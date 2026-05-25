from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import calculate_metrics
from nvda_quant_model.candidate_promotion import (
    DEFAULT_BASELINE_LABEL,
    LONG_WINDOW_MIN_ACTIVE_DAYS,
    LONG_WINDOW_MIN_TRADES,
    classify_candidate,
    evaluate_candidates,
    hard_gate_pass,
)
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.optimizer_csv import read_live_optimizer_csv
from nvda_quant_model.precision_search import (
    build_evaluation_context,
    build_walk_forward_rule_signals,
    latest_prediction_for_rule,
)
from nvda_quant_model.strict_model_selection import rule_from_row


OOS_WINDOWS = (45, 63, 84, 126)
MIN_OOS_TRADES = 3
MIN_REGIME_ACTIVE_DAYS = 3


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _source_score(row: pd.Series) -> float:
    for column in ("high_sample_score", "long_score", "score"):
        if column in row and pd.notna(row[column]):
            return _safe_float(row[column])
    return 0.0


def select_stable_candidates(
    high_sample_rows: pd.DataFrame,
    strict_rows: pd.DataFrame,
    *,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    top_n: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a small validation set; this does not promote anything by itself."""

    candidates: list[pd.Series] = []
    baseline = strict_rows[strict_rows["label"] == baseline_label].copy() if "label" in strict_rows else pd.DataFrame()
    baseline_found = not baseline.empty
    if baseline.empty and not strict_rows.empty:
        baseline = strict_rows.assign(_source_score=strict_rows.apply(_source_score, axis=1)).sort_values("_source_score", ascending=False).head(1)
    if not baseline.empty:
        item = baseline.iloc[0].copy()
        item["roles"] = "current_baseline"
        candidates.append(item)

    if not high_sample_rows.empty:
        high = high_sample_rows.copy()
        if {"sample_gate", "quality_gate"}.issubset(high.columns):
            high = high[(high["sample_gate"] == True) & (high["quality_gate"] == True)]  # noqa: E712
        if not high.empty:
            high["_source_score"] = high.apply(_source_score, axis=1)
            for _, row in high.sort_values("_source_score", ascending=False).head(top_n).iterrows():
                row = row.copy()
                row["roles"] = "high_sample_stability_probe"
                candidates.append(row)

    if not strict_rows.empty:
        strict = strict_rows.copy()
        strict["_source_score"] = strict.apply(_source_score, axis=1)
        strict = strict[strict["bt_num_trades"].fillna(0) >= 30] if "bt_num_trades" in strict else strict
        for _, row in strict.sort_values("_source_score", ascending=False).head(max(2, top_n // 2)).iterrows():
            if any(str(item.get("label")) == str(row.get("label")) for item in candidates):
                continue
            row = row.copy()
            row["roles"] = "strict_stability_probe"
            candidates.append(row)

    selected = pd.DataFrame(candidates).drop_duplicates("label", keep="first") if candidates else pd.DataFrame()
    metadata = {
        "baseline_label": baseline_label,
        "baseline_found": baseline_found,
        "selected_count": int(len(selected)),
        "high_sample_rows": int(len(high_sample_rows)),
        "strict_rows": int(len(strict_rows)),
    }
    return selected.reset_index(drop=True), metadata


def _prepare_window(ticker: str, months: int) -> tuple[StrategyConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end, _ = resolve_data_window("auto", "latest", ticker=ticker, lookback_months=months)
    config = StrategyConfig(ticker=ticker, start_date=start, end_date=end, lookback_months=months, include_peer_events=True)
    prices, external = load_market_data(config.ticker, config.start_date, config.end_date, cache_dir=PROJECT_ROOT / "cache")
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(config.start_date), config.end_date, cache_dir=PROJECT_ROOT / "cache")
    frame, _ = build_model_frame(
        prices,
        external,
        config.start_date,
        config.end_date,
        config.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
    )
    return config, prices, external, frame


def rolling_oos_diagnostics(
    candidate: pd.Series,
    config: StrategyConfig,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = OOS_WINDOWS,
    min_oos_trades: int = MIN_OOS_TRADES,
) -> pd.DataFrame:
    """Evaluate trailing OOS slices for one candidate on the prepared window."""

    rule = rule_from_row(candidate)
    context = build_evaluation_context(config, frame, prices, external)
    signals, _ = build_walk_forward_rule_signals(frame, rule, config.start_date)
    signals = signals.reindex(context.report_prices.index).dropna(subset=["prob_up", "expected_return"])
    cfg = StrategyConfig(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
        lookback_months=config.lookback_months,
        train_window=rule.train_window,
        test_window=rule.test_window,
        stop_loss_pct=rule.stop_loss_pct,
        take_profit_pct=rule.take_profit_pct,
        max_exposure=rule.max_exposure,
        include_peer_events=True,
    )
    full = BacktestEngine(cfg).backtest(signals, context.report_prices)
    full_sharpe = _safe_float(full.metrics.get("sharpe_ratio"))
    rows: list[dict[str, Any]] = []
    for window in windows:
        tail_index = context.report_prices.tail(window).index
        oos_prices = context.report_prices.loc[tail_index]
        oos_signals = signals.reindex(tail_index).dropna(subset=["prob_up", "expected_return"])
        active_days = int((oos_signals.get("position", pd.Series(dtype=float)) > 0).sum())
        if len(oos_prices) < 10 or oos_signals.empty:
            rows.append(_oos_row(candidate, window, full_sharpe, np.nan, 0, active_days, "oos_insufficient_sample"))
            continue
        result = BacktestEngine(cfg).backtest(oos_signals, oos_prices)
        returns = result.daily_returns.dropna()
        equity = cfg.initial_capital * (1.0 + returns).cumprod()
        metrics = calculate_metrics(equity, returns, result.trades, cfg.initial_capital)
        oos_sharpe = _safe_float(metrics.get("sharpe_ratio"), np.nan)
        degradation = (full_sharpe - oos_sharpe) / abs(full_sharpe) if full_sharpe and np.isfinite(oos_sharpe) else np.inf
        trades = int(metrics.get("num_trades", 0))
        status = "pass" if trades >= min_oos_trades and degradation <= 0.20 else "reject_oos_degradation"
        if trades < min_oos_trades:
            status = "oos_insufficient_sample"
        rows.append(_oos_row(candidate, window, full_sharpe, oos_sharpe, trades, active_days, status, degradation, metrics))
    return pd.DataFrame(rows)


def _oos_row(
    candidate: pd.Series,
    window: int,
    full_sharpe: float,
    oos_sharpe: float,
    trades: int,
    active_days: int,
    status: str,
    degradation: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    if degradation is None:
        degradation = (full_sharpe - oos_sharpe) / abs(full_sharpe) if full_sharpe and np.isfinite(oos_sharpe) else np.inf
    return {
        "label": candidate["label"],
        "window_days": window,
        "full_sharpe_ratio": full_sharpe,
        "oos_sharpe_ratio": oos_sharpe,
        "sharpe_degradation": float(degradation),
        "oos_num_trades": trades,
        "oos_active_days": active_days,
        "oos_status": status,
        "oos_annualized_return": metrics.get("annualized_return"),
        "oos_max_drawdown": metrics.get("max_drawdown"),
    }


def regime_diagnostics(
    candidate: pd.Series,
    config: StrategyConfig,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    min_active_days: int = MIN_REGIME_ACTIVE_DAYS,
) -> pd.DataFrame:
    rule = rule_from_row(candidate)
    signals, _ = build_walk_forward_rule_signals(frame, rule, config.start_date)
    report_frame = frame.loc[config.start_date : config.end_date]
    aligned = report_frame.join(signals[["position"]], how="inner")
    active = aligned["position"] > 0
    regimes: dict[str, pd.Series] = {}
    if {"price_60ma_ratio", "volatility_20", "volatility_60"}.issubset(aligned.columns):
        regimes["high_vol_downtrend"] = (aligned["price_60ma_ratio"] < 1.0) & (aligned["volatility_20"] > aligned["volatility_60"])
    if {"price_60ma_ratio", "20d_return"}.issubset(aligned.columns):
        regimes["ai_uptrend"] = (aligned["price_60ma_ratio"] > 1.0) & (aligned["20d_return"] > 0)
    if "pre_holiday_session" in aligned.columns:
        regimes["holiday_week"] = aligned["pre_holiday_session"] == 1

    rows: list[dict[str, Any]] = []
    for name, mask in regimes.items():
        sample = aligned.loc[active & mask].dropna(subset=["target_return"])
        precision = float((sample["target_return"] > 0).mean()) if len(sample) else np.nan
        status = "pass"
        if len(sample) < min_active_days:
            status = "regime_insufficient_sample"
        elif precision < 0.50:
            status = "watchlist_regime_weakness"
        rows.append(
            {
                "label": candidate["label"],
                "regime": name,
                "active_days": int(len(sample)),
                "precision": precision,
                "avg_next_return": float(sample["target_return"].mean()) if len(sample) else np.nan,
                "regime_status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize_stable_validation(
    stress: pd.DataFrame,
    rolling_oos: pd.DataFrame,
    regimes: pd.DataFrame,
    baseline_label: str,
) -> pd.DataFrame:
    baseline_group = stress[stress["label"] == baseline_label]
    rows: list[dict[str, Any]] = []
    for label, group in stress.groupby("label", sort=False):
        status = "current_baseline" if label == baseline_label else classify_candidate(group, baseline_group)
        oos_group = rolling_oos[rolling_oos["label"] == label]
        regime_group = regimes[regimes["label"] == label] if not regimes.empty else pd.DataFrame()
        if status in {"promote_candidate", "watchlist_all_windows"}:
            if oos_group.empty or (oos_group["oos_status"] == "oos_insufficient_sample").any():
                status = "oos_insufficient_sample"
            elif (oos_group["oos_status"] == "reject_oos_degradation").any():
                status = "reject_oos_degradation"
            elif not regime_group.empty and (regime_group["regime_status"] == "watchlist_regime_weakness").any():
                status = "watchlist_regime_weakness"
        row: dict[str, Any] = {
            "status": status,
            "label": label,
            "roles": str(group["roles"].iloc[0]),
            "stable_score": stable_score(group, oos_group, regime_group),
            "rolling_oos_worst_degradation": float(oos_group["sharpe_degradation"].replace([np.inf, -np.inf], np.nan).max()) if not oos_group.empty else np.nan,
            "rolling_oos_median_degradation": float(oos_group["sharpe_degradation"].replace([np.inf, -np.inf], np.nan).median()) if not oos_group.empty else np.nan,
            "overfit_proxy_status": (
                "BASELINE_AUDITED_SEPARATELY"
                if status == "current_baseline"
                else "PASS_WITH_WARNINGS"
                if status == "promote_candidate"
                else "REJECT_OR_WATCHLIST"
            ),
        }
        for months in (24, 36, 60):
            window = group[group["lookback_months"] == months]
            if window.empty:
                continue
            item = window.iloc[0]
            for col in ["annualized_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "num_trades", "dir_active_days", "dir_precision"]:
                row[f"m{months}_{col}"] = item.get(col)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["status", "stable_score"], ascending=[True, False]).reset_index(drop=True)


def stable_score(stress_group: pd.DataFrame, oos_group: pd.DataFrame, regime_group: pd.DataFrame) -> float:
    windows = {int(row["lookback_months"]): row for _, row in stress_group.iterrows()}
    if 60 not in windows:
        return -999.0
    min_ann = min(_safe_float(row.get("annualized_return"), -1.0) for row in windows.values())
    min_sharpe = min(_safe_float(row.get("sharpe_ratio"), -1.0) for row in windows.values())
    max_dd = max(abs(_safe_float(row.get("max_drawdown"))) for row in windows.values())
    min_win = min(_safe_float(row.get("win_rate")) for row in windows.values())
    min_pf = min(_safe_float(row.get("profit_factor")) for row in windows.values())
    long = windows[60]
    worst_oos = 1.0
    if not oos_group.empty:
        cleaned = oos_group["sharpe_degradation"].replace([np.inf, -np.inf], np.nan).dropna()
        worst_oos = float(cleaned.max()) if not cleaned.empty else 1.0
    weak_regimes = int((regime_group.get("regime_status", pd.Series(dtype=str)) == "watchlist_regime_weakness").sum()) if not regime_group.empty else 0
    gate_bonus = 3.0 if all(hard_gate_pass(row, enforce_long_sample=int(row["lookback_months"]) == 60) for _, row in stress_group.iterrows()) else 0.0
    return float(
        gate_bonus
        + min_ann * 4.0
        + min_sharpe * 1.5
        + min_win * 1.5
        + min_pf * 0.6
        + _safe_float(long.get("annualized_return")) * 1.5
        - max_dd * 4.0
        - max(0.0, worst_oos) * 2.0
        - weak_regimes * 0.75
    )


def write_stable_report(output_dir: Path, summary: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    lines = [
        "# Stable Candidate Validation",
        "",
        "## Metadata",
        "",
        f"- Baseline label: `{metadata['baseline_label']}`",
        f"- Baseline found: {metadata['baseline_found']}",
        f"- Selected candidates: {metadata['selected_count']}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No candidates.",
        "",
        "## Interpretation",
        "",
        "- `promote_candidate` requires 24/36/60M hard gates, rolling OOS pass, and no key regime precision below 50%.",
        "- `reject_long_window_failure` means 60M failed hard gates or sample requirements.",
        "- `reject_oos_degradation` means trailing OOS Sharpe degradation exceeded 20%.",
    ]
    path = output_dir / "candidate_promotion_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_stable_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_rows = read_live_optimizer_csv(args.strict_results) if Path(args.strict_results).exists() else pd.DataFrame()
    high_rows = read_live_optimizer_csv(args.high_sample_results) if Path(args.high_sample_results).exists() else pd.DataFrame()
    candidates, metadata = select_stable_candidates(
        high_rows,
        strict_rows,
        baseline_label=args.baseline_label,
        top_n=args.top_n,
    )
    if candidates.empty:
        raise RuntimeError("No stable validation candidates selected")

    lookbacks = [24, 36, 60]
    stress, latest = evaluate_candidates(candidates, lookbacks, args.ticker, args.price_override)
    config60, prices60, external60, frame60 = _prepare_window(args.ticker, 60)
    oos_frames = []
    regime_frames = []
    for _, candidate in candidates.iterrows():
        oos_frames.append(rolling_oos_diagnostics(candidate, config60, prices60, external60, frame60))
        regime_frames.append(regime_diagnostics(candidate, config60, prices60, external60, frame60))
    rolling_oos = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    regimes = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    summary = summarize_stable_validation(stress, rolling_oos, regimes, metadata["baseline_label"])
    report = write_stable_report(output_dir, summary, metadata)

    candidates.to_csv(output_dir / "stable_source_candidates.csv", index=False)
    stress.to_csv(output_dir / "stable_results.csv", index=False)
    summary.to_csv(output_dir / "stable_validation.csv", index=False)
    rolling_oos.to_csv(output_dir / "rolling_oos.csv", index=False)
    regimes.to_csv(output_dir / "regime_diagnostics.csv", index=False)
    latest.to_csv(output_dir / "latest_snapshot.csv", index=False)
    best = summary[summary["status"] == "promote_candidate"].sort_values("stable_score", ascending=False).head(1)
    payload = {
        "updated_at": _now(),
        "metadata": metadata,
        "promotion_counts": {str(k): int(v) for k, v in summary["status"].value_counts().to_dict().items()},
        "report": str(report.resolve()),
        "best_promoted_label": None if best.empty else str(best.iloc[0]["label"]),
        "top_rows": summary.head(10).to_dict(orient="records"),
    }
    (output_dir / "stable_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if not best.empty:
        (output_dir / "best_stable_candidate.json").write_text(json.dumps(best.iloc[0].to_dict(), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    else:
        (output_dir / "best_stable_candidate.json").write_text(json.dumps({"status": "no_promoted_candidate"}, indent=2), encoding="utf-8")
    return payload


def run_stable_candidate_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "stable_state.json"
    started_at = _now()
    deadline = time.time() + args.hours * 3600.0
    cycles = 0
    latest_payload: dict[str, Any] = {}
    while time.time() < deadline and (args.max_cycles is None or cycles < args.max_cycles):
        latest_payload = run_stable_validation(args)
        cycles += 1
        state = {
            "started_at": started_at,
            "updated_at": _now(),
            "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(timespec="seconds"),
            "cycles": cycles,
            "best_label": latest_payload.get("best_promoted_label"),
            "output_dir": str(output_dir.resolve()),
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        if args.max_cycles is not None or time.time() + args.cycle_seconds >= deadline:
            break
        time.sleep(args.cycle_seconds)
    return latest_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NVDA candidates with 60M and rolling OOS hard gates")
    parser.add_argument("--strict-results", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer_repaired" / "long_run_results.csv"))
    parser.add_argument("--high-sample-results", default=str(PROJECT_ROOT / "outputs" / "high_sample_optimizer_repaired" / "high_sample_results.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "stable_candidate_optimizer"))
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=215.34)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--cycle-seconds", type=float, default=900.0)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    payload = run_stable_candidate_optimizer(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
