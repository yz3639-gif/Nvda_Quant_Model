from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.candidate_promotion import DEFAULT_BASELINE_LABEL
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import build_walk_forward_rule_signals, latest_prediction_for_rule
from nvda_quant_model.strict_model_selection import rule_from_row


WATCHLIST_STATUSES = ("watchlist_strong_recent_mid", "watchlist_recent_only")


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):
        return default
    return value


def load_reaction_candidates(
    promotion_dir: Path,
    statuses: tuple[str, ...] = WATCHLIST_STATUSES,
    top_n: int = 8,
) -> tuple[pd.Series, pd.DataFrame]:
    summary = pd.read_csv(promotion_dir / "candidate_promotion_summary.csv")
    source = pd.read_csv(promotion_dir / "candidate_source_rows.csv")
    merged = source.merge(summary[["label", "status", "promotion_score"]], on="label", how="left")
    baseline_rows = merged[merged["label"] == DEFAULT_BASELINE_LABEL]
    if baseline_rows.empty:
        raise RuntimeError(f"Baseline rule not found: {DEFAULT_BASELINE_LABEL}")
    baseline = baseline_rows.sort_values("long_score", ascending=False).iloc[0]

    candidates = merged[merged["status"].isin(statuses)].copy()
    if candidates.empty:
        raise RuntimeError(f"No reaction candidates found for statuses: {statuses}")
    candidates["promotion_score"] = pd.to_numeric(candidates["promotion_score"], errors="coerce").fillna(0.0)
    candidates["m24_dir_precision"] = pd.to_numeric(
        summary.set_index("label").reindex(candidates["label"])["m24_dir_precision"].to_numpy(),
        errors="coerce",
    )
    candidates = candidates.sort_values(["status", "promotion_score", "long_score"], ascending=[True, False, False]).head(top_n)
    return baseline, candidates


def _rule_weight(row: pd.Series) -> float:
    recent_precision = _safe_float(row.get("m24_dir_precision"), _safe_float(row.get("dir_precision"), 0.6))
    score = max(_safe_float(row.get("promotion_score")), 0.0)
    sample = min(max(_safe_float(row.get("bt_num_trades")), 1.0) / 60.0, 1.0)
    status_bonus = 1.15 if row.get("status") == "watchlist_strong_recent_mid" else 1.0
    return float(max(recent_precision - 0.50, 0.02) * (1.0 + score / 8.0) * (0.70 + 0.30 * sample) * status_bonus)


def _panel_for_rules(candidates: pd.DataFrame, frame: pd.DataFrame, report_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_columns: dict[str, pd.Series] = {}
    prob_columns: dict[str, pd.Series] = {}
    expected_columns: dict[str, pd.Series] = {}
    meta_rows: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(candidates.iterrows(), start=1):
        rule = rule_from_row(row)
        signals, _ = build_walk_forward_rule_signals(frame, rule, report_start)
        name = f"watch_{idx:02d}"
        signal_columns[name] = (signals["position"] > 0).astype(float)
        prob_columns[name] = signals["prob_up"]
        expected_columns[name] = signals["expected_return"]
        meta_rows.append(
            {
                "name": name,
                "status": row.get("status"),
                "label": row["label"],
                "weight": _rule_weight(row),
                "promotion_score": _safe_float(row.get("promotion_score")),
                "source_trades": int(_safe_float(row.get("bt_num_trades"))),
                "source_precision": _safe_float(row.get("dir_precision")),
            }
        )
    active = pd.DataFrame(signal_columns).sort_index().fillna(0.0)
    probs = pd.DataFrame(prob_columns).sort_index().fillna(0.5)
    expected = pd.DataFrame(expected_columns).sort_index().fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    panel = pd.concat({"active": active, "prob": probs, "expected": expected}, axis=1)
    return panel, meta


def build_reaction_signals(
    baseline_signals: pd.DataFrame,
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    report_index: pd.Index,
    min_vote_count: int,
    min_vote_weight: float,
    prob_threshold: float,
    max_shadow_exposure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = panel["active"].reindex(report_index).fillna(0.0)
    probs = panel["prob"].reindex(report_index).fillna(0.5)
    expected = panel["expected"].reindex(report_index).fillna(0.0)
    weights = meta.set_index("name")["weight"].reindex(active.columns).fillna(1.0)
    weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(active.columns), index=active.columns)

    active_weight = active.mul(weights, axis=1).sum(axis=1)
    active_count = active.sum(axis=1)
    active_weight_sum = active.mul(weights, axis=1).sum(axis=1).replace(0.0, np.nan)
    weighted_prob = probs.where(active > 0).mul(weights, axis=1).sum(axis=1) / active_weight_sum
    weighted_expected = expected.where(active > 0).mul(weights, axis=1).sum(axis=1) / active_weight_sum
    weighted_prob = weighted_prob.fillna(0.5).clip(0.01, 0.99)
    weighted_expected = weighted_expected.fillna(0.0).clip(lower=0.0)

    baseline_position = baseline_signals.reindex(report_index)["position"].fillna(0.0)
    baseline_flat = baseline_position <= 0
    alert = (
        baseline_flat
        & (active_count >= min_vote_count)
        & (active_weight >= min_vote_weight)
        & (weighted_prob >= prob_threshold)
    )
    confidence_strength = ((weighted_prob - 0.50) / max(prob_threshold - 0.50, 0.01)).clip(lower=0.0, upper=1.5) / 1.5
    vote_strength = (active_weight / max(min_vote_weight, 1e-9)).clip(upper=1.5) / 1.5
    shadow_position = (max_shadow_exposure * confidence_strength * vote_strength).where(alert, 0.0)

    reaction = pd.DataFrame(
        {
            "direction": alert.astype(int),
            "confidence": np.maximum(weighted_prob, 1.0 - weighted_prob),
            "expected_return": weighted_expected.where(alert, 0.0),
            "prob_up": weighted_prob.where(alert, 0.5),
            "prob_down": (1.0 - weighted_prob).where(alert, 0.5),
            "position": shadow_position,
            "active_rule_count": active_count,
            "active_rule_weight": active_weight,
            "baseline_flat": baseline_flat.astype(int),
        },
        index=report_index,
    )
    diagnostics = pd.DataFrame(
        {
            "active_rule_count": active_count,
            "active_rule_weight": active_weight,
            "weighted_prob_up": weighted_prob,
            "weighted_expected_return": weighted_expected,
            "baseline_position": baseline_position,
            "alert": alert.astype(int),
            "shadow_position": shadow_position,
        },
        index=report_index,
    )
    return reaction, diagnostics


def combine_baseline_and_reaction(baseline_signals: pd.DataFrame, reaction_signals: pd.DataFrame, report_index: pd.Index) -> pd.DataFrame:
    baseline = baseline_signals.reindex(report_index).fillna(
        {
            "direction": 0,
            "confidence": 0.0,
            "expected_return": 0.0,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "position": 0.0,
        }
    )
    reaction = reaction_signals.reindex(report_index).fillna(
        {
            "direction": 0,
            "confidence": 0.0,
            "expected_return": 0.0,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "position": 0.0,
        }
    )
    combined = baseline.copy()
    use_reaction = (combined["position"] <= 0) & (reaction["position"] > 0)
    for column in ["direction", "confidence", "expected_return", "prob_up", "prob_down", "position"]:
        combined.loc[use_reaction, column] = reaction.loc[use_reaction, column]
    combined["reaction_overlay"] = use_reaction.astype(int)
    return combined


def directional_metrics(signals: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    aligned = frame[["target_return", "target_direction"]].join(signals[["position", "prob_up"]], how="inner")
    active = aligned.loc[aligned["position"] > 0].dropna(subset=["target_return", "target_direction"])
    if active.empty:
        return {
            "active_days": 0,
            "direction_precision": 0.0,
            "avg_next_return": 0.0,
            "median_next_return": 0.0,
            "worst_year_precision": 0.0,
            "brier_active": None,
        }
    yearly = active.assign(year=active.index.year).groupby("year")["target_direction"].mean()
    prob = active["prob_up"].clip(0.01, 0.99)
    y = active["target_direction"].astype(float)
    return {
        "active_days": int(len(active)),
        "direction_precision": float((active["target_return"] > 0).mean()),
        "avg_next_return": float(active["target_return"].mean()),
        "median_next_return": float(active["target_return"].median()),
        "worst_year_precision": float(yearly.min()) if not yearly.empty else 0.0,
        "brier_active": float(np.mean((prob - y) ** 2)),
        "year_precision": {str(year): float(value) for year, value in yearly.items()},
    }


def latest_reaction_snapshot(
    baseline: pd.Series,
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    price_override: float | None,
    min_vote_count: int,
    prob_threshold: float,
) -> dict[str, Any]:
    baseline_rule = rule_from_row(baseline)
    baseline_latest = latest_prediction_for_rule(baseline_rule, frame, price_override)
    rows = []
    active_weights = []
    for _, row in candidates.iterrows():
        rule = rule_from_row(row)
        latest = latest_prediction_for_rule(rule, frame, price_override)
        weight = _rule_weight(row)
        rows.append(
            {
                "label": row["label"],
                "status": row.get("status"),
                "signal": latest["signal"],
                "prob_up": latest["prob_up"],
                "expected_return": latest["expected_return"],
                "weight": weight,
            }
        )
        if latest["signal"] == 1:
            active_weights.append((latest, weight))
    active_count = len(active_weights)
    total_active_weight = sum(weight for _, weight in active_weights)
    if active_weights and total_active_weight > 0:
        weighted_prob = sum(latest["prob_up"] * weight for latest, weight in active_weights) / total_active_weight
        weighted_expected = sum(latest["expected_return"] * weight for latest, weight in active_weights) / total_active_weight
    else:
        weighted_prob = 0.5
        weighted_expected = 0.0
    alert = baseline_latest["signal"] == 0 and active_count >= min_vote_count and weighted_prob >= prob_threshold
    current_price = float(baseline_latest["current_price"])
    return {
        "date": baseline_latest["date"],
        "as_of_date": baseline_latest["as_of_date"],
        "baseline_signal": baseline_latest["signal"],
        "reaction_alert": int(alert),
        "active_watchlist_rules": active_count,
        "weighted_prob_up": round(float(weighted_prob), 4),
        "expected_return": round(float(weighted_expected), 6) if alert else 0.0,
        "target_price": round(current_price * (1.0 + weighted_expected), 4) if alert else current_price,
        "current_price": current_price,
        "reason": (
            "watchlist confirms a short-term long alert while baseline is flat"
            if alert
            else "no reaction alert; baseline is active or watchlist consensus is insufficient"
        ),
        "watchlist_rows": rows,
    }


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2%}"


def _num(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2f}"


def write_report(
    output_dir: Path,
    payload: dict[str, Any],
    metrics_table: pd.DataFrame,
    meta: pd.DataFrame,
) -> Path:
    display = metrics_table.copy()
    for column in ["annualized_return", "max_drawdown", "win_rate", "direction_precision", "avg_next_return", "median_next_return", "worst_year_precision"]:
        if column in display:
            display[column] = display[column].map(_pct)
    for column in ["sharpe_ratio", "profit_factor", "brier_active"]:
        if column in display:
            display[column] = display[column].map(_num)

    meta_display = meta.copy()
    for column in ["weight", "source_precision"]:
        if column in meta_display:
            meta_display[column] = meta_display[column].map(_num)

    latest = payload["latest_reaction"]
    lines = [
        "# NVDA Reaction Pool Backtest",
        "",
        "## Purpose",
        "",
        "This layer is a shadow alert system. It only fires when the baseline model is flat and watchlist rules agree. It is not a replacement for the production baseline.",
        "",
        "## Backtest Metrics",
        "",
        display.to_markdown(index=False),
        "",
        "## Latest Reaction Snapshot",
        "",
        f"- Date: {latest['date']}",
        f"- Baseline signal: {latest['baseline_signal']}",
        f"- Reaction alert: {latest['reaction_alert']}",
        f"- Active watchlist rules: {latest['active_watchlist_rules']}",
        f"- Weighted prob up: {_pct(latest['weighted_prob_up'])}",
        f"- Expected return: {_pct(latest['expected_return'])}",
        f"- Target price: {latest['target_price']}",
        f"- Reason: {latest['reason']}",
        "",
        "## Watchlist Members",
        "",
        meta_display.to_markdown(index=False),
        "",
        "## Files",
        "",
        f"- Reaction signals: {output_dir / 'reaction_signals.csv'}",
        f"- Combined shadow signals: {output_dir / 'combined_shadow_signals.csv'}",
        f"- Diagnostics: {output_dir / 'reaction_diagnostics.csv'}",
        "",
    ]
    path = output_dir / "reaction_pool_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_reaction_pool(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = tuple(value.strip() for value in args.statuses.split(",") if value.strip())
    baseline, candidates = load_reaction_candidates(Path(args.promotion_dir), statuses, args.top_n)

    start, end, metadata = resolve_data_window("auto", "latest", ticker=args.ticker, lookback_months=args.lookback_months)
    baseline_rule = rule_from_row(baseline)
    config = StrategyConfig(
        ticker=args.ticker,
        start_date=start,
        end_date=end,
        lookback_months=args.lookback_months,
        stop_loss_pct=baseline_rule.stop_loss_pct,
        take_profit_pct=baseline_rule.take_profit_pct,
        max_exposure=baseline_rule.max_exposure,
        include_peer_events=True,
    )
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
    report_prices = prices.loc[config.start_date : config.end_date]

    baseline_rule = rule_from_row(baseline)
    baseline_signals, _ = build_walk_forward_rule_signals(frame, baseline_rule, config.start_date)
    baseline_signals = baseline_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
    panel, meta = _panel_for_rules(candidates, frame, config.start_date)
    reaction_signals, diagnostics = build_reaction_signals(
        baseline_signals,
        panel,
        meta,
        report_prices.index,
        args.min_vote_count,
        args.min_vote_weight,
        args.prob_threshold,
        args.max_shadow_exposure,
    )
    combined = combine_baseline_and_reaction(baseline_signals, reaction_signals, report_prices.index)

    baseline_result = BacktestEngine(config).backtest(baseline_signals, report_prices)
    reaction_config = replace(
        config,
        stop_loss_pct=args.reaction_stop_loss_pct,
        take_profit_pct=args.reaction_take_profit_pct,
        max_exposure=args.max_shadow_exposure,
    )
    reaction_result = BacktestEngine(reaction_config).backtest(reaction_signals, report_prices)
    combined_result = BacktestEngine(config).backtest(combined, report_prices)

    report_frame = frame.loc[config.start_date : config.end_date]
    rows = []
    for name, result, signals in [
        ("baseline", baseline_result, baseline_signals),
        ("reaction_shadow", reaction_result, reaction_signals),
        ("combined_shadow", combined_result, combined),
    ]:
        dmetrics = directional_metrics(signals, report_frame)
        rows.append({"layer": name, **result.metrics, **dmetrics})
    metrics_table = pd.DataFrame(rows)
    latest = latest_reaction_snapshot(
        baseline,
        candidates,
        frame.loc[frame.index <= pd.Timestamp(config.end_date)],
        args.price_override,
        args.min_vote_count,
        args.prob_threshold,
    )

    reaction_signals.to_csv(output_dir / "reaction_signals.csv")
    combined.to_csv(output_dir / "combined_shadow_signals.csv")
    diagnostics.to_csv(output_dir / "reaction_diagnostics.csv")
    meta.to_csv(output_dir / "reaction_members.csv", index=False)
    metrics_table.to_csv(output_dir / "reaction_metrics.csv", index=False)

    payload = {
        "data_window": {
            **metadata,
            "resolved_start": config.start_date,
            "resolved_end": config.end_date,
            "rows": int(len(report_prices)),
        },
        "filters": {
            "statuses": statuses,
            "top_n": args.top_n,
            "min_vote_count": args.min_vote_count,
            "min_vote_weight": args.min_vote_weight,
            "prob_threshold": args.prob_threshold,
            "max_shadow_exposure": args.max_shadow_exposure,
        },
        "baseline_label": str(baseline["label"]),
        "member_count": int(len(meta)),
        "metrics": metrics_table.to_dict(orient="records"),
        "latest_reaction": latest,
    }
    report_path = write_report(output_dir, payload, metrics_table, meta)
    payload["report"] = str(report_path.resolve())
    (output_dir / "reaction_pool_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shadow reaction pool from promoted watchlist rules")
    parser.add_argument("--promotion-dir", default=str(PROJECT_ROOT / "outputs" / "candidate_promotion"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "reaction_pool"))
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--statuses", default="watchlist_strong_recent_mid,watchlist_recent_only")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--min-vote-count", type=int, default=2)
    parser.add_argument("--min-vote-weight", type=float, default=0.18)
    parser.add_argument("--prob-threshold", type=float, default=0.54)
    parser.add_argument("--max-shadow-exposure", type=float, default=0.25)
    parser.add_argument("--reaction-stop-loss-pct", type=float, default=0.025)
    parser.add_argument("--reaction-take-profit-pct", type=float, default=0.035)
    parser.add_argument("--price-override", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    run_reaction_pool(parse_args())


if __name__ == "__main__":
    main()
