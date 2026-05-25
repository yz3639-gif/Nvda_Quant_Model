from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import (
    build_evaluation_context,
    evaluate_rule,
    latest_prediction_for_rule,
)
from nvda_quant_model.strict_model_selection import rule_from_row


DEFAULT_BASELINE_LABEL = "tw147_sw63_10d_return_mq0.60_vq0.45_smh++obv++rsi<75+p60>0.96+vix<0.9"
HARD_GATE_THRESHOLDS = {
    "annualized_return": 0.15,
    "sharpe_ratio": 1.0,
    "max_abs_drawdown": 0.20,
    "win_rate": 0.55,
    "profit_factor": 1.50,
}
LONG_WINDOW_MIN_TRADES = 80
LONG_WINDOW_MIN_ACTIVE_DAYS = 120
METRIC_COLUMNS = [
    "long_score",
    "bt_annualized_return",
    "bt_sharpe_ratio",
    "bt_max_drawdown",
    "bt_win_rate",
    "bt_profit_factor",
    "bt_num_trades",
    "dir_precision",
    "dir_active_days",
    "dir_worst_year_precision",
]


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


def _coerce_numeric_columns(rows: pd.DataFrame) -> pd.DataFrame:
    cleaned = rows.copy()
    for column in METRIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def strict_candidate_mask(
    rows: pd.DataFrame,
    min_trades: int = 30,
    min_annualized: float = 0.15,
    min_sharpe: float = 1.0,
    max_drawdown: float = 0.20,
    min_win_rate: float = 0.55,
    min_profit_factor: float = 1.5,
) -> pd.Series:
    return (
        (rows["bt_num_trades"] >= min_trades)
        & (rows["bt_annualized_return"] >= min_annualized)
        & (rows["bt_sharpe_ratio"] >= min_sharpe)
        & (rows["bt_max_drawdown"].abs() <= max_drawdown)
        & (rows["bt_win_rate"] >= min_win_rate)
        & (rows["bt_profit_factor"] >= min_profit_factor)
    )


def _add_candidate(candidates: dict[str, dict[str, Any]], row: pd.Series, role: str) -> None:
    label = str(row["label"])
    if label not in candidates:
        candidates[label] = {"roles": [], "row": row}
    if role not in candidates[label]["roles"]:
        candidates[label]["roles"].append(role)


def select_promotion_candidates(
    rows: pd.DataFrame,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    top_per_bucket: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = _coerce_numeric_columns(rows)
    candidates: dict[str, dict[str, Any]] = {}

    baseline = rows[rows["label"] == baseline_label]
    if not baseline.empty:
        _add_candidate(candidates, baseline.sort_values("long_score", ascending=False).iloc[0], "current_baseline")

    strict30 = rows[strict_candidate_mask(rows, min_trades=30)].copy()
    for metric, role in [
        ("long_score", "strict30_best_score"),
        ("bt_annualized_return", "strict30_best_return"),
        ("bt_sharpe_ratio", "strict30_best_sharpe"),
        ("bt_win_rate", "strict30_best_win_rate"),
        ("bt_profit_factor", "strict30_best_profit_factor"),
    ]:
        for _, row in strict30.sort_values(metric, ascending=False).head(top_per_bucket).iterrows():
            _add_candidate(candidates, row, role)

    for min_trades in [40, 50, 60]:
        sampled = rows[strict_candidate_mask(rows, min_trades=min_trades)].copy()
        for metric, role_suffix in [("long_score", "score"), ("bt_annualized_return", "return")]:
            for _, row in sampled.sort_values(metric, ascending=False).head(top_per_bucket).iterrows():
                _add_candidate(candidates, row, f"sample{min_trades}_best_{role_suffix}")

    selected_rows = []
    for label, item in candidates.items():
        selected = item["row"].to_dict()
        selected["roles"] = "+".join(item["roles"])
        selected_rows.append(selected)
    selected_df = pd.DataFrame(selected_rows)
    if not selected_df.empty:
        selected_df = selected_df.sort_values(["roles", "long_score"], ascending=[True, False]).reset_index(drop=True)

    metadata = {
        "source_rows": int(len(rows)),
        "strict30_count": int(len(strict30)),
        "selected_count": int(len(selected_df)),
        "baseline_found": bool(not baseline.empty),
        "baseline_label": baseline_label,
    }
    return selected_df, metadata


def _metric_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def hard_gate_failures(row: pd.Series, *, enforce_long_sample: bool = False) -> list[str]:
    """Return production hard-gate failures for one stress-test window."""

    failures: list[str] = []
    if _metric_value(row, "annualized_return") < HARD_GATE_THRESHOLDS["annualized_return"]:
        failures.append("annualized_return")
    if _metric_value(row, "sharpe_ratio") < HARD_GATE_THRESHOLDS["sharpe_ratio"]:
        failures.append("sharpe_ratio")
    if abs(_metric_value(row, "max_drawdown")) > HARD_GATE_THRESHOLDS["max_abs_drawdown"]:
        failures.append("max_drawdown")
    if _metric_value(row, "win_rate") < HARD_GATE_THRESHOLDS["win_rate"]:
        failures.append("win_rate")
    if _metric_value(row, "profit_factor") < HARD_GATE_THRESHOLDS["profit_factor"]:
        failures.append("profit_factor")
    if enforce_long_sample:
        if _metric_value(row, "num_trades") < LONG_WINDOW_MIN_TRADES:
            failures.append("num_trades")
        if _metric_value(row, "dir_active_days") < LONG_WINDOW_MIN_ACTIVE_DAYS:
            failures.append("dir_active_days")
    return failures


def hard_gate_pass(row: pd.Series, *, enforce_long_sample: bool = False) -> bool:
    return not hard_gate_failures(row, enforce_long_sample=enforce_long_sample)


def promotion_score(group: pd.DataFrame) -> float:
    by_window = {int(row["lookback_months"]): row for _, row in group.iterrows()}
    recent = by_window.get(24)
    mid = by_window.get(36)
    long = by_window.get(60)
    if recent is None:
        return -999.0

    score = 0.0
    score += _metric_value(recent, "dir_precision") * 4.0
    score += _metric_value(recent, "sharpe_ratio") * 0.65
    score += _metric_value(recent, "annualized_return") * 1.2
    score += min(_metric_value(recent, "profit_factor"), 6.0) * 0.25
    score -= abs(_metric_value(recent, "max_drawdown")) * 4.0
    score += min(_metric_value(recent, "num_trades") / 50.0, 1.0) * 0.6

    if mid is not None:
        score += _metric_value(mid, "annualized_return") * 1.0
        score += _metric_value(mid, "sharpe_ratio") * 0.35
        score += _metric_value(mid, "win_rate") * 0.5
        score -= max(0.0, 0.12 - _metric_value(mid, "annualized_return")) * 3.0
        score -= max(0.0, abs(_metric_value(mid, "max_drawdown")) - 0.15) * 3.0

    if long is not None:
        score += _metric_value(long, "annualized_return") * 0.35
        score += _metric_value(long, "sharpe_ratio") * 0.15
        score -= max(0.0, -_metric_value(long, "annualized_return")) * 4.0
        score -= max(0.0, abs(_metric_value(long, "max_drawdown")) - 0.25) * 2.5

    return float(score)


def classify_candidate(group: pd.DataFrame, baseline_group: pd.DataFrame | None) -> str:
    by_window = {int(row["lookback_months"]): row for _, row in group.iterrows()}
    recent = by_window.get(24)
    mid = by_window.get(36)
    long = by_window.get(60)
    if recent is None or mid is None or long is None:
        return "reject_missing_window"

    recent_pass = hard_gate_pass(recent)
    mid_pass = hard_gate_pass(mid)
    long_pass = hard_gate_pass(long, enforce_long_sample=True)
    if not long_pass:
        return "reject_long_window_failure"
    if not recent_pass or not mid_pass:
        return "reject_weak_metrics"

    beats_recent_baseline = False
    if baseline_group is not None and not baseline_group.empty:
        base_recent = baseline_group[baseline_group["lookback_months"] == 24]
        if not base_recent.empty:
            base = base_recent.iloc[0]
            beats_recent_baseline = (
                _metric_value(recent, "annualized_return") >= _metric_value(base, "annualized_return")
                and _metric_value(recent, "sharpe_ratio") >= _metric_value(base, "sharpe_ratio")
                and _metric_value(recent, "win_rate") >= _metric_value(base, "win_rate")
                and abs(_metric_value(recent, "max_drawdown")) <= abs(_metric_value(base, "max_drawdown")) * 1.25
                and _metric_value(recent, "profit_factor") >= _metric_value(base, "profit_factor") * 0.90
            )

    if recent_pass and mid_pass and long_pass and beats_recent_baseline:
        return "promote_candidate"
    if recent_pass and mid_pass:
        return "watchlist_all_windows"
    if recent_pass:
        return "watchlist_recent_only"
    return "reject_weak_metrics"


def evaluate_candidates(
    candidates: pd.DataFrame,
    lookback_months: list[int],
    ticker: str,
    price_override: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stress_rows: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []

    for months in lookback_months:
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
        context = build_evaluation_context(config, frame, prices, external)

        for _, candidate in candidates.iterrows():
            rule = rule_from_row(candidate)
            row, _, periods = evaluate_rule(rule, config, frame, prices, external, 0, 0, context)
            precision_series = periods["precision"].dropna() if "precision" in periods else pd.Series(dtype=float)
            stress_rows.append(
                {
                    "roles": candidate["roles"],
                    "label": candidate["label"],
                    "lookback_months": months,
                    "resolved_start": config.start_date,
                    "resolved_end": config.end_date,
                    "annualized_return": row["bt_annualized_return"],
                    "sharpe_ratio": row["bt_sharpe_ratio"],
                    "max_drawdown": row["bt_max_drawdown"],
                    "win_rate": row["bt_win_rate"],
                    "profit_factor": row["bt_profit_factor"],
                    "num_trades": row["bt_num_trades"],
                    "dir_precision": row["dir_precision"],
                    "dir_active_days": row["dir_active_days"],
                    "dir_worst_year_precision": row["dir_worst_year_precision"],
                    "period_precision_std": row["dir_period_precision_std"],
                    "calibration_samples": row.get("dir_calibration_samples"),
                    "brier_score": row.get("dir_brier_score"),
                    "calibration_error": row.get("dir_calibration_error"),
                    "mean_abs_calibration_error": row.get("dir_mean_abs_calibration_error"),
                    "max_abs_bin_error": row.get("dir_max_abs_bin_error"),
                    "losing_periods": int((precision_series < 0.5).sum()) if not precision_series.empty else 0,
                    "source_long_score": candidate.get("long_score"),
                }
            )
            if months == min(lookback_months):
                latest = latest_prediction_for_rule(rule, frame.loc[frame.index <= pd.Timestamp(config.end_date)], price_override)
                latest_rows.append(
                    {
                        "roles": candidate["roles"],
                        "label": candidate["label"],
                        "latest_date": latest["date"],
                        "signal": latest["signal"],
                        "position": latest["position"],
                        "prob_up": latest["prob_up"],
                        "expected_return": latest["expected_return"],
                        "target_price": latest["target_price"],
                        "stop_loss": latest["stop_loss"],
                        "take_profit": latest["take_profit"],
                    }
                )

    return pd.DataFrame(stress_rows), pd.DataFrame(latest_rows)


def summarize_promotions(
    stress: pd.DataFrame,
    baseline_label: str,
) -> pd.DataFrame:
    baseline_group = stress[stress["label"] == baseline_label]
    if baseline_group.empty:
        baseline_group = None
    rows: list[dict[str, Any]] = []
    for label, group in stress.groupby("label", sort=False):
        scored = group.copy()
        score = promotion_score(scored)
        status = "current_baseline" if label == baseline_label else classify_candidate(scored, baseline_group)
        recent = scored[scored["lookback_months"] == 24]
        mid = scored[scored["lookback_months"] == 36]
        long = scored[scored["lookback_months"] == 60]
        row = {
            "status": status,
            "promotion_score": score,
            "roles": str(scored["roles"].iloc[0]),
            "label": label,
        }
        for prefix, frame in [("m24", recent), ("m36", mid), ("m60", long)]:
            if frame.empty:
                continue
            item = frame.iloc[0]
            for column in [
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "num_trades",
                "dir_precision",
                "brier_score",
                "calibration_error",
                "mean_abs_calibration_error",
                "max_abs_bin_error",
            ]:
                if column in item:
                    row[f"{prefix}_{column}"] = item[column]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["status", "promotion_score"], ascending=[True, False]).reset_index(drop=True)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2%}"


def _num(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2f}"


def write_report(output_dir: Path, metadata: dict[str, Any], summary: pd.DataFrame, stress: pd.DataFrame, latest: pd.DataFrame) -> Path:
    display = summary.copy()
    for column in display.columns:
        if any(key in column for key in ["annualized_return", "max_drawdown", "win_rate", "dir_precision", "calibration_error"]):
            display[column] = display[column].map(_pct)
        elif any(key in column for key in ["sharpe_ratio", "profit_factor", "promotion_score"]):
            display[column] = display[column].map(_num)

    latest_display = latest.copy()
    for column in ["prob_up", "expected_return"]:
        if column in latest_display:
            latest_display[column] = latest_display[column].map(_pct)

    lines = [
        "# Candidate Promotion Backtest",
        "",
        "## Metadata",
        "",
        f"- Source rows: {metadata['source_rows']}",
        f"- Strict 30+ candidates: {metadata['strict30_count']}",
        f"- Selected candidates: {metadata['selected_count']}",
        f"- Baseline found: {metadata['baseline_found']}",
        "",
        "## Promotion Summary",
        "",
        display.to_markdown(index=False),
        "",
        "## Latest Reaction Snapshot",
        "",
        latest_display.to_markdown(index=False) if not latest_display.empty else "No latest rows.",
        "",
        "## Interpretation",
        "",
        "- `promote_candidate`: can replace baseline only if recent, mid-term, and long-term gates all hold.",
        "- `watchlist_*`: useful for timely reaction, but not stable enough to replace the main rule.",
        "- `reject_*`: keep out of production even if one short window looks attractive.",
        "",
        "## Files",
        "",
        f"- Stress backtests: {output_dir / 'candidate_stress_backtests.csv'}",
        f"- Promotion summary: {output_dir / 'candidate_promotion_summary.csv'}",
        f"- Latest snapshot: {output_dir / 'candidate_latest_snapshot.csv'}",
        "",
    ]
    path = output_dir / "candidate_promotion_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.results_csv)
    candidates, metadata = select_promotion_candidates(rows, args.baseline_label, args.top_per_bucket)
    if candidates.empty:
        raise RuntimeError("No promotion candidates selected")

    lookbacks = [int(value.strip()) for value in args.lookback_months.split(",") if value.strip()]
    stress, latest = evaluate_candidates(candidates, lookbacks, args.ticker, args.price_override)
    summary = summarize_promotions(stress, args.baseline_label)

    candidates.to_csv(output_dir / "candidate_source_rows.csv", index=False)
    stress.to_csv(output_dir / "candidate_stress_backtests.csv", index=False)
    latest.to_csv(output_dir / "candidate_latest_snapshot.csv", index=False)
    summary.to_csv(output_dir / "candidate_promotion_summary.csv", index=False)
    report_path = write_report(output_dir, metadata, summary, stress, latest)

    payload = {
        "metadata": metadata,
        "lookback_months": lookbacks,
        "report": str(report_path.resolve()),
        "promotion_counts": summary["status"].value_counts().to_dict(),
        "top_rows": summary.head(10).to_dict(orient="records"),
    }
    (output_dir / "candidate_promotion.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote or reject optimizer candidates with multi-window backtests")
    parser.add_argument("--results-csv", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer" / "long_run_results.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "candidate_promotion"))
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--top-per-bucket", type=int, default=3)
    parser.add_argument("--lookback-months", default="24,36,60")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    run_promotion(parse_args())


if __name__ == "__main__":
    main()
