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
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import build_walk_forward_rule_signals
from nvda_quant_model.reaction_pool import (
    _panel_for_rules,
    build_reaction_signals,
    combine_baseline_and_reaction,
    directional_metrics,
    latest_reaction_snapshot,
    load_reaction_candidates,
)
from nvda_quant_model.strict_model_selection import rule_from_row


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _metric_row(
    layer: str,
    segment: str,
    result: Any,
    signals: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    dmetrics = directional_metrics(signals, frame)
    return {"segment": segment, "layer": layer, **result.metrics, **dmetrics}


def _evaluate_segment(
    segment: str,
    index: pd.Index,
    config: StrategyConfig,
    reaction_config: StrategyConfig,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_signals: pd.DataFrame,
    reaction_signals: pd.DataFrame,
    combined_signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    segment_prices = prices.reindex(index).dropna(subset=["Close"])
    segment_frame = frame.reindex(segment_prices.index)
    rows: list[dict[str, Any]] = []
    for layer, cfg, signals in [
        ("baseline", config, baseline_signals),
        ("reaction_shadow", reaction_config, reaction_signals),
        ("combined_shadow", config, combined_signals),
    ]:
        scoped_signals = signals.reindex(segment_prices.index).dropna(subset=["prob_up", "expected_return"])
        result = BacktestEngine(cfg).backtest(scoped_signals, segment_prices)
        rows.append(_metric_row(layer, segment, result, scoped_signals, segment_frame))
    return rows


def _score_train_candidate(metrics: pd.DataFrame, min_reaction_days: int) -> float:
    baseline = metrics[metrics["layer"] == "baseline"].iloc[0]
    reaction = metrics[metrics["layer"] == "reaction_shadow"].iloc[0]
    combined = metrics[metrics["layer"] == "combined_shadow"].iloc[0]
    reaction_days = int(reaction["active_days"])
    if reaction_days < min_reaction_days:
        return -999.0 - (min_reaction_days - reaction_days)
    score = 0.0
    score += _float(reaction["direction_precision"]) * 4.0
    score += _float(reaction["avg_next_return"]) * 60.0
    score += _float(reaction["sharpe_ratio"]) * 0.35
    score += min(_float(reaction["profit_factor"]), 6.0) * 0.20
    score += (_float(combined["annualized_return"]) - _float(baseline["annualized_return"])) * 2.0
    score += (_float(combined["sharpe_ratio"]) - _float(baseline["sharpe_ratio"])) * 0.60
    drawdown_worse = abs(_float(combined["max_drawdown"])) - abs(_float(baseline["max_drawdown"]))
    score -= max(0.0, drawdown_worse) * 6.0
    score -= max(0.0, 0.60 - _float(reaction["worst_year_precision"])) * 0.75
    score -= _float(reaction.get("brier_active"), 0.25) * 0.75
    return float(score)


def _passes_validation(metrics: pd.DataFrame, min_reaction_days: int) -> bool:
    baseline = metrics[metrics["layer"] == "baseline"].iloc[0]
    reaction = metrics[metrics["layer"] == "reaction_shadow"].iloc[0]
    combined = metrics[metrics["layer"] == "combined_shadow"].iloc[0]
    return bool(
        _float(combined["annualized_return"]) >= _float(baseline["annualized_return"])
        and _float(combined["sharpe_ratio"]) >= _float(baseline["sharpe_ratio"])
        and abs(_float(combined["max_drawdown"])) <= abs(_float(baseline["max_drawdown"])) + 0.02
        and _float(reaction["direction_precision"]) >= 0.60
        and int(reaction["active_days"]) >= min_reaction_days
    )


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = tuple(value.strip() for value in args.statuses.split(",") if value.strip())
    baseline, candidates = load_reaction_candidates(Path(args.promotion_dir), statuses, args.max_top_n)

    start, end, metadata = resolve_data_window("auto", "latest", ticker=args.ticker, lookback_months=args.lookback_months)
    config = StrategyConfig(
        ticker=args.ticker,
        start_date=start,
        end_date=end,
        lookback_months=args.lookback_months,
        stop_loss_pct=float(baseline["stop_loss_pct"]),
        take_profit_pct=float(baseline["take_profit_pct"]),
        max_exposure=float(baseline["max_exposure"]),
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
    report_index = report_prices.index
    split_at = max(args.min_train_days, int(len(report_index) * args.train_fraction))
    split_at = min(split_at, len(report_index) - args.min_validation_days)
    if split_at <= 0:
        raise RuntimeError("Not enough data to create train/validation split")
    train_index = report_index[:split_at]
    validation_index = report_index[split_at:]

    baseline_signals, _ = build_walk_forward_rule_signals(frame, rule_from_row(baseline), config.start_date)
    baseline_signals = baseline_signals.reindex(report_index).dropna(subset=["prob_up", "expected_return"])
    panel, meta = _panel_for_rules(candidates, frame, config.start_date)

    reaction_config = replace(
        config,
        stop_loss_pct=args.reaction_stop_loss_pct,
        take_profit_pct=args.reaction_take_profit_pct,
        max_exposure=args.max_shadow_exposure,
    )
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    top_values = [value for value in _parse_int_list(args.top_n_grid) if value <= len(meta)]
    for top_n in top_values:
        scoped_meta = meta.head(top_n).copy()
        keep = list(scoped_meta["name"])
        scoped_panel = pd.concat({group: panel[group][keep] for group in ["active", "prob", "expected"]}, axis=1)
        for min_vote_count in _parse_int_list(args.min_vote_count_grid):
            for min_vote_weight in _parse_float_list(args.min_vote_weight_grid):
                for prob_threshold in _parse_float_list(args.prob_threshold_grid):
                    reaction_signals, _ = build_reaction_signals(
                        baseline_signals,
                        scoped_panel,
                        scoped_meta,
                        report_index,
                        min_vote_count,
                        min_vote_weight,
                        prob_threshold,
                        args.max_shadow_exposure,
                    )
                    combined_signals = combine_baseline_and_reaction(baseline_signals, reaction_signals, report_index)
                    train_metrics = pd.DataFrame(
                        _evaluate_segment(
                            "train",
                            train_index,
                            config,
                            reaction_config,
                            report_prices,
                            frame.loc[config.start_date : config.end_date],
                            baseline_signals,
                            reaction_signals,
                            combined_signals,
                        )
                    )
                    score = _score_train_candidate(train_metrics, args.min_reaction_days)
                    row = {
                        "top_n": top_n,
                        "min_vote_count": min_vote_count,
                        "min_vote_weight": min_vote_weight,
                        "prob_threshold": prob_threshold,
                        "train_score": score,
                    }
                    for _, metric in train_metrics.iterrows():
                        prefix = f"train_{metric['layer']}"
                        row[f"{prefix}_annualized_return"] = metric["annualized_return"]
                        row[f"{prefix}_sharpe_ratio"] = metric["sharpe_ratio"]
                        row[f"{prefix}_max_drawdown"] = metric["max_drawdown"]
                        row[f"{prefix}_direction_precision"] = metric["direction_precision"]
                        row[f"{prefix}_active_days"] = metric["active_days"]
                        row[f"{prefix}_num_trades"] = metric["num_trades"]
                    rows.append(row)
                    if score > -900:
                        for metric in train_metrics.to_dict(orient="records"):
                            metric.update(row)
                            metric_rows.append(metric)

    sweep = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    if sweep.empty:
        raise RuntimeError("No reaction pool parameters evaluated")
    best = sweep.iloc[0]
    best_top_n = int(best["top_n"])
    best_meta = meta.head(best_top_n).copy()
    keep = list(best_meta["name"])
    best_panel = pd.concat({group: panel[group][keep] for group in ["active", "prob", "expected"]}, axis=1)
    reaction_signals, diagnostics = build_reaction_signals(
        baseline_signals,
        best_panel,
        best_meta,
        report_index,
        int(best["min_vote_count"]),
        float(best["min_vote_weight"]),
        float(best["prob_threshold"]),
        args.max_shadow_exposure,
    )
    combined_signals = combine_baseline_and_reaction(baseline_signals, reaction_signals, report_index)
    final_metrics = pd.DataFrame(
        _evaluate_segment(
            "train",
            train_index,
            config,
            reaction_config,
            report_prices,
            frame.loc[config.start_date : config.end_date],
            baseline_signals,
            reaction_signals,
            combined_signals,
        )
        + _evaluate_segment(
            "validation",
            validation_index,
            config,
            reaction_config,
            report_prices,
            frame.loc[config.start_date : config.end_date],
            baseline_signals,
            reaction_signals,
            combined_signals,
        )
        + _evaluate_segment(
            "full",
            report_index,
            config,
            reaction_config,
            report_prices,
            frame.loc[config.start_date : config.end_date],
            baseline_signals,
            reaction_signals,
            combined_signals,
        )
    )
    validation_metrics = final_metrics[final_metrics["segment"] == "validation"]
    validation_pass = _passes_validation(validation_metrics, args.min_reaction_days)
    latest = latest_reaction_snapshot(
        baseline,
        candidates.head(best_top_n),
        frame.loc[frame.index <= pd.Timestamp(config.end_date)],
        args.price_override,
        int(best["min_vote_count"]),
        float(best["prob_threshold"]),
    )

    sweep.to_csv(output_dir / "reaction_pool_time_split_sweep.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "reaction_pool_time_split_train_metrics.csv", index=False)
    final_metrics.to_csv(output_dir / "reaction_pool_time_split_final_metrics.csv", index=False)
    best_meta.to_csv(output_dir / "reaction_pool_time_split_members.csv", index=False)
    reaction_signals.to_csv(output_dir / "reaction_pool_time_split_reaction_signals.csv")
    combined_signals.to_csv(output_dir / "reaction_pool_time_split_combined_signals.csv")
    diagnostics.to_csv(output_dir / "reaction_pool_time_split_diagnostics.csv")

    payload = {
        "data_window": {
            **metadata,
            "resolved_start": config.start_date,
            "resolved_end": config.end_date,
            "rows": int(len(report_index)),
            "train_start": train_index[0].strftime("%Y-%m-%d"),
            "train_end": train_index[-1].strftime("%Y-%m-%d"),
            "validation_start": validation_index[0].strftime("%Y-%m-%d"),
            "validation_end": validation_index[-1].strftime("%Y-%m-%d"),
        },
        "selected_params": {
            "top_n": best_top_n,
            "min_vote_count": int(best["min_vote_count"]),
            "min_vote_weight": float(best["min_vote_weight"]),
            "prob_threshold": float(best["prob_threshold"]),
            "max_shadow_exposure": args.max_shadow_exposure,
            "train_score": float(best["train_score"]),
        },
        "validation_pass": validation_pass,
        "latest_reaction": latest,
        "metrics": final_metrics.to_dict(orient="records"),
    }
    report_path = write_report(output_dir, payload, sweep.head(12), final_metrics)
    payload["report"] = str(report_path.resolve())
    (output_dir / "reaction_pool_time_split_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def _pct(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.2%}"


def _num(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.2f}"


def write_report(output_dir: Path, payload: dict[str, Any], top_sweep: pd.DataFrame, final_metrics: pd.DataFrame) -> Path:
    metrics_display = final_metrics[
        [
            "segment",
            "layer",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "num_trades",
            "direction_precision",
            "active_days",
        ]
    ].copy()
    for column in ["annualized_return", "max_drawdown", "win_rate", "direction_precision"]:
        metrics_display[column] = metrics_display[column].map(_pct)
    for column in ["sharpe_ratio", "profit_factor"]:
        metrics_display[column] = metrics_display[column].map(_num)

    sweep_display = top_sweep[
        [
            "top_n",
            "min_vote_count",
            "min_vote_weight",
            "prob_threshold",
            "train_score",
            "train_reaction_shadow_direction_precision",
            "train_reaction_shadow_active_days",
            "train_combined_shadow_annualized_return",
            "train_combined_shadow_sharpe_ratio",
        ]
    ].copy()
    for column in ["train_reaction_shadow_direction_precision", "train_combined_shadow_annualized_return"]:
        sweep_display[column] = sweep_display[column].map(_pct)
    for column in ["train_score", "train_combined_shadow_sharpe_ratio"]:
        sweep_display[column] = sweep_display[column].map(_num)

    latest = payload["latest_reaction"]
    lines = [
        "# Reaction Pool Time-Split Validation",
        "",
        "## Decision",
        "",
        f"- Validation pass: {payload['validation_pass']}",
        f"- Selected params: `{payload['selected_params']}`",
        "",
        "## Data Split",
        "",
        f"- Train: {payload['data_window']['train_start']} ~ {payload['data_window']['train_end']}",
        f"- Validation: {payload['data_window']['validation_start']} ~ {payload['data_window']['validation_end']}",
        "",
        "## Final Metrics",
        "",
        metrics_display.to_markdown(index=False),
        "",
        "## Top Train Sweep Rows",
        "",
        sweep_display.to_markdown(index=False),
        "",
        "## Latest Reaction",
        "",
        f"- Date: {latest['date']}",
        f"- Baseline signal: {latest['baseline_signal']}",
        f"- Reaction alert: {latest['reaction_alert']}",
        f"- Active watchlist rules: {latest['active_watchlist_rules']}",
        f"- Weighted prob up: {_pct(latest['weighted_prob_up'])}",
        f"- Reason: {latest['reason']}",
        "",
        "## Files",
        "",
        f"- Sweep: {output_dir / 'reaction_pool_time_split_sweep.csv'}",
        f"- Final metrics: {output_dir / 'reaction_pool_time_split_final_metrics.csv'}",
        f"- Signals: {output_dir / 'reaction_pool_time_split_combined_signals.csv'}",
        "",
    ]
    path = output_dir / "reaction_pool_time_split_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-split validation for reaction pool parameter selection")
    parser.add_argument("--promotion-dir", default=str(PROJECT_ROOT / "outputs" / "candidate_promotion"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "reaction_pool_time_split"))
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--lookback-months", type=int, default=60)
    parser.add_argument("--statuses", default="watchlist_strong_recent_mid,watchlist_recent_only")
    parser.add_argument("--max-top-n", type=int, default=8)
    parser.add_argument("--top-n-grid", default="4,6,8")
    parser.add_argument("--min-vote-count-grid", default="2,3,4")
    parser.add_argument("--min-vote-weight-grid", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--prob-threshold-grid", default="0.52,0.53,0.54,0.55,0.56,0.58")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--min-train-days", type=int, default=420)
    parser.add_argument("--min-validation-days", type=int, default=180)
    parser.add_argument("--min-reaction-days", type=int, default=8)
    parser.add_argument("--max-shadow-exposure", type=float, default=0.25)
    parser.add_argument("--reaction-stop-loss-pct", type=float, default=0.025)
    parser.add_argument("--reaction-take-profit-pct", type=float, default=0.035)
    parser.add_argument("--price-override", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    run_validation(parse_args())


if __name__ == "__main__":
    main()
