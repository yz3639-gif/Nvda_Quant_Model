from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import benchmark_metrics, calculate_metrics
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.optimize import evaluate_config, flatten_result, score_metrics


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def variant_grid(base: StrategyConfig) -> list[tuple[str, StrategyConfig]]:
    variants: list[tuple[str, StrategyConfig]] = [
        ("base_optimized", base),
        ("wf_189_42", replace(base, train_window=189, test_window=42)),
        ("wf_252_42", replace(base, train_window=252, test_window=42)),
        ("wf_315_63", replace(base, train_window=315, test_window=63)),
        ("wf_378_63", replace(base, train_window=378, test_window=63)),
        ("double_cost", replace(base, commission=0.002, slippage=0.001)),
        ("half_cost", replace(base, commission=0.0005, slippage=0.00025)),
        ("no_cost", replace(base, commission=0.0, slippage=0.0)),
        ("exposure_075", replace(base, max_exposure=0.75)),
        ("stop_025_take_050", replace(base, stop_loss_pct=0.025, take_profit_pct=0.05)),
        ("stop_030_take_050", replace(base, stop_loss_pct=0.03, take_profit_pct=0.05)),
        ("stop_040_take_040", replace(base, stop_loss_pct=0.04, take_profit_pct=0.04)),
        ("stop_040_take_060", replace(base, stop_loss_pct=0.04, take_profit_pct=0.06)),
        ("stop_050_take_060", replace(base, stop_loss_pct=0.05, take_profit_pct=0.06)),
        ("rule_q_060", replace(base, rule_quantile=0.60)),
        ("rule_filters_0", replace(base, rule_max_filters=0)),
        ("rule_filters_1", replace(base, rule_max_filters=1)),
        ("top_k_10", replace(base, top_k_features=10)),
        ("top_k_16", replace(base, top_k_features=16)),
        ("threshold_055", replace(base, signal_threshold=0.55)),
        ("threshold_060", replace(base, signal_threshold=0.60)),
    ]
    unique: dict[tuple[Any, ...], tuple[str, StrategyConfig]] = {}
    for label, cfg in variants:
        key = tuple(asdict(cfg).items())
        unique[key] = (label, cfg)
    return list(unique.values())


def evaluate_variant(
    label: str,
    cfg: StrategyConfig,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    payload, signals, wf = evaluate_config(cfg, prices, external, frame, feature_columns)
    row = flatten_result(payload)
    row["label"] = label
    row["score"] = score_metrics(payload["metrics"], payload["statistical_tests"], payload["oos"], wf)
    payload["label"] = label
    return row, payload, signals, wf


def rebuild_result_from_signals(cfg_payload: dict[str, Any], signals: pd.DataFrame, prices: pd.DataFrame) -> tuple[StrategyConfig, Any]:
    selected = cfg_payload["selected_config"]
    selected_cfg = StrategyConfig(**selected)
    report_prices = prices.loc[(prices.index >= pd.Timestamp(selected_cfg.start_date)) & (prices.index <= pd.Timestamp(selected_cfg.end_date))]
    aligned_signals = signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
    result = BacktestEngine(selected_cfg).backtest(aligned_signals, report_prices)
    return selected_cfg, result


def period_breakdown(selected_cfg: StrategyConfig, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    start_ts = pd.Timestamp(selected_cfg.start_date)
    end_ts = pd.Timestamp(selected_cfg.end_date)
    periods: list[tuple[str, str, str]] = []
    for year in range(start_ts.year, end_ts.year + 1):
        year_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        label = f"{year}" if year_start.month == 1 and year_start.day == 1 else f"{year}_partial"
        if year_start <= year_end:
            periods.append((label, year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
    periods.append(
        ("last_126d", (end_ts - pd.Timedelta(days=190)).strftime("%Y-%m-%d"), selected_cfg.end_date)
    )
    rows = []
    for label, start, end in periods:
        seg_prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
        seg_signals = signals.reindex(seg_prices.index).dropna(subset=["prob_up", "expected_return"])
        if len(seg_prices) < 10 or seg_signals.empty:
            continue
        result = BacktestEngine(selected_cfg).backtest(seg_signals, seg_prices)
        rows.append({"period": label, "start": start, "end": end, **result.metrics})
    return pd.DataFrame(rows)


def oos_windows(selected_cfg: StrategyConfig, result: Any) -> pd.DataFrame:
    rows = []
    for days in [30, 45, 60, 90, 126]:
        cutoff = result.equity_curve.index.max() - pd.Timedelta(days=days)
        returns = result.daily_returns.loc[result.daily_returns.index >= cutoff]
        if returns.empty:
            continue
        equity = selected_cfg.initial_capital * (1 + returns).cumprod()
        metrics = calculate_metrics(equity, returns, pd.DataFrame(), selected_cfg.initial_capital)
        rows.append({"window_days": days, "start": cutoff.strftime("%Y-%m-%d"), **metrics})
    return pd.DataFrame(rows)


def benchmark_comparison(selected_cfg: StrategyConfig, prices: pd.DataFrame, external: pd.DataFrame, strategy_metrics: dict[str, float]) -> pd.DataFrame:
    report_index = prices.loc[selected_cfg.start_date : selected_cfg.end_date].index
    rows = [{"name": "strategy", **strategy_metrics}]
    nvda_metrics, _, _ = benchmark_metrics(prices["Close"].reindex(report_index).ffill(), selected_cfg.initial_capital)
    rows.append({"name": "NVDA_buy_hold", **nvda_metrics})
    spx = external[MACRO_TICKERS["SP500"]].reindex(report_index).ffill()
    spx_metrics, _, _ = benchmark_metrics(spx, selected_cfg.initial_capital)
    rows.append({"name": "SP500_buy_hold", **spx_metrics})
    if "QQQ" in external.columns:
        qqq_metrics, _, _ = benchmark_metrics(external["QQQ"].reindex(report_index).ffill(), selected_cfg.initial_capital)
        rows.append({"name": "QQQ_buy_hold", **qqq_metrics})
    if "SMH" in external.columns:
        smh_metrics, _, _ = benchmark_metrics(external["SMH"].reindex(report_index).ffill(), selected_cfg.initial_capital)
        rows.append({"name": "SMH_buy_hold", **smh_metrics})
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    result_rows: pd.DataFrame,
    period_rows: pd.DataFrame,
    oos_rows: pd.DataFrame,
    benchmark_rows: pd.DataFrame,
) -> Path:
    best = result_rows.sort_values("score", ascending=False).iloc[0]
    lines = [
        "# NVDA Extended Backtest Summary",
        "",
        "## Best Variant",
        "",
        f"- Label: {best['label']}",
        f"- Score: {best['score']:.3f}",
        f"- Annualized return: {best['annualized_return']:.2%}",
        f"- Sharpe: {best['sharpe_ratio']:.2f}",
        f"- Max drawdown: {best['max_drawdown']:.2%}",
        f"- Calmar: {best['calmar_ratio']:.2f}",
        f"- Win rate: {best['win_rate']:.2%}",
        f"- Profit factor: {best['profit_factor']:.2f}",
        "",
        "## Top Variants",
        "",
        result_rows.head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Period Breakdown",
        "",
        period_rows.to_markdown(index=False, floatfmt=".4f") if not period_rows.empty else "No period rows.",
        "",
        "## OOS Windows",
        "",
        oos_rows.to_markdown(index=False, floatfmt=".4f") if not oos_rows.empty else "No OOS rows.",
        "",
        "## Benchmark Comparison",
        "",
        benchmark_rows.to_markdown(index=False, floatfmt=".4f") if not benchmark_rows.empty else "No benchmark rows.",
        "",
    ]
    path = output_dir / "extended_backtest_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded NVDA backtest validation matrix")
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--include-fundamentals", action="store_true")
    parser.add_argument("--no-peer-events", action="store_true")
    parser.add_argument("--max-variants", type=int, default=21)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "extended_backtests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start, end, _ = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(
        ticker=args.ticker,
        start_date=start,
        end_date=end,
        lookback_months=args.lookback_months,
        include_fundamentals=args.include_fundamentals,
        include_peer_events=not args.no_peer_events,
    )

    prices, external = load_market_data(
        base.ticker,
        base.start_date,
        base.end_date,
        cache_dir=PROJECT_ROOT / "cache",
        force_refresh=args.force_refresh or args.end.lower() in {"latest", "auto", "today", "now"},
    )
    peer_ohlcv = (
        load_peer_ohlcv_panel(
            warmup_start(base.start_date),
            base.end_date,
            cache_dir=PROJECT_ROOT / "cache",
            force_refresh=args.force_refresh or args.end.lower() in {"latest", "auto", "today", "now"},
        )
        if base.include_peer_events
        else {}
    )
    frame, feature_columns = build_model_frame(
        prices,
        external,
        base.start_date,
        base.end_date,
        base.ticker,
        include_fundamentals=base.include_fundamentals,
        peer_ohlcv=peer_ohlcv,
    )

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    signal_map: dict[str, pd.DataFrame] = {}
    wf_rows: list[pd.DataFrame] = []
    variants = variant_grid(base)[: args.max_variants]
    for idx, (label, cfg) in enumerate(variants, start=1):
        print(f"[{idx}/{len(variants)}] {label}", flush=True)
        row, payload, signals, wf = evaluate_variant(label, cfg, prices, external, frame, feature_columns)
        rows.append(row)
        signal_map[label] = signals
        payloads.append(payload)
        wf_labeled = wf.copy()
        wf_labeled.insert(0, "label", label)
        wf_rows.append(wf_labeled)

    result_rows = pd.DataFrame(rows).sort_values("score", ascending=False)
    result_rows.to_csv(output_dir / "extended_results.csv", index=False)
    wf_all = pd.concat(wf_rows, ignore_index=True) if wf_rows else pd.DataFrame()
    wf_all.to_csv(output_dir / "walk_forward_all_variants.csv", index=False)

    best_label = str(result_rows.iloc[0]["label"])
    best_payload = next(payload for payload in payloads if payload["label"] == best_label)
    selected_cfg, best_result = rebuild_result_from_signals(best_payload, signal_map[best_label], prices)
    period_rows = period_breakdown(selected_cfg, signal_map[best_label], prices)
    period_rows.to_csv(output_dir / "period_breakdown.csv", index=False)
    oos_rows = oos_windows(selected_cfg, best_result)
    oos_rows.to_csv(output_dir / "oos_windows.csv", index=False)
    benchmark_rows = benchmark_comparison(selected_cfg, prices, external, best_result.metrics)
    benchmark_rows.to_csv(output_dir / "benchmark_comparison.csv", index=False)

    best_payload["best_label"] = best_label
    (output_dir / "best_payload.json").write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    summary_path = write_summary(output_dir, result_rows, period_rows, oos_rows, benchmark_rows)
    print(f"Best: {best_label}")
    print(result_rows.head(5).to_string(index=False))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
