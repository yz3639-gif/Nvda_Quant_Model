from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.analysis.robustness_check import monte_carlo_bootstrap, oos_degradation
from nvda_quant_model.analysis.statistical_tests import statistical_tests
from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import benchmark_metrics, calculate_metrics
from nvda_quant_model.backtest.walk_forward import walk_forward_signals
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.main import apply_signal_policy, optimize_signal_policy


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def candidate_configs(base: StrategyConfig) -> list[StrategyConfig]:
    """Deterministic, compact grid for expensive walk-forward optimization."""
    seeds = [
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=0, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=1, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.60, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.025, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.04, take_profit_pct=0.05, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.04, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.06, max_exposure=1.0, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=0.75, top_k_features=12),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=10),
        replace(base, rule_quantile=0.55, rule_max_filters=2, stop_loss_pct=0.03, take_profit_pct=0.05, max_exposure=1.0, top_k_features=16),
    ]
    grid = list(
        product(
            [0.55, 0.60],
            [0, 1, 2],
            [0.025, 0.03, 0.04],
            [0.04, 0.05, 0.06],
            [0.75, 1.0],
            [10, 12, 16],
        )
    )
    configs: list[StrategyConfig] = []
    for rule_quantile, rule_max_filters, stop_loss, take_profit, max_exposure, top_k in grid:
        if take_profit <= stop_loss:
            continue
        configs.append(
            replace(
                base,
                rule_quantile=rule_quantile,
                rule_max_filters=rule_max_filters,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                max_exposure=max_exposure,
                top_k_features=top_k,
            )
        )

    def priority(cfg: StrategyConfig) -> tuple[float, ...]:
        return (
            abs(cfg.rule_quantile - base.rule_quantile),
            abs(cfg.rule_max_filters - base.rule_max_filters),
            abs(cfg.stop_loss_pct - base.stop_loss_pct),
            abs(cfg.take_profit_pct - base.take_profit_pct),
            abs(cfg.max_exposure - base.max_exposure),
            abs(cfg.top_k_features - base.top_k_features),
        )

    unique: dict[tuple[Any, ...], StrategyConfig] = {}
    for cfg in [*seeds, *sorted(configs, key=priority)]:
        key = (
            cfg.rule_quantile,
            cfg.rule_max_filters,
            cfg.stop_loss_pct,
            cfg.take_profit_pct,
            cfg.max_exposure,
            cfg.top_k_features,
        )
        unique[key] = cfg
    return list(unique.values())


def score_metrics(metrics: dict[str, float], stats: dict[str, Any], oos: dict[str, Any], wf: pd.DataFrame) -> float:
    hard_fail_penalty = 0.0
    hard_fail_penalty += 2.0 if metrics["annualized_return"] <= 0.15 else 0.0
    hard_fail_penalty += 2.0 if metrics["sharpe_ratio"] <= 1.0 else 0.0
    hard_fail_penalty += 2.0 if abs(metrics["max_drawdown"]) >= 0.20 else 0.0
    hard_fail_penalty += 1.5 if metrics["win_rate"] <= 0.55 else 0.0
    hard_fail_penalty += 1.5 if metrics["profit_factor"] <= 1.5 else 0.0
    hard_fail_penalty += 1.5 if stats["p_value"] >= 0.05 else 0.0
    hard_fail_penalty += 2.0 if not oos["passes_20pct_rule"] else 0.0

    wf_stability = wf["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).std(ddof=0) if not wf.empty else 2.0
    return float(
        metrics["sharpe_ratio"]
        + 0.75 * metrics["annualized_return"]
        + 0.30 * metrics["calmar_ratio"]
        + 0.12 * min(metrics["profit_factor"], 4.0)
        - 0.15 * wf_stability
        - hard_fail_penalty
    )


def evaluate_config(
    cfg: StrategyConfig,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    trainable = frame.dropna(subset=["target_return", "target_direction"])
    raw_signals, wf, feature_importance = walk_forward_signals(trainable, feature_columns, cfg, cfg.start_date)
    report_prices = prices.loc[(prices.index >= pd.Timestamp(cfg.start_date)) & (prices.index <= pd.Timestamp(cfg.end_date))]
    raw_signals = raw_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])

    selected_cfg = optimize_signal_policy(raw_signals, report_prices, cfg)
    signals = apply_signal_policy(raw_signals, selected_cfg)
    result = BacktestEngine(selected_cfg).backtest(signals, report_prices)

    benchmark_close = external[MACRO_TICKERS["SP500"]].reindex(report_prices.index).ffill().dropna()
    _, _, benchmark_returns = benchmark_metrics(benchmark_close, selected_cfg.initial_capital)
    benchmark_returns = benchmark_returns.reindex(result.daily_returns.index).fillna(0.0)
    active_mask = signals["position"].shift().reindex(result.daily_returns.index).fillna(0.0) != 0.0
    stats = statistical_tests(result.daily_returns, benchmark_returns, active_mask=active_mask)
    robustness = monte_carlo_bootstrap(result.daily_returns, n_sims=500, seed=selected_cfg.random_state)

    oos_cutoff = result.equity_curve.index.max() - pd.Timedelta(days=45)
    oos_returns = result.daily_returns.loc[result.daily_returns.index >= oos_cutoff]
    oos_equity = selected_cfg.initial_capital * (1 + oos_returns).cumprod()
    oos_metrics = calculate_metrics(oos_equity, oos_returns, pd.DataFrame(), selected_cfg.initial_capital)
    oos = oos_degradation(result.metrics, oos_metrics)
    score = score_metrics(result.metrics, stats, oos, wf)

    payload = {
        "score": score,
        "config": asdict(cfg),
        "selected_config": asdict(selected_cfg),
        "metrics": result.metrics,
        "statistical_tests": stats,
        "robustness": robustness,
        "oos": oos,
        "walk_forward_mean_sharpe": float(wf["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()) if not wf.empty else 0.0,
        "walk_forward_std_sharpe": float(wf["sharpe_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).std(ddof=0)) if not wf.empty else 0.0,
        "active_days": int((signals["position"] != 0).sum()),
        "feature_importance_top": feature_importance.head(12).to_dict(),
    }
    return payload, signals, wf


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    selected = result["selected_config"]
    metrics = result["metrics"]
    stats = result["statistical_tests"]
    oos = result["oos"]
    return {
        "score": result["score"],
        "rule_quantile": cfg["rule_quantile"],
        "rule_max_filters": cfg["rule_max_filters"],
        "stop_loss_pct": cfg["stop_loss_pct"],
        "take_profit_pct": cfg["take_profit_pct"],
        "max_exposure": cfg["max_exposure"],
        "top_k_features": cfg["top_k_features"],
        "selected_threshold": selected["signal_threshold"],
        "selected_min_expected_return": selected["min_expected_return"],
        "annualized_return": metrics["annualized_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar_ratio": metrics["calmar_ratio"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "num_trades": metrics["num_trades"],
        "p_value": stats["p_value"],
        "oos_degradation": oos["sharpe_degradation"],
        "oos_pass": oos["passes_20pct_rule"],
        "wf_mean_sharpe": result["walk_forward_mean_sharpe"],
        "wf_std_sharpe": result["walk_forward_std_sharpe"],
        "active_days": result["active_days"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize NVDA model parameters with walk-forward evaluation")
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--include-fundamentals", action="store_true")
    parser.add_argument("--no-peer-events", action="store_true")
    parser.add_argument("--max-runs", type=int, default=12)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "optimization"))
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

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    configs = candidate_configs(base)[: args.max_runs]
    for idx, cfg in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] rule_q={cfg.rule_quantile} filters={cfg.rule_max_filters} "
            f"stop={cfg.stop_loss_pct:.3f} take={cfg.take_profit_pct:.3f} exposure={cfg.max_exposure:.2f} top_k={cfg.top_k_features}",
            flush=True,
        )
        payload, _, _ = evaluate_config(cfg, prices, external, frame, feature_columns)
        results.append(payload)
        if best is None or payload["score"] > best["score"]:
            best = payload
            print(
                f"  new best score={payload['score']:.3f} "
                f"sharpe={payload['metrics']['sharpe_ratio']:.2f} "
                f"ann={payload['metrics']['annualized_return']:.2%} "
                f"mdd={payload['metrics']['max_drawdown']:.2%}",
                flush=True,
            )

    rows = pd.DataFrame([flatten_result(result) for result in results]).sort_values("score", ascending=False)
    rows.to_csv(output_dir / "optimization_results.csv", index=False)
    if best is not None:
        (output_dir / "best_result.json").write_text(
            json.dumps(best, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print("BEST")
        print(json.dumps(flatten_result(best), indent=2, ensure_ascii=False, default=_json_default))
    print(f"Saved: {output_dir / 'optimization_results.csv'}")


if __name__ == "__main__":
    main()
