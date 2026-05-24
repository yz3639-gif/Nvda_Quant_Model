from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import benchmark_metrics
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.main import factor_signals, next_us_trading_day
from nvda_quant_model.news_sentiment import fetch_live_news, save_news_outputs
from nvda_quant_model.precision_search import build_walk_forward_rule_signals
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


def filter_candidates(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = rows.copy()
    filtered = filtered[filtered["dir_active_days"] >= args.min_active_days]
    filtered = filtered[filtered["bt_num_trades"] >= args.min_trades]
    filtered = filtered[filtered["dir_precision"] >= args.min_precision]
    filtered = filtered[filtered["dir_worst_year_precision"] >= args.min_worst_year_precision]
    filtered = filtered[filtered["bt_sharpe_ratio"] >= args.min_sharpe]
    filtered = filtered[filtered["bt_annualized_return"] >= args.min_annualized]
    filtered = filtered[filtered["bt_max_drawdown"].abs() <= args.max_drawdown]
    filtered = filtered[filtered["bt_profit_factor"] >= args.min_profit_factor]
    sort_col = "long_score" if "long_score" in filtered.columns else "score"
    return filtered.sort_values([sort_col, "dir_precision", "bt_sharpe_ratio"], ascending=[False, False, False])


def candidate_weight(row: pd.Series) -> float:
    precision_edge = max(float(row["dir_precision"]) - 0.50, 0.0)
    stability = max(float(row["dir_worst_year_precision"]) - 0.50, 0.0)
    sharpe = max(float(row["bt_sharpe_ratio"]), 0.0)
    drawdown_penalty = 1.0 / (1.0 + 8.0 * abs(float(row["bt_max_drawdown"])))
    sample_bonus = min(float(row["dir_active_days"]) / 80.0, 1.0)
    return float((precision_edge * 4.0 + stability * 2.0 + sharpe * 0.40) * drawdown_penalty * (0.75 + 0.25 * sample_bonus))


def build_rule_signal_panel(
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    report_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_cols: dict[str, pd.Series] = {}
    active_cols: dict[str, pd.Series] = {}
    expected_cols: dict[str, pd.Series] = {}
    meta_rows: list[dict[str, Any]] = []

    for idx, (_, row) in enumerate(candidates.iterrows(), start=1):
        rule = rule_from_row(row)
        signals, _ = build_walk_forward_rule_signals(frame, rule, report_start)
        name = f"rule_{idx:02d}"
        prob_cols[name] = signals["prob_up"]
        active_cols[name] = signals["position"] > 0
        expected_cols[name] = signals["expected_return"]
        meta_rows.append(
            {
                "name": name,
                "label": row["label"],
                "weight": candidate_weight(row),
                "dir_precision": float(row["dir_precision"]),
                "dir_active_days": int(row["dir_active_days"]),
                "bt_num_trades": int(row["bt_num_trades"]),
                "bt_sharpe_ratio": float(row["bt_sharpe_ratio"]),
                "bt_annualized_return": float(row["bt_annualized_return"]),
                "bt_max_drawdown": float(row["bt_max_drawdown"]),
            }
        )

    probs = pd.DataFrame(prob_cols).sort_index()
    active = pd.DataFrame(active_cols).sort_index().fillna(False).infer_objects(copy=False)
    expected = pd.DataFrame(expected_cols).sort_index().fillna(0.0)
    meta = pd.DataFrame(meta_rows)
    return probs, active, expected, meta


def build_ensemble_signals(
    probs: pd.DataFrame,
    active: pd.DataFrame,
    expected: pd.DataFrame,
    meta: pd.DataFrame,
    frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    aligned_index = probs.index.union(active.index).sort_values()
    probs = probs.reindex(aligned_index).fillna(0.5)
    active = active.reindex(aligned_index).fillna(False)
    expected = expected.reindex(aligned_index).fillna(0.0)
    weights = meta.set_index("name")["weight"].reindex(probs.columns).fillna(1.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=probs.columns)
    weights = weights / weights.sum()

    active_weight = active.astype(float).mul(weights, axis=1).sum(axis=1)
    active_count = active.sum(axis=1)
    weighted_prob = probs.mul(weights, axis=1).sum(axis=1)
    active_prob = probs.where(active).mul(weights, axis=1).sum(axis=1) / active.astype(float).mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
    prob_up = active_prob.fillna(weighted_prob).clip(0.01, 0.99)
    expected_return = expected.where(active).mul(weights, axis=1).sum(axis=1) / active.astype(float).mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
    expected_return = expected_return.fillna(0.0).clip(lower=0.0)

    direction = ((active_weight >= args.min_vote_weight) & (active_count >= args.min_vote_count) & (prob_up >= args.prob_threshold)).astype(int)
    confidence = np.maximum(prob_up, 1.0 - prob_up)
    vote_strength = (active_weight / max(args.min_vote_weight, 1e-9)).clip(upper=1.5) / 1.5
    probability_strength = ((prob_up - 0.50) / 0.25).clip(lower=0.0, upper=1.0)

    if args.dynamic_size:
        volatility = frame.reindex(aligned_index)["volatility_20"].replace(0, np.nan)
        annual_vol = volatility * np.sqrt(252)
        vol_scalar = (args.target_vol / annual_vol).clip(lower=args.min_vol_scalar, upper=args.max_vol_scalar).fillna(1.0)
        raw_position = args.max_exposure * vote_strength * probability_strength * vol_scalar
        position = raw_position.clip(lower=args.min_position, upper=args.max_exposure)
        position = position.where(direction == 1, 0.0)
    else:
        position = pd.Series(args.max_exposure, index=aligned_index).where(direction == 1, 0.0)

    signals = pd.DataFrame(
        {
            "direction": direction,
            "confidence": confidence,
            "expected_return": expected_return.where(direction == 1, 0.0),
            "prob_up": prob_up.where(direction == 1, 0.5),
            "prob_down": 1.0 - prob_up.where(direction == 1, 0.5),
            "position": position,
            "active_rule_count": active_count,
            "active_rule_weight": active_weight,
            "raw_prob_up": prob_up,
        },
        index=aligned_index,
    )
    return signals


def directional_accuracy(signals: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    aligned = frame[["target_return", "target_direction"]].join(signals[["position", "prob_up", "raw_prob_up"]], how="inner")
    active = aligned["position"] > 0
    active_rows = aligned.loc[active].dropna(subset=["target_return", "target_direction"])
    if active_rows.empty:
        return {
            "active_days": 0,
            "coverage": 0.0,
            "direction_precision": 0.0,
            "avg_next_return": 0.0,
            "median_next_return": 0.0,
            "brier_active": None,
        }
    y = active_rows["target_direction"].astype(float)
    p = active_rows["prob_up"].clip(0.01, 0.99)
    yearly = active_rows.assign(year=active_rows.index.year).groupby("year")["target_direction"].mean()
    return {
        "active_days": int(len(active_rows)),
        "coverage": float(len(active_rows) / len(aligned)),
        "direction_precision": float((active_rows["target_return"] > 0).mean()),
        "avg_next_return": float(active_rows["target_return"].mean()),
        "median_next_return": float(active_rows["target_return"].median()),
        "brier_active": float(np.mean((p - y) ** 2)),
        "worst_year_precision": float(yearly.min()) if not yearly.empty else 0.0,
        "year_precision": {str(year): float(value) for year, value in yearly.items()},
    }


def calibration_table(signals: pd.DataFrame, frame: pd.DataFrame, bins: int = 6) -> pd.DataFrame:
    aligned = frame[["target_direction"]].join(signals[["position", "prob_up", "raw_prob_up"]], how="inner")
    active = aligned.loc[aligned["position"] > 0].dropna(subset=["target_direction", "prob_up"])
    if active.empty:
        return pd.DataFrame()
    active = active.copy()
    active["bin"] = pd.cut(active["prob_up"], bins=np.linspace(0.50, 0.90, bins + 1), include_lowest=True)
    table = active.groupby("bin", observed=False).agg(
        samples=("target_direction", "size"),
        avg_prob=("prob_up", "mean"),
        realized_up=("target_direction", "mean"),
    )
    table["calibration_error"] = table["avg_prob"] - table["realized_up"]
    return table.reset_index().astype({"bin": str})


def regime_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    scoped = frame.copy()
    masks: dict[str, pd.Series] = {}
    if "VIX" in scoped:
        vix = scoped["VIX"]
        masks["vix_low"] = vix <= vix.rolling(252, min_periods=60).quantile(0.35).fillna(vix.quantile(0.35))
        masks["vix_high"] = vix >= vix.rolling(252, min_periods=60).quantile(0.70).fillna(vix.quantile(0.70))
    if "holiday_week" in scoped:
        masks["holiday_week"] = scoped["holiday_week"] == 1
        masks["non_holiday_week"] = scoped["holiday_week"] == 0
    if "SMH_return" in scoped:
        masks["semiconductor_strong"] = scoped["SMH_return"] > 0
        masks["semiconductor_weak"] = scoped["SMH_return"] <= 0
    if "peer_positive_breadth_5d" in scoped:
        masks["peer_breadth_strong"] = scoped["peer_positive_breadth_5d"] >= 0.625
        masks["peer_breadth_weak"] = scoped["peer_positive_breadth_5d"] <= 0.375
    if {"volatility_20", "volatility_60"}.issubset(scoped.columns):
        masks["realized_vol_calm"] = scoped["volatility_20"] < scoped["volatility_60"]
        masks["realized_vol_hot"] = scoped["volatility_20"] >= scoped["volatility_60"] * 1.2
    return {name: mask.fillna(False).astype(bool) for name, mask in masks.items()}


def regime_report(signals: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame[["target_return", "target_direction"]].join(signals[["position", "prob_up"]], how="inner")
    rows: list[dict[str, Any]] = []
    for name, mask in regime_masks(frame).items():
        mask = mask.reindex(aligned.index).fillna(False).astype(bool)
        scoped = aligned.loc[mask]
        active = scoped.loc[scoped["position"] > 0].dropna(subset=["target_return", "target_direction"])
        rows.append(
            {
                "regime": name,
                "regime_days": int(mask.sum()),
                "active_days": int(len(active)),
                "direction_precision": float((active["target_return"] > 0).mean()) if len(active) else np.nan,
                "avg_next_return": float(active["target_return"].mean()) if len(active) else 0.0,
                "median_next_return": float(active["target_return"].median()) if len(active) else 0.0,
                "avg_prob_up": float(active["prob_up"].mean()) if len(active) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def apply_regime_gates(signals: pd.DataFrame, frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    gated = signals.copy()
    masks = regime_masks(frame)
    blocked = pd.Series(False, index=gated.index)
    if args.block_realized_vol_hot and "realized_vol_hot" in masks:
        blocked |= masks["realized_vol_hot"].reindex(gated.index).fillna(False).astype(bool)
    if args.block_vix_high and "vix_high" in masks:
        blocked |= masks["vix_high"].reindex(gated.index).fillna(False).astype(bool)
    if args.block_holiday_week and "holiday_week" in masks:
        blocked |= masks["holiday_week"].reindex(gated.index).fillna(False).astype(bool)
    if args.block_semiconductor_weak and "semiconductor_weak" in masks:
        blocked |= masks["semiconductor_weak"].reindex(gated.index).fillna(False).astype(bool)
    if blocked.any():
        gated.loc[blocked, ["direction", "position", "expected_return"]] = 0.0
        gated.loc[blocked, ["prob_up", "prob_down"]] = 0.5
    gated["regime_blocked"] = blocked.astype(int)
    return gated


def latest_ensemble_prediction(signals: pd.DataFrame, frame: pd.DataFrame, price_override: float | None) -> dict[str, Any]:
    latest_row = frame.tail(1)
    as_of_date = latest_row.index[-1]
    latest_signal = signals.reindex([as_of_date]).fillna(
        {
            "prob_up": 0.5,
            "prob_down": 0.5,
            "position": 0.0,
            "direction": 0,
            "expected_return": 0.0,
            "active_rule_count": 0,
            "active_rule_weight": 0.0,
        }
    ).iloc[-1]
    model_close = float(latest_row["Close"].iloc[0])
    current_price = float(price_override) if price_override is not None else model_close
    expected_return = float(latest_signal.get("expected_return", 0.0))
    next_date = next_us_trading_day(as_of_date)
    return {
        "date": next_date.strftime("%Y-%m-%d"),
        "weekday": next_date.day_name(),
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "model_close": round(model_close, 4),
        "current_price": round(current_price, 4),
        "price_source": "user_price_override" if price_override is not None else "latest_daily_close",
        "signal": int(latest_signal.get("direction", 0)),
        "position": round(float(latest_signal.get("position", 0.0)), 4),
        "prob_up": round(float(latest_signal.get("prob_up", 0.5)), 4),
        "prob_down": round(float(latest_signal.get("prob_down", 0.5)), 4),
        "expected_return": round(expected_return, 6),
        "target_price": round(current_price * (1.0 + expected_return), 4),
        "factor_signals": factor_signals(latest_row.iloc[0]),
        "active_rule_count": int(0 if pd.isna(latest_signal.get("active_rule_count", 0)) else latest_signal.get("active_rule_count", 0)),
        "active_rule_weight": round(float(0.0 if pd.isna(latest_signal.get("active_rule_weight", 0.0)) else latest_signal.get("active_rule_weight", 0.0)), 4),
    }


def run_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.results_csv)
    candidates = filter_candidates(rows, args).head(args.top_n)
    if candidates.empty:
        raise RuntimeError("No candidates satisfy ensemble filters")

    start, end, metadata = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(ticker=args.ticker, start_date=start, end_date=end, lookback_months=args.lookback_months, include_peer_events=True)
    prices, external = load_market_data(base.ticker, base.start_date, base.end_date, cache_dir=PROJECT_ROOT / "cache")
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(base.start_date), base.end_date, cache_dir=PROJECT_ROOT / "cache")
    frame, _ = build_model_frame(
        prices,
        external,
        base.start_date,
        base.end_date,
        base.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
    )
    probs, active, expected, meta = build_rule_signal_panel(candidates, frame, base.start_date)
    signals = build_ensemble_signals(probs, active, expected, meta, frame, args)
    signals = apply_regime_gates(signals, frame, args)
    report_prices = prices.loc[base.start_date : base.end_date]
    signals = signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])

    stop_loss = float(candidates["stop_loss_pct"].median())
    take_profit = float(candidates["take_profit_pct"].median())
    cfg = replace(
        base,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        max_exposure=args.max_exposure,
    )
    result = BacktestEngine(cfg).backtest(signals, report_prices)
    dmetrics = directional_accuracy(signals, frame.loc[base.start_date : base.end_date])
    benchmark_close = external[MACRO_TICKERS["SP500"]].reindex(report_prices.index).ffill().dropna()
    benchmark, _, _ = benchmark_metrics(benchmark_close, cfg.initial_capital)
    calibration = calibration_table(signals, frame.loc[base.start_date : base.end_date], bins=args.calibration_bins)
    regimes = regime_report(signals, frame.loc[base.start_date : base.end_date])
    latest = latest_ensemble_prediction(signals, frame.loc[frame.index <= pd.Timestamp(base.end_date)], args.price_override)
    news_overlay = None
    if args.include_live_news:
        articles = fetch_live_news(days=args.news_days)
        news_payload = save_news_outputs(articles, output_dir / "news_live", price_index=report_prices.index)
        news_overlay = news_payload["overlay"]
        latest["news_overlay"] = {
            "signal": news_overlay.get("signal", 0),
            "sentiment_score": round(float(news_overlay.get("sentiment_score", 0.0)), 4),
            "article_count": int(news_overlay.get("article_count", 0)),
            "risk_flags": news_overlay.get("risk_flags", []),
            "confidence_adjustment": round(float(news_overlay.get("confidence_adjustment", 0.0)), 4),
        }
        if latest["signal"] == 1 and news_overlay.get("signal") == -1:
            latest["news_adjusted_signal"] = 0
            latest["news_adjustment_reason"] = "Live news layer blocks a long signal because bearish/risk news is active."
        else:
            latest["news_adjusted_signal"] = latest["signal"]

    signals.to_csv(output_dir / "ensemble_signals.csv")
    meta.to_csv(output_dir / "ensemble_members.csv", index=False)
    calibration.to_csv(output_dir / "calibration.csv", index=False)
    regimes.to_csv(output_dir / "regime_report.csv", index=False)
    result.equity_curve.to_csv(output_dir / "ensemble_equity_curve.csv")
    result.trades.to_csv(output_dir / "ensemble_trades.csv", index=False)
    payload = {
        "data_window": {
            **metadata,
            "resolved_start": base.start_date,
            "resolved_end": base.end_date,
            "rows": int(len(report_prices)),
        },
        "filters": {
            "top_n": args.top_n,
            "min_active_days": args.min_active_days,
            "min_trades": args.min_trades,
            "min_precision": args.min_precision,
            "min_worst_year_precision": args.min_worst_year_precision,
            "min_sharpe": args.min_sharpe,
            "min_annualized": args.min_annualized,
            "max_drawdown": args.max_drawdown,
            "min_profit_factor": args.min_profit_factor,
            "min_vote_weight": args.min_vote_weight,
            "min_vote_count": args.min_vote_count,
            "prob_threshold": args.prob_threshold,
            "dynamic_size": args.dynamic_size,
            "block_realized_vol_hot": args.block_realized_vol_hot,
            "block_vix_high": args.block_vix_high,
            "block_holiday_week": args.block_holiday_week,
            "block_semiconductor_weak": args.block_semiconductor_weak,
        },
        "member_count": int(len(meta)),
        "stop_loss_pct": stop_loss,
        "take_profit_pct": take_profit,
        "metrics": result.metrics,
        "benchmark_metrics": benchmark,
        "directional_metrics": dmetrics,
        "latest_prediction": latest,
        "news_overlay": news_overlay,
        "paths": {
            "signals": str((output_dir / "ensemble_signals.csv").resolve()),
            "members": str((output_dir / "ensemble_members.csv").resolve()),
            "calibration": str((output_dir / "calibration.csv").resolve()),
            "regime_report": str((output_dir / "regime_report.csv").resolve()),
            "trades": str((output_dir / "ensemble_trades.csv").resolve()),
        },
    }
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-rule ensemble, calibration, regimes, and dynamic sizing")
    parser.add_argument("--results-csv", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer" / "long_run_results.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "ensemble_research"))
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-active-days", type=int, default=50)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-precision", type=float, default=0.64)
    parser.add_argument("--min-worst-year-precision", type=float, default=0.58)
    parser.add_argument("--min-sharpe", type=float, default=1.10)
    parser.add_argument("--min-annualized", type=float, default=0.08)
    parser.add_argument("--max-drawdown", type=float, default=0.12)
    parser.add_argument("--min-profit-factor", type=float, default=1.6)
    parser.add_argument("--min-vote-weight", type=float, default=0.12)
    parser.add_argument("--min-vote-count", type=int, default=2)
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--dynamic-size", action="store_true")
    parser.add_argument("--target-vol", type=float, default=0.30)
    parser.add_argument("--min-vol-scalar", type=float, default=0.40)
    parser.add_argument("--max-vol-scalar", type=float, default=1.25)
    parser.add_argument("--min-position", type=float, default=0.10)
    parser.add_argument("--calibration-bins", type=int, default=6)
    parser.add_argument("--block-realized-vol-hot", action="store_true")
    parser.add_argument("--block-vix-high", action="store_true")
    parser.add_argument("--block-holiday-week", action="store_true")
    parser.add_argument("--block-semiconductor-weak", action="store_true")
    parser.add_argument("--include-live-news", action="store_true")
    parser.add_argument("--news-days", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    run_ensemble(parse_args())


if __name__ == "__main__":
    main()
