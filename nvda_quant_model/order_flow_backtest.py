from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def load_yfinance_minute_bars(symbol: str, period: str = "8d", interval: str = "1m") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, prepost=False, threads=False)
    if raw.empty:
        raise RuntimeError(f"No intraday bars returned for {symbol} period={period} interval={interval}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    columns = {col: col.lower().replace(" ", "_") for col in raw.columns}
    bars = raw.rename(columns=columns)
    keep = [col for col in ["open", "high", "low", "close", "volume", "vwap"] if col in bars.columns]
    bars = bars[keep].dropna(subset=["open", "close"])
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    return bars.sort_index()


def build_intraday_features(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    close = frame["close"]
    frame["return_5m"] = close.pct_change(5)
    frame["return_15m"] = close.pct_change(15)
    frame["return_30m"] = close.pct_change(30)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    session = frame.index.tz_convert("America/New_York").date
    grouped = frame.assign(_session=session).groupby("_session", group_keys=False)
    cumulative_volume = grouped["volume"].cumsum().replace(0, np.nan)
    cumulative_pv = (typical * frame["volume"]).groupby(session).cumsum()
    frame["session_vwap"] = cumulative_pv / cumulative_volume
    frame["vwap_gap"] = close / frame["session_vwap"] - 1.0
    session_high = grouped["high"].cummax()
    session_low = grouped["low"].cummin()
    frame["day_range_position"] = (close - session_low) / (session_high - session_low).replace(0, np.nan)
    volume = frame["volume"].replace(0, np.nan)
    recent_volume = volume.rolling(5, min_periods=5).mean()
    baseline_volume = volume.shift(5).rolling(20, min_periods=10).mean()
    frame["volume_accel"] = recent_volume / baseline_volume
    frame["session"] = pd.Series(session, index=frame.index).astype(str)
    return frame


def score_intraday_proxy(features: pd.DataFrame, threshold: int = 2, mode: str = "trend") -> pd.DataFrame:
    out = features.copy()
    components = pd.DataFrame(index=out.index)
    components["return_5m"] = np.select([out["return_5m"] > 0.001, out["return_5m"] < -0.001], [1, -1], default=0)
    components["return_15m"] = np.select([out["return_15m"] > 0.002, out["return_15m"] < -0.002], [1, -1], default=0)
    components["vwap_gap"] = np.select([out["vwap_gap"] > 0.0005, out["vwap_gap"] < -0.0005], [1, -1], default=0)
    components["day_range"] = np.select([out["day_range_position"] > 0.68, out["day_range_position"] < 0.32], [1, -1], default=0)
    volume_direction = np.sign(out["return_5m"].fillna(0.0))
    components["volume_accel"] = np.where(out["volume_accel"] > 1.35, volume_direction, 0)
    out["micro_score"] = components.sum(axis=1).astype(float)
    signal = np.select([out["micro_score"] >= threshold, out["micro_score"] <= -threshold], [1, -1], default=0)
    if mode == "inverse":
        signal = -signal
    elif mode != "trend":
        raise ValueError("mode must be trend or inverse")
    out["signal"] = signal
    return out


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def backtest_intraday_signals(
    scored: pd.DataFrame,
    horizon_minutes: int = 15,
    cost_per_trade: float | None = None,
    allow_short: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = StrategyConfig()
    cost_per_trade = config.round_trip_cost if cost_per_trade is None else cost_per_trade
    rows: list[dict[str, Any]] = []
    index = scored.index
    i = 0
    while i < len(scored) - horizon_minutes - 1:
        signal = int(scored["signal"].iloc[i])
        if signal == 0 or (signal < 0 and not allow_short):
            i += 1
            continue
        entry_idx = i + 1
        exit_idx = i + horizon_minutes
        entry_time = index[entry_idx]
        exit_time = index[exit_idx]
        if scored["session"].iloc[entry_idx] != scored["session"].iloc[exit_idx]:
            i += 1
            continue
        entry = float(scored["open"].iloc[entry_idx])
        exit_price = float(scored["close"].iloc[exit_idx])
        gross = signal * (exit_price / entry - 1.0)
        net = gross - cost_per_trade
        rows.append(
            {
                "signal_time": index[i],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": "long" if signal > 0 else "short",
                "score": float(scored["micro_score"].iloc[i]),
                "entry_price": entry,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": net,
                "win": net > 0,
            }
        )
        i = exit_idx + 1
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "cost_per_trade": float(cost_per_trade),
        "horizon_minutes": int(horizon_minutes),
        }
    equity = (1.0 + trades["net_return"]).cumprod()
    wins = trades.loc[trades["net_return"] > 0, "net_return"]
    losses = trades.loc[trades["net_return"] < 0, "net_return"]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty else np.inf
    metrics = {
        "num_trades": int(len(trades)),
        "win_rate": float(trades["win"].mean()),
        "avg_net_return": float(trades["net_return"].mean()),
        "median_net_return": float(trades["net_return"].median()),
        "profit_factor": profit_factor,
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": _max_drawdown(equity),
        "long_trades": int((trades["side"] == "long").sum()),
        "short_trades": int((trades["side"] == "short").sum()),
        "gross_avg_return": float(trades["gross_return"].mean()),
        "cost_per_trade": float(cost_per_trade),
        "horizon_minutes": int(horizon_minutes),
    }
    return trades, metrics


def _score_policy(metrics: dict[str, Any], min_train_trades: int) -> float:
    if metrics["num_trades"] < min_train_trades:
        return -np.inf
    return float(
        metrics["avg_net_return"] * 1000.0
        + min(metrics["profit_factor"], 3.0)
        + metrics["win_rate"]
        + metrics["total_return"]
        - abs(metrics["max_drawdown"]) * 2.0
    )


def walk_forward_calibrate_proxy(
    features: pd.DataFrame,
    train_sessions: int = 4,
    thresholds: list[int] | None = None,
    horizons: list[int] | None = None,
    modes: list[str] | None = None,
    min_train_trades: int = 12,
    min_train_avg_net_return: float = 0.0005,
    min_train_profit_factor: float = 1.25,
    min_train_win_rate: float = 0.55,
    cost_per_trade: float | None = None,
    allow_short: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose proxy settings on past sessions, then test the next session.

    If no setting has a positive net edge in the rolling training window, the
    next session is explicitly disabled. This keeps the proxy from manufacturing
    trades when the recent intraday regime is hostile.
    """
    thresholds = thresholds or [2, 3, 4, 5]
    horizons = horizons or [5, 15, 30]
    modes = modes or ["trend", "inverse"]
    sessions = list(pd.Series(features["session"].dropna().unique()).sort_values())
    all_trades: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []

    for idx in range(train_sessions, len(sessions)):
        train_session_set = set(sessions[idx - train_sessions : idx])
        test_session = sessions[idx]
        train = features[features["session"].isin(train_session_set)]
        test = features[features["session"] == test_session]
        best: dict[str, Any] | None = None
        best_score = -np.inf
        for mode in modes:
            for threshold in thresholds:
                for horizon in horizons:
                    candidate = score_intraday_proxy(train, threshold=threshold, mode=mode)
                    _, train_metrics = backtest_intraday_signals(
                        candidate,
                        horizon_minutes=horizon,
                        cost_per_trade=cost_per_trade,
                        allow_short=allow_short,
                    )
                    score = _score_policy(train_metrics, min_train_trades)
                    if score > best_score:
                        best_score = score
                        best = {
                            "mode": mode,
                            "threshold": threshold,
                            "horizon_minutes": horizon,
                            "train_score": score,
                            "train_metrics": train_metrics,
                        }

        enabled = bool(
            best
            and best["train_metrics"]["num_trades"] >= min_train_trades
            and best["train_metrics"]["avg_net_return"] >= min_train_avg_net_return
            and best["train_metrics"]["profit_factor"] >= min_train_profit_factor
            and best["train_metrics"]["win_rate"] >= min_train_win_rate
        )
        if enabled:
            scored_test = score_intraday_proxy(test, threshold=best["threshold"], mode=best["mode"])
            test_trades, test_metrics = backtest_intraday_signals(
                scored_test,
                horizon_minutes=best["horizon_minutes"],
                cost_per_trade=cost_per_trade,
                allow_short=allow_short,
            )
            if not test_trades.empty:
                test_trades["selected_mode"] = best["mode"]
                test_trades["selected_threshold"] = best["threshold"]
                test_trades["selected_horizon"] = best["horizon_minutes"]
                all_trades.append(test_trades)
        else:
            test_metrics = {
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_net_return": 0.0,
                "median_net_return": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "cost_per_trade": float(StrategyConfig().round_trip_cost if cost_per_trade is None else cost_per_trade),
                "horizon_minutes": int(best["horizon_minutes"]) if best else 0,
            }
        period_rows.append(
            {
                "test_session": test_session,
                "enabled": enabled,
                "mode": best["mode"] if best else "none",
                "threshold": best["threshold"] if best else 0,
                "horizon_minutes": best["horizon_minutes"] if best else 0,
                "train_score": best_score if np.isfinite(best_score) else 0.0,
                "train_num_trades": best["train_metrics"]["num_trades"] if best else 0,
                "train_win_rate": best["train_metrics"]["win_rate"] if best else 0.0,
                "train_avg_net_return": best["train_metrics"]["avg_net_return"] if best else 0.0,
                "train_profit_factor": best["train_metrics"]["profit_factor"] if best else 0.0,
                "test_num_trades": test_metrics["num_trades"],
                "test_win_rate": test_metrics["win_rate"],
                "test_avg_net_return": test_metrics["avg_net_return"],
                "test_profit_factor": test_metrics["profit_factor"],
                "test_total_return": test_metrics["total_return"],
                "test_max_drawdown": test_metrics["max_drawdown"],
            }
        )

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    periods = pd.DataFrame(period_rows)
    return trades, periods


def run_backtest(
    symbol: str,
    period: str,
    horizon_minutes: int,
    output_dir: Path,
    allow_short: bool = True,
    cost_per_trade: float | None = None,
    calibrate: bool = False,
    train_sessions: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bars = load_yfinance_minute_bars(symbol, period=period)
    features = build_intraday_features(bars)
    scored = score_intraday_proxy(features)
    if calibrate:
        trades, periods = walk_forward_calibrate_proxy(
            features,
            train_sessions=train_sessions,
            cost_per_trade=cost_per_trade,
            allow_short=allow_short,
        )
        enabled_sessions = int(periods["enabled"].sum()) if not periods.empty else 0
        if trades.empty:
            metrics = {
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_net_return": 0.0,
                "median_net_return": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "cost_per_trade": float(StrategyConfig().round_trip_cost if cost_per_trade is None else cost_per_trade),
                "horizon_minutes": 0,
            }
        else:
            equity = (1.0 + trades["net_return"]).cumprod()
            wins = trades.loc[trades["net_return"] > 0, "net_return"]
            losses = trades.loc[trades["net_return"] < 0, "net_return"]
            metrics = {
                "num_trades": int(len(trades)),
                "win_rate": float((trades["net_return"] > 0).mean()),
                "avg_net_return": float(trades["net_return"].mean()),
                "median_net_return": float(trades["net_return"].median()),
                "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else np.inf,
                "total_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": _max_drawdown(equity),
                "long_trades": int((trades["side"] == "long").sum()),
                "short_trades": int((trades["side"] == "short").sum()),
                "cost_per_trade": float(StrategyConfig().round_trip_cost if cost_per_trade is None else cost_per_trade),
                "horizon_minutes": "walk_forward",
            }
        periods.to_csv(output_dir / "order_flow_proxy_walk_forward.csv", index=False)
    else:
        periods = pd.DataFrame()
        enabled_sessions = 0
        trades, metrics = backtest_intraday_signals(scored, horizon_minutes, cost_per_trade, allow_short)
    bars.to_csv(output_dir / f"{symbol}_1m_bars.csv")
    scored.to_csv(output_dir / "order_flow_proxy_features.csv")
    trades.to_csv(output_dir / "order_flow_proxy_trades.csv", index=False)
    payload = {
        "symbol": symbol,
        "period": period,
        "start": bars.index.min(),
        "end": bars.index.max(),
        "bars": int(len(bars)),
        "sessions": int(scored["session"].nunique()),
        "data_source": "yfinance_1m_bars",
        "important_limit": "This is a proxy backtest for minute trend/VWAP/volume. It does not include historical bid/ask size or Level-2 depth.",
        "mode": "walk_forward_calibrated" if calibrate else "fixed_proxy",
        "decision": "disabled_no_valid_edge" if calibrate and metrics["num_trades"] == 0 else "research_only_not_standalone",
        "enabled_sessions": enabled_sessions,
        "metrics": metrics,
    }
    (output_dir / "order_flow_proxy_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    lines = [
        "# NVDA Order-Flow Proxy Backtest",
        "",
        f"- Symbol: {symbol}",
        f"- Period: {period}",
        f"- Data: {payload['start']} ~ {payload['end']}",
        f"- Bars: {payload['bars']} / Sessions: {payload['sessions']}",
        f"- Mode: {'walk-forward calibrated' if calibrate else 'fixed proxy'}",
        f"- Horizon: {horizon_minutes} minutes" if not calibrate else "- Horizon: selected by rolling training windows",
        f"- Cost per trade: {metrics['cost_per_trade']:.4%}",
        f"- Decision: {payload['decision']}",
    ]
    if calibrate:
        lines.append(f"- Enabled test sessions: {enabled_sessions} / {max(payload['sessions'] - train_sessions, 0)}")
        lines.append("- Calibration gate: train avg net return >= 0.05%, PF >= 1.25, win rate >= 55%, enough trades")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Trades: {metrics['num_trades']}",
            f"- Win rate: {metrics['win_rate']:.2%}",
            f"- Avg net return/trade: {metrics['avg_net_return']:.4%}",
            f"- Median net return/trade: {metrics.get('median_net_return', 0.0):.4%}",
            f"- Profit factor: {metrics['profit_factor']:.2f}",
            f"- Total compounded return: {metrics['total_return']:.2%}",
            f"- Max drawdown: {metrics['max_drawdown']:.2%}",
            f"- Long / Short trades: {metrics['long_trades']} / {metrics['short_trades']}",
            "",
            "## Limitation",
            "",
            "This backtest uses historical 1-minute OHLCV bars as a proxy for the live order-flow layer. "
            "It does not include historical bid/ask size, quote imbalance, or Level-2 depth. "
            "A true buy/sell book backtest needs point-in-time historical quotes or depth data.",
        ]
    )
    report = output_dir / "order_flow_proxy_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    payload["report_path"] = str(report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest live order-flow proxy signals on recent 1-minute bars")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--period", default="8d")
    parser.add_argument("--horizon-minutes", type=int, default=15)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--cost-per-trade", type=float, default=None)
    parser.add_argument("--cost-per-side", type=float, default=None, help="Deprecated alias; converted to round-trip cost by doubling")
    parser.add_argument("--calibrate", action="store_true", help="Walk-forward select mode/threshold/horizon and disable hostile regimes")
    parser.add_argument("--train-sessions", type=int, default=4)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "order_flow_backtest"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_backtest(
        symbol=args.symbol,
        period=args.period,
        horizon_minutes=args.horizon_minutes,
        output_dir=Path(args.output_dir),
        allow_short=not args.long_only,
        cost_per_trade=args.cost_per_trade if args.cost_per_trade is not None else (args.cost_per_side * 2.0 if args.cost_per_side is not None else None),
        calibrate=args.calibrate,
        train_sessions=args.train_sessions,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
