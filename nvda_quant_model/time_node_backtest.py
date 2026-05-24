from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _directional_node_metrics(signals: pd.DataFrame, frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    aligned = frame[["target_return", "target_direction"]].join(signals[["position", "prob_up"]], how="inner")
    mask = mask.reindex(aligned.index).fillna(False).astype(bool)
    scoped = aligned.loc[mask]
    active = scoped[scoped["position"] > 0].dropna(subset=["target_return", "target_direction"])
    if active.empty:
        return {
            "node_days": int(mask.sum()),
            "active_days": 0,
            "direction_precision": np.nan,
            "avg_next_return": 0.0,
            "median_next_return": 0.0,
        }
    return {
        "node_days": int(mask.sum()),
        "active_days": int(len(active)),
        "direction_precision": float((active["target_return"] > 0).mean()),
        "avg_next_return": float(active["target_return"].mean()),
        "median_next_return": float(active["target_return"].median()),
    }


def _backtest_window(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: StrategyConfig,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any] | None:
    seg_prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if len(seg_prices) < 5:
        return None
    seg_signals = signals.reindex(seg_prices.index).dropna(subset=["prob_up", "expected_return"])
    if seg_signals.empty:
        return None
    result = BacktestEngine(cfg).backtest(seg_signals, seg_prices)
    mask = pd.Series(False, index=frame.index)
    mask.loc[(mask.index >= start) & (mask.index <= end)] = True
    dmetrics = _directional_node_metrics(signals, frame, mask)
    return {
        "label": label,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        **result.metrics,
        **dmetrics,
    }


def run_time_node_backtest(signals_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (signals_dir / "best_precision.json").open() as f:
        payload = json.load(f)
    best = payload["best"]
    start, end, _ = resolve_data_window("auto", "latest", ticker="NVDA", lookback_months=24)
    cfg = StrategyConfig(
        start_date=start,
        end_date=end,
        stop_loss_pct=float(best["stop_loss_pct"]),
        take_profit_pct=float(best["take_profit_pct"]),
        max_exposure=float(best["max_exposure"]),
        include_peer_events=True,
    )

    prices, external = load_market_data("NVDA", start, end, cache_dir=PROJECT_ROOT / "cache")
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(start), end, cache_dir=PROJECT_ROOT / "cache")
    frame, _ = build_model_frame(prices, external, start, end, "NVDA", include_fundamentals=False, peer_ohlcv=peer_ohlcv)
    report_prices = prices.loc[start:end]
    signals = pd.read_csv(signals_dir / "best_signals.csv", parse_dates=["Date"], index_col="Date")
    signals = signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])

    rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for year in range(start_ts.year, end_ts.year + 1):
        ys = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        ye = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        row = _backtest_window(f"year_{year}", ys, ye, cfg, signals, report_prices, frame)
        if row:
            rows.append(row)

    quarter_starts = pd.date_range(start_ts, end_ts, freq="QS")
    if len(quarter_starts) == 0 or quarter_starts[0] > start_ts:
        quarter_starts = quarter_starts.insert(0, start_ts)
    for qs in quarter_starts:
        qe = min(qs + pd.DateOffset(months=3) - pd.Timedelta(days=1), end_ts)
        row = _backtest_window(f"quarter_{qs.year}_Q{((qs.month - 1) // 3) + 1}", qs, qe, cfg, signals, report_prices, frame)
        if row:
            rows.append(row)

    idx = report_prices.index
    for i in range(0, max(len(idx) - 63, 1), 21):
        window_idx = idx[i : min(i + 63, len(idx))]
        if len(window_idx) < 21:
            continue
        row = _backtest_window(
            f"rolling_63d_{window_idx[0].strftime('%Y%m%d')}",
            window_idx[0],
            window_idx[-1],
            cfg,
            signals,
            report_prices,
            frame,
        )
        if row:
            rows.append(row)

    node_masks: dict[str, pd.Series] = {}
    scoped_frame = frame.loc[start:end]
    if "peer_event_shock_count" in scoped_frame:
        node_masks["peer_event_days"] = scoped_frame["peer_event_shock_count"] > 0
    if "peer_positive_event_count" in scoped_frame:
        node_masks["peer_positive_event_days"] = scoped_frame["peer_positive_event_count"] > 0
    if "peer_negative_event_count" in scoped_frame:
        node_masks["peer_negative_event_days"] = scoped_frame["peer_negative_event_count"] > 0
    if "peer_business_stress" in scoped_frame:
        node_masks["peer_business_stress_top_quartile"] = scoped_frame["peer_business_stress"] >= scoped_frame["peer_business_stress"].quantile(0.75)
    if "peer_positive_breadth_5d" in scoped_frame:
        node_masks["peer_breadth_strong"] = scoped_frame["peer_positive_breadth_5d"] >= 0.75
        node_masks["peer_breadth_weak"] = scoped_frame["peer_positive_breadth_5d"] <= 0.35
    if "holiday_week" in scoped_frame:
        node_masks["holiday_week"] = scoped_frame["holiday_week"] == 1

    node_rows = []
    for label, mask in node_masks.items():
        node_rows.append({"label": label, **_directional_node_metrics(signals, frame, mask)})

    time_rows = pd.DataFrame(rows)
    event_rows = pd.DataFrame(node_rows)
    time_rows.to_csv(output_dir / "time_node_backtest.csv", index=False)
    event_rows.to_csv(output_dir / "event_node_backtest.csv", index=False)
    summary = {
        "source_signals": str(signals_dir),
        "best_label": best["label"],
        "time_rows": int(len(time_rows)),
        "event_rows": int(len(event_rows)),
        "best": best,
        "time_node_csv": str((output_dir / "time_node_backtest.csv").resolve()),
        "event_node_csv": str((output_dir / "event_node_backtest.csv").resolve()),
    }
    (output_dir / "time_node_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-node validation for selected NVDA precision signals")
    parser.add_argument("--signals-dir", default=str(PROJECT_ROOT / "outputs" / "precision_search_peer_8000"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "time_node_peer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_time_node_backtest(Path(args.signals_dir), Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
