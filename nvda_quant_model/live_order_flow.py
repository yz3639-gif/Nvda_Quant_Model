from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from nvda_quant_model.config import PROJECT_ROOT


DATA_BASE_URL = "https://data.alpaca.markets"


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


def _credentials() -> tuple[str, str]:
    _load_env_file(PROJECT_ROOT.parent / ".env")
    _load_env_file(PROJECT_ROOT / ".env")
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    return key, secret


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _request_data(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    key, secret = _credentials()
    base_url = os.getenv("APCA_DATA_BASE_URL", DATA_BASE_URL).rstrip("/")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"}
    response = requests.get(f"{base_url}{endpoint}", headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _first_present(mapping: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def fetch_stock_snapshot(symbol: str, feed: str = "iex", currency: str = "USD") -> dict[str, Any]:
    """Fetch latest stock snapshot from Alpaca market-data API.

    Alpaca's stock snapshot endpoint returns top-of-book bid/ask, latest trade,
    latest minute bar, daily bar, and previous daily bar. It is not a full
    Level-2 depth book for equities.
    """
    symbol = symbol.upper()
    payload = _request_data(
        "/v2/stocks/snapshots",
        {"symbols": symbol, "feed": feed, "currency": currency},
    )
    snapshots = payload.get("snapshots", payload)
    item = snapshots.get(symbol)
    if not item:
        raise RuntimeError(f"No Alpaca snapshot returned for {symbol}")
    return {"source": "alpaca_stock_snapshot", "feed": feed, "currency": currency, "snapshots": {symbol: item}}


def fetch_recent_bars(symbol: str, minutes: int = 60, feed: str = "iex", currency: str = "USD") -> pd.DataFrame:
    symbol = symbol.upper()
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=max(minutes + 20, 40))
    payload = _request_data(
        "/v2/stocks/bars",
        {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start.isoformat(timespec="seconds"),
            "end": end.isoformat(timespec="seconds"),
            "limit": min(max(minutes + 30, 50), 1000),
            "feed": feed,
            "currency": currency,
        },
    )
    bars = payload.get("bars", {}).get(symbol, [])
    return bars_frame(bars)


def _bar_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": pd.Timestamp(_first_present(item, ["timestamp", "t"])),
        "open": _number(_first_present(item, ["open", "o"])),
        "high": _number(_first_present(item, ["high", "h"])),
        "low": _number(_first_present(item, ["low", "l"])),
        "close": _number(_first_present(item, ["close", "c"])),
        "volume": _number(_first_present(item, ["volume", "v"])),
        "trade_count": _number(_first_present(item, ["trade_count", "n"])),
        "vwap": _number(_first_present(item, ["vwap", "vw"])),
    }


def bars_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
    frame = pd.DataFrame(_bar_row(item) for item in items)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.set_index("timestamp")
    return frame[~frame.index.duplicated(keep="last")]


def _snapshot_item(snapshot_payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    snapshots = snapshot_payload.get("snapshots", snapshot_payload)
    if symbol in snapshots:
        return snapshots[symbol]
    if "symbol" in snapshot_payload and snapshot_payload.get("symbol") == symbol:
        return snapshot_payload
    raise ValueError(f"Snapshot payload does not contain {symbol}")


def _quote_fields(item: dict[str, Any]) -> dict[str, Any]:
    quote = item.get("latest_quote") or item.get("latestQuote") or {}
    return {
        "timestamp": _timestamp(_first_present(quote, ["timestamp", "t"])),
        "bid_price": _number(_first_present(quote, ["bid_price", "bp"])),
        "bid_size": _number(_first_present(quote, ["bid_size", "bs"])),
        "bid_exchange": _first_present(quote, ["bid_exchange", "bx"]),
        "ask_price": _number(_first_present(quote, ["ask_price", "ap"])),
        "ask_size": _number(_first_present(quote, ["ask_size", "as"])),
        "ask_exchange": _first_present(quote, ["ask_exchange", "ax"]),
    }


def _trade_fields(item: dict[str, Any]) -> dict[str, Any]:
    trade = item.get("latest_trade") or item.get("latestTrade") or {}
    return {
        "timestamp": _timestamp(_first_present(trade, ["timestamp", "t"])),
        "price": _number(_first_present(trade, ["price", "p"])),
        "size": _number(_first_present(trade, ["size", "s"])),
        "exchange": _first_present(trade, ["exchange", "x"]),
    }


def _minute_bar_frame(item: dict[str, Any]) -> pd.DataFrame:
    bar = item.get("minute_bar") or item.get("minuteBar") or {}
    return bars_frame([bar]) if bar else bars_frame([])


def _daily_bar(item: dict[str, Any], name: str) -> dict[str, Any]:
    bar = item.get(name) or item.get({"daily_bar": "dailyBar", "previous_daily_bar": "prevDailyBar"}[name]) or {}
    return _bar_row(bar) if bar else {}


def _return_over(frame: pd.DataFrame, minutes: int) -> float | None:
    if len(frame) < 2 or "close" not in frame:
        return None
    scoped = frame.dropna(subset=["close"]).tail(minutes + 1)
    if len(scoped) < 2:
        return None
    start = float(scoped["close"].iloc[0])
    end = float(scoped["close"].iloc[-1])
    return end / start - 1.0 if start else None


def analyze_order_flow(snapshot_payload: dict[str, Any], symbol: str = "NVDA", bars: pd.DataFrame | None = None) -> dict[str, Any]:
    item = _snapshot_item(snapshot_payload, symbol)
    quote = _quote_fields(item)
    trade = _trade_fields(item)
    bars = bars if bars is not None and not bars.empty else _minute_bar_frame(item)
    latest_bar = bars.tail(1).iloc[0].to_dict() if bars is not None and not bars.empty else {}
    daily = _daily_bar(item, "daily_bar")
    previous = _daily_bar(item, "previous_daily_bar")

    bid = quote["bid_price"]
    ask = quote["ask_price"]
    bid_size = quote["bid_size"] or 0.0
    ask_size = quote["ask_size"] or 0.0
    spread = ask - bid if bid is not None and ask is not None else None
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    spread_pct = spread / mid if spread is not None and mid else None
    size_total = bid_size + ask_size
    quote_imbalance = (bid_size - ask_size) / size_total if size_total else None

    trade_price = trade["price"]
    trade_vs_mid = trade_price / mid - 1.0 if trade_price is not None and mid else None
    latest_close = _number(latest_bar.get("close")) if latest_bar else trade_price
    vwap = _number(latest_bar.get("vwap")) if latest_bar else None
    vwap_gap = latest_close / vwap - 1.0 if latest_close is not None and vwap else None

    day_return = None
    if daily.get("close") and previous.get("close"):
        day_return = float(daily["close"]) / float(previous["close"]) - 1.0
    day_range_position = None
    if daily.get("high") and daily.get("low") and latest_close is not None and daily["high"] != daily["low"]:
        day_range_position = (latest_close - float(daily["low"])) / (float(daily["high"]) - float(daily["low"]))

    ret_5m = _return_over(bars, 5) if bars is not None else None
    ret_15m = _return_over(bars, 15) if bars is not None else None
    ret_30m = _return_over(bars, 30) if bars is not None else None
    volume_accel = None
    if bars is not None and len(bars.dropna(subset=["volume"])) >= 10:
        volume = bars["volume"].dropna()
        recent = volume.tail(5).mean()
        baseline = volume.iloc[:-5].tail(20).mean()
        volume_accel = recent / baseline if baseline else None

    score = 0.0
    components: dict[str, int] = {}
    components["quote_imbalance"] = 1 if quote_imbalance is not None and quote_imbalance > 0.25 else -1 if quote_imbalance is not None and quote_imbalance < -0.25 else 0
    components["trade_vs_mid"] = 1 if trade_vs_mid is not None and trade_vs_mid > 0.0002 else -1 if trade_vs_mid is not None and trade_vs_mid < -0.0002 else 0
    components["return_5m"] = 1 if ret_5m is not None and ret_5m > 0.001 else -1 if ret_5m is not None and ret_5m < -0.001 else 0
    components["return_15m"] = 1 if ret_15m is not None and ret_15m > 0.002 else -1 if ret_15m is not None and ret_15m < -0.002 else 0
    components["vwap_gap"] = 1 if vwap_gap is not None and vwap_gap > 0.0005 else -1 if vwap_gap is not None and vwap_gap < -0.0005 else 0
    components["day_range"] = 1 if day_range_position is not None and day_range_position > 0.68 else -1 if day_range_position is not None and day_range_position < 0.32 else 0
    components["volume_accel"] = 1 if volume_accel is not None and volume_accel > 1.35 and (ret_5m or 0) > 0 else -1 if volume_accel is not None and volume_accel > 1.35 and (ret_5m or 0) < 0 else 0
    components["wide_spread"] = -1 if spread_pct is not None and spread_pct > 0.004 else 0
    score = float(sum(components.values()))
    signal = 1 if score >= 2 else -1 if score <= -2 else 0
    label = "bullish" if signal > 0 else "bearish" if signal < 0 else "neutral"
    confidence = min(0.85, 0.50 + abs(score) / 16.0)
    wide_spread = bool(spread_pct is not None and spread_pct > 0.004)
    execution_filter = {
        "use_as_entry_signal": False,
        "can_open_new_position": not wide_spread,
        "recommendation": "block_execution_wide_spread" if wide_spread else "liquidity_ok_monitor_only",
        "reason": (
            "Top-of-book spread is too wide for execution."
            if wide_spread
            else "Recent proxy backtests did not validate this layer as a standalone entry signal."
        ),
    }

    return {
        "symbol": symbol.upper(),
        "source": snapshot_payload.get("source", "alpaca_stock_snapshot"),
        "feed": snapshot_payload.get("feed"),
        "as_of": trade["timestamp"] or quote["timestamp"],
        "latest_trade": trade,
        "top_of_book": {
            **quote,
            "mid_price": mid,
            "spread": spread,
            "spread_pct": spread_pct,
            "quote_imbalance": quote_imbalance,
        },
        "minute_trend": {
            "latest_close": latest_close,
            "latest_volume": _number(latest_bar.get("volume")) if latest_bar else None,
            "latest_vwap": vwap,
            "return_5m": ret_5m,
            "return_15m": ret_15m,
            "return_30m": ret_30m,
            "vwap_gap": vwap_gap,
            "volume_accel_5m_vs_20m": volume_accel,
        },
        "daily_context": {
            "day_return": day_return,
            "day_range_position": day_range_position,
            "day_volume": _number(daily.get("volume")),
            "previous_close": _number(previous.get("close")),
        },
        "micro_signal": {
            "signal": signal,
            "label": label,
            "score": score,
            "confidence": confidence,
            "components": components,
        },
        "execution_filter": execution_filter,
        "notes": [
            "US equities snapshot is top-of-book best bid/ask, not full Level-2 depth.",
            "Live order-flow overlay is current-market context; it is not a standalone entry engine unless a point-in-time quote/depth backtest validates it.",
        ],
    }


def fetch_live_order_flow(symbol: str = "NVDA", feed: str = "iex", minutes: int = 60, currency: str = "USD") -> tuple[dict[str, Any], pd.DataFrame]:
    snapshot = fetch_stock_snapshot(symbol, feed=feed, currency=currency)
    bars = fetch_recent_bars(symbol, minutes=minutes, feed=feed, currency=currency)
    return analyze_order_flow(snapshot, symbol=symbol, bars=bars), bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and analyze live NVDA top-of-book order-flow context")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--feed", default="iex", choices=["iex", "sip", "delayed_sip", "boats", "overnight", "otc"])
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--snapshot-json", default=None, help="Optional existing Alpaca snapshot JSON")
    parser.add_argument("--bars-csv", default=None, help="Optional existing recent 1-minute bars CSV")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "live_order_flow"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshot_json:
        snapshot = json.loads(Path(args.snapshot_json).read_text(encoding="utf-8"))
        bars = pd.read_csv(args.bars_csv, parse_dates=["timestamp"]).set_index("timestamp") if args.bars_csv else None
        overlay = snapshot if "micro_signal" in snapshot and "top_of_book" in snapshot else analyze_order_flow(snapshot, symbol=args.symbol, bars=bars)
    else:
        overlay, bars = fetch_live_order_flow(args.symbol, args.feed, args.minutes, args.currency)
        if bars is not None and not bars.empty:
            bars.to_csv(output_dir / f"{args.symbol.upper()}_recent_1min_bars.csv")
    out = output_dir / f"{args.symbol.upper()}_order_flow.json"
    out.write_text(json.dumps(overlay, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(overlay, indent=2, ensure_ascii=False, default=_json_default))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
