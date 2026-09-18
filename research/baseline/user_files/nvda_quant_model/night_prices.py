from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nvda_quant_model.config import PROJECT_ROOT


DEFAULT_SYMBOLS = [
    "NVDA",
    "SOXL",
    "SMH",
    "SOXX",
    "AMD",
    "AVGO",
    "TSM",
    "ASML",
    "MU",
    "QCOM",
    "ARM",
    "INTC",
]

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ALPACA_DATA_URL = "https://data.alpaca.markets"


@dataclass(frozen=True)
class NightPrice:
    symbol: str
    fetched_at_utc: str
    price: float | None
    price_time_utc: str | None
    provider: str
    session: str
    regular_price: float | None = None
    pre_market_price: float | None = None
    post_market_price: float | None = None
    previous_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    error: str = ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_from_epoch(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return None


def _request_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_alpaca(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    _load_env_file(PROJECT_ROOT.parent / ".env")
    _load_env_file(PROJECT_ROOT / ".env")
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("missing Alpaca credentials")
    query = urllib.parse.urlencode(params)
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    return _request_json(f"{ALPACA_DATA_URL}{endpoint}?{query}", headers=headers)


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


def _latest_close_from_quote(result: dict[str, Any]) -> tuple[float | None, str | None, float | None]:
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    for idx in range(len(closes) - 1, -1, -1):
        close = _number(closes[idx])
        if close is None:
            continue
        ts = _iso_from_epoch(timestamps[idx]) if idx < len(timestamps) else None
        volume = _number(volumes[idx]) if idx < len(volumes) else None
        return close, ts, volume
    return None, None, None


def fetch_yahoo_price(symbol: str) -> NightPrice:
    fetched_at = _now_utc()
    url = YAHOO_CHART_URL.format(symbol=urllib.parse.quote(symbol.upper()))
    params = urllib.parse.urlencode(
        {
            "range": "1d",
            "interval": "1m",
            "includePrePost": "true",
            "events": "div,splits",
        }
    )
    try:
        payload = _request_json(f"{url}?{params}")
        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            raise RuntimeError(json.dumps(error, ensure_ascii=False))
        result = (chart.get("result") or [{}])[0]
        meta = result.get("meta") or {}
        bar_price, bar_time, volume = _latest_close_from_quote(result)
        regular = _number(meta.get("regularMarketPrice"))
        pre = _number(meta.get("preMarketPrice"))
        post = _number(meta.get("postMarketPrice"))
        previous_close = _number(meta.get("chartPreviousClose") or meta.get("previousClose"))

        session = "regular_or_last"
        price = regular or bar_price
        price_time = _iso_from_epoch(meta.get("regularMarketTime")) or bar_time
        if post is not None:
            session = "post_market"
            price = post
            price_time = _iso_from_epoch(meta.get("postMarketTime")) or bar_time
        elif pre is not None:
            session = "pre_market"
            price = pre
            price_time = _iso_from_epoch(meta.get("preMarketTime")) or bar_time
        elif bar_price is not None:
            price = bar_price
            price_time = bar_time
            regular_time = _iso_from_epoch(meta.get("regularMarketTime"))
            if regular_time and bar_time and bar_time > regular_time:
                session = "extended_hours_last"

        change = price - previous_close if price is not None and previous_close else None
        change_pct = change / previous_close if change is not None and previous_close else None
        return NightPrice(
            symbol=symbol.upper(),
            fetched_at_utc=fetched_at,
            price=price,
            price_time_utc=price_time,
            provider="yahoo_chart_include_prepost",
            session=session,
            regular_price=regular,
            pre_market_price=pre,
            post_market_price=post,
            previous_close=previous_close,
            change=change,
            change_pct=change_pct,
            volume=volume,
        )
    except Exception as exc:
        return NightPrice(
            symbol=symbol.upper(),
            fetched_at_utc=fetched_at,
            price=None,
            price_time_utc=None,
            provider="yahoo_chart_include_prepost",
            session="unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )


def fetch_alpaca_overnight_price(symbol: str) -> NightPrice:
    fetched_at = _now_utc()
    try:
        payload = _request_alpaca(
            "/v2/stocks/snapshots",
            {"symbols": symbol.upper(), "feed": "overnight", "currency": "USD"},
        )
        item = (payload.get("snapshots") or {}).get(symbol.upper()) or {}
        trade = item.get("latest_trade") or item.get("latestTrade") or {}
        quote = item.get("latest_quote") or item.get("latestQuote") or {}
        daily = item.get("daily_bar") or item.get("dailyBar") or {}
        previous = item.get("previous_daily_bar") or item.get("prevDailyBar") or {}
        price = _number(trade.get("price") or trade.get("p"))
        bid = _number(quote.get("bid_price") or quote.get("bp"))
        ask = _number(quote.get("ask_price") or quote.get("ap"))
        if price is None and bid is not None and ask is not None:
            price = (bid + ask) / 2.0
        previous_close = _number(previous.get("close") or previous.get("c"))
        change = price - previous_close if price is not None and previous_close else None
        change_pct = change / previous_close if change is not None and previous_close else None
        return NightPrice(
            symbol=symbol.upper(),
            fetched_at_utc=fetched_at,
            price=price,
            price_time_utc=trade.get("timestamp") or trade.get("t") or quote.get("timestamp") or quote.get("t"),
            provider="alpaca_overnight_snapshot",
            session="overnight",
            previous_close=previous_close,
            change=change,
            change_pct=change_pct,
            bid=bid,
            ask=ask,
            volume=_number(daily.get("volume") or daily.get("v")),
        )
    except Exception as exc:
        return NightPrice(
            symbol=symbol.upper(),
            fetched_at_utc=fetched_at,
            price=None,
            price_time_utc=None,
            provider="alpaca_overnight_snapshot",
            session="unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )


def best_price(symbol: str, prefer_alpaca: bool = True) -> NightPrice:
    if prefer_alpaca:
        alpaca = fetch_alpaca_overnight_price(symbol)
        if alpaca.price is not None:
            return alpaca
    return fetch_yahoo_price(symbol)


def write_outputs(rows: list[NightPrice], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": _now_utc(),
        "symbols": [row.symbol for row in rows],
        "prices": [asdict(row) for row in rows],
    }
    (output_dir / "latest_prices.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = list(asdict(rows[0]).keys()) if rows else list(NightPrice("", "", None, None, "", "").__dict__.keys())
    with (output_dir / "latest_prices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    history_path = output_dir / "price_history.csv"
    write_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch extended-hours / overnight prices for NVDA semiconductor watchlist.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated ticker list.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "night_prices"))
    parser.add_argument("--no-alpaca", action="store_true", help="Skip Alpaca overnight feed and use Yahoo pre/post data only.")
    parser.add_argument("--loop", action="store_true", help="Run continuously.")
    parser.add_argument("--interval-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    output_dir = Path(args.output_dir)
    while True:
        rows = [best_price(symbol, prefer_alpaca=not args.no_alpaca) for symbol in symbols]
        write_outputs(rows, output_dir)
        print(
            json.dumps(
                {
                    "updated_at_utc": _now_utc(),
                    "output_dir": str(output_dir),
                    "prices": [asdict(row) for row in rows],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not args.loop:
            break
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    main()
