from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from nvda_quant_model.config import PROJECT_ROOT


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval=1d"


def fetch_vix_snapshot(symbol: str = "^VIX", range_: str = "1mo", timeout: int = 20) -> dict[str, Any]:
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    url = YAHOO_CHART_URL.format(symbol=quote(symbol, safe=""), range=range_)
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        payload = response.json()
        result = payload["chart"]["result"][0]
        row = parse_chart_result(result, symbol=symbol, fetched_at_utc=fetched_at)
    except Exception as exc:  # pragma: no cover - network dependent
        return {
            "symbol": symbol,
            "fetched_at_utc": fetched_at,
            "provider": "yahoo_chart",
            "error": f"{type(exc).__name__}: {exc}",
            "risk_gate": "unavailable",
        }
    row["risk_gate"] = interpret_vix(row.get("level"), row.get("pct_from_prev_close"))
    row["model_guidance"] = vix_model_guidance(row["risk_gate"])
    return row


def parse_chart_result(result: dict[str, Any], *, symbol: str, fetched_at_utc: str) -> dict[str, Any]:
    timestamps = result.get("timestamp") or []
    quote_data = result["indicators"]["quote"][0]
    closes = quote_data.get("close") or []
    opens = quote_data.get("open") or []
    valid = [idx for idx, close in enumerate(closes) if close is not None]
    if not valid:
        raise ValueError("no valid VIX close rows")
    latest_idx = valid[-1]
    prev_idx = valid[-2] if len(valid) >= 2 else None
    five_idx = valid[-6] if len(valid) >= 6 else None
    level = float(closes[latest_idx])
    open_price = float(opens[latest_idx]) if latest_idx < len(opens) and opens[latest_idx] is not None else None
    previous_close = float(closes[prev_idx]) if prev_idx is not None else None
    five_day_reference = float(closes[five_idx]) if five_idx is not None else None
    market_date = datetime.fromtimestamp(timestamps[latest_idx], tz=timezone.utc).date().isoformat()
    return {
        "symbol": symbol,
        "date": market_date,
        "level": round(level, 4),
        "open": round_or_none(open_price),
        "previous_close": round_or_none(previous_close),
        "pct_from_prev_close": round_or_none(pct_change(level, previous_close)),
        "pct_from_open": round_or_none(pct_change(level, open_price)),
        "pct_5d": round_or_none(pct_change(level, five_day_reference)),
        "provider": "yahoo_chart_1mo_1d",
        "fetched_at_utc": fetched_at_utc,
        "error": "",
    }


def interpret_vix(level: Any, pct_from_prev_close: Any = None) -> str:
    if level in {None, ""}:
        return "unavailable"
    level = float(level)
    prev_change = None if pct_from_prev_close in {None, ""} else float(pct_from_prev_close)
    rising_fast = prev_change is not None and prev_change >= 8.0
    falling_fast = prev_change is not None and prev_change <= -8.0
    if level >= 30.0 or (level >= 25.0 and rising_fast):
        return "risk_off"
    if level >= 22.0 or rising_fast:
        return "risk_watch"
    if level <= 15.0 and not rising_fast:
        return "risk_on"
    if level <= 18.0 or falling_fast:
        return "risk_neutral_to_on"
    return "neutral"


def vix_model_guidance(state: str) -> str:
    guidance = {
        "risk_off": "block fresh SOXL leverage adds; require stronger NVDA confirmation.",
        "risk_watch": "cap SOXL size; reduce confidence on bullish semi headlines until volatility cools.",
        "risk_on": "macro volatility supports risk-taking if sector breadth and earnings revisions agree.",
        "risk_neutral_to_on": "VIX is not a blocker; confirm with SMH/SOXX breadth before adding leverage.",
        "neutral": "no standalone VIX adjustment; keep news and technical gates in control.",
        "unavailable": "do not adjust sizing from VIX; rely on cached model VIX features and other gates.",
    }
    return guidance.get(state, guidance["neutral"])


def pct_change(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in {None, 0}:
        return None
    return (value / reference - 1.0) * 100.0


def round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current VIX index snapshot for the NVDA automation.")
    parser.add_argument("--symbol", default="^VIX")
    parser.add_argument("--range", default="1mo")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "news_live" / "vix_snapshot.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = fetch_vix_snapshot(symbol=args.symbol, range_=args.range)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
