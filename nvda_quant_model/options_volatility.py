from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT
from nvda_quant_model.data.load_data import load_market_data, resolve_data_window


OCC_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<yymmdd>\d{6})(?P<type>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OptionQuote:
    symbol: str
    expiration: pd.Timestamp
    option_type: str
    strike: float
    bid: float | None
    ask: float | None
    trade: float | None
    implied_volatility: float | None
    delta: float | None
    theta: float | None
    vega: float | None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid >= 0 and self.ask >= 0:
            return (self.bid + self.ask) / 2.0
        return self.trade


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def parse_occ_symbol(symbol: str) -> tuple[pd.Timestamp, str, float]:
    match = OCC_RE.match(symbol)
    if not match:
        raise ValueError(f"Unsupported OCC option symbol: {symbol}")
    expiration = pd.Timestamp("20" + match.group("yymmdd"))
    option_type = "call" if match.group("type") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return expiration, option_type, strike


def parse_snapshot_payload(payload: dict[str, Any]) -> list[OptionQuote]:
    snapshots = payload.get("snapshots", payload)
    quotes: list[OptionQuote] = []
    for symbol, item in snapshots.items():
        try:
            expiration, option_type, strike = parse_occ_symbol(symbol)
        except ValueError:
            continue
        latest_quote = item.get("latest_quote") or {}
        latest_trade = item.get("latest_trade") or {}
        greeks = item.get("greeks") or {}
        quotes.append(
            OptionQuote(
                symbol=symbol,
                expiration=expiration,
                option_type=option_type,
                strike=strike,
                bid=_maybe_float(latest_quote.get("bid_price")),
                ask=_maybe_float(latest_quote.get("ask_price")),
                trade=_maybe_float(latest_trade.get("price")),
                implied_volatility=_maybe_float(item.get("implied_volatility")),
                delta=_maybe_float(greeks.get("delta")),
                theta=_maybe_float(greeks.get("theta")),
                vega=_maybe_float(greeks.get("vega")),
            )
        )
    return quotes


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def quotes_frame(quotes: list[OptionQuote]) -> pd.DataFrame:
    rows = []
    for quote in quotes:
        rows.append(
            {
                "symbol": quote.symbol,
                "expiration": quote.expiration,
                "type": quote.option_type,
                "strike": quote.strike,
                "bid": quote.bid,
                "ask": quote.ask,
                "mid": quote.mid,
                "trade": quote.trade,
                "iv": quote.implied_volatility,
                "delta": quote.delta,
                "theta": quote.theta,
                "vega": quote.vega,
            }
        )
    return pd.DataFrame(rows)


def realized_volatility(ticker: str = "NVDA") -> dict[str, float]:
    start, end, _ = resolve_data_window("auto", "latest", ticker=ticker, lookback_months=24)
    prices, _ = load_market_data(ticker, start, end, cache_dir=PROJECT_ROOT / "cache")
    returns = prices.loc[start:end, "Close"].pct_change().dropna()
    return {
        "hv_10d": float(returns.tail(10).std(ddof=0) * np.sqrt(252)),
        "hv_20d": float(returns.tail(20).std(ddof=0) * np.sqrt(252)),
        "hv_60d": float(returns.tail(60).std(ddof=0) * np.sqrt(252)),
    }


def _closest_row(frame: pd.DataFrame, mask: pd.Series, target_delta_abs: float | None = None, strike: float | None = None) -> pd.Series | None:
    scoped = frame.loc[mask].dropna(subset=["iv"])
    if scoped.empty:
        return None
    if target_delta_abs is not None and "delta" in scoped:
        scoped = scoped.dropna(subset=["delta"]).copy()
        if scoped.empty:
            return None
        return scoped.iloc[(scoped["delta"].abs() - target_delta_abs).abs().argsort().iloc[0]]
    if strike is not None:
        return scoped.iloc[(scoped["strike"] - strike).abs().argsort().iloc[0]]
    return scoped.iloc[0]


def _select_expiry(expiries: list[dict[str, Any]], target_days: int, prefer_covering_window: bool = False) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        days = int(row["days_to_expiry"])
        miss = abs(days - target_days)
        under_window_penalty = 1 if prefer_covering_window and days < target_days else 0
        return (miss, under_window_penalty, days)

    return min(expiries, key=score)


def analyze_options(
    snapshot_path: Path,
    underlying_price: float,
    range_projection_path: Path | None = None,
    ticker: str = "NVDA",
) -> dict[str, Any]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    frame = quotes_frame(parse_snapshot_payload(payload))
    if frame.empty:
        raise ValueError(f"No option snapshots parsed from {snapshot_path}")

    hv = realized_volatility(ticker)
    expiries: list[dict[str, Any]] = []
    for expiration, exp_frame in frame.groupby("expiration"):
        days = max((pd.Timestamp(expiration).normalize() - pd.Timestamp(payload.get("as_of_date", pd.Timestamp.today())).normalize()).days, 1)
        strikes = exp_frame["strike"].dropna().unique()
        atm_strike = float(strikes[np.argmin(np.abs(strikes - underlying_price))])
        call_atm = _closest_row(exp_frame, (exp_frame["type"] == "call"), strike=atm_strike)
        put_atm = _closest_row(exp_frame, (exp_frame["type"] == "put"), strike=atm_strike)
        call_25 = _closest_row(exp_frame, exp_frame["type"] == "call", target_delta_abs=0.25)
        put_25 = _closest_row(exp_frame, exp_frame["type"] == "put", target_delta_abs=0.25)
        call_10 = _closest_row(exp_frame, exp_frame["type"] == "call", target_delta_abs=0.10)
        put_10 = _closest_row(exp_frame, exp_frame["type"] == "put", target_delta_abs=0.10)
        atm_iv_values = [row["iv"] for row in [call_atm, put_atm] if row is not None and pd.notna(row["iv"])]
        atm_iv = float(np.mean(atm_iv_values)) if atm_iv_values else None
        straddle_mid = None
        if call_atm is not None and put_atm is not None and pd.notna(call_atm["mid"]) and pd.notna(put_atm["mid"]):
            straddle_mid = float(call_atm["mid"] + put_atm["mid"])
        expected_move_pct = straddle_mid / underlying_price if straddle_mid is not None else None
        iv_expected_move_pct = atm_iv * math.sqrt(days / 365.0) if atm_iv is not None else None
        skew_25 = None
        if call_25 is not None and put_25 is not None:
            skew_25 = float(put_25["iv"] - call_25["iv"])
        skew_10 = None
        if call_10 is not None and put_10 is not None:
            skew_10 = float(put_10["iv"] - call_10["iv"])

        expiries.append(
            {
                "expiration": pd.Timestamp(expiration).strftime("%Y-%m-%d"),
                "days_to_expiry": int(days),
                "atm_strike": atm_strike,
                "atm_call_iv": _row_value(call_atm, "iv"),
                "atm_put_iv": _row_value(put_atm, "iv"),
                "atm_iv_avg": atm_iv,
                "atm_straddle_mid": straddle_mid,
                "straddle_expected_move_pct": expected_move_pct,
                "straddle_low": underlying_price * (1.0 - expected_move_pct) if expected_move_pct is not None else None,
                "straddle_high": underlying_price * (1.0 + expected_move_pct) if expected_move_pct is not None else None,
                "iv_expected_move_pct": iv_expected_move_pct,
                "iv_low": underlying_price * (1.0 - iv_expected_move_pct) if iv_expected_move_pct is not None else None,
                "iv_high": underlying_price * (1.0 + iv_expected_move_pct) if iv_expected_move_pct is not None else None,
                "skew_25d_put_minus_call_iv": skew_25,
                "skew_10d_put_minus_call_iv": skew_10,
                "call_25d": _row_summary(call_25),
                "put_25d": _row_summary(put_25),
                "call_10d": _row_summary(call_10),
                "put_10d": _row_summary(put_10),
            }
        )

    two_week = _select_expiry(expiries, 14, prefer_covering_window=True)
    near_week = _select_expiry(expiries, 7)
    skew = two_week.get("skew_25d_put_minus_call_iv")
    if skew is None:
        skew_signal = "unknown"
    elif skew > 0.03:
        skew_signal = "downside_put_skew"
    elif skew < -0.03:
        skew_signal = "upside_call_skew"
    else:
        skew_signal = "balanced"

    iv_vs_hv20 = two_week["atm_iv_avg"] / hv["hv_20d"] if two_week.get("atm_iv_avg") and hv.get("hv_20d") else None
    if iv_vs_hv20 is None:
        volatility_signal = "unknown"
    elif iv_vs_hv20 >= 1.20:
        volatility_signal = "implied_vol_rich"
    elif iv_vs_hv20 <= 0.85:
        volatility_signal = "implied_vol_cheap"
    else:
        volatility_signal = "implied_vol_fair"

    model_range = None
    if range_projection_path and range_projection_path.exists():
        model_range = json.loads(range_projection_path.read_text(encoding="utf-8")).get("combined_latest_trade_anchor")

    fused_range = None
    if model_range and two_week.get("straddle_low") and two_week.get("straddle_high"):
        fused_range = {
            "low_base": float(np.mean([model_range["base_low"], two_week["straddle_low"]])),
            "high_base": float(np.mean([model_range["base_high"], two_week["straddle_high"]])),
            "low_stress": float(np.mean([model_range["stress_low"], two_week["iv_low"]])),
            "high_optimistic": float(np.mean([model_range["optimistic_high"], two_week["iv_high"]])),
        }

    return {
        "as_of_date": payload.get("as_of_date"),
        "underlying_price": underlying_price,
        "source": payload.get("source", "alpaca_option_snapshot"),
        "feed": payload.get("feed", "indicative"),
        "realized_volatility": hv,
        "near_week": near_week,
        "two_week": two_week,
        "iv_vs_hv20": iv_vs_hv20,
        "volatility_signal": volatility_signal,
        "skew_signal": skew_signal,
        "model_range": model_range,
        "fused_two_week_range": fused_range,
        "expiries": sorted(expiries, key=lambda row: row["days_to_expiry"]),
    }


def _row_value(row: pd.Series | None, key: str) -> float | None:
    if row is None or key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def _row_summary(row: pd.Series | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "symbol": row["symbol"],
        "strike": float(row["strike"]),
        "type": row["type"],
        "mid": _row_value(row, "mid"),
        "iv": _row_value(row, "iv"),
        "delta": _row_value(row, "delta"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NVDA option implied volatility snapshots")
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--underlying-price", type=float, default=215.34)
    parser.add_argument("--range-projection", default=str(PROJECT_ROOT / "outputs" / "two_week_range_projection.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "options_volatility_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_options(
        Path(args.snapshot_json),
        args.underlying_price,
        Path(args.range_projection),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
