from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from nvda_quant_model.config import PROJECT_ROOT


OCC_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<yymmdd>\d{6})(?P<type>[CP])(?P<strike>\d{8})$")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def parse_occ_symbol(symbol: str) -> tuple[pd.Timestamp, str, float]:
    match = OCC_RE.match(symbol)
    if not match:
        raise ValueError(f"Unsupported OCC option symbol: {symbol}")
    expiration = pd.Timestamp("20" + match.group("yymmdd"))
    option_type = "call" if match.group("type") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return expiration, option_type, strike


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(spot: float, strike: float, tau: float, sigma: float, option_type: str, rate: float = 0.04) -> float:
    if spot <= 0 or strike <= 0 or tau <= 0 or sigma <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    vol_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * tau) * _norm_cdf(d2)
    return strike * math.exp(-rate * tau) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_volatility(price: float, spot: float, strike: float, tau: float, option_type: str, rate: float = 0.04) -> float | None:
    if price <= 0 or spot <= 0 or strike <= 0 or tau <= 0:
        return None
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if price < intrinsic * 0.98:
        return None
    low, high = 0.01, 5.0
    for _ in range(80):
        mid = (low + high) / 2.0
        estimate = black_scholes_price(spot, strike, tau, mid, option_type, rate)
        if estimate > price:
            high = mid
        else:
            low = mid
    return float((low + high) / 2.0)


def fetch_alpaca_option_bars(
    symbols: list[str],
    start: str,
    end: str,
    output_csv: Path,
    timeframe: str = "1Day",
    feed: str = "indicative",
    limit: int = 10000,
) -> Path:
    """Download historical option bars from Alpaca's data API.

    Requires `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` environment variables
    and an Alpaca account with options market-data permission. Alpaca bars are
    prices/volume, not a full historical IV surface; IV/skew features are
    derived locally from option close prices.
    """
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY or ALPACA_API_KEY/ALPACA_SECRET_KEY")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    url = "https://data.alpaca.markets/v1beta1/options/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
            "feed": feed,
            "limit": limit,
        }
        if page_token:
            params["page_token"] = page_token
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", {})
        for symbol, items in bars.items():
            for item in items:
                rows.append(
                    {
                        "symbol": symbol,
                        "date": pd.Timestamp(item.get("t")).strftime("%Y-%m-%d"),
                        "open": item.get("o"),
                        "high": item.get("h"),
                        "low": item.get("l"),
                        "close": item.get("c"),
                        "volume": item.get("v"),
                        "trade_count": item.get("n"),
                        "vwap": item.get("vw"),
                    }
                )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def load_option_bars(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    required = {"symbol", "date", "close", "volume"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Option bars missing required columns: {missing}")
    parsed = data["symbol"].apply(parse_occ_symbol)
    data["expiration"] = [row[0] for row in parsed]
    data["type"] = [row[1] for row in parsed]
    data["strike"] = [row[2] for row in parsed]
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data["expiration"] = pd.to_datetime(data["expiration"]).dt.tz_localize(None)
    return data.sort_values(["date", "expiration", "strike", "type"])


def _select_expiry(scoped: pd.DataFrame, date: pd.Timestamp, target_dte: int) -> pd.Timestamp | None:
    expiries = scoped["expiration"].dropna().drop_duplicates()
    if expiries.empty:
        return None
    dtes = (expiries - date).dt.days
    valid = pd.DataFrame({"expiration": expiries, "dte": dtes})
    valid = valid[valid["dte"] > 0]
    if valid.empty:
        return None
    valid["under_penalty"] = (valid["dte"] < target_dte).astype(int)
    valid["distance"] = (valid["dte"] - target_dte).abs()
    return pd.Timestamp(valid.sort_values(["distance", "under_penalty", "dte"]).iloc[0]["expiration"])


def build_option_history_features(
    option_bars: pd.DataFrame,
    underlying_close: pd.Series,
    target_dte: int = 14,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """Build daily option-derived features from point-in-time option bars."""
    close = underlying_close.sort_index()
    rows: list[dict[str, Any]] = []
    for date, day in option_bars.groupby("date"):
        date = pd.Timestamp(date).normalize()
        if date not in close.index:
            continue
        spot = float(close.loc[date])
        expiration = _select_expiry(day, date, target_dte)
        if expiration is None:
            continue
        scoped = day[day["expiration"] == expiration].copy()
        if scoped.empty:
            continue
        dte = max((expiration - date).days, 1)
        tau = dte / 365.0
        strikes = scoped["strike"].dropna().unique()
        if len(strikes) == 0:
            continue
        atm_strike = float(strikes[np.argmin(np.abs(strikes - spot))])
        call_atm = scoped[(scoped["type"] == "call") & (scoped["strike"] == atm_strike)]
        put_atm = scoped[(scoped["type"] == "put") & (scoped["strike"] == atm_strike)]
        if call_atm.empty or put_atm.empty:
            continue
        call_close = float(call_atm.iloc[0]["close"])
        put_close = float(put_atm.iloc[0]["close"])
        call_iv = implied_volatility(call_close, spot, atm_strike, tau, "call", risk_free_rate)
        put_iv = implied_volatility(put_close, spot, atm_strike, tau, "put", risk_free_rate)
        iv_values = [value for value in [call_iv, put_iv] if value is not None]
        atm_iv = float(np.mean(iv_values)) if iv_values else np.nan

        calls = scoped[scoped["type"] == "call"]
        puts = scoped[scoped["type"] == "put"]
        call_otm = calls.iloc[(calls["strike"] - spot * 1.05).abs().argsort().iloc[0]] if not calls.empty else None
        put_otm = puts.iloc[(puts["strike"] - spot * 0.95).abs().argsort().iloc[0]] if not puts.empty else None
        call_otm_iv = (
            implied_volatility(float(call_otm["close"]), spot, float(call_otm["strike"]), tau, "call", risk_free_rate)
            if call_otm is not None
            else None
        )
        put_otm_iv = (
            implied_volatility(float(put_otm["close"]), spot, float(put_otm["strike"]), tau, "put", risk_free_rate)
            if put_otm is not None
            else None
        )
        total_call_volume = float(calls["volume"].fillna(0).sum())
        total_put_volume = float(puts["volume"].fillna(0).sum())
        rows.append(
            {
                "Date": date,
                "opt_expiration": expiration.strftime("%Y-%m-%d"),
                "opt_dte": dte,
                "opt_atm_strike": atm_strike,
                "opt_atm_straddle_move": (call_close + put_close) / spot,
                "opt_atm_iv": atm_iv,
                "opt_atm_call_iv": call_iv,
                "opt_atm_put_iv": put_iv,
                "opt_atm_put_minus_call_iv": (put_iv - call_iv) if put_iv is not None and call_iv is not None else np.nan,
                "opt_5pct_put_minus_call_iv": (put_otm_iv - call_otm_iv) if put_otm_iv is not None and call_otm_iv is not None else np.nan,
                "opt_put_call_volume_ratio": total_put_volume / total_call_volume if total_call_volume > 0 else np.nan,
                "opt_total_volume": total_call_volume + total_put_volume,
            }
        )
    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(index=close.index)
    features = features.set_index("Date").sort_index()
    numeric_cols = [col for col in features.columns if col != "opt_expiration"]
    features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="coerce")
    features["opt_iv_rank_60"] = features["opt_atm_iv"].rolling(60, min_periods=20).rank(pct=True)
    features["opt_straddle_move_rank_60"] = features["opt_atm_straddle_move"].rolling(60, min_periods=20).rank(pct=True)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or transform historical option bars into model features")
    parser.add_argument("--symbols", default="", help="Comma-separated OCC option symbols for Alpaca historical bars")
    parser.add_argument("--start", default="2024-05-22")
    parser.add_argument("--end", default="2026-05-22")
    parser.add_argument("--feed", default="indicative")
    parser.add_argument("--bars-csv", default="")
    parser.add_argument("--underlying-csv", default="")
    parser.add_argument("--target-dte", type=int, default=14)
    parser.add_argument("--download-output", default=str(PROJECT_ROOT / "outputs" / "historical_option_bars.csv"))
    parser.add_argument("--features-output", default=str(PROJECT_ROOT / "outputs" / "historical_option_features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars_csv = Path(args.bars_csv) if args.bars_csv else None
    if args.symbols:
        symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
        bars_csv = fetch_alpaca_option_bars(symbols, args.start, args.end, Path(args.download_output), feed=args.feed)
        print(f"Downloaded option bars: {bars_csv}")
    if not bars_csv:
        raise RuntimeError("Provide --bars-csv or --symbols")
    if not args.underlying_csv:
        raise RuntimeError("Provide --underlying-csv with Date and Close columns")
    bars = load_option_bars(bars_csv)
    underlying = pd.read_csv(args.underlying_csv, parse_dates=["Date"], index_col="Date")["Close"]
    features = build_option_history_features(bars, underlying, target_dte=args.target_dte)
    output = Path(args.features_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output)
    print(json.dumps({"bars": str(bars_csv), "features": str(output), "rows": int(len(features))}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
