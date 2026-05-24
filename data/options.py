"""Option-implied event pricing from yfinance option chains."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from data.prices import ensure_dir


@dataclass(slots=True)
class OptionEventPricing:
    event_id: str
    event_date: str
    ticker: str
    spot: float
    expiration: str
    days_to_expiry: int
    atm_strike: float
    call_mid: float
    put_mid: float
    straddle_price: float
    implied_move: float
    atm_iv: float
    next_atm_iv: float | None
    term_slope: float | None
    put_25d_iv: float | None
    call_25d_iv: float | None
    skew_25d: float | None
    source: str
    warnings: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def price_event_options(
    event_id: str,
    event_date: pd.Timestamp,
    ticker: str,
    spot: float,
    as_of_date: pd.Timestamp,
    cache_dir: str | Path,
) -> OptionEventPricing:
    warnings: list[str] = []
    try:
        chain = _get_chain_for_event(ticker, event_date, as_of_date, cache_dir)
        calls, puts, expiration, source = chain
        atm = _atm_pair(calls, puts, spot)
        call_mid = _mid(atm["call"])
        put_mid = _mid(atm["put"])
        if not np.isfinite(call_mid) or not np.isfinite(put_mid) or call_mid <= 0 or put_mid <= 0:
            raise ValueError("ATM call/put mid is unavailable.")
        straddle = call_mid + put_mid
        implied_move = straddle / spot
        atm_iv = _median_iv([atm["call"], atm["put"]])
        next_iv = _next_expiration_atm_iv(ticker, expiration, spot, as_of_date, cache_dir, warnings)
        slope = None if next_iv is None or not np.isfinite(atm_iv) else float(atm_iv - next_iv)
        put25, call25 = _skew(calls, puts, spot, expiration, as_of_date)
        skew = None if put25 is None or call25 is None else float(put25 - call25)
        return OptionEventPricing(
            event_id=event_id,
            event_date=event_date.date().isoformat(),
            ticker=ticker,
            spot=float(spot),
            expiration=expiration.date().isoformat(),
            days_to_expiry=max(0, int((expiration.normalize() - as_of_date.normalize()).days)),
            atm_strike=float(atm["strike"]),
            call_mid=float(call_mid),
            put_mid=float(put_mid),
            straddle_price=float(straddle),
            implied_move=float(implied_move),
            atm_iv=float(atm_iv) if np.isfinite(atm_iv) else float("nan"),
            next_atm_iv=float(next_iv) if next_iv is not None and np.isfinite(next_iv) else None,
            term_slope=slope,
            put_25d_iv=put25,
            call_25d_iv=call25,
            skew_25d=skew,
            source=source,
            warnings="; ".join(warnings),
        )
    except Exception as exc:  # pragma: no cover - depends on live option chain
        warnings.append(f"Option pricing failed: {exc}")
        return OptionEventPricing(
            event_id=event_id,
            event_date=event_date.date().isoformat(),
            ticker=ticker,
            spot=float(spot),
            expiration="",
            days_to_expiry=0,
            atm_strike=float("nan"),
            call_mid=float("nan"),
            put_mid=float("nan"),
            straddle_price=float("nan"),
            implied_move=float("nan"),
            atm_iv=float("nan"),
            next_atm_iv=None,
            term_slope=None,
            put_25d_iv=None,
            call_25d_iv=None,
            skew_25d=None,
            source="missing",
            warnings="; ".join(warnings),
        )


def _get_chain_for_event(
    ticker: str,
    event_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
    cache_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, str]:
    import yfinance as yf

    cache = ensure_dir(cache_dir)
    tk = yf.Ticker(ticker)
    expirations = [pd.Timestamp(x) for x in tk.options]
    if not expirations:
        raise ValueError(f"No option expirations for {ticker}")
    expiration = min([x for x in expirations if x >= event_date] or expirations, key=lambda x: abs((x - event_date).days))
    cache_path = cache / f"options_{_safe(ticker)}_{expiration.date()}_{as_of_date.date()}.pkl"
    if cache_path.exists():
        payload = pd.read_pickle(cache_path)
        return payload["calls"], payload["puts"], expiration, f"cache:yfinance:{cache_path.name}"
    raw = tk.option_chain(expiration.date().isoformat())
    payload = {"calls": raw.calls.copy(), "puts": raw.puts.copy()}
    pd.to_pickle(payload, cache_path)
    return payload["calls"], payload["puts"], expiration, "yfinance delayed option chain"


def _next_expiration_atm_iv(
    ticker: str,
    expiration: pd.Timestamp,
    spot: float,
    as_of_date: pd.Timestamp,
    cache_dir: str | Path,
    warnings: list[str],
) -> float | None:
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        expirations = [pd.Timestamp(x) for x in tk.options if pd.Timestamp(x) > expiration + pd.Timedelta(days=7)]
        if not expirations:
            return None
        next_exp = min(expirations, key=lambda x: abs((x - (expiration + pd.Timedelta(days=30))).days))
        cache = ensure_dir(cache_dir)
        cache_path = cache / f"options_{_safe(ticker)}_{next_exp.date()}_{as_of_date.date()}.pkl"
        if cache_path.exists():
            payload = pd.read_pickle(cache_path)
        else:
            chain = tk.option_chain(next_exp.date().isoformat())
            payload = {"calls": chain.calls.copy(), "puts": chain.puts.copy()}
            pd.to_pickle(payload, cache_path)
        atm = _atm_pair(payload["calls"], payload["puts"], spot)
        return _median_iv([atm["call"], atm["put"]])
    except Exception as exc:  # pragma: no cover
        warnings.append(f"Term-structure IV unavailable: {exc}")
        return None


def _atm_pair(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> dict[str, Any]:
    strikes = sorted(set(pd.to_numeric(calls["strike"], errors="coerce").dropna()).intersection(
        set(pd.to_numeric(puts["strike"], errors="coerce").dropna())
    ))
    if not strikes:
        raise ValueError("No common call/put strikes.")
    strike = min(strikes, key=lambda x: abs(float(x) - spot))
    call = calls.loc[pd.to_numeric(calls["strike"], errors="coerce") == strike].iloc[0]
    put = puts.loc[pd.to_numeric(puts["strike"], errors="coerce") == strike].iloc[0]
    return {"strike": float(strike), "call": call, "put": put}


def _mid(row: pd.Series) -> float:
    bid = float(pd.to_numeric(row.get("bid"), errors="coerce"))
    ask = float(pd.to_numeric(row.get("ask"), errors="coerce"))
    last = float(pd.to_numeric(row.get("lastPrice"), errors="coerce"))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return last if np.isfinite(last) else float("nan")


def _median_iv(rows: list[pd.Series]) -> float:
    vals = []
    for row in rows:
        val = float(pd.to_numeric(row.get("impliedVolatility"), errors="coerce"))
        if np.isfinite(val) and 0.001 < val < 5:
            vals.append(val)
    return float(np.median(vals)) if vals else float("nan")


def _skew(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiration: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> tuple[float | None, float | None]:
    t = max((expiration - as_of_date).days / 365.25, 1 / 365.25)
    call = _row_closest_delta(calls, spot, t, target=0.25, option_type="call")
    put = _row_closest_delta(puts, spot, t, target=-0.25, option_type="put")
    call_iv = None if call is None else float(call.get("impliedVolatility"))
    put_iv = None if put is None else float(put.get("impliedVolatility"))
    return put_iv, call_iv


def _row_closest_delta(
    frame: pd.DataFrame,
    spot: float,
    t: float,
    target: float,
    option_type: str,
) -> pd.Series | None:
    rows = []
    for _, row in frame.iterrows():
        strike = float(pd.to_numeric(row.get("strike"), errors="coerce"))
        iv = float(pd.to_numeric(row.get("impliedVolatility"), errors="coerce"))
        if not np.isfinite(strike) or not np.isfinite(iv) or iv <= 0:
            continue
        d1 = (math.log(spot / strike) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
        delta = stats.norm.cdf(d1) if option_type == "call" else stats.norm.cdf(d1) - 1.0
        rows.append((abs(delta - target), row))
    if not rows:
        return None
    return min(rows, key=lambda x: x[0])[1]


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.replace("^", "idx_"))
