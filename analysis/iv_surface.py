"""Option IV surface extraction from yfinance chains."""

from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from data.prices import ensure_dir


def extract_iv_surface(ticker: str, spot: float, as_of_date: pd.Timestamp, target_days: list[int], cache_dir: str | Path) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    try:
        tk = yf.Ticker(ticker)
        expirations = [pd.Timestamp(x) for x in tk.options]
    except Exception as exc:
        return pd.DataFrame([{"ticker": ticker, "status": "BLACK", "warning": f"option expirations unavailable: {exc}"}])
    cache = ensure_dir(cache_dir)
    for target in target_days:
        exp = min(expirations, key=lambda d: abs((d - (as_of_date + pd.Timedelta(days=int(target)))).days))
        try:
            cache_path = cache / f"ivsurface_{_safe(ticker)}_{exp.date()}_{as_of_date.date()}.pkl"
            if cache_path.exists():
                payload = pd.read_pickle(cache_path)
            else:
                chain = tk.option_chain(exp.date().isoformat())
                payload = {"calls": chain.calls.copy(), "puts": chain.puts.copy()}
                pd.to_pickle(payload, cache_path)
            for side, frame in [("call", payload["calls"]), ("put", payload["puts"])]:
                for delta_target in ([0.25, 0.40, 0.50, 0.60, 0.75] if side == "call" else [-0.25, -0.40, -0.50, -0.60, -0.75]):
                    row = _closest_delta(frame, side, delta_target, spot, exp, as_of_date)
                    if row is None:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "target_days": target,
                            "expiration": exp.date().isoformat(),
                            "side": side,
                            "target_delta": delta_target,
                            "strike": float(row.get("strike", np.nan)),
                            "iv": float(row.get("impliedVolatility", np.nan)),
                            "bid": float(row.get("bid", np.nan)),
                            "ask": float(row.get("ask", np.nan)),
                            "status": "GREEN",
                            "source": "yfinance delayed option chain",
                        }
                    )
        except Exception as exc:
            rows.append({"ticker": ticker, "target_days": target, "expiration": exp.date().isoformat(), "status": "BLACK", "warning": str(exc)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return add_surface_metrics(out)


def add_surface_metrics(surface: pd.DataFrame) -> pd.DataFrame:
    out = surface.copy()
    metrics = []
    for (ticker, exp), group in out.groupby(["ticker", "expiration"], dropna=False):
        atm = group.loc[group["target_delta"].abs().sub(0.5).abs().idxmin()] if "target_delta" in group and group["target_delta"].notna().any() else None
        call25 = group[(group["side"] == "call") & (group["target_delta"].round(2) == 0.25)]
        put25 = group[(group["side"] == "put") & (group["target_delta"].round(2) == -0.25)]
        metrics.append(
            {
                "ticker": ticker,
                "expiration": exp,
                "atm_iv": float(atm["iv"]) if atm is not None else np.nan,
                "risk_reversal_25d": float(call25["iv"].mean() - put25["iv"].mean()) if not call25.empty and not put25.empty else np.nan,
                "butterfly_25d": float((call25["iv"].mean() + put25["iv"].mean()) / 2 - float(atm["iv"])) if atm is not None and not call25.empty and not put25.empty else np.nan,
            }
        )
    return out.merge(pd.DataFrame(metrics), on=["ticker", "expiration"], how="left")


def _closest_delta(frame: pd.DataFrame, side: str, target: float, spot: float, exp: pd.Timestamp, as_of: pd.Timestamp) -> pd.Series | None:
    t = max((exp - as_of).days / 365.25, 1 / 365.25)
    best = None
    best_dist = np.inf
    for _, row in frame.iterrows():
        strike = float(pd.to_numeric(row.get("strike"), errors="coerce"))
        iv = float(pd.to_numeric(row.get("impliedVolatility"), errors="coerce"))
        if not np.isfinite(strike) or not np.isfinite(iv) or iv <= 0:
            continue
        d1 = (math.log(spot / strike) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
        delta = stats.norm.cdf(d1) if side == "call" else stats.norm.cdf(d1) - 1.0
        dist = abs(delta - target)
        if dist < best_dist:
            best = row
            best_dist = dist
    return best


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.replace("^", "idx_"))
