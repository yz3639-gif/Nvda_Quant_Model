"""Event-specific pre-drift baskets for Bayesian conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def basket_for_event(event, config: dict) -> list[str]:
    baskets = config.get("event_pre_drift_baskets", {})
    for category in event.categories:
        if category in baskets:
            return list(baskets[category])
    return list(baskets.get("DEFAULT", ["SPY"]))


def current_event_pre_drift(event, prices: pd.DataFrame, config: dict, as_of_date: pd.Timestamp) -> dict:
    basket = basket_for_event(event, config)
    drift = basket_drift(prices, basket, as_of_date, lookback=20)
    return {
        "event_id": event.id,
        "basket": ",".join(basket),
        "lookback_days": 20,
        "current_pre_drift": drift,
        "definition": "equal-weight basket return over [-20,-1] trading days using event-relevant assets",
    }


def historical_sample_pre_drifts(samples: pd.DataFrame, event, prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    basket = basket_for_event(event, config)
    rows = []
    for _, row in samples.iterrows():
        date = pd.Timestamp(row["date"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "event_id": event.id,
                "basket": ",".join(basket),
                "pre_drift": basket_drift(prices, basket, date - pd.Timedelta(days=1), lookback=20),
            }
        )
    return pd.DataFrame(rows)


def basket_drift(prices: pd.DataFrame, basket: list[str], end_date: pd.Timestamp, lookback: int = 20) -> float:
    returns = []
    for symbol in basket:
        if symbol not in prices:
            continue
        series = pd.to_numeric(prices[symbol], errors="coerce").dropna().sort_index()
        series = series.loc[series.index <= end_date]
        if series.shape[0] <= lookback:
            continue
        returns.append(float(series.iloc[-1] / series.iloc[-lookback - 1] - 1.0))
    return float(np.nanmean(returns)) if returns else float("nan")
