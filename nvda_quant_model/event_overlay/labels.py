from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.event_overlay.event_classifier import classify_events


def _safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    if "Close" not in data.columns:
        raise ValueError("market_data must contain Close")
    return data.sort_index()


def _event_anchor_position(index: pd.DatetimeIndex, published_at: pd.Timestamp) -> int | None:
    event_date = pd.Timestamp(published_at).normalize()
    pos = int(np.searchsorted(index.normalize().to_numpy(), np.datetime64(event_date), side="left"))
    if pos <= 0 or pos >= len(index):
        return None
    return pos - 1


def _future_return(close: pd.Series, anchor_pos: int, horizon: int) -> float:
    future_pos = anchor_pos + horizon
    if future_pos >= len(close) or close.iloc[anchor_pos] == 0:
        return np.nan
    return float(close.iloc[future_pos] / close.iloc[anchor_pos] - 1.0)


def _direction(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > threshold:
        return "bullish"
    if value < -threshold:
        return "bearish"
    return "neutral"


def build_event_labels(
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    direction_threshold: float = 0.005,
    volatility_multiplier: float = 1.25,
    downside_tail_threshold: float = -0.03,
) -> pd.DataFrame:
    """Create future labels for evaluation/training only.

    Labels intentionally use future prices and must never be merged back into
    the event feature matrix.
    """

    classified = classify_events(events)
    prices = _safe_prices(market_data)
    close = prices["Close"].astype(float)
    returns = close.pct_change().dropna()
    rows: list[dict[str, float | str | int]] = []
    for _, row in classified.iterrows():
        anchor = _event_anchor_position(prices.index, row["published_at"])
        labels: dict[str, float | str | int] = {"event_id": row["event_id"]}
        if anchor is None:
            for horizon in horizons:
                labels[f"future_{horizon}d_return"] = np.nan
                labels[f"future_{horizon}d_direction"] = "unknown"
            labels["volatility_shock"] = 0
            labels["downside_tail_risk"] = 0
            rows.append(labels)
            continue
        for horizon in horizons:
            ret = _future_return(close, anchor, horizon)
            labels[f"future_{horizon}d_return"] = ret
            labels[f"future_{horizon}d_direction"] = _direction(ret, direction_threshold)
        pre_vol = float(returns.iloc[max(0, anchor - 20) : anchor].std(ddof=0)) if anchor > 1 else 0.0
        future_returns = returns.iloc[anchor : min(anchor + max(horizons), len(returns))]
        future_vol = float(future_returns.std(ddof=0)) if len(future_returns) else 0.0
        labels["volatility_shock"] = int(future_vol > max(pre_vol * volatility_multiplier, 0.01))
        path = close.iloc[anchor : min(anchor + max(horizons) + 1, len(close))]
        path_returns = path / close.iloc[anchor] - 1.0 if close.iloc[anchor] else pd.Series(dtype=float)
        labels["downside_tail_risk"] = int(float(path_returns.min()) <= downside_tail_threshold) if len(path_returns) else 0
        rows.append(labels)
    return pd.DataFrame(rows)
