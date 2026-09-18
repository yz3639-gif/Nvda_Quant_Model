from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.data.session_calendar import as_utc, session_labels, session_schedule


def _safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    data.index = session_labels(data.index)
    if data.index.has_duplicates:
        raise ValueError("Duplicate market sessions")
    if "Close" not in data.columns:
        raise ValueError("market_data must contain Close")
    return data.sort_index()


def _event_anchor_position(index: pd.DatetimeIndex, published_at: pd.Timestamp) -> int | None:
    if pd.isna(published_at):
        return None
    closes = session_schedule(index, allow_non_sessions=True)["market_close"]
    eligible = np.flatnonzero((closes >= as_utc(published_at)).to_numpy())
    return int(eligible[0]) if len(eligible) else None


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
    """Delayed-close forward-return labels for evaluation/training only.

    Baseline is the FIRST session close at or after event availability; outcomes
    end 1/3/5 sessions later. This excludes pre-publication movement and the
    immediate event reaction, and is not a causal event-impact estimate.
    Features separately use only completed closes at event availability.
    Labels intentionally use future prices and must never enter features.
    """

    if not horizons or any(not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError("Label horizons must be positive session counts")
    classified = classify_events(events)
    prices = _safe_prices(market_data)
    schedule = session_schedule(prices.index, allow_non_sessions=True)
    prices = prices.loc[schedule["market_close"].notna()]
    closes_at = session_schedule(prices.index)["market_close"]
    close = prices["Close"].astype(float)
    returns = close.pct_change().dropna()
    rows: list[dict[str, float | str | int]] = []
    for _, row in classified.iterrows():
        anchor = _event_anchor_position(prices.index, row["available_at"])
        labels: dict = {"event_id": row["event_id"], "decision_at": row["available_at"],
                        "label_end_at": pd.NaT, "anchor_at": pd.NaT,
                        "label_method": "first_close_at_or_after_availability_to_subsequent_close"}
        if anchor is None:
            for horizon in horizons:
                labels[f"future_{horizon}d_return"] = np.nan
                labels[f"future_{horizon}d_direction"] = "unknown"
            labels["volatility_shock"] = np.nan
            labels["downside_tail_risk"] = np.nan
            rows.append(labels)
            continue
        labels["anchor_at"] = closes_at.iloc[anchor]
        for horizon in horizons:
            labels[f"label_end_at_{horizon}d"] = closes_at.iloc[anchor + horizon] if anchor + horizon < len(prices) else pd.NaT
            ret = _future_return(close, anchor, horizon)
            labels[f"future_{horizon}d_return"] = ret
            labels[f"future_{horizon}d_direction"] = _direction(ret, direction_threshold)
        if anchor + max(horizons) >= len(prices):
            labels["volatility_shock"] = np.nan
            labels["downside_tail_risk"] = np.nan
            rows.append(labels)
            continue
        labels["label_end_at"] = closes_at.iloc[anchor + max(horizons)]
        known = np.flatnonzero((closes_at <= row["available_at"]).to_numpy())
        known_pos = int(known[-1]) if len(known) else 0
        pre_vol = float(returns.iloc[max(0, known_pos - 20) : known_pos].std(ddof=0)) if known_pos > 1 else 0.0
        future_returns = returns.iloc[anchor : min(anchor + max(horizons), len(returns))]
        future_vol = float(future_returns.std(ddof=0)) if len(future_returns) else 0.0
        labels["volatility_shock"] = int(future_vol > max(pre_vol * volatility_multiplier, 0.01))
        path = close.iloc[anchor : min(anchor + max(horizons) + 1, len(close))]
        path_returns = path / close.iloc[anchor] - 1.0 if close.iloc[anchor] else pd.Series(dtype=float)
        labels["downside_tail_risk"] = int(float(path_returns.min()) <= downside_tail_threshold) if len(path_returns) else 0
        rows.append(labels)
    return pd.DataFrame(rows)
