from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.event_overlay.schema import EVENT_TYPES, assert_no_event_feature_leakage


def _safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    required = {"Close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"market_data missing required columns: {sorted(missing)}")
    if "Volume" not in data.columns:
        data["Volume"] = np.nan
    return data.sort_index()


def _history_before(prices: pd.DataFrame, published_at: pd.Timestamp) -> pd.DataFrame:
    """Use only daily bars whose session date is strictly before the event date."""

    event_date = pd.Timestamp(published_at).normalize()
    return prices.loc[prices.index.normalize() < event_date]


def _pct(close: pd.Series, days: int) -> float:
    if len(close) <= days or close.iloc[-days - 1] == 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1.0)


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    value = 100.0 - 100.0 / (1.0 + rs.iloc[-1]) if len(rs) and pd.notna(rs.iloc[-1]) else 50.0
    return float(np.clip(value, 0.0, 100.0))


def _market_context_features(history: pd.DataFrame) -> dict[str, float]:
    close = history["Close"].astype(float).dropna()
    volume = history["Volume"].astype(float).dropna()
    out = {
        "pre_return_1d": 0.0,
        "pre_return_3d": 0.0,
        "pre_return_5d": 0.0,
        "pre_return_20d": 0.0,
        "pre_realized_vol_5d": 0.0,
        "pre_realized_vol_20d": 0.0,
        "volume_anomaly_20d": 0.0,
        "ema_20_ratio": 0.0,
        "ema_50_ratio": 0.0,
        "ema_20_50_spread": 0.0,
        "macd_hist": 0.0,
        "rsi_14": 50.0,
        "boll_percent_b_20": 0.5,
        "boll_bandwidth_20": 0.0,
        "market_regime_high_vol": 0.0,
        "market_regime_uptrend": 0.0,
        "market_regime_downtrend": 0.0,
    }
    if len(close) < 2:
        return out
    returns = close.pct_change().dropna()
    out.update(
        {
            "pre_return_1d": _pct(close, 1),
            "pre_return_3d": _pct(close, min(3, len(close) - 1)),
            "pre_return_5d": _pct(close, min(5, len(close) - 1)),
            "pre_return_20d": _pct(close, min(20, len(close) - 1)),
            "pre_realized_vol_5d": float(returns.tail(5).std(ddof=0)) if len(returns) else 0.0,
            "pre_realized_vol_20d": float(returns.tail(20).std(ddof=0)) if len(returns) else 0.0,
            "rsi_14": _rsi(close),
        }
    )
    if len(volume) >= 20:
        base = float(volume.tail(20).mean())
        out["volume_anomaly_20d"] = float(volume.iloc[-1] / base - 1.0) if base else 0.0

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=1).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    last_close = float(close.iloc[-1])
    out["ema_20_ratio"] = float(last_close / ema20.iloc[-1] - 1.0) if ema20.iloc[-1] else 0.0
    out["ema_50_ratio"] = float(last_close / ema50.iloc[-1] - 1.0) if ema50.iloc[-1] else 0.0
    out["ema_20_50_spread"] = float(ema20.iloc[-1] / ema50.iloc[-1] - 1.0) if ema50.iloc[-1] else 0.0
    out["macd_hist"] = float((macd.iloc[-1] - macd_signal.iloc[-1]) / last_close) if last_close else 0.0

    sma20 = close.rolling(20, min_periods=2).mean()
    std20 = close.rolling(20, min_periods=2).std(ddof=0)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    width = upper.iloc[-1] - lower.iloc[-1] if len(upper) else np.nan
    if pd.notna(width) and width:
        out["boll_percent_b_20"] = float(np.clip((last_close - lower.iloc[-1]) / width, -1.0, 2.0))
        out["boll_bandwidth_20"] = float(width / last_close) if last_close else 0.0

    vol20 = out["pre_realized_vol_20d"]
    vol60 = float(returns.tail(60).std(ddof=0)) if len(returns) else vol20
    out["market_regime_high_vol"] = float(vol20 > max(vol60 * 1.20, 0.02))
    out["market_regime_uptrend"] = float(out["ema_20_50_spread"] > 0.0 and out["pre_return_20d"] > 0.0)
    out["market_regime_downtrend"] = float(out["ema_20_50_spread"] < 0.0 and out["pre_return_20d"] < 0.0)
    return out


def _external_features(external_data: pd.DataFrame | None, published_at: pd.Timestamp) -> dict[str, float]:
    out = {
        "SMH_return_5d": 0.0,
        "QQQ_return_5d": 0.0,
        "SPY_return_5d": 0.0,
        "VIX_level": 0.0,
        "VIX_change_5d": 0.0,
    }
    if external_data is None or external_data.empty:
        return out
    data = external_data.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    hist = data.loc[data.index.normalize() < pd.Timestamp(published_at).normalize()].sort_index()
    if hist.empty:
        return out
    for ticker, key in [("SMH", "SMH_return_5d"), ("QQQ", "QQQ_return_5d"), ("SPY", "SPY_return_5d")]:
        if ticker in hist.columns and hist[ticker].dropna().shape[0] > 5:
            series = hist[ticker].dropna()
            out[key] = float(series.iloc[-1] / series.iloc[-6] - 1.0) if series.iloc[-6] else 0.0
    vix_col = "VIX" if "VIX" in hist.columns else "^VIX" if "^VIX" in hist.columns else None
    if vix_col and hist[vix_col].dropna().shape[0] > 5:
        series = hist[vix_col].dropna()
        out["VIX_level"] = float(series.iloc[-1])
        out["VIX_change_5d"] = float(series.iloc[-1] / series.iloc[-6] - 1.0) if series.iloc[-6] else 0.0
    return out


def build_event_feature_frame(
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    include_event_context: bool = True,
) -> pd.DataFrame:
    """Build point-in-time event features from pre-publication market data."""

    classified = classify_events(events)
    prices = _safe_prices(market_data)
    rows: list[dict[str, float]] = []
    for _, row in classified.iterrows():
        history = _history_before(prices, row["published_at"])
        features = _market_context_features(history)
        features.update(_external_features(external_data, row["published_at"]))
        if include_event_context:
            features["source_quality"] = float(row["source_quality"])
            features["nvda_relevance"] = float(row["nvda_relevance"])
            features["event_importance"] = float(row["event_importance"])
            features["event_sentiment_keyword"] = float(_keyword_sentiment(row))
            for event_type in EVENT_TYPES:
                features[f"event_type_{event_type}"] = float(row["event_type"] == event_type)
        rows.append(features)
    out = pd.DataFrame(rows, index=classified.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_event_feature_leakage(list(out.columns))
    return out


def _keyword_sentiment(row: pd.Series) -> float:
    text = f"{row.get('headline', '')} {row.get('body', '')}".lower()
    positive = ("beat", "raise", "upgrade", "strong", "growth", "record", "demand")
    negative = ("miss", "cut", "downgrade", "delay", "restriction", "probe", "weak")
    raw = sum(term in text for term in positive) - sum(term in text for term in negative)
    return float(np.tanh(raw / 3.0))
