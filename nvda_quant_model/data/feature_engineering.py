from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from nvda_quant_model.config import MACRO_TICKERS, PEER_TICKERS, SECTOR_TICKERS
from nvda_quant_model.data.data_validation import clean_model_frame, validate_ohlcv
from nvda_quant_model.data.load_data import fetch_nvda_fundamentals


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = prices["High"] - prices["Low"]
    high_close = (prices["High"] - prices["Close"].shift()).abs()
    low_close = (prices["Low"] - prices["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def _external_column(external: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series | None:
    if primary in external.columns:
        return external[primary]
    if fallback and fallback in external.columns:
        return external[fallback]
    return None


def build_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = validate_ohlcv(prices)
    close = prices["Close"]
    returns = close.pct_change()
    features = prices.copy()
    features["daily_return"] = returns

    for window in [5, 10, 20, 60]:
        features[f"{window}d_return"] = close.pct_change(window)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma60 = close.rolling(60, min_periods=60).mean()
    std20 = close.rolling(20, min_periods=20).std()
    features["price_20ma_ratio"] = close / sma20
    features["price_60ma_ratio"] = close / sma60
    features["price_zscore_20"] = (close - sma20) / std20

    features["volatility_20"] = returns.rolling(20, min_periods=20).std()
    features["volatility_60"] = returns.rolling(60, min_periods=60).std()
    features["atr_ratio"] = _atr(prices, 14) / close

    features["volume_sma_ratio"] = prices["Volume"] / prices["Volume"].rolling(20, min_periods=20).mean()
    obv = _obv(close, prices["Volume"])
    features["obv_signal"] = obv - obv.ewm(span=10, adjust=False).mean()

    rsi = _rsi(close, 14)
    features["rsi_14"] = rsi
    macd_line, macd_signal, macd_hist = _macd(close)
    features["macd_line"] = macd_line / close
    features["macd_signal"] = macd_signal / close
    features["macd_hist"] = macd_hist / close
    rolling_high = close.eq(close.rolling(20, min_periods=20).max())
    features["rsi_divergence"] = ((rolling_high) & (rsi < rsi.shift(5))).astype(float)
    return features


def build_holiday_features(index: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    """Trading-calendar sentiment features for long weekends and holidays.

    Signals are computed from information known at the current close. For
    example, 2025-05-23 knows the next trading session is 2025-05-27 and that
    2025-05-26 is a US holiday.
    """
    holidays = USFederalHolidayCalendar().holidays(
        start=index.min() - pd.Timedelta(days=10),
        end=index.max() + pd.Timedelta(days=10),
    )
    holiday_dates = set(pd.to_datetime(holidays).normalize())
    out = pd.DataFrame(index=index)

    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    current = pd.Series(index, index=index)
    next_values = list(index[1:]) + [index[-1] + us_bday]
    prev_values = [index[0] - us_bday] + list(index[:-1])
    next_session = pd.Series(pd.DatetimeIndex(next_values), index=index)
    prev_session = pd.Series(pd.DatetimeIndex(prev_values), index=index)
    out["next_session_gap_days"] = (next_session - current).dt.days.fillna(1).clip(lower=1)
    out["prev_session_gap_days"] = (current - prev_session).dt.days.fillna(1).clip(lower=1)

    def has_holiday_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
        if pd.isna(start) or pd.isna(end) or end <= start:
            return 0
        days = pd.date_range(start + pd.Timedelta(days=1), end, freq="D").normalize()
        return int(any(day in holiday_dates for day in days))

    out["next_session_has_holiday"] = [
        has_holiday_between(cur, nxt) for cur, nxt in zip(current, next_session)
    ]
    out["prev_session_had_holiday"] = [
        has_holiday_between(prev, cur) for prev, cur in zip(prev_session, current)
    ]
    out["pre_holiday_session"] = (
        (out["next_session_gap_days"] >= 3) & (out["next_session_has_holiday"] == 1)
    ).astype(float)
    out["post_holiday_session"] = (
        (out["prev_session_gap_days"] >= 3) & (out["prev_session_had_holiday"] == 1)
    ).astype(float)
    out["holiday_week"] = [
        int(any(day in holiday_dates for day in pd.date_range(day - pd.Timedelta(days=2), day + pd.Timedelta(days=5), freq="D").normalize()))
        for day in index.normalize()
    ]
    recent_return = close.pct_change(5)
    out["pre_holiday_momentum"] = recent_return.where(out["pre_holiday_session"] == 1, 0.0)
    return out


def build_external_features(external: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    external = external.sort_index().reindex(index).ffill()
    features = pd.DataFrame(index=index)

    vix = _external_column(external, MACRO_TICKERS["VIX"])
    spx = _external_column(external, MACRO_TICKERS["SP500"])
    nasdaq = _external_column(external, MACRO_TICKERS["NASDAQ"])
    usd = _external_column(external, MACRO_TICKERS["USD"], MACRO_TICKERS["USD_FALLBACK"])
    tnx = _external_column(external, MACRO_TICKERS["TREASURY_10Y"])
    hyg = _external_column(external, MACRO_TICKERS["HYG"])
    lqd = _external_column(external, MACRO_TICKERS["LQD"])

    if vix is not None:
        features["VIX"] = vix
        features["VIX_weekly_change"] = vix.pct_change(5)
    if spx is not None:
        features["SP500_return"] = spx.pct_change(5)
        features["SP500_daily_return"] = spx.pct_change()
    if nasdaq is not None:
        features["NASDAQ_return"] = nasdaq.pct_change(5)
    if usd is not None:
        features["USD_index"] = usd
        features["USD_index_return"] = usd.pct_change(5)
    if tnx is not None:
        features["Treasury_10Y"] = tnx / 10.0
        features["Treasury_10Y_change"] = (tnx / 10.0).diff(5)
    if hyg is not None and lqd is not None:
        ratio = hyg / lqd
        features["HY_spread"] = -ratio.pct_change(5)

    for name, ticker in SECTOR_TICKERS.items():
        col = _external_column(external, ticker)
        if col is not None:
            features[f"{name}_return"] = col.pct_change(5)

    return features


def build_peer_market_features(external: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Peer close features from the semiconductor supply chain.

    These features use only prices known at the current close, then predict the
    next session. They add industry breadth and relative-strength context even
    when peer OHLCV event data is unavailable.
    """
    external = external.sort_index().reindex(index).ffill()
    peer_returns_1d: dict[str, pd.Series] = {}
    peer_returns_5d: dict[str, pd.Series] = {}
    peer_above_20ma: dict[str, pd.Series] = {}
    out = pd.DataFrame(index=index)

    for name, ticker in PEER_TICKERS.items():
        if ticker not in external.columns:
            continue
        close = external[ticker]
        ret1 = close.pct_change()
        ret5 = close.pct_change(5)
        peer_returns_1d[name] = ret1
        peer_returns_5d[name] = ret5
        peer_above_20ma[name] = close > close.rolling(20, min_periods=20).mean()
        if name in {"AMD", "AVGO", "TSM", "MU"}:
            out[f"peer_{name}_return_5d"] = ret5

    if not peer_returns_1d:
        return out

    peer1 = pd.DataFrame(peer_returns_1d)
    peer5 = pd.DataFrame(peer_returns_5d)
    above = pd.DataFrame(peer_above_20ma).astype(float)
    out["peer_mean_return_1d"] = peer1.mean(axis=1)
    out["peer_median_return_1d"] = peer1.median(axis=1)
    out["peer_mean_return_5d"] = peer5.mean(axis=1)
    out["peer_median_return_5d"] = peer5.median(axis=1)
    out["peer_return_dispersion_5d"] = peer5.std(axis=1)
    out["peer_positive_breadth_1d"] = (peer1 > 0).mean(axis=1)
    out["peer_positive_breadth_5d"] = (peer5 > 0).mean(axis=1)
    out["peer_above_20ma_breadth"] = above.mean(axis=1)
    if "QQQ_return" in out:
        out["peer_vs_qqq_5d"] = out["peer_mean_return_5d"] - out["QQQ_return"]
    elif "QQQ" in external.columns:
        out["peer_vs_qqq_5d"] = out["peer_mean_return_5d"] - external["QQQ"].pct_change(5)
    return out


def build_peer_business_event_features(peer_ohlcv: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time proxy for peer earnings/business shocks.

    We do not claim to know every historical earnings surprise from free data.
    Instead, we infer business-event pressure from peer close-to-close moves
    and volume spikes relative to each ticker's trailing distribution. The
    threshold is shifted one day, so a day's event classification never uses
    that day's value to define its own cutoff.
    """
    out = pd.DataFrame(index=index)
    if not peer_ohlcv:
        return out

    ret1_map: dict[str, pd.Series] = {}
    vol_ratio_map: dict[str, pd.Series] = {}
    shock_map: dict[str, pd.Series] = {}
    pos_map: dict[str, pd.Series] = {}
    neg_map: dict[str, pd.Series] = {}

    for name, raw in peer_ohlcv.items():
        data = raw.sort_index().reindex(index).ffill()
        if data.empty or "Close" not in data or "Volume" not in data:
            continue
        close = data["Close"]
        volume = data["Volume"].replace(0, np.nan)
        ret1 = close.pct_change()
        vol_ratio = volume / volume.rolling(20, min_periods=20).mean()
        move_cutoff = ret1.abs().rolling(126, min_periods=40).quantile(0.90).shift(1)
        vol_cutoff = vol_ratio.rolling(126, min_periods=40).quantile(0.90).shift(1)
        shock = ((ret1.abs() > move_cutoff) | (vol_ratio > vol_cutoff)).fillna(False)
        ret1_map[name] = ret1
        vol_ratio_map[name] = vol_ratio
        shock_map[name] = shock.astype(float)
        pos_map[name] = (shock & (ret1 > 0)).astype(float)
        neg_map[name] = (shock & (ret1 < 0)).astype(float)

    if not ret1_map:
        return out

    ret1_frame = pd.DataFrame(ret1_map)
    vol_frame = pd.DataFrame(vol_ratio_map)
    shock_frame = pd.DataFrame(shock_map)
    pos_frame = pd.DataFrame(pos_map)
    neg_frame = pd.DataFrame(neg_map)

    out["peer_event_shock_count"] = shock_frame.sum(axis=1)
    out["peer_positive_event_count"] = pos_frame.sum(axis=1)
    out["peer_negative_event_count"] = neg_frame.sum(axis=1)
    out["peer_event_net_score"] = out["peer_positive_event_count"] - out["peer_negative_event_count"]
    out["peer_event_net_score_3d"] = out["peer_event_net_score"].rolling(3, min_periods=1).sum()
    out["peer_max_abs_event_return_1d"] = ret1_frame.where(shock_frame.astype(bool)).abs().max(axis=1).fillna(0.0)
    out["peer_event_return_mean_1d"] = ret1_frame.where(shock_frame.astype(bool)).mean(axis=1).fillna(0.0)
    out["peer_volume_spike_count"] = (vol_frame > vol_frame.rolling(126, min_periods=40).quantile(0.90).shift(1)).sum(axis=1)
    out["peer_business_stress"] = (
        out["peer_negative_event_count"]
        + out["peer_max_abs_event_return_1d"].rolling(3, min_periods=1).max() * 10.0
        - out["peer_positive_event_count"] * 0.5
    )
    return out


def add_correlation_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "SP500_daily_return" in out.columns:
        sp = out["SP500_daily_return"]
        nv = out["daily_return"]
        out["correlation_sp500_20"] = nv.rolling(20, min_periods=20).corr(sp)
        cov = nv.rolling(60, min_periods=60).cov(sp)
        var = sp.rolling(60, min_periods=60).var()
        out["beta_60"] = cov / var.replace(0, np.nan)
    else:
        out["correlation_sp500_20"] = np.nan
        out["beta_60"] = np.nan
    return out


def build_model_frame(
    prices: pd.DataFrame,
    external: pd.DataFrame,
    start_date: str,
    end_date: str,
    ticker: str = "NVDA",
    include_fundamentals: bool = True,
    peer_ohlcv: dict[str, pd.DataFrame] | None = None,
    option_features: pd.DataFrame | None = None,
    news_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    technical = build_technical_features(prices)
    exog = build_external_features(external, technical.index)
    peer_market = build_peer_market_features(external, technical.index)
    frame = technical.join(exog, how="left").join(peer_market, how="left")
    if peer_ohlcv:
        frame = frame.join(build_peer_business_event_features(peer_ohlcv, technical.index), how="left")
    if "peer_mean_return_5d" in frame:
        frame["nvda_vs_peer_mean_5d"] = frame["5d_return"] - frame["peer_mean_return_5d"]
    if "peer_median_return_5d" in frame:
        frame["nvda_vs_peer_median_5d"] = frame["5d_return"] - frame["peer_median_return_5d"]
    if {"peer_event_net_score", "daily_return"}.issubset(frame.columns):
        frame["nvda_after_peer_positive_event"] = ((frame["peer_event_net_score"].shift(1) > 0) & (frame["daily_return"] > 0)).astype(float)
        frame["nvda_after_peer_negative_event"] = ((frame["peer_event_net_score"].shift(1) < 0) & (frame["daily_return"] < 0)).astype(float)
    if option_features is not None and not option_features.empty:
        aligned_options = option_features.sort_index().reindex(frame.index).ffill()
        option_numeric = aligned_options[[col for col in aligned_options.columns if pd.api.types.is_numeric_dtype(aligned_options[col])]]
        frame = frame.join(option_numeric.add_prefix("hist_"), how="left")
    if news_features is not None and not news_features.empty:
        aligned_news = news_features.sort_index().reindex(frame.index).ffill()
        news_numeric = aligned_news[[col for col in aligned_news.columns if pd.api.types.is_numeric_dtype(aligned_news[col])]]
        frame = frame.join(news_numeric, how="left")
    frame = frame.join(build_holiday_features(frame.index, frame["Close"]), how="left")
    frame = add_correlation_features(frame)

    if include_fundamentals:
        fundamentals = fetch_nvda_fundamentals(frame.index, frame["Close"], ticker)
        frame = frame.join(fundamentals, how="left")
    else:
        for col in ["price_to_revenue", "pe_ratio", "earnings_surprise", "revenue_growth_yoy"]:
            frame[col] = np.nan

    frame["target_return"] = frame["Close"].pct_change().shift(-1)
    frame["target_direction"] = np.where(frame["target_return"].isna(), np.nan, (frame["target_return"] > 0).astype(int))

    raw = {"Open", "High", "Low", "Close", "Volume", "target_return", "target_direction"}
    if not include_fundamentals:
        raw.update({"price_to_revenue", "pe_ratio", "earnings_surprise", "revenue_growth_yoy"})
    feature_columns = [col for col in frame.columns if col not in raw and pd.api.types.is_numeric_dtype(frame[col])]
    frame = clean_model_frame(frame, feature_columns, require_target=False)
    frame = frame.loc[(frame.index >= pd.Timestamp(start_date) - pd.Timedelta(days=370)) & (frame.index <= pd.Timestamp(end_date))]
    return frame, feature_columns
