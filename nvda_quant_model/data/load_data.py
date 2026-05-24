from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from nvda_quant_model.config import MACRO_TICKERS, PEER_TICKERS, PROJECT_ROOT, SECTOR_TICKERS


LOGGER = logging.getLogger(__name__)
LATEST_DATE_ALIASES = {"latest", "auto", "today", "now"}
AUTO_START_ALIASES = {"auto", "rolling", "rolling_24m", "latest-24m", "24m"}


def _safe_ticker_name(ticker: str) -> str:
    return (
        ticker.replace("^", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _cache_path(ticker: str, start: str, end: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_safe_ticker_name(ticker)}_{start}_{end}.csv"


def _download_one(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    auto_adjust: bool = True,
    retries: int = 2,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end, cache_dir)
    if path.exists() and not force_refresh:
        data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if not data.empty:
            return data.sort_index()

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(-1):
                    data = data.xs(ticker, axis=1, level=-1)
                elif ticker in data.columns.get_level_values(0):
                    data = data.xs(ticker, axis=1, level=0)
                else:
                    data.columns = data.columns.get_level_values(0)
            data = data.rename_axis("Date").sort_index()
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data = data.dropna(how="all")
            if not data.empty:
                data.to_csv(path)
                return data
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            LOGGER.warning("download failed for %s attempt %s: %s", ticker, attempt + 1, exc)

    if last_error:
        raise RuntimeError(f"Unable to download {ticker}: {last_error}") from last_error
    raise RuntimeError(f"Unable to download {ticker}: empty response")


def _end_exclusive(end_date: str) -> str:
    end = pd.Timestamp(end_date) + timedelta(days=1)
    return end.strftime("%Y-%m-%d")


def _is_latest_alias(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in LATEST_DATE_ALIASES


def _is_auto_start(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in AUTO_START_ALIASES


def fetch_recent_daily_prices(ticker: str, period: str = "45d") -> pd.DataFrame:
    """Fetch recent daily prices without cache, used to resolve live date bounds."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)
        else:
            data.columns = data.columns.get_level_values(0)
    data = data.rename_axis("Date").sort_index()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.dropna(how="all")


def resolve_data_window(
    start_date: str | None,
    end_date: str | None,
    ticker: str = "NVDA",
    lookback_months: int = 24,
) -> tuple[str, str, dict[str, object]]:
    """Resolve dynamic user-friendly date aliases into concrete market dates."""
    recent = pd.DataFrame()
    if _is_latest_alias(end_date) or _is_auto_start(start_date):
        recent = fetch_recent_daily_prices(ticker)
        if recent.empty:
            raise RuntimeError(f"Unable to resolve latest trading day for {ticker}")

    if _is_latest_alias(end_date):
        resolved_end = recent.index.max().normalize()
        end_source = "latest_yfinance_daily"
    else:
        resolved_end = pd.Timestamp(end_date).normalize()
        end_source = "user_supplied"

    if _is_auto_start(start_date):
        resolved_start = (resolved_end - pd.DateOffset(months=lookback_months)).normalize()
        start_source = f"rolling_{lookback_months}_months"
    else:
        resolved_start = pd.Timestamp(start_date).normalize()
        start_source = "user_supplied"

    if resolved_start >= resolved_end:
        raise ValueError(f"start_date must be before end_date: {resolved_start.date()} >= {resolved_end.date()}")

    latest_close = None
    if not recent.empty and resolved_end in recent.index and "Close" in recent.columns:
        latest_close = float(recent.loc[resolved_end, "Close"])

    metadata = {
        "requested_start": start_date,
        "requested_end": end_date,
        "resolved_start": resolved_start.strftime("%Y-%m-%d"),
        "resolved_end": resolved_end.strftime("%Y-%m-%d"),
        "start_source": start_source,
        "end_source": end_source,
        "latest_daily_close": latest_close,
        "lookback_months": lookback_months,
    }
    return resolved_start.strftime("%Y-%m-%d"), resolved_end.strftime("%Y-%m-%d"), metadata


def warmup_start(start_date: str, calendar_days: int = 560) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=calendar_days)).strftime("%Y-%m-%d")


def load_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load split-adjusted OHLCV data for the target ticker."""
    cache_dir = cache_dir or PROJECT_ROOT / "cache"
    data = _download_one(ticker, start_date, _end_exclusive(end_date), cache_dir, force_refresh=force_refresh)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"{ticker} missing OHLCV columns: {missing}")
    return data[required].copy()


def load_external_closes(
    start_date: str,
    end_date: str,
    tickers: Iterable[str] | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load macro and sector close levels used by weekly features."""
    cache_dir = cache_dir or PROJECT_ROOT / "cache"
    requested = list(tickers or {
        *MACRO_TICKERS.values(),
        *SECTOR_TICKERS.values(),
        *PEER_TICKERS.values(),
    })
    frames: list[pd.Series] = []
    for ticker in requested:
        if ticker == "USD_FALLBACK":
            continue
        try:
            raw = _download_one(ticker, start_date, _end_exclusive(end_date), cache_dir, force_refresh=force_refresh)
        except RuntimeError:
            if ticker == MACRO_TICKERS["USD"]:
                LOGGER.warning("USD index failed; falling back to UUP ETF")
                raw = _download_one(
                    MACRO_TICKERS["USD_FALLBACK"],
                    start_date,
                    _end_exclusive(end_date),
                    cache_dir,
                    force_refresh=force_refresh,
                )
                ticker = MACRO_TICKERS["USD_FALLBACK"]
            else:
                LOGGER.warning("Skipping unavailable external ticker %s", ticker)
                continue
        close_col = "Close" if "Close" in raw.columns else raw.columns[0]
        frames.append(raw[close_col].rename(ticker))

    if not frames:
        raise RuntimeError("No external market data downloaded")
    external = pd.concat(frames, axis=1).sort_index()
    return external.ffill()


def load_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    include_warmup: bool = True,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return target OHLCV plus external closes with enough lookback for features."""
    start = warmup_start(start_date) if include_warmup else start_date
    ohlcv = load_ohlcv(ticker, start, end_date, cache_dir, force_refresh=force_refresh)
    external = load_external_closes(start, end_date, cache_dir=cache_dir, force_refresh=force_refresh)
    return ohlcv, external


def load_peer_ohlcv_panel(
    start_date: str,
    end_date: str,
    tickers: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load peer OHLCV data for business-event and industry-realism features."""
    cache_dir = cache_dir or PROJECT_ROOT / "cache"
    peers = tickers or PEER_TICKERS
    panel: dict[str, pd.DataFrame] = {}
    for name, ticker in peers.items():
        try:
            raw = _download_one(ticker, start_date, _end_exclusive(end_date), cache_dir, force_refresh=force_refresh)
        except RuntimeError as exc:
            LOGGER.warning("Skipping peer ticker %s (%s): %s", name, ticker, exc)
            continue
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            LOGGER.warning("Skipping peer ticker %s (%s), missing %s", name, ticker, missing)
            continue
        panel[name] = raw[required].copy()
    return panel


def fetch_nvda_fundamentals(
    price_index: pd.DatetimeIndex,
    close: pd.Series,
    ticker: str = "NVDA",
) -> pd.DataFrame:
    """Best-effort historical fundamental features from yfinance.

    Yahoo does not provide a clean point-in-time fundamentals API. This function
    uses quarterly statement dates and shifts them forward one day before
    forward-filling, so the backtest never uses a quarter's values before the
    report date in the returned index. If the source is unavailable, neutral
    proxy columns are returned and flagged by missing values before validation.
    """
    out = pd.DataFrame(index=price_index)
    for col in ["price_to_revenue", "pe_ratio", "earnings_surprise", "revenue_growth_yoy"]:
        out[col] = pd.NA

    try:  # pragma: no cover - yfinance fundamentals are network/schema dependent
        stock = yf.Ticker(ticker)
        income = stock.quarterly_income_stmt
        if income is None or income.empty:
            return out
        q = income.T.sort_index()
        q.index = pd.to_datetime(q.index).tz_localize(None)

        revenue_col = next((c for c in ["Total Revenue", "Operating Revenue"] if c in q.columns), None)
        eps_col = next((c for c in ["Diluted EPS", "Basic EPS"] if c in q.columns), None)
        shares = None
        try:
            shares = stock.get_shares_full(
                start=price_index.min().strftime("%Y-%m-%d"),
                end=(price_index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if shares is not None and not shares.empty:
                shares.index = pd.to_datetime(shares.index).tz_localize(None)
                shares = shares.reindex(price_index).ffill()
        except Exception:
            shares = None
        if shares is None or shares.empty:
            info_shares = stock.info.get("sharesOutstanding")
            if info_shares:
                shares = pd.Series(float(info_shares), index=price_index)

        if revenue_col:
            revenue = q[revenue_col].astype(float)
            revenue_ttm = revenue.rolling(4, min_periods=2).sum()
            revenue_growth = revenue / revenue.shift(4) - 1.0
            daily_revenue_ttm = revenue_ttm.rename("revenue_ttm")
            daily_revenue_ttm.index = daily_revenue_ttm.index + pd.Timedelta(days=1)
            daily_revenue_ttm = daily_revenue_ttm.reindex(price_index).ffill()
            daily_growth = revenue_growth.rename("revenue_growth_yoy")
            daily_growth.index = daily_growth.index + pd.Timedelta(days=1)
            out["revenue_growth_yoy"] = daily_growth.reindex(price_index).ffill()
            if shares is not None and not shares.empty:
                revenue_per_share = daily_revenue_ttm / shares.reindex(price_index).ffill()
                out["price_to_revenue"] = close / revenue_per_share.replace(0, pd.NA)

        if eps_col:
            eps_ttm = q[eps_col].astype(float).rolling(4, min_periods=2).sum()
            eps_ttm.index = eps_ttm.index + pd.Timedelta(days=1)
            daily_eps = eps_ttm.reindex(price_index).ffill()
            out["pe_ratio"] = close / daily_eps.replace(0, pd.NA)

        try:
            earnings = stock.get_earnings_dates(limit=24)
            if earnings is not None and not earnings.empty and "Surprise(%)" in earnings.columns:
                surprise = earnings["Surprise(%)"].sort_index()
                surprise.index = pd.to_datetime(surprise.index).tz_localize(None) + pd.Timedelta(days=1)
                out["earnings_surprise"] = surprise.reindex(price_index).ffill().fillna(0.0)
        except Exception:
            out["earnings_surprise"] = 0.0
    except Exception as exc:
        LOGGER.warning("fundamentals unavailable: %s", exc)

    return out
