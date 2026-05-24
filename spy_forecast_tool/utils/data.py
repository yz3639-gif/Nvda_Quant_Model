"""Data access, caching, and market-calendar helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RiskFreeRateResult:
    """Risk-free rate value plus provenance and warnings."""

    rate: float
    source: str
    warnings: list[str]


@dataclass(slots=True)
class OptionChainData:
    """Selected option chain data for one expiration."""

    calls: pd.DataFrame
    puts: pd.DataFrame
    expiration: pd.Timestamp
    source: str
    warnings: list[str]


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def next_business_day(start: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the next weekday business day after the supplied date."""

    base = (start or pd.Timestamp.today()).normalize()
    candidate = base + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate


def read_price_history(
    ticker: str,
    lookback_years: float,
    cache_dir: Path,
    end_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Read adjusted close history from cache or yfinance."""

    ensure_directory(cache_dir)
    end = (end_date or pd.Timestamp.today()).normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(math.ceil(lookback_years * 365.25)) + 10)
    cache_path = cache_dir / _cache_name("prices", ticker, start.date().isoformat(), end.date().isoformat(), "csv")
    if cache_path.exists():
        return _read_cached_prices(cache_path)

    yf = _import_yfinance()
    data = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    adjusted_close = _extract_adjusted_close(data, ticker)
    if adjusted_close.empty:
        raise ValueError(f"No adjusted close data returned for {ticker}.")
    adjusted_close = adjusted_close.dropna().sort_index()
    if adjusted_close.empty:
        raise ValueError(f"Adjusted close data for {ticker} is empty after cleaning.")
    adjusted_close.to_frame("adj_close").to_csv(cache_path, index_label="date")
    return adjusted_close


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from adjusted close prices."""

    clean_prices = pd.to_numeric(prices, errors="coerce").dropna()
    if clean_prices.shape[0] < 3:
        raise ValueError("At least three price observations are required.")
    returns = np.log(clean_prices / clean_prices.shift(1)).dropna()
    returns.name = "log_return"
    return returns


def read_risk_free_rate(cache_dir: Path, override: float | None = None) -> RiskFreeRateResult:
    """Read current 13-week Treasury bill proxy from yfinance or use override."""

    if override is not None:
        return RiskFreeRateResult(rate=float(override), source="CLI override", warnings=[])
    ensure_directory(cache_dir)
    today = pd.Timestamp.today().date().isoformat()
    cache_path = cache_dir / _cache_name("risk_free", "^IRX", today, "csv")
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        return RiskFreeRateResult(
            rate=float(frame.loc[0, "rate"]),
            source=str(frame.loc[0, "source"]),
            warnings=[],
        )
    warnings: list[str] = []
    try:
        yf = _import_yfinance()
        data = yf.download("^IRX", period="10d", auto_adjust=False, progress=False, threads=False)
        close = _extract_close(data, "^IRX").dropna()
        if close.empty:
            raise ValueError("No ^IRX close values returned.")
        rate = float(close.iloc[-1]) / 100.0
        source = "^IRX 13-week Treasury bill yield via yfinance"
    except Exception as exc:  # pragma: no cover - exercised only on network/data failure
        rate = 0.04
        source = "fallback constant"
        warnings.append(f"Could not fetch ^IRX risk-free rate ({exc}); using 4.00% fallback.")
    pd.DataFrame([{"rate": rate, "source": source}]).to_csv(cache_path, index=False)
    return RiskFreeRateResult(rate=rate, source=source, warnings=warnings)


def read_option_chain(
    ticker: str,
    horizon_days: int,
    cache_dir: Path,
    source: str = "yfinance",
    as_of_date: pd.Timestamp | None = None,
) -> OptionChainData:
    """Read an option chain closest to the forecast horizon from cache or yfinance."""

    if source != "yfinance":
        raise ValueError("Only yfinance is implemented for --options-source in this build.")
    ensure_directory(cache_dir)
    as_of = (as_of_date or pd.Timestamp.today()).normalize()
    target_calendar_days = max(1, int(round(horizon_days * 365.0 / 252.0)))
    target_expiration = as_of + pd.Timedelta(days=target_calendar_days)
    try:
        yf = _import_yfinance()
        ticker_object = yf.Ticker(ticker)
        expirations = list(ticker_object.options)
        if not expirations:
            raise ValueError(f"No option expirations returned for {ticker}.")
        expiration = min(
            (pd.Timestamp(item) for item in expirations),
            key=lambda item: abs((item - target_expiration).days),
        )
        cache_path = cache_dir / _cache_name(
            "option_chain",
            ticker,
            expiration.date().isoformat(),
            as_of.date().isoformat(),
            "pkl",
        )
        if cache_path.exists():
            cached = pd.read_pickle(cache_path)
            return OptionChainData(
                calls=cached["calls"],
                puts=cached["puts"],
                expiration=expiration,
                source=source,
                warnings=["Loaded option chain from local cache."],
            )
        chain = ticker_object.option_chain(expiration.date().isoformat())
        payload = {"calls": chain.calls.copy(), "puts": chain.puts.copy()}
        pd.to_pickle(payload, cache_path)
        return OptionChainData(
            calls=payload["calls"],
            puts=payload["puts"],
            expiration=expiration,
            source=source,
            warnings=[],
        )
    except Exception as exc:
        cached = _read_latest_option_cache(cache_dir, ticker)
        if cached is not None:
            cached.warnings.append(f"Live option chain fetch failed ({exc}); used latest cached chain.")
            return cached
        raise ValueError(f"Could not read option chain for {ticker}: {exc}") from exc


def _read_cached_prices(cache_path: Path) -> pd.Series:
    """Read cached adjusted close data from CSV."""

    frame = pd.read_csv(cache_path, parse_dates=["date"])
    if "adj_close" not in frame:
        raise ValueError(f"Cached price file is missing adj_close: {cache_path}")
    series = frame.set_index("date")["adj_close"].astype(float)
    series.name = "adj_close"
    return series.sort_index()


def _read_latest_option_cache(cache_dir: Path, ticker: str) -> OptionChainData | None:
    """Read the newest cached option chain for a ticker when live fetch fails."""

    candidates = sorted(cache_dir.glob(f"option_chain_{_safe_key(ticker)}_*.pkl"), reverse=True)
    for path in candidates:
        try:
            cached: dict[str, Any] = pd.read_pickle(path)
            parts = path.stem.split("_")
            expiration = pd.Timestamp(parts[-2])
            return OptionChainData(
                calls=cached["calls"],
                puts=cached["puts"],
                expiration=expiration,
                source="yfinance-cache",
                warnings=["Loaded latest available option chain cache."],
            )
        except Exception:
            continue
    return None


def _extract_adjusted_close(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract adjusted close from yfinance output with single or multi-index columns."""

    if data.empty:
        return pd.Series(dtype=float, name="adj_close")
    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in data.columns:
            series = data[("Adj Close", ticker)]
        elif ("Close", ticker) in data.columns:
            series = data[("Close", ticker)]
        else:
            first_column = data.columns[0]
            series = data[first_column]
    elif "Adj Close" in data:
        series = data["Adj Close"]
    elif "Close" in data:
        series = data["Close"]
    else:
        series = data.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    series.name = "adj_close"
    return series


def _extract_close(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract close values from yfinance output with single or multi-index columns."""

    if data.empty:
        return pd.Series(dtype=float, name="close")
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            series = data[("Close", ticker)]
        else:
            series = data.iloc[:, 0]
    elif "Close" in data:
        series = data["Close"]
    else:
        series = data.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    series.name = "close"
    return series


def _import_yfinance() -> Any:
    """Import yfinance lazily with a helpful error if it is unavailable."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. Install dependencies with: pip install -r requirements.txt") from exc
    return yf


def _cache_name(*parts: str) -> str:
    """Build a safe cache filename from parts."""

    extension = parts[-1]
    stem_parts = parts[:-1]
    return "_".join(_safe_key(part) for part in stem_parts) + f".{extension}"


def _safe_key(value: str) -> str:
    """Return a filesystem-safe cache key segment."""

    return "".join(char if char.isalnum() or char in {"-", "."} else "_" for char in str(value))
