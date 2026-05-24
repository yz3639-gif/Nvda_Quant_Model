"""Market price and macro time-series loading with local caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import re

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PriceLoadResult:
    prices: pd.DataFrame
    warnings: list[str]
    sources: dict[str, str]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_price_panel(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path,
    stale_ok: bool = True,
) -> PriceLoadResult:
    """Load adjusted closes for a list of yfinance tickers."""

    cache = ensure_dir(cache_dir)
    unique = sorted({ticker for ticker in tickers if ticker})
    frames: dict[str, pd.Series] = {}
    warnings: list[str] = []
    sources: dict[str, str] = {}
    for ticker in unique:
        cache_path = cache / f"price_{_safe(ticker)}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.csv"
        if cache_path.exists():
            series = _read_cached_series(cache_path)
            frames[ticker] = series
            sources[ticker] = f"cache:{cache_path}"
            continue
        try:
            series = _download_yfinance_close(ticker, start, end)
            if series.dropna().empty:
                raise ValueError("empty adjusted close")
            series.to_frame("adj_close").to_csv(cache_path, index_label="date")
            frames[ticker] = series
            sources[ticker] = "yfinance adjusted close"
        except Exception as exc:  # pragma: no cover - depends on network state
            latest = _latest_cached_series(cache, ticker)
            if latest is not None and stale_ok:
                frames[ticker] = latest[1]
                sources[ticker] = f"stale-cache:{latest[0]}"
                warnings.append(f"{ticker}: live download failed ({exc}); used stale cache {latest[0].name}.")
            else:
                warnings.append(f"{ticker}: price download failed ({exc}); series omitted.")
    if not frames:
        raise RuntimeError("No price series could be loaded.")
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.loc[~panel.index.duplicated(keep="last")]
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return PriceLoadResult(prices=panel, warnings=warnings, sources=sources)


def load_fred_series(
    series_ids: dict[str, str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path,
) -> PriceLoadResult:
    """Load FRED time series by direct CSV endpoint."""

    cache = ensure_dir(cache_dir)
    frames: dict[str, pd.Series] = {}
    warnings: list[str] = []
    sources: dict[str, str] = {}
    for name, sid in series_ids.items():
        cache_path = cache / f"fred_{_safe(sid)}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.csv"
        try:
            if cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["date"])
                series = df.set_index("date")["value"].astype(float)
                sources[name] = f"cache:{cache_path}"
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                raw = pd.read_csv(url)
                date_col = "observation_date"
                raw[date_col] = pd.to_datetime(raw[date_col])
                raw[sid] = pd.to_numeric(raw[sid].replace(".", np.nan), errors="coerce")
                raw = raw.rename(columns={date_col: "date", sid: "value"})
                raw[["date", "value"]].to_csv(cache_path, index=False)
                series = raw.set_index("date")["value"].astype(float)
                sources[name] = f"FRED:{sid}"
            series = series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
            frames[name] = series
        except Exception as exc:  # pragma: no cover
            warnings.append(f"{name}/{sid}: FRED load failed ({exc}).")
    if not frames:
        return PriceLoadResult(pd.DataFrame(), warnings, sources)
    panel = pd.concat(frames, axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return PriceLoadResult(panel, warnings, sources)


def pct_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.astype(float).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def simple_window_return(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compounded return between two dates using available observations."""

    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return float("nan")
    subset = clean.loc[(clean.index >= start) & (clean.index <= end)]
    if subset.shape[0] < 2:
        return float("nan")
    return float(subset.iloc[-1] / subset.iloc[0] - 1.0)


def prior_business_day(date: pd.Timestamp, index: pd.Index) -> pd.Timestamp | None:
    eligible = pd.DatetimeIndex(index)[pd.DatetimeIndex(index) <= date]
    if len(eligible) == 0:
        return None
    return pd.Timestamp(eligible[-1])


def next_business_day(date: pd.Timestamp, index: pd.Index) -> pd.Timestamp | None:
    eligible = pd.DatetimeIndex(index)[pd.DatetimeIndex(index) >= date]
    if len(eligible) == 0:
        return None
    return pd.Timestamp(eligible[0])


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (nth - 1))


def nearest_weekday_to_day(year: int, month: int, day: int) -> pd.Timestamp:
    last_day = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).day
    candidate = pd.Timestamp(year=year, month=month, day=min(day, last_day))
    if candidate.weekday() == 5:
        candidate -= pd.Timedelta(days=1)
    elif candidate.weekday() == 6:
        candidate += pd.Timedelta(days=1)
    return candidate


def _download_yfinance_close(ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=pd.Timestamp(start).date().isoformat(),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _extract_adjusted_close(df, ticker)


def _extract_adjusted_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        for key in [("Adj Close", ticker), ("Close", ticker)]:
            if key in df.columns:
                out = df[key]
                break
        else:
            level0 = df.columns.get_level_values(0)
            if "Adj Close" in level0:
                out = df.xs("Adj Close", axis=1, level=0).iloc[:, 0]
            elif "Close" in level0:
                out = df.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                out = df.iloc[:, 0]
    elif "Adj Close" in df:
        out = df["Adj Close"]
    elif "Close" in df:
        out = df["Close"]
    else:
        out = df.iloc[:, 0]
    out = pd.to_numeric(out, errors="coerce")
    out.name = ticker
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna().sort_index()


def _read_cached_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    value_col = "adj_close" if "adj_close" in df.columns else df.columns[-1]
    series = df.set_index("date")[value_col].astype(float)
    series.name = re.sub(r"^price_|_\\d{4}-\\d{2}-\\d{2}.*$", "", path.stem)
    return series.sort_index()


def _latest_cached_series(cache: Path, ticker: str) -> tuple[Path, pd.Series] | None:
    candidates = sorted(cache.glob(f"price_{_safe(ticker)}_*.csv"), reverse=True)
    for path in candidates:
        try:
            series = _read_cached_series(path)
            series.name = ticker
            return path, series
        except Exception:
            continue
    return None


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.replace("^", "idx_"))
