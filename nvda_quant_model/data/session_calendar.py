"""Exchange-session timing, separating daily date labels from UTC instants.

The pinned exchange_calendars XNYS calendar supplies holidays, early closes, and
DST. An explicit schedule may be injected. There is no weekday-only fallback.
Daily close availability is a modeling assumption, not a vendor delivery SLA.
"""
from __future__ import annotations
from functools import lru_cache
import pandas as pd

CALENDAR_VERSION = "exchange_calendars-4.11.2-XNYS"


@lru_cache(maxsize=8)
def _calendar(start_year: int, end_year: int):
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ImportError("Install exchange-calendars==4.11.2 for verified exchange-session timing") from exc
    return xcals.get_calendar("XNYS", start=f"{start_year}-01-01", end=f"{end_year}-12-31")


def session_dates(start, end) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    dates = _calendar(start.year, end.year).sessions.tz_localize(None)
    return dates[(dates >= start) & (dates <= end)]


def as_utc(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def session_labels(index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    if not dates.equals(dates.normalize()):
        raise ValueError("Daily price index must contain midnight session labels")
    return dates.normalize()


def session_schedule(index, schedule: pd.DataFrame | None = None, *, allow_non_sessions: bool = False) -> pd.DataFrame:
    """UTC opens/closes. Non-session rows fail unless explicitly allowed as NaT."""
    dates = session_labels(index)
    if len(dates) == 0:
        return pd.DataFrame(index=dates, columns=["market_open", "market_close"])
    if schedule is None:
        source = _calendar(dates.min().year, dates.max().year).schedule.rename(columns={"open": "market_open", "close": "market_close"})
        source.index = source.index.tz_localize(None)
    else:
        source = schedule
    result = source.reindex(dates)[["market_open", "market_close"]].copy()
    for col in result:
        result[col] = pd.to_datetime(result[col], utc=True)
    if not allow_non_sessions and result.isna().any().any():
        raise ValueError("Calendar missing a requested session (holiday, weekend, or unavailable schedule)")
    return result


def attach_daily_time_contract(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    result = frame.copy()
    timing = session_schedule(result.index, allow_non_sessions=True)
    result["available_at"] = timing["market_close"].to_numpy()
    result["decision_at"] = result["available_at"]
    result["label_end_at"] = result["available_at"].shift(-horizon)
    result.attrs["time_contract"] = {"calendar": CALENDAR_VERSION, "daily_available_at": "session_close_assumption", "price_adjustment": "caller_supplied_consistent_OHLC"}
    return result
