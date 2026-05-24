"""Upcoming event calendar discovery.

The free-data implementation intentionally combines a small verified schedule
with manual YAML input. Paid calendars can be wired into this module later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(slots=True)
class CalendarEvent:
    id: str
    name: str
    date: pd.Timestamp
    category: str
    importance: int = 3
    end_date: pd.Timestamp | None = None
    secondary_categories: list[str] | None = None
    geography: str = ""
    source: str = ""
    notes: str = ""
    theme_tickers: list[str] | None = None

    @property
    def categories(self) -> list[str]:
        return [self.category] + list(self.secondary_categories or [])


def load_manual_events(path: str | Path) -> list[CalendarEvent]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payload = yaml.safe_load(file_path.read_text()) or {}
    return [_event_from_dict(item) for item in payload.get("events", [])]


def events_for_next_week(as_of_date: str | pd.Timestamp, manual_path: str | Path) -> list[CalendarEvent]:
    """Return events dated in the next Monday-Friday window after as_of_date."""

    as_of = pd.Timestamp(as_of_date).normalize()
    start = as_of + pd.Timedelta(days=1)
    while start.weekday() != 0:
        start += pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=6)
    events = load_manual_events(manual_path)
    return sorted(
        [event for event in events if start <= event.date.normalize() <= end],
        key=lambda event: (event.date, -event.importance, event.name),
    )


def _event_from_dict(item: dict[str, Any]) -> CalendarEvent:
    return CalendarEvent(
        id=str(item["id"]),
        name=str(item["name"]),
        date=pd.Timestamp(item["date"]),
        end_date=pd.Timestamp(item["end_date"]) if item.get("end_date") else None,
        category=str(item["category"]),
        secondary_categories=[str(x) for x in item.get("secondary_categories", [])],
        importance=int(item.get("importance", 3)),
        geography=str(item.get("geography", "")),
        source=str(item.get("source", "")),
        notes=str(item.get("notes", "")),
        theme_tickers=[str(x) for x in item.get("theme_tickers", [])],
    )
