"""Historical event sample construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data.calendar import CalendarEvent

def load_historical_events(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    payload = yaml.safe_load(file_path.read_text()) if file_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for item in (payload or {}).get("events", []):
        row = dict(item)
        row["date"] = pd.Timestamp(row["date"])
        row.setdefault("secondary_categories", [])
        rows.append(row)
    return pd.DataFrame(rows)


def samples_for_event(
    event: CalendarEvent,
    historical_events: pd.DataFrame,
    config: dict[str, Any],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build matching samples using only explicit, traceable event dates.

    v2 deliberately refuses recurring-calendar proxy dates for macro releases.
    If no true release + surprise history has been loaded, macro events receive
    no historical samples and are downgraded by the orchestration layer.
    """

    categories = set(event.categories)
    rows: list[dict[str, Any]] = []
    if not historical_events.empty:
        for _, row in historical_events.iterrows():
            row_categories = {row.get("category")}
            row_categories.update(row.get("secondary_categories") or [])
            if categories.intersection(row_categories):
                rows.append(
                    {
                        "sample_id": row.get("id", f"{row.get('category')}_{row.get('date')}"),
                        "event_name": row.get("name", ""),
                        "date": pd.Timestamp(row["date"]),
                        "category": row.get("category", ""),
                        "match_type": "explicit",
                        "comparability": 1.0 if row.get("category") == event.category else 0.75,
                        "notes": row.get("notes", ""),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["sample_id", "date"]).sort_values("date")
    frame = frame[frame["date"] < as_of_date]
    start = pd.Timestamp(config.get("historical_sample_start", "1900-01-01"))
    frame = frame[frame["date"] >= start].reset_index(drop=True)
    return frame
