from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def _rewrite_with_schema(path: Path, fieldnames: list[str]) -> None:
    old_header = _read_header(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open(newline="", encoding="utf-8") as source, temp_path.open("w", newline="", encoding="utf-8") as target:
        reader = csv.reader(source)
        next(reader, None)
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            if len(row) == len(fieldnames):
                writer.writerow(dict(zip(fieldnames, row)))
            elif len(row) == len(old_header):
                writer.writerow(dict(zip(old_header, row)))
    temp_path.replace(path)


def append_rows_schema_safe(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = csv_path.exists()
    if exists and _read_header(csv_path) != fieldnames:
        _rewrite_with_schema(csv_path, fieldnames)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_live_optimizer_csv(path: str | Path, attempts: int = 5, delay_seconds: float = 0.25) -> pd.DataFrame:
    """Read an optimizer CSV that may be actively appended by a long-running job."""

    csv_path = Path(path)
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            last_error = exc
            time.sleep(delay_seconds)

    try:
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except Exception:
        if last_error is not None:
            raise last_error
        raise
