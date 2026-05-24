"""Macro data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.prices import load_fred_series


def load_macro_panel(
    fred_series: dict[str, str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    result = load_fred_series(fred_series, start, end, cache_dir)
    return result.prices, result.warnings, result.sources
