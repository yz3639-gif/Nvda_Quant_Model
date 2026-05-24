"""CFTC COT loader skeleton.

The official files are public, but parsing contract mappings robustly is a
separate data-engineering task. This module exposes an honest empty result until
that parser is completed.
"""

from __future__ import annotations

import pandas as pd


def load_cftc_cot(year: int) -> tuple[pd.DataFrame, list[str]]:
    return pd.DataFrame(), [f"CFTC COT parser for {year} is not configured; use manual_sentiment.yaml."]
