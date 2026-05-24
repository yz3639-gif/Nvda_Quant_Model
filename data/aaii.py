"""AAII sentiment loader placeholder with honest failure semantics."""

from __future__ import annotations

import pandas as pd


def load_aaii_sentiment() -> tuple[pd.DataFrame, list[str]]:
    return pd.DataFrame(), ["AAII automatic scrape is not enabled in this build; use manual_sentiment.yaml."]
