"""CBOE put/call and volatility data helpers."""

from __future__ import annotations

import pandas as pd


def load_put_call_ratios() -> tuple[pd.DataFrame, list[str]]:
    return pd.DataFrame(), ["CBOE put/call automatic CSV endpoint unavailable or not configured; use manual_sentiment.yaml."]
