"""Alternative data hooks returning NaN until credentials are configured."""

from __future__ import annotations

import pandas as pd


def load_alt_data_signals() -> tuple[pd.DataFrame, list[str]]:
    return pd.DataFrame(), ["Alternative data credentials are not configured; no synthetic values generated."]
