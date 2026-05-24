"""FRED/ALFRED release calendar helpers.

Free ALFRED vintages can provide vintages but not a clean actual-vs-consensus
surprise history for CPI/PPI/retail sales without additional mapping. v2 refuses
to generate proxy events from this module.
"""

from __future__ import annotations

import pandas as pd


def load_true_macro_release_history(series_id: str) -> tuple[pd.DataFrame, list[str]]:
    return pd.DataFrame(), [f"No verified release-date + consensus surprise history loaded for {series_id}; macro event must be downgraded."]
