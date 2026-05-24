"""Historical sample matching helpers."""

from __future__ import annotations

import pandas as pd


def sample_counts(samples: pd.DataFrame) -> dict[str, int]:
    if samples.empty:
        return {"raw": 0, "clean": 0, "environment": 0}
    return {
        "raw": int(len(samples)),
        "clean": int((~samples.get("excluded_noise", False)).sum()) if "excluded_noise" in samples else int(len(samples)),
        "environment": int(samples.get("environment_match", pd.Series(False, index=samples.index)).sum()),
    }
