"""Scenario helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_scenarios(posteriors: pd.DataFrame, weights: pd.DataFrame) -> dict[str, float]:
    merged = weights.merge(posteriors, on="event_id", how="left")
    scenarios = {"bull_75": 0.0, "base_50": 0.0, "bear_25": 0.0}
    for _, row in merged.iterrows():
        mean = float(row.get("posterior_mean", np.nan))
        sd = float(row.get("posterior_sd", np.nan))
        weight = float(row.get("weight", 0.0))
        if not np.isfinite(mean):
            continue
        sd = sd if np.isfinite(sd) and sd > 0 else 0.0
        scenarios["bull_75"] += weight * (mean + 0.67449 * sd)
        scenarios["base_50"] += weight * mean
        scenarios["bear_25"] += weight * (mean - 0.67449 * sd)
    return scenarios
