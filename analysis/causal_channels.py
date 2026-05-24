"""Causal-channel classification from observable proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def identify_channels(event_id: str, cross_asset: pd.DataFrame, sector_rotation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    channels = {
        "risk": ["VIX", "VVIX", "HY_CREDIT", "HYG", "JNK"],
        "rates": ["UST2Y", "UST10Y", "UST30Y", "TLT", "IEF", "SHY", "TIP"],
        "credit": ["HY_CREDIT", "IG_CREDIT", "HYG", "LQD"],
        "risk_appetite": ["RUT", "NDX", "EEM", "XLY", "XLP"],
        "safe_haven": ["DXY", "Gold", "USDJPY", "GLD"],
        "cyclicals": ["XLI", "XLB", "XLE", "CPER", "Oil"],
    }
    combined = pd.concat([cross_asset, sector_rotation], ignore_index=True, sort=False)
    for channel, keys in channels.items():
        subset = combined[
            combined.get("event_id", pd.Series(dtype=str)).astype(str).eq(event_id)
            & (
                combined.get("asset_name", combined.get("symbol", pd.Series(dtype=str))).astype(str).isin(keys)
                | combined.get("symbol", pd.Series(dtype=str)).astype(str).isin(keys)
            )
        ]
        score = float(pd.to_numeric(subset.get("mean", pd.Series(dtype=float)), errors="coerce").abs().mean()) if not subset.empty else np.nan
        rows.append(
            {
                "event_id": event_id,
                "channel": channel,
                "channel_score_abs_car": score,
                "evidence_count": int(len(subset)),
                "quality": "BLACK" if subset.empty else ("GREEN" if len(subset) >= 3 else "YELLOW"),
            }
        )
    return pd.DataFrame(rows)
