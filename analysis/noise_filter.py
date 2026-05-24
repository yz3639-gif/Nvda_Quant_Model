"""Noise-sample exclusion rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def annotate_noise(samples: pd.DataFrame, prices: pd.DataFrame, noise_path: str | Path) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    payload = yaml.safe_load(Path(noise_path).read_text()) if Path(noise_path).exists() else {}
    periods = payload.get("periods", []) if payload else []
    dynamic = payload.get("dynamic_rules", {}) if payload else {}
    out = samples.copy()
    excluded = []
    reasons = []
    for _, row in out.iterrows():
        date = pd.Timestamp(row["date"])
        window_start = date - pd.Timedelta(days=10)
        window_end = date + pd.Timedelta(days=10)
        reason_list: list[str] = []
        for period in periods:
            start = pd.Timestamp(period["start"])
            end = pd.Timestamp(period["end"])
            if start <= window_end and end >= window_start:
                reason_list.append(str(period["name"]))
        vix_threshold = dynamic.get("vix_above")
        if vix_threshold is not None and "^VIX" in prices:
            vix = prices["^VIX"].loc[(prices.index >= window_start) & (prices.index <= window_end)].dropna()
            if not vix.empty and float(vix.max()) > float(vix_threshold):
                reason_list.append(f"VIX>{vix_threshold}")
        dd_threshold = dynamic.get("weekly_spx_drawdown_less_than")
        if dd_threshold is not None and "SPY" in prices:
            spy = prices["SPY"].loc[(prices.index >= window_start) & (prices.index <= window_end)].dropna()
            if spy.shape[0] >= 5:
                roll = spy.pct_change(5, fill_method=None)
                if np.isfinite(roll.min()) and float(roll.min()) < float(dd_threshold):
                    reason_list.append(f"5d SPY return < {float(dd_threshold):.1%}")
        excluded.append(bool(reason_list))
        reasons.append("; ".join(dict.fromkeys(reason_list)))
    out["excluded_noise"] = excluded
    out["noise_reason"] = reasons
    return out
