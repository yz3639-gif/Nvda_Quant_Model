"""Cross-asset event transmission summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.statistics import summarize_cars


def summarize_cross_assets(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    assets: dict[str, str],
    benchmark: str,
    window: tuple[int, int] = (1, 5),
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).sort_index()
    rows: list[dict[str, Any]] = []
    if samples.empty or benchmark not in returns:
        return pd.DataFrame()
    for name, symbol in assets.items():
        if symbol not in returns:
            continue
        vals = []
        corr_deltas = []
        for _, sample in samples.iterrows():
            pos = _event_pos(returns.index, pd.Timestamp(sample["date"]))
            if pos is None:
                continue
            start, end = pos + window[0], pos + window[1]
            if start < 0 or end >= len(returns):
                continue
            vals.append(float(returns[symbol].iloc[start : end + 1].sum()))
            pre = returns[[symbol, benchmark]].iloc[max(0, pos - 20) : pos].dropna()
            post = returns[[symbol, benchmark]].iloc[pos + 1 : min(len(returns), pos + 21)].dropna()
            if pre.shape[0] >= 10 and post.shape[0] >= 10:
                corr_deltas.append(float(post.corr().iloc[0, 1] - pre.corr().iloc[0, 1]))
        stats = summarize_cars(vals, n_boot=n_boot, rng=rng)
        stats.update(
            {
                "asset_name": name,
                "symbol": symbol,
                "window": f"{window[0]}_{window[1]}",
                "corr_delta_mean": float(np.nanmean(corr_deltas)) if corr_deltas else np.nan,
            }
        )
        rows.append(stats)
    return pd.DataFrame(rows).sort_values("mean", ascending=False) if rows else pd.DataFrame()


def _event_pos(index: pd.Index, event_date: pd.Timestamp) -> int | None:
    idx = pd.DatetimeIndex(index)
    locs = np.where(idx >= event_date.normalize())[0]
    return int(locs[0]) if len(locs) else None
