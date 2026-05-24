"""Sector and factor rotation analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.statistics import summarize_cars


def summarize_assets_for_window(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    assets: dict[str, str],
    benchmark: str,
    window: tuple[int, int] = (1, 5),
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
    group_label: str = "sector",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if samples.empty or benchmark not in prices:
        return pd.DataFrame()
    returns = prices.pct_change(fill_method=None).sort_index()
    for symbol, name in assets.items():
        if symbol not in returns:
            continue
        vals = []
        excess = []
        for _, sample in samples.iterrows():
            pos = _event_pos(returns.index, pd.Timestamp(sample["date"]))
            if pos is None:
                continue
            start, end = pos + window[0], pos + window[1]
            if start < 0 or end >= len(returns):
                continue
            asset_ret = float(returns[symbol].iloc[start : end + 1].sum())
            bench_ret = float(returns[benchmark].iloc[start : end + 1].sum())
            if np.isfinite(asset_ret):
                vals.append(asset_ret)
                excess.append(asset_ret - bench_ret if np.isfinite(bench_ret) else np.nan)
        stats = summarize_cars(vals, n_boot=n_boot, rng=rng)
        ex_stats = summarize_cars(excess, n_boot=n_boot, rng=rng)
        stats.update(
            {
                "group": group_label,
                "symbol": symbol,
                "name": name,
                "window": f"{window[0]}_{window[1]}",
                "excess_mean_vs_benchmark": ex_stats["mean"],
                "excess_median_vs_benchmark": ex_stats["median"],
            }
        )
        rows.append(stats)
    return pd.DataFrame(rows).sort_values("mean", ascending=False) if rows else pd.DataFrame()


def summarize_factor_spreads(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    factors: dict[str, list[str]],
    window: tuple[int, int] = (1, 5),
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    rows = []
    returns = prices.pct_change(fill_method=None).sort_index()
    for factor_name, pair in factors.items():
        if len(pair) != 2 or pair[0] not in returns or pair[1] not in returns:
            continue
        vals = []
        for _, sample in samples.iterrows():
            pos = _event_pos(returns.index, pd.Timestamp(sample["date"]))
            if pos is None:
                continue
            start, end = pos + window[0], pos + window[1]
            if start < 0 or end >= len(returns):
                continue
            vals.append(float(returns[pair[0]].iloc[start : end + 1].sum() - returns[pair[1]].iloc[start : end + 1].sum()))
        stats = summarize_cars(vals, n_boot=n_boot, rng=rng)
        stats.update({"factor": factor_name, "long": pair[0], "short": pair[1], "window": f"{window[0]}_{window[1]}"})
        rows.append(stats)
    return pd.DataFrame(rows).sort_values("mean", ascending=False) if rows else pd.DataFrame()


def _event_pos(index: pd.Index, event_date: pd.Timestamp) -> int | None:
    idx = pd.DatetimeIndex(index)
    locs = np.where(idx >= event_date.normalize())[0]
    return int(locs[0]) if len(locs) else None
