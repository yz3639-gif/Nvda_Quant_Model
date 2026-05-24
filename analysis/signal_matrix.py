"""Large long-format signal matrix builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


STAT_COLUMNS = [
    "n", "mean", "median", "win_rate", "std", "skew", "t_stat", "p_value",
    "wilcoxon_p", "patell_p", "bmp_p", "boot_ci_low", "boot_ci_high",
]


def build_signal_matrices(
    events: list,
    samples_by_event: dict[str, dict[str, pd.DataFrame]],
    prices: pd.DataFrame,
    asset_map: dict[str, str],
    windows: dict[str, list[int]],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).sort_index()
    matrix_boot = int(min(n_boot, 200))
    wide_rows = []
    long_rows = []
    for event in events:
        subsets = samples_by_event.get(event.id, {})
        for subset_name, samples in subsets.items():
            sample_positions = []
            if samples is not None and not samples.empty:
                for _, sample in samples.iterrows():
                    pos = _event_pos(returns.index, pd.Timestamp(sample["date"]))
                    if pos is not None:
                        sample_positions.append(pos)
            for asset_name, symbol in asset_map.items():
                if symbol not in returns:
                    continue
                for window_name, (w0, w1) in windows.items():
                    vals = []
                    for pos in sample_positions:
                        start, end = pos + int(w0), pos + int(w1)
                        if start < 0 or end >= len(returns):
                            continue
                        vals.append(float(returns[symbol].iloc[start : end + 1].sum()))
                    stats = _summarize_matrix_values(vals, n_boot=matrix_boot, rng=rng)
                    quality = quality_label(int(stats["n"]), float(stats.get("p_value", np.nan)))
                    row = {
                        "event_id": event.id,
                        "event_name": event.name,
                        "subset": subset_name,
                        "asset_name": asset_name,
                        "asset": symbol,
                        "window": window_name,
                        "quality": quality,
                        **{k: stats.get(k, np.nan) for k in STAT_COLUMNS},
                    }
                    wide_rows.append(row)
                    for stat in STAT_COLUMNS:
                        long_rows.append(
                            {
                                "event_id": event.id,
                                "event_name": event.name,
                                "subset": subset_name,
                                "asset_name": asset_name,
                                "asset": symbol,
                                "window": window_name,
                                "statistic": stat,
                                "value": stats.get(stat, np.nan),
                                "quality": quality,
                                "source": "yfinance/FRED price event window",
                            }
                        )
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def quality_label(n: int, p_value: float) -> str:
    if n <= 0:
        return "BLACK"
    if n < 10:
        return "RED"
    if n < 30:
        return "YELLOW"
    if np.isfinite(p_value) and p_value < 0.05:
        return "GREEN"
    return "YELLOW"


def _summarize_matrix_values(values: list[float], n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    out = {col: np.nan for col in STAT_COLUMNS}
    out["n"] = int(arr.size)
    if arr.size == 0:
        return out
    out["mean"] = float(np.mean(arr))
    out["median"] = float(np.median(arr))
    out["win_rate"] = float(np.mean(arr > 0))
    out["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    out["skew"] = float(stats.skew(arr, bias=False)) if arr.size > 2 else np.nan
    if arr.size > 1 and out["std"] > 0:
        t_stat = float(np.mean(arr) / (out["std"] / np.sqrt(arr.size)))
        out["t_stat"] = t_stat
        out["p_value"] = float(2 * (1 - stats.t.cdf(abs(t_stat), df=arr.size - 1)))
        boot = int(n_boot)
        if boot > 0:
            draws = rng.choice(arr, size=(boot, arr.size), replace=True).mean(axis=1)
            out["boot_ci_low"] = float(np.quantile(draws, 0.025))
            out["boot_ci_high"] = float(np.quantile(draws, 0.975))
    else:
        out["boot_ci_low"] = float(arr[0])
        out["boot_ci_high"] = float(arr[0])
    return out


def _event_pos(index: pd.Index, event_date: pd.Timestamp) -> int | None:
    idx = pd.DatetimeIndex(index)
    locs = np.where(idx >= event_date.normalize())[0]
    return int(locs[0]) if len(locs) else None
