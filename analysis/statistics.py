"""Statistical summaries and tests."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def summarize_cars(values: Iterable[float], standardized: Iterable[float] | None = None, n_boot: int = 10000, rng: np.random.Generator | None = None) -> dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    out = {
        "n": int(arr.size),
        "mean": np.nan,
        "median": np.nan,
        "win_rate": np.nan,
        "std": np.nan,
        "skew": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "wilcoxon_p": np.nan,
        "patell_z": np.nan,
        "patell_p": np.nan,
        "bmp_t": np.nan,
        "bmp_p": np.nan,
        "boot_ci_low": np.nan,
        "boot_ci_high": np.nan,
        "economic_significant": False,
    }
    if arr.size == 0:
        return out
    out.update(
        {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "win_rate": float(np.mean(arr > 0)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "skew": float(stats.skew(arr, bias=False)) if arr.size > 2 else np.nan,
            "economic_significant": bool(abs(np.mean(arr)) >= 0.005),
        }
    )
    if arr.size > 1 and np.std(arr, ddof=1) > 0:
        t_stat, p_value = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
        out["t_stat"] = float(t_stat)
        out["p_value"] = float(p_value)
    if arr.size > 1 and not np.allclose(arr, 0):
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(arr, zero_method="wilcox").pvalue)
        except ValueError:
            out["wilcoxon_p"] = np.nan
    if standardized is not None:
        std_arr = np.asarray([x for x in standardized if np.isfinite(x)], dtype=float)
        if std_arr.size > 0:
            z = float(np.sum(std_arr) / np.sqrt(std_arr.size))
            out["patell_z"] = z
            out["patell_p"] = float(2 * (1 - stats.norm.cdf(abs(z))))
        if std_arr.size > 1 and np.std(std_arr, ddof=1) > 0:
            bmp = float(np.mean(std_arr) / (np.std(std_arr, ddof=1) / np.sqrt(std_arr.size)))
            out["bmp_t"] = bmp
            out["bmp_p"] = float(2 * (1 - stats.t.cdf(abs(bmp), df=std_arr.size - 1)))
    if arr.size > 1:
        rng = rng or np.random.default_rng(42)
        boot_count = int(n_boot)
        if boot_count * arr.size <= 2_000_000:
            draws = rng.choice(arr, size=(boot_count, arr.size), replace=True).mean(axis=1)
        else:
            chunks = []
            remaining = boot_count
            while remaining > 0:
                chunk = min(1000, remaining)
                chunks.append(rng.choice(arr, size=(chunk, arr.size), replace=True).mean(axis=1))
                remaining -= chunk
            draws = np.concatenate(chunks)
        out["boot_ci_low"] = float(np.quantile(draws, 0.025))
        out["boot_ci_high"] = float(np.quantile(draws, 0.975))
    else:
        out["boot_ci_low"] = float(arr[0])
        out["boot_ci_high"] = float(arr[0])
    return out


def holm_adjust(frame: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    """Add Holm-Bonferroni adjusted p-values to a summary frame."""

    if frame.empty or p_col not in frame:
        frame[f"{p_col}_holm"] = np.nan
        return frame
    pvals = pd.to_numeric(frame[p_col], errors="coerce")
    valid = pvals.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=frame.index, dtype=float)
    m = len(valid)
    running = 0.0
    for rank, (idx, pval) in enumerate(valid.items(), start=1):
        val = min(1.0, (m - rank + 1) * float(pval))
        running = max(running, val)
        adjusted.loc[idx] = running
    frame[f"{p_col}_holm"] = adjusted
    return frame


def normal_kl(mu0: float, sd0: float, mu1: float, sd1: float) -> float:
    if not all(np.isfinite([mu0, sd0, mu1, sd1])) or sd0 <= 0 or sd1 <= 0:
        return float("nan")
    return float(np.log(sd1 / sd0) + (sd0 * sd0 + (mu0 - mu1) ** 2) / (2 * sd1 * sd1) - 0.5)


def pct(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"
