"""Gaussian copula simulation for event posterior aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def gaussian_copula_draws(
    posteriors: pd.DataFrame,
    weights: pd.DataFrame,
    scheme: str,
    n_sims: int,
    seed: int,
    corr: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    merged = weights.merge(posteriors, on="event_id", how="left")
    merged = merged[pd.to_numeric(merged["posterior_mean"], errors="coerce").notna()].reset_index(drop=True)
    if merged.empty:
        return np.array([], dtype=float), merged
    n = len(merged)
    if corr is None or corr.shape != (n, n):
        rho = 0.35
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
    eig = np.linalg.eigvalsh(corr)
    if eig.min() <= 1e-8:
        corr = corr + np.eye(n) * (abs(float(eig.min())) + 1e-6)
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(n), corr, size=int(n_sims))
    u = stats.norm.cdf(z)
    components = np.zeros_like(u)
    for i, row in merged.iterrows():
        mean = float(row["posterior_mean"])
        sd = float(row.get("posterior_sd", np.nan))
        if not np.isfinite(sd) or sd <= 0:
            components[:, i] = mean
        else:
            components[:, i] = stats.norm.ppf(u[:, i], loc=mean, scale=sd)
    weight_col = f"weight_{scheme}"
    w = merged[weight_col].to_numpy(dtype=float)
    totals = components @ w
    return totals, merged
