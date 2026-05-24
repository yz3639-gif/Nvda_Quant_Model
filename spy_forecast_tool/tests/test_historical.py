"""Tests for historical block bootstrap behavior."""

from __future__ import annotations

import numpy as np

from methods.historical import simulate_block_bootstrap_paths


def test_block_bootstrap_preserves_return_stationarity_moments() -> None:
    """Block bootstrap sampled returns should preserve first two moments."""

    rng = np.random.default_rng(321)
    innovations = rng.normal(0.0004, 0.01, size=1_500)
    returns = np.empty_like(innovations)
    returns[0] = innovations[0]
    for index in range(1, innovations.size):
        returns[index] = 0.25 * returns[index - 1] + innovations[index]
    paths = simulate_block_bootstrap_paths(
        returns=returns,
        spot=100.0,
        horizon_days=80,
        n_sims=6_000,
        block_size=5,
        rng=np.random.default_rng(999),
    )
    sampled_returns = np.diff(np.log(paths), axis=1).ravel()
    assert abs(sampled_returns.mean() - returns.mean()) < 7.5e-4
    assert abs(sampled_returns.std(ddof=1) - returns.std(ddof=1)) / returns.std(ddof=1) < 0.06

