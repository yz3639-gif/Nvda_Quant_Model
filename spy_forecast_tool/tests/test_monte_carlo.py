"""Tests for GBM Monte Carlo behavior."""

from __future__ import annotations

import numpy as np

from methods.monte_carlo import simulate_gbm_paths


def test_gbm_terminal_mean_converges_to_theoretical_expectation() -> None:
    """GBM terminal price mean should converge toward S0 * exp(mu * T)."""

    spot = 100.0
    mu = 0.08
    sigma = 0.20
    horizon_days = 63
    n_sims = 50_000
    rng = np.random.default_rng(123)
    paths = simulate_gbm_paths(
        spot=spot,
        horizon_days=horizon_days,
        n_sims=n_sims,
        mu_annual=mu,
        sigma_annual=sigma,
        rng=rng,
    )
    expected = spot * np.exp(mu * horizon_days / 252.0)
    assert abs(paths[:, -1].mean() - expected) / expected < 0.01

