"""Tests for GBM Monte Carlo behavior."""

from __future__ import annotations

import numpy as np
import pytest

from methods.monte_carlo import (
    MAX_STUDENT_T_DF,
    MIN_STUDENT_T_DF,
    estimate_gbm_parameters,
    fit_student_t_distribution,
    simulate_gbm_paths,
    simulate_student_t_gbm_paths,
)


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


def test_gbm_parameter_estimation_rejects_zero_volatility() -> None:
    """Degenerate returns should not produce silent NaN or zero-vol forecasts."""

    with pytest.raises(ValueError, match="volatility"):
        estimate_gbm_parameters(np.zeros(20))


def test_student_t_fit_clips_extreme_degrees_of_freedom() -> None:
    """Student-t degrees of freedom should stay in a numerically reasonable range."""

    returns = np.linspace(-0.02, 0.02, 200)
    params = fit_student_t_distribution(returns)
    assert MIN_STUDENT_T_DF <= params.df <= MAX_STUDENT_T_DF


def test_student_t_simulation_rejects_invalid_shape_parameters() -> None:
    """Student-t simulation should fail clearly for infinite-variance df values."""

    with pytest.raises(ValueError, match="degrees of freedom"):
        simulate_student_t_gbm_paths(
            spot=100.0,
            horizon_days=5,
            n_sims=10,
            mu_annual=0.05,
            sigma_annual=0.2,
            df=2.0,
            rng=np.random.default_rng(1),
        )
