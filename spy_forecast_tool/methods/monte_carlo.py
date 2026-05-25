"""Geometric Brownian motion Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
from scipy import stats

from utils.stats import ScenarioResult, build_result, make_price_paths


TRADING_DAYS_PER_YEAR = 252
MIN_STUDENT_T_DF = 2.1
MAX_STUDENT_T_DF = 30.0


@dataclass(slots=True)
class GBMParameters:
    """Annualized GBM parameters estimated from daily log returns."""

    mean_daily: float
    std_daily: float
    mu_annual: float
    sigma_annual: float


@dataclass(slots=True)
class StudentTParameters:
    """Student t distribution parameters fitted by maximum likelihood."""

    df: float
    loc: float
    scale: float


def estimate_gbm_parameters(returns: np.ndarray) -> GBMParameters:
    """Estimate annualized drift and volatility from daily log returns."""

    clean_returns = np.asarray(returns, dtype=float)
    clean_returns = clean_returns[np.isfinite(clean_returns)]
    if clean_returns.size < 2:
        raise ValueError("GBM parameter estimation requires at least two valid returns.")
    mean_daily = float(np.mean(clean_returns))
    std_daily = float(np.std(clean_returns, ddof=1))
    if not np.isfinite(std_daily) or std_daily <= 0:
        raise ValueError("GBM volatility estimate must be finite and positive.")
    return GBMParameters(
        mean_daily=mean_daily,
        std_daily=std_daily,
        mu_annual=mean_daily * TRADING_DAYS_PER_YEAR,
        sigma_annual=std_daily * np.sqrt(TRADING_DAYS_PER_YEAR),
    )


def fit_student_t_distribution(returns: np.ndarray) -> StudentTParameters:
    """Fit a Student t distribution to daily log returns by maximum likelihood."""

    clean_returns = np.asarray(returns, dtype=float)
    clean_returns = clean_returns[np.isfinite(clean_returns)]
    if clean_returns.size < 10:
        raise ValueError("Student t fit requires at least ten valid returns.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, scale = stats.t.fit(clean_returns)
    if not np.isfinite(df):
        df = MAX_STUDENT_T_DF
    df = float(np.clip(df, MIN_STUDENT_T_DF, MAX_STUDENT_T_DF))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(clean_returns, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Student t scale estimate must be finite and positive.")
    if not np.isfinite(loc):
        loc = float(np.mean(clean_returns))
    return StudentTParameters(df=float(df), loc=float(loc), scale=float(scale))


def simulate_gbm_paths(
    spot: float,
    horizon_days: int,
    n_sims: int,
    mu_annual: float,
    sigma_annual: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate vectorized GBM price paths with normally distributed shocks."""

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if sigma_annual <= 0:
        raise ValueError("sigma_annual must be positive.")
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    shocks = rng.standard_normal(size=(n_sims, horizon_days))
    increments = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * shocks
    return make_price_paths(spot, increments)


def simulate_student_t_gbm_paths(
    spot: float,
    horizon_days: int,
    n_sims: int,
    mu_annual: float,
    sigma_annual: float,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate vectorized GBM paths with standardized Student t shocks."""

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if sigma_annual <= 0:
        raise ValueError("sigma_annual must be positive.")
    if df <= 2.0 or not np.isfinite(df):
        raise ValueError("Student t degrees of freedom must exceed 2 for finite variance.")
    df = float(np.clip(df, MIN_STUDENT_T_DF, MAX_STUDENT_T_DF))
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    raw_shocks = rng.standard_t(df=df, size=(n_sims, horizon_days))
    standardized_shocks = raw_shocks / np.sqrt(df / (df - 2.0))
    increments = (
        (mu_annual - 0.5 * sigma_annual**2) * dt
        + sigma_annual * np.sqrt(dt) * standardized_shocks
    )
    return make_price_paths(spot, increments)


def run_monte_carlo_methods(
    returns: np.ndarray,
    spot: float,
    horizon_days: int,
    n_sims: int,
    rng: np.random.Generator,
    thresholds: Iterable[float],
    lookback_years: float,
) -> tuple[ScenarioResult, ScenarioResult]:
    """Run normal GBM and Student t GBM variants and return both summaries."""

    params = estimate_gbm_parameters(returns)
    t_params = fit_student_t_distribution(returns)
    normal_paths = simulate_gbm_paths(
        spot=spot,
        horizon_days=horizon_days,
        n_sims=n_sims,
        mu_annual=params.mu_annual,
        sigma_annual=params.sigma_annual,
        rng=rng,
    )
    t_paths = simulate_student_t_gbm_paths(
        spot=spot,
        horizon_days=horizon_days,
        n_sims=n_sims,
        mu_annual=params.mu_annual,
        sigma_annual=params.sigma_annual,
        df=t_params.df,
        rng=rng,
    )
    base_assumptions = [
        f"Uses the last {lookback_years:g} years of daily log returns.",
        f"Annualized drift mu = {params.mu_annual:.6f}, annualized volatility sigma = {params.sigma_annual:.6f}.",
        "Uses 252 trading days per year and constant drift/volatility over the forecast horizon.",
        "GBM assumes continuous compounding and independent daily shocks.",
    ]
    normal = build_result(
        name="GBM Normal Monte Carlo",
        paths=normal_paths,
        spot=spot,
        thresholds=thresholds,
        assumptions=base_assumptions + ["Shock distribution is standard normal."],
        extras={
            "method_detail": "gbm_normal",
            "mu_annual": params.mu_annual,
            "sigma_annual": params.sigma_annual,
        },
    )
    student_t = build_result(
        name="GBM Student-t Monte Carlo",
        paths=t_paths,
        spot=spot,
        thresholds=thresholds,
        assumptions=base_assumptions
        + [
            f"Shock distribution is standardized Student t with df = {t_params.df:.3f}, estimated by MLE.",
            "Student t shocks are scaled to unit variance before applying annualized sigma.",
        ],
        extras={
            "method_detail": "gbm_student_t",
            "mu_annual": params.mu_annual,
            "sigma_annual": params.sigma_annual,
            "student_t_df": t_params.df,
            "student_t_loc": t_params.loc,
            "student_t_scale": t_params.scale,
        },
        is_primary=False,
    )
    return normal, student_t
