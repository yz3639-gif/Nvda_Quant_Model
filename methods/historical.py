"""Historical block bootstrap simulation for SPY forward returns."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.stats import ScenarioResult, build_result, make_price_paths


def simulate_block_bootstrap_paths(
    returns: np.ndarray,
    spot: float,
    horizon_days: int,
    n_sims: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate forward price paths using vectorized moving block bootstrap."""

    clean_returns = np.asarray(returns, dtype=float)
    clean_returns = clean_returns[np.isfinite(clean_returns)]
    if clean_returns.size < 2:
        raise ValueError("Historical bootstrap requires at least two valid return observations.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    effective_block_size = min(block_size, clean_returns.size)
    n_blocks = int(np.ceil(horizon_days / effective_block_size))
    max_start = clean_returns.size - effective_block_size + 1
    starts = rng.integers(0, max_start, size=(n_sims, n_blocks))
    offsets = np.arange(effective_block_size, dtype=int)
    indices = starts[..., None] + offsets
    sampled_returns = clean_returns[indices].reshape(n_sims, -1)[:, :horizon_days]
    return make_price_paths(spot, sampled_returns)


def run_historical_bootstrap(
    returns: np.ndarray,
    spot: float,
    horizon_days: int,
    n_sims: int,
    block_size: int,
    rng: np.random.Generator,
    thresholds: Iterable[float],
    lookback_years: float,
) -> ScenarioResult:
    """Run the historical block bootstrap method and return summarized results."""

    paths = simulate_block_bootstrap_paths(
        returns=returns,
        spot=spot,
        horizon_days=horizon_days,
        n_sims=n_sims,
        block_size=block_size,
        rng=rng,
    )
    assumptions = [
        f"Uses the last {lookback_years:g} years of adjusted close log returns.",
        f"Moving block bootstrap with block size {block_size} trading days.",
        "Forward paths are sampled from historical return blocks with replacement.",
        "Assumes the historical return distribution remains informative under the current market regime.",
    ]
    return build_result(
        name="Historical Bootstrap",
        paths=paths,
        spot=spot,
        thresholds=thresholds,
        assumptions=assumptions,
        extras={"method_detail": "moving_block_bootstrap"},
    )
