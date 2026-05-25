from __future__ import annotations

import numpy as np
import pandas as pd

def monte_carlo_bootstrap(
    strategy_returns: pd.Series,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    returns = strategy_returns.dropna().to_numpy()
    if len(returns) < 20:
        return {"bootstrap_sharpe_p05": 0.0, "bootstrap_mdd_p95": 0.0, "observed_sharpe": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns, size=(n_sims, len(returns)), replace=True)
    vols = samples.std(axis=1)
    sharpes = np.divide(
        np.sqrt(252) * samples.mean(axis=1),
        vols,
        out=np.zeros(n_sims, dtype=float),
        where=vols != 0,
    )
    equity = np.cumprod(1.0 + samples, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / running_max - 1.0
    mdds = np.abs(drawdowns.min(axis=1))
    observed_vol = returns.std()
    observed_sharpe = np.sqrt(252) * returns.mean() / observed_vol if observed_vol else 0.0
    return {
        "observed_sharpe": float(observed_sharpe),
        "bootstrap_sharpe_p05": float(np.percentile(sharpes, 5)),
        "bootstrap_sharpe_p50": float(np.percentile(sharpes, 50)),
        "bootstrap_mdd_p95": float(np.percentile(mdds, 95)),
    }


def oos_degradation(full_metrics: dict[str, float], oos_metrics: dict[str, float]) -> dict[str, float | bool]:
    full = full_metrics.get("sharpe_ratio", 0.0)
    oos = oos_metrics.get("sharpe_ratio", 0.0)
    degradation = (full - oos) / abs(full) if full else 0.0
    return {
        "sharpe_degradation": float(degradation),
        "passes_20pct_rule": bool(degradation <= 0.20),
    }
