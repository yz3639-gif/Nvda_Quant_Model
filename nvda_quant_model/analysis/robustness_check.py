from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.metrics import max_drawdown


def monte_carlo_bootstrap(
    strategy_returns: pd.Series,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    returns = strategy_returns.dropna().to_numpy()
    if len(returns) < 20:
        return {"bootstrap_sharpe_p05": 0.0, "bootstrap_mdd_p95": 0.0, "observed_sharpe": 0.0}
    rng = np.random.default_rng(seed)
    sharpes = []
    mdds = []
    for _ in range(n_sims):
        sample = rng.choice(returns, size=len(returns), replace=True)
        vol = sample.std()
        sharpes.append(np.sqrt(252) * sample.mean() / vol if vol else 0.0)
        equity = pd.Series((1 + sample).cumprod())
        mdds.append(abs(max_drawdown(equity)[0]))
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

