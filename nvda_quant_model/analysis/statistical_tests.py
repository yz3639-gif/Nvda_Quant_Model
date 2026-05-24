from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson


def statistical_tests(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    active_mask: pd.Series | None = None,
) -> dict[str, float | str]:
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {"p_value": 1.0, "t_stat": 0.0, "conclusion": "样本不足，无法检验"}

    diff = aligned["strategy"] - aligned["benchmark"]
    excess_two_sided_t, excess_two_sided_p = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
    positive_return_t, positive_return_p = stats.ttest_1samp(
        aligned["strategy"],
        0.0,
        nan_policy="omit",
        alternative="greater",
    )
    active_excess_t = np.nan
    active_excess_p = np.nan
    if active_mask is not None:
        active = active_mask.reindex(aligned.index).fillna(False).astype(bool)
        active_diff = diff.loc[active].dropna()
        if len(active_diff) >= 3:
            active_excess_t, active_excess_p = stats.ttest_1samp(
                active_diff,
                0.0,
                nan_policy="omit",
                alternative="greater",
            )
    primary_t = active_excess_t if not np.isnan(active_excess_t) else positive_return_t
    primary_p = active_excess_p if not np.isnan(active_excess_p) else positive_return_p
    strategy_sharpe = np.sqrt(252) * aligned["strategy"].mean() / aligned["strategy"].std(ddof=0)
    benchmark_sharpe = np.sqrt(252) * aligned["benchmark"].mean() / aligned["benchmark"].std(ddof=0)
    shapiro_p = stats.shapiro(diff.sample(min(len(diff), 500), random_state=42)).pvalue if len(diff) >= 3 else np.nan
    dw = durbin_watson(diff)
    conclusion = "策略有统计显著性 (p < 0.05)" if primary_p < 0.05 else "策略无显著性，可能是随机波动"
    return {
        "t_stat": float(primary_t) if not np.isnan(primary_t) else 0.0,
        "p_value": float(primary_p) if not np.isnan(primary_p) else 1.0,
        "positive_return_t": float(positive_return_t) if not np.isnan(positive_return_t) else 0.0,
        "positive_return_p": float(positive_return_p) if not np.isnan(positive_return_p) else 1.0,
        "excess_two_sided_t": float(excess_two_sided_t) if not np.isnan(excess_two_sided_t) else 0.0,
        "excess_two_sided_p": float(excess_two_sided_p) if not np.isnan(excess_two_sided_p) else 1.0,
        "active_excess_t": float(active_excess_t) if not np.isnan(active_excess_t) else 0.0,
        "active_excess_p": float(active_excess_p) if not np.isnan(active_excess_p) else 1.0,
        "strategy_sharpe": float(strategy_sharpe) if not np.isnan(strategy_sharpe) else 0.0,
        "benchmark_sharpe": float(benchmark_sharpe) if not np.isnan(benchmark_sharpe) else 0.0,
        "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else 1.0,
        "durbin_watson": float(dw) if not np.isnan(dw) else 0.0,
        "conclusion": conclusion,
    }
