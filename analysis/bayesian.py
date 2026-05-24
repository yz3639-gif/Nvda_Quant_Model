"""Bayesian prior/posterior update from pre-event drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.statistics import normal_kl


def bayesian_update_from_drift(
    event_id: str,
    detail: pd.DataFrame,
    current_pre_drift: float,
    sample_pre_drifts: pd.DataFrame | None = None,
    prior_window: str = "post_1_5",
    pre_window: str = "pre_5",
) -> dict:
    if detail.empty:
        return _empty(event_id, current_pre_drift, "no event-study detail")
    if sample_pre_drifts is not None and not sample_pre_drifts.empty:
        pre = sample_pre_drifts[["sample_id", "pre_drift"]].copy()
        pre_source = "event_specific_basket"
    else:
        pre = detail.loc[detail["window"] == pre_window, ["sample_id", "raw_return"]].rename(columns={"raw_return": "pre_drift"})
        pre_source = "fallback_asset_pre_window_return"
    post = detail.loc[detail["window"] == prior_window, ["sample_id", "car"]].rename(columns={"car": "post_car"})
    joined = pre.merge(post, on="sample_id", how="inner").dropna()
    arr = joined["post_car"].to_numpy(dtype=float)
    if arr.size < 3:
        return _empty(event_id, current_pre_drift, "sample不足，无法估计条件分布")
    prior_mean = float(np.mean(arr))
    prior_sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    posterior_mean = prior_mean
    posterior_sd = prior_sd
    beta = np.nan
    alpha = np.nan
    model_note = "prior fallback"
    effective_n = int(arr.size)
    if joined.shape[0] >= 5 and np.std(joined["pre_drift"], ddof=1) > 0 and np.isfinite(current_pre_drift):
        x = joined["pre_drift"].to_numpy(dtype=float)
        y = joined["post_car"].to_numpy(dtype=float)
        bandwidth = max(float(np.nanstd(x, ddof=1)) * 0.75, 1e-6)
        weights = np.exp(-0.5 * ((x - current_pre_drift) / bandwidth) ** 2)
        if float(weights.sum()) > 1e-8:
            weights = weights / weights.sum()
            posterior_mean = float(np.sum(weights * y))
            posterior_sd = float(np.sqrt(np.sum(weights * (y - posterior_mean) ** 2)))
            effective_n = int(np.ceil(1.0 / np.sum(weights**2)))
            model_note = "kernel conditional update P(CAR_post | event-specific pre-drift)"
    kl = normal_kl(prior_mean, max(prior_sd, 1e-8), posterior_mean, max(posterior_sd, 1e-8))
    return {
        "event_id": event_id,
        "n": int(arr.size),
        "current_pre_drift": float(current_pre_drift),
        "pre_drift_source": pre_source,
        "prior_mean": prior_mean,
        "prior_sd": prior_sd,
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "reg_alpha": alpha,
        "reg_beta": beta,
        "kl_divergence": kl,
        "p_up": _norm_prob_gt(0.0, posterior_mean, posterior_sd),
        "p_gt_1pct": _norm_prob_gt(0.01, posterior_mean, posterior_sd),
        "p_gt_2pct": _norm_prob_gt(0.02, posterior_mean, posterior_sd),
        "p_lt_minus_1pct": _norm_prob_lt(-0.01, posterior_mean, posterior_sd),
        "effective_n": effective_n,
        "warning": "" if arr.size >= 10 else "N<10 小样本，后验仅作方向性参考",
        "model_note": model_note,
    }


def _norm_prob_gt(threshold: float, mean: float, sd: float) -> float:
    if not np.isfinite(sd) or sd <= 0:
        return float(mean > threshold)
    from scipy import stats

    return float(1.0 - stats.norm.cdf(threshold, loc=mean, scale=sd))


def _norm_prob_lt(threshold: float, mean: float, sd: float) -> float:
    if not np.isfinite(sd) or sd <= 0:
        return float(mean < threshold)
    from scipy import stats

    return float(stats.norm.cdf(threshold, loc=mean, scale=sd))


def _empty(event_id: str, current_pre_drift: float, warning: str) -> dict:
    return {
        "event_id": event_id,
        "n": 0,
        "current_pre_drift": float(current_pre_drift) if np.isfinite(current_pre_drift) else np.nan,
        "pre_drift_source": "unavailable",
        "prior_mean": np.nan,
        "prior_sd": np.nan,
        "posterior_mean": np.nan,
        "posterior_sd": np.nan,
        "reg_alpha": np.nan,
        "reg_beta": np.nan,
        "kl_divergence": np.nan,
        "p_up": np.nan,
        "p_gt_1pct": np.nan,
        "p_gt_2pct": np.nan,
        "p_lt_minus_1pct": np.nan,
        "warning": warning,
        "effective_n": 0,
        "model_note": "unavailable",
    }
