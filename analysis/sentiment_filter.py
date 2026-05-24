"""Environment-matched historical subset selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_DIMS = [
    "VIX",
    "MOVE",
    "HY_OAS",
    "IG_OAS",
    "NFCI",
    "curve_2s10s",
    "T10YIE",
    "DFII10",
    "WALCL_YOY",
    "SPY_20d_realized_vol",
    "USEPUINDXD",
]

SENTIMENT_DIMS = [
    "AAII_bull_bear_spread",
    "CFTC_emini_spx_net_percentile",
    "CFTC_10y_ust_net_percentile",
    "CBOE_equity_pc_10dma",
    "CBOE_total_pc_10dma",
    "NAAIM_exposure",
    "FINRA_margin_debt_yoy",
    "ICI_4w_equity_flow_b",
]


def build_environment_vector(snapshot_df: pd.DataFrame) -> dict:
    """Return the current environment vector using only available dimensions."""

    available_dims = {}
    if snapshot_df is None or snapshot_df.empty:
        return available_dims
    for dim in PRICE_DIMS + SENTIMENT_DIMS:
        row = snapshot_df[snapshot_df["metric"] == dim]
        if row.empty:
            continue
        item = row.iloc[0]
        if item.get("status") == "available" and pd.notna(item.get("value")):
            available_dims[dim] = {
                "value": float(item.get("value")),
                "percentile": float(item.get("percentile")) if pd.notna(item.get("percentile")) else np.nan,
                "z_score": float(item.get("z_score")) if pd.notna(item.get("z_score")) else np.nan,
            }
    return available_dims


def match_historical_environment(
    current_env: dict,
    historical_sentiment: pd.DataFrame,
    min_dims_required: int = 3,
    top_k_pct: float = 0.5,
) -> tuple[list, dict]:
    """Match sample-indexed historical rows to the current environment."""

    if len(current_env) < min_dims_required:
        return [], {
            "status": "insufficient_dimensions",
            "available_dims": list(current_env.keys()),
            "min_required": min_dims_required,
        }
    if historical_sentiment is None or historical_sentiment.empty:
        return [], {
            "status": "no_historical_sentiment",
            "available_dims": list(current_env.keys()),
            "min_required": min_dims_required,
        }

    distances = []
    used_dims = [dim for dim in current_env if dim in historical_sentiment.columns]
    for sample_id, sample_row in historical_sentiment.iterrows():
        z_current = []
        z_sample = []
        for dim in used_dims:
            sample_value = sample_row.get(dim)
            current_z = current_env[dim].get("z_score", np.nan)
            if pd.notna(sample_value) and np.isfinite(current_z):
                z_current.append(current_z)
                z_sample.append(float(sample_value))
        if len(z_sample) >= min_dims_required:
            distances.append((sample_id, float(np.linalg.norm(np.array(z_current) - np.array(z_sample)))))

    if not distances:
        return [], {
            "status": "insufficient_historical_overlap",
            "available_dims": list(current_env.keys()),
            "used_dimensions": used_dims,
            "min_required": min_dims_required,
        }

    distances.sort(key=lambda x: x[1])
    cutoff = max(1, int(np.ceil(len(distances) * top_k_pct)))
    matched_ids = [item[0] for item in distances[:cutoff]]
    return matched_ids, {
        "status": "success",
        "used_dimensions": used_dims,
        "n_dimensions": len(used_dims),
        "n_matched": len(matched_ids),
        "distance_p25": distances[cutoff // 4][1] if cutoff > 4 else None,
        "distance_p50": distances[cutoff // 2][1] if cutoff > 2 else None,
    }


def attach_environment_distances(
    samples: pd.DataFrame,
    sentiment_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    nearest_fraction: float = 0.5,
    snapshot_df: pd.DataFrame | None = None,
    min_dims_required: int = 3,
) -> pd.DataFrame:
    out = samples.copy()
    if out.empty:
        out["env_distance"] = np.nan
        out["environment_match"] = False
        out.attrs["env_match_metadata"] = {
            "status": "no_samples",
            "used_dimensions": [],
            "n_dimensions": 0,
            "n_matched": 0,
        }
        return out
    if sentiment_history is None or sentiment_history.empty:
        out["env_distance"] = np.nan
        out["environment_match"] = False
        out.attrs["env_match_metadata"] = {
            "status": "no_historical_sentiment",
            "used_dimensions": [],
            "n_dimensions": 0,
            "n_matched": 0,
        }
        return out

    hist = sentiment_history.copy().sort_index()
    current_env = build_environment_vector(snapshot_df) if snapshot_df is not None else _current_env_from_history(hist, as_of_date)
    candidate_dims = [dim for dim in current_env if dim in hist.columns and hist[dim].dropna().shape[0] >= 3]
    if len(candidate_dims) < min_dims_required:
        out["env_distance"] = np.nan
        out["environment_match"] = False
        out["env_feature_date"] = ""
        out.attrs["env_match_metadata"] = {
            "status": "insufficient_dimensions",
            "available_dims": list(current_env.keys()),
            "used_dimensions": candidate_dims,
            "n_dimensions": len(candidate_dims),
            "n_matched": 0,
            "min_required": min_dims_required,
        }
        return out

    means = hist[candidate_dims].mean()
    stds = hist[candidate_dims].std(ddof=1).replace(0, np.nan)
    current_z = {}
    for dim in candidate_dims:
        z = current_env[dim].get("z_score", np.nan)
        if not np.isfinite(z):
            value = current_env[dim].get("value", np.nan)
            z = (value - means[dim]) / stds[dim] if np.isfinite(value) and np.isfinite(stds[dim]) else np.nan
        current_z[dim] = z

    distances = []
    matched_dates = []
    used_count = []
    for _, row in out.iterrows():
        event_date = pd.Timestamp(row["date"])
        sample_dates = hist.index[hist.index <= event_date - pd.Timedelta(days=1)]
        if len(sample_dates) == 0:
            distances.append(np.nan)
            matched_dates.append("")
            used_count.append(0)
            continue
        sample_date = sample_dates[-1]
        sample_slice = hist.loc[[sample_date], candidate_dims]
        sample = sample_slice.iloc[-1] if isinstance(sample_slice, pd.DataFrame) else sample_slice
        z_current_vec = []
        z_sample_vec = []
        for dim in candidate_dims:
            sample_value = sample.get(dim, np.nan)
            if pd.notna(sample_value) and np.isfinite(current_z[dim]) and np.isfinite(stds[dim]):
                z_current_vec.append(current_z[dim])
                z_sample_vec.append((float(sample_value) - means[dim]) / stds[dim])
        if len(z_sample_vec) >= min_dims_required:
            distances.append(float(np.sqrt(np.mean((np.array(z_current_vec) - np.array(z_sample_vec)) ** 2))))
        else:
            distances.append(np.nan)
        matched_dates.append(pd.Timestamp(sample_date).date().isoformat())
        used_count.append(len(z_sample_vec))

    out["env_distance"] = distances
    out["env_feature_date"] = matched_dates
    out["env_dimensions_used"] = used_count
    valid = out["env_distance"].dropna()
    if valid.empty:
        out["environment_match"] = False
        meta = {
            "status": "insufficient_historical_overlap",
            "used_dimensions": candidate_dims,
            "n_dimensions": len(candidate_dims),
            "n_matched": 0,
        }
    else:
        n_keep = max(1, int(np.ceil(len(valid) * float(nearest_fraction))))
        cutoff = valid.nsmallest(n_keep).max()
        out["environment_match"] = out["env_distance"] <= cutoff
        sorted_dist = valid.sort_values().to_numpy()
        meta = {
            "status": "success",
            "used_dimensions": candidate_dims,
            "n_dimensions": len(candidate_dims),
            "n_matched": int(out["environment_match"].sum()),
            "distance_p25": float(np.quantile(sorted_dist, 0.25)),
            "distance_p50": float(np.quantile(sorted_dist, 0.50)),
            "distance_mean": float(np.mean(sorted_dist)),
        }
    out.attrs["env_match_metadata"] = meta
    return out


def _current_env_from_history(hist: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    current_dates = hist.index[(hist.index <= as_of_date) & (pd.DatetimeIndex(hist.index).weekday < 5)]
    if len(current_dates) == 0:
        return {}
    current = hist.loc[current_dates[-1]]
    out = {}
    for dim in PRICE_DIMS + SENTIMENT_DIMS:
        if dim not in hist:
            continue
        value = current.get(dim, np.nan)
        if pd.notna(value):
            series = hist[dim].dropna()
            std = series.std(ddof=1)
            out[dim] = {
                "value": float(value),
                "percentile": float((series <= value).mean() * 100.0) if not series.empty else np.nan,
                "z_score": float((value - series.mean()) / std) if std and np.isfinite(std) else np.nan,
            }
    return out
