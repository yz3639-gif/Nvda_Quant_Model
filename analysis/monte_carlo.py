"""Aggregate event posterior distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_event_weights(events_detail: pd.DataFrame, posteriors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in events_detail.iterrows():
        event_id = row["event_id"]
        posterior = posteriors.loc[posteriors["event_id"] == event_id]
        if posterior.empty:
            impact = 0.0
        else:
            p = posterior.iloc[0]
            impact = abs(float(p.get("prior_mean", 0) or 0)) + abs(float(p.get("posterior_mean", 0) or 0))
            if np.isfinite(p.get("p_up", np.nan)):
                impact += abs(float(p["p_up"]) - 0.5) * 0.01
        importance = float(row.get("importance", 3))
        raw_weight = max(0.0001, impact) * importance
        rows.append({"event_id": event_id, "event_name": row.get("event_name", event_id), "raw_weight": raw_weight})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["weight"] = out["raw_weight"] / out["raw_weight"].sum()
    return out


def run_aggregate_monte_carlo(
    posteriors: pd.DataFrame,
    weights: pd.DataFrame,
    n_sims: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if posteriors.empty or weights.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = weights.merge(posteriors, on="event_id", how="left")
    draws = np.zeros(int(n_sims), dtype=float)
    component_rows = []
    for _, row in merged.iterrows():
        mean = float(row.get("posterior_mean", np.nan))
        sd = float(row.get("posterior_sd", np.nan))
        if not np.isfinite(mean):
            continue
        if not np.isfinite(sd) or sd <= 0:
            component = np.full(int(n_sims), mean)
        else:
            component = rng.normal(mean, sd, size=int(n_sims))
        weighted = float(row["weight"]) * component
        draws += weighted
        component_rows.append(
            {
                "event_id": row["event_id"],
                "event_name": row.get("event_name", row["event_id"]),
                "weight": float(row["weight"]),
                "posterior_mean": mean,
                "posterior_sd": sd,
                "weighted_mean": float(np.mean(weighted)),
            }
        )
    summary = {
        "n_sims": int(n_sims),
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "p25": float(np.quantile(draws, 0.25)),
        "p75": float(np.quantile(draws, 0.75)),
        "best": float(np.max(draws)),
        "worst": float(np.min(draws)),
        "p_gt_0": float(np.mean(draws > 0)),
        "p_gt_1pct": float(np.mean(draws > 0.01)),
        "p_gt_2pct": float(np.mean(draws > 0.02)),
        "p_lt_minus_1pct": float(np.mean(draws < -0.01)),
        "p_lt_minus_2pct": float(np.mean(draws < -0.02)),
        "p_lt_minus_3pct": float(np.mean(draws < -0.03)),
    }
    paths_summary = pd.DataFrame([summary])
    components = pd.DataFrame(component_rows)
    return paths_summary, components
