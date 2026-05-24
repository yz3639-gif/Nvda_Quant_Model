"""v2 event weighting with explicit small-sample penalties and caps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_weight_sets(
    events_detail: pd.DataFrame,
    posteriors: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if events_detail.empty:
        return pd.DataFrame()
    merged = events_detail.merge(posteriors, on="event_id", how="left", suffixes=("", "_posterior"))
    sig = _event_significance(signals)
    merged = merged.merge(sig, on="event_id", how="left")
    rows = []
    cap = float(config.get("weight_formula", {}).get("economic_signal_cap", 0.05))
    penalties = config.get("weight_formula", {}).get("small_sample_penalty", {})
    macro_cap = float(config.get("weight_formula", {}).get("macro_no_surprise_weight_cap", 0.05))
    n_events = len(merged)
    for _, row in merged.iterrows():
        n_clean = int(row.get("clean_samples", 0) or 0)
        p_min = float(row.get("min_holm_p", 1.0)) if np.isfinite(row.get("min_holm_p", np.nan)) else 1.0
        significance = max(0.0, 1.0 - min(1.0, p_min))
        signal = min(abs(float(row.get("posterior_mean", 0.0) or 0.0)), cap)
        importance = float(row.get("importance", 3) or 3)
        raw = (np.sqrt(max(n_clean, 0)) * significance * max(signal, 1e-6) * importance)
        if n_clean < 10:
            raw *= float(penalties.get("n_lt_10", 0.3))
        elif n_clean < 30:
            raw *= float(penalties.get("n_lt_30", 0.6))
        if bool(row.get("macro_no_surprise", False)):
            raw = min(raw, macro_cap)
        rows.append(
            {
                "event_id": row["event_id"],
                "event_name": row.get("event_name", row["event_id"]),
                "n_clean": n_clean,
                "importance": importance,
                "min_holm_p": p_min,
                "significance_score": significance,
                "economic_signal_abs_capped": signal,
                "macro_no_surprise": bool(row.get("macro_no_surprise", False)),
                "raw_composite": raw,
                "raw_equal": 1.0 / n_events if n_events else 0.0,
                "raw_n_weighted": np.sqrt(max(n_clean, 0)),
            }
        )
    out = pd.DataFrame(rows)
    for scheme, col in [("composite", "raw_composite"), ("equal", "raw_equal"), ("n_weighted", "raw_n_weighted")]:
        denom = float(out[col].sum())
        if denom <= 0:
            out[f"weight_{scheme}"] = 1.0 / len(out) if len(out) else np.nan
        else:
            out[f"weight_{scheme}"] = out[col] / denom
    if "macro_no_surprise" in out:
        over = out["macro_no_surprise"] & (out["weight_composite"] > macro_cap)
        if over.any():
            excess = float((out.loc[over, "weight_composite"] - macro_cap).sum())
            out.loc[over, "weight_composite"] = macro_cap
            recipients = ~over
            if recipients.any() and out.loc[recipients, "weight_composite"].sum() > 0:
                out.loc[recipients, "weight_composite"] += (
                    out.loc[recipients, "weight_composite"] / out.loc[recipients, "weight_composite"].sum()
                ) * excess
    return out


def _event_significance(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["event_id", "min_holm_p"])
    target = signals[
        (signals.get("asset", "") == "SPY")
        & (signals.get("subset", "") == "clean")
        & (signals.get("window", "").isin(["post_1_5", "post_1_20"]))
    ].copy()
    if target.empty:
        return pd.DataFrame({"event_id": signals["event_id"].drop_duplicates(), "min_holm_p": 1.0})
    p_col = "p_value_holm" if "p_value_holm" in target else "p_value"
    return target.groupby("event_id")[p_col].min().reset_index(name="min_holm_p")


def assert_small_sample_weight_bound(weights: pd.DataFrame, bound: float = 1.5) -> bool:
    if weights.empty:
        return True
    small = weights[weights["n_clean"] < 10]
    large = weights[weights["n_clean"] >= 100]
    if small.empty or large.empty:
        return True
    return float(small["weight_composite"].max()) <= bound * float(large["weight_composite"].max())
