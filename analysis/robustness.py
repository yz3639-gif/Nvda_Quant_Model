"""Lightweight robustness diagnostics for v2 reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_out(samples_by_event: dict[str, pd.DataFrame], detail_by_event: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for event_id, detail in detail_by_event.items():
        target = detail[detail.get("window", "") == "post_1_5"] if not detail.empty else pd.DataFrame()
        if target.empty:
            rows.append({"event_id": event_id, "status": "BLACK", "reason": "no post_1_5 detail"})
            continue
        base = float(target["car"].mean())
        for sample_id in target["sample_id"].dropna().unique():
            loo = target[target["sample_id"] != sample_id]["car"]
            rows.append(
                {
                    "event_id": event_id,
                    "left_out_sample_id": sample_id,
                    "base_mean": base,
                    "loo_mean": float(loo.mean()) if not loo.empty else np.nan,
                    "delta": float(loo.mean() - base) if not loo.empty else np.nan,
                    "status": "GREEN" if len(target) >= 30 else ("YELLOW" if len(target) >= 10 else "RED"),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_sample_selection(detail_by_event: dict[str, pd.DataFrame], n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for event_id, detail in detail_by_event.items():
        target = detail[detail.get("window", "") == "post_1_5"] if not detail.empty else pd.DataFrame()
        vals = pd.to_numeric(target.get("car", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        if vals.size < 3:
            rows.append({"event_id": event_id, "status": "RED", "reason": "sample too small"})
            continue
        draws = rng.choice(vals, size=(int(n_boot), vals.size), replace=True).mean(axis=1)
        rows.append(
            {
                "event_id": event_id,
                "n_boot": int(n_boot),
                "mean": float(draws.mean()),
                "p05": float(np.quantile(draws, 0.05)),
                "p50": float(np.quantile(draws, 0.50)),
                "p95": float(np.quantile(draws, 0.95)),
                "status": "GREEN" if vals.size >= 30 else ("YELLOW" if vals.size >= 10 else "RED"),
            }
        )
    return pd.DataFrame(rows)


def placebo_test(prices: pd.DataFrame, n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if "SPY" not in prices:
        return pd.DataFrame([{"status": "BLACK", "reason": "SPY missing"}])
    ret = prices["SPY"].pct_change(fill_method=None).dropna()
    if len(ret) < 200:
        return pd.DataFrame([{"status": "BLACK", "reason": "not enough price data"}])
    positions = rng.choice(np.arange(60, len(ret) - 10), size=min(n, len(ret) - 70), replace=False)
    vals = [float(ret.iloc[pos + 1 : pos + 6].sum()) for pos in positions]
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    t = mean / (sd / np.sqrt(len(vals))) if sd > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "test": "random_non_event_days_post_1_5",
                "n": len(vals),
                "mean": mean,
                "t_stat": t,
                "passes": bool(abs(t) < 1.96) if np.isfinite(t) else False,
                "status": "GREEN" if np.isfinite(t) and abs(t) < 1.96 else "YELLOW",
            }
        ]
    )


def oos_backtest_placeholder() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "test": "2021-2025 out-of-sample",
                "status": "BLACK",
                "reason": "True macro surprise history and historical option surfaces are unavailable in free sources; no synthetic backtest is produced.",
            }
        ]
    )


def sensitivity_placeholder() -> pd.DataFrame:
    rows = []
    for vix in [35, 40, 50]:
        for dd in [-0.05, -0.07, -0.10]:
            rows.append({"vix_threshold": vix, "weekly_spx_drawdown": dd, "status": "YELLOW", "note": "requires rerun with alternate noise filters"})
    return pd.DataFrame(rows)
