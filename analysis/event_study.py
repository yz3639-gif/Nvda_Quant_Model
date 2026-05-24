"""Market-model event study implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analysis.statistics import holm_adjust, summarize_cars


@dataclass(slots=True)
class EventStudyResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[str]


def run_event_study(
    samples: pd.DataFrame,
    prices: pd.DataFrame,
    asset: str,
    benchmark: str,
    windows: dict[str, list[int]],
    n_boot: int,
    rng: np.random.Generator,
    label: str = "full",
) -> EventStudyResult:
    """Run event study for one asset over matched historical samples."""

    warnings: list[str] = []
    if samples.empty or asset not in prices or benchmark not in prices:
        return EventStudyResult(pd.DataFrame(), pd.DataFrame(), [f"No samples or missing prices for {asset}/{benchmark}."])
    columns = [asset] if asset == benchmark else [asset, benchmark]
    aligned = prices[columns].dropna(how="any").sort_index()
    returns = aligned.pct_change(fill_method=None).dropna(how="any")
    rows: list[dict[str, Any]] = []
    for _, sample in samples.iterrows():
        event_date = pd.Timestamp(sample["date"])
        pos = _event_position(returns.index, event_date)
        if pos is None:
            warnings.append(f"{sample.get('sample_id')}: no trading date on/after {event_date.date()}.")
            continue
        alpha, beta, sigma, enough_est = _market_model_params(returns, asset, benchmark, pos)
        if not enough_est:
            warnings.append(f"{sample.get('sample_id')}: insufficient estimation window for Market Model.")
        for window_name, (start_offset, end_offset) in windows.items():
            start_pos = pos + int(start_offset)
            end_pos = pos + int(end_offset)
            if start_pos < 0 or end_pos >= len(returns) or start_pos > end_pos:
                continue
            asset_ret = returns[asset].iloc[start_pos : end_pos + 1]
            if asset == benchmark:
                bench_ret = asset_ret
                # Avoid zero abnormal returns for the primary index against itself.
                expected = pd.Series(alpha, index=asset_ret.index)
            else:
                bench_ret = returns[benchmark].iloc[start_pos : end_pos + 1]
                expected = alpha + beta * bench_ret
            ar = asset_ret - expected
            car = float(ar.sum())
            raw = float(asset_ret.sum())
            expected_sum = float(expected.sum())
            sigma_safe = sigma if np.isfinite(sigma) and sigma > 0 else float(asset_ret.std(ddof=1) or np.nan)
            standardized = car / (sigma_safe * np.sqrt(len(asset_ret))) if np.isfinite(sigma_safe) and sigma_safe > 0 else np.nan
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "event_name": sample.get("event_name", ""),
                    "event_date": event_date.date().isoformat(),
                    "trading_date": returns.index[pos].date().isoformat(),
                    "category": sample.get("category", ""),
                    "match_type": sample.get("match_type", ""),
                    "comparability": sample.get("comparability", np.nan),
                    "window": window_name,
                    "window_start": returns.index[start_pos].date().isoformat(),
                    "window_end": returns.index[end_pos].date().isoformat(),
                    "asset": asset,
                    "benchmark": benchmark,
                    "raw_return": raw,
                    "expected_return": expected_sum,
                    "car": car,
                    "standardized_car": standardized,
                    "alpha": alpha,
                    "beta": beta,
                    "residual_sigma": sigma,
                    "excluded_noise": bool(sample.get("excluded_noise", False)),
                    "noise_reason": sample.get("noise_reason", ""),
                    "subset": label,
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return EventStudyResult(detail, pd.DataFrame(), warnings)
    summary_rows = []
    for window_name, group in detail.groupby("window", sort=False):
        stats = summarize_cars(group["car"], group["standardized_car"], n_boot=n_boot, rng=rng)
        stats.update({"window": window_name, "asset": asset, "benchmark": benchmark, "subset": label})
        summary_rows.append(stats)
    summary = holm_adjust(pd.DataFrame(summary_rows), p_col="p_value")
    return EventStudyResult(detail, summary, warnings)


def _event_position(index: pd.Index, event_date: pd.Timestamp) -> int | None:
    idx = pd.DatetimeIndex(index)
    locs = np.where(idx >= event_date.normalize())[0]
    if len(locs) == 0:
        return None
    return int(locs[0])


def _market_model_params(returns: pd.DataFrame, asset: str, benchmark: str, event_pos: int) -> tuple[float, float, float, bool]:
    start = max(0, event_pos - 250)
    end = event_pos - 20
    if end <= start + 30:
        hist = returns.iloc[:event_pos]
    else:
        hist = returns.iloc[start:end]
    hist = hist[[asset]].dropna() if asset == benchmark else hist[[asset, benchmark]].dropna()
    if hist.shape[0] < 30:
        mu = float(returns[asset].iloc[max(0, event_pos - 60) : event_pos].mean())
        return (mu if np.isfinite(mu) else 0.0), 0.0, float("nan"), False
    y = hist[asset].to_numpy(dtype=float)
    if asset == benchmark:
        resid = y - np.mean(y)
        return float(np.mean(y)), 0.0, float(np.std(resid, ddof=1)), True
    x = hist[benchmark].to_numpy(dtype=float)
    var = float(np.var(x, ddof=1))
    if var <= 0 or not np.isfinite(var):
        resid = y - np.mean(y)
        return float(np.mean(y)), 0.0, float(np.std(resid, ddof=1)), True
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var)
    alpha = float(np.mean(y) - beta * np.mean(x))
    resid = y - (alpha + beta * x)
    return alpha, beta, float(np.std(resid, ddof=1)), True
