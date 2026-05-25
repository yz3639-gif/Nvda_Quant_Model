from __future__ import annotations

from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd


def probability_calibration(
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    bins: int = 5,
) -> dict[str, Any]:
    """Score active long-signal probability calibration against realized next-day direction."""

    required_signal_cols = {"position", "prob_up"}
    required_frame_cols = {"target_direction"}
    if not required_signal_cols.issubset(signals.columns) or not required_frame_cols.issubset(frame.columns):
        raise ValueError("Calibration requires signal position/prob_up and frame target_direction columns.")

    aligned = frame[["target_direction"]].join(signals[["position", "prob_up"]], how="inner")
    active = aligned.loc[aligned["position"] > 0].dropna(subset=["target_direction", "prob_up"]).copy()
    if active.empty:
        return {
            "samples": 0,
            "brier_score": np.nan,
            "avg_prob_up": np.nan,
            "realized_up": np.nan,
            "calibration_error": np.nan,
            "mean_abs_calibration_error": np.nan,
            "max_abs_bin_error": np.nan,
            "bins": [],
        }

    y = active["target_direction"].astype(float).clip(0.0, 1.0)
    p = active["prob_up"].astype(float).clip(0.01, 0.99)
    bin_count = max(2, int(bins))
    active["calibration_bin"] = pd.cut(p, bins=np.linspace(0.50, 0.90, bin_count + 1), include_lowest=True)
    grouped = active.assign(prob_up=p, target_direction=y).groupby("calibration_bin", observed=False)
    bin_rows = []
    abs_errors = []
    for bin_label, group in grouped:
        if group.empty:
            continue
        avg_prob = float(group["prob_up"].mean())
        realized = float(group["target_direction"].mean())
        error = avg_prob - realized
        abs_errors.append(abs(error))
        bin_rows.append(
            {
                "bin": str(bin_label),
                "samples": int(len(group)),
                "avg_prob_up": avg_prob,
                "realized_up": realized,
                "calibration_error": error,
            }
        )

    avg_prob = float(p.mean())
    realized_up = float(y.mean())
    return {
        "samples": int(len(active)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "avg_prob_up": avg_prob,
        "realized_up": realized_up,
        "calibration_error": avg_prob - realized_up,
        "mean_abs_calibration_error": float(np.mean(abs_errors)) if abs_errors else np.nan,
        "max_abs_bin_error": float(max(abs_errors)) if abs_errors else np.nan,
        "bins": bin_rows,
    }


def rolling_interval_calibration(
    close: pd.Series,
    horizon_days: int = 10,
    train_window: int = 252,
    step: int = 21,
    confidence_levels: Iterable[float] = (0.80, 0.95),
    volatility_mode: str = "ewma",
    ewma_span: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest rolling log-return interval coverage for NVDA volatility assumptions."""

    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    prices = prices[prices > 0]
    if horizon_days <= 0 or train_window <= 5 or step <= 0:
        raise ValueError("horizon_days, train_window, and step must be positive.")
    if prices.shape[0] <= train_window + horizon_days:
        raise ValueError("Not enough prices for rolling interval calibration.")
    levels = tuple(float(level) for level in confidence_levels)
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("confidence_levels must be between 0 and 1.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rows: list[dict[str, Any]] = []
    normal = NormalDist()
    for as_of_pos in range(train_window, prices.shape[0] - horizon_days, step):
        as_of_date = prices.index[as_of_pos]
        train_returns = log_returns.loc[:as_of_date].tail(train_window)
        train_returns = train_returns[np.isfinite(train_returns)]
        if train_returns.shape[0] < max(30, min(train_window, 60)):
            continue
        if volatility_mode == "ewma":
            daily_sigma = float(train_returns.ewm(span=ewma_span, adjust=False).std(bias=False).iloc[-1])
        elif volatility_mode == "rolling":
            daily_sigma = float(train_returns.std(ddof=1))
        else:
            raise ValueError("volatility_mode must be 'ewma' or 'rolling'.")
        daily_mu = float(train_returns.tail(min(60, len(train_returns))).mean())
        if not np.isfinite(daily_sigma) or daily_sigma <= 0:
            continue

        terminal_date = prices.index[as_of_pos + horizon_days]
        actual_log_return = float(np.log(prices.iloc[as_of_pos + horizon_days] / prices.iloc[as_of_pos]))
        row: dict[str, Any] = {
            "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
            "terminal_date": pd.Timestamp(terminal_date).strftime("%Y-%m-%d"),
            "actual_log_return": actual_log_return,
            "daily_mu": daily_mu,
            "daily_sigma": daily_sigma,
            "volatility_mode": volatility_mode,
            "horizon_days": horizon_days,
        }
        forecast_mean = daily_mu * horizon_days
        forecast_sigma = daily_sigma * np.sqrt(horizon_days)
        for level in levels:
            z = normal.inv_cdf(0.5 + level / 2.0)
            low = forecast_mean - z * forecast_sigma
            high = forecast_mean + z * forecast_sigma
            key = int(round(level * 100))
            row[f"ci_{key}_low"] = low
            row[f"ci_{key}_high"] = high
            row[f"inside_{key}"] = bool(low <= actual_log_return <= high)
        rows.append(row)

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No rolling calibration windows could be evaluated.")

    summary_rows = []
    for level in levels:
        key = int(round(level * 100))
        coverage = float(windows[f"inside_{key}"].mean())
        summary_rows.append(
            {
                "confidence_level": level,
                "coverage": coverage,
                "coverage_error": coverage - level,
                "n_windows": int(len(windows)),
                "horizon_days": horizon_days,
                "train_window": train_window,
                "step": step,
                "volatility_mode": volatility_mode,
            }
        )
    return pd.DataFrame(summary_rows), windows
