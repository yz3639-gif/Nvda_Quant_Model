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


def engine_rolling_calibration(
    close: pd.Series | pd.DataFrame,
    horizon_days: int = 21,
    train_window_years: float = 5.0,
    step: int = 21,
    n_sims: int = 5_000,
    confidence_levels: Iterable[float] = (0.50, 0.80, 0.95),
    seed: int = 42,
    volatility_mode: str = "rolling",
    ewma_span: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate the NVDA terminal-return simulation engine on rolling historical as-of dates.

    The PIT value is the fraction of simulated terminal log returns below the realized
    terminal log return. A calibrated engine should have PIT values centered near 0.5
    and interval coverage close to the requested confidence levels.
    """

    prices = _coerce_close_series(close)
    if horizon_days <= 0 or train_window_years <= 0 or step <= 0 or n_sims <= 100:
        raise ValueError("horizon_days, train_window_years, step, and n_sims must be positive.")
    levels = tuple(float(level) for level in confidence_levels)
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("confidence_levels must be between 0 and 1.")

    train_window = int(round(train_window_years * 252))
    if prices.shape[0] <= train_window + horizon_days:
        raise ValueError("Not enough prices for engine rolling calibration.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for as_of_pos in range(train_window, prices.shape[0] - horizon_days, step):
        as_of_date = prices.index[as_of_pos]
        train_returns = log_returns.loc[:as_of_date].tail(train_window)
        train_returns = train_returns[np.isfinite(train_returns)]
        if train_returns.shape[0] < max(60, int(train_window * 0.5)):
            continue
        daily_mu = float(train_returns.mean())
        if volatility_mode == "rolling":
            daily_sigma = float(train_returns.std(ddof=1))
        elif volatility_mode == "ewma":
            daily_sigma = float(train_returns.ewm(span=ewma_span, adjust=False).std(bias=False).iloc[-1])
        else:
            raise ValueError("volatility_mode must be 'rolling' or 'ewma'.")
        if not np.isfinite(daily_mu) or not np.isfinite(daily_sigma) or daily_sigma <= 0.0:
            continue

        shocks = rng.standard_normal(size=(int(n_sims), int(horizon_days)))
        simulated_terminal_returns = (daily_mu + daily_sigma * shocks).sum(axis=1)
        terminal_date = prices.index[as_of_pos + horizon_days]
        actual_log_return = float(np.log(prices.iloc[as_of_pos + horizon_days] / prices.iloc[as_of_pos]))
        pit = float(np.mean(simulated_terminal_returns <= actual_log_return))
        row: dict[str, Any] = {
            "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
            "terminal_date": pd.Timestamp(terminal_date).strftime("%Y-%m-%d"),
            "actual_log_return": actual_log_return,
            "simulated_mean_log_return": float(np.mean(simulated_terminal_returns)),
            "simulated_median_log_return": float(np.median(simulated_terminal_returns)),
            "simulated_std_log_return": float(np.std(simulated_terminal_returns, ddof=1)),
            "pit": pit,
            "pit_centered": pit - 0.5,
            "daily_mu": daily_mu,
            "daily_sigma": daily_sigma,
            "annualized_mu": daily_mu * 252,
            "annualized_sigma": daily_sigma * np.sqrt(252),
            "horizon_days": int(horizon_days),
            "train_window_days": int(train_window),
            "n_sims": int(n_sims),
            "volatility_mode": volatility_mode,
        }
        for level in levels:
            low_q = (1.0 - level) / 2.0
            high_q = 1.0 - low_q
            low, high = np.quantile(simulated_terminal_returns, [low_q, high_q])
            key = int(round(level * 100))
            row[f"ci_{key}_low"] = float(low)
            row[f"ci_{key}_high"] = float(high)
            row[f"ci_{key}_width"] = float(high - low)
            row[f"inside_{key}"] = bool(low <= actual_log_return <= high)
            row[f"below_{key}"] = bool(actual_log_return < low)
            row[f"above_{key}"] = bool(actual_log_return > high)
        rows.append(row)

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No engine calibration windows could be evaluated.")

    summary_rows: list[dict[str, Any]] = []
    for level in levels:
        key = int(round(level * 100))
        coverage = float(windows[f"inside_{key}"].mean())
        summary_rows.append(
            {
                "confidence_level": level,
                "expected_coverage": level,
                "actual_coverage": coverage,
                "coverage_error": coverage - level,
                "below_rate": float(windows[f"below_{key}"].mean()),
                "above_rate": float(windows[f"above_{key}"].mean()),
                "avg_interval_width": float(windows[f"ci_{key}_width"].mean()),
                "pit_mean": float(windows["pit"].mean()),
                "pit_mean_error": float(windows["pit"].mean() - 0.5),
                "pit_std": float(windows["pit"].std(ddof=0)),
                "n_windows": int(len(windows)),
                "horizon_days": int(horizon_days),
                "train_window_years": float(train_window_years),
                "train_window_days": int(train_window),
                "n_sims": int(n_sims),
                "volatility_mode": volatility_mode,
            }
        )
    return pd.DataFrame(summary_rows), windows


def _coerce_close_series(close: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize yfinance Series/DataFrame output to one adjusted-close Series."""

    if isinstance(close, pd.DataFrame):
        if isinstance(close.columns, pd.MultiIndex):
            if "Adj Close" in close.columns.get_level_values(0):
                close = close.xs("Adj Close", axis=1, level=0).iloc[:, 0]
            else:
                close = close.iloc[:, 0]
        elif "Adj Close" in close.columns:
            close = close["Adj Close"]
        elif "Close" in close.columns:
            close = close["Close"]
        else:
            close = close.iloc[:, 0]
    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices[prices > 0]
    prices.name = "adj_close"
    return prices
