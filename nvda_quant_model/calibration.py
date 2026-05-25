from __future__ import annotations

from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from methods.historical import run_historical_bootstrap
from methods.monte_carlo import run_monte_carlo_methods


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
    hac_lags: int | None = None,
    block_size: int = 5,
    engine_methods: Iterable[str] = ("historical_bootstrap", "gbm_normal", "gbm_student_t"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate NVDA terminal-return simulation engines on rolling historical as-of dates.

    The PIT value is the fraction of simulated terminal simple returns below the
    realized terminal simple return. A calibrated engine should have PIT values
    centered near 0.5 and interval coverage close to the requested confidence
    levels. The returned window rows are per engine method, not a normal proxy.
    """

    prices = _coerce_close_series(close)
    if horizon_days <= 0 or train_window_years <= 0 or step <= 0 or n_sims <= 100 or block_size <= 0:
        raise ValueError("horizon_days, train_window_years, step, n_sims, and block_size must be positive.")
    levels = tuple(float(level) for level in confidence_levels)
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("confidence_levels must be between 0 and 1.")
    requested_methods = tuple(dict.fromkeys(str(method) for method in engine_methods))
    valid_methods = {"historical_bootstrap", "gbm_normal", "gbm_student_t"}
    unknown_methods = set(requested_methods).difference(valid_methods)
    if unknown_methods:
        raise ValueError(f"Unsupported engine_methods: {sorted(unknown_methods)}")
    if not requested_methods:
        raise ValueError("At least one engine method is required.")

    train_window = int(round(train_window_years * 252))
    if prices.shape[0] <= train_window + horizon_days:
        raise ValueError("Not enough prices for engine rolling calibration.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    rows: list[dict[str, Any]] = []
    as_of_positions = list(range(train_window, prices.shape[0] - horizon_days, step))
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(max(len(as_of_positions) * 2, 1))

    for window_id, as_of_pos in enumerate(as_of_positions):
        as_of_date = prices.index[as_of_pos]
        train_returns = log_returns.loc[:as_of_date].tail(train_window)
        train_returns = train_returns[np.isfinite(train_returns)]
        if train_returns.shape[0] < max(60, int(train_window * 0.5)):
            continue
        daily_mu = float(train_returns.mean())
        daily_sigma = float(train_returns.std(ddof=1))
        if not np.isfinite(daily_mu) or not np.isfinite(daily_sigma) or daily_sigma <= 0.0:
            continue

        spot = float(prices.iloc[as_of_pos])
        terminal_date = prices.index[as_of_pos + horizon_days]
        actual_simple_return = float(prices.iloc[as_of_pos + horizon_days] / spot - 1.0)
        actual_log_return = float(np.log(prices.iloc[as_of_pos + horizon_days] / spot))
        rng_hist = np.random.default_rng(child_seeds[window_id * 2])
        rng_mc = np.random.default_rng(child_seeds[window_id * 2 + 1])

        method_results: dict[str, Any] = {}
        if "historical_bootstrap" in requested_methods:
            method_results["historical_bootstrap"] = run_historical_bootstrap(
                returns=train_returns.to_numpy(dtype=float),
                spot=spot,
                horizon_days=int(horizon_days),
                n_sims=int(n_sims),
                block_size=int(block_size),
                rng=rng_hist,
                thresholds=[0.0],
                lookback_years=float(train_window_years),
            )
        if "gbm_normal" in requested_methods or "gbm_student_t" in requested_methods:
            gbm_normal, gbm_student_t = run_monte_carlo_methods(
                returns=train_returns.to_numpy(dtype=float),
                spot=spot,
                horizon_days=int(horizon_days),
                n_sims=int(n_sims),
                rng=rng_mc,
                thresholds=[0.0],
                lookback_years=float(train_window_years),
            )
            if "gbm_normal" in requested_methods:
                method_results["gbm_normal"] = gbm_normal
            if "gbm_student_t" in requested_methods:
                method_results["gbm_student_t"] = gbm_student_t

        for method_name in requested_methods:
            result = method_results.get(method_name)
            if result is None:
                continue
            simulated_terminal_returns = np.asarray(result.terminal_returns, dtype=float)
            simulated_terminal_returns = simulated_terminal_returns[np.isfinite(simulated_terminal_returns)]
            if simulated_terminal_returns.size <= 1:
                continue
            pit = float(np.mean(simulated_terminal_returns <= actual_simple_return))
            row: dict[str, Any] = {
                "method": method_name,
                "method_label": result.name,
                "method_detail": result.extras.get("method_detail", method_name),
                "as_of_date": pd.Timestamp(as_of_date).strftime("%Y-%m-%d"),
                "terminal_date": pd.Timestamp(terminal_date).strftime("%Y-%m-%d"),
                "spot": spot,
                "actual_simple_return": actual_simple_return,
                "actual_log_return": actual_log_return,
                "simulated_mean_return": float(np.mean(simulated_terminal_returns)),
                "simulated_median_return": float(np.median(simulated_terminal_returns)),
                "simulated_std_return": float(np.std(simulated_terminal_returns, ddof=1)),
                "pit": pit,
                "pit_centered": pit - 0.5,
                "daily_mu": daily_mu,
                "daily_sigma": daily_sigma,
                "annualized_mu": daily_mu * 252,
                "annualized_sigma": daily_sigma * np.sqrt(252),
                "horizon_days": int(horizon_days),
                "train_window_days": int(train_window),
                "n_sims": int(n_sims),
                "block_size": int(block_size),
                "engine_source": "simulation_engine",
                "student_t_df": result.extras.get("student_t_df"),
            }
            for level in levels:
                low_q = (1.0 - level) / 2.0
                high_q = 1.0 - low_q
                low, high = np.quantile(simulated_terminal_returns, [low_q, high_q])
                key = int(round(level * 100))
                row[f"ci_{key}_low"] = float(low)
                row[f"ci_{key}_high"] = float(high)
                row[f"ci_{key}_width"] = float(high - low)
                row[f"inside_{key}"] = bool(low <= actual_simple_return <= high)
                row[f"below_{key}"] = bool(actual_simple_return < low)
                row[f"above_{key}"] = bool(actual_simple_return > high)
            rows.append(row)

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("No engine calibration windows could be evaluated.")

    summary_rows: list[dict[str, Any]] = []
    effective_hac_lags = max(0, int(hac_lags if hac_lags is not None else np.ceil(horizon_days / step)))
    for method_name in requested_methods:
        method_windows = windows.loc[windows["method"] == method_name]
        if method_windows.empty:
            continue
        pit_values = method_windows["pit"].to_numpy(dtype=float)
        pit_ks = stats.kstest(pit_values, "uniform")
        pit_mean = float(np.mean(pit_values))
        pit_mean_error = pit_mean - 0.5
        pit_naive_se = float(np.std(pit_values, ddof=1) / np.sqrt(len(pit_values))) if len(pit_values) > 1 else np.nan
        pit_hac_se = _newey_west_mean_se(pit_values, effective_hac_lags)
        for level in levels:
            key = int(round(level * 100))
            hits = method_windows[f"inside_{key}"].astype(float).to_numpy()
            coverage = float(np.mean(hits))
            coverage_error = coverage - level
            naive_se = float(np.sqrt(max(level * (1.0 - level), 0.0) / len(hits)))
            hac_se = _newey_west_mean_se(hits, effective_hac_lags)
            summary_rows.append(
                {
                    "method": method_name,
                    "method_label": str(method_windows["method_label"].iloc[0]),
                    "confidence_level": level,
                    "expected_coverage": level,
                    "actual_coverage": coverage,
                    "coverage_error": coverage_error,
                    "coverage_naive_se": naive_se,
                    "coverage_hac_se": hac_se,
                    "coverage_z_hac": coverage_error / hac_se if hac_se and np.isfinite(hac_se) and hac_se > 0 else np.nan,
                    "below_rate": float(method_windows[f"below_{key}"].mean()),
                    "above_rate": float(method_windows[f"above_{key}"].mean()),
                    "avg_interval_width": float(method_windows[f"ci_{key}_width"].mean()),
                    "pit_mean": pit_mean,
                    "pit_mean_error": pit_mean_error,
                    "pit_naive_se": pit_naive_se,
                    "pit_hac_se": pit_hac_se,
                    "pit_z_hac": pit_mean_error / pit_hac_se if pit_hac_se and np.isfinite(pit_hac_se) and pit_hac_se > 0 else np.nan,
                    "pit_std": float(np.std(pit_values, ddof=0)),
                    "pit_ks_stat": float(pit_ks.statistic),
                    "pit_ks_pvalue": float(pit_ks.pvalue),
                    "n_windows": int(len(method_windows)),
                    "hac_lags": effective_hac_lags,
                    "horizon_days": int(horizon_days),
                    "train_window_years": float(train_window_years),
                    "train_window_days": int(train_window),
                    "n_sims": int(n_sims),
                    "block_size": int(block_size),
                    "engine_source": "simulation_engine",
                }
            )
    return pd.DataFrame(summary_rows), windows


def _newey_west_mean_se(values: np.ndarray | pd.Series, lags: int) -> float:
    """Estimate Newey-West standard error for a sample mean."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = array.size
    if n <= 1:
        return np.nan
    centered = array - np.mean(array)
    max_lag = min(max(int(lags), 0), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    return float(np.sqrt(max(long_run_var, 0.0) / n))


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
