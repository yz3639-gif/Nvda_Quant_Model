"""Matplotlib visualizations for forecast distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils.data import ensure_directory
from utils.stats import ScenarioResult


def plot_histogram_overlay(results: list[ScenarioResult], output_dir: Path) -> Path:
    """Save an overlaid histogram of terminal return distributions."""

    ensure_directory(output_dir)
    path = output_dir / "terminal_return_histogram.png"
    plt.figure(figsize=(11, 7))
    for result in results:
        plt.hist(
            result.terminal_returns * 100.0,
            bins=80,
            density=True,
            alpha=0.35,
            label=result.name,
        )
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    plt.title("Terminal Return Distribution Overlay")
    plt.xlabel("Terminal return (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_fan_chart(results: list[ScenarioResult], output_dir: Path) -> Path:
    """Save price-path fan charts for each method."""

    ensure_directory(output_dir)
    path = output_dir / "price_path_fan_chart.png"
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.8 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    x_values = np.arange(results[0].paths.shape[1])
    for axis, result in zip(axes, results):
        q025, q10, q25, q50, q75, q90, q975 = np.quantile(
            result.paths,
            [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975],
            axis=0,
        )
        axis.fill_between(x_values, q025, q975, alpha=0.16, label="95% band")
        axis.fill_between(x_values, q10, q90, alpha=0.24, label="80% band")
        axis.fill_between(x_values, q25, q75, alpha=0.34, label="50% band")
        axis.plot(x_values, q50, color="black", linewidth=1.4, label="Median")
        axis.set_title(result.name)
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Trading days from start")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cdf_comparison(results: list[ScenarioResult], output_dir: Path) -> Path:
    """Save a cumulative distribution function comparison chart."""

    ensure_directory(output_dir)
    path = output_dir / "terminal_return_cdf.png"
    plt.figure(figsize=(11, 7))
    for result in results:
        sorted_returns = np.sort(result.terminal_returns * 100.0)
        cdf = np.arange(1, sorted_returns.size + 1) / sorted_returns.size
        plt.plot(sorted_returns, cdf, linewidth=1.8, label=result.name)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    plt.title("Terminal Return CDF Comparison")
    plt.xlabel("Terminal return (%)")
    plt.ylabel("Cumulative probability")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_sample_paths(
    results: list[ScenarioResult],
    output_dir: Path,
    rng: np.random.Generator,
    n_paths: int = 100,
) -> Path:
    """Save random sample path charts for each method."""

    ensure_directory(output_dir)
    path = output_dir / "sample_price_paths.png"
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.8 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    x_values = np.arange(results[0].paths.shape[1])
    for axis, result in zip(axes, results):
        count = min(n_paths, result.paths.shape[0])
        indices = rng.choice(result.paths.shape[0], size=count, replace=False)
        axis.plot(x_values, result.paths[indices].T, color="#1f77b4", alpha=0.12, linewidth=0.8)
        axis.plot(x_values, np.median(result.paths, axis=0), color="black", linewidth=1.5)
        axis.set_title(result.name)
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Trading days from start")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_risk_neutral_vs_historical_density(
    historical_result: ScenarioResult,
    option_result: ScenarioResult,
    output_dir: Path,
) -> Path:
    """Save a risk-neutral versus historical terminal-return density chart."""

    ensure_directory(output_dir)
    path = output_dir / "risk_neutral_vs_historical_density.png"
    plt.figure(figsize=(11, 7))
    _plot_kde_or_hist(historical_result.terminal_returns * 100.0, "Historical bootstrap")
    _plot_kde_or_hist(option_result.terminal_returns * 100.0, "Option-implied risk-neutral")
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
    plt.title("Risk-Neutral vs Historical Density")
    plt.xlabel("Terminal return (%)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_all(
    results: list[ScenarioResult],
    output_dir: Path,
    rng: np.random.Generator,
) -> list[Path]:
    """Save all required forecast charts and return their paths."""

    if not results:
        raise ValueError("At least one result is required to plot forecast charts.")
    historical_result = results[0]
    option_result = next((result for result in results if result.name.startswith("Option-Implied")), results[-1])
    return [
        plot_histogram_overlay(results, output_dir),
        plot_fan_chart(results, output_dir),
        plot_cdf_comparison(results, output_dir),
        plot_sample_paths(results, output_dir, rng),
        plot_risk_neutral_vs_historical_density(historical_result, option_result, output_dir),
    ]


def _plot_kde_or_hist(values: np.ndarray, label: str) -> None:
    """Plot a KDE when possible, otherwise a normalized histogram."""

    clean_values = values[np.isfinite(values)]
    if clean_values.size < 3 or np.std(clean_values) <= 0.0:
        plt.hist(clean_values, bins=40, density=True, alpha=0.35, label=label)
        return
    try:
        kde = stats.gaussian_kde(clean_values)
        grid = np.linspace(np.quantile(clean_values, 0.005), np.quantile(clean_values, 0.995), 400)
        plt.plot(grid, kde(grid), linewidth=2.0, label=label)
    except Exception:
        plt.hist(clean_values, bins=60, density=True, alpha=0.35, label=label)
