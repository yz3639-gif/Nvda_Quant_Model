from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nvda_quant_model.backtest.metrics import max_drawdown


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_equity_curve(strategy_equity: pd.Series, benchmark_equity: pd.Series, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    strategy_norm = strategy_equity / strategy_equity.iloc[0]
    benchmark_norm = benchmark_equity / benchmark_equity.iloc[0]
    ax.plot(strategy_norm.index, strategy_norm, label="Strategy", linewidth=2)
    ax.plot(benchmark_norm.index, benchmark_norm, label="S&P 500", linewidth=1.6, alpha=0.8)
    ax.set_title("Strategy Equity vs Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, output_dir / "equity_curve.png")


def plot_drawdown(strategy_equity: pd.Series, benchmark_equity: pd.Series, output_dir: Path) -> Path:
    _, dd_strategy = max_drawdown(strategy_equity)
    _, dd_bench = max_drawdown(benchmark_equity)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd_strategy.index, dd_strategy * 100, 0, label="Strategy", alpha=0.35)
    ax.plot(dd_bench.index, dd_bench * 100, label="S&P 500", linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, output_dir / "drawdown.png")


def plot_monthly_heatmap(strategy_returns: pd.Series, output_dir: Path) -> Path:
    monthly = (1 + strategy_returns).resample("ME").prod() - 1
    heat = monthly.to_frame("return")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    table = heat.pivot(index="year", columns="month", values="return") * 100
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Monthly Returns (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return _save(fig, output_dir / "monthly_returns_heatmap.png")


def plot_rolling_sharpe(strategy_returns: pd.Series, output_dir: Path, window: int = 63) -> Path:
    rolling = strategy_returns.rolling(window).mean() / strategy_returns.rolling(window).std()
    rolling = rolling * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rolling.index, rolling, linewidth=1.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"Rolling {window}-Day Sharpe")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir / "rolling_sharpe.png")


def create_all_charts(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    strategy_returns: pd.Series,
    output_dir: Path,
) -> dict[str, Path]:
    return {
        "equity_curve": plot_equity_curve(strategy_equity, benchmark_equity, output_dir),
        "drawdown": plot_drawdown(strategy_equity, benchmark_equity, output_dir),
        "monthly_returns": plot_monthly_heatmap(strategy_returns, output_dir),
        "rolling_sharpe": plot_rolling_sharpe(strategy_returns, output_dir),
    }

