#!/usr/bin/env python3
"""Render README evidence from the frozen CSVs, without running new experiments.

Run from any directory:
    .venv-research/bin/python scripts/build_readme_figures.py

The two panels concern separate saved experiments and retain their own dates.
All 17 strategy/control rows are shown in source order, not selected by outcome.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/results/frozen_20260918"
OUT = ROOT / "docs/assets"
INK = "#192c3f"
MUTED = "#526478"
GRID = "#dce4eb"
BLUE = "#4c7193"
ORANGE = "#b97750"
TEAL = "#487f81"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build() -> None:
    replay = read_rows(RUN / "historical_replay/historical_replay.csv")
    replay = [
        row for row in replay
        if row["family"] == "strict_baseline" and row["lookback_months"] == "60"
    ]
    variants = [
        "legacy_close_daily_reset",
        "corrected_close_daily_reset",
        "corrected_next_open_entry",
    ]
    by_variant = {row["variant"]: row for row in replay}
    if set(by_variant) != set(variants) or len(replay) != 3:
        raise ValueError("Expected exactly three strict 60-month replay variants")
    if len({(row["date_start"], row["date_end"]) for row in replay}) != 1:
        raise ValueError("Replay comparison must use a shared evaluation period")
    returns = [float(by_variant[key]["annualized_return"]) for key in variants]

    strategies = read_rows(RUN / "strategy_summary.csv")
    if len(strategies) != 17 or len({row["model"] for row in strategies}) != 17:
        raise ValueError("Expected the complete 17-row strategy/control register")
    if len({(row["evaluation_first"], row["evaluation_last"]) for row in strategies}) != 1:
        raise ValueError("Strategy rows must share the saved evaluation dates")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.hashsalt": "nvda-frozen-research-evidence-v1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })
    fig = plt.figure(figsize=(16, 10.3), facecolor="white")
    fig.text(0.045, 0.95, "Execution assumptions and strategy evidence",
             fontsize=24, weight="bold", ha="left")
    fig.text(0.045, 0.909,
             "NVDA  /  Saved run: 2026-09-18  /  Historical development evaluation",
             fontsize=14, color=MUTED)
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.882, 0.882],
                             transform=fig.transFigure, color=GRID, linewidth=1.2))

    fig.text(0.045, 0.838, "01  Same frozen signals, different execution",
             fontsize=16, weight="bold")
    replay_first = by_variant[variants[0]]["date_start"]
    replay_last = by_variant[variants[0]]["date_end"]
    fig.text(0.045, 0.803, f"60-month replay  |  {replay_first} to {replay_last}",
             fontsize=12.5, color=MUTED)

    left = fig.add_axes([0.082, 0.385, 0.355, 0.365])
    positions = range(3)
    left.bar(positions, returns, width=0.52,
             color=[BLUE, BLUE, ORANGE], zorder=3)
    left.axhline(0, color=INK, linewidth=1.1, zorder=2)
    left.set_ylim(-0.035, 0.205)
    left.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_ylabel("Annualized return", labelpad=12, fontsize=13)
    left.set_xticks(list(positions), [
        "Legacy\nClose fill\nDaily-reset stop",
        "Corrected accounting\nClose fill\nDaily-reset stop",
        "Corrected accounting\nNext-open fill\nEntry-reference stop",
    ], fontsize=11.3, linespacing=1.6)
    left.tick_params(axis="both", length=0, pad=10)
    left.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for pos, value in zip(positions, returns):
        left.text(pos, value + (0.009 if value >= 0 else -0.007),
                  f"{value:.2%}", ha="center",
                  va="bottom" if value >= 0 else "top",
                  fontsize=18, weight="bold", color=INK)

    fig.text(0.045, 0.23, "What this comparison establishes", fontsize=14,
             weight="bold")
    fig.text(0.045, 0.193,
             "Results are sensitive to execution assumptions.\n"
             "The next-open case changes fill timing and stop\n"
             "semantics together; it is not a pure accounting fix.",
             fontsize=12.7, color=MUTED, va="top", linespacing=1.7)

    fig.text(0.515, 0.838, "02  Every saved strategy and control",
             fontsize=16, weight="bold")
    first = strategies[0]["evaluation_first"]
    last = strategies[0]["evaluation_last"]
    fig.text(0.515, 0.803, f"Net Sharpe  |  {first} to {last}",
             fontsize=12.5, color=MUTED)

    suffix_names = {
        "raw": "raw",
        "calibrated": "calibrated",
        "vol_target": "volatility target",
        "fixed_risk_control": "fixed-risk control",
        "exposure_control": "exposure control",
    }

    def label(model: str) -> str:
        if model == "buy_hold":
            return "Buy & hold benchmark"
        if model == "trend":
            return "Trend benchmark"
        family, variant = model.split("_", 1)
        return f"{family.title()} / {suffix_names[variant]}"

    sharpes = [float(row["sharpe_ratio"]) for row in strategies]
    right = fig.add_axes([0.727, 0.215, 0.214, 0.542])
    right.barh(range(17), sharpes, height=0.63,
               color=[TEAL if value > 0 else BLUE for value in sharpes], zorder=3)
    right.set_yticks(range(17), [label(row["model"]) for row in strategies],
                    fontsize=11.6)
    right.invert_yaxis()
    right.set_xlim(-1.25, 1.57)
    right.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5])
    right.set_xlabel("Net Sharpe ratio", labelpad=9, fontsize=12)
    right.tick_params(axis="both", length=0, pad=8)
    right.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    right.axvline(0, color=INK, linewidth=1.0, zorder=2)
    for ypos in (4.5, 9.5, 14.5):
        right.axhline(ypos, color=GRID, linewidth=0.8, zorder=1)
    for ypos, value in enumerate(sharpes):
        right.text(value + (0.04 if value >= 0 else -0.04), ypos,
                   f"{value:.2f}", va="center",
                   ha="left" if value >= 0 else "right", fontsize=11.5)
    fig.text(0.515, 0.129,
             "All 17 rows shown in source order. Baseline costs included.\n"
             "Exposure and risk halts differ; Sharpe alone is not a ranking.",
             fontsize=11.8, color=MUTED, va="top", linespacing=1.6)

    fig.add_artist(plt.Line2D([0.045, 0.955], [0.071, 0.071],
                             transform=fig.transFigure, color=GRID, linewidth=1.2))
    fig.text(0.045, 0.037,
             "Previously researched data; not an untouched holdout or live performance. "
             "Panels are separate experiments with different evaluation periods.",
             fontsize=11.1, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = OUT / f"research_evidence.{extension}"
        metadata = {"Date": None} if extension == "svg" else {"Software": "NVDA Research"}
        fig.savefig(target, dpi=160, facecolor="white", metadata=metadata)
        print(target.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    build()
