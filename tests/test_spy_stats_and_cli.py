"""Tests for SPY forecast statistics and CLI guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from spy_forecast import build_parser
from utils.stats import compute_metrics, comparison_frame, make_price_paths


def test_cli_rejects_non_positive_simulation_counts() -> None:
    """Invalid positive-only CLI arguments should fail before a forecast starts."""

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--n-sims", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--horizon-days", "-5"])


def test_compute_metrics_uses_fixed_worst_tail_for_cvar() -> None:
    """CVaR should average the worst fixed tail count rather than equality matches."""

    terminal_returns = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    increments = np.log1p(terminal_returns)[:, None]
    paths = make_price_paths(100.0, increments)

    metrics = compute_metrics(paths, spot=100.0, thresholds=[0.0])

    assert metrics["var_95"] == pytest.approx(-0.30)
    assert metrics["cvar_95"] == pytest.approx(-0.30)
    assert metrics["var_99"] == pytest.approx(-0.30)
    assert metrics["cvar_99"] == pytest.approx(-0.30)


def test_comparison_divergence_flag_is_broadcast_column() -> None:
    """Divergence flags should be explicit broadcast columns, not hidden scalar state."""

    class DummyResult:
        def __init__(self, name: str, p_up: float) -> None:
            self.name = name
            self.metrics = {
                "p_up": p_up,
                "expected_return": 0.01,
                "return_std": 0.02,
                "var_95": -0.04,
            }

    frame = comparison_frame([DummyResult("A", 0.45), DummyResult("B", 0.60)])  # type: ignore[list-item]

    assert frame["P(up)_diverges_gt_5pp"].tolist() == [True, True]
