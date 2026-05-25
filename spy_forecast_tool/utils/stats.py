"""Statistical helpers and common result structures for forecast methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


CONFIDENCE_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99)
BARRIER_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)


@dataclass(slots=True)
class ScenarioResult:
    """Container for simulated paths, terminal returns, metrics, and metadata."""

    name: str
    paths: np.ndarray
    terminal_returns: np.ndarray
    terminal_prices: np.ndarray
    metrics: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    is_primary: bool = True


def parse_thresholds(raw: str) -> list[float]:
    """Parse comma-separated percentage thresholds into decimal returns."""

    values = [float(item.strip()) / 100.0 for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold is required.")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Thresholds must be finite numbers.")
    return sorted(set(values))


def format_pct(value: float, precision: int = 2) -> str:
    """Format a decimal return or probability as a percentage string."""

    return f"{value * 100:.{precision}f}%"


def central_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return the central interval for the requested confidence level."""

    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def make_price_paths(spot: float, log_returns: np.ndarray) -> np.ndarray:
    """Convert a matrix of forward log returns into price paths with spot at t=0."""

    cumulative = np.cumsum(log_returns, axis=1)
    future_paths = spot * np.exp(cumulative)
    return np.concatenate(
        [np.full((log_returns.shape[0], 1), spot, dtype=float), future_paths],
        axis=1,
    )


def terminal_returns_from_paths(paths: np.ndarray, spot: float) -> np.ndarray:
    """Compute terminal simple returns from simulated price paths."""

    return (paths[:, -1] / spot) - 1.0


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Compute each path's maximum drawdown as a negative decimal return."""

    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths / running_max) - 1.0
    return np.min(drawdowns, axis=1)


def bucket_probabilities(returns: np.ndarray, thresholds: Iterable[float]) -> dict[str, float]:
    """Compute probabilities for return buckets defined by sorted thresholds."""

    edges = list(sorted(thresholds))
    probabilities: dict[str, float] = {}
    lower = -np.inf
    for upper in edges:
        label = f"{_edge_label(lower)} to {_edge_label(upper)}"
        probabilities[label] = float(np.mean((returns > lower) & (returns <= upper)))
        lower = upper
    probabilities[f"{_edge_label(lower)} to +inf"] = float(np.mean(returns > lower))
    return probabilities


def exceedance_probabilities(returns: np.ndarray, thresholds: Iterable[float]) -> dict[str, float]:
    """Compute probabilities that terminal returns exceed each threshold."""

    return {f"P(return > {format_pct(t)})": float(np.mean(returns > t)) for t in thresholds}


def barrier_probabilities(paths: np.ndarray, spot: float) -> dict[str, float]:
    """Compute path-dependent probabilities of touching +/- barrier levels."""

    probabilities: dict[str, float] = {}
    for level in BARRIER_LEVELS:
        upper = spot * (1.0 + level)
        lower = spot * (1.0 - level)
        probabilities[f"touch +{format_pct(level, 0)}"] = float(np.mean(np.any(paths >= upper, axis=1)))
        probabilities[f"touch -{format_pct(level, 0)}"] = float(np.mean(np.any(paths <= lower, axis=1)))
    return probabilities


def compute_metrics(paths: np.ndarray, spot: float, thresholds: Iterable[float]) -> dict[str, Any]:
    """Compute terminal, tail-risk, interval, drawdown, and barrier statistics."""

    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2:
        raise ValueError("Metrics require at least two simulated paths with one forward step.")
    if not np.isfinite(paths).all():
        raise ValueError("Simulated paths contain NaN or infinite values.")
    terminal_prices = paths[:, -1]
    terminal_returns = (terminal_prices / spot) - 1.0
    mdd = max_drawdowns(paths)
    return_intervals = {
        f"{int(level * 100)}%": {
            "low": central_interval(terminal_returns, level)[0],
            "high": central_interval(terminal_returns, level)[1],
        }
        for level in CONFIDENCE_LEVELS
    }
    price_intervals = {
        f"{int(level * 100)}%": {
            "low": central_interval(terminal_prices, level)[0],
            "high": central_interval(terminal_prices, level)[1],
        }
        for level in CONFIDENCE_LEVELS
    }
    var_95, cvar_95 = _var_cvar(terminal_returns, 0.05)
    var_99, cvar_99 = _var_cvar(terminal_returns, 0.01)
    mdd_loss = -mdd
    return {
        "p_up": float(np.mean(terminal_returns > 0.0)),
        "p_down": float(np.mean(terminal_returns < 0.0)),
        "bucket_probabilities": bucket_probabilities(terminal_returns, thresholds),
        "exceedance_probabilities": exceedance_probabilities(terminal_returns, thresholds),
        "expected_return": float(np.mean(terminal_returns)),
        "median_return": float(np.median(terminal_returns)),
        "return_std": float(np.std(terminal_returns, ddof=1)),
        "skewness": float(stats.skew(terminal_returns, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(terminal_returns, fisher=True, bias=False)),
        "return_confidence_intervals": return_intervals,
        "price_confidence_intervals": price_intervals,
        "var_95": var_95,
        "var_99": var_99,
        "var_loss_95": float(max(0.0, -var_95)),
        "var_loss_99": float(max(0.0, -var_99)),
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "cvar_loss_95": float(max(0.0, -cvar_95)),
        "cvar_loss_99": float(max(0.0, -cvar_99)),
        "max_drawdown": {
            "mean": float(np.mean(mdd)),
            "median": float(np.median(mdd)),
            "p95_worst_loss": float(np.quantile(mdd_loss, 0.95)),
        },
        "barrier_probabilities": barrier_probabilities(paths, spot),
        "terminal_price_mean": float(np.mean(terminal_prices)),
        "terminal_price_median": float(np.median(terminal_prices)),
    }


def build_result(
    name: str,
    paths: np.ndarray,
    spot: float,
    thresholds: Iterable[float],
    assumptions: list[str],
    warnings: list[str] | None = None,
    extras: dict[str, Any] | None = None,
    is_primary: bool = True,
) -> ScenarioResult:
    """Build a fully populated ScenarioResult from simulated paths."""

    terminal_returns = terminal_returns_from_paths(paths, spot)
    terminal_prices = paths[:, -1]
    metrics = compute_metrics(paths, spot, thresholds)
    return ScenarioResult(
        name=name,
        paths=paths,
        terminal_returns=terminal_returns,
        terminal_prices=terminal_prices,
        metrics=metrics,
        assumptions=assumptions,
        warnings=warnings or [],
        extras=extras or {},
        is_primary=is_primary,
    )


def build_ensemble_metrics(results: list[ScenarioResult]) -> dict[str, Any]:
    """Build equal-weight average metrics across primary methods."""

    if not results:
        raise ValueError("At least one result is required to build ensemble metrics.")
    return _average_metric_values([result.metrics for result in results])


def metrics_to_records(result: ScenarioResult | tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested metrics into row records for CSV output."""

    if isinstance(result, ScenarioResult):
        method = result.name
        metrics = result.metrics
    else:
        method, metrics = result
    return [
        {"method": method, "metric": key, "value": value}
        for key, value in _flatten_dict(metrics).items()
    ]


def comparison_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    """Create a compact comparison table with divergence flags."""

    rows = []
    for result in results:
        mean = result.metrics["expected_return"]
        std = result.metrics["return_std"]
        rows.append(
            {
                "method": result.name,
                "P(up)": result.metrics["p_up"],
                "expected_return": mean,
                "VaR_95_return": result.metrics["var_95"],
                "one_sigma_low": mean - std,
                "one_sigma_high": mean + std,
            }
        )
    frame = pd.DataFrame(rows)
    numeric_cols = ["P(up)", "expected_return", "VaR_95_return", "one_sigma_low", "one_sigma_high"]
    for col in numeric_cols:
        has_divergence = bool(frame[col].max() - frame[col].min() > 0.05)
        frame[f"{col}_diverges_gt_5pp"] = has_divergence
    return frame


def format_comparison_for_terminal(frame: pd.DataFrame) -> str:
    """Format the comparison table with markers for columns diverging by more than 5pp."""

    display = frame.copy()
    marker_cols = {
        col.replace("_diverges_gt_5pp", "")
        for col in display.columns
        if col.endswith("_diverges_gt_5pp") and bool(display[col].iloc[0])
    }
    for col in ["P(up)", "expected_return", "VaR_95_return", "one_sigma_low", "one_sigma_high"]:
        display[col] = display[col].map(lambda value: format_pct(float(value)))
        if col in marker_cols:
            display[col] = display[col] + " *"
    display = display[[col for col in display.columns if not col.endswith("_diverges_gt_5pp")]]
    return display.to_string(index=False)


def json_safe(value: Any) -> Any:
    """Convert NumPy, pandas, and nested objects into JSON-serializable values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _var_cvar(terminal_returns: np.ndarray, tail_probability: float) -> tuple[float, float]:
    """Compute left-tail VaR and CVaR using a fixed worst-tail sample count."""

    ordered = np.sort(np.asarray(terminal_returns, dtype=float))
    tail_count = max(1, int(np.ceil(ordered.size * tail_probability)))
    tail = ordered[:tail_count]
    var_index = min(tail_count - 1, ordered.size - 1)
    return float(ordered[var_index]), float(np.mean(tail))


def result_summary_for_json(result: ScenarioResult) -> dict[str, Any]:
    """Create a compact JSON-safe summary for a scenario result."""

    extras = {
        key: value
        for key, value in result.extras.items()
        if key
        in {
            "method_detail",
            "expiration",
            "days_to_expiration",
            "atm_iv",
            "risk_free_rate",
            "rnd_reconstructed",
            "risk_neutral_density_grid",
            "risk_neutral_density_values",
        }
    }
    return json_safe(
        {
            "name": result.name,
            "metrics": result.metrics,
            "assumptions": result.assumptions,
            "warnings": result.warnings,
            "extras": extras,
        }
    )


def _average_metric_values(values: list[Any]) -> Any:
    """Average a list of metric values recursively where possible."""

    first = values[0]
    if isinstance(first, dict):
        keys = set(first.keys())
        for value in values[1:]:
            keys &= set(value.keys())
        return {key: _average_metric_values([value[key] for value in values]) for key in sorted(keys)}
    if isinstance(first, (int, float, np.number)):
        return float(np.mean(values))
    return first


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary using dotted metric names."""

    records: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            records.update(_flatten_dict(value, full_key))
        else:
            records[full_key] = json_safe(value)
    return records


def _edge_label(value: float) -> str:
    """Format a bucket edge label."""

    if value == -np.inf:
        return "-inf"
    if value == np.inf:
        return "+inf"
    return format_pct(value)
