from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT
from nvda_quant_model.production_model import BASELINE_PRODUCTION_LABEL, PRODUCTION_RULE_OVERRIDES


WINDOWS = (24, 36, 60)
THRESHOLDS = {
    "annualized_return": 0.15,
    "sharpe_ratio": 1.0,
    "max_abs_drawdown": 0.20,
    "win_rate": 0.55,
    "profit_factor": 1.50,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _safe_literal_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): _safe_float(value) for key, value in parsed.items()}


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def _baseline_row(summary: pd.DataFrame, label: str) -> pd.Series:
    rows = summary[summary["label"] == label]
    if rows.empty:
        rows = summary[summary["status"] == "current_baseline"]
    if rows.empty:
        raise ValueError(f"Baseline row not found: {label}")
    return rows.iloc[0]


def window_threshold_table(row: pd.Series) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for months in WINDOWS:
        annualized = _safe_float(row.get(f"m{months}_annualized_return"))
        sharpe = _safe_float(row.get(f"m{months}_sharpe_ratio"))
        max_drawdown = _safe_float(row.get(f"m{months}_max_drawdown"))
        win_rate = _safe_float(row.get(f"m{months}_win_rate"))
        profit_factor = _safe_float(row.get(f"m{months}_profit_factor"))
        num_trades = int(_safe_float(row.get(f"m{months}_num_trades")))
        dir_precision = _safe_float(row.get(f"m{months}_dir_precision"))
        brier_score = _safe_float(row.get(f"m{months}_brier_score"), np.nan)
        calibration_error = _safe_float(row.get(f"m{months}_calibration_error"), np.nan)
        records.append(
            {
                "lookback_months": months,
                "annualized_return": annualized,
                "annualized_margin": annualized - THRESHOLDS["annualized_return"],
                "sharpe_ratio": sharpe,
                "sharpe_margin": sharpe - THRESHOLDS["sharpe_ratio"],
                "max_drawdown": max_drawdown,
                "drawdown_margin": THRESHOLDS["max_abs_drawdown"] - abs(max_drawdown),
                "win_rate": win_rate,
                "win_rate_margin": win_rate - THRESHOLDS["win_rate"],
                "profit_factor": profit_factor,
                "profit_factor_margin": profit_factor - THRESHOLDS["profit_factor"],
                "num_trades": num_trades,
                "dir_precision": dir_precision,
                "brier_score": brier_score,
                "calibration_error": calibration_error,
                "passes_hard_gates": bool(
                    annualized >= THRESHOLDS["annualized_return"]
                    and sharpe >= THRESHOLDS["sharpe_ratio"]
                    and abs(max_drawdown) <= THRESHOLDS["max_abs_drawdown"]
                    and win_rate >= THRESHOLDS["win_rate"]
                    and profit_factor >= THRESHOLDS["profit_factor"]
                ),
            }
        )
    return pd.DataFrame(records)


def train_validation_table(meta_metrics: pd.DataFrame) -> pd.DataFrame:
    base = meta_metrics[(meta_metrics["layer"] == "baseline") & (meta_metrics["segment"].isin(["train", "validation", "full"]))].copy()
    records: list[dict[str, Any]] = []
    for months in WINDOWS:
        group = base[base["lookback_months"] == months]
        if group.empty:
            continue
        by_segment = {str(row["segment"]): row for _, row in group.iterrows()}
        train = by_segment.get("train")
        validation = by_segment.get("validation")
        full = by_segment.get("full")
        if train is None or validation is None or full is None:
            continue
        train_years = _safe_literal_dict(train.get("year_precision"))
        full_years = _safe_literal_dict(full.get("year_precision"))
        records.append(
            {
                "lookback_months": months,
                "train_annualized_return": _safe_float(train.get("annualized_return")),
                "validation_annualized_return": _safe_float(validation.get("annualized_return")),
                "annualized_degradation": _safe_float(validation.get("annualized_return")) - _safe_float(train.get("annualized_return")),
                "train_sharpe_ratio": _safe_float(train.get("sharpe_ratio")),
                "validation_sharpe_ratio": _safe_float(validation.get("sharpe_ratio")),
                "sharpe_degradation": _safe_float(validation.get("sharpe_ratio")) - _safe_float(train.get("sharpe_ratio")),
                "train_win_rate": _safe_float(train.get("win_rate")),
                "validation_win_rate": _safe_float(validation.get("win_rate")),
                "train_num_trades": int(_safe_float(train.get("num_trades"))),
                "validation_num_trades": int(_safe_float(validation.get("num_trades"))),
                "train_worst_year_precision": min(train_years.values()) if train_years else _safe_float(train.get("worst_year_precision")),
                "full_worst_year_precision": min(full_years.values()) if full_years else _safe_float(full.get("worst_year_precision")),
                "full_worst_year": min(full_years, key=full_years.get) if full_years else "",
            }
        )
    return pd.DataFrame(records)


def parameter_sensitivity_table(grid: pd.DataFrame, variant: str, stop_loss: float, take_profit: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = grid[grid["variant"] == variant].copy()
    if rows.empty:
        raise ValueError(f"No parameter-grid rows for variant: {variant}")
    near = rows[
        (rows["stop_loss_pct"].between(stop_loss - 0.0051, stop_loss + 0.0051))
        & (rows["take_profit_pct"].between(take_profit - 0.0051, take_profit + 0.0051))
    ].copy()
    exact = rows[
        np.isclose(rows["stop_loss_pct"], stop_loss)
        & np.isclose(rows["take_profit_pct"], take_profit)
    ]
    pass_all = rows[rows["pass_all_thresholds"] == True]  # noqa: E712
    near_pass = near[near["pass_all_thresholds"] == True]  # noqa: E712
    payload = {
        "variant": variant,
        "production_stop_loss_pct": stop_loss,
        "production_take_profit_pct": take_profit,
        "grid_rows": int(len(rows)),
        "grid_pass_count": int(len(pass_all)),
        "near_rows": int(len(near)),
        "near_pass_count": int(len(near_pass)),
        "exact_found": bool(not exact.empty),
        "exact_min_annualized_return": _safe_float(exact.iloc[0].get("min_annualized_return")) if not exact.empty else np.nan,
        "exact_min_annualized_margin": _safe_float(exact.iloc[0].get("min_annualized_return")) - THRESHOLDS["annualized_return"] if not exact.empty else np.nan,
    }
    return near.sort_values(["stop_loss_pct", "take_profit_pct"]), payload


def engine_calibration_flags(engine: pd.DataFrame) -> dict[str, Any]:
    if engine.empty:
        return {}
    level95 = engine[np.isclose(engine["confidence_level"], 0.95)].copy()
    best95 = level95.sort_values("actual_coverage", ascending=False).head(1)
    return {
        "methods": sorted(str(value) for value in engine["method"].dropna().unique()),
        "best_95_method": str(best95["method"].iloc[0]) if not best95.empty else "",
        "best_95_coverage": _safe_float(best95["actual_coverage"].iloc[0]) if not best95.empty else np.nan,
        "best_95_coverage_error": _safe_float(best95["coverage_error"].iloc[0]) if not best95.empty else np.nan,
        "min_pit_ks_pvalue": _safe_float(engine["pit_ks_pvalue"].min(), np.nan),
    }


def build_audit(
    promotion_summary: pd.DataFrame,
    meta_metrics: pd.DataFrame,
    parameter_grid: pd.DataFrame,
    engine_calibration: pd.DataFrame,
    label: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    baseline = _baseline_row(promotion_summary, label)
    window_table = window_threshold_table(baseline)
    validation_table = train_validation_table(meta_metrics)
    override = PRODUCTION_RULE_OVERRIDES.get(label, {})
    stop_loss = float(override.get("stop_loss_pct", baseline.get("stop_loss_pct", 0.025)))
    take_profit = float(override.get("take_profit_pct", baseline.get("take_profit_pct", 0.040)))
    sensitivity_table, sensitivity = parameter_sensitivity_table(parameter_grid, "baseline", stop_loss, take_profit)
    engine_flags = engine_calibration_flags(engine_calibration)

    warnings: list[str] = []
    min_annualized_margin = float(window_table["annualized_margin"].min())
    if min_annualized_margin < 0.01:
        warnings.append("Annualized return passes, but the minimum margin is below 1 percentage point.")
    if int(sensitivity["near_pass_count"]) <= 1:
        warnings.append("Only one nearby stop/take parameter set passes all hard gates; parameter fragility is elevated.")
    if int(window_table["num_trades"].min()) < 60:
        warnings.append("Shorter windows still have fewer than 60 trades; use 24M/36M as supporting evidence, not sole proof.")
    if validation_table["full_worst_year_precision"].min() < 0.50:
        warnings.append("At least one yearly regime has precision below 50%; 2022-like regimes need separate handling.")
    if window_table["calibration_error"].abs().max() > 0.10:
        warnings.append("Probability calibration is weak in at least one window; direction signal is better than prob_up scale.")
    if engine_flags.get("best_95_coverage", 1.0) < 0.95:
        warnings.append("Terminal-return engine 95% interval remains under-covered; tail risk is still understated.")

    severe = [
        warning
        for warning in warnings
        if "Only one nearby" in warning or "minimum margin" in warning or "below 50%" in warning
    ]
    overall_status = "PASS_WITH_WARNINGS" if not severe else "PASS_BUT_FRAGILE"
    if not bool(window_table["passes_hard_gates"].all()):
        overall_status = "REJECT_HARD_GATE_FAILURE"

    payload = {
        "label": str(baseline["label"]),
        "overall_status": overall_status,
        "hard_gates_pass": bool(window_table["passes_hard_gates"].all()),
        "min_annualized_margin": min_annualized_margin,
        "min_sharpe_margin": float(window_table["sharpe_margin"].min()),
        "max_drawdown_margin": float(window_table["drawdown_margin"].min()),
        "min_win_rate_margin": float(window_table["win_rate_margin"].min()),
        "min_profit_factor_margin": float(window_table["profit_factor_margin"].min()),
        "min_num_trades": int(window_table["num_trades"].min()),
        "parameter_sensitivity": sensitivity,
        "engine_calibration": engine_flags,
        "warnings": warnings,
        "recommended_controls": [
            "Keep the audited production override, but do not promote any new rule without 24/36/60 stress validation.",
            "Treat 2022-like/high-volatility regimes as a separate model bucket before increasing exposure.",
            "Do not use prob_up as a sizing scalar until calibration error is reduced.",
            "Require parameter-neighborhood evidence before future stop/take changes become production.",
        ],
    }
    tables = {
        "window_thresholds": window_table,
        "train_validation": validation_table,
        "parameter_sensitivity_nearby": sensitivity_table,
        "engine_calibration": engine_calibration,
    }
    return payload, tables


def _pct(value: float) -> str:
    return f"{value:.2%}"


def write_report(output_dir: Path, payload: dict[str, Any], tables: dict[str, pd.DataFrame]) -> Path:
    lines = [
        "# NVDA Overfit Audit",
        "",
        f"- Label: `{payload['label']}`",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Hard gates pass: {payload['hard_gates_pass']}",
        f"- Minimum annualized margin: {_pct(payload['min_annualized_margin'])}",
        f"- Minimum trade count: {payload['min_num_trades']}",
        "",
        "## Warnings",
    ]
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Controls"])
    lines.extend(f"- {item}" for item in payload["recommended_controls"])
    lines.extend(["", "## Window Thresholds", ""])
    window_cols = [
        "lookback_months",
        "annualized_return",
        "annualized_margin",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "num_trades",
        "passes_hard_gates",
    ]
    lines.append(tables["window_thresholds"][window_cols].to_markdown(index=False))
    lines.extend(["", "## Train Vs Validation", ""])
    validation_cols = [
        "lookback_months",
        "train_annualized_return",
        "validation_annualized_return",
        "train_sharpe_ratio",
        "validation_sharpe_ratio",
        "train_num_trades",
        "validation_num_trades",
        "full_worst_year",
        "full_worst_year_precision",
    ]
    lines.append(tables["train_validation"][validation_cols].to_markdown(index=False))
    lines.extend(["", "## Parameter Neighborhood", ""])
    param_cols = [
        "stop_loss_pct",
        "take_profit_pct",
        "pass_all_thresholds",
        "min_annualized_return",
        "min_sharpe_ratio",
        "max_abs_drawdown",
        "min_win_rate",
        "min_profit_factor",
        "min_num_trades",
    ]
    lines.append(tables["parameter_sensitivity_nearby"][param_cols].to_markdown(index=False))
    path = output_dir / "overfit_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, tables = build_audit(
        _read_csv(args.promotion_summary),
        _read_csv(args.meta_metrics),
        _read_csv(args.parameter_grid),
        _read_csv(args.engine_calibration),
        args.label,
    )
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    report_path = write_report(output_dir, payload, tables)
    payload["report"] = str(report_path.resolve())
    (output_dir / "overfit_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    default_root = PROJECT_ROOT / "outputs" / "performance_retest"
    parser = argparse.ArgumentParser(description="Audit NVDA production rule for overfitting and fragility")
    parser.add_argument("--label", default=BASELINE_PRODUCTION_LABEL)
    parser.add_argument(
        "--promotion-summary",
        default=str(default_root / "strict_candidate_promotion_override" / "candidate_promotion_summary.csv"),
    )
    parser.add_argument(
        "--meta-metrics",
        default=str(default_root / "meta_decision_layer_override" / "meta_decision_metrics.csv"),
    )
    parser.add_argument(
        "--parameter-grid",
        default=str(default_root / "adjustment_search" / "stop_take_grid_summary.csv"),
    )
    parser.add_argument(
        "--engine-calibration",
        default=str(default_root / "engine_calibration" / "engine_calibration_summary.csv"),
    )
    parser.add_argument("--output-dir", default=str(default_root / "overfit_audit"))
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
