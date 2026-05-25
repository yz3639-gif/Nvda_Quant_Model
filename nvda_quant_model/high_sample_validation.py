from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.candidate_promotion import (
    DEFAULT_BASELINE_LABEL,
    evaluate_candidates,
    summarize_promotions,
    write_report,
)
from nvda_quant_model.config import PROJECT_ROOT


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def select_high_sample_candidates(
    high_sample_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qualified = high_sample_rows.copy()
    if {"sample_gate", "quality_gate"}.issubset(qualified.columns):
        qualified = qualified[(qualified["sample_gate"] == True) & (qualified["quality_gate"] == True)]  # noqa: E712
    qualified = qualified.sort_values("high_sample_score", ascending=False).head(top_n)
    qualified = qualified.copy()
    if not qualified.empty:
        qualified["roles"] = "high_sample_qualified"
        qualified["long_score"] = qualified["high_sample_score"]

    baseline = baseline_rows[baseline_rows["label"] == baseline_label].copy()
    baseline_found = not baseline.empty
    effective_baseline_label = baseline_label
    if baseline.empty and not baseline_rows.empty:
        sort_col = "long_score" if "long_score" in baseline_rows.columns else "score"
        baseline = baseline_rows.sort_values(sort_col, ascending=False).head(1).copy()
        effective_baseline_label = str(baseline["label"].iloc[0])
    if not baseline.empty:
        baseline["roles"] = "current_baseline"

    candidates = pd.concat([baseline.head(1), qualified], ignore_index=True, sort=False)
    metadata = {
        "source_rows": int(len(high_sample_rows)),
        "strict30_count": int(len(qualified)),
        "high_sample_rows": int(len(high_sample_rows)),
        "qualified_rows": int(len(qualified)),
        "selected_count": int(len(candidates)),
        "baseline_found": bool(baseline_found),
        "baseline_label": baseline_label,
        "effective_baseline_label": effective_baseline_label,
    }
    return candidates, metadata


def _status_counts(summary: pd.DataFrame) -> dict[str, int]:
    if summary.empty or "status" not in summary:
        return {}
    return {str(key): int(value) for key, value in summary["status"].value_counts().items()}


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    high_sample_rows = pd.read_csv(args.high_sample_results)
    baseline_rows = pd.read_csv(args.baseline_results)
    candidates, metadata = select_high_sample_candidates(
        high_sample_rows,
        baseline_rows,
        baseline_label=args.baseline_label,
        top_n=args.top_n,
    )
    if candidates.empty:
        raise RuntimeError("No high-sample validation candidates selected")

    lookbacks = [int(value.strip()) for value in args.lookback_months.split(",") if value.strip()]
    stress, latest = evaluate_candidates(candidates, lookbacks, args.ticker, args.price_override)
    summary = summarize_promotions(stress, metadata["effective_baseline_label"])
    report_path = write_report(output_dir, metadata, summary, stress, latest)

    candidates.to_csv(output_dir / "high_sample_validation_candidates.csv", index=False)
    stress.to_csv(output_dir / "high_sample_stress_backtests.csv", index=False)
    latest.to_csv(output_dir / "high_sample_latest_snapshot.csv", index=False)
    summary.to_csv(output_dir / "high_sample_validation_summary.csv", index=False)
    payload = {
        "metadata": metadata,
        "lookback_months": lookbacks,
        "report": str(report_path.resolve()),
        "promotion_counts": _status_counts(summary),
        "top_rows": summary.head(10).to_dict(orient="records"),
    }
    (output_dir / "high_sample_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate high-sample optimizer candidates across 24/36/60 month windows")
    parser.add_argument(
        "--high-sample-results",
        default=str(PROJECT_ROOT / "outputs" / "high_sample_optimizer" / "high_sample_results.csv"),
    )
    parser.add_argument(
        "--baseline-results",
        default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer" / "long_run_results.csv"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "high_sample_validation"))
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--lookback-months", default="24,36,60")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_validation(parse_args())


if __name__ == "__main__":
    main()
