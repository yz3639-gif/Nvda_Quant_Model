from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import (
    PrecisionRule,
    build_evaluation_context,
    evaluate_rule,
    latest_prediction_for_rule,
    write_summary,
)
from nvda_quant_model.production_model import apply_production_rule_overrides


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


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def rule_from_row(row: pd.Series) -> PrecisionRule:
    fields = {}
    for name in PrecisionRule.__dataclass_fields__:
        value = _none_if_nan(row[name])
        if name.startswith("require_") or name == "exclude_negative_pre_holiday":
            value = _bool_value(value)
        elif name in {"train_window", "test_window"}:
            value = int(value)
        elif name in {
            "momentum_quantile",
            "volume_quantile",
            "max_rsi",
            "min_price_60ma",
            "vix_quantile_cap",
            "stop_loss_pct",
            "take_profit_pct",
            "max_exposure",
        } and value is not None:
            value = float(value)
        fields[name] = value
    label = str(row.get("label", ""))
    fields = apply_production_rule_overrides(label, fields)
    return PrecisionRule(**fields)


def strict_filter(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = rows.copy()
    required = required[required["dir_active_days"] >= args.min_active_days]
    required = required[required["bt_num_trades"] >= args.min_trades]
    required = required[required["dir_precision"] >= args.min_precision]
    required = required[required["dir_worst_year_precision"] >= args.min_worst_year_precision]
    required = required[required["bt_sharpe_ratio"] >= args.min_sharpe]
    required = required[required["bt_annualized_return"] >= args.min_annualized]
    required = required[required["bt_max_drawdown"].abs() <= args.max_drawdown]
    required = required[required["bt_profit_factor"] >= args.min_profit_factor]
    return required.sort_values(
        [
            "long_score" if "long_score" in required.columns else "score",
            "dir_precision",
            "bt_sharpe_ratio",
            "bt_annualized_return",
        ],
        ascending=[False, False, False, False],
    )


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    results_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(results_path)
    filtered = strict_filter(rows, args)
    if filtered.empty:
        raise RuntimeError("No model satisfied strict selection gates")

    start, end, metadata = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(ticker=args.ticker, start_date=start, end_date=end, lookback_months=args.lookback_months, include_peer_events=True)
    prices, external = load_market_data(base.ticker, base.start_date, base.end_date, cache_dir=PROJECT_ROOT / "cache")
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(base.start_date), base.end_date, cache_dir=PROJECT_ROOT / "cache")
    frame, _ = build_model_frame(
        prices,
        external,
        base.start_date,
        base.end_date,
        base.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
    )
    context = build_evaluation_context(base, frame, prices, external)

    selected_row = filtered.iloc[0]
    rule = rule_from_row(selected_row)
    row, signals, periods = evaluate_rule(rule, base, frame, prices, external, args.min_active_days, args.min_trades, context)
    if "long_score" in selected_row:
        row["long_score"] = float(selected_row["long_score"])
    latest = latest_prediction_for_rule(rule, frame, args.price_override)

    signals.to_csv(output_dir / "strict_best_signals.csv")
    periods.to_csv(output_dir / "strict_best_periods.csv", index=False)
    filtered.head(args.top_n).to_csv(output_dir / "strict_candidates.csv", index=False)
    write_summary(output_dir, filtered.head(max(args.top_n, 20)), filtered.iloc[0])
    payload = {
        "source_results": str(results_path.resolve()),
        "gates": {
            "min_active_days": args.min_active_days,
            "min_trades": args.min_trades,
            "min_precision": args.min_precision,
            "min_worst_year_precision": args.min_worst_year_precision,
            "min_sharpe": args.min_sharpe,
            "min_annualized": args.min_annualized,
            "max_drawdown": args.max_drawdown,
            "min_profit_factor": args.min_profit_factor,
        },
        "data_window": {
            **metadata,
            "resolved_start": base.start_date,
            "resolved_end": base.end_date,
            "rows": int(len(prices.loc[base.start_date : base.end_date])),
        },
        "candidate_count": int(len(rows)),
        "strict_candidate_count": int(len(filtered)),
        "best": row,
        "latest_prediction": latest,
    }
    (output_dir / "strict_best_precision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a stricter best model from long-run optimizer results")
    parser.add_argument("--results-csv", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer" / "long_run_results.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "strict_selection_current"))
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--min-active-days", type=int, default=30)
    parser.add_argument("--min-trades", type=int, default=18)
    parser.add_argument("--min-precision", type=float, default=0.70)
    parser.add_argument("--min-worst-year-precision", type=float, default=0.68)
    parser.add_argument("--min-sharpe", type=float, default=1.50)
    parser.add_argument("--min-annualized", type=float, default=0.12)
    parser.add_argument("--max-drawdown", type=float, default=0.08)
    parser.add_argument("--min-profit-factor", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    run_selection(parse_args())


if __name__ == "__main__":
    main()
