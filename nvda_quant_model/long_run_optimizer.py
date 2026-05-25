from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import (
    build_evaluation_context,
    MOMENTUM_FEATURES,
    PrecisionRule,
    evaluate_rule,
    latest_prediction_for_rule,
    write_summary,
)
from nvda_quant_model.optimizer_csv import append_rows_schema_safe


FLAG_SETS: list[dict[str, bool]] = [
    {},
    {"require_smh_positive": True},
    {"require_qqq_positive": True},
    {"require_sp500_positive": True},
    {"require_smh_positive": True, "require_qqq_positive": True},
    {"require_smh_positive": True, "require_sp500_positive": True},
    {"require_smh_positive": True, "require_macd_positive": True},
    {"require_smh_positive": True, "require_obv_positive": True},
    {"require_smh_positive": True, "require_vol_calm": True},
    {"require_smh_positive": True, "exclude_negative_pre_holiday": True},
    {"require_smh_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_peer_mean_positive": True},
    {"require_smh_positive": True, "require_peer_event_net_positive": True},
    {"require_smh_positive": True, "require_no_peer_business_stress": True},
    {"require_smh_positive": True, "require_peer_breadth_positive": True, "require_no_peer_business_stress": True},
    {"require_smh_positive": True, "require_qqq_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_obv_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_obv_positive": True, "require_peer_event_net_positive": True},
    {"require_smh_positive": True, "require_macd_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_qqq_positive": True, "require_obv_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_peer_mean_positive": True, "require_peer_event_net_positive": True},
]


SEARCH_SPACE: dict[str, list[Any]] = {
    "train_window": [84, 105, 126, 147, 168, 189, 210, 252, 315],
    "test_window": [21, 28, 42, 63],
    "momentum_feature": MOMENTUM_FEATURES,
    "momentum_quantile": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "volume_quantile": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "max_rsi": [None, 60.0, 65.0, 68.0, 72.0, 75.0, 78.0, 82.0],
    "min_price_60ma": [None, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04],
    "vix_quantile_cap": [None, 0.60, 0.70, 0.80, 0.90],
    "stop_loss_pct": [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060],
    "take_profit_pct": [0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070, 0.080],
    "max_exposure": [0.50, 0.75, 1.00],
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sample_rule(rng: random.Random) -> PrecisionRule:
    values = {name: rng.choice(options) for name, options in SEARCH_SPACE.items()}
    while values["take_profit_pct"] <= values["stop_loss_pct"]:
        values["take_profit_pct"] = rng.choice(SEARCH_SPACE["take_profit_pct"])
        values["stop_loss_pct"] = rng.choice(SEARCH_SPACE["stop_loss_pct"])
    flags = rng.choice(FLAG_SETS)
    return PrecisionRule(
        train_window=int(values["train_window"]),
        test_window=int(values["test_window"]),
        momentum_feature=str(values["momentum_feature"]),
        momentum_quantile=float(values["momentum_quantile"]),
        volume_quantile=float(values["volume_quantile"]),
        max_rsi=values["max_rsi"],
        min_price_60ma=values["min_price_60ma"],
        require_smh_positive=flags.get("require_smh_positive", False),
        require_qqq_positive=flags.get("require_qqq_positive", False),
        require_sp500_positive=flags.get("require_sp500_positive", False),
        require_macd_positive=flags.get("require_macd_positive", False),
        require_obv_positive=flags.get("require_obv_positive", False),
        require_vol_calm=flags.get("require_vol_calm", False),
        require_peer_breadth_positive=flags.get("require_peer_breadth_positive", False),
        require_peer_mean_positive=flags.get("require_peer_mean_positive", False),
        require_peer_event_net_positive=flags.get("require_peer_event_net_positive", False),
        require_no_peer_business_stress=flags.get("require_no_peer_business_stress", False),
        vix_quantile_cap=values["vix_quantile_cap"],
        exclude_negative_pre_holiday=flags.get("exclude_negative_pre_holiday", False),
        stop_loss_pct=float(values["stop_loss_pct"]),
        take_profit_pct=float(values["take_profit_pct"]),
        max_exposure=float(values["max_exposure"]),
    )


def _robust_score(row: dict[str, Any], periods: pd.DataFrame) -> float:
    precision = float(row.get("dir_precision", 0.0))
    worst_year = float(row.get("dir_worst_year_precision", 0.0))
    sharpe = float(row.get("bt_sharpe_ratio", 0.0))
    annualized = float(row.get("bt_annualized_return", 0.0))
    drawdown = abs(float(row.get("bt_max_drawdown", 0.0)))
    profit_factor = min(float(row.get("bt_profit_factor", 0.0)), 8.0)
    active_days = int(row.get("dir_active_days", 0))
    trades = int(row.get("bt_num_trades", 0))

    period_precision = periods["precision"].dropna() if "precision" in periods else pd.Series(dtype=float)
    weak_periods = int((period_precision < 0.50).sum()) if not period_precision.empty else 9
    period_std = float(period_precision.std(ddof=0)) if len(period_precision) > 1 else 0.0
    sample_penalty = max(0, 25 - active_days) * 0.15 + max(0, 18 - trades) * 0.20
    drawdown_penalty = max(0.0, drawdown - 0.12) * 10.0
    instability_penalty = weak_periods * 0.35 + period_std * 2.0 + max(0.0, 0.58 - worst_year) * 4.0

    return float(
        precision * 6.0
        + worst_year * 2.0
        + sharpe * 0.90
        + annualized * 1.25
        + profit_factor * 0.30
        + min(active_days, 80) * 0.015
        - sample_penalty
        - drawdown_penalty
        - instability_penalty
    )


def _load_seen(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    try:
        return set(pd.read_csv(results_path, usecols=["label"])["label"].dropna().astype(str))
    except (ValueError, pd.errors.EmptyDataError):
        return set()


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    values = row.to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in values.items()}


def _load_existing_best(output_dir: Path, results_path: Path) -> dict[str, Any] | None:
    if results_path.exists():
        try:
            rows = pd.read_csv(results_path)
            if not rows.empty and "long_score" in rows.columns:
                return _row_to_dict(rows.sort_values("long_score", ascending=False).iloc[0])
        except (ValueError, pd.errors.EmptyDataError):
            pass
    best_path = output_dir / "best_precision.json"
    if best_path.exists():
        try:
            payload = json.loads(best_path.read_text(encoding="utf-8"))
            best = payload.get("best")
            return best if isinstance(best, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    append_rows_schema_safe(path, rows)


def _write_best(
    output_dir: Path,
    best_row: dict[str, Any],
    best_signals: pd.DataFrame,
    best_periods: pd.DataFrame,
    latest_prediction: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    best_signals.to_csv(output_dir / "best_signals.csv")
    best_periods.to_csv(output_dir / "best_periods.csv", index=False)
    payload = {
        "updated_at": _now(),
        "data_window": metadata,
        "best": best_row,
        "latest_prediction": latest_prediction,
    }
    (output_dir / "best_precision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _state_payload(
    started_at: str,
    deadline_ts: float,
    evaluated: int,
    skipped_duplicates: int,
    skipped_errors: int,
    best_row: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "updated_at": _now(),
        "deadline_utc": datetime.fromtimestamp(deadline_ts, tz=timezone.utc).isoformat(timespec="seconds"),
        "evaluated": evaluated,
        "skipped_duplicates": skipped_duplicates,
        "skipped_errors": skipped_errors,
        "best_score": None if best_row is None else best_row.get("long_score"),
        "best_label": None if best_row is None else best_row.get("label"),
        "output_dir": str(output_dir.resolve()),
    }


def run_long_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "long_run_results.csv"
    state_path = output_dir / "long_run_state.json"
    rng = random.Random(args.seed)
    started_at = _now()
    deadline_ts = time.time() + args.hours * 3600.0

    start, end, metadata = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(ticker=args.ticker, start_date=start, end_date=end, lookback_months=args.lookback_months, include_peer_events=True)
    prices, external = load_market_data(
        base.ticker,
        base.start_date,
        base.end_date,
        cache_dir=PROJECT_ROOT / "cache",
        force_refresh=args.force_refresh or str(args.end).lower() in {"latest", "auto", "today", "now"},
    )
    peer_ohlcv = load_peer_ohlcv_panel(
        warmup_start(base.start_date),
        base.end_date,
        cache_dir=PROJECT_ROOT / "cache",
        force_refresh=args.force_refresh or str(args.end).lower() in {"latest", "auto", "today", "now"},
    )
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
    metadata = {
        **metadata,
        "resolved_start": base.start_date,
        "resolved_end": base.end_date,
        "rows": int(len(prices.loc[base.start_date : base.end_date])),
    }

    seen = _load_seen(results_path) if args.resume else set()
    pending_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = _load_existing_best(output_dir, results_path) if args.resume else None
    best_signals = pd.DataFrame()
    best_periods = pd.DataFrame()
    evaluated = len(seen) if args.resume else 0
    skipped_duplicates = 0
    skipped_errors = 0
    last_report = time.time()
    if best_row is not None:
        best_score = best_row.get("long_score", best_row.get("score", 0.0)) or 0.0
        print(
            f"resumed best long_score={float(best_score):.3f} "
            f"label={best_row.get('label')}",
            flush=True,
        )

    while time.time() < deadline_ts and (args.max_runs is None or evaluated < args.max_runs):
        rule = _sample_rule(rng)
        if rule.label in seen:
            skipped_duplicates += 1
            continue
        seen.add(rule.label)
        try:
            row, signals, periods = evaluate_rule(
                rule,
                base,
                frame,
                prices,
                external,
                args.min_active_days,
                args.min_trades,
                context,
            )
            row["long_score"] = _robust_score(row, periods)
            row["evaluated_at"] = _now()
            row["run_seed"] = args.seed
        except Exception as exc:
            skipped_errors += 1
            if args.verbose:
                print(f"skipped {rule.label}: {exc}", flush=True)
            continue

        evaluated += 1
        pending_rows.append(row)
        if best_row is None or row["long_score"] > best_row["long_score"]:
            best_row = row
            best_signals = signals
            best_periods = periods
            latest_prediction = latest_prediction_for_rule(rule, frame, args.price_override)
            _write_best(output_dir, best_row, best_signals, best_periods, latest_prediction, metadata)
            print(
                f"new best #{evaluated}: long_score={row['long_score']:.3f} "
                f"precision={row['dir_precision']:.2%} ann={row['bt_annualized_return']:.2%} "
                f"sharpe={row['bt_sharpe_ratio']:.2f} mdd={row['bt_max_drawdown']:.2%} "
                f"trades={int(row['bt_num_trades'])} label={row['label']}",
                flush=True,
            )

        if len(pending_rows) >= args.checkpoint_every:
            _append_rows(results_path, pending_rows)
            pending_rows = []
            state_path.write_text(
                json.dumps(
                    _state_payload(started_at, deadline_ts, evaluated, skipped_duplicates, skipped_errors, best_row, output_dir),
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )

        if time.time() - last_report >= args.progress_seconds:
            best_text = "none" if best_row is None else f"{best_row['long_score']:.3f} / {best_row['label']}"
            print(
                f"progress evaluated={evaluated} duplicates={skipped_duplicates} errors={skipped_errors} best={best_text}",
                flush=True,
            )
            last_report = time.time()

    _append_rows(results_path, pending_rows)
    state = _state_payload(started_at, deadline_ts, evaluated, skipped_duplicates, skipped_errors, best_row, output_dir)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    if results_path.exists():
        rows = pd.read_csv(results_path).sort_values("long_score", ascending=False)
        rows.to_csv(results_path, index=False)
        if not rows.empty:
            write_summary(output_dir, rows, rows.iloc[0])
    print(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed long-running NVDA precision optimizer")
    parser.add_argument("--hours", type=float, default=30.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--min-active-days", type=int, default=25)
    parser.add_argument("--min-trades", type=int, default=18)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_long_optimizer(parse_args())


if __name__ == "__main__":
    main()
