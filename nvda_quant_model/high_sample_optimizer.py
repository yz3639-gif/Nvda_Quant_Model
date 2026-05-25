from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
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
    MOMENTUM_FEATURES,
    write_summary,
)
from nvda_quant_model.optimizer_csv import append_rows_schema_safe, read_live_optimizer_csv


FLAG_SETS: list[dict[str, bool]] = [
    {},
    {"require_smh_positive": True},
    {"require_qqq_positive": True},
    {"require_sp500_positive": True},
    {"require_smh_positive": True, "require_qqq_positive": True},
    {"require_smh_positive": True, "require_sp500_positive": True},
    {"require_smh_positive": True, "require_obv_positive": True},
    {"require_smh_positive": True, "require_macd_positive": True},
    {"require_smh_positive": True, "require_vol_calm": True},
    {"require_peer_breadth_positive": True},
    {"require_peer_mean_positive": True},
    {"require_smh_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_peer_mean_positive": True},
    {"require_smh_positive": True, "exclude_negative_pre_holiday": True},
]


SEARCH_SPACE: dict[str, list[Any]] = {
    "train_window": [63, 84, 105, 126, 147, 168, 189, 210, 252],
    "test_window": [21, 28, 42, 63],
    "momentum_feature": MOMENTUM_FEATURES,
    "momentum_quantile": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "volume_quantile": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "max_rsi": [None, 68.0, 72.0, 75.0, 78.0, 82.0, 85.0],
    "min_price_60ma": [None, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02],
    "vix_quantile_cap": [None, 0.70, 0.80, 0.90, 0.95],
    "stop_loss_pct": [0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.070],
    "take_profit_pct": [0.035, 0.040, 0.050, 0.060, 0.070, 0.080, 0.100],
    "max_exposure": [0.35, 0.50, 0.75, 1.00],
}


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


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["label"])["label"].dropna().astype(str))
    except (ValueError, pd.errors.EmptyDataError):
        return set()


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    append_rows_schema_safe(path, rows)


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    values = row.to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in values.items()}


def _load_existing_best(results_path: Path) -> dict[str, Any] | None:
    if not results_path.exists():
        return None
    try:
        rows = read_live_optimizer_csv(results_path)
    except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if rows.empty or "high_sample_score" not in rows.columns:
        return None
    return _row_to_dict(rows.sort_values("high_sample_score", ascending=False).iloc[0])


def high_sample_quality(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    trades = int(row.get("bt_num_trades", 0))
    active_days = int(row.get("dir_active_days", 0))
    precision = float(row.get("dir_precision", 0.0))
    worst_year = float(row.get("dir_worst_year_precision", 0.0))
    sharpe = float(row.get("bt_sharpe_ratio", 0.0))
    annualized = float(row.get("bt_annualized_return", 0.0))
    drawdown = abs(float(row.get("bt_max_drawdown", 0.0)))
    win_rate = float(row.get("bt_win_rate", 0.0))
    profit_factor = float(row.get("bt_profit_factor", 0.0))

    return {
        "sample_gate": trades >= args.min_trades and active_days >= args.min_active_days,
        "quality_gate": (
            precision >= args.min_precision
            and worst_year >= args.min_worst_year_precision
            and sharpe >= args.min_sharpe
            and annualized >= args.min_annualized
            and drawdown <= args.max_drawdown
            and win_rate >= args.min_win_rate
            and profit_factor >= args.min_profit_factor
        ),
        "trades": trades,
        "active_days": active_days,
        "precision": precision,
        "worst_year": worst_year,
        "sharpe": sharpe,
        "annualized": annualized,
        "drawdown": drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def high_sample_score(row: dict[str, Any], periods: pd.DataFrame, args: argparse.Namespace) -> float:
    quality = high_sample_quality(row, args)
    precision = quality["precision"]
    worst_year = quality["worst_year"]
    sharpe = quality["sharpe"]
    annualized = quality["annualized"]
    drawdown = quality["drawdown"]
    win_rate = quality["win_rate"]
    profit_factor = min(quality["profit_factor"], 6.0)
    trades = quality["trades"]
    active_days = quality["active_days"]

    period_precision = periods["precision"].dropna() if "precision" in periods else pd.Series(dtype=float)
    weak_periods = int((period_precision < 0.50).sum()) if not period_precision.empty else 9
    period_std = float(period_precision.std(ddof=0)) if len(period_precision) > 1 else 0.0

    sample_shortfall = max(0, args.min_trades - trades) * 0.18 + max(0, args.min_active_days - active_days) * 0.08
    sample_bonus = min(trades, args.target_trades) / max(args.target_trades, 1) * 1.4
    active_bonus = min(active_days, args.target_active_days) / max(args.target_active_days, 1) * 0.8
    drawdown_penalty = max(0.0, drawdown - args.max_drawdown) * 12.0 + max(0.0, drawdown - 0.12) * 3.0
    instability_penalty = weak_periods * 0.25 + period_std * 1.75 + max(0.0, args.min_worst_year_precision - worst_year) * 3.0
    gate_bonus = 2.0 if quality["sample_gate"] and quality["quality_gate"] else 0.0

    return float(
        precision * 4.0
        + worst_year * 1.5
        + win_rate * 1.5
        + sharpe * 0.75
        + annualized * 1.0
        + profit_factor * 0.35
        + sample_bonus
        + active_bonus
        + gate_bonus
        - sample_shortfall
        - drawdown_penalty
        - instability_penalty
    )


def _state_payload(
    started_at: str,
    deadline_ts: float,
    evaluated: int,
    skipped_duplicates: int,
    skipped_errors: int,
    best_row: dict[str, Any] | None,
    best_qualified_row: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "updated_at": _now(),
        "deadline_utc": datetime.fromtimestamp(deadline_ts, tz=timezone.utc).isoformat(timespec="seconds"),
        "evaluated": evaluated,
        "skipped_duplicates": skipped_duplicates,
        "skipped_errors": skipped_errors,
        "best_score": None if best_row is None else best_row.get("high_sample_score"),
        "best_label": None if best_row is None else best_row.get("label"),
        "best_qualified_score": None if best_qualified_row is None else best_qualified_row.get("high_sample_score"),
        "best_qualified_label": None if best_qualified_row is None else best_qualified_row.get("label"),
        "output_dir": str(output_dir.resolve()),
    }


def _write_best(
    output_dir: Path,
    name: str,
    row: dict[str, Any],
    signals: pd.DataFrame,
    periods: pd.DataFrame,
    latest: dict[str, Any],
    metadata: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    signals.to_csv(output_dir / f"{name}_signals.csv")
    periods.to_csv(output_dir / f"{name}_periods.csv", index=False)
    payload = {
        "updated_at": _now(),
        "data_window": metadata,
        "gates": gates,
        "best": row,
        "latest_prediction": latest,
    }
    (output_dir / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _gates_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_trades": args.min_trades,
        "target_trades": args.target_trades,
        "min_active_days": args.min_active_days,
        "target_active_days": args.target_active_days,
        "min_precision": args.min_precision,
        "min_worst_year_precision": args.min_worst_year_precision,
        "min_sharpe": args.min_sharpe,
        "min_annualized": args.min_annualized,
        "max_drawdown": args.max_drawdown,
        "min_win_rate": args.min_win_rate,
        "min_profit_factor": args.min_profit_factor,
    }


def run_high_sample_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "high_sample_results.csv"
    state_path = output_dir / "high_sample_state.json"
    rng = random.Random(args.seed)
    started_at = _now()
    deadline_ts = time.time() + args.hours * 3600.0

    start, end, metadata = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(ticker=args.ticker, start_date=start, end_date=end, lookback_months=args.lookback_months, include_peer_events=True)
    force_refresh = args.force_refresh or str(args.end).lower() in {"latest", "auto", "today", "now"}
    prices, external = load_market_data(base.ticker, base.start_date, base.end_date, cache_dir=PROJECT_ROOT / "cache", force_refresh=force_refresh)
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(base.start_date), base.end_date, cache_dir=PROJECT_ROOT / "cache", force_refresh=force_refresh)
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
    gates = _gates_from_args(args)

    seen = _load_seen(results_path) if args.resume else set()
    best_row = _load_existing_best(results_path) if args.resume else None
    best_qualified_row: dict[str, Any] | None = None
    if args.resume and results_path.exists():
        try:
            rows = read_live_optimizer_csv(results_path)
            qualified = rows[(rows["sample_gate"] == True) & (rows["quality_gate"] == True)]  # noqa: E712
            if not qualified.empty:
                best_qualified_row = _row_to_dict(qualified.sort_values("high_sample_score", ascending=False).iloc[0])
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
            best_qualified_row = None

    best_signals = pd.DataFrame()
    best_periods = pd.DataFrame()
    best_qualified_signals = pd.DataFrame()
    best_qualified_periods = pd.DataFrame()
    evaluated = len(seen) if args.resume else 0
    skipped_duplicates = 0
    skipped_errors = 0
    pending_rows: list[dict[str, Any]] = []
    last_report = time.time()

    if best_row is not None:
        print(
            f"resumed best high_sample_score={float(best_row.get('high_sample_score', 0.0)):.3f} "
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
            row, signals, periods = evaluate_rule(rule, base, frame, prices, external, 0, 0, context)
            quality = high_sample_quality(row, args)
            row["sample_gate"] = bool(quality["sample_gate"])
            row["quality_gate"] = bool(quality["quality_gate"])
            row["high_sample_score"] = high_sample_score(row, periods, args)
            row["evaluated_at"] = _now()
            row["run_seed"] = args.seed
        except Exception as exc:
            skipped_errors += 1
            if args.verbose:
                print(f"skipped {rule.label}: {exc}", flush=True)
            continue

        evaluated += 1
        pending_rows.append(row)

        if best_row is None or row["high_sample_score"] > best_row["high_sample_score"]:
            best_row = row
            best_signals = signals
            best_periods = periods
            latest = latest_prediction_for_rule(rule, frame, args.price_override)
            _write_best(output_dir, "best_high_sample", row, signals, periods, latest, metadata, gates)
            print(
                f"new best #{evaluated}: score={row['high_sample_score']:.3f} "
                f"trades={int(row['bt_num_trades'])} precision={row['dir_precision']:.2%} "
                f"win={row['bt_win_rate']:.2%} ann={row['bt_annualized_return']:.2%} "
                f"sharpe={row['bt_sharpe_ratio']:.2f} mdd={row['bt_max_drawdown']:.2%} "
                f"qualified={row['sample_gate'] and row['quality_gate']} label={row['label']}",
                flush=True,
            )

        if row["sample_gate"] and row["quality_gate"] and (
            best_qualified_row is None or row["high_sample_score"] > best_qualified_row["high_sample_score"]
        ):
            best_qualified_row = row
            best_qualified_signals = signals
            best_qualified_periods = periods
            latest = latest_prediction_for_rule(rule, frame, args.price_override)
            _write_best(output_dir, "best_qualified_high_sample", row, signals, periods, latest, metadata, gates)
            print(
                f"new qualified #{evaluated}: score={row['high_sample_score']:.3f} "
                f"trades={int(row['bt_num_trades'])} precision={row['dir_precision']:.2%} "
                f"win={row['bt_win_rate']:.2%} ann={row['bt_annualized_return']:.2%} "
                f"sharpe={row['bt_sharpe_ratio']:.2f} mdd={row['bt_max_drawdown']:.2%} label={row['label']}",
                flush=True,
            )

        if len(pending_rows) >= args.checkpoint_every:
            _append_rows(results_path, pending_rows)
            pending_rows = []
            state_path.write_text(
                json.dumps(
                    _state_payload(
                        started_at,
                        deadline_ts,
                        evaluated,
                        skipped_duplicates,
                        skipped_errors,
                        best_row,
                        best_qualified_row,
                        output_dir,
                    ),
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )

        if time.time() - last_report >= args.progress_seconds:
            best_text = "none" if best_row is None else f"{best_row['high_sample_score']:.3f} / {best_row['label']}"
            qualified_text = (
                "none"
                if best_qualified_row is None
                else f"{best_qualified_row['high_sample_score']:.3f} / {best_qualified_row['label']}"
            )
            print(
                f"progress evaluated={evaluated} duplicates={skipped_duplicates} errors={skipped_errors} "
                f"best={best_text} qualified={qualified_text}",
                flush=True,
            )
            last_report = time.time()

    _append_rows(results_path, pending_rows)
    if results_path.exists():
        rows = pd.read_csv(results_path).sort_values("high_sample_score", ascending=False)
        rows.to_csv(results_path, index=False)
        if not rows.empty:
            write_summary(output_dir, rows, rows.iloc[0])
    state = _state_payload(started_at, deadline_ts, evaluated, skipped_duplicates, skipped_errors, best_row, best_qualified_row, output_dir)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-sample NVDA optimizer with hard trade-count gates")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=60)
    parser.add_argument("--target-trades", type=int, default=90)
    parser.add_argument("--min-active-days", type=int, default=90)
    parser.add_argument("--target-active-days", type=int, default=130)
    parser.add_argument("--min-precision", type=float, default=0.58)
    parser.add_argument("--min-worst-year-precision", type=float, default=0.52)
    parser.add_argument("--min-sharpe", type=float, default=1.0)
    parser.add_argument("--min-annualized", type=float, default=0.10)
    parser.add_argument("--max-drawdown", type=float, default=0.20)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--min-profit-factor", type=float, default=1.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-seconds", type=int, default=300)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "high_sample_optimizer"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_high_sample_optimizer(parse_args())


if __name__ == "__main__":
    main()
