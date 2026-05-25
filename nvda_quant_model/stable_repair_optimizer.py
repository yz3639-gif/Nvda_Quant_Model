from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import calculate_metrics
from nvda_quant_model.candidate_promotion import DEFAULT_BASELINE_LABEL
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.optimizer_csv import append_rows_schema_safe, read_live_optimizer_csv
from nvda_quant_model.precision_search import (
    MOMENTUM_FEATURES,
    PrecisionRule,
    build_evaluation_context,
    build_walk_forward_rule_signals,
    directional_metrics,
    latest_prediction_for_rule,
)
from nvda_quant_model.production_model import BASELINE_PRODUCTION_LABEL
from nvda_quant_model.strict_model_selection import rule_from_row


LOOKBACK_WINDOWS = (24, 36, 60)
RECENT_CONSTRAINT_WINDOWS = (24, 36)
OOS_WINDOWS = (45, 63, 84, 126)
BASELINE_CONSTRAINT_METRICS = ("annualized_return", "sharpe_ratio", "win_rate", "profit_factor")
BASELINE_DRAWDOWN_TOLERANCE = 0.001
MIN_OOS_TRADES = 3


REPAIR_FLAG_SETS: list[dict[str, bool]] = [
    {"require_smh_positive": True},
    {"require_qqq_positive": True},
    {"require_sp500_positive": True},
    {"require_smh_positive": True, "require_qqq_positive": True},
    {"require_smh_positive": True, "require_sp500_positive": True},
    {"require_smh_positive": True, "require_macd_positive": True},
    {"require_smh_positive": True, "require_obv_positive": True},
    {"require_smh_positive": True, "require_vol_calm": True},
    {"require_smh_positive": True, "exclude_negative_pre_holiday": True},
    {"require_peer_breadth_positive": True},
    {"require_peer_mean_positive": True},
    {"require_smh_positive": True, "require_peer_breadth_positive": True},
    {"require_smh_positive": True, "require_peer_mean_positive": True},
    {"require_smh_positive": True, "require_peer_event_net_positive": True},
    {"require_smh_positive": True, "require_no_peer_business_stress": True},
    {"require_smh_positive": True, "require_peer_breadth_positive": True, "require_no_peer_business_stress": True},
]


REPAIR_SEARCH_SPACE: dict[str, list[Any]] = {
    "train_window": [126, 147, 168, 189, 210, 252],
    "test_window": [21, 42, 63],
    "momentum_feature": MOMENTUM_FEATURES,
    "momentum_quantile": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "volume_quantile": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "max_rsi": [65.0, 68.0, 72.0, 75.0, 78.0],
    "min_price_60ma": [0.94, 0.96, 0.98, 1.00, 1.02],
    "vix_quantile_cap": [0.70, 0.80, 0.90, 0.95, None],
    "repair_exposure": [0.10, 0.15, 0.20, 0.25],
    "min_repair_prob": [0.58, 0.62, 0.66, 0.70],
}


@dataclass(frozen=True)
class RepairCandidate:
    repair_rule: PrecisionRule
    repair_exposure: float
    min_repair_prob: float

    @property
    def label(self) -> str:
        return (
            f"{BASELINE_PRODUCTION_LABEL}__repair{self.repair_exposure:.2f}"
            f"_p{self.min_repair_prob:.2f}__{self.repair_rule.label}"
        )


@dataclass
class WindowBundle:
    months: int
    config: StrategyConfig
    prices: pd.DataFrame
    frame: pd.DataFrame
    baseline_signals: pd.DataFrame
    baseline_metrics: dict[str, Any]
    baseline_directional: dict[str, Any]


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def fallback_baseline_rule() -> PrecisionRule:
    """Return the audited NVDA production rule if the optimizer CSV is unavailable."""

    row = pd.Series(
        {
            "label": BASELINE_PRODUCTION_LABEL,
            "train_window": 147,
            "test_window": 63,
            "momentum_feature": "10d_return",
            "momentum_quantile": 0.60,
            "volume_quantile": 0.45,
            "max_rsi": 75.0,
            "min_price_60ma": 0.96,
            "require_smh_positive": True,
            "require_qqq_positive": False,
            "require_sp500_positive": False,
            "require_macd_positive": False,
            "require_obv_positive": True,
            "require_vol_calm": False,
            "require_peer_breadth_positive": False,
            "require_peer_mean_positive": False,
            "require_peer_event_net_positive": False,
            "require_no_peer_business_stress": False,
            "vix_quantile_cap": 0.90,
            "exclude_negative_pre_holiday": False,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.040,
            "max_exposure": 1.0,
        }
    )
    return rule_from_row(row)


def baseline_rule_from_results(rows: pd.DataFrame, baseline_label: str = DEFAULT_BASELINE_LABEL) -> PrecisionRule:
    if not rows.empty and "label" in rows.columns:
        baseline = rows[rows["label"] == baseline_label]
        if not baseline.empty:
            return rule_from_row(baseline.iloc[0])
    return fallback_baseline_rule()


def combine_core_and_repair_signals(
    core_signals: pd.DataFrame,
    repair_signals: pd.DataFrame,
    index: pd.Index,
    *,
    repair_exposure: float,
    min_repair_prob: float = 0.0,
) -> pd.DataFrame:
    """Keep the production core untouched; use repair only on core-flat days."""

    columns = ["direction", "confidence", "expected_return", "prob_up", "prob_down", "position"]
    core = core_signals.reindex(index)[columns]
    repair = repair_signals.reindex(index)[columns]
    for frame in (core, repair):
        frame[["direction", "confidence", "expected_return", "position"]] = frame[
            ["direction", "confidence", "expected_return", "position"]
        ].fillna(0.0)
        frame["prob_up"] = frame["prob_up"].fillna(0.5)
        frame["prob_down"] = frame["prob_down"].fillna(0.5)
    combined = core.copy()
    overlay_mask = (core["position"] <= 0.0) & (repair["position"] > 0.0) & (repair["prob_up"] >= min_repair_prob)
    if overlay_mask.any():
        combined.loc[overlay_mask, ["direction", "confidence", "expected_return", "prob_up", "prob_down"]] = repair.loc[
            overlay_mask, ["direction", "confidence", "expected_return", "prob_up", "prob_down"]
        ]
        combined.loc[overlay_mask, "position"] = np.minimum(repair.loc[overlay_mask, "position"], repair_exposure)
    combined["core_active"] = (core["position"] > 0.0).astype(int)
    combined["repair_overlay_active"] = overlay_mask.astype(int)
    return combined


def baseline_constraint_failures(candidate: pd.Series, baseline: pd.Series, *, months: int) -> list[str]:
    """Hard reject any 24M/36M candidate that weakens current baseline metrics."""

    failures: list[str] = []
    for metric in BASELINE_CONSTRAINT_METRICS:
        if _safe_float(candidate.get(metric), -np.inf) < _safe_float(baseline.get(metric), -np.inf):
            failures.append(f"{months}m_{metric}")
    if abs(_safe_float(candidate.get("max_drawdown"))) > abs(_safe_float(baseline.get("max_drawdown"))) + BASELINE_DRAWDOWN_TOLERANCE:
        failures.append(f"{months}m_max_drawdown")
    return failures


def sixty_month_improvement_failures(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    min_annualized_target: float,
    min_sharpe_target: float,
) -> list[str]:
    failures: list[str] = []
    if _safe_float(candidate.get("annualized_return")) <= _safe_float(baseline.get("annualized_return")):
        failures.append("60m_annualized_not_improved")
    if _safe_float(candidate.get("annualized_return")) < min_annualized_target:
        failures.append("60m_annualized_target")
    if _safe_float(candidate.get("sharpe_ratio")) < max(_safe_float(baseline.get("sharpe_ratio")), min_sharpe_target):
        failures.append("60m_sharpe")
    if abs(_safe_float(candidate.get("max_drawdown"))) > abs(_safe_float(baseline.get("max_drawdown"))) + BASELINE_DRAWDOWN_TOLERANCE:
        failures.append("60m_max_drawdown")
    if _safe_float(candidate.get("win_rate")) < _safe_float(baseline.get("win_rate")):
        failures.append("60m_win_rate")
    if _safe_float(candidate.get("profit_factor")) < _safe_float(baseline.get("profit_factor")):
        failures.append("60m_profit_factor")
    if _safe_float(candidate.get("num_trades")) < _safe_float(baseline.get("num_trades")):
        failures.append("60m_num_trades")
    return failures


def sixty_month_first_score(
    stress_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    rolling_oos: pd.DataFrame,
    *,
    constraint_failures: list[str],
    improvement_failures: list[str],
) -> float:
    by_window = {int(row["lookback_months"]): row for _, row in stress_rows.iterrows()}
    baseline_by_window = {int(row["lookback_months"]): row for _, row in baseline_rows.iterrows()}
    long = by_window.get(60)
    baseline_long = baseline_by_window.get(60)
    if long is None or baseline_long is None:
        partial_score = -100.0 - len(constraint_failures) * 8.0
        for months in sorted(by_window):
            item = by_window[months]
            base = baseline_by_window.get(months)
            if base is None:
                continue
            partial_score += (_safe_float(item.get("annualized_return")) - _safe_float(base.get("annualized_return"))) * 10.0
            partial_score += (_safe_float(item.get("sharpe_ratio")) - _safe_float(base.get("sharpe_ratio"))) * 2.0
            partial_score += (_safe_float(item.get("win_rate")) - _safe_float(base.get("win_rate"))) * 2.0
            partial_score += (_safe_float(item.get("profit_factor")) - _safe_float(base.get("profit_factor"))) * 0.5
            partial_score -= max(0.0, abs(_safe_float(item.get("max_drawdown"))) - abs(_safe_float(base.get("max_drawdown")))) * 8.0
        return float(partial_score)
    worst_oos = 0.0
    if not rolling_oos.empty:
        cleaned = rolling_oos["sharpe_degradation"].replace([np.inf, -np.inf], np.nan).dropna()
        worst_oos = float(cleaned.max()) if not cleaned.empty else 1.0
    ann_delta = _safe_float(long.get("annualized_return")) - _safe_float(baseline_long.get("annualized_return"))
    sharpe_delta = _safe_float(long.get("sharpe_ratio")) - _safe_float(baseline_long.get("sharpe_ratio"))
    pf_delta = _safe_float(long.get("profit_factor")) - _safe_float(baseline_long.get("profit_factor"))
    win_delta = _safe_float(long.get("win_rate")) - _safe_float(baseline_long.get("win_rate"))
    drawdown_penalty = max(
        0.0,
        abs(_safe_float(long.get("max_drawdown"))) - abs(_safe_float(baseline_long.get("max_drawdown"))),
    )
    return float(
        ann_delta * 18.0
        + sharpe_delta * 4.0
        + pf_delta * 1.2
        + win_delta * 3.0
        + min(_safe_float(long.get("num_trades")) / 120.0, 1.5)
        + min(_safe_float(long.get("dir_active_days")) / 180.0, 1.5)
        - drawdown_penalty * 12.0
        - max(0.0, worst_oos) * 2.0
        - len(constraint_failures) * 25.0
        - len(improvement_failures) * 5.0
    )


def _sample_repair_candidate(rng: random.Random, baseline_rule: PrecisionRule) -> RepairCandidate:
    values = {name: rng.choice(options) for name, options in REPAIR_SEARCH_SPACE.items()}
    flags = rng.choice(REPAIR_FLAG_SETS)
    # Keep execution risk comparable to production baseline; repair only changes entry timing and size.
    rule = PrecisionRule(
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
        stop_loss_pct=baseline_rule.stop_loss_pct,
        take_profit_pct=baseline_rule.take_profit_pct,
        max_exposure=1.0,
    )
    return RepairCandidate(
        repair_rule=rule,
        repair_exposure=float(values["repair_exposure"]),
        min_repair_prob=float(values["min_repair_prob"]),
    )


def _repair_candidate_from_row(row: pd.Series, baseline_rule: PrecisionRule, exposure: float, min_repair_prob: float) -> RepairCandidate:
    rule = rule_from_row(row)
    comparable_rule = PrecisionRule(
        **{
            **asdict(rule),
            "stop_loss_pct": baseline_rule.stop_loss_pct,
            "take_profit_pct": baseline_rule.take_profit_pct,
            "max_exposure": 1.0,
        }
    )
    return RepairCandidate(repair_rule=comparable_rule, repair_exposure=exposure, min_repair_prob=min_repair_prob)


def _load_seed_candidates(
    strict_rows: pd.DataFrame,
    high_rows: pd.DataFrame,
    baseline_rule: PrecisionRule,
    *,
    limit: int,
) -> list[RepairCandidate]:
    seeds: list[RepairCandidate] = []
    frames = []
    for frame, score_column in [(high_rows, "high_sample_score"), (strict_rows, "long_score"), (strict_rows, "score")]:
        if not frame.empty and score_column in frame.columns:
            item = frame.copy()
            item["_seed_score"] = pd.to_numeric(item[score_column], errors="coerce")
            frames.append(item.sort_values("_seed_score", ascending=False).head(limit))
    if not frames:
        return seeds
    rows = pd.concat(frames, ignore_index=True).drop_duplicates("label", keep="first")
    for _, row in rows.head(limit).iterrows():
        for exposure in (0.10, 0.15, 0.25):
            min_prob = 0.62 if exposure <= 0.15 else 0.66
            try:
                seeds.append(_repair_candidate_from_row(row, baseline_rule, exposure, min_prob))
            except (KeyError, TypeError, ValueError):
                continue
    return seeds


def _evaluate_signals(
    label: str,
    roles: str,
    months: int,
    config: StrategyConfig,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
    signals: pd.DataFrame,
) -> dict[str, Any]:
    result = BacktestEngine(config).backtest(signals, prices.loc[config.start_date : config.end_date])
    dmetrics = directional_metrics(signals, frame.loc[config.start_date : config.end_date])
    row: dict[str, Any] = {
        "label": label,
        "roles": roles,
        "lookback_months": months,
        "resolved_start": config.start_date,
        "resolved_end": config.end_date,
        **result.metrics,
        "dir_active_days": dmetrics["active_days"],
        "dir_precision": dmetrics["precision"],
        "dir_worst_year_precision": dmetrics["worst_year_precision"],
        "dir_avg_next_return": dmetrics["avg_next_return"],
        "dir_median_next_return": dmetrics["median_next_return"],
    }
    if "repair_overlay_active" in signals.columns:
        row["repair_overlay_days"] = int(signals["repair_overlay_active"].sum())
    return row


def prepare_window_bundle(ticker: str, months: int, baseline_rule: PrecisionRule) -> WindowBundle:
    start, end, _ = resolve_data_window("auto", "latest", ticker=ticker, lookback_months=months)
    config = StrategyConfig(
        ticker=ticker,
        start_date=start,
        end_date=end,
        lookback_months=months,
        include_peer_events=True,
        train_window=baseline_rule.train_window,
        test_window=baseline_rule.test_window,
        stop_loss_pct=baseline_rule.stop_loss_pct,
        take_profit_pct=baseline_rule.take_profit_pct,
        max_exposure=baseline_rule.max_exposure,
    )
    prices, external = load_market_data(config.ticker, config.start_date, config.end_date, cache_dir=PROJECT_ROOT / "cache")
    peer_ohlcv = load_peer_ohlcv_panel(warmup_start(config.start_date), config.end_date, cache_dir=PROJECT_ROOT / "cache")
    frame, _ = build_model_frame(
        prices,
        external,
        config.start_date,
        config.end_date,
        config.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
    )
    build_evaluation_context(config, frame, prices, external)
    baseline_signals, _ = build_walk_forward_rule_signals(frame, baseline_rule, config.start_date)
    baseline_signals = baseline_signals.reindex(prices.loc[config.start_date : config.end_date].index).dropna(
        subset=["prob_up", "expected_return"]
    )
    baseline_row = _evaluate_signals(
        BASELINE_PRODUCTION_LABEL,
        "current_baseline",
        months,
        config,
        prices,
        frame,
        baseline_signals,
    )
    baseline_directional = {
        "dir_active_days": baseline_row["dir_active_days"],
        "dir_precision": baseline_row["dir_precision"],
        "dir_worst_year_precision": baseline_row["dir_worst_year_precision"],
    }
    return WindowBundle(
        months=months,
        config=config,
        prices=prices,
        frame=frame,
        baseline_signals=baseline_signals,
        baseline_metrics=baseline_row,
        baseline_directional=baseline_directional,
    )


def evaluate_repair_candidate(candidate: RepairCandidate, bundles: dict[int, WindowBundle], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    constraints: list[str] = []
    improvement_failures: list[str] = []
    candidate_signals60 = pd.DataFrame()

    baseline_rows = pd.DataFrame([bundles[months].baseline_metrics for months in LOOKBACK_WINDOWS])
    for months in LOOKBACK_WINDOWS:
        bundle = bundles[months]
        repair_signals, _ = build_walk_forward_rule_signals(bundle.frame, candidate.repair_rule, bundle.config.start_date)
        combined = combine_core_and_repair_signals(
            bundle.baseline_signals,
            repair_signals,
            bundle.prices.loc[bundle.config.start_date : bundle.config.end_date].index,
            repair_exposure=candidate.repair_exposure,
            min_repair_prob=candidate.min_repair_prob,
        )
        row = _evaluate_signals(candidate.label, "stable_repair_overlay", months, bundle.config, bundle.prices, bundle.frame, combined)
        row["repair_rule_label"] = candidate.repair_rule.label
        row["repair_exposure"] = candidate.repair_exposure
        rows.append(row)
        if months in RECENT_CONSTRAINT_WINDOWS:
            constraints.extend(
                baseline_constraint_failures(
                    pd.Series(row),
                    pd.Series(bundle.baseline_metrics),
                    months=months,
                )
            )
            if constraints:
                break
        if months == 60:
            improvement_failures = sixty_month_improvement_failures(
                pd.Series(row),
                pd.Series(bundle.baseline_metrics),
                min_annualized_target=args.min_60m_annualized,
                min_sharpe_target=args.min_60m_sharpe,
            )
            candidate_signals60 = combined

    stress = pd.DataFrame(rows)
    rolling_oos = pd.DataFrame()
    status = "reject_recent_or_mid_degradation" if constraints else "reject_missing_60m"
    if not constraints and 60 in set(stress["lookback_months"]):
        bundle60 = bundles[60]
        rolling_oos = rolling_oos_diagnostics_for_signals(candidate.label, stress[stress["lookback_months"] == 60].iloc[0], bundle60, candidate_signals60)
        if improvement_failures:
            status = (
                "watchlist_60m_lift_below_target"
                if "60m_annualized_not_improved" not in improvement_failures
                else "reject_no_60m_improvement"
            )
        elif not rolling_oos.empty and (rolling_oos["oos_status"] != "pass").any():
            status = "reject_oos_degradation"
        else:
            status = "promote_repair_candidate"
    score = sixty_month_first_score(
        stress,
        baseline_rows,
        rolling_oos,
        constraint_failures=constraints,
        improvement_failures=improvement_failures,
    )
    summary = summarize_repair_candidate(
        candidate,
        stress,
        baseline_rows,
        rolling_oos,
        status=status,
        score=score,
        constraint_failures=constraints,
        improvement_failures=improvement_failures,
    )
    return stress, rolling_oos, summary


def rolling_oos_diagnostics_for_signals(
    label: str,
    full_row: pd.Series,
    bundle: WindowBundle,
    signals: pd.DataFrame,
    *,
    windows: tuple[int, ...] = OOS_WINDOWS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_sharpe = _safe_float(full_row.get("sharpe_ratio"))
    report_prices = bundle.prices.loc[bundle.config.start_date : bundle.config.end_date]
    for window in windows:
        tail_index = report_prices.tail(window).index
        oos_prices = report_prices.loc[tail_index]
        oos_signals = signals.reindex(tail_index).dropna(subset=["prob_up", "expected_return"])
        if len(oos_prices) < 10 or oos_signals.empty:
            rows.append(
                {
                    "label": label,
                    "window_days": window,
                    "full_sharpe_ratio": full_sharpe,
                    "oos_sharpe_ratio": np.nan,
                    "sharpe_degradation": np.inf,
                    "oos_num_trades": 0,
                    "oos_active_days": int((oos_signals.get("position", pd.Series(dtype=float)) > 0).sum()),
                    "oos_status": "oos_insufficient_sample",
                }
            )
            continue
        result = BacktestEngine(bundle.config).backtest(oos_signals, oos_prices)
        returns = result.daily_returns.dropna()
        equity = bundle.config.initial_capital * (1.0 + returns).cumprod()
        metrics = calculate_metrics(equity, returns, result.trades, bundle.config.initial_capital)
        oos_sharpe = _safe_float(metrics.get("sharpe_ratio"), np.nan)
        degradation = (full_sharpe - oos_sharpe) / abs(full_sharpe) if full_sharpe and np.isfinite(oos_sharpe) else np.inf
        trades = int(metrics.get("num_trades", 0))
        status = "pass"
        if trades < MIN_OOS_TRADES:
            status = "oos_insufficient_sample"
        elif degradation > 0.20:
            status = "reject_oos_degradation"
        rows.append(
            {
                "label": label,
                "window_days": window,
                "full_sharpe_ratio": full_sharpe,
                "oos_sharpe_ratio": oos_sharpe,
                "sharpe_degradation": float(degradation),
                "oos_num_trades": trades,
                "oos_active_days": int((oos_signals["position"] > 0).sum()),
                "oos_status": status,
                "oos_annualized_return": metrics.get("annualized_return"),
                "oos_max_drawdown": metrics.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def summarize_repair_candidate(
    candidate: RepairCandidate,
    stress: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    rolling_oos: pd.DataFrame,
    *,
    status: str,
    score: float,
    constraint_failures: list[str],
    improvement_failures: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "label": candidate.label,
        "repair_rule_label": candidate.repair_rule.label,
        "repair_exposure": candidate.repair_exposure,
        "min_repair_prob": candidate.min_repair_prob,
        "stable_repair_score": score,
        "constraint_failures": "|".join(constraint_failures),
        "improvement_failures": "|".join(improvement_failures),
        "rolling_oos_worst_degradation": np.nan,
        "rolling_oos_median_degradation": np.nan,
    }
    if not rolling_oos.empty:
        cleaned = rolling_oos["sharpe_degradation"].replace([np.inf, -np.inf], np.nan).dropna()
        row["rolling_oos_worst_degradation"] = float(cleaned.max()) if not cleaned.empty else np.nan
        row["rolling_oos_median_degradation"] = float(cleaned.median()) if not cleaned.empty else np.nan
    baseline_by_window = {int(item["lookback_months"]): item for _, item in baseline_rows.iterrows()}
    for _, item in stress.iterrows():
        months = int(item["lookback_months"])
        base = baseline_by_window.get(months, pd.Series(dtype=object))
        for metric in ["annualized_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "num_trades", "dir_active_days", "dir_precision", "repair_overlay_days"]:
            if metric in item:
                row[f"m{months}_{metric}"] = item.get(metric)
            if metric in base:
                row[f"base_m{months}_{metric}"] = base.get(metric)
        if "annualized_return" in item and "annualized_return" in base:
            row[f"m{months}_annualized_delta"] = _safe_float(item.get("annualized_return")) - _safe_float(base.get("annualized_return"))
        if "sharpe_ratio" in item and "sharpe_ratio" in base:
            row[f"m{months}_sharpe_delta"] = _safe_float(item.get("sharpe_ratio")) - _safe_float(base.get("sharpe_ratio"))
    return row


def write_repair_report(output_dir: Path, summary: pd.DataFrame, baseline_rows: pd.DataFrame, metadata: dict[str, Any]) -> Path:
    lines = [
        "# NVDA Stable Repair Optimizer",
        "",
        "## Purpose",
        "",
        "Baseline core stays untouched. Repair overlay may only add small exposure on core-flat days.",
        "Promotion requires 24M/36M to be no worse than baseline and 60M to improve under rolling OOS checks.",
        "",
        "## Metadata",
        "",
        f"- Baseline label: `{metadata['baseline_label']}`",
        f"- Evaluated repair candidates: {metadata['evaluated']}",
        f"- Skipped duplicates: {metadata['skipped_duplicates']}",
        f"- Skipped errors: {metadata['skipped_errors']}",
        "",
        "## Baseline Metrics",
        "",
        baseline_rows.to_markdown(index=False),
        "",
        "## Repair Summary",
        "",
        summary.head(30).to_markdown(index=False) if not summary.empty else "No repair candidates evaluated.",
        "",
        "## Interpretation",
        "",
        "- `reject_recent_or_mid_degradation`: 24M or 36M fell below the current production baseline.",
        "- `reject_no_60m_improvement`: 24M/36M held, but 60M did not beat baseline and target gates.",
        "- `watchlist_60m_lift_below_target`: 24M/36M held and 60M improved, but not enough for production targets.",
        "- `reject_oos_degradation`: 60M looked better, but trailing OOS Sharpe degradation exceeded 20% or sample was insufficient.",
        "- `promote_repair_candidate`: candidate can enter watchlist; it still does not replace production until manually audited.",
    ]
    path = output_dir / "stable_repair_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_stable_repair_optimizer(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "stable_repair_results.csv"
    stress_path = output_dir / "stable_repair_stress_results.csv"
    oos_path = output_dir / "stable_repair_rolling_oos.csv"
    state_path = output_dir / "stable_repair_state.json"

    strict_rows = read_live_optimizer_csv(args.strict_results) if Path(args.strict_results).exists() else pd.DataFrame()
    high_rows = read_live_optimizer_csv(args.high_sample_results) if Path(args.high_sample_results).exists() else pd.DataFrame()
    baseline_rule = baseline_rule_from_results(strict_rows, args.baseline_label)
    bundles = {months: prepare_window_bundle(args.ticker, months, baseline_rule) for months in LOOKBACK_WINDOWS}
    baseline_rows = pd.DataFrame([bundles[months].baseline_metrics for months in LOOKBACK_WINDOWS])
    baseline_rows.to_csv(output_dir / "stable_repair_baseline.csv", index=False)

    rng = random.Random(args.seed)
    seen = set(read_live_optimizer_csv(results_path)["label"].dropna().astype(str)) if args.resume and results_path.exists() else set()
    seed_candidates = _load_seed_candidates(strict_rows, high_rows, baseline_rule, limit=args.seed_candidate_limit)
    started_at = _now()
    deadline = time.time() + args.hours * 3600.0
    evaluated = len(seen) if args.resume else 0
    skipped_duplicates = 0
    skipped_errors = 0
    pending_summary: list[dict[str, Any]] = []
    pending_stress: list[dict[str, Any]] = []
    pending_oos: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    last_report = time.time()

    while time.time() < deadline and (args.max_runs is None or evaluated < args.max_runs):
        if seed_candidates:
            candidate = seed_candidates.pop(0)
        else:
            candidate = _sample_repair_candidate(rng, baseline_rule)
        if candidate.label in seen:
            skipped_duplicates += 1
            continue
        seen.add(candidate.label)
        try:
            stress, oos, summary = evaluate_repair_candidate(candidate, bundles, args)
        except Exception as exc:
            skipped_errors += 1
            if args.verbose:
                print(f"skipped {candidate.label}: {exc}", flush=True)
            continue
        evaluated += 1
        pending_summary.append({**summary, "evaluated_at": _now(), "run_seed": args.seed})
        pending_stress.extend(stress.to_dict(orient="records"))
        if not oos.empty:
            pending_oos.extend(oos.to_dict(orient="records"))

        if best_row is None or summary["stable_repair_score"] > best_row["stable_repair_score"]:
            best_row = summary
            (output_dir / "best_stable_repair.json").write_text(
                json.dumps(
                    {
                        "updated_at": _now(),
                        "best": best_row,
                        "baseline_metrics": baseline_rows.to_dict(orient="records"),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            print(
                f"new repair best #{evaluated}: status={summary['status']} score={summary['stable_repair_score']:.3f} "
                f"m60_ann={_safe_float(summary.get('m60_annualized_return')):.2%} "
                f"m60_sharpe={_safe_float(summary.get('m60_sharpe_ratio')):.2f} "
                f"label={summary['repair_rule_label']}",
                flush=True,
            )

        if len(pending_summary) >= args.checkpoint_every:
            _flush_results(results_path, stress_path, oos_path, pending_summary, pending_stress, pending_oos)
            pending_summary, pending_stress, pending_oos = [], [], []
            _write_state(state_path, started_at, deadline, evaluated, skipped_duplicates, skipped_errors, best_row, output_dir)

        if time.time() - last_report >= args.progress_seconds:
            best_text = "none" if best_row is None else f"{best_row['stable_repair_score']:.3f} / {best_row['status']}"
            print(
                f"progress repair evaluated={evaluated} duplicates={skipped_duplicates} errors={skipped_errors} best={best_text}",
                flush=True,
            )
            last_report = time.time()

    _flush_results(results_path, stress_path, oos_path, pending_summary, pending_stress, pending_oos)
    summary = read_live_optimizer_csv(results_path) if results_path.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values("stable_repair_score", ascending=False)
        summary.to_csv(results_path, index=False)
    metadata = {
        "baseline_label": args.baseline_label,
        "evaluated": evaluated,
        "skipped_duplicates": skipped_duplicates,
        "skipped_errors": skipped_errors,
    }
    report = write_repair_report(output_dir, summary, baseline_rows, metadata)
    payload = {
        "updated_at": _now(),
        "metadata": metadata,
        "promotion_counts": {} if summary.empty else {str(k): int(v) for k, v in summary["status"].value_counts().to_dict().items()},
        "best_promoted_label": None
        if summary.empty or summary[summary["status"] == "promote_repair_candidate"].empty
        else str(summary[summary["status"] == "promote_repair_candidate"].iloc[0]["label"]),
        "best_label": None if best_row is None else best_row["label"],
        "best_status": None if best_row is None else best_row["status"],
        "report": str(report.resolve()),
        "top_rows": [] if summary.empty else summary.head(10).to_dict(orient="records"),
    }
    (output_dir / "stable_repair_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_state(state_path, started_at, deadline, evaluated, skipped_duplicates, skipped_errors, best_row, output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return payload


def _flush_results(
    results_path: Path,
    stress_path: Path,
    oos_path: Path,
    summary_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
) -> None:
    _append_union_rows(results_path, summary_rows)
    _append_union_rows(stress_path, stress_rows)
    _append_union_rows(oos_path, oos_rows)


def _append_union_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    if path.exists():
        try:
            fieldnames.extend(pd.read_csv(path, nrows=0).columns.tolist())
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    normalized = [{key: row.get(key) for key in fieldnames} for row in rows]
    append_rows_schema_safe(path, normalized)


def _write_state(
    state_path: Path,
    started_at: str,
    deadline: float,
    evaluated: int,
    skipped_duplicates: int,
    skipped_errors: int,
    best_row: dict[str, Any] | None,
    output_dir: Path,
) -> None:
    state = {
        "started_at": started_at,
        "updated_at": _now(),
        "deadline_utc": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(timespec="seconds"),
        "evaluated": evaluated,
        "skipped_duplicates": skipped_duplicates,
        "skipped_errors": skipped_errors,
        "best_score": None if best_row is None else best_row.get("stable_repair_score"),
        "best_label": None if best_row is None else best_row.get("label"),
        "best_status": None if best_row is None else best_row.get("status"),
        "output_dir": str(output_dir.resolve()),
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search baseline-preserving repair overlays with 60M-first scoring")
    parser.add_argument("--strict-results", default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer_repaired" / "long_run_results.csv"))
    parser.add_argument("--high-sample-results", default=str(PROJECT_ROOT / "outputs" / "high_sample_optimizer_repaired" / "high_sample_results.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "stable_repair_optimizer"))
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price-override", type=float, default=215.34)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--seed-candidate-limit", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-seconds", type=int, default=300)
    parser.add_argument("--min-60m-annualized", type=float, default=0.18)
    parser.add_argument("--min-60m-sharpe", type=float, default=1.40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_stable_repair_optimizer(parse_args())


if __name__ == "__main__":
    main()
