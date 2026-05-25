from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.candidate_promotion import DEFAULT_BASELINE_LABEL
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.precision_search import build_walk_forward_rule_signals, latest_prediction_for_rule
from nvda_quant_model.reaction_pool import directional_metrics
from nvda_quant_model.strict_model_selection import rule_from_row


@dataclass(frozen=True)
class RiskOffRule:
    family: str
    min_votes: int
    use_downtrend: bool = False
    use_high_vix_regime: bool = False
    use_vix_jump: bool = False
    use_macro_negative: bool = False
    use_sector_negative: bool = False
    use_peer_weak: bool = False
    use_peer_stress: bool = False
    use_vol_expansion: bool = False
    use_trend_break: bool = False
    use_holiday_weak: bool = False
    market_negative_count: int = 2
    peer_breadth_threshold: float = 0.45
    peer_stress_quantile: float = 0.75
    vix_jump_quantile: float = 0.75
    vol_multiplier: float = 1.05

    @property
    def label(self) -> str:
        parts = [self.family, f"votes>={self.min_votes}"]
        for name in [
            "downtrend",
            "high_vix_regime",
            "vix_jump",
            "macro_negative",
            "sector_negative",
            "peer_weak",
            "peer_stress",
            "vol_expansion",
            "trend_break",
            "holiday_weak",
        ]:
            if getattr(self, f"use_{name}"):
                parts.append(name)
        parts.append(f"mneg{self.market_negative_count}")
        parts.append(f"breadth<{self.peer_breadth_threshold:g}")
        parts.append(f"pstress{self.peer_stress_quantile:g}")
        parts.append(f"vixq{self.vix_jump_quantile:g}")
        parts.append(f"volx{self.vol_multiplier:g}")
        return "_".join(parts)


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def load_baseline_row(promotion_dir: Path, baseline_label: str = DEFAULT_BASELINE_LABEL) -> pd.Series:
    rows = pd.read_csv(promotion_dir / "candidate_source_rows.csv")
    baseline = rows[rows["label"] == baseline_label].copy()
    if baseline.empty:
        raise RuntimeError(f"Baseline rule not found in {promotion_dir}: {baseline_label}")
    sort_cols = [col for col in ["long_score", "bt_sharpe_ratio", "bt_annualized_return"] if col in baseline.columns]
    return baseline.sort_values(sort_cols, ascending=False).iloc[0] if sort_cols else baseline.iloc[0]


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame:
        return frame[column].fillna(default)
    return pd.Series(default, index=frame.index)


def build_risk_components(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["downtrend"] = (_series(frame, "regime_downtrend") > 0).astype(float)
    out["high_vix_regime"] = (_series(frame, "regime_high_vol") > 0).astype(float)

    vix_change = _series(frame, "VIX_weekly_change")
    out["vix_jump"] = (
        vix_change > vix_change.rolling(126, min_periods=40).quantile(0.75).shift(1)
    ).fillna(False).astype(float)

    market_returns = pd.DataFrame(
        {
            "SP500_return": _series(frame, "SP500_return"),
            "NASDAQ_return": _series(frame, "NASDAQ_return"),
            "QQQ_return": _series(frame, "QQQ_return"),
            "SMH_return": _series(frame, "SMH_return"),
        },
        index=frame.index,
    )
    out["negative_market_count"] = (market_returns < 0).sum(axis=1).astype(float)
    out["sector_negative"] = ((_series(frame, "SMH_return") < 0) & (_series(frame, "QQQ_return") < 0)).astype(float)
    out["peer_weak"] = (
        (_series(frame, "peer_positive_breadth_5d", 1.0) <= 0.45) | (_series(frame, "peer_mean_return_5d") < 0)
    ).astype(float)

    peer_stress = _series(frame, "peer_business_stress")
    out["peer_stress"] = (
        peer_stress > peer_stress.rolling(126, min_periods=40).quantile(0.75).shift(1)
    ).fillna(False).astype(float)
    out["vol_expansion"] = (
        _series(frame, "volatility_20") > _series(frame, "volatility_60") * 1.05
    ).fillna(False).astype(float)
    out["trend_break"] = (
        (_series(frame, "price_20ma_ratio", 1.0) < 1.0) & (_series(frame, "10d_return") < 0)
    ).astype(float)
    out["holiday_weak"] = (
        (_series(frame, "pre_holiday_session") == 1) & (_series(frame, "pre_holiday_momentum") < 0)
    ).astype(float)
    return out


def risk_mask_for_rule(frame: pd.DataFrame, rule: RiskOffRule) -> tuple[pd.Series, pd.DataFrame]:
    components = build_risk_components(frame)
    votes = pd.Series(0.0, index=frame.index)
    used = pd.DataFrame(index=frame.index)

    def add_component(name: str, condition: pd.Series) -> None:
        used[name] = condition.astype(float)

    if rule.use_downtrend:
        add_component("downtrend", components["downtrend"] > 0)
    if rule.use_high_vix_regime:
        add_component("high_vix_regime", components["high_vix_regime"] > 0)
    if rule.use_vix_jump:
        vix_change = _series(frame, "VIX_weekly_change")
        threshold = vix_change.rolling(126, min_periods=40).quantile(rule.vix_jump_quantile).shift(1)
        add_component("vix_jump", vix_change > threshold)
    if rule.use_macro_negative:
        add_component("macro_negative", components["negative_market_count"] >= rule.market_negative_count)
    if rule.use_sector_negative:
        add_component("sector_negative", components["sector_negative"] > 0)
    if rule.use_peer_weak:
        peer_condition = (
            (_series(frame, "peer_positive_breadth_5d", 1.0) <= rule.peer_breadth_threshold)
            | (_series(frame, "peer_mean_return_5d") < 0)
        )
        add_component("peer_weak", peer_condition)
    if rule.use_peer_stress:
        peer_stress = _series(frame, "peer_business_stress")
        threshold = peer_stress.rolling(126, min_periods=40).quantile(rule.peer_stress_quantile).shift(1)
        add_component("peer_stress", peer_stress > threshold)
    if rule.use_vol_expansion:
        add_component("vol_expansion", _series(frame, "volatility_20") > _series(frame, "volatility_60") * rule.vol_multiplier)
    if rule.use_trend_break:
        add_component("trend_break", (_series(frame, "price_20ma_ratio", 1.0) < 1.0) & (_series(frame, "10d_return") < 0))
    if rule.use_holiday_weak:
        add_component("holiday_weak", components["holiday_weak"] > 0)

    if not used.empty:
        votes = used.sum(axis=1)
    mask = (votes >= rule.min_votes).fillna(False)
    diagnostics = used.copy()
    diagnostics["risk_votes"] = votes
    diagnostics["risk_off"] = mask.astype(int)
    return mask.astype(bool), diagnostics


def apply_risk_off_guard(
    baseline_signals: pd.DataFrame,
    risk_mask: pd.Series,
    report_index: pd.Index,
) -> pd.DataFrame:
    guarded = baseline_signals.reindex(report_index).fillna(
        {
            "direction": 0,
            "confidence": 0.0,
            "expected_return": 0.0,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "position": 0.0,
        }
    )
    mask = risk_mask.reindex(report_index).fillna(False).astype(bool)
    baseline_position = guarded["position"].copy()
    blocked = (baseline_position > 0) & mask
    guarded["blocked_long"] = blocked.astype(int)
    guarded["risk_off"] = mask.astype(int)
    guarded["baseline_position_before_guard"] = baseline_position
    guarded.loc[blocked, ["direction", "confidence", "expected_return", "position"]] = 0.0
    guarded.loc[blocked, ["prob_up", "prob_down"]] = 0.5
    return guarded


def blocked_trade_diagnostics(
    baseline_signals: pd.DataFrame,
    guarded_signals: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    aligned = frame[["target_return", "target_direction"]].join(
        baseline_signals[["position"]].rename(columns={"position": "baseline_position"}),
        how="inner",
    )
    aligned = aligned.join(guarded_signals[["blocked_long", "risk_off"]], how="left")
    blocked = aligned[(aligned["baseline_position"] > 0) & (aligned["blocked_long"] == 1)].dropna(subset=["target_return"])
    if blocked.empty:
        return {
            "blocked_days": 0,
            "blocked_loss_rate": 0.0,
            "blocked_avg_next_return": 0.0,
            "blocked_median_next_return": 0.0,
            "blocked_loss_capture": 0.0,
        }
    baseline_active = aligned[aligned["baseline_position"] > 0].dropna(subset=["target_return"])
    baseline_losses = int((baseline_active["target_return"] <= 0).sum())
    blocked_losses = int((blocked["target_return"] <= 0).sum())
    return {
        "blocked_days": int(len(blocked)),
        "blocked_loss_rate": float((blocked["target_return"] <= 0).mean()),
        "blocked_avg_next_return": float(blocked["target_return"].mean()),
        "blocked_median_next_return": float(blocked["target_return"].median()),
        "blocked_loss_capture": float(blocked_losses / baseline_losses) if baseline_losses else 0.0,
    }


def candidate_risk_rules() -> list[RiskOffRule]:
    families: dict[str, dict[str, bool]] = {
        "macro_trend": {
            "use_downtrend": True,
            "use_macro_negative": True,
            "use_sector_negative": True,
            "use_vix_jump": True,
            "use_trend_break": True,
        },
        "peer_stress": {
            "use_peer_weak": True,
            "use_peer_stress": True,
            "use_sector_negative": True,
            "use_trend_break": True,
        },
        "high_vol_break": {
            "use_high_vix_regime": True,
            "use_vix_jump": True,
            "use_vol_expansion": True,
            "use_trend_break": True,
            "use_macro_negative": True,
        },
        "holiday_risk": {
            "use_holiday_weak": True,
            "use_macro_negative": True,
            "use_sector_negative": True,
        },
        "composite_risk": {
            "use_downtrend": True,
            "use_high_vix_regime": True,
            "use_macro_negative": True,
            "use_sector_negative": True,
            "use_peer_weak": True,
            "use_peer_stress": True,
            "use_vol_expansion": True,
            "use_trend_break": True,
        },
    }
    rules: list[RiskOffRule] = []
    for family, flags in families.items():
        component_count = sum(flags.values())
        min_vote_values = [2, 3] if component_count <= 5 else [3, 4]
        for min_votes in min_vote_values:
            for market_negative_count in [2, 3]:
                for peer_breadth_threshold in [0.40, 0.45, 0.50]:
                    for peer_stress_quantile in [0.70, 0.80]:
                        for vix_jump_quantile in [0.70, 0.80]:
                            for vol_multiplier in [1.00, 1.10]:
                                if min_votes > component_count:
                                    continue
                                rules.append(
                                    RiskOffRule(
                                        family=family,
                                        min_votes=min_votes,
                                        market_negative_count=market_negative_count,
                                        peer_breadth_threshold=peer_breadth_threshold,
                                        peer_stress_quantile=peer_stress_quantile,
                                        vix_jump_quantile=vix_jump_quantile,
                                        vol_multiplier=vol_multiplier,
                                        **flags,
                                    )
                                )
    unique: dict[str, RiskOffRule] = {}
    for rule in rules:
        unique[rule.label] = rule
    return list(unique.values())


def _metric_row(
    segment: str,
    layer: str,
    result: Any,
    signals: pd.DataFrame,
    frame: pd.DataFrame,
    baseline_signals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    dmetrics = directional_metrics(signals, frame)
    row = {"segment": segment, "layer": layer, **result.metrics, **dmetrics}
    if baseline_signals is not None:
        row.update(blocked_trade_diagnostics(baseline_signals, signals, frame))
    return row


def _evaluate_layer(
    index: pd.Index,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
    config: StrategyConfig,
    baseline_signals: pd.DataFrame,
    guarded_signals: pd.DataFrame,
    segment: str,
) -> pd.DataFrame:
    scoped_prices = prices.reindex(index).dropna(subset=["Close"])
    scoped_frame = frame.reindex(scoped_prices.index)
    rows = []
    for layer, signals, base_signals in [
        ("baseline", baseline_signals, None),
        ("risk_guarded", guarded_signals, baseline_signals),
    ]:
        scoped_signals = signals.reindex(scoped_prices.index).dropna(subset=["prob_up", "expected_return"])
        result = BacktestEngine(config).backtest(scoped_signals, scoped_prices)
        rows.append(_metric_row(segment, layer, result, scoped_signals, scoped_frame, base_signals))
    return pd.DataFrame(rows)


def _score_train(metrics: pd.DataFrame, min_blocked_days: int) -> float:
    baseline = metrics[metrics["layer"] == "baseline"].iloc[0]
    guarded = metrics[metrics["layer"] == "risk_guarded"].iloc[0]
    blocked_days = int(_safe_float(guarded.get("blocked_days")))
    if blocked_days < min_blocked_days:
        return -999.0 - (min_blocked_days - blocked_days)
    score = 0.0
    score += (_safe_float(guarded["direction_precision"]) - _safe_float(baseline["direction_precision"])) * 5.0
    score += (_safe_float(guarded["sharpe_ratio"]) - _safe_float(baseline["sharpe_ratio"])) * 1.25
    score += (_safe_float(guarded["annualized_return"]) - _safe_float(baseline["annualized_return"])) * 1.5
    drawdown_delta = abs(_safe_float(baseline["max_drawdown"])) - abs(_safe_float(guarded["max_drawdown"]))
    score += drawdown_delta * 4.0
    score += _safe_float(guarded.get("blocked_loss_rate")) * 0.75
    score += max(0.0, -_safe_float(guarded.get("blocked_avg_next_return"))) * 25.0
    score += _safe_float(guarded.get("blocked_loss_capture")) * 0.50
    return float(score)


def _passes_production_gate(metrics: pd.DataFrame, min_blocked_days: int) -> bool:
    baseline = metrics[metrics["layer"] == "baseline"].iloc[0]
    guarded = metrics[metrics["layer"] == "risk_guarded"].iloc[0]
    blocked_days = int(_safe_float(guarded.get("blocked_days")))
    return bool(
        blocked_days >= min_blocked_days
        and _safe_float(guarded.get("blocked_loss_rate")) >= 0.55
        and _safe_float(guarded["direction_precision"]) >= _safe_float(baseline["direction_precision"])
        and _safe_float(guarded["annualized_return"]) >= _safe_float(baseline["annualized_return"])
        and _safe_float(guarded["sharpe_ratio"]) >= _safe_float(baseline["sharpe_ratio"])
        and abs(_safe_float(guarded["max_drawdown"])) <= abs(_safe_float(baseline["max_drawdown"])) + 0.005
    )


def _is_watchlist_guard(metrics: pd.DataFrame, min_blocked_days: int) -> bool:
    baseline = metrics[metrics["layer"] == "baseline"].iloc[0]
    guarded = metrics[metrics["layer"] == "risk_guarded"].iloc[0]
    blocked_days = int(_safe_float(guarded.get("blocked_days")))
    return bool(
        blocked_days >= min_blocked_days
        and _safe_float(guarded.get("blocked_loss_rate")) >= 0.55
        and _safe_float(guarded["direction_precision"]) >= _safe_float(baseline["direction_precision"])
        and _safe_float(guarded["sharpe_ratio"]) >= _safe_float(baseline["sharpe_ratio"]) - 0.05
        and abs(_safe_float(guarded["max_drawdown"])) <= abs(_safe_float(baseline["max_drawdown"])) + 0.005
        and _safe_float(guarded["annualized_return"]) >= _safe_float(baseline["annualized_return"]) - 0.015
    )


def _format_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2%}"


def _format_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.2f}"


def _prepare_window(
    ticker: str,
    lookback_months: int,
    baseline: pd.Series,
) -> tuple[StrategyConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end, _ = resolve_data_window("auto", "latest", ticker=ticker, lookback_months=lookback_months)
    config = StrategyConfig(
        ticker=ticker,
        start_date=start,
        end_date=end,
        lookback_months=lookback_months,
        stop_loss_pct=float(baseline["stop_loss_pct"]),
        take_profit_pct=float(baseline["take_profit_pct"]),
        max_exposure=float(baseline["max_exposure"]),
        include_peer_events=True,
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
    report_prices = prices.loc[config.start_date : config.end_date]
    baseline_signals, _ = build_walk_forward_rule_signals(frame, rule_from_row(baseline), config.start_date)
    baseline_signals = baseline_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
    return config, prices, report_prices, frame, baseline_signals


def evaluate_window(
    ticker: str,
    lookback_months: int,
    baseline: pd.Series,
    rules: list[RiskOffRule],
    train_fraction: float,
    min_train_days: int,
    min_validation_days: int,
    min_blocked_days: int,
    price_override: float | None = None,
) -> dict[str, Any]:
    config, prices, report_prices, frame, baseline_signals = _prepare_window(ticker, lookback_months, baseline)
    report_index = report_prices.index
    split_at = max(min_train_days, int(len(report_index) * train_fraction))
    split_at = min(split_at, len(report_index) - min_validation_days)
    if split_at <= 0:
        raise RuntimeError(f"Not enough data for split at {lookback_months} months")
    train_index = report_index[:split_at]
    validation_index = report_index[split_at:]
    report_frame = frame.loc[config.start_date : config.end_date]

    sweep_rows: list[dict[str, Any]] = []
    for rule in rules:
        risk_mask, _ = risk_mask_for_rule(frame, rule)
        guarded = apply_risk_off_guard(baseline_signals, risk_mask, report_index)
        train_metrics = _evaluate_layer(
            train_index,
            report_prices,
            report_frame,
            config,
            baseline_signals,
            guarded,
            "train",
        )
        score = _score_train(train_metrics, min_blocked_days)
        guarded_train = train_metrics[train_metrics["layer"] == "risk_guarded"].iloc[0]
        baseline_train = train_metrics[train_metrics["layer"] == "baseline"].iloc[0]
        sweep_rows.append(
            {
                "lookback_months": lookback_months,
                "label": rule.label,
                "train_score": score,
                "train_baseline_annualized_return": baseline_train["annualized_return"],
                "train_guarded_annualized_return": guarded_train["annualized_return"],
                "train_baseline_sharpe_ratio": baseline_train["sharpe_ratio"],
                "train_guarded_sharpe_ratio": guarded_train["sharpe_ratio"],
                "train_baseline_direction_precision": baseline_train["direction_precision"],
                "train_guarded_direction_precision": guarded_train["direction_precision"],
                "train_baseline_max_drawdown": baseline_train["max_drawdown"],
                "train_guarded_max_drawdown": guarded_train["max_drawdown"],
                "train_blocked_days": guarded_train["blocked_days"],
                "train_blocked_loss_rate": guarded_train["blocked_loss_rate"],
                "train_blocked_avg_next_return": guarded_train["blocked_avg_next_return"],
                **{f"rule_{key}": value for key, value in asdict(rule).items()},
            }
        )

    sweep = pd.DataFrame(sweep_rows).sort_values("train_score", ascending=False)
    best_rule = rules[[rule.label for rule in rules].index(sweep.iloc[0]["label"])]
    best_mask, diagnostics = risk_mask_for_rule(frame, best_rule)
    diagnostics = diagnostics.reindex(report_index)
    best_guarded = apply_risk_off_guard(baseline_signals, best_mask, report_index)

    final_metrics = pd.concat(
        [
            _evaluate_layer(train_index, report_prices, report_frame, config, baseline_signals, best_guarded, "train"),
            _evaluate_layer(validation_index, report_prices, report_frame, config, baseline_signals, best_guarded, "validation"),
            _evaluate_layer(report_index, report_prices, report_frame, config, baseline_signals, best_guarded, "full"),
        ],
        ignore_index=True,
    )
    validation_metrics = final_metrics[final_metrics["segment"] == "validation"]
    production_pass = _passes_production_gate(validation_metrics, min_blocked_days=max(1, min_blocked_days // 2))
    watchlist_guard = _is_watchlist_guard(validation_metrics, min_blocked_days=max(1, min_blocked_days // 2))
    latest = latest_risk_snapshot(baseline, frame, best_rule, price_override=price_override)

    return {
        "lookback_months": lookback_months,
        "config": config,
        "split": {
            "train_start": train_index.min().strftime("%Y-%m-%d"),
            "train_end": train_index.max().strftime("%Y-%m-%d"),
            "validation_start": validation_index.min().strftime("%Y-%m-%d"),
            "validation_end": validation_index.max().strftime("%Y-%m-%d"),
            "train_days": int(len(train_index)),
            "validation_days": int(len(validation_index)),
        },
        "best_rule": best_rule,
        "sweep": sweep,
        "diagnostics": diagnostics,
        "guarded_signals": best_guarded,
        "final_metrics": final_metrics,
        "production_pass": production_pass,
        "watchlist_guard": watchlist_guard,
        "latest": latest,
    }


def latest_risk_snapshot(
    baseline: pd.Series,
    frame: pd.DataFrame,
    rule: RiskOffRule,
    price_override: float | None,
) -> dict[str, Any]:
    baseline_latest = latest_prediction_for_rule(rule_from_row(baseline), frame, price_override)
    mask, diagnostics = risk_mask_for_rule(frame, rule)
    risk_off = bool(mask.iloc[-1])
    baseline_signal = int(baseline_latest["signal"])
    return {
        "date": baseline_latest["date"],
        "as_of_date": baseline_latest["as_of_date"],
        "baseline_signal": baseline_signal,
        "risk_off": int(risk_off),
        "would_block_baseline_long": int(risk_off and baseline_signal == 1),
        "current_price": baseline_latest["current_price"],
        "risk_votes": float(diagnostics["risk_votes"].iloc[-1]),
        "rule": rule.label,
        "reason": (
            "risk guard would block a baseline long"
            if risk_off and baseline_signal == 1
            else "no baseline long is blocked"
        ),
    }


def write_report(output_dir: Path, payload: dict[str, Any], summary: pd.DataFrame) -> Path:
    display = summary.copy()
    for column in [
        "annualized_return",
        "max_drawdown",
        "win_rate",
        "direction_precision",
        "avg_next_return",
        "median_next_return",
        "worst_year_precision",
        "blocked_loss_rate",
        "blocked_avg_next_return",
        "blocked_median_next_return",
        "blocked_loss_capture",
    ]:
        if column in display:
            display[column] = display[column].map(_format_pct)
    for column in ["sharpe_ratio", "profit_factor", "brier_active"]:
        if column in display:
            display[column] = display[column].map(_format_num)

    lines = [
        "# NVDA Risk-Off Guard Backtest",
        "",
        "## Purpose",
        "",
        "This module tests a BLOCK_LONG layer. It can only remove baseline long exposure; it cannot add trades or short the stock.",
        "",
        "## Validation Policy",
        "",
        "- Select guard parameters on the train segment only.",
        "- Evaluate production eligibility on the validation segment only.",
        "- Production pass requires no drop in annualized return, Sharpe, or direction precision, and no meaningful drawdown increase.",
        "",
        "## Overall Decision",
        "",
        f"- Production-ready windows: {payload['production_ready_windows']} / {payload['window_count']}",
        f"- Watchlist-only windows: {payload['watchlist_windows']} / {payload['window_count']}",
        f"- Decision: {payload['decision']}",
        "",
        "## Metrics",
        "",
        display.to_markdown(index=False),
        "",
        "## Best Rules",
        "",
    ]
    for item in payload["windows"]:
        latest = item["latest"]
        lines.extend(
            [
                f"### {item['lookback_months']}M",
                "",
                f"- Rule: {item['best_rule']['label']}",
                f"- Production pass: {item['production_pass']}",
                f"- Watchlist guard: {item['watchlist_guard']}",
                f"- Train: {item['split']['train_start']} to {item['split']['train_end']} ({item['split']['train_days']} sessions)",
                f"- Validation: {item['split']['validation_start']} to {item['split']['validation_end']} ({item['split']['validation_days']} sessions)",
                f"- Latest baseline signal: {latest['baseline_signal']}",
                f"- Latest risk off: {latest['risk_off']}",
                f"- Would block latest baseline long: {latest['would_block_baseline_long']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            f"- Summary metrics: {output_dir / 'risk_off_guard_metrics.csv'}",
            f"- Sweep rows: {output_dir / 'risk_off_guard_sweep.csv'}",
            f"- Latest snapshots: {output_dir / 'risk_off_guard_latest.json'}",
        ]
    )
    path = output_dir / "risk_off_guard_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_risk_off_guard(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_row(Path(args.promotion_dir), args.baseline_label)
    rules = candidate_risk_rules()

    windows: list[dict[str, Any]] = []
    all_sweeps: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for lookback in [int(value.strip()) for value in args.lookback_months.split(",") if value.strip()]:
        result = evaluate_window(
            args.ticker,
            lookback,
            baseline,
            rules,
            args.train_fraction,
            args.min_train_days,
            args.min_validation_days,
            args.min_blocked_days,
            args.price_override,
        )
        sweep = result["sweep"].copy()
        metrics = result["final_metrics"].copy()
        metrics.insert(0, "lookback_months", lookback)
        sweep.to_csv(output_dir / f"risk_off_guard_sweep_{lookback}m.csv", index=False)
        metrics.to_csv(output_dir / f"risk_off_guard_metrics_{lookback}m.csv", index=False)
        result["guarded_signals"].to_csv(output_dir / f"risk_off_guard_signals_{lookback}m.csv")
        result["diagnostics"].to_csv(output_dir / f"risk_off_guard_diagnostics_{lookback}m.csv")
        all_sweeps.append(sweep)
        all_metrics.append(metrics)
        windows.append(
            {
                "lookback_months": lookback,
                "split": result["split"],
                "best_rule": {"label": result["best_rule"].label, **asdict(result["best_rule"])},
                "production_pass": result["production_pass"],
                "watchlist_guard": result["watchlist_guard"],
                "latest": result["latest"],
            }
        )

    sweep_table = pd.concat(all_sweeps, ignore_index=True)
    metrics_table = pd.concat(all_metrics, ignore_index=True)
    sweep_table.to_csv(output_dir / "risk_off_guard_sweep.csv", index=False)
    metrics_table.to_csv(output_dir / "risk_off_guard_metrics.csv", index=False)
    production_ready = sum(1 for window in windows if window["production_pass"])
    watchlist_ready = sum(1 for window in windows if window["watchlist_guard"])
    decision = (
        "production_candidate"
        if production_ready == len(windows)
        else "watchlist_only" if watchlist_ready > 0 else "reject_for_now"
    )
    payload = {
        "baseline_label": args.baseline_label,
        "window_count": len(windows),
        "production_ready_windows": production_ready,
        "watchlist_windows": watchlist_ready,
        "decision": decision,
        "windows": windows,
    }
    (output_dir / "risk_off_guard_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, payload, metrics_table)
    payload["report_path"] = str(report_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a BLOCK_LONG risk-off guard for the NVDA baseline rule")
    parser.add_argument("--promotion-dir", default=str(PROJECT_ROOT / "outputs" / "candidate_promotion"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "risk_off_guard"))
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--lookback-months", default="24,36,60")
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--min-validation-days", type=int, default=90)
    parser.add_argument("--min-blocked-days", type=int, default=3)
    parser.add_argument("--price-override", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    run_risk_off_guard(parse_args())


if __name__ == "__main__":
    main()
