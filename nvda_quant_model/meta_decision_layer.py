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
from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.news_sentiment import fetch_live_news, save_news_outputs
from nvda_quant_model.options_volatility import analyze_options
from nvda_quant_model.precision_search import build_walk_forward_rule_signals, latest_prediction_for_rule
from nvda_quant_model.reaction_pool import (
    _panel_for_rules,
    build_reaction_signals,
    directional_metrics,
    latest_reaction_snapshot,
    load_reaction_candidates,
)
from nvda_quant_model.risk_off_guard import (
    RiskOffRule,
    load_baseline_row,
    risk_mask_for_rule,
)
from nvda_quant_model.strict_model_selection import rule_from_row


DECISION_PRIORITY = {
    "NO_TRADE": 0,
    "WATCH": 1,
    "RISK_OFF": 2,
    "TACTICAL_LONG": 3,
    "BASELINE_LONG": 4,
    "BLOCK_LONG": 5,
}


@dataclass(frozen=True)
class MetaPolicy:
    name: str
    allow_reaction_trades: bool = False
    allow_risk_blocks: bool = False
    reaction_validation_pass: bool = False
    risk_validation_pass: bool = False
    notes: str = ""


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


def _zero_signals(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "direction": 0,
            "confidence": 0.0,
            "expected_return": 0.0,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "position": 0.0,
        },
        index=index,
    )


def _align_signal_frame(signals: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    defaults = {
        "direction": 0,
        "confidence": 0.0,
        "expected_return": 0.0,
        "prob_up": 0.5,
        "prob_down": 0.5,
        "position": 0.0,
    }
    aligned = signals.reindex(index).copy()
    for column, default in defaults.items():
        if column not in aligned:
            aligned[column] = default
    return aligned.fillna(defaults)


def build_meta_signals(
    baseline_signals: pd.DataFrame,
    reaction_signals: pd.DataFrame,
    risk_mask: pd.Series,
    report_index: pd.Index,
    policy: MetaPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _align_signal_frame(baseline_signals, report_index)
    reaction = _align_signal_frame(reaction_signals, report_index)
    risk_off = risk_mask.reindex(report_index).fillna(False).astype(bool)

    meta = baseline.copy()
    meta["decision"] = "NO_TRADE"
    meta["decision_code"] = DECISION_PRIORITY["NO_TRADE"]
    meta["source_layer"] = "none"
    meta["risk_off"] = risk_off.astype(int)
    meta["reaction_monitor"] = (reaction["position"] > 0).astype(int)
    meta["baseline_position_before_meta"] = baseline["position"]
    meta["reaction_position_before_meta"] = reaction["position"]

    baseline_long = baseline["position"] > 0
    reaction_long = (reaction["position"] > 0) & ~baseline_long
    blocked = baseline_long & risk_off & policy.allow_risk_blocks
    baseline_allowed = baseline_long & ~blocked
    tactical = reaction_long & ~risk_off if policy.allow_reaction_trades else pd.Series(False, index=report_index)
    reaction_watch = reaction_long if not policy.allow_reaction_trades else pd.Series(False, index=report_index)
    risk_only = risk_off & ~baseline_allowed & ~blocked & ~tactical

    meta.loc[:, ["direction", "confidence", "expected_return", "prob_up", "prob_down", "position"]] = 0.0
    meta.loc[:, ["prob_up", "prob_down"]] = 0.5

    for column in ["direction", "confidence", "expected_return", "prob_up", "prob_down", "position"]:
        meta.loc[baseline_allowed, column] = baseline.loc[baseline_allowed, column]
    meta.loc[baseline_allowed, "decision"] = "BASELINE_LONG"
    meta.loc[baseline_allowed, "source_layer"] = "baseline"

    for column in ["direction", "confidence", "expected_return", "prob_up", "prob_down", "position"]:
        meta.loc[tactical, column] = reaction.loc[tactical, column]
    meta.loc[tactical, "decision"] = "TACTICAL_LONG"
    meta.loc[tactical, "source_layer"] = "reaction"

    meta.loc[blocked, "decision"] = "BLOCK_LONG"
    meta.loc[blocked, "source_layer"] = "risk_off"
    meta.loc[risk_only, "decision"] = "RISK_OFF"
    meta.loc[risk_only, "source_layer"] = "risk_off"
    watch = reaction_watch & ~risk_only & ~blocked
    meta.loc[watch, "decision"] = "WATCH"
    meta.loc[watch, "source_layer"] = "reaction_monitor"
    meta["decision_code"] = meta["decision"].map(DECISION_PRIORITY).fillna(0).astype(int)
    meta["blocked_long"] = blocked.astype(int)
    meta["tactical_long"] = tactical.astype(int)
    meta["policy_name"] = policy.name

    diagnostics = meta[
        [
            "decision",
            "decision_code",
            "source_layer",
            "risk_off",
            "blocked_long",
            "tactical_long",
            "reaction_monitor",
            "baseline_position_before_meta",
            "reaction_position_before_meta",
            "position",
            "prob_up",
            "expected_return",
        ]
    ].copy()
    diagnostics["allow_reaction_trades"] = int(policy.allow_reaction_trades)
    diagnostics["allow_risk_blocks"] = int(policy.allow_risk_blocks)
    return meta, diagnostics


def _risk_rule_from_payload(payload: dict[str, Any], lookback_months: int) -> tuple[RiskOffRule | None, bool, bool]:
    windows = payload.get("windows") or []
    if not windows:
        return None, False, False
    exact = [row for row in windows if int(row.get("lookback_months", -1)) == lookback_months]
    selected = exact[0] if exact else min(windows, key=lambda row: abs(int(row.get("lookback_months", 0)) - lookback_months))
    raw_rule = selected.get("best_rule") or {}
    fields = {name: raw_rule[name] for name in RiskOffRule.__dataclass_fields__ if name in raw_rule}
    rule = RiskOffRule(**fields) if fields else None
    production_ready = bool(payload.get("decision") == "production_candidate" and selected.get("production_pass"))
    watchlist_ready = bool(selected.get("production_pass") or selected.get("watchlist_guard"))
    return rule, production_ready, watchlist_ready


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_inputs(
    ticker: str,
    lookback_months: int,
    baseline: pd.Series,
    news_history_path: Path | None,
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
    news_features = None
    if news_history_path is not None and news_history_path.exists():
        from nvda_quant_model.news_sentiment import load_news_feature_cache

        news_features = load_news_feature_cache(news_history_path, prices.index)
    frame, _ = build_model_frame(
        prices,
        external,
        config.start_date,
        config.end_date,
        config.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
        news_features=news_features,
    )
    report_prices = prices.loc[config.start_date : config.end_date]
    baseline_signals, _ = build_walk_forward_rule_signals(frame, rule_from_row(baseline), config.start_date)
    baseline_signals = baseline_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
    return config, prices, report_prices, frame, baseline_signals


def _build_reaction_layer(
    promotion_dir: Path,
    reaction_summary_path: Path,
    frame: pd.DataFrame,
    baseline_signals: pd.DataFrame,
    report_index: pd.Index,
    price_override: float | None,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    summary = _load_json(reaction_summary_path)
    validation_pass = bool(summary.get("validation_pass", False))
    params = summary.get("selected_params") or {}
    try:
        baseline_row, candidates = load_reaction_candidates(promotion_dir, top_n=int(params.get("top_n", 4)))
        panel, meta = _panel_for_rules(candidates, frame, str(report_index.min().date()))
        reaction, diagnostics = build_reaction_signals(
            baseline_signals,
            panel,
            meta.head(int(params.get("top_n", len(meta)))),
            report_index,
            int(params.get("min_vote_count", 3)),
            float(params.get("min_vote_weight", 0.20)),
            float(params.get("prob_threshold", 0.52)),
            float(params.get("max_shadow_exposure", 0.25)),
        )
        latest = latest_reaction_snapshot(
            baseline_row,
            candidates.head(int(params.get("top_n", len(candidates)))),
            frame.loc[frame.index <= report_index.max()],
            price_override,
            int(params.get("min_vote_count", 3)),
            float(params.get("prob_threshold", 0.52)),
        )
        latest["validation_pass"] = validation_pass
        latest["diagnostic_active_days"] = int(diagnostics["alert"].sum()) if "alert" in diagnostics else 0
        return reaction, latest, validation_pass
    except Exception as exc:
        return _zero_signals(report_index), {"validation_pass": validation_pass, "error": str(exc)}, False


def _decision_counts(diagnostics: pd.DataFrame) -> dict[str, int]:
    counts = diagnostics["decision"].value_counts().to_dict() if "decision" in diagnostics else {}
    return {decision: int(counts.get(decision, 0)) for decision in DECISION_PRIORITY}


def _metric_row(
    lookback_months: int,
    segment: str,
    layer: str,
    config: StrategyConfig,
    prices: pd.DataFrame,
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    scoped_signals = signals.reindex(prices.index).dropna(subset=["prob_up", "expected_return"])
    result = BacktestEngine(config).backtest(scoped_signals, prices)
    dmetrics = directional_metrics(scoped_signals, frame)
    row = {
        "lookback_months": lookback_months,
        "segment": segment,
        "layer": layer,
        **result.metrics,
        **dmetrics,
    }
    if diagnostics is not None:
        counts = _decision_counts(diagnostics.reindex(prices.index).dropna(how="all"))
        row.update({f"decision_{key.lower()}": value for key, value in counts.items()})
    return row


def _evaluate_segments(
    lookback_months: int,
    config: StrategyConfig,
    report_prices: pd.DataFrame,
    frame: pd.DataFrame,
    layers: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]],
    train_fraction: float,
    min_train_days: int,
    min_validation_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report_index = report_prices.index
    split_at = max(min_train_days, int(len(report_index) * train_fraction))
    split_at = min(split_at, len(report_index) - min_validation_days)
    if split_at <= 0:
        raise RuntimeError(f"Not enough data to split {lookback_months}M window")
    segments = {
        "train": report_index[:split_at],
        "validation": report_index[split_at:],
        "full": report_index,
    }
    rows = []
    report_frame = frame.loc[config.start_date : config.end_date]
    for segment, index in segments.items():
        prices = report_prices.reindex(index).dropna(subset=["Close"])
        scoped_frame = report_frame.reindex(prices.index)
        for layer, (signals, diagnostics) in layers.items():
            rows.append(
                _metric_row(
                    lookback_months,
                    segment,
                    layer,
                    config,
                    prices,
                    scoped_frame,
                    signals.reindex(prices.index),
                    diagnostics.reindex(prices.index) if diagnostics is not None else None,
                )
            )
    split = {
        "train_start": report_index[:split_at].min().strftime("%Y-%m-%d"),
        "train_end": report_index[:split_at].max().strftime("%Y-%m-%d"),
        "validation_start": report_index[split_at:].min().strftime("%Y-%m-%d"),
        "validation_end": report_index[split_at:].max().strftime("%Y-%m-%d"),
        "train_days": int(split_at),
        "validation_days": int(len(report_index) - split_at),
    }
    return pd.DataFrame(rows), split


def _passes_no_degradation(metrics: pd.DataFrame, layer: str) -> bool:
    validation = metrics[metrics["segment"] == "validation"]
    baseline = validation[validation["layer"] == "baseline"].iloc[0]
    candidate = validation[validation["layer"] == layer].iloc[0]
    return bool(
        _safe_float(candidate["annualized_return"]) >= _safe_float(baseline["annualized_return"]) - 0.0025
        and _safe_float(candidate["sharpe_ratio"]) >= _safe_float(baseline["sharpe_ratio"]) - 0.02
        and abs(_safe_float(candidate["max_drawdown"])) <= abs(_safe_float(baseline["max_drawdown"])) + 0.005
        and _safe_float(candidate["direction_precision"]) >= _safe_float(baseline["direction_precision"]) - 0.005
    )


def _latest_from_meta(
    meta_signals: pd.DataFrame,
    diagnostics: pd.DataFrame,
    baseline: pd.Series,
    frame: pd.DataFrame,
    config: StrategyConfig,
    price_override: float | None,
    live_overlays: dict[str, Any],
) -> dict[str, Any]:
    base_latest = latest_prediction_for_rule(rule_from_row(baseline), frame.loc[frame.index <= pd.Timestamp(config.end_date)], price_override)
    as_of = pd.Timestamp(base_latest["as_of_date"])
    signal_frame = meta_signals.loc[meta_signals.index <= as_of].tail(1)
    diag_frame = diagnostics.loc[diagnostics.index <= as_of].tail(1)
    if signal_frame.empty or diag_frame.empty:
        signal_frame = meta_signals.tail(1)
        diag_frame = diagnostics.tail(1)
    signal = signal_frame.iloc[-1].copy()
    diag = diag_frame.iloc[-1].copy()
    current_price = float(base_latest["current_price"])
    expected_return = float(signal.get("expected_return", 0.0))
    decision = str(diag.get("decision", "NO_TRADE"))
    live_flags: list[str] = []

    news = live_overlays.get("news_overlay") or {}
    if news.get("signal") == -1 and (news.get("risk_flags") or _safe_float(news.get("sentiment_score")) < -0.18):
        live_flags.append("news_bearish_or_risk")
        if float(signal.get("position", 0.0)) > 0:
            decision = "BLOCK_LONG"
            signal["position"] = 0.0
            signal["direction"] = 0
            signal["prob_up"] = 0.5
            signal["prob_down"] = 0.5
            signal["expected_return"] = 0.0
    options = live_overlays.get("options_volatility") or {}
    if options.get("skew_signal") == "downside_put_skew":
        live_flags.append("options_downside_put_skew")
    if options.get("volatility_signal") == "implied_vol_rich":
        live_flags.append("options_implied_vol_rich")
    order_flow = live_overlays.get("order_flow_overlay") or {}
    execution_filter = order_flow.get("execution_filter") or {}
    if execution_filter.get("recommendation") == "block_execution_wide_spread":
        live_flags.append("wide_spread_execution_block")
    micro = order_flow.get("micro_signal") or {}
    if micro.get("signal") == -1:
        live_flags.append("order_flow_bearish")

    return {
        "date": base_latest["date"],
        "weekday": base_latest["weekday"],
        "as_of_date": base_latest["as_of_date"],
        "current_price": current_price,
        "decision": decision,
        "position": round(float(signal.get("position", 0.0)), 4),
        "prob_up": round(float(signal.get("prob_up", 0.5)), 4),
        "expected_return": round(float(signal.get("expected_return", 0.0)), 6),
        "target_price": round(current_price * (1.0 + expected_return), 4),
        "stop_loss": round(current_price * (1.0 - config.stop_loss_pct), 4),
        "take_profit": round(current_price * (1.0 + config.take_profit_pct), 4),
        "source_layer": str(diag.get("source_layer", "none")),
        "baseline_signal": base_latest["signal"],
        "risk_off": int(diag.get("risk_off", 0)),
        "reaction_monitor": int(diag.get("reaction_monitor", 0)),
        "live_overlay_flags": live_flags,
        "live_overlays": live_overlays,
    }


def _load_live_overlays(args: argparse.Namespace, output_dir: Path, report_prices: pd.DataFrame, current_price: float) -> dict[str, Any]:
    overlays: dict[str, Any] = {}
    if args.news_overlay_json:
        overlays["news_overlay"] = _load_json(Path(args.news_overlay_json))
    elif args.include_live_news:
        try:
            news_dir = output_dir / "news_live"
            articles = fetch_live_news(days=args.news_days)
            overlays["news_overlay"] = save_news_outputs(articles, news_dir, price_index=report_prices.index)["overlay"]
        except Exception as exc:
            overlays["news_overlay"] = {
                "article_count": 0,
                "sentiment_score": 0.0,
                "signal": 0,
                "risk_flags": [f"news_fetch_failed:{exc}"],
            }

    if args.options_snapshot_json:
        try:
            overlays["options_volatility"] = analyze_options(
                Path(args.options_snapshot_json),
                current_price,
                output_dir / "two_week_range_projection.json",
                ticker=args.ticker,
            )
            (output_dir / "options_volatility_report.json").write_text(
                json.dumps(overlays["options_volatility"], indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        except Exception as exc:
            overlays["options_volatility"] = {"error": str(exc), "skew_signal": "unknown", "volatility_signal": "unknown"}

    if args.order_flow_json or args.include_live_order_flow:
        from nvda_quant_model.live_order_flow import analyze_order_flow, fetch_live_order_flow

        try:
            if args.order_flow_json:
                payload = _load_json(Path(args.order_flow_json))
                overlays["order_flow_overlay"] = (
                    payload if "micro_signal" in payload and "top_of_book" in payload else analyze_order_flow(payload, symbol=args.ticker)
                )
            else:
                overlays["order_flow_overlay"], bars = fetch_live_order_flow(args.ticker, args.order_flow_feed, args.order_flow_minutes)
                if bars is not None and not bars.empty:
                    order_dir = output_dir / "live_order_flow"
                    order_dir.mkdir(parents=True, exist_ok=True)
                    bars.to_csv(order_dir / f"{args.ticker}_recent_1min_bars.csv")
            order_dir = output_dir / "live_order_flow"
            order_dir.mkdir(parents=True, exist_ok=True)
            (order_dir / f"{args.ticker}_order_flow.json").write_text(
                json.dumps(overlays["order_flow_overlay"], indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        except Exception as exc:
            overlays["order_flow_overlay"] = {
                "error": str(exc),
                "micro_signal": {"signal": 0, "label": "unavailable", "score": 0.0, "confidence": 0.5},
                "execution_filter": {"use_as_entry_signal": False, "recommendation": "order_flow_unavailable"},
            }
    return overlays


def _pct(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.2%}"


def _num(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.2f}"


def write_report(output_dir: Path, payload: dict[str, Any], metrics: pd.DataFrame) -> Path:
    display = metrics.copy()
    for column in ["annualized_return", "max_drawdown", "win_rate", "direction_precision", "avg_next_return"]:
        if column in display:
            display[column] = display[column].map(_pct)
    for column in ["sharpe_ratio", "profit_factor", "brier_active"]:
        if column in display:
            display[column] = display[column].map(_num)
    keep = [
        "lookback_months",
        "segment",
        "layer",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "num_trades",
        "direction_precision",
        "active_days",
        "decision_baseline_long",
        "decision_tactical_long",
        "decision_block_long",
        "decision_risk_off",
        "decision_watch",
    ]
    keep = [col for col in keep if col in display]
    latest = payload["latest_decision"]
    lines = [
        "# NVDA Meta Decision Layer Backtest",
        "",
        "## Purpose",
        "",
        "This layer is the final decision outlet. It combines the baseline rule, reaction pool, risk-off guard, and live event overlays while keeping historical backtests point-in-time clean.",
        "",
        "## Production Rules",
        "",
        "- Baseline can trade when active.",
        "- Reaction can only create TACTICAL_LONG if its time-split validation passed.",
        "- Risk-off can only create BLOCK_LONG if the guard is production-approved across windows.",
        "- Live news/options/order-flow are latest overlays unless a point-in-time history is provided.",
        "",
        "## Decision",
        "",
        f"- Strict production pass windows: {payload['strict_pass_windows']} / {payload['window_count']}",
        f"- Research watchlist pass windows: {payload['research_pass_windows']} / {payload['window_count']}",
        f"- Production decision: {payload['production_decision']}",
        f"- Reaction validation pass: {payload['reaction_validation_pass']}",
        f"- Risk global decision: {payload['risk_global_decision']}",
        "",
        "## Latest Meta Decision",
        "",
        f"- Date: {latest['date']} ({latest['weekday']})",
        f"- Decision: {latest['decision']}",
        f"- Position: {latest['position']}",
        f"- Prob up: {_pct(latest['prob_up'])}",
        f"- Expected return: {_pct(latest['expected_return'])}",
        f"- Source layer: {latest['source_layer']}",
        f"- Baseline signal: {latest['baseline_signal']}",
        f"- Risk off: {latest['risk_off']}",
        f"- Reaction monitor: {latest['reaction_monitor']}",
        f"- Live overlay flags: {', '.join(latest['live_overlay_flags']) if latest['live_overlay_flags'] else 'none'}",
        "",
        "## Metrics",
        "",
        display[keep].to_markdown(index=False),
        "",
        "## Files",
        "",
        f"- Metrics: {output_dir / 'meta_decision_metrics.csv'}",
        f"- Latest: {output_dir / 'meta_decision_latest.json'}",
        f"- Summary: {output_dir / 'meta_decision_summary.json'}",
    ]
    path = output_dir / "meta_decision_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_meta_decision(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    promotion_dir = Path(args.promotion_dir)
    baseline = load_baseline_row(promotion_dir, args.baseline_label)
    reaction_summary = _load_json(Path(args.reaction_summary_json))
    risk_payload = _load_json(Path(args.risk_summary_json))
    reaction_validation_pass = bool(reaction_summary.get("validation_pass", False))

    all_metrics: list[pd.DataFrame] = []
    windows: list[dict[str, Any]] = []
    latest_decision: dict[str, Any] | None = None
    for lookback in [int(value.strip()) for value in args.lookback_months.split(",") if value.strip()]:
        config, _, report_prices, frame, baseline_signals = _prepare_inputs(
            args.ticker,
            lookback,
            baseline,
            Path(args.news_history_csv) if args.news_history_csv else None,
        )
        report_index = report_prices.index
        reaction_signals, reaction_latest, reaction_pass = _build_reaction_layer(
            promotion_dir,
            Path(args.reaction_summary_json),
            frame,
            baseline_signals,
            report_index,
            args.price_override,
        )
        risk_rule, risk_production_ready, risk_watchlist_ready = _risk_rule_from_payload(risk_payload, lookback)
        risk_mask = pd.Series(False, index=report_index)
        risk_rule_label = None
        if risk_rule is not None:
            risk_rule_label = risk_rule.label
            full_mask, _ = risk_mask_for_rule(frame, risk_rule)
            risk_mask = full_mask.reindex(report_index).fillna(False).astype(bool)

        strict_policy = MetaPolicy(
            name="strict_production",
            allow_reaction_trades=reaction_pass,
            allow_risk_blocks=risk_production_ready,
            reaction_validation_pass=reaction_pass,
            risk_validation_pass=risk_production_ready,
            notes="Only validated layers can change exposure.",
        )
        research_policy = MetaPolicy(
            name="watchlist_research",
            allow_reaction_trades=args.allow_watchlist_overlays,
            allow_risk_blocks=args.allow_watchlist_overlays and risk_watchlist_ready,
            reaction_validation_pass=reaction_pass,
            risk_validation_pass=risk_watchlist_ready,
            notes="Research mode can test watchlist overlays; not production by default.",
        )
        strict_signals, strict_diag = build_meta_signals(
            baseline_signals,
            reaction_signals,
            risk_mask,
            report_index,
            strict_policy,
        )
        research_signals, research_diag = build_meta_signals(
            baseline_signals,
            reaction_signals,
            risk_mask,
            report_index,
            research_policy,
        )
        layers = {
            "baseline": (baseline_signals, None),
            "meta_strict_production": (strict_signals, strict_diag),
            "meta_watchlist_research": (research_signals, research_diag),
        }
        metrics, split = _evaluate_segments(
            lookback,
            config,
            report_prices,
            frame,
            layers,
            args.train_fraction,
            args.min_train_days,
            args.min_validation_days,
        )
        strict_pass = _passes_no_degradation(metrics, "meta_strict_production")
        research_pass = _passes_no_degradation(metrics, "meta_watchlist_research")
        all_metrics.append(metrics)

        for name, signals, diag in [
            ("strict", strict_signals, strict_diag),
            ("research", research_signals, research_diag),
        ]:
            signals.to_csv(output_dir / f"meta_decision_{name}_signals_{lookback}m.csv")
            diag.to_csv(output_dir / f"meta_decision_{name}_diagnostics_{lookback}m.csv")

        live_overlays: dict[str, Any] = {}
        if lookback == int(args.lookback_months.split(",")[0].strip()):
            latest_base = latest_prediction_for_rule(rule_from_row(baseline), frame.loc[frame.index <= pd.Timestamp(config.end_date)], args.price_override)
            live_overlays = _load_live_overlays(args, output_dir, report_prices, float(latest_base["current_price"]))
            latest_decision = _latest_from_meta(
                strict_signals,
                strict_diag,
                baseline,
                frame,
                config,
                args.price_override,
                live_overlays,
            )

        windows.append(
            {
                "lookback_months": lookback,
                "split": split,
                "strict_pass": strict_pass,
                "research_pass": research_pass,
                "reaction_latest": reaction_latest,
                "risk_rule": risk_rule_label,
                "risk_production_ready": risk_production_ready,
                "risk_watchlist_ready": risk_watchlist_ready,
            }
        )

    metrics_table = pd.concat(all_metrics, ignore_index=True)
    metrics_table.to_csv(output_dir / "meta_decision_metrics.csv", index=False)
    strict_pass_windows = sum(1 for row in windows if row["strict_pass"])
    research_pass_windows = sum(1 for row in windows if row["research_pass"])
    production_decision = (
        "meta_orchestrator_ready_baseline_only"
        if strict_pass_windows == len(windows)
        else "do_not_promote"
    )
    payload = {
        "baseline_label": args.baseline_label,
        "window_count": len(windows),
        "strict_pass_windows": strict_pass_windows,
        "research_pass_windows": research_pass_windows,
        "production_decision": production_decision,
        "reaction_validation_pass": reaction_validation_pass,
        "risk_global_decision": risk_payload.get("decision", "unknown"),
        "allow_watchlist_overlays": args.allow_watchlist_overlays,
        "windows": windows,
        "latest_decision": latest_decision or {},
    }
    (output_dir / "meta_decision_latest.json").write_text(
        json.dumps(payload["latest_decision"], indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, payload, metrics_table)
    payload["report_path"] = str(report_path)
    (output_dir / "meta_decision_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and backtest the NVDA meta decision layer")
    parser.add_argument("--promotion-dir", default=str(PROJECT_ROOT / "outputs" / "candidate_promotion"))
    parser.add_argument("--reaction-summary-json", default=str(PROJECT_ROOT / "outputs" / "reaction_pool_time_split" / "reaction_pool_time_split_summary.json"))
    parser.add_argument("--risk-summary-json", default=str(PROJECT_ROOT / "outputs" / "risk_off_guard" / "risk_off_guard_latest.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "meta_decision_layer"))
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--baseline-label", default="tw147_sw63_10d_return_mq0.60_vq0.45_smh++obv++rsi<75+p60>0.96+vix<0.9")
    parser.add_argument("--lookback-months", default="24,36,60")
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--min-validation-days", type=int, default=90)
    parser.add_argument("--allow-watchlist-overlays", action="store_true")
    parser.add_argument("--news-history-csv", default="")
    parser.add_argument("--include-live-news", action="store_true")
    parser.add_argument("--news-overlay-json", default="")
    parser.add_argument("--news-days", type=int, default=3)
    parser.add_argument("--options-snapshot-json", default="")
    parser.add_argument("--include-live-order-flow", action="store_true")
    parser.add_argument("--order-flow-json", default="")
    parser.add_argument("--order-flow-feed", default="iex", choices=["iex", "sip", "delayed_sip", "boats", "overnight", "otc"])
    parser.add_argument("--order-flow-minutes", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    run_meta_decision(parse_args())


if __name__ == "__main__":
    main()
