#!/usr/bin/env python3
"""Macro event-study market forecast system.

Generates Markdown and CSV outputs requested in the task:

    python main.py --as-of 2026-05-10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.bayesian import bayesian_update_from_drift
from analysis.causal_channels import identify_channels
from analysis.copula import gaussian_copula_draws
from analysis.correlation_network import correlation_shift
from analysis.cross_asset import summarize_cross_assets
from analysis.event_study import run_event_study
from analysis.implied_vs_realized import compare_implied_realized
from analysis.iv_surface import extract_iv_surface
from analysis.matching import sample_counts
from analysis.noise_filter import annotate_noise
from analysis.pre_drift import current_event_pre_drift, historical_sample_pre_drifts
from analysis.robustness import bootstrap_sample_selection, leave_one_out, oos_backtest_placeholder, placebo_test, sensitivity_placeholder
from analysis.sector_rotation import summarize_assets_for_window, summarize_factor_spreads
from analysis.sentiment_filter import attach_environment_distances, build_environment_vector
from analysis.signal_matrix import build_signal_matrices
from analysis.weighting import assert_small_sample_weight_bound, build_weight_sets
from data.calendar import CalendarEvent, events_for_next_week
from data.events_db import load_historical_events, samples_for_event
from data.macro import load_macro_panel
from data.options import price_event_options
from data.prices import ensure_dir, load_price_panel
from data.sentiment import CORE_SENTIMENT_METRICS, build_sentiment
from report.markdown_writer import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Event-study market forecast for next week.")
    parser.add_argument("--as-of", default=None, help="Analysis as-of date, e.g. 2026-05-10.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--manual-events", default="manual_events.yaml", help="Manual event calendar YAML path.")
    parser.add_argument("--historical-events", default="historical_events.yaml", help="Historical event DB YAML path.")
    parser.add_argument("--noise-periods", default="noise_periods.yaml", help="Noise period YAML path.")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--no-live-options", action="store_true", help="Skip live option chain lookup.")
    parser.add_argument("--no-auto-fetch", action="store_true", help="Skip automatic sentiment web fetchers and only read manual_sentiment.yaml.")
    parser.add_argument(
        "--sentiment-mode",
        choices=["strict", "permissive"],
        default="strict",
        help="strict stops if core sentiment/positioning is unavailable; permissive emits BLACK/NaN diagnostics.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    as_of = pd.Timestamp(args.as_of or config.get("as_of_date") or pd.Timestamp.today()).normalize()
    output_dir = ensure_dir(args.output_dir or config["output_dir"])
    cache_dir = ensure_dir(config["cache_dir"])
    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    warnings: list[str] = []

    print(f"As of {as_of.date()} | loading event calendar...")
    events = events_for_next_week(as_of, args.manual_events)
    if not events:
        raise RuntimeError("No next-week events found. Add events to manual_events.yaml.")
    historical_db = load_historical_events(args.historical_events)

    print(f"Found {len(events)} events. Loading market data...")
    tickers = _all_tickers(config, events)
    start = as_of - pd.Timedelta(days=int(float(config.get("lookback_years", 100)) * 365.25) + 370)
    price_result = load_price_panel(tickers, start, as_of, cache_dir)
    prices = price_result.prices
    warnings.extend(price_result.warnings)
    macro, macro_warnings, _ = load_macro_panel(config.get("fred_series", {}), start, as_of, cache_dir)
    warnings.extend(macro_warnings)

    sentiment = build_sentiment(
        prices=prices,
        macro=macro,
        as_of_date=as_of,
        manual_path=config.get("sentiment", {}).get("manual_overrides_path"),
        enable_auto_fetch=bool(config.get("enable_auto_fetch", True)) and not args.no_auto_fetch,
        cache_dir=Path(cache_dir) / "sentiment",
    )
    warnings.extend(sentiment.warnings)
    source_summary = _sentiment_source_summary(sentiment.snapshot)
    current_env = build_environment_vector(sentiment.snapshot)
    min_env_dims = int(config.get("sentiment", {}).get("min_dims_required", 3))
    env_status = "ENABLED" if len(current_env) >= min_env_dims else "DISABLED"
    print(
        "Sentiment sources: "
        f"{source_summary['auto']} auto-fetched, "
        f"{source_summary['manual']} manual, "
        f"{source_summary['missing']} missing"
    )
    print(f"Environment matching: {env_status} ({len(current_env)} dimensions)")
    if env_status == "DISABLED":
        warnings.append(
            f"Environment matching disabled: {len(current_env)} available dimensions, "
            f"minimum required is {min_env_dims}. Fill manual_sentiment.yaml to improve matching."
        )

    all_event_rows: list[dict[str, Any]] = []
    all_samples: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_sector: list[pd.DataFrame] = []
    options_rows: list[dict[str, Any]] = []
    bayes_rows: list[dict[str, Any]] = []
    pre_drift_rows: list[dict[str, Any]] = []
    study_detail_by_event: dict[str, pd.DataFrame] = {}
    samples_by_event: dict[str, dict[str, pd.DataFrame]] = {}

    spot = _last_value(prices.get(config["primary_symbol"]), as_of)
    n_boot = int(config.get("bootstrap_iterations", 10000))
    windows = {name: [int(v[0]), int(v[1])] for name, v in config["windows"].items()}
    post_window = tuple(config["windows"]["post_1_5"])
    flat_assets = _flat_asset_map(config)
    macro_categories = set(config.get("macro_event_policy", {}).get("categories", []))

    for event in events:
        print(f"Analyzing {event.date.date()} {event.name}...")
        samples = samples_for_event(event, historical_db, config, as_of)
        samples = annotate_noise(samples, prices, args.noise_periods)
        macro_no_surprise = event.category in macro_categories and samples.empty
        samples = attach_environment_distances(
            samples,
            sentiment.history,
            as_of,
            nearest_fraction=float(config.get("sentiment", {}).get("nearest_fraction", 0.5)),
            snapshot_df=sentiment.snapshot,
            min_dims_required=min_env_dims,
        )
        env_meta = samples.attrs.get("env_match_metadata", {})
        if env_meta.get("status") != "success":
            warnings.append(f"{event.id}: environment matching {env_meta.get('status', 'unavailable')}.")
        momentum_history = pd.concat([sentiment.history, sentiment.momentum_history], axis=1)
        samples_with_momentum = attach_environment_distances(
            samples.copy(),
            momentum_history,
            as_of,
            nearest_fraction=float(config.get("sentiment", {}).get("nearest_fraction", 0.5)),
            snapshot_df=sentiment.snapshot,
            min_dims_required=min_env_dims,
        )
        samples["event_id"] = event.id
        samples["future_event_name"] = event.name
        all_samples.append(samples)
        clean = samples[~samples.get("excluded_noise", pd.Series(False, index=samples.index))]
        env_subset = clean[clean.get("environment_match", pd.Series(False, index=clean.index))]
        clean_momentum = samples_with_momentum[~samples_with_momentum.get("excluded_noise", pd.Series(False, index=samples_with_momentum.index))]
        momentum_subset = clean_momentum[clean_momentum.get("environment_match", pd.Series(False, index=clean_momentum.index))]
        samples_by_event[event.id] = {
            "full": samples,
            "clean": clean,
            "environment_macro_only": env_subset,
            "environment_with_momentum": momentum_subset,
        }

        full_result = run_event_study(samples, prices, config["primary_symbol"], config["benchmark_symbol"], windows, n_boot, rng, label="full")
        clean_result = run_event_study(clean, prices, config["primary_symbol"], config["benchmark_symbol"], windows, n_boot, rng, label="clean")
        env_result = run_event_study(env_subset, prices, config["primary_symbol"], config["benchmark_symbol"], windows, n_boot, rng, label="environment_macro_only")
        momentum_result = run_event_study(momentum_subset, prices, config["primary_symbol"], config["benchmark_symbol"], windows, n_boot, rng, label="environment_with_momentum")
        warnings.extend([f"{event.id}: {w}" for w in full_result.warnings + clean_result.warnings + env_result.warnings + momentum_result.warnings])
        study_detail = pd.concat([clean_result.detail], ignore_index=True) if not clean_result.detail.empty else pd.DataFrame()
        study_detail_by_event[event.id] = study_detail

        for result in [full_result, clean_result, env_result, momentum_result]:
            if result.summary.empty:
                continue
            summary = result.summary.copy()
            summary["event_id"] = event.id
            summary["event_name"] = event.name
            summary["matrix_type"] = "event_study"
            all_signals.append(summary)

        sector = summarize_assets_for_window(clean, prices, config.get("markets", {}).get("sectors", {}), config["benchmark_symbol"], post_window, n_boot, rng, "sector")
        if not sector.empty:
            sector["event_id"] = event.id
            all_sector.append(sector)
        factors = summarize_factor_spreads(clean, prices, config.get("factor_spreads", {}), post_window, n_boot, rng)
        if not factors.empty:
            factors["event_id"] = event.id
            factors["group"] = "factor"
            all_sector.append(factors)
        theme_assets = _theme_assets_for_event(event, config)
        theme = summarize_assets_for_window(clean, prices, theme_assets, config["benchmark_symbol"], post_window, n_boot, rng, "theme")
        if not theme.empty:
            theme["event_id"] = event.id
            all_sector.append(theme)

        cross = summarize_cross_assets(clean, prices, _cross_asset_map(config), config["benchmark_symbol"], post_window, n_boot, rng)
        if not cross.empty:
            cross["event_id"] = event.id
            cross["event_name"] = event.name
            cross["matrix_type"] = "cross_asset"
            all_signals.append(cross)

        pre_current = current_event_pre_drift(event, prices, config, as_of)
        sample_pre = historical_sample_pre_drifts(clean, event, prices, config)
        pre_drift_rows.append(pre_current)
        if not sample_pre.empty:
            pre_drift_rows.extend(sample_pre.to_dict("records"))
        posterior = bayesian_update_from_drift(event.id, study_detail, pre_current["current_pre_drift"], sample_pre_drifts=sample_pre)
        bayes_rows.append(posterior)

        option_payload = {}
        if not args.no_live_options and np.isfinite(spot):
            option_payload = price_event_options(event.id, event.date, config.get("options", {}).get("ticker", "SPY"), spot, as_of, cache_dir).to_dict()
        else:
            option_payload = _empty_option_row(event.id, event.date, spot)
        implied = compare_implied_realized(event.id, option_payload, study_detail)
        option_payload.update(implied)
        option_payload["skew_25d"] = option_payload.get("skew_25d", option_payload.get("option_skew_25d"))
        options_rows.append(option_payload)

        counts = sample_counts(samples)
        event_row = _event_row(event, counts, posterior, _policy_note(event, clean, study_detail))
        event_row["env_match_status"] = env_meta.get("status", "")
        event_row["env_used_dimensions"] = ",".join(env_meta.get("used_dimensions", []) or env_meta.get("available_dims", []))
        event_row["env_n_dimensions"] = int(env_meta.get("n_dimensions", len(env_meta.get("used_dimensions", []) or [])) or 0)
        event_row["env_distance_p50"] = env_meta.get("distance_p50", np.nan)
        event_row["macro_no_surprise"] = macro_no_surprise
        event_row["original_importance"] = event.importance
        if macro_no_surprise:
            event_row["importance"] = min(event.importance, int(config.get("macro_event_policy", {}).get("no_surprise_importance_cap", 2)))
            event_row["data_quality"] = "BLACK"
            event_row["data_quality_note"] = "Macro event has no true release-date + surprise history; no proxy samples used."
        else:
            event_row["data_quality"] = _event_quality(counts["clean"])
            event_row["data_quality_note"] = ""
        all_event_rows.append(event_row)

    events_detail = pd.DataFrame(all_event_rows)
    historical_samples = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    signals_matrix = pd.concat(all_signals, ignore_index=True, sort=False) if all_signals else pd.DataFrame()
    sector_rotation = pd.concat(all_sector, ignore_index=True, sort=False) if all_sector else pd.DataFrame()
    options_pricing = pd.DataFrame(options_rows)
    bayesian_posteriors = pd.DataFrame(bayes_rows)
    bayesian_pre_drift = pd.DataFrame(pre_drift_rows)

    weights = build_weight_sets(events_detail, bayesian_posteriors, signals_matrix, config)
    n_sims = int(config.get("monte_carlo_iterations", 100000))
    mc_summary, mc_full = _run_three_scheme_mc(bayesian_posteriors, weights, n_sims, int(config.get("random_seed", 42)))
    mc_components = weights.merge(bayesian_posteriors, on="event_id", how="left")
    if not mc_components.empty:
        mc_components["weight"] = mc_components["weight_composite"]
        mc_components["weighted_mean"] = mc_components["weight_composite"] * mc_components["posterior_mean"].fillna(0.0)

    if not bayesian_posteriors.empty:
        events_detail = events_detail.merge(
            bayesian_posteriors[["event_id", "posterior_mean", "posterior_sd", "p_up"]],
            on="event_id",
            how="left",
        )
    if not weights.empty:
        events_detail = events_detail.merge(
            weights[["event_id", "weight_equal", "weight_n_weighted", "weight_composite", "significance_score", "min_holm_p"]],
            on="event_id",
            how="left",
        )

    print("Building expanded signal matrix and diagnostics...")
    signals_long, signals_wide = build_signal_matrices(events, samples_by_event, prices, flat_assets, windows, n_boot, rng)
    clean_samples_by_event = {event_id: subsets.get("clean", pd.DataFrame()) for event_id, subsets in samples_by_event.items()}
    corr_shift = correlation_shift(clean_samples_by_event, prices, dict(list(flat_assets.items())[:40]))
    iv_surface = extract_iv_surface(
        config.get("options", {}).get("ticker", "SPY"),
        spot,
        as_of,
        config.get("options", {}).get("expiries_days", [7, 14, 30, 90, 180, 365]),
        cache_dir,
    ) if not args.no_live_options and np.isfinite(spot) else pd.DataFrame()
    robustness_loo = leave_one_out(clean_samples_by_event, study_detail_by_event)
    robustness_bootstrap = bootstrap_sample_selection(study_detail_by_event, seed=int(config.get("random_seed", 42)))
    robustness_placebo = placebo_test(prices, seed=int(config.get("random_seed", 42)))
    robustness_sensitivity = sensitivity_placeholder()
    robustness_oos = oos_backtest_placeholder()
    causal_channels = pd.concat(
        [identify_channels(event.id, signals_matrix[signals_matrix.get("matrix_type", "") == "cross_asset"], sector_rotation) for event in events],
        ignore_index=True,
        sort=False,
    ) if events else pd.DataFrame()
    alt_data_signals = _alt_data_placeholder(events)
    policy_path_predictions = _policy_paths(events, events_detail)
    mispricing_opportunities = options_pricing[options_pricing.get("signal", pd.Series(dtype=str)).astype(str).str.contains("候选", na=False)].copy()
    factor_attribution = sector_rotation[sector_rotation.get("group", pd.Series(dtype=str)) == "factor"].copy() if not sector_rotation.empty else pd.DataFrame()
    cross_asset_matrix = signals_matrix[signals_matrix.get("matrix_type", pd.Series(dtype=str)) == "cross_asset"].copy() if not signals_matrix.empty else pd.DataFrame()
    sector_csv = sector_rotation[sector_rotation.get("group", pd.Series(dtype=str)) == "sector"].copy() if not sector_rotation.empty else pd.DataFrame()
    subindustry_csv = sector_rotation[sector_rotation.get("group", pd.Series(dtype=str)).isin(["theme"])].copy() if not sector_rotation.empty else pd.DataFrame()
    environment_matching = _environment_matching_summary(events_detail, historical_samples)

    confidence_label, confidence_note = _confidence(events_detail, bayesian_posteriors)
    context = {
        "as_of_date": as_of.date().isoformat(),
        "confidence_label": confidence_label,
        "confidence_note": confidence_note,
        "sentiment_available": int(source_summary["available"]),
        "sentiment_total": int(source_summary["total"]),
        "sentiment_missing": source_summary["missing_metrics"],
        "sentiment_auto_loaded": int(source_summary["auto"]),
        "sentiment_manual_loaded": int(source_summary["manual"]),
        "environment_status": env_status,
        "environment_dimensions": list(current_env.keys()),
    }

    print("Writing CSV and Markdown outputs...")
    _write_csv(output_dir / "events_detail.csv", events_detail)
    _write_csv(output_dir / "historical_samples.csv", historical_samples)
    _write_csv(output_dir / "environment_matching.csv", environment_matching)
    _write_csv(output_dir / "signals_matrix.csv", signals_matrix)
    _write_csv(output_dir / "signals_matrix_long.csv", signals_long)
    _write_csv(output_dir / "signals_matrix_wide.csv", signals_wide)
    _write_csv(output_dir / "sector_rotation.csv", sector_csv)
    _write_csv(output_dir / "subindustry_rotation.csv", subindustry_csv)
    _write_csv(output_dir / "factor_attribution.csv", factor_attribution)
    _write_csv(output_dir / "cross_asset_matrix.csv", cross_asset_matrix)
    _write_csv(output_dir / "correlation_shift_matrix.csv", corr_shift)
    _write_csv(output_dir / "options_surface.csv", iv_surface)
    _write_csv(output_dir / "sentiment_snapshot.csv", sentiment.snapshot)
    _write_csv(output_dir / "sentiment_historical.csv", sentiment.history.reset_index().rename(columns={"index": "date"}))
    _write_csv(output_dir / "options_pricing.csv", options_pricing)
    _write_csv(output_dir / "bayesian_posteriors.csv", bayesian_posteriors)
    _write_csv(output_dir / "bayesian_pre_drift.csv", bayesian_pre_drift)
    _write_csv(output_dir / "monte_carlo_paths.csv", mc_summary)
    _write_csv(output_dir / "monte_carlo_full_distribution.csv", mc_full)
    _write_csv(output_dir / "robustness_loo.csv", robustness_loo)
    _write_csv(output_dir / "robustness_bootstrap.csv", robustness_bootstrap)
    _write_csv(output_dir / "robustness_sensitivity.csv", robustness_sensitivity)
    _write_csv(output_dir / "robustness_placebo.csv", robustness_placebo)
    _write_csv(output_dir / "robustness_oos.csv", robustness_oos)
    _write_csv(output_dir / "causal_channels.csv", causal_channels)
    _write_csv(output_dir / "alt_data_signals.csv", alt_data_signals)
    _write_csv(output_dir / "policy_path_predictions.csv", policy_path_predictions)
    _write_csv(output_dir / "mispricing_opportunities.csv", mispricing_opportunities)

    write_report(
        output_path=output_dir / "final_report.md",
        context=context,
        events_detail=events_detail,
        sentiment_snapshot=sentiment.snapshot,
        options_pricing=options_pricing,
        bayesian_posteriors=bayesian_posteriors,
        mc_summary=mc_summary[mc_summary["scheme"] == "composite"] if "scheme" in mc_summary else mc_summary,
        mc_components=mc_components,
        sector_rotation=sector_csv,
        signals_matrix=signals_matrix,
        historical_samples=historical_samples,
        warnings=warnings,
    )
    _write_markdown_extras(output_dir, context, events_detail, sentiment.snapshot, warnings, weights, mc_summary)
    print(f"Done. Report: {output_dir / 'final_report.md'}")
    return 0


def _all_tickers(config: dict[str, Any], events: list[CalendarEvent]) -> list[str]:
    tickers: set[str] = set(_flat_asset_map(config).values())
    tickers.add(config["primary_symbol"])
    tickers.add(config["benchmark_symbol"])
    for pair in config.get("factor_spreads", {}).values():
        tickers.update(pair)
    for basket in config.get("event_pre_drift_baskets", {}).values():
        tickers.update(basket)
    for category in config.get("theme_baskets", {}).values():
        for basket in category.values():
            tickers.update(basket)
    for event in events:
        tickers.update(event.theme_tickers or [])
    tickers.add(config.get("options", {}).get("ticker", "SPY"))
    return sorted(t for t in tickers if t)


def _flat_asset_map(config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group, mapping in config.get("markets", {}).items():
        if not isinstance(mapping, dict):
            continue
        for name, symbol_or_label in mapping.items():
            if group in {"sectors", "sub_industries", "factors", "rates", "credit", "commodities"}:
                out[f"{group}:{name}"] = name
            else:
                out[f"{group}:{name}"] = str(symbol_or_label)
    for symbol in ["SPY", "QQQ", "IWM", "TLT", "GLD", "KWEB", "FXI", "SOXX", "ARKK"]:
        out.setdefault(f"extra:{symbol}", symbol)
    return out


def _cross_asset_map(config: dict[str, Any]) -> dict[str, str]:
    out = {}
    for group in ["major", "rates", "credit", "currencies", "commodities", "crypto", "international", "vol"]:
        mapping = config.get("markets", {}).get(group, {})
        for name, symbol_or_label in mapping.items():
            out[name] = name if group in {"rates", "credit", "commodities"} else str(symbol_or_label)
    return out


def _theme_assets_for_event(event: CalendarEvent, config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    category_map = config.get("theme_baskets", {}).get(event.category, {})
    for basket_name, tickers in category_map.items():
        for ticker in tickers:
            out[ticker] = basket_name
    for ticker in event.theme_tickers or []:
        out[ticker] = ticker
    return out


def _last_value(series: pd.Series | None, as_of: pd.Timestamp) -> float:
    if series is None:
        return float("nan")
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean.loc[clean.index <= as_of]
    return float(clean.iloc[-1]) if not clean.empty else float("nan")


def _current_pre_drift(series: pd.Series, as_of: pd.Timestamp, days: int = 5) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean = clean.loc[clean.index <= as_of]
    if clean.shape[0] <= days:
        return float("nan")
    returns = clean.pct_change(fill_method=None).dropna()
    return float(returns.tail(days).sum())


def _empty_option_row(event_id: str, event_date: pd.Timestamp, spot: float) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_date": event_date.date().isoformat(),
        "ticker": "SPY",
        "spot": spot,
        "expiration": "",
        "days_to_expiry": np.nan,
        "atm_strike": np.nan,
        "call_mid": np.nan,
        "put_mid": np.nan,
        "straddle_price": np.nan,
        "implied_move": np.nan,
        "atm_iv": np.nan,
        "next_atm_iv": np.nan,
        "term_slope": np.nan,
        "skew_25d": np.nan,
        "source": "skipped",
        "warnings": "live options skipped",
    }


def _event_row(event: CalendarEvent, counts: dict[str, int], posterior: dict[str, Any], policy_note: str) -> dict[str, Any]:
    return {
        "event_id": event.id,
        "event_name": event.name,
        "date": event.date.date().isoformat(),
        "end_date": event.end_date.date().isoformat() if event.end_date is not None else "",
        "category": event.category,
        "secondary_categories": ",".join(event.secondary_categories or []),
        "importance": event.importance,
        "geography": event.geography,
        "source": event.source,
        "notes": event.notes,
        "raw_samples": counts["raw"],
        "clean_samples": counts["clean"],
        "env_samples": counts["environment"],
        "policy_note": policy_note,
    }


def _policy_note(event: CalendarEvent, clean: pd.DataFrame, study_detail: pd.DataFrame) -> str:
    if event.category == "PRESIDENT_CHINA_VISIT":
        n = int(len(clean))
        return (
            f"历史模式: 样本 N={n}。外交访问后 30-60 日常见路径是声明性缓和先行、实质协议滞后；"
            "当前条件下最可能情景为关税/出口管制边际缓和声明 + 元首热线或工作组机制，"
            "但台湾与科技限制议题使左尾风险不能忽略。"
        )
    if event.category == "FED_CHAIR_TRANSITION":
        return "政策走向取决于提名人鹰鸽标签与参议院不确定性；历史样本显示长端利率、区域银行和黄金最敏感。"
    return "宏观数据类事件无政策路径预测；主要通过 surprise 改写 Fed path、实际利率和增长预期。"


def _weighted_average(values: pd.DataFrame, weights: pd.DataFrame, col: str) -> float:
    merged = weights.merge(values[["event_id", col]], on="event_id", how="left")
    valid = pd.to_numeric(merged[col], errors="coerce").notna()
    if not valid.any():
        return float("nan")
    return float(np.average(merged.loc[valid, col].astype(float), weights=merged.loc[valid, "weight"].astype(float)))


def _run_three_scheme_mc(posteriors: pd.DataFrame, weights: pd.DataFrame, n_sims: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    full = pd.DataFrame({"simulation": np.arange(int(n_sims))})
    for i, scheme in enumerate(["equal", "n_weighted", "composite"]):
        draws, _ = gaussian_copula_draws(posteriors, weights, scheme, n_sims, seed + i)
        if draws.size == 0:
            continue
        full[scheme] = draws
        summaries.append(_summarize_draws(draws, scheme))
    return pd.DataFrame(summaries), full


def _summarize_draws(draws: np.ndarray, scheme: str) -> dict[str, float | str]:
    quantiles = {f"p{int(q * 100):02d}": float(np.quantile(draws, q)) for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]}
    return {
        "scheme": scheme,
        "n_sims": int(draws.size),
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "p25": quantiles["p25"],
        "p75": quantiles["p75"],
        "p_gt_0": float(np.mean(draws > 0)),
        "p_gt_1pct": float(np.mean(draws > 0.01)),
        "p_gt_2pct": float(np.mean(draws > 0.02)),
        "p_lt_minus_1pct": float(np.mean(draws < -0.01)),
        "p_lt_minus_2pct": float(np.mean(draws < -0.02)),
        "p_lt_minus_3pct": float(np.mean(draws < -0.03)),
        "tail_cvar_p05": float(np.mean(draws[draws <= quantiles["p05"]])) if np.any(draws <= quantiles["p05"]) else np.nan,
        "tail_cvar_p95": float(np.mean(draws[draws >= quantiles["p95"]])) if np.any(draws >= quantiles["p95"]) else np.nan,
        **quantiles,
    }


def _event_quality(n_clean: int) -> str:
    if n_clean <= 0:
        return "BLACK"
    if n_clean < 10:
        return "RED"
    if n_clean < 30:
        return "YELLOW"
    return "GREEN"


def _sentiment_source_summary(snapshot: pd.DataFrame) -> dict[str, Any]:
    total = len(CORE_SENTIMENT_METRICS)
    if snapshot is None or snapshot.empty:
        return {
            "total": total,
            "available": 0,
            "auto": 0,
            "manual": 0,
            "missing": total,
            "missing_metrics": CORE_SENTIMENT_METRICS.copy(),
        }
    core = snapshot[snapshot["metric"].isin(CORE_SENTIMENT_METRICS)].copy()
    available = core[core["status"].eq("available")]
    auto = int(available["source"].astype(str).str.contains("auto|AAII|NAAIM|CFTC|FINRA|ICI|CBOE|cboe|aaii|naaim|cftc|finra", case=False, regex=True).sum())
    manual = int(available["source"].astype(str).str.contains("manual", case=False, regex=False).sum())
    auto = int(snapshot.attrs.get("auto_loaded", auto))
    manual = int(snapshot.attrs.get("manual_loaded", manual))
    missing_metrics = [
        metric for metric in CORE_SENTIMENT_METRICS
        if core.loc[core["metric"] == metric, "status"].empty or core.loc[core["metric"] == metric, "status"].iloc[0] != "available"
    ]
    return {
        "total": total,
        "available": int(available.shape[0]),
        "auto": auto,
        "manual": manual,
        "missing": len(missing_metrics),
        "missing_metrics": missing_metrics,
    }


def _environment_matching_summary(events_detail: pd.DataFrame, historical_samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if events_detail is None or events_detail.empty:
        return pd.DataFrame()
    for _, event in events_detail.iterrows():
        event_id = event.get("event_id")
        subset = historical_samples[historical_samples.get("event_id", pd.Series(dtype=str)) == event_id] if historical_samples is not None and not historical_samples.empty else pd.DataFrame()
        rows.append(
            {
                "event_id": event_id,
                "event_name": event.get("event_name", ""),
                "status": event.get("env_match_status", ""),
                "used_dimensions": event.get("env_used_dimensions", ""),
                "n_dimensions": event.get("env_n_dimensions", 0),
                "env_samples": event.get("env_samples", 0),
                "distance_p50": event.get("env_distance_p50", np.nan),
                "avg_distance": pd.to_numeric(subset.get("env_distance", pd.Series(dtype=float)), errors="coerce").mean() if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _alt_data_placeholder(events: list[CalendarEvent]) -> pd.DataFrame:
    rows = []
    sources = ["Google Trends", "GDELT/News sentiment", "Capitol Trades", "SEC Form 4", "GPR", "EPU", "Shipping rates"]
    for event in events:
        for source in sources:
            rows.append({"event_id": event.id, "source": source, "value": np.nan, "status": "BLACK", "note": "Not configured or unavailable; no synthetic value generated."})
    return pd.DataFrame(rows)


def _policy_paths(events: list[CalendarEvent], events_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events:
        if event.category == "PRESIDENT_CHINA_VISIT":
            path = "Tariff/export-control tone easing statement; concrete enforcement details uncertain."
        elif event.category == "FED_CHAIR_TRANSITION":
            path = "Nomination/confirmation risk transmits through policy reaction-function expectations."
        else:
            path = "Macro release changes policy path only through actual surprise; surprise history unavailable."
        rows.append({"event_id": event.id, "policy_path": path, "status": "YELLOW" if event.category in {"PRESIDENT_CHINA_VISIT", "FED_CHAIR_TRANSITION"} else "BLACK"})
    return pd.DataFrame(rows)


def _write_sentiment_blocker(output_dir: Path, snapshot: pd.DataFrame, warnings: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output_dir / "sentiment_snapshot.csv", index=False)
    template = Path("manual_sentiment.yaml")
    (output_dir / "data_quality_report.md").write_text(
        "# Data Quality Report\n\n"
        "BLACK: core sentiment/positioning metrics are unavailable, so v2 strict mode stopped before event analysis.\n\n"
        f"Fill `{template}` with current AAII, CFTC, CBOE put/call, NAAIM, FINRA margin debt, and ICI flow values, then rerun.\n\n"
        "Warnings:\n" + "\n".join(f"- {w}" for w in warnings),
        encoding="utf-8",
    )


def _manual_core_count(path: str | Path | None) -> int:
    if not path or not Path(path).exists():
        return 0
    payload = yaml.safe_load(Path(path).read_text()) or {}
    core = {
        "AAII_bull_bear_spread",
        "CFTC_emini_spx_net_percentile",
        "CFTC_10y_ust_net_percentile",
        "CBOE_equity_pc_10dma",
        "CBOE_total_pc_10dma",
        "NAAIM_exposure",
        "FINRA_margin_debt_yoy",
        "ICI_4w_equity_flow_b",
    }
    count = 0
    for item in payload.get("metrics", []):
        if item.get("metric") in core and item.get("value") is not None:
            try:
                if np.isfinite(float(item["value"])):
                    count += 1
            except Exception:
                pass
    return count


def _write_markdown_extras(
    output_dir: Path,
    context: dict[str, Any],
    events_detail: pd.DataFrame,
    sentiment_snapshot: pd.DataFrame,
    warnings: list[str],
    weights: pd.DataFrame,
    mc_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    composite = mc_summary[mc_summary["scheme"] == "composite"].iloc[0] if not mc_summary.empty and "scheme" in mc_summary and (mc_summary["scheme"] == "composite").any() else None
    summary_lines = [f"# Executive Summary ({context['as_of_date']})", ""]
    if composite is not None:
        summary_lines.append(f"- Composite median: {float(composite['median']):.2%}")
        summary_lines.append(f"- P(up): {float(composite['p_gt_0']):.1%}")
        summary_lines.append(f"- P(<-2%): {float(composite['p_lt_minus_2pct']):.1%}")
    summary_lines.append(f"- Confidence: {context.get('confidence_label')} - {context.get('confidence_note')}")
    summary_lines.append(
        f"- Sentiment: {context.get('sentiment_available', 0)}/{context.get('sentiment_total', 8)} available; "
        f"environment matching {context.get('environment_status', 'DISABLED')}"
    )
    output_dir.joinpath("executive_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    methodology = """# Methodology

v2 refuses proxy macro dates, uses Market Model event studies, labels every signal GREEN/YELLOW/RED/BLACK, uses event-specific pre-drift baskets for Bayesian conditioning, and aggregates equal/N-weighted/composite scenarios with Gaussian Copula draws.

Small-sample events receive explicit penalties in `analysis/weighting.py`. Environment matching excludes recent equity returns from the main path; the momentum-inclusive path is diagnostic only.
"""
    output_dir.joinpath("methodology.md").write_text(methodology, encoding="utf-8")

    data_quality = ["# Data Quality Report", ""]
    data_quality.append("## Event Quality")
    event_cols = [col for col in ["event_id", "clean_samples", "env_samples", "env_match_status", "data_quality", "data_quality_note"] if col in events_detail]
    data_quality.append("```text\n" + events_detail[event_cols].to_string(index=False) + "\n```")
    data_quality.append("\n## Sentiment Source Summary")
    data_quality.extend(
        [
            f"- Available / total: {context.get('sentiment_available', 0)} / {context.get('sentiment_total', 8)}",
            f"- Auto-fetched: {context.get('sentiment_auto_loaded', 0)}",
            f"- Manual: {context.get('sentiment_manual_loaded', 0)}",
            f"- Missing: {', '.join(context.get('sentiment_missing', [])) if context.get('sentiment_missing') else 'none'}",
            f"- Environment matching: {context.get('environment_status', 'DISABLED')} ({len(context.get('environment_dimensions', []))} dimensions)",
        ]
    )
    data_quality.append("\n## Sentiment Quality")
    data_quality.append("```text\n" + sentiment_snapshot[["metric", "status", "source"]].to_string(index=False) + "\n```")
    if warnings:
        data_quality.append("\n## Warnings")
        data_quality.extend(f"- {w}" for w in warnings[:100])
    output_dir.joinpath("data_quality_report.md").write_text("\n".join(data_quality), encoding="utf-8")

    introspection = """# Model Self-Reflection

1. 最大不确定性来自宏观数据 surprise 与核心情绪/持仓数据的可得性。
2. 如果模型错了，最可能错在把历史事件类别当作当前政策环境的充分类比。
3. 历史样本与当前环境最大差异是估值、关税基线和地缘风险叠加。
4. 若再多 6 个月开发，会优先接入付费一致预期、OptionMetrics/ORATS、真实 ETF flows 与 CFTC 全历史解析。
5. 用户只看一个数字时，应看 composite `p50`，但必须同时看 BLACK/RED 数据质量标签。
"""
    output_dir.joinpath("model_self_reflection.md").write_text(introspection, encoding="utf-8")


def _confidence(events_detail: pd.DataFrame, posteriors: pd.DataFrame) -> tuple[str, str]:
    if posteriors.empty:
        return "低", "后验不可用"
    valid = posteriors["posterior_mean"].dropna()
    if valid.empty:
        return "低", "后验均值缺失"
    direction_consistency = max(float((valid > 0).mean()), float((valid < 0).mean()))
    total_clean = int(events_detail.get("clean_samples", pd.Series(dtype=int)).sum())
    small_share = float((events_detail.get("clean_samples", pd.Series(dtype=int)) < 10).mean()) if not events_detail.empty else 1.0
    if direction_consistency >= 0.75 and total_clean >= 100 and small_share < 0.35:
        return "中高", f"{direction_consistency:.0%} 事件方向一致，总 clean 样本 {total_clean}"
    if direction_consistency >= 0.60 and total_clean >= 50:
        return "中等", f"{direction_consistency:.0%} 事件方向一致，总 clean 样本 {total_clean}"
    return "低到中等", f"{direction_consistency:.0%} 事件方向一致，小样本或方向冲突较多"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
