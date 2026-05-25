#!/usr/bin/env python3
"""Command-line SPY probability distribution estimator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from methods.historical import run_historical_bootstrap
from methods.monte_carlo import estimate_gbm_parameters, run_monte_carlo_methods
from methods.option_implied import run_option_implied_method, simulate_lognormal_option_paths
from utils.data import (
    OptionChainData,
    compute_log_returns,
    ensure_directory,
    next_business_day,
    read_option_chain,
    read_price_history,
    read_risk_free_rate,
)
from utils.plotting import plot_all
from utils.stats import (
    ScenarioResult,
    build_ensemble_metrics,
    build_result,
    comparison_frame,
    format_comparison_for_terminal,
    format_pct,
    json_safe,
    metrics_to_records,
    parse_thresholds,
    result_summary_for_json,
)


logger = logging.getLogger(__name__)

DISCLAIMER = """
IMPORTANT DISCLAIMER
This tool produces statistical probability distributions under explicit modeling assumptions.
It is not investment advice, a recommendation, or a statement about what the future will do.
Historical bootstrap, GBM, and option-implied methods can all fail when market regimes change,
liquidity shifts, volatility risk premia move, or option chains are stale/sparse.
"""


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a strictly positive float CLI value."""

    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Estimate SPY forward return probability distributions.")
    parser.add_argument("--ticker", default="SPY", help="Ticker to analyze. Default: SPY")
    parser.add_argument("--horizon-days", type=_positive_int, default=63, help="Forecast horizon in trading days.")
    parser.add_argument("--start-date", type=str, default=None, help="Forecast start date. Default: next business day.")
    parser.add_argument("--lookback-years", type=_positive_float, default=20.0, help="Historical lookback window in years.")
    parser.add_argument("--n-sims", type=_positive_int, default=10_000, help="Number of simulation paths.")
    parser.add_argument(
        "--block-size",
        type=_positive_int,
        default=5,
        help=(
            "Historical bootstrap block size in trading days. Smaller values assume more independence; "
            "larger values preserve more short-run clustering."
        ),
    )
    parser.add_argument("--risk-free-rate", type=float, default=None, help="Annual risk-free rate override as decimal.")
    parser.add_argument(
        "--thresholds",
        default="-10,-5,-2,0,2,5,10",
        help="Comma-separated return thresholds in percent.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory.")
    parser.add_argument("--options-source", default="yfinance", help="Options data source. Default: yfinance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--validate", action="store_true", help="Run a one-year-ago historical validation.")
    parser.add_argument("--verbose", action="store_true", help="Print traceback details when a forecast fails.")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--plot", dest="plot", action="store_true", default=True, help="Generate PNG charts.")
    plot_group.add_argument("--no-plot", dest="plot", action="store_false", help="Skip PNG charts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(DISCLAIMER.strip())
    try:
        run_forecast(args)
    except KeyboardInterrupt:
        logger.warning("Forecast interrupted by user.")
        return 130
    except Exception as exc:
        if args.verbose:
            logger.exception("Forecast failed")
        else:
            logger.error("Forecast failed: %s", exc)
        print(f"\nERROR: {exc}\nRun with --verbose for traceback.")
        return 1
    return 0


def run_forecast(args: argparse.Namespace) -> None:
    """Run data loading, simulations, reporting, and output persistence."""

    thresholds = parse_thresholds(args.thresholds)
    output_dir = ensure_directory(args.output_dir)
    cache_dir = ensure_directory(Path("./cache"))
    start_date = _resolve_start_date(args.start_date)
    seed_sequence = np.random.SeedSequence(args.seed)
    rng_names = ("historical", "mc", "option", "plot", "validation")
    rngs = {
        name: np.random.default_rng(child_seed)
        for name, child_seed in zip(rng_names, seed_sequence.spawn(len(rng_names)), strict=True)
    }

    logger.info("Loading %s price history for %.2f lookback years.", args.ticker, args.lookback_years)
    prices = read_price_history(args.ticker, args.lookback_years, cache_dir)
    returns = compute_log_returns(prices)
    spot = float(prices.iloc[-1])
    as_of_date = pd.Timestamp(prices.index[-1]).normalize()
    risk_free = read_risk_free_rate(cache_dir, args.risk_free_rate)

    print(f"\nAs of: {as_of_date.date().isoformat()} | Forecast start: {start_date.date().isoformat()}")
    print(f"Ticker: {args.ticker} | Spot: {spot:.2f} | Horizon: {args.horizon_days} trading days")
    print(f"Risk-free rate: {risk_free.rate:.4%} ({risk_free.source})")
    for warning in risk_free.warnings:
        print(f"WARNING: {warning}")

    historical = run_historical_bootstrap(
        returns=returns.to_numpy(dtype=float),
        spot=spot,
        horizon_days=args.horizon_days,
        n_sims=args.n_sims,
        block_size=args.block_size,
        rng=rngs["historical"],
        thresholds=thresholds,
        lookback_years=args.lookback_years,
    )
    gbm_normal, gbm_student_t = run_monte_carlo_methods(
        returns=returns.to_numpy(dtype=float),
        spot=spot,
        horizon_days=args.horizon_days,
        n_sims=args.n_sims,
        rng=rngs["mc"],
        thresholds=thresholds,
        lookback_years=args.lookback_years,
    )
    option_result = _run_option_method_with_fallback(
        args=args,
        cache_dir=cache_dir,
        spot=spot,
        as_of_date=as_of_date,
        returns=returns.to_numpy(dtype=float),
        risk_free_rate=risk_free.rate,
        rng=rngs["option"],
        thresholds=thresholds,
    )

    primary_results = [historical, gbm_normal, option_result]
    all_results = primary_results + [gbm_student_t]
    ensemble_metrics = build_ensemble_metrics(primary_results)
    comparison = comparison_frame(primary_results)
    divergence_notes = _explain_divergences(primary_results, comparison)

    _print_results(all_results, ("Ensemble Average", ensemble_metrics), comparison, divergence_notes)
    saved_files = _save_outputs(
        args=args,
        output_dir=output_dir,
        spot=spot,
        as_of_date=as_of_date,
        start_date=start_date,
        risk_free_source=risk_free.source,
        results=all_results,
        ensemble_metrics=ensemble_metrics,
        comparison=comparison,
        divergence_notes=divergence_notes,
    )

    if args.plot:
        chart_paths = plot_all(results=primary_results, output_dir=output_dir, rng=rngs["plot"])
        saved_files.extend(chart_paths)

    if args.validate:
        validation = run_validation(prices, args, thresholds, rngs["validation"])
        validation_paths = _save_validation(validation, output_dir)
        saved_files.extend(validation_paths)
        _print_validation(validation)

    print("\nSaved files:")
    for path in saved_files:
        print(f"  {path}")


def run_validation(
    prices: pd.Series,
    args: argparse.Namespace,
    thresholds: list[float],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Backtest forecast distributions from approximately one year ago."""

    clean_prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if clean_prices.shape[0] <= args.horizon_days + 30:
        return [{"warning": "Not enough price history to run validation."}]
    target_date = clean_prices.index[-1] - pd.Timedelta(days=365)
    as_of_position = int(np.searchsorted(clean_prices.index.to_numpy(), np.datetime64(target_date), side="right") - 1)
    as_of_position = min(as_of_position, clean_prices.shape[0] - args.horizon_days - 1)
    as_of_position = max(as_of_position, 30)
    if as_of_position >= clean_prices.shape[0] - args.horizon_days:
        return [{"warning": "Not enough future price history to run validation at the requested horizon."}]
    validation_prices = clean_prices.iloc[: as_of_position + 1]
    future_terminal = float(clean_prices.iloc[as_of_position + args.horizon_days])
    validation_spot = float(clean_prices.iloc[as_of_position])
    validation_returns = compute_log_returns(validation_prices).to_numpy(dtype=float)
    seed_sequence = np.random.SeedSequence(rng.integers(0, 2**32 - 1))
    rng_historical, rng_mc = [np.random.default_rng(child) for child in seed_sequence.spawn(2)]
    historical = run_historical_bootstrap(
        returns=validation_returns,
        spot=validation_spot,
        horizon_days=args.horizon_days,
        n_sims=args.n_sims,
        block_size=args.block_size,
        rng=rng_historical,
        thresholds=thresholds,
        lookback_years=args.lookback_years,
    )
    gbm_normal, gbm_student_t = run_monte_carlo_methods(
        returns=validation_returns,
        spot=validation_spot,
        horizon_days=args.horizon_days,
        n_sims=args.n_sims,
        rng=rng_mc,
        thresholds=thresholds,
        lookback_years=args.lookback_years,
    )
    actual_return = (future_terminal / validation_spot) - 1.0
    rows = []
    for result in [historical, gbm_normal, gbm_student_t]:
        intervals = result.metrics["return_confidence_intervals"]
        rows.append(
            {
                "method": result.name,
                "as_of_date": pd.Timestamp(clean_prices.index[as_of_position]).date().isoformat(),
                "terminal_date": pd.Timestamp(clean_prices.index[as_of_position + args.horizon_days]).date().isoformat(),
                "actual_return": actual_return,
                "predicted_mean_return": result.metrics["expected_return"],
                "predicted_median_return": result.metrics["median_return"],
                "actual_percentile": float(np.mean(result.terminal_returns <= actual_return)),
                "inside_50_ci": _inside_interval(actual_return, intervals["50%"]),
                "inside_80_ci": _inside_interval(actual_return, intervals["80%"]),
                "inside_95_ci": _inside_interval(actual_return, intervals["95%"]),
            }
        )
    rows.append(
        {
            "warning": "Option-implied validation is skipped because free yfinance chains do not provide historical option chains.",
        }
    )
    return rows


def _run_option_method_with_fallback(
    args: argparse.Namespace,
    cache_dir: Path,
    spot: float,
    as_of_date: pd.Timestamp,
    returns: np.ndarray,
    risk_free_rate: float,
    rng: np.random.Generator,
    thresholds: list[float],
) -> ScenarioResult:
    """Run option-implied method, falling back clearly when chain data is unavailable."""

    try:
        chain = read_option_chain(
            ticker=args.ticker,
            horizon_days=args.horizon_days,
            cache_dir=cache_dir,
            source=args.options_source,
            as_of_date=as_of_date,
        )
        result = run_option_implied_method(
            calls=chain.calls,
            puts=chain.puts,
            spot=spot,
            expiration=chain.expiration,
            as_of_date=as_of_date,
            horizon_days=args.horizon_days,
            n_sims=args.n_sims,
            risk_free_rate=risk_free_rate,
            rng=rng,
            thresholds=thresholds,
            options_source=chain.source,
        )
        result.warnings.extend(chain.warnings)
        return result
    except Exception as exc:
        params = estimate_gbm_parameters(returns)
        paths = simulate_lognormal_option_paths(
            spot=spot,
            horizon_days=args.horizon_days,
            n_sims=args.n_sims,
            risk_free_rate=risk_free_rate,
            implied_volatility_value=params.sigma_annual,
            rng=rng,
        )
        return build_result(
            name="Option-Implied Distribution",
            paths=paths,
            spot=spot,
            thresholds=thresholds,
            assumptions=[
                "Option chain was unavailable, so this is a clearly marked emergency fallback.",
                "Fallback uses historical annualized volatility with risk-neutral drift.",
                "This fallback is not an option-implied probability distribution.",
            ],
            warnings=[f"Could not run option-implied method: {exc}"],
            extras={
                "method_detail": "historical_vol_unavailable_fallback",
                "risk_free_rate": risk_free_rate,
                "atm_iv": params.sigma_annual,
                "rnd_reconstructed": False,
            },
        )


def _save_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    spot: float,
    as_of_date: pd.Timestamp,
    start_date: pd.Timestamp,
    risk_free_source: str,
    results: list[ScenarioResult],
    ensemble_metrics: dict[str, Any],
    comparison: pd.DataFrame,
    divergence_notes: list[str],
) -> list[Path]:
    """Save JSON and CSV output files."""

    ensure_directory(output_dir)
    json_path = output_dir / "forecast_results.json"
    metrics_path = output_dir / "probability_metrics.csv"
    comparison_path = output_dir / "method_comparison.csv"
    records: list[dict[str, Any]] = []
    for result in results:
        records.extend(metrics_to_records(result))
    records.extend(metrics_to_records(("Ensemble Average", ensemble_metrics)))
    pd.DataFrame(records).to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    payload = {
        "disclaimer": DISCLAIMER.strip(),
        "run": {
            "ticker": args.ticker,
            "spot": spot,
            "as_of_date": as_of_date.date().isoformat(),
            "forecast_start_date": start_date.date().isoformat(),
            "horizon_days": args.horizon_days,
            "lookback_years": args.lookback_years,
            "n_sims": args.n_sims,
            "block_size": args.block_size,
            "thresholds_percent": args.thresholds,
            "seed": args.seed,
            "risk_free_source": risk_free_source,
            "options_source": args.options_source,
        },
        "results": [result_summary_for_json(result) for result in results],
        "ensemble_average": json_safe(ensemble_metrics),
        "comparison": json_safe(comparison.to_dict(orient="records")),
        "divergence_notes": divergence_notes,
    }
    try:
        json_payload = json.dumps(json_safe(payload), indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.exception("JSON serialization failed for forecast payload.")
        raise ValueError(f"Could not serialize forecast output: {exc}") from exc
    json_path.write_text(json_payload, encoding="utf-8")
    return [json_path, metrics_path, comparison_path]


def _save_validation(validation: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Save validation output files."""

    json_path = output_dir / "validation_results.json"
    csv_path = output_dir / "validation_results.csv"
    try:
        json_payload = json.dumps(json_safe(validation), indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.exception("JSON serialization failed for validation payload.")
        raise ValueError(f"Could not serialize validation output: {exc}") from exc
    json_path.write_text(json_payload, encoding="utf-8")
    pd.DataFrame(validation).to_csv(csv_path, index=False)
    return [json_path, csv_path]


def _print_results(
    results: list[ScenarioResult],
    ensemble: tuple[str, dict[str, Any]],
    comparison: pd.DataFrame,
    divergence_notes: list[str],
) -> None:
    """Print method summaries, ensemble metrics, and comparison table."""

    print("\nProbability tables:")
    for result in results:
        _print_result_summary(result.name, result.metrics, result.assumptions, result.warnings)
    _print_result_summary(ensemble[0], ensemble[1], ["Equal-weight average of the three primary method metrics."], [])
    print("\nThree-method comparison (* marks columns with >5 percentage point spread):")
    print(format_comparison_for_terminal(comparison))
    if divergence_notes:
        print("\nDivergence notes:")
        for note in divergence_notes:
            print(f"  - {note}")


def _print_result_summary(
    name: str,
    metrics: dict[str, Any],
    assumptions: list[str],
    warnings: list[str],
) -> None:
    """Print a single method's key statistics and probability buckets."""

    print(f"\n[{name}]")
    print(
        "P(up)={up} | P(down)={down} | mean={mean} | median={median} | std={std}".format(
            up=format_pct(metrics["p_up"]),
            down=format_pct(metrics["p_down"]),
            mean=format_pct(metrics["expected_return"]),
            median=format_pct(metrics["median_return"]),
            std=format_pct(metrics["return_std"]),
        )
    )
    print(
        "skew={skew:.3f} | excess kurtosis={kurt:.3f} | VaR95={var95} | VaR99={var99} | "
        "CVaR95={cvar95} | CVaR99={cvar99}".format(
            skew=metrics["skewness"],
            kurt=metrics["excess_kurtosis"],
            var95=format_pct(metrics["var_95"]),
            var99=format_pct(metrics["var_99"]),
            cvar95=format_pct(metrics["cvar_95"]),
            cvar99=format_pct(metrics["cvar_99"]),
        )
    )
    mdd = metrics["max_drawdown"]
    print(
        "max drawdown: mean={mean}, median={median}, 95% worst loss={p95}".format(
            mean=format_pct(mdd["mean"]),
            median=format_pct(mdd["median"]),
            p95=format_pct(mdd["p95_worst_loss"]),
        )
    )
    print("Return buckets:")
    for bucket, probability in metrics["bucket_probabilities"].items():
        print(f"  {bucket}: {format_pct(probability)}")
    print("Terminal return confidence intervals:")
    for level, interval in metrics["return_confidence_intervals"].items():
        print(f"  {level}: {format_pct(interval['low'])} to {format_pct(interval['high'])}")
    print("Terminal price confidence intervals:")
    for level, interval in metrics["price_confidence_intervals"].items():
        print(f"  {level}: {interval['low']:.2f} to {interval['high']:.2f}")
    print("Barrier touch probabilities:")
    for barrier, probability in metrics["barrier_probabilities"].items():
        print(f"  {barrier}: {format_pct(probability)}")
    if assumptions:
        print("Assumptions:")
        for assumption in assumptions:
            print(f"  - {assumption}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _print_validation(validation: list[dict[str, Any]]) -> None:
    """Print validation results to terminal."""

    print("\nValidation:")
    for row in validation:
        if "warning" in row and "method" not in row:
            print(f"  WARNING: {row['warning']}")
            continue
        print(
            "  {method}: actual={actual}, percentile={percentile:.2%}, inside 80% CI={inside80}, inside 95% CI={inside95}".format(
                method=row["method"],
                actual=format_pct(float(row["actual_return"])),
                percentile=float(row["actual_percentile"]),
                inside80=row["inside_80_ci"],
                inside95=row["inside_95_ci"],
            )
        )


def _explain_divergences(results: list[ScenarioResult], comparison: pd.DataFrame) -> list[str]:
    """Explain material differences among the primary methods."""

    marker_columns = [col for col in comparison.columns if col.endswith("_diverges_gt_5pp")]
    if not any(bool(comparison[col].iloc[0]) for col in marker_columns):
        return ["No primary comparison column differs by more than 5 percentage points."]
    notes = [
        "Methods differ structurally: bootstrap is empirical, GBM is parametric physical-measure, and option-implied is risk-neutral.",
    ]
    gbm = next((result for result in results if result.name.startswith("GBM Normal")), None)
    option = next((result for result in results if result.name.startswith("Option-Implied")), None)
    if gbm and option:
        hist_sigma = float(gbm.extras.get("sigma_annual", np.nan))
        atm_iv = float(option.extras.get("atm_iv", np.nan))
        if np.isfinite(hist_sigma) and np.isfinite(atm_iv):
            if atm_iv - hist_sigma > 0.05:
                notes.append(
                    "ATM implied volatility is more than 5 percentage points above historical volatility, "
                    "which can indicate pricing for event risk, hedging demand, or volatility risk premium."
                )
            elif hist_sigma - atm_iv > 0.05:
                notes.append(
                    "Historical volatility is more than 5 percentage points above ATM IV, "
                    "which can happen after a realized volatility shock not fully priced into options."
                )
    if option and option.extras.get("method_detail") != "breeden_litzenberger_rnd":
        notes.append("Option method used a fallback/lognormal approximation, so density shape may understate skew or smile effects.")
    return notes


def _resolve_start_date(raw_start_date: str | None) -> pd.Timestamp:
    """Resolve CLI start-date input into a normalized timestamp."""

    if raw_start_date:
        return pd.Timestamp(raw_start_date).normalize()
    return next_business_day()


def _inside_interval(value: float, interval: dict[str, float]) -> bool:
    """Return whether a value sits inside a low/high interval dictionary."""

    return bool(interval["low"] <= value <= interval["high"])


if __name__ == "__main__":
    raise SystemExit(main())
