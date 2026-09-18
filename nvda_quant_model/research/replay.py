"""Offline execution replay of frozen historical signals, not model refitting.

The historical signals were produced during development and are not a fresh
holdout. Replay establishes accounting and execution sensitivity conditional on
those signals. In particular, next-open/entry results change trading assumptions
and must not be called a pure bug-fix attribution or new out-of-sample evidence.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine, FILL_COLUMNS, TRADE_COLUMNS
from nvda_quant_model.backtest.execution import ExecutionConfig
from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.research.provenance import code_state, dependencies, sha256, utc_now, write_json


STRICT_SIGNALS = Path("performance_retest/meta_decision_layer_override")
STRICT_SCORES = Path("performance_retest/strict_candidate_promotion_override/candidate_stress_backtests.csv")
STRICT_CACHES = {
    24: "NVDA_2022-11-09_2026-05-23.csv",
    36: "NVDA_2021-11-08_2026-05-23.csv",
    60: "NVDA_2019-11-09_2026-05-23.csv",
}
DEFAULT_CACHE = "NVDA_2023-01-11_2026-07-25.csv"
VARIANTS = {
    "legacy_close_daily_reset": ExecutionConfig(mode="legacy_close", stop_reference="daily_reset"),
    "corrected_close_daily_reset": ExecutionConfig(mode="close", stop_reference="daily_reset"),
    "corrected_next_open_entry": ExecutionConfig(mode="next_open", stop_reference="entry"),
}
CAVEATS = [
    "Frozen precomputed signals only; no feature, training, parameter-selection or signal-generation reconstruction.",
    "Historical development data and selected strategies; not an untouched holdout and not evidence of prospective alpha.",
    "Corrected close changes accounting and gap fills; next-open plus entry additionally changes execution and stop semantics.",
    "Cached adjusted OHLC is replayed as stored; original vendor vintage and corporate-action availability are not independently reconstructed.",
    "Near matches are explicitly not exact reproductions; small differences may reflect cache vintages or numeric processing, with attribution unproven.",
]


def _daily(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError(f"{path}: nonempty unique chronological dates required")
    return data


def _input_record(path: Path, baseline_dir: Path, freeze: dict) -> dict:
    digest = sha256(path)
    expected = None
    try:
        relative = path.relative_to(baseline_dir).as_posix()
    except ValueError:
        relative = None
    if relative is not None:
        matches = [item for item in freeze.get("historical_artifacts", [])
                   if item["path"].endswith("historical_outputs/" + relative)]
    else:
        matches = [item for item in freeze.get("data", []) if Path(item["path"]).name == path.name]
    if len(matches) == 1:
        expected = matches[0]["sha256"]
        if digest != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size,
            "frozen_sha256": expected, "freeze_verified": expected == digest if expected else False}


def _metric_comparison(actual: dict, expected: dict) -> dict:
    comparisons = {}
    for name, recorded in expected.items():
        if name not in actual or not isinstance(recorded, (int, float, np.number)):
            continue
        computed = float(actual[name])
        recorded = float(recorded)
        if np.isfinite(recorded) and np.isfinite(computed):
            comparisons[name] = {"recorded": recorded, "replayed": computed, "delta": computed - recorded,
                                 "exact_tolerance": bool(np.isclose(computed, recorded, rtol=1e-10, atol=1e-10)),
                                 "near_tolerance": bool(np.isclose(computed, recorded, rtol=1e-5, atol=1e-6))}
    if not comparisons:
        status = "unverifiable_no_reference_metrics"
    elif all(item["exact_tolerance"] for item in comparisons.values()):
        status = "matched_within_1e-10"
    elif all(item["near_tolerance"] for item in comparisons.values()):
        status = "near_match_not_exact"
    else:
        status = "mismatch"
    return {"status": status, "metrics": comparisons,
            "exact_tolerance": {"rtol": 1e-10, "atol": 1e-10},
            "near_tolerance": {"rtol": 1e-5, "atol": 1e-6}}


def _specifications(baseline_dir: Path, cache_dir: Path) -> list[dict]:
    stress_path = baseline_dir / STRICT_SCORES
    stress = pd.read_csv(stress_path)
    meta_path = baseline_dir / STRICT_SIGNALS / "meta_decision_metrics.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    meta_summary_path = baseline_dir / STRICT_SIGNALS / "meta_decision_summary.json"
    meta_summary = json.loads(meta_summary_path.read_text()) if meta_summary_path.exists() else {}
    specs = []
    for months in [24, 36, 60]:
        # The first row in each window is the current baseline in the frozen
        # file. Select the explicit role instead of relying on row order.
        role = stress.roles.fillna("").map(lambda x: "current_baseline" in x.split("+"))
        candidates = stress.loc[role & stress.lookback_months.eq(months)]
        if len(candidates) != 1:
            raise ValueError(f"Expected one current_baseline source row for {months}m; found {len(candidates)}")
        row = candidates.iloc[0]
        expected = {key: row[key] for key in ["annualized_return", "sharpe_ratio", "max_drawdown",
                                             "win_rate", "profit_factor", "num_trades"] if key in row}
        references = [stress_path]
        label_matches = (str(meta_summary["baseline_label"]) == str(row.label)
                         if "baseline_label" in meta_summary else None)
        if meta_summary_path.exists():
            references.append(meta_summary_path)
        if not meta.empty:
            full = meta.loc[meta.lookback_months.eq(months) & meta.segment.eq("full") & meta.layer.eq("baseline")]
            if len(full) != 1:
                raise ValueError(f"Expected one full baseline meta reference for {months}m")
            expected.update({key: full.iloc[0][key] for key in
                             ["initial_capital", "final_equity", "total_return"] if key in full})
            references.append(meta_path)
        price_path = cache_dir / STRICT_CACHES[months]
        fallback = not price_path.exists()
        if fallback:
            price_path = cache_dir / STRICT_CACHES[60]
        cfg = StrategyConfig(start_date=str(row.resolved_start), end_date=str(row.resolved_end),
                             lookback_months=months, stop_loss_pct=0.025, take_profit_pct=0.04)
        specs.append({"family": "strict_baseline", "lookback_months": months, "source_label": row.label,
                      "signal_path": baseline_dir / STRICT_SIGNALS / f"meta_decision_strict_signals_{months}m.csv",
                      "price_path": price_path, "cache_fallback": fallback, "config": cfg,
                      "signal_label_matches_reference": label_matches,
                      "expected": expected, "reference_paths": references,
                      "config_basis": "Historical strict baseline constructor defaults plus explicit 2.5% stop / 4% take-profit override."})
    summary_path = baseline_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    allowed = {field.name for field in fields(StrategyConfig)}
    unknown = sorted(set(summary["config"]) - allowed)
    if unknown:
        raise ValueError(f"Unknown historical StrategyConfig fields: {unknown}")
    cfg = StrategyConfig(**summary["config"])
    specs.append({"family": "default_model", "lookback_months": cfg.lookback_months,
                  "source_label": "historical_default_summary", "signal_path": baseline_dir / "signals.csv",
                  "price_path": cache_dir / DEFAULT_CACHE, "cache_fallback": False, "config": cfg,
                  "signal_label_matches_reference": None,
                  "expected": summary["metrics"], "reference_paths": [summary_path],
                  "config_basis": "All stored StrategyConfig fields from frozen summary.json; new additive fields use current defaults."})
    return specs


def _replay_historical(baseline_dir: Path, cache_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Replay strict 24/36/60m and the default model under three explicit modes.

    Inputs are read-only. Missing files/hash changes raise rather than fetching
    or synthesizing historical signals. Existing variant output directories are
    rejected to prevent silently mixing different runs. The returned table is
    also written to ``historical_replay.csv``; every variant has full artifacts.
    """
    baseline_dir, cache_dir, output_dir = map(Path, (baseline_dir, cache_dir, output_dir))
    output_resolved = output_dir.resolve()
    for source in [baseline_dir.resolve(), cache_dir.resolve()]:
        if output_resolved == source or source in output_resolved.parents:
            raise ValueError("Replay outputs must not be written inside a source input directory")
    freeze_path = baseline_dir.parent / "manifest.json"
    freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else {}
    specs = _specifications(baseline_dir, cache_dir)
    # Validate every input before writing any result, and retain actual slices.
    prepared = []
    for spec in specs:
        cfg = spec["config"]
        paths = [spec["signal_path"], spec["price_path"], *spec["reference_paths"]]
        inputs = [_input_record(path, baseline_dir, freeze) for path in paths]
        signals, full_prices = _daily(spec["signal_path"]), _daily(spec["price_path"])
        prices = full_prices.loc[cfg.start_date:cfg.end_date]
        if prices.empty:
            raise ValueError(f"No cached prices in replay interval: {spec['price_path']}")
        if not signals.index.isin(prices.index).all():
            raise ValueError("Frozen signals contain dates absent from scoped price data")
        # Read all available session prices, including the historical default's
        # final un-signaled session. Engine's explicit missing-signal rule is 0.
        missing_signal_days = prices.index.difference(signals.index)
        identifier = f"{spec['family']}_{spec['lookback_months']}m"
        for variant in VARIANTS:
            if (output_dir / identifier / variant).exists():
                raise FileExistsError(f"Replay variant already exists: {identifier}/{variant}")
        prepared.append((spec, inputs, signals, prices, missing_signal_days, identifier))
    runtime = dependencies()
    code = code_state(Path(__file__).resolve().parents[2])
    created_at = utc_now()
    rows = []
    for spec, inputs, signals, prices, missing_signal_days, identifier in prepared:
        cfg = spec["config"]
        legacy_result = BacktestEngine(cfg, VARIANTS["legacy_close_daily_reset"]).backtest(signals, prices)
        comparison = _metric_comparison(legacy_result.metrics, spec["expected"])
        baseline_positions_match = (bool(np.allclose(signals.position, signals.baseline_position_before_meta,
                                                     rtol=0, atol=1e-12))
                                    if "baseline_position_before_meta" in signals else None)
        comparison["metric_status"] = comparison["status"]
        if spec["signal_label_matches_reference"] is False:
            comparison["status"] = "source_label_mismatch"
        elif baseline_positions_match is False:
            comparison["status"] = "signal_positions_differ_from_baseline"
        for variant, execution in VARIANTS.items():
            result = (legacy_result if execution.mode == "legacy_close" else
                      BacktestEngine(cfg, execution).backtest(signals, prices))
            directory = output_dir / identifier / variant
            directory.mkdir(parents=True)
            result.equity_curve.to_csv(directory / "equity_curve.csv")
            (result.trades if len(result.trades.columns) else pd.DataFrame(columns=TRADE_COLUMNS)).to_csv(directory / "trades.csv", index=False)
            (result.fills if len(result.fills.columns) else pd.DataFrame(columns=FILL_COLUMNS)).to_csv(directory / "fills.csv", index=False)
            signals.to_csv(directory / "signals.csv")
            prices.to_csv(directory / "prices.csv")
            write_json(directory / "account.json", result.account)
            write_json(directory / "metrics.json", result.metrics)
            record = {"schema_version": 1, "created_at": created_at, "experiment_id": identifier,
                      "family": spec["family"], "variant": variant, "source_label": spec["source_label"],
                      "signal_label_matches_reference": spec["signal_label_matches_reference"],
                      "frozen_positions_equal_baseline_positions": baseline_positions_match,
                      "replay_scope": "frozen_signal_execution_only", "holdout_status": "historical_development_data",
                      "signal_model_refitted": False, "config": asdict(cfg), "execution": asdict(execution),
                      "effective_cost_per_side": cfg.commission + cfg.slippage,
                      "config_basis": spec["config_basis"], "inputs": inputs,
                      "source_freeze_manifest_sha256": sha256(freeze_path) if freeze_path.exists() else None,
                      "source_commit": freeze.get("source_commit"), "code": code, "dependencies": runtime,
                      "seed": cfg.random_state, "cache_fallback": spec["cache_fallback"],
                      "date_start": str(prices.index.min().date()), "date_end": str(prices.index.max().date()),
                      "price_rows": len(prices), "signal_rows": len(signals),
                      "missing_signal_days": [str(x.date()) for x in missing_signal_days],
                      "missing_signal_policy": "zero target; next_open executes that target at the following observed session",
                      "legacy_reference_comparison": comparison,
                      "ledger_available": execution.mode != "legacy_close",
                      "caveats": CAVEATS,
                      "artifacts": {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}}
            write_json(directory / "manifest.json", record)
            rows.append({"experiment_id": identifier, "family": spec["family"], "lookback_months": spec["lookback_months"],
                         "variant": variant, "source_label": spec["source_label"],
                         "execution_mode": execution.mode, "stop_reference": execution.stop_reference,
                         "replay_scope": "frozen_signal_execution_only", "legacy_reference_status": comparison["status"],
                         "date_start": prices.index.min().date(), "date_end": prices.index.max().date(),
                         "price_rows": len(prices), "signal_rows": len(signals),
                         "missing_signal_days": len(missing_signal_days), "ledger_available": execution.mode != "legacy_close",
                         "recorded_annualized_return": spec["expected"].get("annualized_return"),
                         "recorded_sharpe_ratio": spec["expected"].get("sharpe_ratio"),
                         "annualized_delta_vs_recorded": result.metrics["annualized_return"] - spec["expected"].get("annualized_return", np.nan),
                         "sharpe_delta_vs_recorded": result.metrics["sharpe_ratio"] - spec["expected"].get("sharpe_ratio", np.nan),
                         "manifest_path": str((directory / "manifest.json").resolve()), **result.metrics})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "historical_replay.csv", index=False)
    write_json(output_dir / "historical_replay_manifest.json", {
        "created_at": created_at, "replay_scope": "frozen_signal_execution_only", "caveats": CAVEATS,
        "runs": [{"experiment_id": row["experiment_id"], "variant": row["variant"],
                  "manifest_path": row["manifest_path"], "manifest_sha256": sha256(Path(row["manifest_path"]))} for row in rows],
        "summary_sha256": sha256(output_dir / "historical_replay.csv")})
    return table


def replay_historical(baseline_dir: Path, cache_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Publish a whole validated replay atomically; failed attempts never block retry.

    Failed staging directories are retained next to the run directory for audit.
    Successful outputs are immutable and cannot be overwritten.
    """
    output_dir = Path(output_dir)
    for source in [Path(baseline_dir).resolve(), Path(cache_dir).resolve()]:
        if output_dir.resolve() == source or source in output_dir.resolve().parents:
            raise ValueError('Replay outputs must not be written inside a source input directory')
    if output_dir.exists():
        raise FileExistsError(f'Replay output already exists: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.replay-attempt-', dir=output_dir.parent.parent))
    try:
        table = _replay_historical(baseline_dir, cache_dir, staging)
        old, new = str(staging.resolve()), str(output_dir.resolve())
        table['manifest_path'] = table.manifest_path.str.replace(old, new, regex=False)
        table.to_csv(staging/'historical_replay.csv', index=False)
        master = json.loads((staging/'historical_replay_manifest.json').read_text())
        for entry in master['runs']:
            entry['manifest_path'] = entry['manifest_path'].replace(old, new)
        master['summary_sha256'] = sha256(staging/'historical_replay.csv')
        write_json(staging/'historical_replay_manifest.json', master)
        if output_dir.exists():
            raise FileExistsError(f'Replay output appeared during computation: {output_dir}')
        staging.rename(output_dir)
        return table
    except Exception:
        staging.rename(staging.with_name(staging.name + '.failed'))
        raise
