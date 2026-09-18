"""Portable offline replay tests use synthetic frozen signals and caches."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.execution import ExecutionConfig
from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.research.provenance import sha256
from nvda_quant_model.research.replay import DEFAULT_CACHE, STRICT_CACHES, STRICT_SCORES, STRICT_SIGNALS, replay_historical


@pytest.fixture
def replay_inputs(tmp_path):
    baseline = tmp_path / "baseline" / "historical_outputs"
    cache = tmp_path / "cache"
    (baseline / STRICT_SIGNALS).mkdir(parents=True)
    (baseline / STRICT_SCORES).parent.mkdir(parents=True)
    cache.mkdir()
    dates = pd.bdate_range("2025-01-02", periods=5, name="Date")
    prices = pd.DataFrame({"Open": [100, 100, 103, 80, 82], "High": [100, 104, 104, 83, 82],
                           "Low": [100, 99, 102, 79, 82], "Close": [100, 103, 103, 82, 82]}, index=dates)
    signals = pd.DataFrame({"position": [0, 1, 1, 0, 0], "direction": [0, 1, 1, 0, 0],
                            "confidence": 0.6, "expected_return": 0.01, "prob_up": 0.6,
                            "prob_down": 0.4}, index=dates)
    source_rows = []
    for months, filename in STRICT_CACHES.items():
        prices.to_csv(cache / filename)
        signals.to_csv(baseline / STRICT_SIGNALS / f"meta_decision_strict_signals_{months}m.csv")
        cfg = StrategyConfig(start_date=str(dates[0].date()), end_date=str(dates[-1].date()),
                             lookback_months=months, stop_loss_pct=0.025, take_profit_pct=0.04)
        r = BacktestEngine(cfg, ExecutionConfig(mode="legacy_close")).backtest(signals, prices)
        # The challenger is first deliberately. Replay must use the explicit role.
        source_rows.extend([{"roles": "challenger", "label": "higher_return_selected_challenger",
                             "lookback_months": months, "resolved_start": cfg.start_date,
                             "resolved_end": cfg.end_date, **r.metrics, "annualized_return": 999},
                            {"roles": "current_baseline+other_role", "label": "actual_baseline",
                             "lookback_months": months, "resolved_start": cfg.start_date,
                             "resolved_end": cfg.end_date, **r.metrics}])
    pd.DataFrame(source_rows).to_csv(baseline / STRICT_SCORES, index=False)
    prices.to_csv(cache / DEFAULT_CACHE)
    # Source default has a last available price without a last-day signal.
    signals.iloc[:-1].to_csv(baseline / "signals.csv")
    cfg = StrategyConfig(start_date=str(dates[0].date()), end_date=str(dates[-1].date()))
    r = BacktestEngine(cfg, ExecutionConfig(mode="legacy_close")).backtest(signals.iloc[:-1], prices)
    (baseline / "summary.json").write_text(json.dumps({"config": asdict(cfg), "metrics": r.metrics}))
    return baseline, cache


def test_replay_selects_baseline_and_writes_reconcilable_artifacts(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    before = {p: sha256(p) for directory in [baseline, cache] for p in directory.rglob("*") if p.is_file()}
    result = replay_historical(baseline, cache, tmp_path / "out")
    assert len(result) == 12
    assert result.legacy_reference_status.eq("matched_within_1e-10").all()
    assert result.loc[result.family.eq("strict_baseline"), "source_label"].eq("actual_baseline").all()
    assert result.loc[result.family.eq("default_model"), "missing_signal_days"].eq(1).all()
    assert {p: sha256(p) for p in before} == before
    for row in result.itertuples():
        path = Path(row.manifest_path)
        manifest = json.loads(path.read_text())
        assert manifest["signal_model_refitted"] is False
        assert manifest["replay_scope"] == "frozen_signal_execution_only"
        assert manifest["holdout_status"] == "historical_development_data"
        assert manifest["execution"]["mode"] == row.execution_mode
        assert all(item["sha256"] == sha256(Path(item["path"])) for item in manifest["inputs"])
        assert all(value == sha256(path.parent / name) for name, value in manifest["artifacts"].items())
        equity = pd.read_csv(path.parent / "equity_curve.csv")
        fills = pd.read_csv(path.parent / "fills.csv")
        if row.ledger_available:
            assert np.allclose(equity.equity, equity.cash + equity.shares * equity.mark_price)
            assert np.allclose(equity.equity, 100_000 + equity.realized_pnl + equity.unrealized_pnl)
            assert fills.fee.sum() == pytest.approx(row.total_fees)
        else:
            assert fills.empty
    repeated = replay_historical(baseline, cache, tmp_path / "out_repeat")
    pd.testing.assert_frame_equal(result.drop(columns="manifest_path"), repeated.drop(columns="manifest_path"))
    with pytest.raises(FileExistsError, match="already exists"):
        replay_historical(baseline, cache, tmp_path / "out")


def test_replay_reports_metric_mismatch_instead_of_faking_reproduction(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    source = baseline / "summary.json"
    summary = json.loads(source.read_text())
    summary["metrics"]["annualized_return"] = 5.0
    source.write_text(json.dumps(summary))
    result = replay_historical(baseline, cache, tmp_path / "mismatch")
    assert result.loc[result.family.eq("default_model"), "legacy_reference_status"].eq("mismatch").all()
    assert result.loc[result.family.eq("strict_baseline"), "legacy_reference_status"].eq("matched_within_1e-10").all()


def test_replay_honors_frozen_hashes_and_does_not_write_on_input_failure(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    source = baseline / "signals.csv"
    frozen = {"historical_artifacts": [{"path": "research/baseline/historical_outputs/signals.csv", "sha256": sha256(source)}]}
    (baseline.parent / "manifest.json").write_text(json.dumps(frozen))
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen input hash mismatch"):
        replay_historical(baseline, cache, tmp_path / "bad_hash")
    assert not (tmp_path / "bad_hash").exists()


def test_replay_records_broad_cache_fallback_and_rejects_source_output(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    (cache / STRICT_CACHES[24]).unlink()
    result = replay_historical(baseline, cache, tmp_path / "fallback")
    path = result.loc[result.experiment_id.eq("strict_baseline_24m"), "manifest_path"].iloc[0]
    assert json.loads(Path(path).read_text())["cache_fallback"] is True
    with pytest.raises(ValueError, match="source input directory"):
        replay_historical(baseline, cache, baseline / "results")


def test_replay_requires_unique_explicit_baseline_role(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    source = baseline / STRICT_SCORES
    rows = pd.read_csv(source)
    rows.loc[rows.roles.eq("challenger"), "roles"] = "current_baseline"
    rows.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Expected one current_baseline"):
        replay_historical(baseline, cache, tmp_path / "duplicate")


def test_replay_does_not_claim_same_strategy_when_meta_source_label_differs(replay_inputs, tmp_path):
    baseline, cache = replay_inputs
    (baseline / STRICT_SIGNALS / 'meta_decision_summary.json').write_text(json.dumps({'baseline_label': 'different_rule'}))
    result = replay_historical(baseline, cache, tmp_path / 'labels')
    assert result.loc[result.family.eq('strict_baseline'), 'legacy_reference_status'].eq('source_label_mismatch').all()


def test_next_open_old_inventory_gap_stop_precedes_queued_reversal():
    dates = pd.bdate_range('2025-01-01', periods=3)
    prices = pd.DataFrame({'Open': [100, 100, 80], 'High': [100, 100, 85],
                           'Low': [100, 100, 75], 'Close': [100, 100, 82]}, index=dates)
    signals = pd.DataFrame({'position': [1, -1, 0]}, index=dates)
    result = BacktestEngine(StrategyConfig(stop_loss_pct=.025), ExecutionConfig(cost_per_side=0)).backtest(signals, prices)
    assert result.fills.price.tolist() == [100, 80]
    assert result.fills.reason.tolist() == ['signal', 'stop_loss']
    assert result.equity_curve.shares.iloc[-1] == 0
    assert result.trades.pnl.sum() == pytest.approx(-20_000)
    assert result.metrics['final_equity'] == pytest.approx(80_000)
