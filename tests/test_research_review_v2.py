"""Independent final review: calibrated exposure controls and resume integrity."""
import json
import numpy as np
import pandas as pd
import pytest
from nvda_quant_model.research.sample import synthetic_prices
from nvda_quant_model.research.experiments import FittedPredictor, fit_fold, research_frame


def test_exposure_control_reflects_selected_probability_mapping(monkeypatch):
    frame = research_frame(synthetic_prices(sessions=310, seed=23))
    frame["target_direction"] = (np.arange(len(frame)) % 10 != 0).astype(float)
    frame["target_return"] = np.where(frame.target_direction == 1, .01, -.01)
    # A raw forecaster can be poorly calibrated: its scores never cross .55,
    # while the forward calibration block supports a much higher occurrence rate.
    monkeypatch.setattr(FittedPredictor, "predict", lambda self, test: np.full(len(test), .52))
    prediction, audit = fit_fold(frame.iloc[:200], frame.iloc[200:210], "rule", 7, .55)
    assert audit["selected_calibrator"] != "raw"
    assert (prediction.calibrated >= .55).all()
    assert (prediction.training_vol_exposure > 0).all(), "Calibrated active strategy must not be compared to a raw-inactive cash control"


@pytest.fixture
def small_registered_run(tmp_path, monkeypatch):
    from nvda_quant_model.research import runner, reporting
    config = json.loads((runner.ROOT / "research/configs/smoke.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    prices_path = tmp_path / "prices.csv"
    prices = synthetic_prices(sessions=150, seed=19)
    prices.to_csv(prices_path)
    monkeypatch.setattr(runner, "code_state", lambda root: {"source_sha256": "test-source"})
    monkeypatch.setattr(runner, "dependencies", lambda: {"python": "3.test", "numpy": "test-version-1"})
    monkeypatch.setattr(runner, "run_predictions", lambda frame, config: (pd.DataFrame({"Date": [prices.index[80]], "raw": [.4]}), []))
    monkeypatch.setattr(runner, "calibration_report", lambda *a, **k: None)
    monkeypatch.setattr(runner, "trading_experiment", lambda *a, **k: None)
    monkeypatch.setattr(runner, "distribution_experiment", lambda *a, **k: (pd.DataFrame({"window": [1]}), pd.DataFrame({"summary": [1]})))
    monkeypatch.setattr(runner, "news_eligibility", lambda *a, **k: {})
    monkeypatch.setattr(reporting, "build_report", lambda *a, **k: None)
    output = tmp_path / "run"
    runner.run(config_path, prices_path, output, synthetic=True)
    return runner, config_path, prices_path, output


def test_resume_rejects_modified_completed_artifact(small_registered_run):
    runner, config, prices, output = small_registered_run
    path = output / "predictions.csv"
    tampered = pd.read_csv(path)
    tampered["raw"] = .99
    tampered.to_csv(path, index=False)
    with pytest.raises(ValueError, match="(?i)(artifact|hash|integrity|modified|mismatch)"):
        runner.run(config, prices, output, synthetic=True, resume=True)


def test_resume_rejects_changed_dependency_identity(small_registered_run, monkeypatch):
    runner, config, prices, output = small_registered_run
    monkeypatch.setattr(runner, "dependencies", lambda: {"python": "3.test", "numpy": "test-version-2"})
    with pytest.raises(ValueError, match="(?i)(dependenc|identity|immutable|mismatch)"):
        runner.run(config, prices, output, synthetic=True, resume=True)


def test_block_uncertainty_does_not_claim_precision_with_only_one_block():
    from nvda_quant_model.research.provenance import block_mean_interval
    interval = block_mean_interval(np.arange(10.), block=21, repetitions=100)
    assert interval["mean"] == 4.5
    assert interval["low"] is None and interval["high"] is None
    assert interval.get("inference") == "insufficient_effective_blocks"


def test_resume_hashes_identical_artifacts_left_by_interrupted_stage(small_registered_run):
    runner, config, prices, output = small_registered_run
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Valid interruption point: predictions/folds were written, but their stage
    # completion manifest was not persisted. A retry reproduces identical bytes.
    manifest["status"] = "failed"
    manifest["stages"] = {"predictions": {"status": "running"}}
    manifest.pop("artifacts", None)
    manifest_path.write_text(json.dumps(manifest))
    runner.run(config, prices, output, synthetic=True, resume=True)
    tampered = pd.read_csv(output / "predictions.csv")
    tampered["raw"] = .99
    tampered.to_csv(output / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="(?i)(artifact|hash|integrity|modified|mismatch)"):
        runner.run(config, prices, output, synthetic=True, resume=True)


def test_calibrated_exposure_uses_same_configured_cap(monkeypatch):
    frame = research_frame(synthetic_prices(sessions=310, seed=23))
    frame["target_direction"] = (np.arange(len(frame)) % 10 != 0).astype(float)
    frame["target_return"] = np.where(frame.target_direction == 1, .01, -.01)
    monkeypatch.setattr(FittedPredictor, "predict", lambda self, test: np.full(len(test), .52))
    prediction, _ = fit_fold(frame.iloc[:200], frame.iloc[200:210], "rule", 7, .55,
                             target_volatility=100, max_exposure=.3)
    assert np.allclose(prediction.training_vol_exposure, .3)
    assert np.allclose(prediction.training_conditional_size, .3)
