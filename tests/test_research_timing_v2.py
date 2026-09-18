"""End-to-end temporal invariants of the actual registered prediction code."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from threadpoolctl import threadpool_limits
from nvda_quant_model.research.sample import synthetic_prices
from nvda_quant_model.research.experiments import FittedPredictor, FEATURES, _mature, fit_fold, research_frame


@pytest.mark.parametrize("family", ["rule", "logistic", "boosting"])
def test_actual_fold_predictions_state_and_inner_fits_invariant_to_future_prices(family, monkeypatch):
    prices = synthetic_prices(sessions=360, seed=913)
    original = research_frame(prices)
    boundary = original.index[205]
    changed = prices.copy()
    changed.loc[changed.index > boundary, ["Open", "High", "Low", "Close"]] *= 7
    changed.loc[changed.index > boundary, "Volume"] *= 17
    future = research_frame(changed)
    assert_frame_equal(original.loc[:boundary, FEATURES], future.loc[:boundary, FEATURES])
    evidence = []
    real_fit = FittedPredictor.fit

    def record_fit(self, train, fit_at):
        result = real_fit(self, train, fit_at)
        mature = _mature(train, fit_at)
        assert mature.label_end_at.notna().all()
        assert mature.available_at.notna().all()
        assert (mature.label_end_at <= fit_at).all()
        assert (mature.available_at <= fit_at).all()
        assert self.preprocessor.fit_at_ == fit_at
        evidence.append({"fit_at": fit_at, "label_end": mature.label_end_at.max(),
                         "rows": mature.index.copy(), "lower": self.preprocessor.lower_.copy(),
                         "upper": self.preprocessor.upper_.copy(), "median": self.preprocessor.medians_.copy()})
        return result

    monkeypatch.setattr(FittedPredictor, "fit", record_fit)
    with threadpool_limits(limits=1):
        a, audit_a = fit_fold(original.iloc[:200], original.iloc[200:220], family, 4, .55)
        split = len(evidence)
        b, audit_b = fit_fold(future.iloc[:200], future.iloc[200:220], family, 4, .55)
    cols = ["raw", "shrink", "sigmoid", "calibrated", "base_rate", "selected_parameter", "selected_calibrator", "fit_at"]
    assert_frame_equal(a.loc[:boundary, cols], b.loc[:boundary, cols])
    assert audit_a["max_training_label_end_at"] <= audit_a["fit_at"]
    assert audit_a["candidate_trials"] == audit_b["candidate_trials"]
    assert audit_a["calibration_trials"] == audit_b["calibration_trials"]
    left, right = evidence[:split], evidence[split:]
    assert len(left) == len(right)
    assert len({e["fit_at"] for e in left}) == 3  # tuning, calibration model, final refit
    for x, y in zip(left, right):
        assert x["rows"].equals(y["rows"])
        assert x["fit_at"] == y["fit_at"]
        for stat in ["lower", "upper", "median"]:
            assert_series_equal(x[stat], y[stat])


def test_research_label_clock_matches_next_open_execution_and_missing_maturity():
    prices = synthetic_prices(sessions=200, seed=8)
    data = research_frame(prices)
    row = data.iloc[30]
    pos = prices.index.get_loc(data.index[30])
    expected = prices.Open.iloc[pos+2]/prices.Open.iloc[pos+1]-1
    assert row.target_return == expected
    assert row.available_at == row.decision_at
    assert row.label_end_at > row.decision_at
    assert data.label_end_at.iloc[-2:].isna().all()
    malformed = data.copy()
    malformed.loc[malformed.index[:3], "label_end_at"] = pd.NaT
    mature = _mature(malformed, data.decision_at.iloc[90])
    assert not set(malformed.index[:3]).intersection(mature.index)
    assert (mature.label_end_at <= data.decision_at.iloc[90]).all()
