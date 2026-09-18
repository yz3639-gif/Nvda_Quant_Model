import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from nvda_quant_model.data.data_validation import TrainingPreprocessor, clean_model_frame, purge_immature_labels, validate_ohlcv
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.session_calendar import attach_daily_time_contract, session_dates, session_schedule


def prices(n=100):
    idx = session_dates("2025-01-02", "2025-12-31")[:n]
    close = pd.Series(100 + np.arange(n) + np.sin(np.arange(n)), index=idx)
    return pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 100000}, index=idx)


def test_preprocessor_future_mutation_invariant_and_batch_independent():
    frame = attach_daily_time_contract(prices(40))
    frame["x"] = np.arange(40, dtype=float)
    frame.loc[frame.index[[0, 2, 20]], "x"] = np.nan
    cutoff = frame.iloc[20].decision_at
    train = purge_immature_labels(frame.iloc[:20], cutoff)
    a = TrainingPreprocessor(["x"]).fit(train, fit_at=cutoff)
    mutated = frame.copy()
    mutated.loc[mutated.index[21:], "x"] = 1e12
    b = TrainingPreprocessor(["x"]).fit(purge_immature_labels(mutated.iloc[:20], cutoff), fit_at=cutoff)
    assert_series_equal(a.medians_, b.medians_)
    assert_series_equal(a.upper_, b.upper_)
    assert_frame_equal(a.transform(frame.iloc[20:21]), b.transform(mutated.iloc[20:21]))
    assert_frame_equal(a.transform(frame.iloc[20:21]), a.transform(frame.iloc[20:]).iloc[:1])
    assert np.isnan(clean_model_frame(frame, ["x"]).x.iloc[0])


def test_missing_and_future_label_maturity_never_train():
    frame = attach_daily_time_contract(prices(12))
    cutoff = frame.iloc[8].decision_at
    frame.loc[frame.index[1], "label_end_at"] = pd.NaT
    frame.loc[frame.index[2], "available_at"] = pd.NaT
    train = purge_immature_labels(frame, cutoff)
    assert frame.index[1] not in train.index and frame.index[2] not in train.index
    assert (train.label_end_at <= cutoff).all()
    assert frame.index[8] not in train.index
    with pytest.raises(ValueError, match="label_end_at"):
        purge_immature_labels(frame.drop(columns="label_end_at"), cutoff)


def test_ohlcv_defects_fail_closed_and_missing_sessions_reported():
    frame = prices(10)
    for col, value in [("Low", 9999), ("Close", np.nan), ("Volume", -1)]:
        bad = frame.copy()
        bad.loc[bad.index[2], col] = value
        with pytest.raises(ValueError):
            validate_ohlcv(bad)
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(pd.concat([frame, frame.iloc[:1]]))
    gap = frame.drop(frame.index[3])
    assert validate_ohlcv(gap).attrs["data_quality"]["missing_sessions"]
    with pytest.raises(ValueError, match="calendar"):
        validate_ohlcv(gap, strict_calendar=True)


def test_feature_frame_future_prices_do_not_change_historical_features():
    bars = prices(100)
    start, end = str(bars.index.min().date()), str(bars.index.max().date())
    first, columns = build_model_frame(bars, pd.DataFrame(index=bars.index), start, end, include_fundamentals=False)
    future = bars.copy()
    future.loc[future.index[80:], ["Open", "High", "Low", "Close"]] *= 3
    second, _ = build_model_frame(future, pd.DataFrame(index=bars.index), start, end, include_fundamentals=False)
    assert_frame_equal(first[columns].iloc[:80], second[columns].iloc[:80])
    assert first.attrs["availability_provenance"]["fundamentals"] == "not_used"
    assert first.label_end_at.iloc[-1] is pd.NaT


def test_exchange_calendar_holidays_early_closes_and_dst():
    idx = pd.to_datetime(["2025-03-07", "2025-03-10", "2025-07-03", "2025-11-28"])
    schedule = session_schedule(idx)
    assert list(schedule.market_close.dt.hour) == [21, 20, 17, 18]
    assert pd.Timestamp("2025-01-09") not in session_dates("2025-01-01", "2025-01-15")
    assert pd.Timestamp("2025-04-18") not in session_dates("2025-04-01", "2025-04-30")
    with pytest.raises(ValueError, match="Calendar"):
        session_schedule(pd.to_datetime(["2025-07-04"]))
