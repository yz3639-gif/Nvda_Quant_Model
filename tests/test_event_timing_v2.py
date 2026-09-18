import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from nvda_quant_model.data.session_calendar import session_dates
from nvda_quant_model.event_overlay.schema import normalize_event_frame
from nvda_quant_model.event_overlay.labels import build_event_labels
from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame
from nvda_quant_model.event_overlay.validation import event_validation_splits
from nvda_quant_model.event_overlay.overlay_model import train_event_overlay_model, score_event_overlay


def bars():
    idx = session_dates("2025-02-03", "2025-08-29")
    close = pd.Series(np.arange(len(idx)) + 100., index=idx)
    return pd.DataFrame({"Close": close, "Volume": 1000}, index=idx)


def events(times):
    return pd.DataFrame([{"event_id": str(i), "headline": f"NVDA update {i}", "published_at": t, "source": "Reuters"} for i, t in enumerate(times)])


def test_session_anchors_pre_intraday_after_close_weekend_dst_and_early_close():
    event = events(["2025-03-07T14:00:00Z", "2025-03-07T18:00:00Z", "2025-03-07T21:01:00Z", "2025-03-09T18:00:00Z", "2025-03-10T20:01:00Z", "2025-07-03T17:01:00Z"])
    labels = build_event_labels(event, bars())
    assert list(labels.anchor_at.dt.strftime("%Y-%m-%d %H:%M")) == ["2025-03-07 21:00", "2025-03-07 21:00", "2025-03-10 20:00", "2025-03-10 20:00", "2025-03-11 20:00", "2025-07-07 20:00"]
    assert labels.label_end_at_1d.iloc[-1] == pd.Timestamp("2025-07-08T20:00:00Z")


def test_availability_delay_controls_anchor_and_timezone_preserved():
    event = events(["2025-03-07T16:01:00-05:00"])
    event["available_at"] = "2025-03-10T20:01:00Z"
    normalized = normalize_event_frame(event)
    assert normalized.published_at.iloc[0] == pd.Timestamp("2025-03-07T21:01:00Z")
    assert build_event_labels(event, bars()).anchor_at.iloc[0] == pd.Timestamp("2025-03-11T20:00:00Z")
    event["available_at"] = "2025-03-01T00:00:00Z"
    with pytest.raises(ValueError, match="precede"):
        normalize_event_frame(event)


def test_unmatured_tail_labels_are_missing_not_false_negatives():
    market = bars()
    event = events([str(market.index[-2].date()) + "T22:00:00Z"])
    labels = build_event_labels(event, market)
    assert pd.isna(labels.label_end_at.iloc[0])
    assert pd.isna(labels.volatility_shock.iloc[0])
    assert pd.isna(labels.downside_tail_risk.iloc[0])


def test_event_feature_future_mutation_invariance():
    market = bars()
    event = events(["2025-03-07T21:01:00Z"])
    first = build_event_feature_frame(event, market)
    market.loc[market.index > "2025-03-07", "Close"] *= 10
    assert_frame_equal(first, build_event_feature_frame(event, market))


def test_timestamp_and_cluster_groups_never_cross_fold_boundary():
    times = pd.date_range("2025-03-01", periods=30, tz="UTC")
    event = events([x.isoformat() for x in times])
    event.loc[10:12, "published_at"] = event.loc[10, "published_at"]
    event["cluster_id"] = [str(i) for i in range(len(event))]
    event.loc[18:21, "cluster_id"] = "same_story"
    folds = list(event_validation_splits(event, 11, 6, 6))
    assert folds
    for train, test in folds:
        assert set(train.available_at).isdisjoint(test.available_at)
        assert set(train.cluster_id).isdisjoint(test.cluster_id)


def test_event_training_purges_labels_past_fit_and_missing_availability():
    market = bars()
    event = events([str(x.date()) + "T22:00:00Z" for x in market.index[:30]])
    fit_at = pd.Timestamp(str(market.index[20].date()) + "T22:00:00Z")
    first, _ = train_event_overlay_model(event, market, text_method="none", fit_at=fit_at)
    future = market.copy()
    future.loc[future.index > market.index[20], "Close"] *= 10
    second, _ = train_event_overlay_model(event, future, text_method="none", fit_at=fit_at)
    np.testing.assert_allclose(first.feature_pipeline.scaler.scale_, second.feature_pipeline.scaler.scale_)
    probe = events([fit_at.isoformat()])
    assert_frame_equal(score_event_overlay(first, probe, market), score_event_overlay(second, probe, future))
    event["available_at"] = pd.NaT
    with pytest.raises(ValueError, match="Not enough"):
        train_event_overlay_model(event, market, text_method="none", fit_at=fit_at)


def test_intraday_label_excludes_already_realized_return():
    market = bars()
    event = events(["2025-03-07T18:00:00Z"])
    first = build_event_labels(event, market)
    market.loc[market.index < "2025-03-07", "Close"] *= 5
    second = build_event_labels(event, market)
    assert first.future_1d_return.iloc[0] == second.future_1d_return.iloc[0]
    assert first.anchor_at.iloc[0] >= pd.Timestamp(event.published_at.iloc[0])


def test_historical_store_preserves_availability_clusters_and_ingestion():
    from nvda_quant_model.event_overlay.news_history import normalize_historical_event_frame, to_event_overlay_frame
    raw = events(["2025-03-07T18:00:00Z"])
    raw["available_at"] = "2025-03-07T21:01:00Z"
    raw["availability_basis"] = "verified_ingestion_log"
    raw["cluster_id"] = "story-1"
    history = normalize_historical_event_frame(raw, ingested_at="2025-03-08T00:00:00Z")
    repeated = normalize_historical_event_frame(history, ingested_at="2026-01-01T00:00:00Z")
    assert_frame_equal(history, repeated)
    overlay = to_event_overlay_frame(repeated)
    assert overlay.available_at.iloc[0] == pd.Timestamp("2025-03-07T21:01:00Z")
    assert overlay.cluster_id.iloc[0] == "story-1"
    assert overlay.published_at.iloc[0].tzinfo is not None
    assert overlay.availability_basis.iloc[0] == "verified_ingestion_log"
    missing = to_event_overlay_frame(events(["2025-03-07T18:00:00Z"]))
    assert pd.isna(missing.available_at.iloc[0])
    assert missing.availability_basis.iloc[0] == "unknown"


def test_event_log_loss_uses_neutral_class_probability():
    from nvda_quant_model.event_overlay.validation import _direction_metrics
    metric = _direction_metrics(pd.Series(["neutral"]), pd.Series([.1]), pd.Series(["neutral"]), pd.Series([.1]))
    assert metric["log_loss"] == pytest.approx(-np.log(.8))
