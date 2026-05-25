from __future__ import annotations

import pandas as pd
import pytest

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.event_overlay.schema import assert_no_event_feature_leakage, normalize_event_frame


def test_event_schema_normalizes_nvda_event_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "id": "a1",
                "published_at": "2025-01-03T14:00:00Z",
                "source": "Reuters",
                "title": "Nvidia Blackwell demand remains strong",
                "content": "NVDA data center customers raise capex.",
                "ticker": "NVDA",
            }
        ]
    )

    events = classify_events(raw)

    assert events.loc[0, "event_id"] == "a1"
    assert events.loc[0, "ticker_scope"] == ("NVDA",)
    assert events.loc[0, "source_quality"] > 0.9
    assert events.loc[0, "nvda_relevance"] > 0.7
    assert events.loc[0, "event_type"] in {"product", "customer_capex"}
    assert "high_nvda_relevance" in events.loc[0, "reason_codes"]


def test_event_schema_rejects_leakage_fields() -> None:
    assert_no_event_feature_leakage(["pre_return_5d", "event_type_earnings"])
    with pytest.raises(ValueError):
        assert_no_event_feature_leakage(["future_1d_return"])
    with pytest.raises(ValueError):
        assert_no_event_feature_leakage(["overall_sma_20"])
    with pytest.raises(ValueError):
        assert_no_event_feature_leakage(["trend"])


def test_normalize_event_frame_keeps_unknown_event_neutral() -> None:
    events = normalize_event_frame(
        pd.DataFrame(
            [
                {
                    "published_at": "2025-01-03",
                    "source": "unknown",
                    "headline": "Quiet market update",
                    "ticker_scope": "NVDA",
                    "event_type": "not_a_real_type",
                }
            ]
        )
    )

    assert events.loc[0, "event_type"] == "neutral"
    assert events.loc[0, "source_quality"] == 0.5
