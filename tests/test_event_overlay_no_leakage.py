from __future__ import annotations

import pandas as pd

from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame
from nvda_quant_model.event_overlay.labels import build_event_labels


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=80)
    close = pd.Series(100 + pd.RangeIndex(80), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 10_000_000 + pd.Series(range(80), index=idx) * 1000,
        }
    )


def test_event_features_do_not_include_label_or_future_fields() -> None:
    event = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "published_at": "2025-02-20T13:00:00Z",
                "source": "Reuters",
                "headline": "Nvidia raises guidance",
                "body": "Strong demand for Blackwell.",
                "ticker_scope": "NVDA",
                "output_window": [999, 1000],
                "future_1d_return": 9.9,
                "trend": "bullish",
                "alignment": "consistent",
            }
        ]
    )

    features = build_event_feature_frame(event, _prices())
    labels = build_event_labels(event, _prices())

    forbidden = {"output_window", "future_1d_return", "trend", "alignment", "future_1d_direction"}
    assert forbidden.isdisjoint(features.columns)
    assert "future_1d_direction" in labels.columns
    assert "event_type_guidance" in features.columns
