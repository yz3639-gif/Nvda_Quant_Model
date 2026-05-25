from __future__ import annotations

import pandas as pd

from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame


def test_event_features_use_only_prices_before_publication_date() -> None:
    idx = pd.bdate_range("2025-01-01", periods=40)
    close = pd.Series(100.0, index=idx)
    close.iloc[:25] = 100.0
    close.iloc[25:] = 150.0
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 10_000_000,
        },
        index=idx,
    )
    event = pd.DataFrame(
        [
            {
                "event_id": "before_jump",
                "published_at": idx[25].strftime("%Y-%m-%dT10:00:00Z"),
                "source": "CNBC",
                "headline": "Nvidia product update",
                "body": "Blackwell chip update before the close.",
                "ticker_scope": "NVDA",
            }
        ]
    )

    features = build_event_feature_frame(event, prices)

    assert features.loc[0, "pre_return_1d"] == 0.0
    assert features.loc[0, "pre_return_5d"] == 0.0
    assert features.loc[0, "ema_20_ratio"] == 0.0


def test_event_features_add_external_market_context_without_future_data() -> None:
    idx = pd.bdate_range("2025-01-01", periods=40)
    close = pd.Series(100 + pd.RangeIndex(40), index=idx, dtype=float)
    prices = pd.DataFrame({"Close": close, "Volume": 10_000_000}, index=idx)
    external = pd.DataFrame(
        {
            "SMH": close * 2,
            "QQQ": close * 3,
            "SPY": close * 4,
            "VIX": 20 - pd.Series(range(40), index=idx) * 0.1,
        },
        index=idx,
    )
    event = pd.DataFrame(
        [
            {
                "event_id": "ctx",
                "published_at": idx[30].strftime("%Y-%m-%dT08:00:00Z"),
                "source": "Reuters",
                "headline": "Nvidia data center demand",
                "body": "Customers raise capex.",
                "ticker_scope": "NVDA",
            }
        ]
    )

    features = build_event_feature_frame(event, prices, external)

    assert features.loc[0, "SMH_return_5d"] > 0
    assert features.loc[0, "QQQ_return_5d"] > 0
    assert features.loc[0, "VIX_level"] > 0
