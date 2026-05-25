from __future__ import annotations

import pandas as pd
import pytest

from nvda_quant_model.event_overlay.news_history import (
    audit_event_store_for_backtest,
    build_historical_event_labels,
    build_historical_event_store,
    fetch_sec_filing_events,
    normalize_historical_event_frame,
    run_historical_event_overlay_backtest,
    to_event_overlay_frame,
    to_legacy_news_article_frame,
)


def _raw_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "published_at": "2025-01-06T14:30:00Z",
                "source": "Reuters",
                "url": "https://example.com/nvda-blackwell",
                "headline": "Nvidia Blackwell demand remains strong",
                "summary": "Customers raise data center capex.",
                "ticker": "NVDA",
            },
            {
                "published_at": "2025-01-06T14:30:00Z",
                "source": "Reuters",
                "url": "https://example.com/nvda-blackwell",
                "headline": "Nvidia Blackwell demand remains strong",
                "summary": "Duplicate row.",
                "ticker": "NVDA",
            },
        ]
    )


def _prices(n: int = 90) -> pd.DataFrame:
    idx = pd.bdate_range("2024-12-01", periods=n)
    close = pd.Series(100.0, index=idx)
    for i in range(1, n):
        close.iloc[i] = close.iloc[i - 1] * (1.01 if i % 3 == 0 else 0.995)
    return pd.DataFrame({"Close": close, "Volume": 10_000_000}, index=idx)


def test_historical_event_store_normalizes_utc_and_deduplicates() -> None:
    events = normalize_historical_event_frame(_raw_events(), ingested_at="2026-05-25T00:00:00Z")

    assert len(events) == 1
    assert events.loc[0, "published_at_utc"].tzinfo is not None
    assert events.loc[0, "event_id"].startswith("nvda_event_")
    assert events.loc[0, "canonical_hash"]
    assert events.loc[0, "event_type"] in {"product", "customer_capex", "neutral"}


def test_historical_event_store_rejects_future_and_label_fields() -> None:
    raw = _raw_events()
    raw["future_1d_return"] = 0.10
    with pytest.raises(ValueError, match="leakage"):
        normalize_historical_event_frame(raw)

    raw = _raw_events()
    raw["output_window"] = "[1, 2, 3]"
    with pytest.raises(ValueError, match="leakage"):
        normalize_historical_event_frame(raw)


def test_historical_event_labels_are_separate_from_feature_events() -> None:
    events = normalize_historical_event_frame(_raw_events())
    overlay_events = to_event_overlay_frame(events)
    labels = build_historical_event_labels(events, _prices())
    legacy = to_legacy_news_article_frame(events)

    assert "future_1d_direction" in labels.columns
    assert "future_1d_direction" not in overlay_events.columns
    assert {"published_at", "title", "link"}.issubset(legacy.columns)
    assert legacy.loc[0, "title"] == events.loc[0, "headline"]


def test_backtest_gate_reports_insufficient_sample_instead_of_fake_validation() -> None:
    events = normalize_historical_event_frame(_raw_events())
    labels = build_historical_event_labels(events, _prices())
    audit = audit_event_store_for_backtest(events, labels)
    result = run_historical_event_overlay_backtest(events, _prices(), min_events=250)

    assert audit["status"] == "insufficient_point_in_time_sample"
    assert audit["serious_backtest_ready"] is False
    assert result["status"] == "insufficient_point_in_time_sample"
    assert result["summary"].empty


def test_sec_adapter_maps_filings_without_future_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001045810-25-000001", "0001045810-25-000002"],
                "acceptanceDateTime": ["2025-02-26T21:05:00.000Z", "2025-03-01T12:00:00.000Z"],
                "filingDate": ["2025-02-26", "2025-03-01"],
                "form": ["10-Q", "4"],
                "primaryDocument": ["nvda-20250126.htm", "xslF345X05/doc4.xml"],
            }
        }
    }

    def fake_read_json_url(url: str, *, user_agent: str, timeout: int) -> dict:
        return payload

    monkeypatch.setattr("nvda_quant_model.event_overlay.news_history._read_json_url", fake_read_json_url)
    events = fetch_sec_filing_events(start="2025-01-01", end="2025-12-31")

    assert len(events) == 1
    assert events.loc[0, "source"] == "SEC EDGAR"
    assert events.loc[0, "event_type"] == "earnings"
    assert "future_1d_return" not in events.columns


def test_build_result_reports_source_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_gdelt(**kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("nvda_quant_model.event_overlay.news_history.fetch_gdelt_doc_events", broken_gdelt)
    result = build_historical_event_store(start="2025-01-01", end="2025-01-31", sources=("gdelt",))

    assert result.events.empty
    assert result.audit["status"] == "insufficient_point_in_time_sample"
    assert result.audit["source_warnings"] == ["gdelt_fetch_failed:rate limited"]
