from pathlib import Path

import numpy as np
import pandas as pd

from data.sentiment import load_sentiment_snapshot


def _manual_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "manual_sentiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_manual_sentiment_full_file_loads_all_core_fields(tmp_path):
    manual = _manual_file(
        tmp_path,
        """
AAII_bull_bear_spread: 7.3
AAII_date: 2026-05-07
CFTC_emini_spx_net: 180000
CFTC_emini_spx_net_percentile: 78
CFTC_10y_ust_net: -75000
CFTC_10y_ust_net_percentile: 22
CFTC_date: 2026-05-05
CBOE_equity_pc_10dma: 0.65
CBOE_total_pc_10dma: 0.92
CBOE_date: 2026-05-08
NAAIM_exposure: 78
NAAIM_date: 2026-05-06
FINRA_margin_debt_yoy: 0.14
FINRA_date: 2026-03-31
ICI_4w_equity_flow_b: 12.5
ICI_date: 2026-05-06
""",
    )

    df = load_sentiment_snapshot("2026-05-10", manual_file=manual, enable_auto_fetch=False)
    available = df[df["status"] == "available"].set_index("metric")

    assert float(available.loc["AAII_bull_bear_spread", "value"]) == 7.3
    assert float(available.loc["CFTC_emini_spx_net_percentile", "value"]) == 78
    assert float(available.loc["CBOE_equity_pc_10dma", "value"]) == 0.65
    assert available.loc["NAAIM_exposure", "source"] == "manual_sentiment.yaml"
    assert df.attrs["manual_loaded"] == 8


def test_manual_sentiment_partial_file_keeps_missing_fields(tmp_path):
    manual = _manual_file(
        tmp_path,
        """
AAII_bull_bear_spread: 4.0
AAII_date: 2026-05-07
NAAIM_exposure: null
""",
    )

    df = load_sentiment_snapshot("2026-05-10", manual_file=manual, enable_auto_fetch=False)
    status = df.set_index("metric")["status"].to_dict()

    assert status["AAII_bull_bear_spread"] == "available"
    assert status["NAAIM_exposure"] == "missing"
    assert "CFTC_emini_spx_net_percentile" in df.attrs["missing_core"]


def test_auto_fetch_failure_falls_back_to_manual(monkeypatch, tmp_path):
    manual = _manual_file(tmp_path, "AAII_bull_bear_spread: 5.0\nAAII_date: 2026-05-07\n")

    class BrokenFetcher:
        def __init__(self, cache_dir):
            self.sources = {}
            self.warnings = []
            self.histories = {}

        def fetch_all(self):
            raise RuntimeError("network down")

    monkeypatch.setattr("data.sentiment.SentimentFetcher", BrokenFetcher)
    df = load_sentiment_snapshot("2026-05-10", manual_file=manual, enable_auto_fetch=True, cache_dir=tmp_path / "cache")

    row = df[df["metric"] == "AAII_bull_bear_spread"].iloc[0]
    assert float(row["value"]) == 5.0
    assert row["status"] == "available"
    assert any("auto-fetch failed" in warning for warning in df.attrs["warnings"])


def test_manual_values_override_auto_fetch(monkeypatch, tmp_path):
    manual = _manual_file(tmp_path, "AAII_bull_bear_spread: 9.0\nAAII_date: 2026-05-07\n")

    class FakeFetcher:
        def __init__(self, cache_dir):
            self.sources = {"AAII_bull_bear_spread": "auto:test"}
            self.warnings = []
            self.histories = {}

        def fetch_all(self):
            return {"AAII_bull_bear_spread": 1.0, "AAII_date": "2026-05-01"}

    monkeypatch.setattr("data.sentiment.SentimentFetcher", FakeFetcher)
    df = load_sentiment_snapshot("2026-05-10", manual_file=manual, enable_auto_fetch=True, cache_dir=tmp_path / "cache")

    row = df[df["metric"] == "AAII_bull_bear_spread"].iloc[0]
    assert np.isclose(float(row["value"]), 9.0)
    assert row["source"] == "manual_sentiment.yaml"
