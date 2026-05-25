"""Tests for SPY data cache hygiene."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import utils.data as data_module
from utils.data import read_price_history


def test_price_history_uses_fresh_stable_cache(tmp_path) -> None:
    """Price cache should be keyed by ticker and reused within the freshness window."""

    dates = pd.bdate_range("2024-01-02", periods=10)
    cache_path = tmp_path / "prices_SPY.csv"
    pd.Series(range(100, 110), index=dates, name="adj_close").to_frame().to_csv(cache_path, index_label="date")

    prices = read_price_history("SPY", lookback_years=0.02, cache_dir=tmp_path, end_date=pd.Timestamp("2024-01-15"))

    assert prices.iloc[0] == 100
    assert prices.iloc[-1] == 109


def test_price_history_recovers_from_corrupt_cache(tmp_path, monkeypatch) -> None:
    """A corrupt ticker-level cache should be replaced by a fresh download."""

    cache_path = tmp_path / "prices_SPY.csv"
    cache_path.write_text("date,adj_close\nbad,row,too,many,fields\n", encoding="utf-8")
    dates = pd.bdate_range("2024-01-02", periods=8)
    downloaded = pd.DataFrame({"Adj Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]}, index=dates)

    fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: downloaded)
    monkeypatch.setattr(data_module, "_import_yfinance", lambda: fake_yfinance)

    prices = read_price_history("SPY", lookback_years=0.02, cache_dir=tmp_path, end_date=pd.Timestamp("2024-01-12"))

    assert prices.iloc[-1] == 107.0
    assert "too,many" not in cache_path.read_text(encoding="utf-8")
