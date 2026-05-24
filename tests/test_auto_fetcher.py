from io import BytesIO
import zipfile

import pandas as pd

from data.sentiment_fetcher import SentimentFetcher


def _xlsx_bytes(frame: pd.DataFrame) -> bytes:
    buf = BytesIO()
    frame.to_excel(buf, index=False)
    return buf.getvalue()


def test_fetch_aaii_from_mock_excel(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-04-01", periods=3, freq="W-THU"),
            "Bullish": [30.0, 35.5, 40.0],
            "Bearish": [25.0, 28.2, 22.0],
        }
    )
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", lambda *args, **kwargs: _xlsx_bytes(frame))

    data = fetcher._fetch_aaii()

    assert data["AAII_bull_bear_spread"] == 18.0
    assert data["AAII_date"] == "2026-04-16"


def test_fetch_naaim_from_mock_csv(monkeypatch, tmp_path):
    csv = b"Date,NAAIM Number Mean\n2026-05-01,70\n2026-05-08,82.5\n"
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", lambda *args, **kwargs: csv)

    data = fetcher._fetch_naaim()

    assert data["NAAIM_exposure"] == 82.5
    assert data["NAAIM_date"] == "2026-05-08"


def test_fetch_cftc_from_mock_zip(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "Market and Exchange Names": [
                "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
                "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
            ],
            "As of Date in Form YYYY-MM-DD": ["2026-05-05", "2026-05-05"],
            "Noncommercial Positions-Long (All)": [450000, 120000],
            "Noncommercial Positions-Short (All)": [270000, 195000],
        }
    )
    xlsx = _xlsx_bytes(frame)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("cftc.xlsx", xlsx)
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", lambda *args, **kwargs: buf.getvalue())

    data = fetcher._fetch_cftc_cot()

    assert data["CFTC_emini_spx_net"] == 180000
    assert data["CFTC_10y_ust_net"] == -75000
    assert data["CFTC_date"] == "2026-05-05"


def test_fetch_finra_margin_from_mock_excel(monkeypatch, tmp_path):
    dates = pd.date_range("2025-01-31", periods=16, freq="ME")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Debit Balances in Customers' Securities Margin Accounts": range(700, 716),
        }
    )
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", lambda *args, **kwargs: _xlsx_bytes(frame))

    data = fetcher._fetch_finra_margin()

    assert data["FINRA_date"] == "2026-04-30"
    assert data["FINRA_margin_debt_yoy"] > 0


def test_fetch_ici_flows_from_mock_html(monkeypatch, tmp_path):
    html = b"""
    <table>
      <tr><th>Date</th><th>Domestic Equity</th><th>World Equity</th></tr>
      <tr><td>2026-04-15</td><td>1.0</td><td>2.0</td></tr>
      <tr><td>2026-04-22</td><td>1.5</td><td>2.5</td></tr>
      <tr><td>2026-04-29</td><td>2.0</td><td>3.0</td></tr>
      <tr><td>2026-05-06</td><td>2.5</td><td>3.5</td></tr>
    </table>
    """
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", lambda *args, **kwargs: html)

    data = fetcher._fetch_ici_flows()

    assert data["ICI_4w_equity_flow_b"] == 18.0
    assert data["ICI_date"] == "2026-05-06"


def test_fetch_cboe_put_call_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_read_url(url, *args, **kwargs):
        calls["count"] += 1
        return b"DATE,CLOSE\n2026-04-27,0.60\n2026-04-28,0.70\n2026-04-29,0.80\n2026-04-30,0.90\n2026-05-01,1.00\n2026-05-04,1.10\n2026-05-05,1.20\n2026-05-06,1.30\n2026-05-07,1.40\n2026-05-08,1.50\n"

    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_read_url", fake_read_url)

    first = fetcher._fetch_cboe_pc()
    second = fetcher._fetch_cboe_pc()

    assert first == second
    assert calls["count"] == 2
    assert first["CBOE_equity_pc_10dma"] == 1.05


def test_fetch_all_failure_isolation(monkeypatch, tmp_path):
    fetcher = SentimentFetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_fetch_aaii", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(fetcher, "_fetch_naaim", lambda: {"NAAIM_exposure": 80.0})
    monkeypatch.setattr(fetcher, "_fetch_cftc_cot", lambda: {})
    monkeypatch.setattr(fetcher, "_fetch_finra_margin", lambda: {})
    monkeypatch.setattr(fetcher, "_fetch_ici_flows", lambda: {})
    monkeypatch.setattr(fetcher, "_fetch_cboe_pc", lambda: {})

    data = fetcher.fetch_all()

    assert data == {"NAAIM_exposure": 80.0}
    assert any("aaii failed" in warning for warning in fetcher.warnings)
