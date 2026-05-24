import pandas as pd

from data.sentiment import build_sentiment


def test_exogenous_sentiment_history_excludes_price_momentum():
    idx = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame({"SPY": range(260), "^VIX": 20.0, "^VIX3M": 22.0}, index=idx)
    macro = pd.DataFrame({"HY_OAS": 3.0, "IG_OAS": 1.0, "UST2Y_YIELD": 4.0, "UST10Y_YIELD": 4.5}, index=idx)
    result = build_sentiment(prices, macro, idx[-1], manual_path=None, enable_auto_fetch=False)
    assert "SPY_20d_return" not in result.history.columns
    assert "SPY_5d_return" not in result.history.columns
    assert "SPY_20d_return" in result.momentum_history.columns
