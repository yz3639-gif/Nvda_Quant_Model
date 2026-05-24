import numpy as np
import pandas as pd

from analysis.pre_drift import current_event_pre_drift
from data.calendar import CalendarEvent


def test_pre_drift_is_event_specific_basket():
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    prices = pd.DataFrame(
        {
            "SPY": np.linspace(100, 110, len(idx)),
            "KWEB": np.linspace(50, 70, len(idx)),
            "FXI": np.linspace(40, 55, len(idx)),
        },
        index=idx,
    )
    event = CalendarEvent(id="e", name="china", date=pd.Timestamp("2026-03-01"), category="PRESIDENT_CHINA_VISIT")
    config = {"event_pre_drift_baskets": {"PRESIDENT_CHINA_VISIT": ["KWEB", "FXI"], "DEFAULT": ["SPY"]}}
    drift = current_event_pre_drift(event, prices, config, idx[-1])
    assert drift["basket"] == "KWEB,FXI"
    assert drift["current_pre_drift"] != prices["SPY"].iloc[-1] / prices["SPY"].iloc[-21] - 1

