import pandas as pd

from data.calendar import CalendarEvent
from data.events_db import samples_for_event


def test_macro_event_does_not_generate_proxy_samples():
    event = CalendarEvent(id="cpi", name="CPI", date=pd.Timestamp("2026-05-12"), category="CPI")
    config = {"historical_sample_start": "1970-01-01", "macro_release_rules": {"CPI": {"weekday": 1, "nth": 2}}}
    samples = samples_for_event(event, pd.DataFrame(), config, pd.Timestamp("2026-05-10"))
    assert samples.empty

