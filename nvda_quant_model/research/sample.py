"""Deterministic, license-free offline test data. Never label this market evidence."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def synthetic_prices(sessions=850, seed=314159):
    from nvda_quant_model.data.session_calendar import session_dates
    dates = session_dates('2019-01-02', '2025-12-31')[:sessions]
    rng = np.random.default_rng(seed)
    overnight = rng.normal(.0002, .008, sessions)
    intraday = rng.normal(.0003, .014, sessions)
    close = 100*np.exp(np.cumsum(overnight+intraday))
    opening = close/np.exp(intraday)
    pad = rng.uniform(.001, .02, sessions)
    return pd.DataFrame({'Open': opening, 'High': np.maximum(opening, close)*(1+pad),
        'Low': np.minimum(opening, close)*(1-pad), 'Close': close,
        'Volume': rng.integers(1000000, 10000000, sessions)}, index=pd.DatetimeIndex(dates, name='Date'))


def write_sample(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_prices().to_csv(path, float_format='%.10f')
