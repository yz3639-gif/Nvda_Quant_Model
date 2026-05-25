"""Tests for Black-Scholes implied volatility inversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from methods.option_implied import black_scholes_call_price, implied_volatility
from methods.option_implied import reconstruct_risk_neutral_density


def test_black_scholes_implied_volatility_inversion_accuracy() -> None:
    """Implied volatility inversion should recover the source volatility."""

    spot = 475.0
    strike = 480.0
    time_to_expiry = 63 / 365.0
    rate = 0.045
    source_vol = 0.22
    price = black_scholes_call_price(spot, strike, time_to_expiry, rate, source_vol)
    recovered_vol = implied_volatility(price, spot, strike, time_to_expiry, rate, "call")
    assert abs(recovered_vol - source_vol) < 1e-6


def test_reconstructed_density_is_normalized_for_smooth_call_curve() -> None:
    """A clean Black-Scholes call curve should produce a normalized density."""

    spot = 100.0
    rate = 0.04
    time_to_expiry = 45 / 365.0
    vol = 0.24
    strikes = np.linspace(65.0, 145.0, 81)
    calls = pd.DataFrame(
        {
            "strike": strikes,
            "bid": [black_scholes_call_price(spot, strike, time_to_expiry, rate, vol) * 0.995 for strike in strikes],
            "ask": [black_scholes_call_price(spot, strike, time_to_expiry, rate, vol) * 1.005 for strike in strikes],
        }
    )

    rnd = reconstruct_risk_neutral_density(calls, risk_free_rate=rate, time_to_expiry=time_to_expiry, spot=spot)

    assert rnd.reconstructed
    assert abs(float(np.trapezoid(rnd.density, rnd.strikes)) - 1.0) < 1e-9
    assert np.all(rnd.density >= 0.0)
