"""Tests for Black-Scholes implied volatility inversion."""

from __future__ import annotations

from methods.option_implied import black_scholes_call_price, implied_volatility


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
