"""Option-implied risk-neutral terminal distribution estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from methods.monte_carlo import TRADING_DAYS_PER_YEAR
from utils.stats import ScenarioResult, build_result, make_price_paths


@dataclass(slots=True)
class RiskNeutralDensity:
    """Risk-neutral density reconstructed across strike prices."""

    strikes: np.ndarray
    density: np.ndarray
    reconstructed: bool
    warning: str | None = None


def black_scholes_call_price(spot: float, strike: float, time_to_expiry: float, rate: float, vol: float) -> float:
    """Return the Black-Scholes European call price."""

    if time_to_expiry <= 0.0:
        return float(max(spot - strike, 0.0))
    if vol <= 0.0:
        forward_intrinsic = spot - strike * np.exp(-rate * time_to_expiry)
        return float(max(forward_intrinsic, 0.0))
    d1, d2 = _black_scholes_d1_d2(spot, strike, time_to_expiry, rate, vol)
    return float(spot * stats.norm.cdf(d1) - strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(d2))


def black_scholes_put_price(spot: float, strike: float, time_to_expiry: float, rate: float, vol: float) -> float:
    """Return the Black-Scholes European put price."""

    if time_to_expiry <= 0.0:
        return float(max(strike - spot, 0.0))
    if vol <= 0.0:
        forward_intrinsic = strike * np.exp(-rate * time_to_expiry) - spot
        return float(max(forward_intrinsic, 0.0))
    d1, d2 = _black_scholes_d1_d2(spot, strike, time_to_expiry, rate, vol)
    return float(strike * np.exp(-rate * time_to_expiry) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1))


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str = "call",
) -> float:
    """Invert Black-Scholes price to implied volatility using Brent's method."""

    if price <= 0.0:
        raise ValueError("Option price must be positive.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    pricing_function = black_scholes_call_price if option_type == "call" else black_scholes_put_price

    def objective(vol: float) -> float:
        """Return pricing error at the candidate volatility."""

        return pricing_function(spot, strike, time_to_expiry, rate, vol) - price

    lower = 1e-6
    upper = 5.0
    lower_value = objective(lower)
    upper_value = objective(upper)
    if lower_value * upper_value > 0:
        raise ValueError("Could not bracket implied volatility.")
    return float(optimize.brentq(objective, lower, upper, xtol=1e-10, rtol=1e-10, maxiter=100))


def extract_atm_implied_volatility(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
) -> tuple[float, list[str]]:
    """Extract ATM implied volatility from yfinance option chain columns."""

    warnings: list[str] = []
    candidates: list[float] = []
    for chain, option_type in ((calls, "call"), (puts, "put")):
        if chain.empty or "strike" not in chain:
            continue
        nearest = chain.iloc[(chain["strike"].astype(float) - spot).abs().argsort()[:3]]
        if "impliedVolatility" in nearest:
            iv_values = pd.to_numeric(nearest["impliedVolatility"], errors="coerce")
            candidates.extend(iv for iv in iv_values.to_numpy(dtype=float) if np.isfinite(iv) and 0.01 <= iv <= 5.0)
        if not candidates:
            for _, row in nearest.iterrows():
                mid = _option_mid_price(row)
                if mid is None:
                    continue
                try:
                    candidates.append(
                        implied_volatility(
                            price=mid,
                            spot=spot,
                            strike=float(row["strike"]),
                            time_to_expiry=time_to_expiry,
                            rate=risk_free_rate,
                            option_type=option_type,
                        )
                    )
                except ValueError:
                    continue
    if candidates:
        return float(np.median(candidates)), warnings
    warnings.append("Could not extract ATM IV from option chain; using 20% placeholder volatility.")
    return 0.20, warnings


def reconstruct_risk_neutral_density(
    calls: pd.DataFrame,
    risk_free_rate: float,
    time_to_expiry: float,
    spot: float,
) -> RiskNeutralDensity:
    """Reconstruct risk-neutral density from call prices using Breeden-Litzenberger."""

    prepared = _prepare_call_curve(calls, spot)
    if prepared.shape[0] < 15:
        return RiskNeutralDensity(
            strikes=np.array([], dtype=float),
            density=np.array([], dtype=float),
            reconstructed=False,
            warning="Option chain has too few usable call strikes for stable RND reconstruction.",
        )
    strikes = prepared["strike"].to_numpy(dtype=float)
    prices = prepared["price"].to_numpy(dtype=float)
    first_derivative = np.gradient(prices, strikes)
    second_derivative = np.gradient(first_derivative, strikes)
    density = np.maximum(np.exp(risk_free_rate * time_to_expiry) * second_derivative, 0.0)
    positive_points = int(np.sum(density > 0.0))
    integral = float(np.trapezoid(density, strikes))
    if positive_points < 5 or not np.isfinite(integral) or integral <= 1e-8:
        return RiskNeutralDensity(
            strikes=np.array([], dtype=float),
            density=np.array([], dtype=float),
            reconstructed=False,
            warning="Breeden-Litzenberger finite differences were unstable; falling back to ATM IV lognormal.",
        )
    density = density / integral
    mean_terminal = float(np.trapezoid(strikes * density, strikes))
    expected_forward = float(spot * np.exp(risk_free_rate * time_to_expiry))
    if not np.isfinite(mean_terminal) or abs(mean_terminal / expected_forward - 1.0) > 0.08:
        return RiskNeutralDensity(
            strikes=np.array([], dtype=float),
            density=np.array([], dtype=float),
            reconstructed=False,
            warning=(
                "Breeden-Litzenberger density failed forward-price consistency checks; "
                "falling back to ATM IV lognormal."
            ),
        )
    return RiskNeutralDensity(strikes=strikes, density=density, reconstructed=True)


def simulate_lognormal_option_paths(
    spot: float,
    horizon_days: int,
    n_sims: int,
    risk_free_rate: float,
    implied_volatility_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate risk-neutral lognormal paths parameterized by ATM IV."""

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    shocks = rng.standard_normal(size=(n_sims, horizon_days))
    increments = (
        (risk_free_rate - 0.5 * implied_volatility_value**2) * dt
        + implied_volatility_value * np.sqrt(dt) * shocks
    )
    return make_price_paths(spot, increments)


def simulate_rnd_terminal_paths(
    spot: float,
    horizon_days: int,
    n_sims: int,
    risk_free_rate: float,
    implied_volatility_value: float,
    rnd: RiskNeutralDensity,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate paths whose terminals are sampled from reconstructed RND."""

    base_paths = simulate_lognormal_option_paths(
        spot=spot,
        horizon_days=horizon_days,
        n_sims=n_sims,
        risk_free_rate=risk_free_rate,
        implied_volatility_value=implied_volatility_value,
        rng=rng,
    )
    target_terminal = sample_from_density(rnd.strikes, rnd.density, n_sims, rng)
    base_terminal = np.maximum(base_paths[:, -1], 1e-12)
    weights = np.linspace(0.0, 1.0, horizon_days + 1)
    adjustment = np.exp(np.log(target_terminal / base_terminal)[:, None] * weights[None, :])
    paths = base_paths * adjustment
    paths[:, 0] = spot
    paths[:, -1] = target_terminal
    return paths


def sample_from_density(
    grid: np.ndarray,
    density: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample values from a density defined on a grid by inverse transform."""

    if grid.size != density.size or grid.size < 2:
        raise ValueError("Density sampling requires matching grid and density arrays.")
    segment_area = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(segment_area)])
    if cdf[-1] <= 0.0 or not np.isfinite(cdf[-1]):
        raise ValueError("Density integral must be positive.")
    cdf = cdf / cdf[-1]
    uniforms = rng.random(n_samples)
    return np.interp(uniforms, cdf, grid)


def run_option_implied_method(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiration: pd.Timestamp,
    as_of_date: pd.Timestamp,
    horizon_days: int,
    n_sims: int,
    risk_free_rate: float,
    rng: np.random.Generator,
    thresholds: Iterable[float],
    options_source: str,
) -> ScenarioResult:
    """Run the option-implied distribution method with RND fallback handling."""

    if calls.empty and puts.empty:
        raise ValueError("Option chain is empty.")
    days_to_expiration = max(int((expiration.normalize() - as_of_date.normalize()).days), 1)
    time_to_expiry = days_to_expiration / 365.0
    atm_iv, iv_warnings = extract_atm_implied_volatility(
        calls=calls,
        puts=puts,
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
    )
    rnd = reconstruct_risk_neutral_density(
        calls=calls,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        spot=spot,
    )
    warnings = list(iv_warnings)
    if rnd.reconstructed:
        paths = simulate_rnd_terminal_paths(
            spot=spot,
            horizon_days=horizon_days,
            n_sims=n_sims,
            risk_free_rate=risk_free_rate,
            implied_volatility_value=atm_iv,
            rnd=rnd,
            rng=rng,
        )
        method_detail = "breeden_litzenberger_rnd"
        assumptions = [
            "Terminal risk-neutral density is reconstructed from call prices via Breeden-Litzenberger finite differences.",
            "Intrahorizon path diagnostics use ATM IV lognormal paths adjusted to match sampled RND terminals.",
        ]
    else:
        paths = simulate_lognormal_option_paths(
            spot=spot,
            horizon_days=horizon_days,
            n_sims=n_sims,
            risk_free_rate=risk_free_rate,
            implied_volatility_value=atm_iv,
            rng=rng,
        )
        method_detail = "atm_iv_lognormal_fallback"
        warnings.append(rnd.warning or "RND reconstruction unavailable; falling back to ATM IV lognormal.")
        assumptions = [
            "Risk-neutral terminal distribution is approximated by a lognormal distribution parameterized by ATM IV.",
            "Fallback was used because the option chain was too sparse or noisy for stable RND reconstruction.",
        ]

    common_assumptions = [
        f"Options source: {options_source}.",
        f"Selected expiration {expiration.date().isoformat()} ({days_to_expiration} calendar days from as-of date).",
        f"ATM implied volatility = {atm_iv:.6f}; risk-free rate = {risk_free_rate:.6f}.",
        "Option-implied probabilities are risk-neutral, not real-world realized probabilities.",
    ]
    extras = {
        "method_detail": method_detail,
        "expiration": expiration.date().isoformat(),
        "days_to_expiration": days_to_expiration,
        "atm_iv": atm_iv,
        "risk_free_rate": risk_free_rate,
        "rnd_reconstructed": rnd.reconstructed,
    }
    if rnd.reconstructed:
        extras["risk_neutral_density_grid"] = rnd.strikes
        extras["risk_neutral_density_values"] = rnd.density
    return build_result(
        name="Option-Implied Distribution",
        paths=paths,
        spot=spot,
        thresholds=thresholds,
        assumptions=assumptions + common_assumptions,
        warnings=warnings,
        extras=extras,
    )


def _prepare_call_curve(calls: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Prepare a clean call price curve for finite-difference density estimation."""

    if calls.empty or "strike" not in calls:
        return pd.DataFrame(columns=["strike", "price"])
    rows: list[dict[str, float]] = []
    lower = spot * 0.55
    upper = spot * 1.60
    for _, row in calls.iterrows():
        try:
            strike = float(row["strike"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(strike) or strike <= 0.0 or strike < lower or strike > upper:
            continue
        price = _option_mid_price(row)
        if price is not None and np.isfinite(price) and price > 0.0:
            rows.append({"strike": strike, "price": float(price)})
    if not rows:
        return pd.DataFrame(columns=["strike", "price"])
    frame = pd.DataFrame(rows).groupby("strike", as_index=False)["price"].mean()
    frame = frame.sort_values("strike").reset_index(drop=True)
    frame["price"] = frame["price"].cummin()
    return frame


def _option_mid_price(row: pd.Series) -> float | None:
    """Return bid/ask midpoint when possible, otherwise last traded price."""

    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    last = _safe_float(row.get("lastPrice"))
    if bid is not None and ask is not None and ask >= bid and ask > 0.0:
        midpoint = 0.5 * (bid + ask)
        if midpoint > 0.0:
            return midpoint
    if last is not None and last > 0.0:
        return last
    return None


def _safe_float(value: object) -> float | None:
    """Convert a scalar to float when finite, otherwise return None."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(converted):
        return converted
    return None


def _black_scholes_d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
) -> tuple[float, float]:
    """Compute Black-Scholes d1 and d2 terms."""

    variance_time = vol * np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * time_to_expiry) / variance_time
    d2 = d1 - variance_time
    return float(d1), float(d2)
