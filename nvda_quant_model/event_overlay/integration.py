from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_event_overlay_to_distribution(
    base_prob_up: float,
    base_mu: float,
    base_sigma: float,
    event_overlay: dict[str, Any] | pd.Series,
    lambda_mu: float = 0.05,
    sigma_bounds: tuple[float, float] = (1.0, 1.30),
    tail_bounds: tuple[float, float] = (1.0, 1.50),
) -> dict[str, float | str]:
    """Adjust a Monte Carlo distribution without creating a trading signal."""

    mu_adjustment = float(event_overlay.get("event_mu_adjustment", 0.0))
    sigma_multiplier = float(event_overlay.get("event_sigma_multiplier", 1.0))
    tail_multiplier = float(event_overlay.get("event_tail_risk_multiplier", 1.0))
    sigma_multiplier = float(np.clip(sigma_multiplier, *sigma_bounds))
    tail_multiplier = float(np.clip(tail_multiplier, *tail_bounds))
    return {
        "base_prob_up": float(base_prob_up),
        "base_mu": float(base_mu),
        "base_sigma": float(base_sigma),
        "event_mu_adjustment": float(np.clip(mu_adjustment, -1.0, 1.0)),
        "event_sigma_multiplier": sigma_multiplier,
        "event_tail_risk_multiplier": tail_multiplier,
        "lambda_mu": float(lambda_mu),
        "adjusted_mu": float(base_mu + lambda_mu * np.clip(mu_adjustment, -1.0, 1.0)),
        "adjusted_sigma": float(base_sigma * sigma_multiplier),
        "adjusted_tail_risk_multiplier": tail_multiplier,
        "distribution_role": "Event Impact Overlay adjusts distribution only; it is not a buy/sell/hold signal.",
    }
