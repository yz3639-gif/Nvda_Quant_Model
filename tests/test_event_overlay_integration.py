from __future__ import annotations

import pandas as pd

from nvda_quant_model.event_overlay.integration import apply_event_overlay_to_distribution
from nvda_quant_model.event_overlay.overlay_model import score_event_overlay, train_event_overlay_model


def _events(n: int = 45) -> pd.DataFrame:
    rows = []
    idx = pd.bdate_range("2025-01-15", periods=n)
    for i, day in enumerate(idx):
        bullish = i % 3 == 0
        bearish = i % 3 == 1
        rows.append(
            {
                "event_id": f"e{i}",
                "published_at": day.strftime("%Y-%m-%dT12:00:00Z"),
                "source": "Reuters" if i % 2 == 0 else "Yahoo Finance",
                "headline": "Nvidia raises guidance" if bullish else "Nvidia export controls risk" if bearish else "Nvidia neutral market update",
                "body": "Strong Blackwell demand and customer capex." if bullish else "China restrictions and weak margin risk." if bearish else "Mixed AI trade update.",
                "ticker_scope": "NVDA",
            }
        )
    return pd.DataFrame(rows)


def _prices(n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(100.0, index=idx)
    for i in range(1, n):
        close.iloc[i] = close.iloc[i - 1] * (1.006 if i % 3 == 0 else 0.994 if i % 3 == 1 else 1.001)
    return pd.DataFrame({"Close": close, "Volume": 10_000_000 + pd.Series(range(n), index=idx) * 1000}, index=idx)


def test_event_overlay_scores_distribution_fields_only() -> None:
    model, metrics = train_event_overlay_model(_events(), _prices(), target_horizon=1, text_method="tfidf")
    scores = score_event_overlay(model, _events(6).tail(3), _prices())

    assert metrics["accuracy"] >= 0.0
    assert scores["event_bullish_score"].between(0.0, 1.0).all()
    assert scores["event_bearish_score"].between(0.0, 1.0).all()
    assert scores["event_sigma_multiplier"].between(1.0, 1.30).all()
    assert scores["event_tail_risk_multiplier"].between(1.0, 1.50).all()
    assert "signal" not in scores.columns
    assert "position" not in scores.columns


def test_event_overlay_distribution_adjustment_does_not_trade() -> None:
    adjusted = apply_event_overlay_to_distribution(
        base_prob_up=0.52,
        base_mu=0.10,
        base_sigma=0.40,
        event_overlay={
            "event_mu_adjustment": 0.50,
            "event_sigma_multiplier": 1.60,
            "event_tail_risk_multiplier": 2.00,
        },
    )

    assert adjusted["adjusted_mu"] == 0.125
    assert adjusted["adjusted_sigma"] == 0.52
    assert adjusted["adjusted_tail_risk_multiplier"] == 1.50
    assert "signal" not in adjusted
    assert "position" not in adjusted
    assert "not a buy/sell/hold signal" in str(adjusted["distribution_role"])
