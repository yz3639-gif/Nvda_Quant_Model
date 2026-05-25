from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.news_impact_overlay import (
    adjust_monte_carlo_distribution,
    assert_no_data_leakage,
    build_price_feature_frame,
    inspect_mtbench_columns,
    predict_news_impact_scores,
    train_news_impact_bundle,
    walk_forward_news_overlay_validation,
)


def _row(idx: int, direction: str) -> dict[str, object]:
    base_time = pd.Timestamp("2024-01-02 14:30:00") + pd.Timedelta(days=idx)
    if direction == "bullish":
        prices = np.linspace(100.0, 103.0 + idx * 0.03, 40)
        output_change = 1.3 + (idx % 3) * 0.15
        text = "Nvidia demand growth strong data center guidance upgrade"
    elif direction == "bearish":
        prices = np.linspace(103.0, 100.0 - idx * 0.02, 40)
        output_change = -1.4 - (idx % 3) * 0.12
        text = "Nvidia export controls delay weak margin downgrade"
    else:
        prices = 101.0 + 0.05 * np.sin(np.arange(40))
        output_change = 0.05
        text = "Nvidia shares little changed mixed neutral update"
    timestamps = np.array([(base_time + pd.Timedelta(minutes=5 * i)).timestamp() for i in range(40)])
    technical = {
        "in_ema_10": pd.Series(prices).ewm(span=10, adjust=False).mean().tolist(),
        "in_sma_10": pd.Series(prices).rolling(10, min_periods=1).mean().tolist(),
        "out_ema_10": [999.0] * 5,
        "overall_sma_10": [888.0] * 5,
    }
    return {
        "input_timestamps": timestamps,
        "input_window": prices,
        "output_timestamps": timestamps[-5:] + 3600,
        "output_window": prices[-1] * (1.0 + output_change / 100.0) + np.linspace(0, 0.2, 5),
        "text": json.dumps(
            {
                "article_url": f"https://example.com/{idx}",
                "content": f"{text}\nPublished on: {base_time.isoformat()}Z",
            }
        ),
        "trend": json.dumps({"output_percentage_change": output_change}),
        "technical": json.dumps(technical),
        "alignment": json.dumps("consistent" if direction != "neutral" else "inconsistent"),
    }


def _sample_frame(n: int = 90) -> pd.DataFrame:
    directions = ["bullish", "bearish", "neutral"]
    return pd.DataFrame([_row(i, directions[i % len(directions)]) for i in range(n)])


def test_mtbench_column_audit_marks_labels_and_leakage() -> None:
    audit = inspect_mtbench_columns(
        [
            "input_timestamps",
            "input_window",
            "output_timestamps",
            "output_window",
            "text",
            "trend",
            "technical",
            "alignment",
        ]
    )

    assert audit.news_text_features == ["text"]
    assert "input_window" in audit.pre_news_price_features
    assert audit.label_fields == ["trend", "alignment"]
    assert "output_window" in audit.excluded_leakage_fields
    assert "technical" in audit.excluded_leakage_fields


def test_price_features_exclude_output_and_mixed_technical_leakage() -> None:
    frame = _sample_frame(12)
    features = build_price_feature_frame(frame)

    assert "pre_return_20" in features.columns
    assert "tech_in_ema_10_last_to_close" in features.columns
    assert not any(name.startswith("out_") or "overall" in name for name in features.columns)
    assert_no_data_leakage(list(features.columns))
    with pytest.raises(ValueError):
        assert_no_data_leakage(["text", "output_window", "trend"])


def test_train_news_overlay_bundle_and_score_outputs_are_bounded() -> None:
    frame = _sample_frame(90)
    bundle, metrics = train_news_impact_bundle(
        frame,
        model_type="logistic",
        text_method="tfidf",
        train_fraction=0.75,
        random_state=11,
    )
    scores = predict_news_impact_scores(bundle, frame.tail(6))

    assert bundle.direction_model is not None
    assert metrics["direction"]["accuracy"] >= 0.0
    for column in [
        "news_bullish_score",
        "news_bearish_score",
        "news_alignment_probability",
        "news_shock_strength",
    ]:
        assert scores[column].between(0.0, 1.0).all()
    assert scores["news_volatility_multiplier"].between(1.0, 1.25).all()


def test_news_overlay_adjusts_distribution_without_trade_signal() -> None:
    adjusted = adjust_monte_carlo_distribution(
        base_prob_up=0.52,
        base_mu=0.10,
        base_sigma=0.45,
        option_implied_volatility=0.50,
        news_scores={
            "news_bullish_score": 0.70,
            "news_bearish_score": 0.20,
            "news_shock_strength": 0.40,
        },
    )

    assert adjusted["adjusted_mu"] == pytest.approx(0.125)
    assert adjusted["adjusted_sigma"] == pytest.approx(0.495)
    assert "position" not in adjusted
    assert "signal" not in adjusted
    assert "not trading decision" in str(adjusted["distribution_role"])


def test_walk_forward_validation_compares_price_only_and_news_overlay() -> None:
    result = walk_forward_news_overlay_validation(
        _sample_frame(96),
        min_train_size=45,
        test_size=15,
        step=15,
        model_type="logistic",
        text_method="tfidf",
        random_state=5,
    )
    summary = result["summary"]
    windows = result["windows"]

    assert set(summary["model"]) == {"base_price_only", "news_overlay"}
    assert {"accuracy", "auc", "brier_score", "log_loss"}.issubset(summary.columns)
    assert len(windows) >= 4
    assert result["principle"] == "This module adjusts the distribution, not the trading decision."
