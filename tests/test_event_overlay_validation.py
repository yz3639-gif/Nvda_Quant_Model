from __future__ import annotations

import pandas as pd

from nvda_quant_model.event_overlay.validation import walk_forward_event_overlay_validation


def _prices(n: int = 130) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(100.0, index=idx)
    for i in range(1, n):
        close.iloc[i] = close.iloc[i - 1] * (1.01 if i % 4 == 0 else 0.99 if i % 4 == 1 else 1.002)
    return pd.DataFrame({"Close": close, "Volume": 10_000_000 + pd.Series(range(n), index=idx) * 1000}, index=idx)


def _events(n: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-20", periods=n)
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "event_id": f"event_{i}",
                "published_at": date.strftime("%Y-%m-%dT13:00:00Z"),
                "source": "Reuters" if i % 2 == 0 else "CNBC",
                "headline": "Nvidia raises guidance" if i % 4 == 0 else "Nvidia export controls risk" if i % 4 == 1 else "Nvidia product update",
                "body": "Strong demand." if i % 4 == 0 else "Restriction risk." if i % 4 == 1 else "Mixed update.",
                "ticker_scope": "NVDA",
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_event_overlay_validation_is_time_based() -> None:
    result = walk_forward_event_overlay_validation(
        _events(),
        _prices(),
        min_train_size=30,
        test_size=10,
        step=10,
        target_horizon=1,
        model_type="logistic",
        text_method="tfidf",
        random_state=3,
    )
    summary = result["summary"]
    windows = result["windows"]
    group_summary = result["group_summary"]

    assert set(summary["model"]) == {"core_only", "core_event_overlay"}
    assert windows["train_size"].is_monotonic_increasing
    assert {"expected_move_coverage", "var_95_coverage", "downside_tail_recall"}.issubset(windows.columns)
    assert not group_summary.empty
    assert result["principle"] == "Event overlay adjusts distribution, not trading decision."
