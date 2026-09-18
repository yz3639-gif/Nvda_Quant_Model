from __future__ import annotations

from nvda_quant_model.vix_snapshot import interpret_vix, parse_chart_result


def test_interpret_vix_risk_states() -> None:
    assert interpret_vix(31.0, -2.0) == "risk_off"
    assert interpret_vix(24.0, 9.0) == "risk_watch"
    assert interpret_vix(14.5, -1.0) == "risk_on"
    assert interpret_vix(16.8, -2.0) == "risk_neutral_to_on"


def test_parse_chart_result_builds_latest_vix_snapshot() -> None:
    result = {
        "timestamp": [1_800_000_000, 1_800_086_400, 1_800_172_800, 1_800_259_200, 1_800_345_600, 1_800_432_000],
        "indicators": {
            "quote": [
                {
                    "close": [18.0, 17.5, 17.2, 17.0, 16.8, 16.6],
                    "open": [18.2, 17.8, 17.4, 17.1, 17.0, 16.9],
                }
            ]
        },
    }

    snapshot = parse_chart_result(result, symbol="^VIX", fetched_at_utc="2026-01-01T00:00:00Z")

    assert snapshot["symbol"] == "^VIX"
    assert snapshot["level"] == 16.6
    assert snapshot["previous_close"] == 16.8
    assert snapshot["pct_from_prev_close"] < 0
    assert snapshot["pct_5d"] < 0
