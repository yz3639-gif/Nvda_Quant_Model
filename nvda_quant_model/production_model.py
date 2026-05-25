from __future__ import annotations

from typing import Any


BASELINE_PRODUCTION_LABEL = "tw147_sw63_10d_return_mq0.60_vq0.45_smh++obv++rsi<75+p60>0.96+vix<0.9"


PRODUCTION_RULE_OVERRIDES: dict[str, dict[str, float]] = {
    # 2026-05-25 retest across 24/36/60M: keeping the 2.5% stop and
    # tightening take-profit from 4.5% to 4.0% lifted 60M annualized return
    # above the 15% gate without breaking Sharpe, drawdown, win-rate, or PF.
    BASELINE_PRODUCTION_LABEL: {
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.040,
    }
}


def apply_production_rule_overrides(label: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Apply audited production overrides to a precision-rule field dict."""

    override = PRODUCTION_RULE_OVERRIDES.get(label)
    if not override:
        return fields
    adjusted = dict(fields)
    adjusted.update(override)
    return adjusted
