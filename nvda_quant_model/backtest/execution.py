"""Explicit daily-bar execution assumptions for reproducible research.

Costs are charged once per fill on absolute traded notional. ``next_open``
uses the preceding observed session's signal; ``close`` uses today's signal
at today's close and is only suitable for a decision available by that close.
``legacy_close`` is the frozen, historically flawed accounting implementation.
Trailing stops use only the prior bar's favorable extreme (and current open),
never an unknowable ordering of today's high and low. Profit targets remain
entry-based except in daily_reset mode, where both brackets use prior close.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "next_open"
    stop_reference: str = "entry"
    terminal_policy: str = "mark"
    intrabar_policy: str = "stop_first"
    cost_per_side: float | None = None

    def __post_init__(self) -> None:
        choices = {
            "mode": {"close", "next_open", "legacy_close"},
            "stop_reference": {"entry", "daily_reset", "trailing"},
            "terminal_policy": {"mark", "liquidate"},
            "intrabar_policy": {"stop_first", "take_first"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if self.cost_per_side is not None and (
            not math.isfinite(self.cost_per_side) or not 0 <= self.cost_per_side < 1
        ):
            raise ValueError("cost_per_side must be finite and in [0, 1)")
