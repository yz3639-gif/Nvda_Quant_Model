from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.execution import ExecutionConfig
from nvda_quant_model.backtest.metrics import calculate_metrics
from nvda_quant_model.config import StrategyConfig


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    daily_returns: pd.Series
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    account: dict = field(default_factory=dict)


TRADE_COLUMNS = ["entry_date", "exit_date", "entry_price", "exit_price", "position", "pnl", "return",
                 "exit_reason", "gross_pnl", "fees", "entry_equity"]
FILL_COLUMNS = ["date", "sequence", "quantity", "price", "notional", "fee", "reason", "cash", "shares",
                "equity", "gross_realized_pnl", "realized_pnl", "cumulative_fees"]
EQUITY_COLUMNS = ["equity", "position", "strategy_return", "cash", "shares", "mark_price", "realized_pnl",
                  "unrealized_pnl", "cumulative_fees", "drawdown", "halted"]


class BacktestEngine:
    """Single-asset, share/cash backtest with one authoritative fill ledger.

    Existing call sites get corrected close execution with daily-reset brackets.
    New research should explicitly pass ExecutionConfig() for next-open/entry.
    A target is a fraction of *post-fee* equity and is rebalanced each session.
    Cash earns no interest; borrowing/stock-loan fees and liquidity are unmodeled.
    Slippage is an explicit monetary fee, not a second price adjustment.
    Risk halts compare open and close equity to the previous closing high-water
    mark (including initial capital), flattening permanently at the observed
    price. They are not an intrabar equity barrier or margin guarantee.
    Gap bracket exits precede signal orders and suppress re-entry that session.
    """

    def __init__(self, config: StrategyConfig, execution: ExecutionConfig | None = None):
        self.config = config
        self.execution = execution or ExecutionConfig(mode="close", stop_reference="daily_reset")
        self.initial_capital = config.initial_capital
        self.position_size = config.position_unit
        self.max_drawdown_limit = config.max_drawdown_limit
        self.cost = (config.commission + config.slippage if self.execution.cost_per_side is None
                     else self.execution.cost_per_side)
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not np.isfinite(self.cost) or not 0 <= self.cost < 1:
            raise ValueError("per-side cost must be finite and in [0, 1)")
        if not np.isfinite(self.max_drawdown_limit) or not 0 < self.max_drawdown_limit <= 1:
            raise ValueError("max_drawdown_limit must be in (0, 1]")
        if any(not np.isfinite(x) or x <= 0 for x in (self.stop_loss_pct, self.take_profit_pct)):
            raise ValueError("stop and take-profit percentages must be finite and positive")
        if not np.isfinite(config.max_exposure) or config.max_exposure < 0:
            raise ValueError("max_exposure must be finite and nonnegative")
        if config.max_exposure * self.cost >= 1:
            raise ValueError("exposure times fee rate must be below one for self-financing sizing")

    def backtest(self, signals: pd.DataFrame, prices: pd.DataFrame) -> BacktestResult:
        if self.execution.mode == "legacy_close":
            from nvda_quant_model.backtest.legacy_engine import BacktestEngine as LegacyEngine
            old = LegacyEngine(self.config).backtest(signals, prices)
            return BacktestResult(old.metrics, old.equity_curve, old.trades, old.daily_returns,
                                  account={"execution_mode": "legacy_close", "ledger_reconciled": False})
        data = self._validated_data(signals, prices)
        cash = float(self.initial_capital)
        shares = basis = realized = fees = 0.0
        peak = previous_equity = cash
        previous_close = None
        trail = None
        halted = False
        episode = None
        fills, trades, rows = [], [], []
        eps = 1e-10

        def equity_at(price):
            return cash + shares * price

        def fill(quantity, price, date, reason):
            # Reversals must be split by the caller so every fee belongs to one
            # unambiguous trade episode. Average cost handles partial reductions.
            nonlocal cash, shares, basis, realized, fees, episode, trail
            if abs(quantity) < eps:
                return
            before = equity_at(price)
            fee = abs(quantity * price) * self.cost
            gross = 0.0
            if abs(shares) < eps:
                episode = {"entry_date": date, "entry_equity": before, "entry_price": price,
                           "position": quantity * price / before if before else 0.0,
                           "gross_pnl": 0.0, "fees": 0.0}
                basis = price
                trail = price
            elif np.sign(quantity) != np.sign(shares):
                if abs(quantity) > abs(shares) + eps:
                    raise AssertionError("reversal must be split into close and open fills")
                gross = min(abs(quantity), abs(shares)) * (price - basis) * np.sign(shares)
            else:
                basis = (abs(shares) * basis + abs(quantity) * price) / abs(shares + quantity)
            cash -= quantity * price + fee
            shares += quantity
            realized += gross - fee
            fees += fee
            episode["gross_pnl"] += gross
            episode["fees"] += fee
            if abs(shares) < eps:
                shares = 0.0
            fills.append({"date": date, "sequence": len(fills), "quantity": quantity, "price": price,
                          "notional": abs(quantity * price), "fee": fee, "reason": reason,
                          "cash": cash, "shares": shares, "equity": equity_at(price),
                          "gross_realized_pnl": gross, "realized_pnl": realized, "cumulative_fees": fees})
            if shares == 0:
                pnl = episode["gross_pnl"] - episode["fees"]
                trades.append({**episode, "exit_date": date, "exit_price": price, "pnl": pnl,
                               "return": pnl / episode["entry_equity"], "exit_reason": reason})
                basis = 0.0
                episode = trail = None

        def rebalance(weight, price, date):
            # Solve T = weight * (equity - rate * abs(T-current_notional)).
            # Unlike subtracting a fee after sizing, weight=1 never borrows cash.
            eq = equity_at(price)
            if eq <= 0:
                return
            current = shares * price
            side = np.sign(weight * eq - current)
            target = weight * (eq + self.cost * side * current) / (1 + weight * self.cost * side)
            target_shares = target / price
            if shares and target_shares and np.sign(shares) != np.sign(target_shares):
                fill(-shares, price, date, "reversal")
            fill(target_shares - shares, price, date, "signal")

        def levels():
            if self.execution.stop_reference == "daily_reset":
                reference = previous_close if previous_close is not None else basis
                # A new position entered at the current open must not inherit a
                # bracket already crossed before it existed.
                if episode["entry_date"] == date and self.execution.mode == "next_open":
                    reference = basis
                stop_ref = take_ref = reference
            else:
                take_ref = basis
                stop_ref = trail if self.execution.stop_reference == "trailing" else basis
            direction = np.sign(shares)
            return stop_ref * (1 - direction * self.stop_loss_pct), take_ref * (1 + direction * self.take_profit_pct)

        def bracket_exit(open_price, high, low, date, gap_only=False):
            if not shares:
                return False
            stop, take = levels()
            long = shares > 0
            stop_gap = open_price <= stop if long else open_price >= stop
            take_gap = open_price >= take if long else open_price <= take
            # Gaps are observed first and have priority over intraday ambiguity.
            if stop_gap or take_gap:
                reason = "stop_loss" if stop_gap else "take_profit"
                fill(-shares, open_price, date, reason)
                return True
            if gap_only:
                return False
            stop_hit = low <= stop if long else high >= stop
            take_hit = high >= take if long else low <= take
            if stop_hit and (not take_hit or self.execution.intrabar_policy == "stop_first"):
                fill(-shares, stop, date, "stop_loss")
                return True
            if take_hit:
                fill(-shares, take, date, "take_profit")
                return True
            return False

        def check_halt(price, date):
            nonlocal halted
            if equity_at(price) <= peak * (1 - self.max_drawdown_limit):
                halted = True
                fill(-shares, price, date, "drawdown_stop")
            return halted

        for date, row in data.iterrows():
            op, high, low, close = (float(row[c]) for c in ("Open", "High", "Low", "Close"))
            # Overnight risk on the old inventory is checked before new orders.
            exited = bracket_exit(op, high, low, date, gap_only=True)
            check_halt(op, date)
            if shares and self.execution.stop_reference == "trailing":
                trail = max(trail, op) if shares > 0 else min(trail, op)
            if self.execution.mode == "next_open" and not halted and not exited:
                rebalance(float(row["target"]), op, date)
                check_halt(op, date)
            if not exited and not halted:
                exited = bracket_exit(op, high, low, date)
            check_halt(close, date)
            if self.execution.mode == "close" and not halted and not exited:
                rebalance(float(row["target"]), close, date)
                check_halt(close, date)
            if date == data.index[-1] and self.execution.terminal_policy == "liquidate":
                fill(-shares, close, date, "terminal")
            eq = equity_at(close)
            peak = max(peak, eq)
            unrealized = shares * (close - basis)
            if not np.isclose(eq, self.initial_capital + realized + unrealized, rtol=1e-10, atol=1e-7):
                raise AssertionError("fill ledger does not reconcile with equity")
            rows.append({"Date": date, "equity": eq, "position": shares * close / eq if eq else 0.0,
                         "strategy_return": eq / previous_equity - 1 if previous_equity else 0.0,
                         "cash": cash, "shares": shares, "mark_price": close, "realized_pnl": realized,
                         "unrealized_pnl": unrealized, "cumulative_fees": fees, "drawdown": eq / peak - 1,
                         "halted": halted})
            if shares and self.execution.stop_reference == "trailing":
                # A position opened at the close cannot use earlier highs/lows.
                just_opened = episode["entry_date"] == date and self.execution.mode == "close"
                extreme = close if just_opened else (high if shares > 0 else low)
                trail = max(trail, extreme) if shares > 0 else min(trail, extreme)
            previous_close, previous_equity = close, eq

        equity_curve = (pd.DataFrame(rows).set_index("Date") if rows else
                        pd.DataFrame(columns=EQUITY_COLUMNS, index=pd.DatetimeIndex([], name="Date")))
        daily_returns = equity_curve["strategy_return"].astype(float)
        trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        metrics = calculate_metrics(equity_curve["equity"], daily_returns, trades_df, self.initial_capital,
                                    include_initial_equity=True)
        metrics.update({"total_fees": fees, "realized_pnl": realized,
                        "unrealized_pnl": shares * (previous_close - basis) if shares else 0.0})
        return BacktestResult(metrics, equity_curve, trades_df, daily_returns,
                              pd.DataFrame(fills, columns=FILL_COLUMNS),
                              {"execution_mode": self.execution.mode, "stop_reference": self.execution.stop_reference,
                               "terminal_policy": self.execution.terminal_policy,
                               "intrabar_policy": self.execution.intrabar_policy, "cost_per_side": self.cost,
                               "ledger_reconciled": True, "open_trade": episode, "halted": halted,
                               "cash": cash, "shares": shares, "cost_basis": basis,
                               "realized_pnl": realized, "unrealized_pnl": metrics["unrealized_pnl"],
                               "cumulative_fees": fees, "equity": metrics["final_equity"]})

    def _validated_data(self, signals, prices):
        for frame, name in ((prices, "prices"), (signals, "signals")):
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError(f"{name} index must be unique and chronological")
        data = prices[["Open", "High", "Low", "Close"]].astype(float).copy()
        if not np.isfinite(data.to_numpy()).all() or (data <= 0).any().any():
            raise ValueError("OHLC prices must be finite and positive")
        if ((data.High < data[["Open", "Close"]].max(axis=1)) |
            (data.Low > data[["Open", "Close"]].min(axis=1)) | (data.High < data.Low)).any():
            raise ValueError("OHLC envelope is inconsistent")
        if "position" not in signals:
            raise ValueError("signals must contain position")
        raw = signals["position"]
        if not np.isfinite(raw.dropna().to_numpy(dtype=float)).all():
            raise ValueError("position must be finite")
        if (raw.abs() > self.config.max_exposure + 1e-10).any():
            raise ValueError("position exceeds configured max_exposure")
        weights = raw.reindex(data.index).fillna(0.0)
        data["target"] = weights.shift(1).fillna(0.0) if self.execution.mode == "next_open" else weights
        return data
