"""Financial invariants, observable fills, and explicit execution semantics."""
import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.execution import ExecutionConfig
from nvda_quant_model.config import StrategyConfig


def bars(closes, opens=None, highs=None, lows=None):
    op = closes if opens is None else opens
    return pd.DataFrame({"Open": op, "High": np.maximum(op, closes) if highs is None else highs,
                         "Low": np.minimum(op, closes) if lows is None else lows, "Close": closes},
                        index=pd.bdate_range("2025-01-01", periods=len(closes)))


def run(prices, weights, mode="close", stop_reference="entry", **kwargs):
    strategy_keys = {"commission", "slippage", "stop_loss_pct", "take_profit_pct", "max_drawdown_limit"}
    strategy = {"commission": 0.001, "slippage": 0.0005, "stop_loss_pct": 0.5,
                "take_profit_pct": 0.5, "max_drawdown_limit": 1.0}
    for key in strategy_keys:
        if key in kwargs:
            strategy[key] = kwargs.pop(key)
    signals = pd.DataFrame({"position": weights}, index=prices.index)
    # Full signal frame keeps the frozen replay interface usable too.
    for key in ["direction", "confidence", "expected_return", "prob_up", "prob_down"]:
        signals[key] = 0.0
    return BacktestEngine(StrategyConfig(**strategy),
                          ExecutionConfig(mode=mode, stop_reference=stop_reference, **kwargs)).backtest(signals, prices)


def assert_reconciles(result):
    curve = result.equity_curve
    assert np.allclose(curve.equity, curve.cash + curve.shares * curve.mark_price)
    assert np.allclose(curve.equity, 100_000 + curve.realized_pnl + curve.unrealized_pnl)
    assert np.isclose(result.fills.fee.sum(), result.metrics["total_fees"])
    cash, shares, fee = 100_000.0, 0.0, 0.0
    for fill in result.fills.itertuples():
        cash -= fill.quantity * fill.price + fill.fee
        shares += fill.quantity
        fee += fill.fee
        assert fill.cash == pytest.approx(cash)
        assert fill.shares == pytest.approx(shares, abs=1e-8)
        assert fill.cumulative_fees == pytest.approx(fee)
    if curve.shares.iloc[-1] == 0:
        assert result.trades.pnl.sum() == pytest.approx(curve.equity.iloc[-1] - 100_000)
    assert np.prod(1 + result.daily_returns) == pytest.approx(curve.equity.iloc[-1] / 100_000)


@pytest.mark.parametrize("weight,ohlc,expected", [(1, (80, 85, 75, 82), 80), (-1, (120, 125, 115, 118), 120)])
def test_gap_stop_fills_at_observed_open(weight, ohlc, expected):
    op, hi, lo, cl = ohlc
    result = run(bars([100, cl], [100, op], [100, hi], [100, lo]), [weight, weight],
                 stop_loss_pct=0.025, cost_per_side=0)
    assert result.trades.iloc[0].exit_price == expected
    assert result.trades.iloc[0].exit_reason == "stop_loss"
    assert result.metrics["final_equity"] == pytest.approx(80_000)
    assert_reconciles(result)


def test_entry_and_exit_costs_make_small_gross_win_a_net_loss():
    prices = bars([100, 100, 100.2])
    result = run(prices, [0, 1, 0])
    qty = 100_000 / (100 * 1.0015)
    expected = qty * 0.2 - qty * (100 + 100.2) * 0.0015
    assert result.trades.iloc[0].pnl == pytest.approx(expected)
    assert expected < 0
    assert result.metrics["win_rate"] == 0
    assert result.equity_curve.cash.iloc[1] == pytest.approx(0, abs=1e-8)
    assert_reconciles(result)
    legacy = run(prices, [0, 1, 0], mode="legacy_close")
    assert legacy.metrics["final_equity"] - 100_000 == pytest.approx(-100.075)
    assert legacy.trades.iloc[0].pnl == pytest.approx(49.925)
    assert legacy.account["ledger_reconciled"] is False


def test_initial_entry_fee_is_in_returns_drawdown_and_unrealized_reconciliation():
    result = run(bars([100]), [1])
    assert result.daily_returns.iloc[0] < 0
    assert result.metrics["max_drawdown"] == pytest.approx(result.daily_returns.iloc[0])
    assert result.trades.empty
    assert result.account["open_trade"] is not None
    assert result.metrics["realized_pnl"] == pytest.approx(-result.metrics["total_fees"])
    assert_reconciles(result)


def test_next_open_uses_prior_signal_and_does_not_fill_final_signal():
    result = run(bars([100, 110, 130], [100, 105, 120]), [1, 0, 1], mode="next_open", cost_per_side=0)
    assert list(result.fills.price) == [105, 120]
    assert list(result.fills.date) == list(result.equity_curve.index[1:])
    assert result.equity_curve.shares.iloc[0] == 0
    assert result.equity_curve.shares.iloc[-1] == 0
    assert_reconciles(result)


def test_scaling_partial_reductions_and_reversals_allocate_all_fees():
    result = run(bars([100, 102, 101, 103, 102, 100]), [0.4, 0.8, 0.3, -0.5, -0.8, 0])
    assert len(result.trades) == 2
    assert result.trades.fees.sum() == pytest.approx(result.fills.fee.sum())
    assert (result.fills.reason == "reversal").sum() == 1
    assert_reconciles(result)


@pytest.mark.parametrize("terminal_policy", ["mark", "liquidate"])
def test_terminal_open_position_is_marked_or_explicitly_liquidated(terminal_policy):
    result = run(bars([100, 102]), [0.5, 0.5], terminal_policy=terminal_policy)
    if terminal_policy == "mark":
        assert result.trades.empty
        assert result.equity_curve.unrealized_pnl.iloc[-1] > 0
    else:
        assert result.trades.iloc[-1].exit_reason == "terminal"
        assert result.account["open_trade"] is None
    assert_reconciles(result)


@pytest.mark.parametrize("policy,price", [("stop_first", 97.5), ("take_first", 104)])
def test_ambiguous_bar_conflict_is_explicit(policy, price):
    result = run(bars([100, 101], [100, 100], [100, 110], [100, 90]), [1, 1],
                 stop_loss_pct=0.025, take_profit_pct=0.04, intrabar_policy=policy, cost_per_side=0)
    assert result.trades.iloc[0].exit_price == price
    assert len(result.fills) == 2  # no re-entry on a bracket exit bar
    assert_reconciles(result)


def test_gap_take_profit_precedes_later_intraday_stop():
    result = run(bars([100, 95], [100, 110], [100, 115], [100, 90]), [1, 1],
                 stop_loss_pct=0.025, take_profit_pct=0.04, cost_per_side=0)
    assert result.trades.iloc[0].exit_reason == "take_profit"
    assert result.trades.iloc[0].exit_price == 110


def test_entry_daily_reset_and_trailing_have_distinct_reference_semantics():
    prices = bars([100, 110, 106], [100, 100, 110], [100, 114, 110], [100, 100, 104])
    entry = run(prices, [1, 1, 1], stop_loss_pct=0.05, cost_per_side=0)
    daily = run(prices, [1, 1, 1], stop_loss_pct=0.05, cost_per_side=0, stop_reference="daily_reset")
    trailing = run(prices, [1, 1, 1], stop_loss_pct=0.05, cost_per_side=0, stop_reference="trailing")
    assert entry.trades.empty
    assert daily.trades.iloc[0].exit_price == pytest.approx(104.5)
    assert trailing.trades.iloc[0].exit_price == pytest.approx(108.3)
    assert_reconciles(trailing)


def test_trailing_does_not_assume_current_high_precedes_current_low():
    result = run(bars([100, 110], [100, 100], [100, 120], [100, 99]), [1, 1],
                 stop_reference="trailing", stop_loss_pct=0.05, cost_per_side=0)
    assert result.trades.empty


def test_drawdown_halt_liquidates_and_charges_exit_fee_and_never_reenters():
    result = run(bars([100, 90, 95, 98]), [1, 1, 1, 1], max_drawdown_limit=0.05)
    assert result.trades.iloc[0].exit_reason == "drawdown_stop"
    assert result.trades.iloc[0].exit_price == 90
    assert result.equity_curve.shares.iloc[1:].eq(0).all()
    assert result.fills.fee.iloc[-1] > 0
    assert_reconciles(result)


def test_future_signal_changes_do_not_change_past_next_open_execution():
    prices = bars([100, 102, 104, 105])
    a = run(prices, [0.2, 0.4, 0.6, 0.8], mode="next_open")
    b = run(prices, [0.2, 0.4, -0.6, -0.8], mode="next_open")
    pd.testing.assert_frame_equal(a.equity_curve.iloc[:3], b.equity_curve.iloc[:3])


def test_next_open_entry_can_stop_same_day_at_entry_based_bracket():
    result = run(bars([100, 90], [100, 100], [100, 101], [100, 89]), [1, 0],
                 mode="next_open", stop_loss_pct=0.025, cost_per_side=0)
    assert list(result.fills.price) == [100, 97.5]
    assert result.trades.iloc[0].entry_date == result.trades.iloc[0].exit_date
    assert_reconciles(result)


def test_next_open_new_daily_reset_position_does_not_inherit_previous_gap():
    result = run(bars([100, 81], [100, 80], [100, 82], [100, 79]), [1, 0],
                 mode="next_open", stop_reference="daily_reset", stop_loss_pct=0.025, cost_per_side=0)
    assert result.trades.empty
    assert list(result.fills.price) == [80]
    assert_reconciles(result)


def test_randomized_scaling_and_reversals_reconcile_with_open_and_closed_positions():
    rng = np.random.default_rng(2026)
    for mode in ["next_open", "close"]:
        for terminal in ["mark", "liquidate"]:
            opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 100)))
            closes = opens * np.exp(rng.normal(0, 0.02, 100))
            highs = np.maximum(opens, closes) * 1.02
            lows = np.minimum(opens, closes) * 0.98
            result = run(bars(closes, opens, highs, lows), rng.choice([-1, -0.5, 0, 0.5, 1], 100),
                         mode=mode, stop_loss_pct=0.025, take_profit_pct=0.04,
                         terminal_policy=terminal)
            assert_reconciles(result)


def test_empty_input_and_invalid_ohlc_are_explicit():
    result = run(bars([]), [])
    assert result.metrics["final_equity"] == 100_000
    assert result.metrics["max_drawdown"] == 0
    prices = bars([100])
    prices["High"] = 99
    with pytest.raises(ValueError, match="envelope"):
        run(prices, [1])
