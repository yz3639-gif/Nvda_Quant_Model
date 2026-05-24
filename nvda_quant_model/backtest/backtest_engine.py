from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.metrics import calculate_metrics
from nvda_quant_model.config import StrategyConfig


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    daily_returns: pd.Series


class BacktestEngine:
    def __init__(self, config: StrategyConfig):
        self.initial_capital = config.initial_capital
        self.position_size = config.position_unit
        self.max_drawdown_limit = config.max_drawdown_limit
        self.cost = config.round_trip_cost
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct

    def backtest(self, signals: pd.DataFrame, prices: pd.DataFrame) -> BacktestResult:
        data = prices[["Open", "High", "Low", "Close"]].join(signals, how="left")
        data[["position", "direction", "confidence", "expected_return", "prob_up", "prob_down"]] = data[
            ["position", "direction", "confidence", "expected_return", "prob_up", "prob_down"]
        ].fillna(0.0)

        equity = self.initial_capital
        peak = equity
        prev_position = 0.0
        prev_close = None
        rows: list[dict[str, float | pd.Timestamp]] = []
        trades: list[dict[str, float | pd.Timestamp | str]] = []
        current_trade: dict[str, float | pd.Timestamp] | None = None

        for date, row in data.iterrows():
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            target_position = float(row["position"])

            if prev_close is None:
                rows.append({"Date": date, "equity": equity, "position": target_position, "strategy_return": 0.0})
                prev_close = close
                prev_position = target_position
                if target_position != 0:
                    current_trade = {"entry_date": date, "entry_equity": equity, "entry_price": close, "position": target_position}
                continue

            active_position = prev_position
            asset_return = close / prev_close - 1.0
            exit_reason = "close"
            if active_position > 0:
                stop_price = prev_close * (1.0 - self.stop_loss_pct)
                take_price = prev_close * (1.0 + self.take_profit_pct)
                if low <= stop_price:
                    asset_return = stop_price / prev_close - 1.0
                    exit_reason = "stop_loss"
                    target_position = 0.0
                elif high >= take_price:
                    asset_return = take_price / prev_close - 1.0
                    exit_reason = "take_profit"
                    target_position = 0.0
            elif active_position < 0:
                stop_price = prev_close * (1.0 + self.stop_loss_pct)
                take_price = prev_close * (1.0 - self.take_profit_pct)
                if high >= stop_price:
                    asset_return = -(stop_price / prev_close - 1.0)
                    exit_reason = "stop_loss"
                    target_position = 0.0
                elif low <= take_price:
                    asset_return = -(take_price / prev_close - 1.0)
                    exit_reason = "take_profit"
                    target_position = 0.0
                else:
                    asset_return = -asset_return

            gross_return = abs(active_position) * asset_return if active_position < 0 else active_position * asset_return
            turnover = abs(target_position - prev_position)
            cost_return = turnover * self.cost
            strategy_return = gross_return - cost_return
            prev_equity = equity
            equity *= 1.0 + strategy_return
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0

            if current_trade is not None and (np.sign(target_position) != np.sign(prev_position) or target_position == 0 or exit_reason != "close"):
                pnl = equity - float(current_trade["entry_equity"])
                trades.append(
                    {
                        "entry_date": current_trade["entry_date"],
                        "exit_date": date,
                        "entry_price": current_trade["entry_price"],
                        "exit_price": close,
                        "position": current_trade["position"],
                        "pnl": pnl,
                        "return": equity / float(current_trade["entry_equity"]) - 1.0,
                        "exit_reason": exit_reason,
                    }
                )
                current_trade = None

            if target_position != 0 and current_trade is None:
                current_trade = {"entry_date": date, "entry_equity": equity, "entry_price": close, "position": target_position}

            if drawdown <= -self.max_drawdown_limit:
                target_position = 0.0

            rows.append({"Date": date, "equity": equity, "position": target_position, "strategy_return": strategy_return})
            prev_close = close
            prev_position = target_position

        equity_curve = pd.DataFrame(rows).set_index("Date")
        daily_returns = equity_curve["strategy_return"].fillna(0.0)
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(equity_curve["equity"], daily_returns, trades_df, self.initial_capital)
        return BacktestResult(metrics, equity_curve, trades_df, daily_returns)

