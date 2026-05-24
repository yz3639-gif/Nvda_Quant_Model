from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()), drawdown


def ulcer_index(equity: pd.Series) -> float:
    running_max = equity.cummax()
    pct_drawdown = (equity / running_max - 1.0).clip(upper=0.0) * 100
    return float(np.sqrt(np.mean(np.square(pct_drawdown))))


def calculate_metrics(
    equity: pd.Series,
    strategy_returns: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float]:
    returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
    days = max(len(returns), 1)
    final_equity = float(equity.iloc[-1]) if not equity.empty else initial_capital
    total_return = final_equity / initial_capital - 1.0
    annualized_return = (1 + total_return) ** (252 / days) - 1 if total_return > -1 else -1.0
    vol = returns.std(ddof=0)
    sharpe = np.sqrt(252) * returns.mean() / vol if vol and not np.isnan(vol) else 0.0
    downside = returns[returns < 0].std(ddof=0)
    sortino = np.sqrt(252) * returns.mean() / downside if downside and not np.isnan(downside) else 0.0
    mdd, _ = max_drawdown(equity)
    calmar = annualized_return / abs(mdd) if mdd < 0 else np.inf

    if trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    else:
        pnl = trades["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / len(pnl) if len(pnl) else 0.0
        gross_profit = wins.sum()
        gross_loss = losses.sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = losses.mean() if not losses.empty else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(mdd),
        "calmar_ratio": float(calmar),
        "ulcer_index": ulcer_index(equity),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "num_trades": int(len(trades)),
    }


def benchmark_metrics(close: pd.Series, initial_capital: float) -> tuple[dict[str, float], pd.Series, pd.Series]:
    returns = close.pct_change().fillna(0.0)
    equity = initial_capital * (1 + returns).cumprod()
    metrics = calculate_metrics(equity, returns, pd.DataFrame(), initial_capital)
    return metrics, equity, returns

