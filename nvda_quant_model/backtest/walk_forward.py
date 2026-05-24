from __future__ import annotations

import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.models.ensemble_model import EnsembleModel


def walk_forward_signals(
    data: pd.DataFrame,
    feature_columns: list[str],
    config: StrategyConfig,
    report_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data = data.dropna(subset=["target_return", "target_direction"]).copy()
    signals: list[pd.DataFrame] = []
    periods: list[dict[str, float | str]] = []
    feature_importance = pd.Series(0.0, index=feature_columns)
    n_models = 0

    start_idx = config.train_window
    while start_idx < len(data):
        train = data.iloc[start_idx - config.train_window : start_idx]
        test = data.iloc[start_idx : min(start_idx + config.test_window, len(data))]
        if test.empty:
            break
        if test.index[-1] < pd.Timestamp(report_start):
            start_idx += config.test_window
            continue

        model = EnsembleModel(config).fit(train, feature_columns)
        pred = model.predict(test)
        signals.append(pred)
        feature_importance = feature_importance.add(model.feature_importance_, fill_value=0.0)
        n_models += 1

        engine = BacktestEngine(config)
        result = engine.backtest(pred, test[["Open", "High", "Low", "Close"]])
        periods.append(
            {
                "period_start": test.index.min().strftime("%Y-%m-%d"),
                "period_end": test.index.max().strftime("%Y-%m-%d"),
                "annualized_return": result.metrics["annualized_return"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "max_drawdown": result.metrics["max_drawdown"],
                "win_rate": result.metrics["win_rate"],
                "profit_factor": result.metrics["profit_factor"],
            }
        )
        start_idx += config.test_window

    if not signals:
        raise ValueError("Not enough data to create walk-forward signals")

    all_signals = pd.concat(signals).sort_index()
    all_signals = all_signals[~all_signals.index.duplicated(keep="last")]
    period_frame = pd.DataFrame(periods)
    if n_models > 0 and feature_importance.sum() > 0:
        feature_importance = (feature_importance / n_models).sort_values(ascending=False)
        feature_importance = feature_importance / feature_importance.sum()
    return all_signals, period_frame, feature_importance


def train_latest_model(data: pd.DataFrame, feature_columns: list[str], config: StrategyConfig) -> EnsembleModel:
    train = data.tail(config.train_window)
    return EnsembleModel(config).fit(train, feature_columns)
