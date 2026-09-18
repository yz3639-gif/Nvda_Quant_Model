from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.models.ensemble_model import EnsembleModel
from nvda_quant_model.data.data_validation import purge_immature_labels


def _walk_forward_windows(
    data: pd.DataFrame,
    config: StrategyConfig,
    report_start: str,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start_idx = config.train_window
    while start_idx < len(data):
        end_idx = min(start_idx + config.test_window, len(data))
        test = data.iloc[start_idx:end_idx]
        if test.empty:
            break
        if test.index[-1] >= pd.Timestamp(report_start):
            windows.append((start_idx, end_idx))
        start_idx += config.test_window
    return windows


def _walk_forward_period(
    args: tuple[pd.DataFrame, list[str], StrategyConfig, int, int],
) -> tuple[pd.DataFrame, dict[str, float | str], pd.Series]:
    data, feature_columns, config, start_idx, end_idx = args
    train = data.iloc[start_idx - config.train_window : start_idx]
    test = data.iloc[start_idx:end_idx]
    fit_at = test["decision_at"].iloc[0]
    train = purge_immature_labels(train, fit_at)
    model = EnsembleModel(config).fit(train, feature_columns)
    model.fit_at_ = fit_at
    pred = model.predict(test)
    pred["fit_at"] = fit_at
    pred["decision_at"] = test["decision_at"]
    pred["available_at"] = test["available_at"]
    pred["label_end_at"] = test["label_end_at"]
    engine = BacktestEngine(config)
    result = engine.backtest(pred, test[["Open", "High", "Low", "Close"]])
    period = {
        "period_start": test.index.min().strftime("%Y-%m-%d"),
        "period_end": test.index.max().strftime("%Y-%m-%d"),
        "annualized_return": result.metrics["annualized_return"],
        "sharpe_ratio": result.metrics["sharpe_ratio"],
        "max_drawdown": result.metrics["max_drawdown"],
        "win_rate": result.metrics["win_rate"],
        "profit_factor": result.metrics["profit_factor"],
    }
    return pred, period, model.feature_importance_


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
    windows = _walk_forward_windows(data, config, report_start)
    jobs = max(1, int(config.walk_forward_jobs))
    tasks = [(data, feature_columns, config, start_idx, end_idx) for start_idx, end_idx in windows]

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(_walk_forward_period, tasks))
    else:
        results = [_walk_forward_period(task) for task in tasks]

    for pred, period, importance in results:
        signals.append(pred)
        periods.append(period)
        feature_importance = feature_importance.add(importance, fill_value=0.0)
        n_models += 1

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
    fit_at = data["decision_at"].max()
    train = purge_immature_labels(data, fit_at).tail(config.train_window)
    model = EnsembleModel(config).fit(train, feature_columns)
    model.fit_at_ = fit_at
    return model
