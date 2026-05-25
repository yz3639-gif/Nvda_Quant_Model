from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import benchmark_metrics
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.main import factor_signals, next_us_trading_day


@dataclass(frozen=True)
class PrecisionRule:
    train_window: int
    test_window: int
    momentum_feature: str
    momentum_quantile: float
    volume_quantile: float
    max_rsi: float | None
    min_price_60ma: float | None
    require_smh_positive: bool
    require_qqq_positive: bool
    require_sp500_positive: bool
    require_macd_positive: bool
    require_obv_positive: bool
    require_vol_calm: bool
    require_peer_breadth_positive: bool
    require_peer_mean_positive: bool
    require_peer_event_net_positive: bool
    require_no_peer_business_stress: bool
    vix_quantile_cap: float | None
    exclude_negative_pre_holiday: bool
    stop_loss_pct: float
    take_profit_pct: float
    max_exposure: float

    @property
    def label(self) -> str:
        flags = []
        for name in [
            "require_smh_positive",
            "require_qqq_positive",
            "require_sp500_positive",
            "require_macd_positive",
            "require_obv_positive",
            "require_vol_calm",
            "require_peer_breadth_positive",
            "require_peer_mean_positive",
            "require_peer_event_net_positive",
            "require_no_peer_business_stress",
            "exclude_negative_pre_holiday",
        ]:
            if getattr(self, name):
                flags.append(name.replace("require_", "").replace("_positive", "+").replace("exclude_", "no_"))
        if self.max_rsi is not None:
            flags.append(f"rsi<{self.max_rsi:g}")
        if self.min_price_60ma is not None:
            flags.append(f"p60>{self.min_price_60ma:g}")
        if self.vix_quantile_cap is not None:
            flags.append(f"vix<{self.vix_quantile_cap:g}")
        suffix = "+".join(flags) if flags else "base"
        return (
            f"tw{self.train_window}_sw{self.test_window}_{self.momentum_feature}"
            f"_mq{self.momentum_quantile:.2f}_vq{self.volume_quantile:.2f}_{suffix}"
        )


@dataclass(frozen=True)
class EvaluationContext:
    report_prices: pd.DataFrame
    report_frame: pd.DataFrame
    benchmark: dict[str, float]


def build_evaluation_context(
    base: StrategyConfig,
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    external: pd.DataFrame,
) -> EvaluationContext:
    report_prices = prices.loc[base.start_date : base.end_date]
    report_frame = frame.loc[base.start_date : base.end_date]
    benchmark_close = external[MACRO_TICKERS["SP500"]].reindex(report_prices.index).ffill().dropna()
    benchmark, _, _ = benchmark_metrics(benchmark_close, base.initial_capital)
    return EvaluationContext(report_prices=report_prices, report_frame=report_frame, benchmark=benchmark)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _rule_mask(data: pd.DataFrame, train: pd.DataFrame, rule: PrecisionRule) -> pd.Series:
    mom_threshold = float(train[rule.momentum_feature].quantile(rule.momentum_quantile))
    vol_threshold = float(train["volume_sma_ratio"].quantile(rule.volume_quantile))
    mask = (data[rule.momentum_feature] > mom_threshold) & (data["volume_sma_ratio"] > vol_threshold)

    if rule.max_rsi is not None and "rsi_14" in data:
        mask &= data["rsi_14"] < rule.max_rsi
    if rule.min_price_60ma is not None and "price_60ma_ratio" in data:
        mask &= data["price_60ma_ratio"] > rule.min_price_60ma
    if rule.require_smh_positive and "SMH_return" in data:
        mask &= data["SMH_return"] > 0
    if rule.require_qqq_positive and "QQQ_return" in data:
        mask &= data["QQQ_return"] > 0
    if rule.require_sp500_positive and "SP500_return" in data:
        mask &= data["SP500_return"] > 0
    if rule.require_macd_positive and "macd_hist" in data:
        mask &= data["macd_hist"] > 0
    if rule.require_obv_positive and "obv_signal" in data:
        mask &= data["obv_signal"] > 0
    if rule.require_vol_calm and {"volatility_20", "volatility_60"}.issubset(data.columns):
        mask &= data["volatility_20"] < data["volatility_60"] * 1.10
    if rule.require_peer_breadth_positive and "peer_positive_breadth_5d" in data:
        mask &= data["peer_positive_breadth_5d"] >= 0.55
    if rule.require_peer_mean_positive and "peer_mean_return_5d" in data:
        mask &= data["peer_mean_return_5d"] > 0
    if rule.require_peer_event_net_positive and "peer_event_net_score_3d" in data:
        mask &= data["peer_event_net_score_3d"] >= 0
    if rule.require_no_peer_business_stress and "peer_business_stress" in data:
        stress_cap = float(train["peer_business_stress"].quantile(0.75))
        mask &= data["peer_business_stress"] <= stress_cap
    if rule.vix_quantile_cap is not None and "VIX_weekly_change" in data:
        vix_cap = float(train["VIX_weekly_change"].quantile(rule.vix_quantile_cap))
        mask &= data["VIX_weekly_change"] < vix_cap
    if rule.exclude_negative_pre_holiday and {"pre_holiday_session", "pre_holiday_momentum"}.issubset(data.columns):
        mask &= ~((data["pre_holiday_session"] == 1) & (data["pre_holiday_momentum"] < 0))
    return mask.fillna(False)


def build_walk_forward_rule_signals(frame: pd.DataFrame, rule: PrecisionRule, report_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.dropna(subset=["target_return", "target_direction"]).copy()
    signals: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []
    start_idx = rule.train_window
    while start_idx < len(data):
        train = data.iloc[start_idx - rule.train_window : start_idx]
        test = data.iloc[start_idx : min(start_idx + rule.test_window, len(data))]
        if test.empty:
            break
        if test.index[-1] < pd.Timestamp(report_start):
            start_idx += rule.test_window
            continue

        train_active = _rule_mask(train, train, rule)
        train_returns = train.loc[train_active, "target_return"].dropna()
        train_precision = float((train.loc[train_active, "target_direction"] == 1).mean()) if train_active.any() else 0.5
        train_median_return = float(train_returns.median()) if not train_returns.empty else 0.0
        test_active = _rule_mask(test, train, rule)

        precision_shrunk = (train_precision * len(train_returns) + 0.5 * 12) / (len(train_returns) + 12)
        prob_up = pd.Series(0.5, index=test.index)
        expected_return = pd.Series(0.0, index=test.index)
        prob_up.loc[test_active] = float(np.clip(precision_shrunk, 0.51, 0.85))
        expected_return.loc[test_active] = max(train_median_return, 0.0005)

        pred = pd.DataFrame(
            {
                "direction": test_active.astype(int),
                "confidence": np.maximum(prob_up, 1.0 - prob_up),
                "expected_return": expected_return,
                "prob_up": prob_up,
                "prob_down": 1.0 - prob_up,
                "position": test_active.astype(float) * rule.max_exposure,
            },
            index=test.index,
        )
        signals.append(pred)

        active_returns = test.loc[test_active, "target_return"].dropna()
        period_rows.append(
            {
                "period_start": test.index.min().strftime("%Y-%m-%d"),
                "period_end": test.index.max().strftime("%Y-%m-%d"),
                "active_days": int(test_active.sum()),
                "precision": float((active_returns > 0).mean()) if len(active_returns) else np.nan,
                "avg_next_return": float(active_returns.mean()) if len(active_returns) else 0.0,
            }
        )
        start_idx += rule.test_window

    if not signals:
        raise ValueError("Not enough data for precision search")
    combined = pd.concat(signals).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, pd.DataFrame(period_rows)


def directional_metrics(signals: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    aligned = frame[["target_return", "target_direction"]].join(signals[["position", "prob_up"]], how="inner")
    active = aligned["position"] > 0
    active_rows = aligned.loc[active].dropna(subset=["target_return", "target_direction"])
    if active_rows.empty:
        return {
            "active_days": 0,
            "coverage": 0.0,
            "precision": 0.0,
            "avg_next_return": 0.0,
            "median_next_return": 0.0,
            "period_precision_std": 1.0,
            "worst_year_precision": 0.0,
        }
    yearly = active_rows.assign(year=active_rows.index.year).groupby("year")["target_direction"].mean()
    return {
        "active_days": int(len(active_rows)),
        "coverage": float(len(active_rows) / len(aligned)),
        "precision": float((active_rows["target_return"] > 0).mean()),
        "avg_next_return": float(active_rows["target_return"].mean()),
        "median_next_return": float(active_rows["target_return"].median()),
        "period_precision_std": float(yearly.std(ddof=0)) if len(yearly) > 1 else 0.0,
        "worst_year_precision": float(yearly.min()) if not yearly.empty else 0.0,
        "year_precision": {str(year): float(value) for year, value in yearly.items()},
    }


def score_candidate(
    metrics: dict[str, float],
    dmetrics: dict[str, Any],
    period_rows: pd.DataFrame,
    min_active_days: int,
    min_trades: int,
) -> float:
    precision = dmetrics["precision"]
    active_days = dmetrics["active_days"]
    trade_count = metrics["num_trades"]
    period_precision = period_rows["precision"].dropna()
    losing_periods = int((period_precision < 0.50).sum()) if not period_precision.empty else 99
    if active_days < min_active_days or trade_count < min_trades:
        shortfall = max(0, min_active_days - active_days) + 2 * max(0, min_trades - trade_count)
        return float(-100.0 - shortfall)
    too_many_penalty = max(0.0, dmetrics["coverage"] - 0.25) * 3.0
    drawdown_penalty = max(0.0, abs(metrics["max_drawdown"]) - 0.18) * 8.0
    instability_penalty = dmetrics["period_precision_std"] * 1.5 + losing_periods * 0.15
    return float(
        precision * 5.0
        + min(metrics["profit_factor"], 5.0) * 0.45
        + metrics["sharpe_ratio"] * 0.55
        + metrics["annualized_return"] * 0.75
        + metrics["calmar_ratio"] * 0.20
        + dmetrics["avg_next_return"] * 50.0
        - drawdown_penalty
        - too_many_penalty
        - instability_penalty
    )


def _priority_groups(values: list[Any], key_func: Any) -> list[list[Any]]:
    groups: dict[float, list[Any]] = {}
    for value in values:
        groups.setdefault(float(key_func(value)), []).append(value)
    return [groups[key] for key in sorted(groups)]


def candidate_rules(base: StrategyConfig, max_candidates: int | None = None) -> list[PrecisionRule]:
    required_sets = [
        {},
        {"require_smh_positive": True},
        {"require_qqq_positive": True},
        {"require_smh_positive": True, "require_qqq_positive": True},
        {"require_smh_positive": True, "require_sp500_positive": True},
        {"require_smh_positive": True, "require_macd_positive": True},
        {"require_smh_positive": True, "require_obv_positive": True},
        {"require_smh_positive": True, "require_vol_calm": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "require_macd_positive": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "require_obv_positive": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "require_vol_calm": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "exclude_negative_pre_holiday": True},
        {"require_smh_positive": True, "require_peer_breadth_positive": True},
        {"require_smh_positive": True, "require_peer_mean_positive": True},
        {"require_smh_positive": True, "require_peer_event_net_positive": True},
        {"require_smh_positive": True, "require_no_peer_business_stress": True},
        {"require_smh_positive": True, "require_peer_breadth_positive": True, "require_no_peer_business_stress": True},
        {"require_smh_positive": True, "require_peer_mean_positive": True, "require_peer_event_net_positive": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "require_peer_breadth_positive": True},
        {"require_smh_positive": True, "require_qqq_positive": True, "require_no_peer_business_stress": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_peer_breadth_positive": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_peer_mean_positive": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_peer_event_net_positive": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_no_peer_business_stress": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_peer_breadth_positive": True, "require_no_peer_business_stress": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_qqq_positive": True, "require_peer_breadth_positive": True},
        {"require_smh_positive": True, "require_obv_positive": True, "require_macd_positive": True, "require_peer_breadth_positive": True},
    ]

    train_windows = [126, 189, 252, 315]
    test_windows = [21, 42, 63]
    momentum_features = ["5d_return", "10d_return", "20d_return", "60d_return"]
    momentum_quantiles = [0.55, 0.60, 0.65, 0.70, 0.75]
    volume_quantiles = [0.50, 0.55, 0.60, 0.65]
    max_rsi_values = [None, 65.0, 72.0, 78.0]
    min_price_60ma_values = [None, 0.98, 1.00, 1.02]
    vix_quantile_cap_values = [None, 0.70, 0.80]
    stop_loss_values = [0.025, 0.035, 0.045]
    take_profit_values = [0.04, 0.05, 0.06]
    max_exposure_values = [0.75, 1.0]

    train_groups = _priority_groups(train_windows, lambda value: abs(value - base.train_window) / 252)
    test_groups = _priority_groups(test_windows, lambda value: abs(value - base.test_window) / 63)
    momentum_quantile_groups = _priority_groups(momentum_quantiles, lambda value: abs(value - 0.65))
    volume_quantile_groups = _priority_groups(volume_quantiles, lambda value: abs(value - 0.55))
    momentum_feature_groups = _priority_groups(
        momentum_features,
        lambda value: 0.0 if value in {"10d_return", "20d_return"} else 0.5,
    )
    flag_groups = _priority_groups(required_sets, lambda value: 0.0 if value.get("require_smh_positive", False) else 0.2)
    stop_loss_groups = _priority_groups(stop_loss_values, lambda value: abs(value - base.stop_loss_pct))
    take_profit_groups = _priority_groups(take_profit_values, lambda value: abs(value - base.take_profit_pct))

    rules: list[PrecisionRule] = []
    for train_group in train_groups:
        for test_group in test_groups:
            for momentum_quantile_group in momentum_quantile_groups:
                for volume_quantile_group in volume_quantile_groups:
                    for momentum_feature_group in momentum_feature_groups:
                        for flag_group in flag_groups:
                            for stop_loss_group in stop_loss_groups:
                                for take_profit_group in take_profit_groups:
                                    for train_window in train_group:
                                        for test_window in test_group:
                                            for momentum_feature in momentum_feature_group:
                                                for momentum_quantile in momentum_quantile_group:
                                                    for volume_quantile in volume_quantile_group:
                                                        for max_rsi in max_rsi_values:
                                                            for min_price_60ma in min_price_60ma_values:
                                                                for vix_quantile_cap in vix_quantile_cap_values:
                                                                    for stop_loss in stop_loss_group:
                                                                        for take_profit in take_profit_group:
                                                                            if take_profit <= stop_loss:
                                                                                continue
                                                                            for max_exposure in max_exposure_values:
                                                                                for flags in flag_group:
                                                                                    rules.append(
                                                                                        PrecisionRule(
                                                                                            train_window=train_window,
                                                                                            test_window=test_window,
                                                                                            momentum_feature=momentum_feature,
                                                                                            momentum_quantile=momentum_quantile,
                                                                                            volume_quantile=volume_quantile,
                                                                                            max_rsi=max_rsi,
                                                                                            min_price_60ma=min_price_60ma,
                                                                                            require_smh_positive=flags.get("require_smh_positive", False),
                                                                                            require_qqq_positive=flags.get("require_qqq_positive", False),
                                                                                            require_sp500_positive=flags.get("require_sp500_positive", False),
                                                                                            require_macd_positive=flags.get("require_macd_positive", False),
                                                                                            require_obv_positive=flags.get("require_obv_positive", False),
                                                                                            require_vol_calm=flags.get("require_vol_calm", False),
                                                                                            require_peer_breadth_positive=flags.get("require_peer_breadth_positive", False),
                                                                                            require_peer_mean_positive=flags.get("require_peer_mean_positive", False),
                                                                                            require_peer_event_net_positive=flags.get("require_peer_event_net_positive", False),
                                                                                            require_no_peer_business_stress=flags.get("require_no_peer_business_stress", False),
                                                                                            vix_quantile_cap=vix_quantile_cap,
                                                                                            exclude_negative_pre_holiday=flags.get("exclude_negative_pre_holiday", False),
                                                                                            stop_loss_pct=stop_loss,
                                                                                            take_profit_pct=take_profit,
                                                                                            max_exposure=max_exposure,
                                                                                        )
                                                                                    )
                                                                                    if max_candidates is not None and len(rules) >= max_candidates:
                                                                                        return rules
    return rules


def evaluate_rule(
    rule: PrecisionRule,
    base: StrategyConfig,
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    min_active_days: int,
    min_trades: int,
    context: EvaluationContext | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    context = context or build_evaluation_context(base, frame, prices, external)
    signals, period_rows = build_walk_forward_rule_signals(frame, rule, base.start_date)
    report_prices = context.report_prices
    signals = signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
    cfg = replace(
        base,
        train_window=rule.train_window,
        test_window=rule.test_window,
        stop_loss_pct=rule.stop_loss_pct,
        take_profit_pct=rule.take_profit_pct,
        max_exposure=rule.max_exposure,
    )
    result = BacktestEngine(cfg).backtest(signals, report_prices)
    dmetrics = directional_metrics(signals, context.report_frame)
    benchmark = context.benchmark
    score = score_candidate(result.metrics, dmetrics, period_rows, min_active_days, min_trades)
    row = {
        "label": rule.label,
        "score": score,
        **asdict(rule),
        **{f"bt_{key}": value for key, value in result.metrics.items()},
        **{f"dir_{key}": value for key, value in dmetrics.items() if key != "year_precision"},
        "benchmark_annualized_return": benchmark["annualized_return"],
        "benchmark_sharpe_ratio": benchmark["sharpe_ratio"],
        "year_precision": json.dumps(dmetrics.get("year_precision", {}), ensure_ascii=False),
    }
    return row, signals, period_rows


def latest_prediction_for_rule(
    rule: PrecisionRule,
    frame: pd.DataFrame,
    price_override: float | None = None,
) -> dict[str, Any]:
    trainable = frame.dropna(subset=["target_return", "target_direction"])
    latest_row = frame.tail(1)
    as_of_date = latest_row.index[-1]
    train = trainable.loc[trainable.index < as_of_date].tail(rule.train_window)
    if train.empty:
        raise ValueError("Not enough training data for latest precision prediction")

    latest_active = _rule_mask(latest_row, train, rule)
    train_active = _rule_mask(train, train, rule)
    train_returns = train.loc[train_active, "target_return"].dropna()
    train_precision = float((train.loc[train_active, "target_direction"] == 1).mean()) if train_active.any() else 0.5
    precision_shrunk = (train_precision * len(train_returns) + 0.5 * 12) / (len(train_returns) + 12)

    signal = int(latest_active.iloc[0])
    model_close = float(latest_row["Close"].iloc[0])
    current_price = float(price_override) if price_override is not None else model_close
    expected_return = max(float(train_returns.median()) if not train_returns.empty else 0.0, 0.0005) if signal else 0.0
    prob_up = float(np.clip(precision_shrunk, 0.51, 0.85)) if signal else 0.5
    next_date = next_us_trading_day(as_of_date)
    return {
        "date": next_date.strftime("%Y-%m-%d"),
        "weekday": next_date.day_name(),
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "model_close": round(model_close, 4),
        "current_price": round(current_price, 4),
        "price_source": "user_price_override" if price_override is not None else "latest_daily_close",
        "signal": signal,
        "position": float(rule.max_exposure if signal else 0.0),
        "prob_up": round(prob_up, 4),
        "prob_down": round(1.0 - prob_up, 4),
        "expected_return": round(expected_return, 6),
        "target_price": round(current_price * (1.0 + expected_return), 4),
        "stop_loss": round(current_price * (1.0 - rule.stop_loss_pct), 4),
        "take_profit": round(current_price * (1.0 + rule.take_profit_pct), 4),
        "factor_signals": factor_signals(latest_row.iloc[0]),
        "rule": rule.label,
        "train_active_days": int(train_active.sum()),
        "train_precision": round(train_precision, 4),
        "train_median_return": round(float(train_returns.median()) if len(train_returns) else 0.0, 6),
    }


def write_summary(output_dir: Path, rows: pd.DataFrame, best_row: pd.Series) -> Path:
    lines = [
        "# NVDA Precision Search Summary",
        "",
        "## Best Candidate",
        "",
        f"- Label: {best_row['label']}",
        f"- Score: {best_row['score']:.3f}",
        f"- Direction precision: {best_row['dir_precision']:.2%}",
        f"- Active days: {int(best_row['dir_active_days'])}",
        f"- Coverage: {best_row['dir_coverage']:.2%}",
        f"- Annualized return: {best_row['bt_annualized_return']:.2%}",
        f"- Sharpe: {best_row['bt_sharpe_ratio']:.2f}",
        f"- Max drawdown: {best_row['bt_max_drawdown']:.2%}",
        f"- Win rate: {best_row['bt_win_rate']:.2%}",
        f"- Profit factor: {best_row['bt_profit_factor']:.2f}",
        f"- Worst year precision: {best_row['dir_worst_year_precision']:.2%}",
        "",
        "## Top 20",
        "",
        rows.head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    path = output_dir / "precision_search_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-precision NVDA signal search")
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--max-candidates", type=int, default=1500)
    parser.add_argument("--min-active-days", type=int, default=25)
    parser.add_argument("--min-trades", type=int, default=18)
    parser.add_argument("--price-override", type=float, default=None)
    parser.add_argument("--no-peer-events", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "precision_search"))
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start, end, metadata = resolve_data_window(args.start, args.end, ticker=args.ticker, lookback_months=args.lookback_months)
    base = StrategyConfig(ticker=args.ticker, start_date=start, end_date=end, lookback_months=args.lookback_months)
    prices, external = load_market_data(
        base.ticker,
        base.start_date,
        base.end_date,
        cache_dir=PROJECT_ROOT / "cache",
        force_refresh=args.force_refresh or str(args.end).lower() in {"latest", "auto", "today", "now"},
    )
    base = replace(base, include_peer_events=not args.no_peer_events)
    peer_ohlcv = (
        load_peer_ohlcv_panel(
            warmup_start(base.start_date),
            base.end_date,
            cache_dir=PROJECT_ROOT / "cache",
            force_refresh=args.force_refresh or str(args.end).lower() in {"latest", "auto", "today", "now"},
        )
        if base.include_peer_events
        else {}
    )
    frame, _ = build_model_frame(
        prices,
        external,
        base.start_date,
        base.end_date,
        base.ticker,
        include_fundamentals=False,
        peer_ohlcv=peer_ohlcv,
    )
    context = build_evaluation_context(base, frame, prices, external)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_signals = pd.DataFrame()
    best_periods = pd.DataFrame()
    rules = candidate_rules(base, args.max_candidates)
    for idx, rule in enumerate(rules, start=1):
        try:
            row, signals, periods = evaluate_rule(
                rule,
                base,
                frame,
                prices,
                external,
                args.min_active_days,
                args.min_trades,
                context,
            )
        except Exception as exc:
            print(f"[{idx}/{len(rules)}] skipped {rule.label}: {exc}", flush=True)
            continue
        rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
            best_signals = signals
            best_periods = periods
            print(
                f"[{idx}/{len(rules)}] new best precision={row['dir_precision']:.2%} "
                f"ann={row['bt_annualized_return']:.2%} sharpe={row['bt_sharpe_ratio']:.2f} "
                f"mdd={row['bt_max_drawdown']:.2%} active={row['dir_active_days']}",
                flush=True,
            )
        elif idx % 100 == 0:
            print(f"[{idx}/{len(rules)}] checked; current best score={best['score']:.3f}", flush=True)

    if not rows:
        raise RuntimeError("No precision search rows produced")
    result_rows = pd.DataFrame(rows).sort_values("score", ascending=False)
    result_rows.to_csv(output_dir / "precision_results.csv", index=False)
    best_signals.to_csv(output_dir / "best_signals.csv")
    best_periods.to_csv(output_dir / "best_periods.csv", index=False)
    best_row = result_rows.iloc[0]
    best_rule_fields = {key: best_row[key] for key in PrecisionRule.__dataclass_fields__}
    for key in ["max_rsi", "min_price_60ma", "vix_quantile_cap"]:
        if isinstance(best_rule_fields[key], float) and np.isnan(best_rule_fields[key]):
            best_rule_fields[key] = None
    best_rule = PrecisionRule(**best_rule_fields)
    latest_prediction = latest_prediction_for_rule(best_rule, frame, args.price_override)
    (output_dir / "latest_precision_prediction.json").write_text(
        json.dumps(latest_prediction, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    payload = {
        "data_window": {
            **metadata,
            "resolved_start": base.start_date,
            "resolved_end": base.end_date,
            "rows": int(len(prices.loc[base.start_date : base.end_date])),
        },
        "best": best_row.to_dict(),
        "latest_prediction": latest_prediction,
    }
    (output_dir / "best_precision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    summary_path = write_summary(output_dir, result_rows, best_row)
    print(f"Saved: {summary_path}")
    print(json.dumps(payload["best"], indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
