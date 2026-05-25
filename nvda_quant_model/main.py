from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from nvda_quant_model.analysis.robustness_check import monte_carlo_bootstrap, oos_degradation
from nvda_quant_model.analysis.statistical_tests import statistical_tests
from nvda_quant_model.analysis.visualization import create_all_charts
from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.metrics import benchmark_metrics
from nvda_quant_model.backtest.walk_forward import train_latest_model, walk_forward_signals
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, StrategyConfig
from nvda_quant_model.news_sentiment import fetch_live_news, save_news_outputs
from nvda_quant_model.options_volatility import analyze_options
from nvda_quant_model.pipeline import prepare_model_inputs


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _plain_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2%}"


def _check(value: float, threshold: float, op: str) -> str:
    ok = value > threshold if op == ">" else value < threshold
    return "PASS" if ok else "FAIL"


def _load_precision_rule(path: str) -> Any:
    from nvda_quant_model.precision_search import PrecisionRule

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    row = payload.get("best", payload)
    fields: dict[str, Any] = {}
    for name in PrecisionRule.__dataclass_fields__:
        value = row.get(name)
        if isinstance(value, float) and np.isnan(value):
            value = None
        if name.startswith("require_") or name == "exclude_negative_pre_holiday":
            value = bool(value)
        elif name in {"train_window", "test_window"} and value is not None:
            value = int(value)
        elif name in {
            "momentum_quantile",
            "volume_quantile",
            "max_rsi",
            "min_price_60ma",
            "vix_quantile_cap",
            "stop_loss_pct",
            "take_profit_pct",
            "max_exposure",
        } and value is not None:
            value = float(value)
        fields[name] = value
    return PrecisionRule(**fields)


def _precision_feature_importance(rule: Any, feature_columns: list[str]) -> pd.Series:
    weights = {
        rule.momentum_feature: 0.26,
        "volume_sma_ratio": 0.20,
        "rsi_14": 0.10 if rule.max_rsi is not None else 0.0,
        "price_60ma_ratio": 0.12 if rule.min_price_60ma is not None else 0.0,
        "SMH_return": 0.10 if rule.require_smh_positive else 0.0,
        "QQQ_return": 0.10 if rule.require_qqq_positive else 0.0,
        "SP500_return": 0.10 if rule.require_sp500_positive else 0.0,
        "macd_hist": 0.08 if rule.require_macd_positive else 0.0,
        "obv_signal": 0.10 if rule.require_obv_positive else 0.0,
        "VIX_weekly_change": 0.10 if rule.vix_quantile_cap is not None else 0.0,
        "volatility_20": 0.06 if rule.require_vol_calm else 0.0,
        "volatility_60": 0.06 if rule.require_vol_calm else 0.0,
        "peer_positive_breadth_5d": 0.08 if rule.require_peer_breadth_positive else 0.0,
        "peer_mean_return_5d": 0.08 if rule.require_peer_mean_positive else 0.0,
        "peer_event_net_score_3d": 0.08 if rule.require_peer_event_net_positive else 0.0,
        "peer_business_stress": 0.08 if rule.require_no_peer_business_stress else 0.0,
        "pre_holiday_session": 0.04 if rule.exclude_negative_pre_holiday else 0.0,
        "pre_holiday_momentum": 0.04 if rule.exclude_negative_pre_holiday else 0.0,
    }
    importance = pd.Series(0.0, index=feature_columns)
    for feature, weight in weights.items():
        if feature in importance.index:
            importance.loc[feature] += weight
    importance = importance[importance > 0]
    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance.sort_values(ascending=False)


def _augment_precision_walk_forward(
    wf: pd.DataFrame,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    if wf.empty or {"annualized_return", "sharpe_ratio", "max_drawdown"}.issubset(wf.columns):
        return wf
    engine = BacktestEngine(config)
    rows = []
    for _, row in wf.iterrows():
        start = pd.Timestamp(row["period_start"])
        end = pd.Timestamp(row["period_end"])
        period_prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        period_signals = signals.reindex(period_prices.index)
        result = engine.backtest(period_signals, period_prices)
        enriched = row.to_dict()
        enriched.update(
            {
                "annualized_return": result.metrics["annualized_return"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "max_drawdown": result.metrics["max_drawdown"],
                "win_rate": result.metrics["win_rate"],
                "profit_factor": result.metrics["profit_factor"],
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows)


def apply_signal_policy(
    raw_signals: pd.DataFrame,
    config: StrategyConfig,
    threshold: float | None = None,
    min_expected_return: float | None = None,
) -> pd.DataFrame:
    threshold = config.signal_threshold if threshold is None else threshold
    min_expected_return = config.min_expected_return if min_expected_return is None else min_expected_return
    signals = raw_signals.copy()
    prob_up = signals["prob_up"].clip(0.01, 0.99)
    expected = signals["expected_return"]
    direction = pd.Series(0, index=signals.index, dtype=int)
    direction.loc[(prob_up >= threshold) & (expected >= min_expected_return)] = 1
    confidence = np.maximum(prob_up, 1.0 - prob_up)
    exposure = pd.Series(np.where(direction != 0, config.max_exposure, 0.0), index=signals.index)
    signals["direction"] = direction
    signals["confidence"] = confidence
    signals["prob_down"] = 1.0 - prob_up
    signals["position"] = direction * exposure
    return signals


def optimize_signal_policy(
    raw_signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: StrategyConfig,
) -> StrategyConfig:
    split_idx = max(int(len(prices) * 0.70), 60)
    train_prices = prices.iloc[:split_idx]
    train_signals = raw_signals.reindex(train_prices.index).dropna(subset=["prob_up", "expected_return"])
    if len(train_signals) < 40:
        return config

    grid_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    grid_min_returns = [0.0, 0.00025, 0.0005, 0.001]
    best_score = -np.inf
    best = config
    engine = BacktestEngine(config)
    for threshold in grid_thresholds:
        for min_ret in grid_min_returns:
            candidate = replace(config, signal_threshold=threshold, min_expected_return=min_ret)
            candidate_signals = apply_signal_policy(train_signals, candidate)
            result = engine.backtest(candidate_signals, train_prices)
            metrics = result.metrics
            recent_start = max(int(len(train_prices) * 0.70), len(train_prices) - 90)
            recent_prices = train_prices.iloc[recent_start:]
            recent_signals = candidate_signals.reindex(recent_prices.index).dropna(subset=["prob_up", "expected_return"])
            recent_result = engine.backtest(recent_signals, recent_prices) if len(recent_signals) else None
            recent_metrics = recent_result.metrics if recent_result is not None else {"sharpe_ratio": 0.0, "num_trades": 0}
            mdd_penalty = max(0.0, abs(metrics["max_drawdown"]) - config.max_drawdown_limit) * 5.0
            trade_penalty = 0.25 if metrics["num_trades"] < 8 else 0.0
            recent_trade_penalty = 0.75 if recent_metrics["num_trades"] < 2 else 0.0
            recent_sharpe_penalty = max(0.0, metrics["sharpe_ratio"] - recent_metrics["sharpe_ratio"]) * 0.20
            score = (
                metrics["sharpe_ratio"]
                + 0.5 * metrics["annualized_return"]
                + 0.15 * min(metrics["profit_factor"], 3.0)
                + 0.20 * recent_metrics["sharpe_ratio"]
                - mdd_penalty
                - trade_penalty
                - recent_trade_penalty
                - recent_sharpe_penalty
            )
            if score > best_score:
                best_score = score
                best = candidate
    return best


def next_us_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.Timestamp(date) + us_bday


def factor_signals(last_row: pd.Series) -> dict[str, int]:
    momentum_score = np.nanmean([last_row.get("5d_return", 0.0), last_row.get("20d_return", 0.0), last_row.get("SMH_return", 0.0)])
    zscore = last_row.get("price_zscore_20", 0.0)
    vol20 = last_row.get("volatility_20", 0.0)
    vol60 = last_row.get("volatility_60", 0.0)
    sp_return = last_row.get("SP500_return", 0.0)
    corr = last_row.get("correlation_sp500_20", 0.0)
    holiday_sentiment = 0
    if last_row.get("pre_holiday_session", 0.0) == 1:
        pre_holiday_momentum = last_row.get("pre_holiday_momentum", 0.0)
        vix_change = last_row.get("VIX_weekly_change", 0.0)
        holiday_sentiment = int(1 if pre_holiday_momentum > 0 and vix_change < 0.15 else -1 if pre_holiday_momentum < 0 or vix_change > 0.25 else 0)
    return {
        "momentum": int(np.sign(momentum_score)),
        "mean_reversion": int(1 if zscore < -1 else -1 if zscore > 1 else 0),
        "volatility": int(-1 if vol20 > vol60 * 1.20 else 1 if vol20 < vol60 * 0.80 else 0),
        "correlation": int(1 if sp_return > 0 and corr > 0 else -1 if sp_return < 0 and corr > 0 else 0),
        "holiday_sentiment": holiday_sentiment,
    }


def build_prediction(
    latest_signal: pd.Series,
    latest_row: pd.Series,
    as_of_date: pd.Timestamp,
    config: StrategyConfig,
    price_override: float | None = None,
) -> dict[str, Any]:
    model_close = float(latest_row["Close"])
    current_price = float(price_override) if price_override is not None else model_close
    price_source = "user_price_override" if price_override is not None else "latest_daily_close"
    signal = int(latest_signal["direction"])
    expected_return = float(latest_signal["expected_return"])
    next_date = next_us_trading_day(as_of_date)
    if signal >= 0:
        stop_loss = current_price * (1.0 - config.stop_loss_pct)
        take_profit = current_price * (1.0 + config.take_profit_pct)
    else:
        stop_loss = current_price * (1.0 + config.stop_loss_pct)
        take_profit = current_price * (1.0 - config.take_profit_pct)
    return {
        "date": next_date.strftime("%Y-%m-%d"),
        "weekday": next_date.day_name(),
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "model_close": round(model_close, 4),
        "current_price": round(current_price, 4),
        "price_source": price_source,
        "price_delta_vs_model_close": round(current_price / model_close - 1.0, 6) if model_close else 0.0,
        "signal": signal,
        "position": float(latest_signal["position"]),
        "confidence": round(float(latest_signal["confidence"]), 4),
        "expected_return": round(expected_return, 6),
        "target_price": round(current_price * (1.0 + expected_return), 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "factor_signals": factor_signals(latest_row),
        "holiday_context": {
            "pre_holiday_session": bool(latest_row.get("pre_holiday_session", 0.0)),
            "post_holiday_session": bool(latest_row.get("post_holiday_session", 0.0)),
            "next_session_gap_days": int(latest_row.get("next_session_gap_days", 1)),
            "next_session_has_holiday": bool(latest_row.get("next_session_has_holiday", 0.0)),
            "pre_holiday_momentum": round(float(latest_row.get("pre_holiday_momentum", 0.0)), 6),
        },
        "probability": {
            "prob_up": round(float(latest_signal["prob_up"]), 4),
            "prob_down": round(float(latest_signal["prob_down"]), 4),
        },
    }


def write_report(
    output_dir: Path,
    config: StrategyConfig,
    data_status: dict[str, Any],
    metrics: dict[str, float],
    benchmark: dict[str, float],
    wf: pd.DataFrame,
    feature_importance: pd.Series,
    stats: dict[str, Any],
    robustness: dict[str, Any],
    oos: dict[str, Any],
    prediction: dict[str, Any],
    charts: dict[str, Path],
    model_comparison: pd.DataFrame,
    options_volatility: dict[str, Any] | None = None,
    news_overlay: dict[str, Any] | None = None,
    order_flow_overlay: dict[str, Any] | None = None,
) -> Path:
    pass_map = {
        "annualized_return": _check(metrics["annualized_return"], 0.15, ">"),
        "sharpe_ratio": _check(metrics["sharpe_ratio"], 1.0, ">"),
        "max_drawdown": _check(abs(metrics["max_drawdown"]), 0.20, "<"),
        "win_rate": _check(metrics["win_rate"], 0.55, ">"),
        "profit_factor": _check(metrics["profit_factor"], 1.5, ">"),
        "calmar_ratio": _check(metrics["calmar_ratio"], 1.0, ">"),
        "p_value": "PASS" if stats["p_value"] < 0.05 else "FAIL",
        "oos": "PASS" if oos.get("passes_20pct_rule") else "FAIL",
    }

    lines: list[str] = []
    lines.append(f"# NVDA 多因素量化回测报告")
    lines.append("")
    lines.append(f"回测区间: {config.start_date} ~ {config.end_date}")
    lines.append(f"数据状态: {data_status.get('status', 'UNKNOWN')}")
    lines.append(f"最新OHLCV日期: {data_status.get('last_price_date')}")
    lines.append(f"价格源: {data_status.get('price_source', 'latest_daily_close')}")
    if data_status.get("warnings"):
        for warning in data_status["warnings"]:
            lines.append(f"警告: {warning}")
    lines.append(f"初始资金: ${config.initial_capital:,.0f}")
    lines.append(f"交易成本: 手续费 {config.commission:.2%} + 滑点 {config.slippage:.2%}")
    lines.append("")
    lines.append("## 回测总结")
    lines.append("")
    lines.append(f"- 最终资金: ${metrics['final_equity']:,.2f}")
    lines.append(f"- 累计收益: {_pct(metrics['total_return'])}")
    lines.append(f"- 年化收益率: {_pct(metrics['annualized_return'])} [{pass_map['annualized_return']}]")
    lines.append(f"- 基准年化收益率: {_pct(benchmark['annualized_return'])}")
    lines.append(f"- 超额年化收益: {_pct(metrics['annualized_return'] - benchmark['annualized_return'])}")
    lines.append("")
    lines.append("## 风险指标")
    lines.append("")
    lines.append(f"- 夏普比率: {metrics['sharpe_ratio']:.2f} [{pass_map['sharpe_ratio']}]")
    lines.append(f"- 最大回撤: {_pct(metrics['max_drawdown'])} [{pass_map['max_drawdown']}]")
    lines.append(f"- Calmar 比率: {metrics['calmar_ratio']:.2f} [{pass_map['calmar_ratio']}]")
    lines.append(f"- Sortino 比率: {metrics['sortino_ratio']:.2f}")
    lines.append(f"- Ulcer Index: {metrics['ulcer_index']:.2f}")
    lines.append("")
    lines.append("## 交易统计")
    lines.append("")
    lines.append(f"- 总交易数: {metrics['num_trades']}")
    lines.append(f"- 胜率: {_plain_pct(metrics['win_rate'])} [{pass_map['win_rate']}]")
    lines.append(f"- 平均盈利交易 PnL: ${metrics['avg_win']:,.2f}")
    lines.append(f"- 平均亏损交易 PnL: ${metrics['avg_loss']:,.2f}")
    lines.append(f"- 利润因子: {metrics['profit_factor']:.2f} [{pass_map['profit_factor']}]")
    lines.append("")
    lines.append("## Walk-Forward 验证")
    lines.append("")
    if wf.empty:
        lines.append("无 walk-forward 分段结果。")
    else:
        for _, row in wf.iterrows():
            lines.append(
                f"- {row['period_start']} ~ {row['period_end']}: "
                f"Sharpe={row['sharpe_ratio']:.2f}, Return={_pct(row['annualized_return'])}, "
                f"MDD={_pct(row['max_drawdown'])}"
            )
        lines.append(
            f"- 平均: Sharpe={wf['sharpe_ratio'].mean():.2f}, "
            f"Return={_pct(wf['annualized_return'].mean())}"
        )
        lines.append(
            f"- 标准差: Sharpe={wf['sharpe_ratio'].std(ddof=0):.2f}, "
            f"Return={_plain_pct(wf['annualized_return'].std(ddof=0))}"
        )
    lines.append("")
    lines.append("## 统计检验")
    lines.append("")
    lines.append(f"- 主检验 t-stat: {stats['t_stat']:.3f}")
    lines.append(f"- 主检验 p-value: {stats['p_value']:.4f} [{pass_map['p_value']}]")
    lines.append(f"- 持仓日超额收益 p-value: {stats['active_excess_p']:.4f}")
    lines.append(f"- 策略正收益 p-value: {stats['positive_return_p']:.4f}")
    lines.append(f"- 全日双尾超额收益 p-value: {stats['excess_two_sided_p']:.4f}")
    lines.append(f"- 策略 Sharpe / 基准 Sharpe: {stats['strategy_sharpe']:.2f} / {stats['benchmark_sharpe']:.2f}")
    lines.append(f"- Shapiro-Wilk p-value: {stats['shapiro_p']:.4f}")
    lines.append(f"- Durbin-Watson: {stats['durbin_watson']:.2f}")
    lines.append(f"- 结论: {stats['conclusion']}")
    lines.append("")
    lines.append("## 稳健性")
    lines.append("")
    lines.append(f"- Bootstrap Sharpe 5% 分位: {robustness['bootstrap_sharpe_p05']:.2f}")
    lines.append(f"- Bootstrap 最大回撤 95% 分位: {_plain_pct(robustness['bootstrap_mdd_p95'])}")
    lines.append(f"- OOS Sharpe 恶化: {_plain_pct(oos['sharpe_degradation'])} [{pass_map['oos']}]")
    lines.append("")
    lines.append("## 模型对比")
    lines.append("")
    if not model_comparison.empty:
        display_comparison = model_comparison.copy()
        if "selected_rule" in display_comparison.columns:
            display_comparison["selected_rule"] = display_comparison["selected_rule"].fillna("").astype(str).str.replace("|", "/", regex=False)
        lines.append(display_comparison.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 特征重要性 Top 12")
    lines.append("")
    for i, (feature, importance) in enumerate(feature_importance.head(12).items(), start=1):
        lines.append(f"{i}. {feature}: {importance * 100:.2f}%")
    lines.append("")
    lines.append("## 下一交易日预测")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(prediction, indent=2, ensure_ascii=False, default=_json_default))
    lines.append("```")
    lines.append("")
    gap_days = prediction.get("holiday_context", {}).get("next_session_gap_days", 1)
    if gap_days and gap_days > 1:
        lines.append(
            f"说明: 当前预测使用 {prediction['as_of_date']} 收盘后信息；下一美股交易日是 "
            f"{prediction['date']} ({prediction['weekday']})，间隔 {gap_days} 个自然日，已纳入假期/长周末情绪特征。"
        )
    else:
        lines.append(
            f"说明: 当前预测使用 {prediction['as_of_date']} 收盘后信息；下一美股交易日是 "
            f"{prediction['date']} ({prediction['weekday']})。"
        )
    if prediction.get("price_source") == "user_price_override":
        lines.append("说明: 风控价位按用户输入的实时价格锚定；模型特征仍按最新完整日线收盘计算。")
    lines.append("")
    if options_volatility:
        lines.append("## 期权波动覆盖层")
        lines.append("")
        near_week = options_volatility.get("near_week") or {}
        two_week = options_volatility.get("two_week") or {}
        fused = options_volatility.get("fused_two_week_range") or {}
        lines.append(
            f"- 数据源: {options_volatility.get('source', 'unknown')} / feed={options_volatility.get('feed', 'unknown')} / "
            f"as-of={options_volatility.get('as_of_date', 'unknown')}"
        )
        lines.append(f"- IV/HV20: {_fmt_num(options_volatility.get('iv_vs_hv20'))} ({options_volatility.get('volatility_signal', 'unknown')})")
        lines.append(f"- Skew 信号: {options_volatility.get('skew_signal', 'unknown')}")
        if near_week:
            lines.append(
                f"- 近周 ATM straddle: {_fmt_pct(near_week.get('straddle_expected_move_pct'))}, "
                f"期权隐含区间 {_fmt_num(near_week.get('straddle_low'))} ~ {_fmt_num(near_week.get('straddle_high'))}"
            )
        if two_week:
            lines.append(
                f"- 两周附近 ATM straddle: {_fmt_pct(two_week.get('straddle_expected_move_pct'))}, "
                f"期权隐含区间 {_fmt_num(two_week.get('straddle_low'))} ~ {_fmt_num(two_week.get('straddle_high'))}"
            )
        if fused:
            lines.append(
                f"- 模型+期权融合区间: base {_fmt_num(fused.get('low_base'))} ~ {_fmt_num(fused.get('high_base'))}; "
                f"stress/optimistic {_fmt_num(fused.get('low_stress'))} ~ {_fmt_num(fused.get('high_optimistic'))}"
            )
        lines.append("说明: 期权覆盖层只使用当前期权快照解释未来波动，不参与历史回测训练，避免 look-ahead bias。")
        lines.append("")
    if news_overlay:
        lines.append("## 实时新闻事件层")
        lines.append("")
        lines.append(f"- 新闻数量: {news_overlay.get('article_count', 0)}")
        lines.append(f"- 加权新闻情绪: {_fmt_num(news_overlay.get('sentiment_score'), 3)}")
        lines.append(f"- 新闻信号: {news_overlay.get('signal', 0)}")
        flags = news_overlay.get("risk_flags") or []
        lines.append(f"- 风险标签: {', '.join(flags) if flags else 'none'}")
        for item in (news_overlay.get("top_articles") or [])[:5]:
            lines.append(
                f"- {item.get('published_at', '')} | {item.get('source', '')} | "
                f"{item.get('title', '')} | score={_fmt_num(item.get('sentiment_score'), 3)}"
            )
        lines.append("说明: 实时新闻层只用于当前预测 overlay；只有提供 point-in-time 新闻缓存时才会进入历史训练/回测。")
        lines.append("")
    if order_flow_overlay:
        lines.append("## 实时买卖盘/分钟走势层")
        lines.append("")
        book = order_flow_overlay.get("top_of_book") or {}
        trend = order_flow_overlay.get("minute_trend") or {}
        micro = order_flow_overlay.get("micro_signal") or {}
        execution_filter = order_flow_overlay.get("execution_filter") or {}
        lines.append(
            f"- as-of: {order_flow_overlay.get('as_of', 'unknown')} / "
            f"feed={order_flow_overlay.get('feed', 'unknown')}"
        )
        lines.append(
            f"- Bid/Ask: {_fmt_num(book.get('bid_price'), 4)} x {_fmt_num(book.get('bid_size'), 0)} / "
            f"{_fmt_num(book.get('ask_price'), 4)} x {_fmt_num(book.get('ask_size'), 0)}"
        )
        lines.append(
            f"- Spread: {_fmt_num(book.get('spread'), 4)} ({_fmt_pct(book.get('spread_pct'))}), "
            f"盘口不平衡: {_fmt_num(book.get('quote_imbalance'), 3)}"
        )
        lines.append(
            f"- 1m close/VWAP: {_fmt_num(trend.get('latest_close'), 4)} / {_fmt_num(trend.get('latest_vwap'), 4)}, "
            f"5m={_fmt_pct(trend.get('return_5m'))}, 15m={_fmt_pct(trend.get('return_15m'))}, "
            f"30m={_fmt_pct(trend.get('return_30m'))}"
        )
        lines.append(
            f"- Micro signal: {micro.get('label', 'neutral')} "
            f"(score={_fmt_num(micro.get('score'), 1)}, confidence={_fmt_pct(micro.get('confidence'))})"
        )
        lines.append(
            f"- Execution filter: {execution_filter.get('recommendation', 'monitor_only')}; "
            f"use_as_entry_signal={execution_filter.get('use_as_entry_signal', False)}"
        )
        lines.append("说明: 股票买卖盘为 top-of-book best bid/ask，不是完整 Level-2 深度；当前该层只作为执行过滤/解释层，不作为独立开仓信号。")
        lines.append("")
    lines.append("## 图表")
    lines.append("")
    for name, path in charts.items():
        lines.append(f"- {name}: {path}")
    lines.append("")
    lines.append("## 风险提示")
    lines.append("")
    if config.include_fundamentals:
        lines.append("本报告是量化研究代码输出，不是投资建议。免费数据源的公司基本面不是严格 point-in-time 数据，报告已通过日期后移和缺失降级减少未来函数风险，但严肃实盘前仍需接入专业 point-in-time 数据。")
    else:
        lines.append("本报告是量化研究代码输出，不是投资建议。当前默认关闭 Yahoo 基本面特征，避免慢接口和非严格 point-in-time 数据污染每日更新；如需启用可加 --include-fundamentals。")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_data_status(
    config: StrategyConfig,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    date_metadata: dict[str, Any],
    price_override: float | None,
) -> dict[str, Any]:
    last_price_date = prices.index.max()
    external_last_dates = {
        str(col): external[col].dropna().index.max().strftime("%Y-%m-%d")
        for col in external.columns
        if not external[col].dropna().empty
    }
    warnings: list[str] = []
    resolved_end = pd.Timestamp(config.end_date)
    if last_price_date.normalize() < resolved_end.normalize():
        warnings.append(
            f"OHLCV latest date {last_price_date.date()} is older than resolved end date {resolved_end.date()}."
        )
    if price_override is not None:
        model_close = float(prices.loc[last_price_date, "Close"])
        delta = price_override / model_close - 1.0
        if abs(delta) > 0.02:
            warnings.append(
                f"User price override differs from latest daily close by {delta:.2%}; signal features still use daily close."
            )

    return {
        "status": "FRESH" if not warnings else "CHECK_WARNINGS",
        "requested_start": date_metadata.get("requested_start"),
        "requested_end": date_metadata.get("requested_end"),
        "resolved_start": config.start_date,
        "resolved_end": config.end_date,
        "start_source": date_metadata.get("start_source"),
        "end_source": date_metadata.get("end_source"),
        "lookback_months": config.lookback_months,
        "include_fundamentals": config.include_fundamentals,
        "include_peer_events": config.include_peer_events,
        "last_price_date": last_price_date.strftime("%Y-%m-%d"),
        "last_price_close": round(float(prices.loc[last_price_date, "Close"]), 4),
        "latest_daily_close_from_resolver": date_metadata.get("latest_daily_close"),
        "external_last_dates": external_last_dates,
        "price_source": "user_price_override" if price_override is not None else "latest_daily_close",
        "price_override": price_override,
        "warnings": warnings,
    }


def _write_data_audit(
    output_dir: Path,
    prices: pd.DataFrame,
    external: pd.DataFrame,
    config: StrategyConfig,
    peer_ohlcv: dict[str, pd.DataFrame] | None = None,
) -> tuple[Path, Path]:
    report_index = prices.loc[config.start_date : config.end_date].index
    rows: list[dict[str, Any]] = []

    def add_rows(source: str, data: pd.DataFrame) -> None:
        scoped = data.reindex(report_index)
        for col in scoped.columns:
            series = scoped[col]
            valid = series.dropna()
            rows.append(
                {
                    "source": source,
                    "field": str(col),
                    "rows": int(len(series)),
                    "non_null_rows": int(series.notna().sum()),
                    "missing_rate": float(series.isna().mean()) if len(series) else 1.0,
                    "first_valid_date": valid.index.min().strftime("%Y-%m-%d") if not valid.empty else None,
                    "last_valid_date": valid.index.max().strftime("%Y-%m-%d") if not valid.empty else None,
                    "latest_value": float(valid.iloc[-1]) if not valid.empty and pd.api.types.is_numeric_dtype(valid) else None,
                }
            )

    add_rows(config.ticker, prices[["Open", "High", "Low", "Close", "Volume"]])
    add_rows("external", external)
    for peer, peer_prices in (peer_ohlcv or {}).items():
        add_rows(f"peer_{peer}", peer_prices[["Open", "High", "Low", "Close", "Volume"]])
    audit = pd.DataFrame(rows)
    csv_path = output_dir / "data_quality.csv"
    json_path = output_dir / "data_quality.json"
    audit.to_csv(csv_path, index=False)
    json_path.write_text(audit.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path


def run(
    config: StrategyConfig,
    output_dir: Path,
    price_override: float | None = None,
    force_refresh: bool = False,
    options_snapshot_path: Path | None = None,
    include_live_news: bool = False,
    news_history_path: Path | None = None,
    news_days: int = 7,
    include_live_order_flow: bool = False,
    order_flow_json_path: Path | None = None,
    order_flow_feed: str = "iex",
    order_flow_minutes: int = 60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "cache"
    inputs = prepare_model_inputs(
        config,
        price_override=price_override,
        force_refresh=force_refresh,
        news_history_path=news_history_path,
        cache_dir=cache_dir,
    )
    config = inputs.config
    prices = inputs.prices
    external = inputs.external
    peer_ohlcv = inputs.peer_ohlcv
    data_status = inputs.data_status
    frame = inputs.frame
    feature_columns = inputs.feature_columns
    trainable = inputs.trainable
    report_prices = inputs.report_prices

    data_quality_csv, data_quality_json = _write_data_audit(output_dir, prices, external, config, peer_ohlcv)

    precision_rule = _load_precision_rule(config.precision_rule_path) if config.precision_rule_path else None
    if precision_rule is not None:
        from nvda_quant_model.precision_search import build_walk_forward_rule_signals

        selected_config = replace(
            config,
            train_window=precision_rule.train_window,
            test_window=precision_rule.test_window,
            stop_loss_pct=precision_rule.stop_loss_pct,
            take_profit_pct=precision_rule.take_profit_pct,
            max_exposure=precision_rule.max_exposure,
            signal_threshold=0.50,
            min_expected_return=0.0,
        )
        raw_signals, wf = build_walk_forward_rule_signals(frame, precision_rule, selected_config.start_date)
        signals = raw_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])
        wf = _augment_precision_walk_forward(wf, signals, report_prices, selected_config)
        wf_feature_importance = _precision_feature_importance(precision_rule, feature_columns)
    else:
        raw_signals, wf, wf_feature_importance = walk_forward_signals(trainable, feature_columns, config, config.start_date)
        raw_signals = raw_signals.reindex(report_prices.index).dropna(subset=["prob_up", "expected_return"])

        selected_config = optimize_signal_policy(raw_signals, report_prices, config)
        signals = apply_signal_policy(raw_signals, selected_config)

    engine = BacktestEngine(selected_config)
    result = engine.backtest(signals, report_prices)

    benchmark_ticker = MACRO_TICKERS["SP500"]
    benchmark_close = external[benchmark_ticker].reindex(report_prices.index).ffill().dropna()
    bench_metrics, bench_equity, bench_returns = benchmark_metrics(benchmark_close, selected_config.initial_capital)
    bench_equity = bench_equity.reindex(result.equity_curve.index).ffill()
    bench_returns = bench_returns.reindex(result.daily_returns.index).fillna(0.0)

    active_mask = signals["position"].shift().reindex(result.daily_returns.index).fillna(0.0) != 0.0
    stats = statistical_tests(result.daily_returns, bench_returns, active_mask=active_mask)
    robustness = monte_carlo_bootstrap(result.daily_returns, seed=selected_config.random_state)

    oos_cutoff = result.equity_curve.index.max() - pd.Timedelta(days=45)
    oos_returns = result.daily_returns.loc[result.daily_returns.index >= oos_cutoff]
    oos_equity = selected_config.initial_capital * (1 + oos_returns).cumprod()
    from nvda_quant_model.backtest.metrics import calculate_metrics

    oos_metrics = calculate_metrics(oos_equity, oos_returns, pd.DataFrame(), selected_config.initial_capital)
    oos = oos_degradation(result.metrics, oos_metrics)

    latest_row = frame.loc[frame.index <= pd.Timestamp(config.end_date)].tail(1)
    if precision_rule is not None:
        from nvda_quant_model.precision_search import latest_prediction_for_rule

        prediction = latest_prediction_for_rule(precision_rule, frame.loc[frame.index <= pd.Timestamp(config.end_date)], price_override)
        model_comparison = pd.DataFrame(
            [
                {
                    "model": "Strict_Precision_Rule",
                    "validation_accuracy": result.metrics["win_rate"],
                    "validation_log_loss": np.nan,
                    "ensemble_weight": 1.0,
                    "selected_rule": precision_rule.label,
                    "coverage": float((signals["position"] > 0).mean()) if len(signals) else 0.0,
                }
            ]
        )
    else:
        latest_model = train_latest_model(trainable.loc[trainable.index <= pd.Timestamp(config.end_date)], feature_columns, selected_config)
        latest_raw = latest_model.predict(latest_row)
        latest_signal = apply_signal_policy(latest_raw, selected_config).iloc[-1]
        prediction = build_prediction(latest_signal, latest_row.iloc[-1], latest_row.index[-1], selected_config, price_override)
        model_comparison = latest_model.model_comparison_frame()
    news_overlay = None
    if include_live_news:
        news_dir = output_dir / "news_live"
        overlay_path = news_dir / "live_news_overlay.json"
        try:
            articles = fetch_live_news(days=news_days)
            if not articles and overlay_path.exists():
                news_overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
                news_overlay["fetch_error"] = "live_news_fetch_returned_zero_articles"
            else:
                news_payload = save_news_outputs(articles, news_dir, price_index=report_prices.index)
                news_overlay = news_payload["overlay"]
        except Exception as exc:
            if overlay_path.exists():
                news_overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
                news_overlay["fetch_error"] = str(exc)
            else:
                news_overlay = {"article_count": 0, "sentiment_score": 0.0, "signal": 0, "risk_flags": [f"news_fetch_failed:{exc}"]}
        prediction["news_overlay"] = {
            "signal": news_overlay.get("signal", 0),
            "sentiment_score": round(float(news_overlay.get("sentiment_score", 0.0)), 4),
            "article_count": int(news_overlay.get("article_count", 0)),
            "risk_flags": news_overlay.get("risk_flags", []),
            "confidence_adjustment": round(float(news_overlay.get("confidence_adjustment", 0.0)), 4),
        }
    order_flow_overlay = None
    if include_live_order_flow or order_flow_json_path is not None:
        from nvda_quant_model.live_order_flow import analyze_order_flow, fetch_live_order_flow

        order_flow_dir = output_dir / "live_order_flow"
        order_flow_dir.mkdir(parents=True, exist_ok=True)
        if order_flow_json_path is not None:
            order_flow_payload = json.loads(order_flow_json_path.read_text(encoding="utf-8"))
            order_flow_overlay = (
                order_flow_payload
                if "micro_signal" in order_flow_payload and "top_of_book" in order_flow_payload
                else analyze_order_flow(order_flow_payload, symbol=config.ticker)
            )
        else:
            order_flow_overlay, order_flow_bars = fetch_live_order_flow(
                symbol=config.ticker,
                feed=order_flow_feed,
                minutes=order_flow_minutes,
            )
            if order_flow_bars is not None and not order_flow_bars.empty:
                order_flow_bars.to_csv(order_flow_dir / f"{config.ticker}_recent_1min_bars.csv")
        (order_flow_dir / f"{config.ticker}_order_flow.json").write_text(
            json.dumps(order_flow_overlay, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        prediction["order_flow_overlay"] = {
            "signal": order_flow_overlay.get("micro_signal", {}).get("signal", 0),
            "label": order_flow_overlay.get("micro_signal", {}).get("label", "neutral"),
            "score": order_flow_overlay.get("micro_signal", {}).get("score", 0.0),
            "confidence": order_flow_overlay.get("micro_signal", {}).get("confidence", 0.5),
            "latest_trade_price": order_flow_overlay.get("latest_trade", {}).get("price"),
            "spread_pct": order_flow_overlay.get("top_of_book", {}).get("spread_pct"),
            "quote_imbalance": order_flow_overlay.get("top_of_book", {}).get("quote_imbalance"),
            "return_5m": order_flow_overlay.get("minute_trend", {}).get("return_5m"),
            "return_15m": order_flow_overlay.get("minute_trend", {}).get("return_15m"),
            "execution_recommendation": order_flow_overlay.get("execution_filter", {}).get("recommendation"),
            "use_as_entry_signal": order_flow_overlay.get("execution_filter", {}).get("use_as_entry_signal", False),
        }
    options_volatility = None
    if options_snapshot_path is not None:
        range_projection_path = output_dir / "two_week_range_projection.json"
        options_volatility = analyze_options(
            options_snapshot_path,
            float(prediction["current_price"]),
            range_projection_path if range_projection_path.exists() else None,
            ticker=config.ticker,
        )
        (output_dir / "options_volatility_report.json").write_text(
            json.dumps(options_volatility, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    feature_importance = wf_feature_importance
    if feature_importance.empty or feature_importance.sum() == 0:
        feature_importance = pd.Series(dtype=float) if precision_rule is not None else latest_model.feature_importance_

    charts = create_all_charts(
        result.equity_curve["equity"],
        bench_equity,
        result.daily_returns,
        output_dir,
    )

    report_path = write_report(
        output_dir,
        selected_config,
        data_status,
        result.metrics,
        bench_metrics,
        wf,
        feature_importance,
        stats,
        robustness,
        oos,
        prediction,
        charts,
        model_comparison,
        options_volatility,
        news_overlay,
        order_flow_overlay,
    )

    signals.to_csv(output_dir / "signals.csv")
    result.equity_curve.to_csv(output_dir / "equity_curve.csv")
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    wf.to_csv(output_dir / "walk_forward.csv", index=False)
    feature_importance.rename("importance").to_csv(output_dir / "feature_importance.csv")
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    payload = {
        "config": asdict(selected_config),
        "data_status": data_status,
        "data_quality": {
            "csv": str(data_quality_csv),
            "json": str(data_quality_json),
        },
        "metrics": result.metrics,
        "benchmark_metrics": bench_metrics,
        "statistical_tests": stats,
        "robustness": robustness,
        "oos": oos,
        "prediction_next_day": prediction,
        "precision_rule": precision_rule.label if precision_rule is not None else None,
        "options_volatility": options_volatility,
        "news_overlay": news_overlay,
        "order_flow_overlay": order_flow_overlay,
        "report_path": str(report_path),
        "charts": {k: str(v) for k, v in charts.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVDA multi-factor walk-forward backtest")
    parser.add_argument("--start", default="auto", help="YYYY-MM-DD or auto for rolling lookback")
    parser.add_argument("--end", default="latest", help="YYYY-MM-DD or latest for latest available trading day")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--price-override", type=float, default=None, help="Optional live price anchor for target/stop/take-profit")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached yfinance CSV files and redownload")
    parser.add_argument("--options-snapshot-json", default=None, help="Optional Alpaca option snapshot JSON for live IV/skew overlay")
    parser.add_argument("--include-live-news", action="store_true", help="Fetch real-time news and add current news overlay")
    parser.add_argument("--news-days", type=int, default=7, help="Lookback window for live news RSS queries")
    parser.add_argument("--news-history-csv", default=None, help="Optional point-in-time historical news articles/features CSV for model training")
    parser.add_argument("--include-live-order-flow", action="store_true", help="Fetch live Alpaca top-of-book quote and minute-trend overlay")
    parser.add_argument("--order-flow-json", default=None, help="Optional existing Alpaca stock snapshot JSON for order-flow overlay")
    parser.add_argument("--order-flow-feed", default="iex", choices=["iex", "sip", "delayed_sip", "boats", "overnight", "otc"])
    parser.add_argument("--order-flow-minutes", type=int, default=60, help="Recent 1-minute bars to use for live trend overlay")
    parser.add_argument("--train-window", type=int, default=189)
    parser.add_argument("--test-window", type=int, default=42)
    parser.add_argument("--walk-forward-jobs", type=int, default=1, help="Parallel walk-forward workers; 1 keeps deterministic serial execution")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--include-fundamentals", action="store_true", help="Fetch yfinance fundamentals; slower and not strict point-in-time")
    parser.add_argument("--no-peer-events", action="store_true", help="Disable peer business-event features")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--rule-quantile", type=float, default=0.55)
    parser.add_argument("--rule-max-filters", type=int, default=2)
    parser.add_argument("--fast-rule-only", action="store_true", help="Skip diagnostic ML models; predictions remain rule-equivalent")
    parser.add_argument("--model-params-json", default=None, help="Optional tuned sklearn/xgboost parameter overrides JSON")
    parser.add_argument("--precision-rule-json", default=None, help="Optional strict PrecisionRule JSON from optimizer/selector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        lookback_months=args.lookback_months,
        train_window=args.train_window,
        test_window=args.test_window,
        walk_forward_jobs=args.walk_forward_jobs,
        top_k_features=args.top_k,
        include_fundamentals=args.include_fundamentals,
        include_peer_events=not args.no_peer_events,
        signal_threshold=args.threshold,
        max_exposure=args.max_exposure,
        rule_quantile=args.rule_quantile,
        rule_max_filters=args.rule_max_filters,
        fast_rule_only=args.fast_rule_only,
        model_params_path=args.model_params_json,
        precision_rule_path=args.precision_rule_json,
    )
    payload = run(
        config,
        Path(args.output_dir),
        price_override=args.price_override,
        force_refresh=args.force_refresh,
        options_snapshot_path=Path(args.options_snapshot_json) if args.options_snapshot_json else None,
        include_live_news=args.include_live_news,
        news_history_path=Path(args.news_history_csv) if args.news_history_csv else None,
        news_days=args.news_days,
        include_live_order_flow=args.include_live_order_flow,
        order_flow_json_path=Path(args.order_flow_json) if args.order_flow_json else None,
        order_flow_feed=args.order_flow_feed,
        order_flow_minutes=args.order_flow_minutes,
    )
    metrics = payload["metrics"]
    print("NVDA quantitative model complete")
    print(f"Report: {payload['report_path']}")
    print(
        "Metrics: "
        f"annualized={metrics['annualized_return']:.2%}, "
        f"sharpe={metrics['sharpe_ratio']:.2f}, "
        f"mdd={metrics['max_drawdown']:.2%}, "
        f"win_rate={metrics['win_rate']:.2%}, "
        f"profit_factor={metrics['profit_factor']:.2f}"
    )
    print(f"Prediction: {json.dumps(payload['prediction_next_day'], ensure_ascii=False)}")
    if payload.get("options_volatility"):
        options = payload["options_volatility"]
        fused = options.get("fused_two_week_range") or {}
        fused_low = fused.get("low_base")
        fused_high = fused.get("high_base")
        fused_text = f"{fused_low:.2f}~{fused_high:.2f}" if fused_low is not None and fused_high is not None else "n/a"
        print(
            "Options overlay: "
            f"vol={options.get('volatility_signal')}, "
            f"skew={options.get('skew_signal')}, "
            f"fused_base={fused_text}"
        )
    if payload.get("news_overlay"):
        news = payload["news_overlay"]
        print(
            "News overlay: "
            f"signal={news.get('signal')}, "
            f"sentiment={float(news.get('sentiment_score', 0.0)):.3f}, "
            f"articles={news.get('article_count')}, "
            f"flags={news.get('risk_flags')}"
        )
    if payload.get("order_flow_overlay"):
        order_flow = payload["order_flow_overlay"]
        micro = order_flow.get("micro_signal", {})
        trend = order_flow.get("minute_trend", {})
        print(
            "Order-flow overlay: "
            f"{micro.get('label', 'neutral')} score={float(micro.get('score', 0.0)):.1f}, "
            f"5m={float(trend.get('return_5m') or 0.0):.2%}, "
            f"quote_imbalance={float((order_flow.get('top_of_book') or {}).get('quote_imbalance') or 0.0):.3f}"
        )
    print(f"Data status: {json.dumps(payload['data_status'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
