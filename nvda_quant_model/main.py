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
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.news_sentiment import fetch_live_news, load_news_feature_cache, save_news_outputs
from nvda_quant_model.options_volatility import analyze_options


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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "cache"

    resolved_start, resolved_end, date_metadata = resolve_data_window(
        config.start_date,
        config.end_date,
        ticker=config.ticker,
        lookback_months=config.lookback_months,
    )
    config = replace(config, start_date=resolved_start, end_date=resolved_end)

    prices, external = load_market_data(
        config.ticker,
        config.start_date,
        config.end_date,
        cache_dir=cache_dir,
        force_refresh=force_refresh or date_metadata.get("end_source") == "latest_yfinance_daily",
    )
    peer_ohlcv = (
        load_peer_ohlcv_panel(
            warmup_start(config.start_date),
            config.end_date,
            cache_dir=cache_dir,
            force_refresh=force_refresh or date_metadata.get("end_source") == "latest_yfinance_daily",
        )
        if config.include_peer_events
        else {}
    )
    data_status = _build_data_status(config, prices, external, date_metadata, price_override)
    data_quality_csv, data_quality_json = _write_data_audit(output_dir, prices, external, config, peer_ohlcv)
    news_features = None
    if news_history_path is not None:
        news_features = load_news_feature_cache(news_history_path, prices.index)
    frame, feature_columns = build_model_frame(
        prices,
        external,
        config.start_date,
        config.end_date,
        config.ticker,
        include_fundamentals=config.include_fundamentals,
        peer_ohlcv=peer_ohlcv,
        news_features=news_features,
    )
    trainable = frame.dropna(subset=["target_return", "target_direction"])

    raw_signals, wf, wf_feature_importance = walk_forward_signals(trainable, feature_columns, config, config.start_date)
    report_prices = prices.loc[(prices.index >= pd.Timestamp(config.start_date)) & (prices.index <= pd.Timestamp(config.end_date))]
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

    latest_model = train_latest_model(trainable.loc[trainable.index <= pd.Timestamp(config.end_date)], feature_columns, selected_config)
    latest_row = frame.loc[frame.index <= pd.Timestamp(config.end_date)].tail(1)
    latest_raw = latest_model.predict(latest_row)
    latest_signal = apply_signal_policy(latest_raw, selected_config).iloc[-1]
    prediction = build_prediction(latest_signal, latest_row.iloc[-1], latest_row.index[-1], selected_config, price_override)
    model_comparison = latest_model.model_comparison_frame()
    news_overlay = None
    if include_live_news:
        news_dir = output_dir / "news_live"
        articles = fetch_live_news(days=news_days)
        news_payload = save_news_outputs(articles, news_dir, price_index=report_prices.index)
        news_overlay = news_payload["overlay"]
        prediction["news_overlay"] = {
            "signal": news_overlay.get("signal", 0),
            "sentiment_score": round(float(news_overlay.get("sentiment_score", 0.0)), 4),
            "article_count": int(news_overlay.get("article_count", 0)),
            "risk_flags": news_overlay.get("risk_flags", []),
            "confidence_adjustment": round(float(news_overlay.get("confidence_adjustment", 0.0)), 4),
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
        feature_importance = latest_model.feature_importance_

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
        "options_volatility": options_volatility,
        "news_overlay": news_overlay,
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
    parser.add_argument("--train-window", type=int, default=189)
    parser.add_argument("--test-window", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--include-fundamentals", action="store_true", help="Fetch yfinance fundamentals; slower and not strict point-in-time")
    parser.add_argument("--no-peer-events", action="store_true", help="Disable peer business-event features")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--rule-quantile", type=float, default=0.55)
    parser.add_argument("--rule-max-filters", type=int, default=2)
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
        top_k_features=args.top_k,
        include_fundamentals=args.include_fundamentals,
        include_peer_events=not args.no_peer_events,
        signal_threshold=args.threshold,
        max_exposure=args.max_exposure,
        rule_quantile=args.rule_quantile,
        rule_max_filters=args.rule_max_filters,
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
    print(f"Data status: {json.dumps(payload['data_status'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
