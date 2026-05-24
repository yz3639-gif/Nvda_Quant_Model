"""Pure Markdown report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_report(
    output_path: str | Path,
    context: dict[str, Any],
    events_detail: pd.DataFrame,
    sentiment_snapshot: pd.DataFrame,
    options_pricing: pd.DataFrame,
    bayesian_posteriors: pd.DataFrame,
    mc_summary: pd.DataFrame,
    mc_components: pd.DataFrame,
    sector_rotation: pd.DataFrame,
    signals_matrix: pd.DataFrame,
    historical_samples: pd.DataFrame,
    warnings: list[str],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    as_of = context["as_of_date"]
    lines.append(f"# 下周市场预测报告 (as of {as_of})")
    lines.append("")
    lines.append("## TL;DR")
    if not mc_summary.empty:
        row = mc_summary.iloc[0]
        opt_move = _weighted_option_move(options_pricing, mc_components)
        lines.extend(
            [
                f"- SPX 下周预期收益 (后验中位数): {_pct(row['median'])}",
                f"- 25/50/75 分位区间: [{_pct(row['p25'])}, {_pct(row['median'])}, {_pct(row['p75'])}]",
                f"- P(上涨) = {_pct(row['p_gt_0'], 0)}, P(>+2%) = {_pct(row['p_gt_2pct'], 0)}, P(<-2%) = {_pct(row['p_lt_minus_2pct'], 0)}",
                f"- 期权市场加权 implied move: ±{_pct(opt_move)}",
                f"- 置信度: {context.get('confidence_label', '中等')} ({context.get('confidence_note', '')})",
                f"- 关键驱动: {', '.join(mc_components.sort_values('weight', ascending=False).head(5)['event_name'].astype(str).tolist()) if not mc_components.empty else 'NA'}",
            ]
        )
    else:
        lines.append("- 蒙特卡洛汇总无法生成，原因通常是样本或价格数据不足。")
    best_sectors = _top_symbols(sector_rotation, n=3, ascending=False)
    worst_sectors = _top_symbols(sector_rotation, n=3, ascending=True)
    lines.append(f"- 最看好行业/主题: {best_sectors}")
    lines.append(f"- 最看弱行业/主题: {worst_sectors}")
    trade_flags = options_pricing.loc[options_pricing.get("signal", pd.Series(dtype=str)).astype(str).str.contains("候选", na=False)]
    lines.append(f"- 交易机会标注: {', '.join(trade_flags['event_id'].astype(str).tolist()) if not trade_flags.empty else '无强信号或数据不足'}")
    lines.append("")

    lines.append("## 数据完整性")
    total_sent = int(context.get("sentiment_total", 8))
    avail_sent = int(context.get("sentiment_available", 0))
    pct = avail_sent / total_sent if total_sent else 0.0
    missing_sent = context.get("sentiment_missing", [])
    env_dims = context.get("environment_dimensions", [])
    lines.extend(
        [
            f"- 情绪指标可用 / 总数: {avail_sent} / {total_sent} ({pct:.0%})",
            f"- 缺失: {', '.join(missing_sent) if missing_sent else '无'}",
            f"- 自动抓取: {int(context.get('sentiment_auto_loaded', 0))} 个",
            f"- 手动填入: {int(context.get('sentiment_manual_loaded', 0))} 个",
            f"- 环境匹配状态: {context.get('environment_status', 'DISABLED')} (使用 {len(env_dims)} 维度)",
        ]
    )
    lines.append("")

    lines.append("## 当前情绪与市场状态")
    lines.append(_df_to_md(_select_cols(sentiment_snapshot, ["metric", "value", "percentile", "z_score", "status", "extreme_flag", "source"])))
    lines.append("")

    lines.append("## 事件分析")
    for _, event in events_detail.sort_values(["date", "importance"], ascending=[True, False]).iterrows():
        event_id = event["event_id"]
        lines.append(f"### {event['event_name']} ({event['date']})")
        lines.append(f"- 类别: {event['category']} | 重要性: {event['importance']} | 来源: {event.get('source', '')}")
        lines.append(f"- 原始样本: N={event.get('raw_samples', 0)}；剔除 noise 后: N={event.get('clean_samples', 0)}；环境匹配子集: N={event.get('env_samples', 0)}")
        lines.append("")
        lines.append("**环境匹配子集**")
        lines.append(_df_to_md(_environment_meta_table(event)))
        if int(event.get("clean_samples", 0)) < 10:
            lines.append("- 小样本警告: N<10，统计检验应以非参数结果和经济直觉为主。")
        noise = historical_samples[(historical_samples["event_id"] == event_id) & (historical_samples["excluded_noise"] == True)]
        if not noise.empty:
            reasons = noise[["sample_id", "date", "noise_reason"]].head(10)
            lines.append("- 剔除原因清单:")
            lines.append(_df_to_md(reasons))
        spx = signals_matrix[
            (signals_matrix["event_id"] == event_id)
            & (signals_matrix["asset"] == "SPY")
            & (signals_matrix["subset"].isin(["full", "clean", "environment_macro_only", "environment_with_momentum"]))
        ]
        if not spx.empty:
            lines.append("")
            lines.append("**SPX 事件研究**")
            lines.append(_df_to_md(_format_study(spx)))
            lines.append("")
            lines.append("**环境匹配子集事件研究 (CAR [+1,+5])**")
            lines.append(_df_to_md(_environment_study_table(spx)))
        sectors = sector_rotation[(sector_rotation["event_id"] == event_id) & (sector_rotation["group"].isin(["sector", "theme"]))]
        if not sectors.empty:
            lines.append("")
            lines.append("**行业 / 主题轮动 (CAR [+1,+5])**")
            show = sectors.sort_values("mean", ascending=False).head(6)
            show = pd.concat([show, sectors.sort_values("mean", ascending=True).head(3)]).drop_duplicates(subset=["event_id", "group", "symbol"])
            lines.append(_df_to_md(_select_cols(show, ["group", "symbol", "name", "mean", "median", "win_rate", "p_value", "excess_mean_vs_benchmark"])))
        factors = sector_rotation[(sector_rotation["event_id"] == event_id) & (sector_rotation["group"] == "factor")]
        if not factors.empty:
            lines.append("")
            lines.append("**因子代理 (spread CAR [+1,+5])**")
            lines.append(_df_to_md(_select_cols(factors, ["factor", "long", "short", "mean", "median", "p_value"])))
        cross = signals_matrix[(signals_matrix["event_id"] == event_id) & (signals_matrix["matrix_type"] == "cross_asset")]
        if not cross.empty:
            lines.append("")
            lines.append("**跨资产 (CAR [+1,+5])**")
            lines.append(_df_to_md(_select_cols(cross.sort_values("mean", ascending=False).head(12), ["asset_name", "symbol", "mean", "median", "p_value", "corr_delta_mean"])))
        opt = options_pricing[options_pricing["event_id"] == event_id]
        if not opt.empty:
            lines.append("")
            lines.append("**隐含 vs 实际**")
            lines.append(_df_to_md(_select_cols(opt, ["ticker", "expiration", "implied_move", "historical_abs_move_median", "mispricing_ratio", "signal", "skew_25d", "historical_car_skew_post_1_5"])))
        post = bayesian_posteriors[bayesian_posteriors["event_id"] == event_id]
        if not post.empty:
            lines.append("")
            lines.append("**贝叶斯后验**")
            lines.append(_df_to_md(_select_cols(post, ["current_pre_drift", "prior_mean", "prior_sd", "posterior_mean", "posterior_sd", "kl_divergence", "p_up", "p_gt_1pct", "p_gt_2pct", "p_lt_minus_1pct", "warning"])))
        policy = str(event.get("policy_note", ""))
        if policy:
            lines.append("")
            lines.append("**政策走向预测**")
            lines.append(policy)
        lines.append("")

    lines.append("## 综合预测")
    lines.append(_df_to_md(mc_summary))
    if not mc_components.empty:
        lines.append("")
        lines.append("**事件权重**")
        lines.append(_df_to_md(_select_cols(mc_components.sort_values("weight", ascending=False), ["event_id", "event_name", "weight", "posterior_mean", "posterior_sd", "weighted_mean"])))
    lines.append("")

    lines.append("## 隐含 vs 实际定价对比")
    lines.append(_df_to_md(_select_cols(options_pricing, ["event_id", "implied_move", "historical_abs_move_median", "mispricing_ratio", "signal", "term_slope", "skew_25d"])))
    lines.append("")

    lines.append("## 行业 / 因子 / 跨资产矩阵")
    lines.append("完整矩阵见 `signals_matrix.csv` 与 `sector_rotation.csv`。下表为每个事件的核心后验。")
    lines.append(_df_to_md(_select_cols(events_detail, ["event_id", "event_name", "posterior_mean", "posterior_sd", "p_up", "raw_samples", "clean_samples", "env_samples"])))
    lines.append("")

    lines.append("## 与历史的关键差异")
    limitation_lines = [
        "- CPI/PPI/零售销售等宏观事件拒绝使用 proxy 日期；若无法取得真实发布日历和 surprise，样本为 0 并降级为 BLACK。",
        "- AAII/NAAIM/CFTC/put-call/资金流等指标若自动抓取失败且无手动文件，会被明确标为缺失；环境匹配只使用当前可用维度。",
        "- 总统访华历史样本数量有限，且 1972-1989 的 ETF 覆盖不足；行业轮动主要从 1998 以后样本估计。",
        "- 多事件同周发生，模型用加权合成降低双重计数，但无法完全分离互相污染的日内冲击。",
    ]
    lines.extend(limitation_lines)
    if warnings:
        lines.append("")
        lines.append("**运行警告**")
        for warning in warnings[:30]:
            lines.append(f"- {warning}")
    lines.append("")

    lines.append("## 方法论与局限")
    lines.extend(
        [
            "- AR 使用 Market Model：估计窗口 [-250,-20]；主指数对自身时使用常数均值模型，避免 beta=1 导致 AR 恒为 0。",
            "- v2 拒绝宏观 proxy 日期；若没有真实发布日历与 surprise，宏观事件样本为 0、标 BLACK，并在权重中封顶。",
            "- 环境匹配主路径排除过去 5/10/20 日价格收益；包含动量的匹配只作为诊断输出。",
            "- 贝叶斯 pre-drift 使用事件相关资产篮子，而不是全市场 5 日涨幅。",
            "- 报告同时输出包含 noise 与剔除 noise 后统计，并列出剔除原因。",
            "- 显著性包含 t 检验、Patell 近似、BMP 近似、Wilcoxon 与 Holm 校正；经济显著定义为 |CAR| > 0.5%。",
            "- 贝叶斯更新使用事件相关资产篮子的 pre-event drift 与 post-event CAR 的核条件分布；样本不足时退回经验先验。",
            "- 本报告不是投资建议。结果是模型假设下的概率分布，可能因数据延迟、样本污染、政策突发和流动性变化失效。",
        ]
    )
    lines.append("")
    lines.append("## 模型自省")
    lines.extend(
        [
            "1. 本次预测最大的不确定性来自宏观 surprise 历史、核心情绪/持仓数据和付费期权曲面的缺口。",
            "2. 如果模型错了，最可能错在历史类比假设：当前政策、估值和地缘叠加状态可能没有足够先例。",
            "3. 历史样本与当前环境的最大差异是关税基线、AI/半导体集中度、财政赤字与期限溢价水平。",
            "4. 如果再多 6 个月开发，会优先补 Bloomberg/Reuters consensus、OptionMetrics/ORATS、CFTC 全历史解析、ETF flows 和真实日内流动性。",
            "5. 如果用户只看一个数字，应看 composite Monte Carlo 的 50 分位，同时必须看事件数据质量标签。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_无可用数据_"
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_float_dtype(clean[col]):
            if any(token in col for token in ["return", "mean", "median", "rate", "p_", "prob", "move", "ratio", "weight", "sd", "ci_low", "ci_high", "skew", "drift", "percentile"]):
                clean[col] = clean[col].map(lambda x: _fmt(x, col))
            else:
                clean[col] = clean[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.4f}")
    headers = list(clean.columns)
    rows = clean.astype(str).replace("nan", "").values.tolist()
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(out)


def _format_study(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["subset", "window", "n", "mean", "median", "win_rate", "t_stat", "p_value", "p_value_holm", "wilcoxon_p", "patell_p", "bmp_p", "boot_ci_low", "boot_ci_high", "economic_significant"]
    return _select_cols(df, cols)


def _environment_meta_table(event: pd.Series) -> pd.DataFrame:
    used = str(event.get("env_used_dimensions", ""))
    used_display = ", ".join([item for item in used.split(",") if item]) if used else ""
    return pd.DataFrame(
        [
            {
                "使用维度数": event.get("env_n_dimensions", 0),
                "维度列表": used_display or "无",
                "匹配样本数": event.get("env_samples", 0),
                "距离 p50": event.get("env_distance_p50", np.nan),
                "状态": event.get("env_match_status", ""),
            }
        ]
    )


def _environment_study_table(spx: pd.DataFrame) -> pd.DataFrame:
    subset = spx[spx["window"] == "post_1_5"].copy()
    if subset.empty:
        return pd.DataFrame()
    label_map = {
        "full": "full",
        "clean": "clean",
        "environment_macro_only": "environment",
        "environment_with_momentum": "environment_with_momentum",
    }
    rows = []
    for key, label in label_map.items():
        row = subset[subset["subset"] == key]
        if row.empty:
            continue
        item = row.iloc[0]
        rows.append(
            {
                "metric": label,
                "n": item.get("n", np.nan),
                "mean": item.get("mean", np.nan),
                "median": item.get("median", np.nan),
                "win_rate": item.get("win_rate", np.nan),
                "p_value": item.get("p_value", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _select_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy() if df is not None and not df.empty else pd.DataFrame()


def _fmt(x: Any, col: str) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(val):
        return ""
    if col.endswith("_ratio") or "ratio" in col:
        return f"{val:.3f}"
    if any(token in col for token in ["p_", "prob", "win_rate", "move", "mean", "median", "return", "sd", "ci_low", "ci_high", "drift", "weight"]):
        return f"{val * 100:.2f}%"
    if "percentile" in col:
        return f"{val:.1f}"
    return f"{val:.3f}"


def _pct(x: float, digits: int = 2) -> str:
    if not np.isfinite(float(x)):
        return "NA"
    return f"{float(x) * 100:.{digits}f}%"


def _weighted_option_move(options: pd.DataFrame, components: pd.DataFrame) -> float:
    if options.empty:
        return float("nan")
    if components.empty or "weight" not in components:
        return float(pd.to_numeric(options["implied_move"], errors="coerce").mean())
    merged = options.merge(components[["event_id", "weight"]], on="event_id", how="left")
    weights = merged["weight"].fillna(1 / len(merged))
    moves = pd.to_numeric(merged["implied_move"], errors="coerce")
    valid = moves.notna()
    return float(np.average(moves[valid], weights=weights[valid])) if valid.any() else float("nan")


def _top_symbols(df: pd.DataFrame, n: int, ascending: bool) -> str:
    if df.empty or "symbol" not in df:
        return "NA"
    subset = df[df.get("group", "") == "sector"] if "group" in df else df
    if subset.empty:
        subset = df
    return ", ".join(subset.sort_values("mean", ascending=ascending).head(n)["symbol"].astype(str).tolist())
