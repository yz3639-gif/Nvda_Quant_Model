"""Compare current option-implied move with historical realized event moves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_implied_realized(
    event_id: str,
    option_row: dict,
    study_detail: pd.DataFrame,
    realized_window: str = "option_realized",
) -> dict:
    hist = study_detail.loc[study_detail["window"] == realized_window, "raw_return"] if not study_detail.empty else pd.Series(dtype=float)
    abs_hist = pd.to_numeric(hist, errors="coerce").abs().dropna()
    hist_median = float(abs_hist.median()) if not abs_hist.empty else np.nan
    hist_mean = float(abs_hist.mean()) if not abs_hist.empty else np.nan
    implied = float(option_row.get("implied_move", np.nan))
    ratio = hist_median / implied if np.isfinite(hist_median) and np.isfinite(implied) and implied > 0 else np.nan
    signal = "数据不足"
    if np.isfinite(ratio):
        if ratio > 1.2:
            signal = "市场低估事件影响 / 买波动率候选"
        elif ratio < 0.8:
            signal = "市场高估事件影响 / 卖波动率候选"
        else:
            signal = "隐含与历史大致匹配"
    skew = float(option_row.get("skew_25d")) if option_row.get("skew_25d") is not None else np.nan
    dist_skew = float(study_detail.loc[study_detail["window"] == "post_1_5", "car"].skew()) if not study_detail.empty else np.nan
    return {
        "event_id": event_id,
        "implied_move": implied,
        "historical_abs_move_median": hist_median,
        "historical_abs_move_mean": hist_mean,
        "mispricing_ratio": ratio,
        "signal": signal,
        "option_skew_25d": skew,
        "historical_car_skew_post_1_5": dist_skew,
        "skew_consistency": _skew_consistency(skew, dist_skew),
    }


def _skew_consistency(option_skew: float, car_skew: float) -> str:
    if not np.isfinite(option_skew) or not np.isfinite(car_skew):
        return "数据不足"
    if option_skew > 0 and car_skew < 0:
        return "期权左尾保护与历史左偏一致"
    if option_skew > 0 and car_skew > 0:
        return "期权左尾保护高于历史右偏分布"
    if option_skew <= 0 and car_skew > 0:
        return "期权未明显定价左尾，历史分布偏右"
    return "方向大致一致"
