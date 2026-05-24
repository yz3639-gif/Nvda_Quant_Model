"""Sentiment and positioning proxies.

Some requested sentiment series (AAII, NAAIM, detailed CFTC COT and ETF flows)
do not expose stable free machine-readable endpoints. This module reports those
as missing unless the user supplies manual overrides, and uses transparent market
proxies for environment matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from data.sentiment_fetcher import SentimentFetcher


CORE_SENTIMENT_METRICS = [
    "AAII_bull_bear_spread",
    "CFTC_emini_spx_net_percentile",
    "CFTC_10y_ust_net_percentile",
    "CBOE_equity_pc_10dma",
    "CBOE_total_pc_10dma",
    "NAAIM_exposure",
    "FINRA_margin_debt_yoy",
    "ICI_4w_equity_flow_b",
]

OPTIONAL_SENTIMENT_METRICS = [
    "CFTC_emini_spx_net",
    "CFTC_10y_ust_net",
]

DATE_KEYS = {
    "AAII_bull_bear_spread": "AAII_date",
    "CFTC_emini_spx_net": "CFTC_date",
    "CFTC_emini_spx_net_percentile": "CFTC_date",
    "CFTC_10y_ust_net": "CFTC_date",
    "CFTC_10y_ust_net_percentile": "CFTC_date",
    "CBOE_equity_pc_10dma": "CBOE_date",
    "CBOE_total_pc_10dma": "CBOE_date",
    "NAAIM_exposure": "NAAIM_date",
    "FINRA_margin_debt_yoy": "FINRA_date",
    "ICI_4w_equity_flow_b": "ICI_date",
}


@dataclass(slots=True)
class SentimentResult:
    snapshot: pd.DataFrame
    history: pd.DataFrame
    momentum_history: pd.DataFrame
    warnings: list[str]
    core_available_count: int


def build_sentiment(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    as_of_date: pd.Timestamp,
    manual_path: str | Path | None = None,
    enable_auto_fetch: bool = True,
    cache_dir: str | Path = Path(".cache/sentiment"),
) -> SentimentResult:
    """Build current sentiment table and historical feature panel."""

    history, momentum_history = _feature_history(prices, macro)

    snapshot = load_sentiment_snapshot(
        as_of_date=as_of_date,
        manual_file=Path(manual_path or "manual_sentiment.yaml"),
        enable_auto_fetch=enable_auto_fetch,
        cache_dir=Path(cache_dir),
        history=history,
    )
    warnings = list(snapshot.attrs.get("warnings", []))
    for metric, hist in snapshot.attrs.get("histories", {}).items():
        if hist is not None and not hist.empty and "date" in hist and metric in hist:
            hist = hist[["date", metric]].copy()
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist = hist.dropna(subset=["date"]).set_index("date").sort_index()
            history = history.join(hist, how="left")
    history = history.ffill(limit=21)
    core_available = int(
        snapshot[
            snapshot["metric"].isin(CORE_SENTIMENT_METRICS)
            & snapshot["status"].isin(["available", "manual", "auto"])
            & pd.to_numeric(snapshot["value"], errors="coerce").notna()
        ].shape[0]
    )
    if core_available == 0:
        warnings.append("AAII/NAAIM/CFTC/put-call/margin/flow metrics are missing unless manual_sentiment.yaml is supplied.")
    return SentimentResult(snapshot, history, momentum_history, warnings, core_available)


def load_sentiment_snapshot(
    as_of_date: str | pd.Timestamp,
    manual_file: Path = Path("manual_sentiment.yaml"),
    enable_auto_fetch: bool = True,
    cache_dir: Path = Path(".cache/sentiment"),
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load current sentiment snapshot from auto fetchers, manual overrides and price-based history."""

    as_of = pd.Timestamp(as_of_date).normalize()
    snapshot: dict[str, Any] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    histories: dict[str, pd.DataFrame] = {}
    auto_loaded = 0
    manual_loaded = 0

    if enable_auto_fetch:
        try:
            fetcher = SentimentFetcher(Path(cache_dir))
            auto_data = fetcher.fetch_all()
            auto_loaded = len([k for k in auto_data if k in CORE_SENTIMENT_METRICS])
            snapshot.update(auto_data)
            sources.update({k: fetcher.sources.get(k, "auto-fetched") for k in auto_data})
            warnings.extend(fetcher.warnings)
            histories.update(fetcher.histories)
        except Exception as exc:
            warnings.append(f"sentiment auto-fetch failed: {exc}")

    if manual_file.exists():
        manual_data = _read_manual_flat(manual_file)
        for key, value in manual_data.items():
            if value is not None and not _is_nan(value):
                snapshot[key] = value
                sources[key] = "manual_sentiment.yaml"
                if key in CORE_SENTIMENT_METRICS:
                    manual_loaded += 1

    rows = _price_based_rows(history, as_of) if history is not None else []
    price_metrics = {row["metric"] for row in rows}
    rows.extend(
        _sentiment_rows_from_snapshot(
            snapshot=snapshot,
            as_of=as_of,
            sources=sources,
            history=history,
            price_metrics=price_metrics,
        )
    )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["metric"], keep="last")
    frame.attrs["warnings"] = warnings
    frame.attrs["histories"] = histories
    frame.attrs["auto_loaded"] = auto_loaded
    frame.attrs["manual_loaded"] = manual_loaded
    frame.attrs["missing_core"] = [
        metric for metric in CORE_SENTIMENT_METRICS
        if frame.empty or frame.loc[frame["metric"] == metric, "status"].empty or frame.loc[frame["metric"] == metric, "status"].iloc[0] == "missing"
    ]
    return frame


def build_historical_sentiment_db(
    historical_events: list | pd.DataFrame,
    aaii_history: pd.DataFrame | None = None,
    naaim_history: pd.DataFrame | None = None,
    cot_history: pd.DataFrame | None = None,
    finra_history: pd.DataFrame | None = None,
    cboe_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backfill each historical event with the latest prior sentiment snapshot."""

    if isinstance(historical_events, pd.DataFrame):
        events_iter = historical_events.to_dict("records")
    else:
        events_iter = list(historical_events)
    history_specs = [
        ("AAII_bull_bear_spread", aaii_history),
        ("NAAIM_exposure", naaim_history),
        ("CFTC_emini_spx_net_percentile", cot_history),
        ("CFTC_10y_ust_net_percentile", cot_history),
        ("FINRA_margin_debt_yoy", finra_history),
        ("CBOE_equity_pc_10dma", cboe_history),
        ("CBOE_total_pc_10dma", cboe_history),
    ]
    rows = []
    for event in events_iter:
        sample_id = event.get("sample_id") or event.get("id")
        event_date = pd.to_datetime(event.get("date"), errors="coerce")
        if not sample_id or pd.isna(event_date):
            continue
        row = {"sample_id": sample_id, "event_date": event_date}
        for metric, history in history_specs:
            if history is None or history.empty or metric not in history:
                row[metric] = np.nan
                continue
            hist = history.copy()
            date_col = "date" if "date" in hist.columns else hist.index.name
            if date_col and date_col in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
                prior = hist[hist["date"] <= event_date].dropna(subset=[metric]).tail(1)
            else:
                hist = hist.copy()
                hist.index = pd.to_datetime(hist.index, errors="coerce")
                prior = hist[hist.index <= event_date].dropna(subset=[metric]).tail(1)
            row[metric] = float(prior.iloc[0][metric]) if not prior.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id") if rows else pd.DataFrame()


def _feature_history(prices: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(index=prices.index)
    momentum = pd.DataFrame(index=prices.index)
    if "SPY" in prices:
        spy = pd.to_numeric(prices["SPY"], errors="coerce").dropna()
        ret = spy.pct_change(fill_method=None)
        momentum.loc[spy.index, "SPY_20d_return"] = spy.pct_change(20, fill_method=None)
        momentum.loc[spy.index, "SPY_5d_return"] = spy.pct_change(5, fill_method=None)
        momentum.loc[spy.index, "SPY_distance_200dma"] = spy / spy.rolling(200).mean() - 1.0
        frame.loc[spy.index, "SPY_20d_realized_vol"] = ret.rolling(20).std() * np.sqrt(252)
    if "^VIX" in prices:
        frame["VIX"] = prices["^VIX"]
    if "^VIX" in prices and "^VIX3M" in prices:
        frame["VIX_term_level"] = prices["^VIX"] - prices["^VIX3M"]
    if "^MOVE" in prices:
        frame["MOVE"] = prices["^MOVE"]
    if "HY_OAS" in macro:
        frame = frame.join(macro[["HY_OAS"]], how="left")
        frame["HY_OAS_1m_change"] = frame["HY_OAS"] - frame["HY_OAS"].shift(21)
    if "IG_OAS" in macro:
        frame = frame.join(macro[["IG_OAS"]], how="left")
    for col in ["UST2Y_YIELD", "UST5Y_YIELD", "UST10Y_YIELD", "UST30Y_YIELD", "T10YIE", "T5YIE", "DFII10", "UNRATE", "FEDFUNDS", "WALCL", "NFCI", "USEPUINDXD"]:
        if col in macro and col not in frame:
            frame = frame.join(macro[[col]], how="left")
    if "UST2Y_YIELD" in frame and "UST10Y_YIELD" in frame:
        frame["curve_2s10s"] = frame["UST10Y_YIELD"] - frame["UST2Y_YIELD"]
    if "UST5Y_YIELD" in frame and "UST30Y_YIELD" in frame:
        frame["curve_5s30s"] = frame["UST30Y_YIELD"] - frame["UST5Y_YIELD"]
    if "CPIAUCSL" in macro:
        cpi = macro["CPIAUCSL"].dropna()
        frame = frame.join(((cpi / cpi.shift(12)) - 1.0).rename("CPI_YOY"), how="left")
    if "WALCL" in frame:
        frame["WALCL_YOY"] = frame["WALCL"] / frame["WALCL"].shift(252) - 1.0
        frame = frame.drop(columns=["WALCL"])
    frame = frame.ffill(limit=21)
    momentum = momentum.ffill(limit=5)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(how="all"), momentum.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _price_based_rows(history: pd.DataFrame | None, as_of: pd.Timestamp) -> list[dict[str, Any]]:
    if history is None or history.empty:
        return []
    as_of_index = history.index[(history.index <= as_of) & (pd.DatetimeIndex(history.index).weekday < 5)]
    current_date = pd.Timestamp(as_of_index[-1]) if len(as_of_index) else as_of
    current = history.loc[current_date] if len(as_of_index) else pd.Series(dtype=float)
    rows = []
    for col in history.columns:
        value = float(current.get(col, np.nan))
        hist = history[col].dropna()
        percentile = _percentile(hist, value)
        rows.append(
            {
                "metric": col,
                "value": value,
                "percentile": percentile,
                "z_score": _zscore(hist, value),
                "as_of": current_date.date().isoformat(),
                "source": _source_for_metric(col),
                "status": "available" if np.isfinite(value) else "missing",
                "extreme_flag": _extreme_flag(percentile),
            }
        )
    return rows


def _sentiment_rows_from_snapshot(
    snapshot: dict[str, Any],
    as_of: pd.Timestamp,
    sources: dict[str, str],
    history: pd.DataFrame | None,
    price_metrics: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for metric in CORE_SENTIMENT_METRICS + OPTIONAL_SENTIMENT_METRICS:
        if metric in price_metrics:
            continue
        value = _maybe_float(snapshot.get(metric))
        source = sources.get(metric, "missing; auto-fetch failed or manual value empty")
        hist = history[metric].dropna() if history is not None and metric in history else pd.Series(dtype=float)
        percentile = _maybe_float(snapshot.get(f"{metric}_percentile"))
        if not np.isfinite(percentile) and metric.endswith("_percentile") and np.isfinite(value):
            percentile = value
        if not np.isfinite(percentile) and not hist.empty and np.isfinite(value):
            percentile = _percentile(hist, value)
        z_score = _maybe_float(snapshot.get(f"{metric}_z_score"))
        if not np.isfinite(z_score) and np.isfinite(percentile):
            z_score = _percentile_to_z(percentile)
        if not np.isfinite(z_score) and not hist.empty and np.isfinite(value):
            z_score = _zscore(hist, value)
        date_key = DATE_KEYS.get(metric)
        metric_date = snapshot.get(date_key) if date_key else None
        rows.append(
            {
                "metric": metric,
                "value": value,
                "percentile": percentile,
                "z_score": z_score,
                "as_of": str(metric_date or as_of.date().isoformat()),
                "source": source,
                "status": "available" if np.isfinite(value) else "missing",
                "extreme_flag": _extreme_flag(percentile),
            }
        )
    return rows


def _read_manual_flat(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if "metrics" in payload and isinstance(payload["metrics"], list):
        # Backward-compatible support for the old list-of-metrics format.
        out = {}
        for item in payload["metrics"]:
            metric = item.get("metric")
            if not metric:
                continue
            out[str(metric)] = item.get("value")
            if item.get("percentile") is not None:
                out[f"{metric}_percentile"] = item.get("percentile")
            if item.get("z_score") is not None:
                out[f"{metric}_z_score"] = item.get("z_score")
            if item.get("as_of") is not None:
                out[DATE_KEYS.get(str(metric), f"{metric}_date")] = item.get("as_of")
        return out
    return dict(payload)


def _read_manual(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    payload = _read_manual_flat(path)
    rows = []
    for metric in CORE_SENTIMENT_METRICS + OPTIONAL_SENTIMENT_METRICS:
        value = payload.get(metric)
        finite_value = value is not None and np.isfinite(float(value))
        percentile = payload.get(f"{metric}_percentile")
        z_val = _maybe_float(payload.get(f"{metric}_z_score"))
        rows.append(
            {
                "metric": metric,
                "value": float(value) if finite_value else np.nan,
                "percentile": float(percentile) if percentile is not None else np.nan,
                "z_score": z_val,
                "as_of": str(payload.get(DATE_KEYS.get(metric, ""), "")),
                "source": "manual_sentiment.yaml",
                "status": "available" if finite_value else "missing",
                "extreme_flag": _extreme_flag(float(percentile)) if percentile is not None else "",
            }
        )
    return rows


def _percentile(history: pd.Series, value: float) -> float:
    if history.empty or not np.isfinite(value):
        return float("nan")
    return float((history <= value).mean() * 100.0)


def _zscore(history: pd.Series, value: float) -> float:
    if history.empty or not np.isfinite(value) or history.std(ddof=1) == 0:
        return float("nan")
    return float((value - history.mean()) / history.std(ddof=1))


def _extreme_flag(percentile: float) -> str:
    if not np.isfinite(percentile):
        return ""
    if percentile >= 90:
        return ">=90分位: 极端偏高"
    if percentile <= 10:
        return "<=10分位: 极端偏低"
    return ""


def _source_for_metric(metric: str) -> str:
    if metric.startswith("VIX"):
        return "yfinance CBOE volatility indices"
    if metric.startswith("HY") or metric.startswith("IG"):
        return "FRED credit spread / ETF proxy"
    return "yfinance market proxy"


def _maybe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        if isinstance(value, str) and not value.strip():
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_value_metric(key: str) -> bool:
    return key in set(CORE_SENTIMENT_METRICS + OPTIONAL_SENTIMENT_METRICS)


def _percentile_to_z(percentile: float) -> float:
    if not np.isfinite(percentile):
        return np.nan
    p = min(max(float(percentile) / 100.0, 0.001), 0.999)
    return float(stats.norm.ppf(p))
