from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_market_data, load_peer_ohlcv_panel, resolve_data_window, warmup_start
from nvda_quant_model.news_sentiment import load_news_feature_cache


@dataclass(frozen=True)
class ModelInputs:
    config: StrategyConfig
    prices: pd.DataFrame
    external: pd.DataFrame
    peer_ohlcv: dict[str, pd.DataFrame]
    frame: pd.DataFrame
    feature_columns: list[str]
    trainable: pd.DataFrame
    report_prices: pd.DataFrame
    data_status: dict[str, Any]
    date_metadata: dict[str, Any]


def build_data_status(
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
        "availability_warnings": ["Historical macro/fundamental release and revision times are unknown; cached values are not verified point-in-time."],
        "availability_provenance": {"prices": "session_close_assumption", "macro": "unknown", "fundamentals": "unknown"},
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


def prepare_model_inputs(
    config: StrategyConfig,
    price_override: float | None = None,
    force_refresh: bool = False,
    news_history_path: Path | None = None,
    cache_dir: Path | None = None,
) -> ModelInputs:
    cache_dir = cache_dir or PROJECT_ROOT / "cache"
    resolved_start, resolved_end, date_metadata = resolve_data_window(
        config.start_date,
        config.end_date,
        ticker=config.ticker,
        lookback_months=config.lookback_months,
    )
    config = replace(config, start_date=resolved_start, end_date=resolved_end)
    latest_end = date_metadata.get("end_source") == "latest_yfinance_daily"

    prices, external = load_market_data(
        config.ticker,
        config.start_date,
        config.end_date,
        cache_dir=cache_dir,
        force_refresh=force_refresh or latest_end,
    )
    peer_ohlcv = (
        load_peer_ohlcv_panel(
            warmup_start(config.start_date),
            config.end_date,
            cache_dir=cache_dir,
            force_refresh=force_refresh or latest_end,
        )
        if config.include_peer_events
        else {}
    )
    news_features = load_news_feature_cache(news_history_path, prices.index) if news_history_path is not None else None
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
    report_prices = prices.loc[
        (prices.index >= pd.Timestamp(config.start_date)) & (prices.index <= pd.Timestamp(config.end_date))
    ]
    return ModelInputs(
        config=config,
        prices=prices,
        external=external,
        peer_ohlcv=peer_ohlcv,
        frame=frame,
        feature_columns=feature_columns,
        trainable=frame.dropna(subset=["target_return", "target_direction"]),
        report_prices=report_prices,
        data_status=build_data_status(config, prices, external, date_metadata, price_override),
        date_metadata=date_metadata,
    )
