from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.config import RAW_PRICE_COLUMNS


def validate_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_PRICE_COLUMNS if col not in prices.columns]
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices[RAW_PRICE_COLUMNS] = prices[RAW_PRICE_COLUMNS].ffill()
    prices = prices.dropna(subset=["Open", "High", "Low", "Close"])
    prices = prices[prices["Close"] > 0]
    return prices


def winsorize_features(frame: pd.DataFrame, feature_columns: list[str], limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    cleaned = frame.copy()
    for col in feature_columns:
        series = cleaned[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        lo, hi = series.quantile(limits[0]), series.quantile(limits[1])
        cleaned[col] = series.clip(lo, hi)
    return cleaned


def clean_model_frame(frame: pd.DataFrame, feature_columns: list[str], require_target: bool = False) -> pd.DataFrame:
    cleaned = frame.sort_index().replace([np.inf, -np.inf], np.nan)
    cleaned[feature_columns] = cleaned[feature_columns].ffill()
    cleaned = winsorize_features(cleaned, feature_columns)
    cleaned[feature_columns] = cleaned[feature_columns].fillna(cleaned[feature_columns].median(numeric_only=True))
    cleaned[feature_columns] = cleaned[feature_columns].fillna(0.0)
    if require_target:
        cleaned = cleaned.dropna(subset=["target_return", "target_direction"])
    return cleaned
