from __future__ import annotations

import numpy as np
import pandas as pd
from nvda_quant_model.config import RAW_PRICE_COLUMNS
from nvda_quant_model.data.session_calendar import as_utc, session_dates, session_labels


def validate_ohlcv(prices: pd.DataFrame, *, strict_calendar: bool = False) -> pd.DataFrame:
    """Validate daily bars without fabricating prices or silently dropping defects.

    Calendar gaps/non-sessions are reported in attrs, or rejected in strict mode.
    Adjustments must have been applied consistently to all OHLC columns upstream.
    """
    missing = [col for col in RAW_PRICE_COLUMNS if col not in prices.columns]
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")
    out = prices.copy()
    out.index = session_labels(out.index)
    if out.index.hasnans or out.index.has_duplicates:
        raise ValueError("OHLCV session index contains missing or duplicate dates")
    out = out.sort_index()
    numeric = out[RAW_PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("OHLCV contains missing, nonnumeric, or infinite values")
    bad = (numeric[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) | (numeric["Volume"] < 0)
    bad |= (numeric["High"] < numeric[["Open", "Low", "Close"]].max(axis=1))
    bad |= (numeric["Low"] > numeric[["Open", "High", "Close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"OHLCV contains invalid prices/ranges/volume on {list(out.index[bad].astype(str)[:5])}")
    out[RAW_PRICE_COLUMNS] = numeric
    expected = session_dates(out.index.min(), out.index.max()) if len(out) else out.index
    report = {"missing_sessions": list(expected.difference(out.index).strftime("%Y-%m-%d")),
              "non_sessions": list(out.index.difference(expected).strftime("%Y-%m-%d")),
              "adjustment_provenance": prices.attrs.get("adjustment_provenance", "unknown; caller must verify consistent OHLC")}
    if strict_calendar and (report["missing_sessions"] or report["non_sessions"]):
        raise ValueError(f"OHLCV calendar mismatch: {report}")
    out.attrs["data_quality"] = report
    return out


def clean_model_frame(frame: pd.DataFrame, feature_columns: list[str], require_target: bool = False) -> pd.DataFrame:
    """Structural cleaning only. Statistical transforms belong inside each fit."""
    if frame.index.has_duplicates:
        raise ValueError("Model frame has duplicate row indices")
    cleaned = frame.sort_index().copy()
    cleaned[feature_columns] = cleaned[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if require_target:
        cleaned = cleaned.dropna(subset=["target_return", "target_direction"])
    return cleaned


class TrainingPreprocessor:
    """Fitted winsor limits and medians, never recomputed during transform.

    An entirely missing training column maps to zero, even if observed in test.
    No cross-row filling ensures batch/incremental transforms are identical.
    """
    def __init__(self, feature_columns: list[str], limits: tuple[float, float] = (0.01, 0.99)):
        if not 0 <= limits[0] <= limits[1] <= 1:
            raise ValueError("Invalid winsorization limits")
        self.feature_columns = list(feature_columns)
        self.limits = limits

    def fit(self, frame: pd.DataFrame, *, fit_at=None) -> "TrainingPreprocessor":
        if frame.empty:
            raise ValueError("Cannot fit preprocessing on empty training data")
        data = clean_model_frame(frame, self.feature_columns)[self.feature_columns]
        self.lower_ = data.quantile(self.limits[0]).fillna(0.0)
        self.upper_ = data.quantile(self.limits[1]).fillna(0.0)
        self.medians_ = data.clip(self.lower_, self.upper_, axis=1).median().fillna(0.0)
        self.fit_at_ = as_utc(fit_at) if fit_at is not None else None
        if fit_at is not None and "available_at" in frame:
            available = pd.to_datetime(frame["available_at"], utc=True)
            if available.isna().any() or (available > self.fit_at_).any():
                raise ValueError("Training features are unavailable at fit_at")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise ValueError("TrainingPreprocessor must be fitted first")
        out = clean_model_frame(frame, self.feature_columns)
        out[self.feature_columns] = out[self.feature_columns].clip(self.lower_, self.upper_, axis=1).fillna(self.medians_)
        return out

    def fit_transform(self, frame: pd.DataFrame, *, fit_at=None) -> pd.DataFrame:
        return self.fit(frame, fit_at=fit_at).transform(frame)


def winsorize_features(frame: pd.DataFrame, feature_columns: list[str], limits: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Training-only compatibility helper. For evaluation use a saved transformer."""
    fitted = TrainingPreprocessor(feature_columns, limits).fit(frame)
    out = frame.copy()
    out[feature_columns] = out[feature_columns].clip(fitted.lower_, fitted.upper_, axis=1)
    return out


def purge_immature_labels(frame: pd.DataFrame, fit_at) -> pd.DataFrame:
    """Fail closed: missing maturity/availability can never enter model training."""
    for col in ("label_end_at", "available_at"):
        if col not in frame:
            raise ValueError(f"Training requires explicit {col}")
    cutoff = as_utc(fit_at)
    end = pd.to_datetime(frame["label_end_at"], utc=True, errors="coerce")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    result = frame.loc[end.notna() & available.notna() & (end <= cutoff) & (available <= cutoff)].copy()
    result["fit_at"] = cutoff
    return result
