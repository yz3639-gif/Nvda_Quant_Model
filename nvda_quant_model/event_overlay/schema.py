from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from nvda_quant_model.data.session_calendar import as_utc


EVENT_TYPES = (
    "earnings",
    "guidance",
    "analyst",
    "product",
    "supply_chain",
    "export_control",
    "competition",
    "customer_capex",
    "valuation",
    "macro_ai_trade",
    "legal_regulatory",
    "neutral",
)

REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "published_at",
    "source",
    "headline",
    "body",
    "ticker_scope",
    "event_type",
    "source_quality",
    "nvda_relevance",
    "event_importance",
)

LABEL_COLUMNS = (
    "future_1d_direction",
    "future_3d_direction",
    "future_5d_direction",
    "volatility_shock",
    "downside_tail_risk",
)

LEAKAGE_EXACT_FIELDS = {
    "label_end_at",
    "fit_at",
    "decision_at",
    "output_window",
    "output_timestamps",
    "future_return",
    "future_returns",
    "future_1d_return",
    "future_3d_return",
    "future_5d_return",
    "future_1d_direction",
    "future_3d_direction",
    "future_5d_direction",
    "volatility_shock",
    "downside_tail_risk",
    "trend",
    "alignment",
    "signal",
    "position",
    "direction",
}

LEAKAGE_PREFIXES = ("out_", "overall_", "output_", "future_")


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    published_at: pd.Timestamp
    source: str
    headline: str
    body: str = ""
    ticker_scope: tuple[str, ...] = ("NVDA",)
    event_type: str = "neutral"
    source_quality: float = 0.5
    nvda_relevance: float = 0.5
    event_importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.headline} {self.body}".strip()


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ticker_scope(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        tickers = re.split(r"[,|;\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        tickers = list(value)
    else:
        tickers = ["NVDA"]
    cleaned = tuple(sorted({str(item).upper().strip() for item in tickers if str(item).strip()}))
    return cleaned or ("NVDA",)


def normalize_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw NVDA news rows into the event overlay schema.

    This function only normalizes fields available at publication time. It does
    not create labels and does not inspect future returns.
    """

    rows: list[dict[str, Any]] = []
    for idx, row in events.iterrows():
        metadata = parse_metadata(row.get("metadata", {}))
        headline = _clean_text(row.get("headline", row.get("title", "")))
        body = _clean_text(row.get("body", row.get("summary", row.get("content", ""))))
        raw_published = pd.Timestamp(row.get("published_at", row.get("date", pd.NaT)))
        published_at = as_utc(raw_published)
        available_at = as_utc(row.get("available_at", published_at))
        if pd.notna(published_at) and pd.notna(available_at) and available_at < published_at:
            raise ValueError("Event available_at cannot precede published_at")
        fingerprint = hashlib.sha256((str(published_at.date()) + "|" + headline.lower()).encode()).hexdigest()[:20]
        cluster = row.get("cluster_id", metadata.get("cluster_id", fingerprint))
        source = _clean_text(row.get("source", "unknown")) or "unknown"
        event_id = _clean_text(row.get("event_id", row.get("id", ""))) or f"event_{idx}"
        rows.append(
            {
                "event_id": event_id,
                "published_at": published_at,
                "available_at": available_at,
                "availability_basis": row.get("availability_basis", "supplied" if "available_at" in row else "publication_time_assumption"),
                "timestamp_basis": row.get("timestamp_basis", "naive_assumed_UTC" if raw_published.tzinfo is None else "timezone_supplied"),
                "cluster_id": str(cluster),
                "source": source,
                "headline": headline,
                "body": body,
                "ticker_scope": _ticker_scope(row.get("ticker_scope", row.get("ticker", "NVDA"))),
                "event_type": _clean_text(row.get("event_type", "neutral")) or "neutral",
                "source_quality": float(row.get("source_quality", 0.5)),
                "nvda_relevance": float(row.get("nvda_relevance", 0.5)),
                "event_importance": float(row.get("event_importance", 0.5)),
                "metadata": metadata,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=list(REQUIRED_EVENT_COLUMNS) + ["metadata"])
    if out["event_id"].duplicated().any():
        raise ValueError("Event IDs must be unique; assign distinct IDs and shared cluster_id to duplicate stories")
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
    out["event_type"] = out["event_type"].where(out["event_type"].isin(EVENT_TYPES), "neutral")
    out["source_quality"] = out["source_quality"].clip(0.0, 1.0)
    out["nvda_relevance"] = out["nvda_relevance"].clip(0.0, 1.0)
    out["event_importance"] = out["event_importance"].clip(0.0, 1.0)
    return out.sort_values(["available_at", "published_at"], kind="mergesort").reset_index(drop=True)


def assert_no_event_feature_leakage(feature_columns: list[str]) -> None:
    leaks: list[str] = []
    for column in feature_columns:
        lowered = str(column).lower()
        if lowered in LEAKAGE_EXACT_FIELDS or any(lowered.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            leaks.append(str(column))
    if leaks:
        raise ValueError("Event overlay feature matrix contains leakage fields: " + ", ".join(sorted(leaks)))


def event_text(frame: pd.DataFrame) -> list[str]:
    return (frame["headline"].fillna("") + " " + frame["body"].fillna("")).str.strip().tolist()
