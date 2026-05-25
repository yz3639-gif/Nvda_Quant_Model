from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nvda_quant_model.event_overlay.schema import EVENT_TYPES, normalize_event_frame


SOURCE_QUALITY = {
    "reuters": 0.95,
    "bloomberg": 0.95,
    "wall street journal": 0.93,
    "the wall street journal": 0.93,
    "nvidia": 0.92,
    "sec": 0.92,
    "cnbc": 0.86,
    "marketwatch": 0.78,
    "barron's": 0.80,
    "yahoo finance": 0.72,
    "zacks": 0.68,
    "seeking alpha": 0.62,
}

EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("earnings", "eps", "quarter", "revenue", "margin", "results"),
    "guidance": ("guidance", "forecast", "outlook", "raised outlook", "lowered outlook", "guide"),
    "analyst": ("upgrade", "downgrade", "price target", "rating", "initiated", "outperform", "underperform"),
    "product": ("blackwell", "gpu", "h100", "h200", "b200", "cuda", "launch", "chip", "server"),
    "supply_chain": ("tsmc", "asml", "coWoS", "supply", "shipment", "lead time", "capacity", "supplier"),
    "export_control": ("export control", "china", "restriction", "ban", "license", "commerce department"),
    "competition": ("amd", "broadcom", "intel", "asic", "competitive", "competition"),
    "customer_capex": ("capex", "capital expenditure", "microsoft", "meta", "amazon", "google", "oracle", "data center spend"),
    "valuation": ("valuation", "multiple", "p/e", "price-to-sales", "expensive", "bubble"),
    "macro_ai_trade": ("ai trade", "nasdaq", "rates", "treasury", "risk appetite", "semiconductor rally"),
    "legal_regulatory": ("lawsuit", "probe", "investigation", "regulator", "antitrust", "subpoena"),
}

POSITIVE_TERMS = (
    "beat",
    "beats",
    "raise",
    "raises",
    "upgrade",
    "strong demand",
    "record",
    "accelerating",
    "wins",
    "outperform",
    "price target raised",
)

NEGATIVE_TERMS = (
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "delay",
    "delayed",
    "restriction",
    "export control",
    "probe",
    "lawsuit",
    "margin pressure",
    "price target cut",
)

NVDA_TERMS = (
    "nvda",
    "nvidia",
    "blackwell",
    "cuda",
    "h100",
    "h200",
    "gb200",
    "gpu",
    "data center",
    "ai chip",
)


def _text(row: pd.Series) -> str:
    return f"{row.get('headline', '')} {row.get('body', '')}".lower()


def source_quality(source: str) -> float:
    lowered = str(source or "").strip().lower()
    for key, quality in SOURCE_QUALITY.items():
        if key in lowered:
            return quality
    return 0.55


def nvda_relevance(text: str, ticker_scope: tuple[str, ...] | list[str] | str = ("NVDA",)) -> float:
    lowered = text.lower()
    scope = {str(item).upper() for item in ticker_scope} if not isinstance(ticker_scope, str) else {ticker_scope.upper()}
    score = 0.25 + (0.35 if "NVDA" in scope else 0.0)
    hits = sum(1 for term in NVDA_TERMS if term in lowered)
    score += min(hits * 0.12, 0.40)
    return float(np.clip(score, 0.0, 1.0))


def classify_event_type(text: str) -> str:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for event_type, terms in EVENT_KEYWORDS.items():
        scores[event_type] = sum(1 for term in terms if term.lower() in lowered)
    best_type, best_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return best_type if best_score > 0 else "neutral"


def keyword_sentiment_score(text: str) -> float:
    lowered = text.lower()
    pos = sum(1 for term in POSITIVE_TERMS if term in lowered)
    neg = sum(1 for term in NEGATIVE_TERMS if term in lowered)
    return float(np.tanh((pos - neg) / 3.0))


def event_importance(text: str, event_type: str, relevance: float, quality: float) -> float:
    lowered = text.lower()
    intensity = abs(keyword_sentiment_score(lowered))
    type_boost = 0.20 if event_type in {"earnings", "guidance", "export_control", "customer_capex"} else 0.10
    long_text_boost = min(len(re.findall(r"\w+", lowered)) / 500.0, 0.20)
    return float(np.clip(0.25 + 0.30 * relevance + 0.20 * quality + 0.20 * intensity + type_boost + long_text_boost, 0.0, 1.0))


def reason_codes_for_event(text: str, event_type: str, relevance: float, quality: float) -> list[str]:
    reasons = [f"event_type:{event_type}"]
    if relevance >= 0.75:
        reasons.append("high_nvda_relevance")
    if quality >= 0.85:
        reasons.append("high_source_quality")
    sentiment = keyword_sentiment_score(text)
    if sentiment >= 0.25:
        reasons.append("positive_event_language")
    elif sentiment <= -0.25:
        reasons.append("negative_event_language")
    return reasons


def classify_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add event type, source quality, relevance, importance, and reason codes."""

    frame = normalize_event_frame(events).copy()
    if frame.empty:
        return frame
    event_types: list[str] = []
    qualities: list[float] = []
    relevances: list[float] = []
    importances: list[float] = []
    reasons: list[list[str]] = []
    for _, row in frame.iterrows():
        text = _text(row)
        inferred_type = row["event_type"] if row["event_type"] in EVENT_TYPES and row["event_type"] != "neutral" else classify_event_type(text)
        quality = source_quality(row["source"]) if row.get("source_quality", 0.5) == 0.5 else float(row["source_quality"])
        relevance = nvda_relevance(text, row["ticker_scope"]) if row.get("nvda_relevance", 0.5) == 0.5 else float(row["nvda_relevance"])
        importance = event_importance(text, inferred_type, relevance, quality)
        event_types.append(inferred_type)
        qualities.append(quality)
        relevances.append(relevance)
        importances.append(importance)
        reasons.append(reason_codes_for_event(text, inferred_type, relevance, quality))
    frame["event_type"] = event_types
    frame["source_quality"] = np.clip(qualities, 0.0, 1.0)
    frame["nvda_relevance"] = np.clip(relevances, 0.0, 1.0)
    frame["event_importance"] = np.clip(importances, 0.0, 1.0)
    frame["reason_codes"] = reasons
    return frame
