from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from nvda_quant_model.config import PROJECT_ROOT


DEFAULT_QUERIES = [
    'NVDA OR NVIDIA',
    '"Nvidia" earnings OR revenue OR guidance',
    '"Nvidia" Blackwell OR GPU OR AI chips',
    '"Nvidia" export controls OR China OR regulation',
    '"AMD" OR "Broadcom" OR "TSMC" semiconductor AI chips',
]


POSITIVE_TERMS = {
    "beat": 2.0,
    "beats": 2.0,
    "raise": 1.5,
    "raises": 1.5,
    "upgrade": 1.5,
    "upgraded": 1.5,
    "bullish": 1.2,
    "strong demand": 1.8,
    "record": 1.0,
    "surge": 1.0,
    "jumps": 0.8,
    "rally": 0.8,
    "partnership": 0.8,
    "wins": 1.0,
    "outperform": 1.2,
    "buy rating": 1.2,
    "price target raised": 1.8,
    "growth": 0.6,
    "data center": 0.5,
}


NEGATIVE_TERMS = {
    "miss": -2.0,
    "misses": -2.0,
    "cut": -1.2,
    "cuts": -1.2,
    "downgrade": -1.6,
    "downgraded": -1.6,
    "bearish": -1.2,
    "weak": -1.0,
    "delay": -1.3,
    "delayed": -1.3,
    "probe": -1.4,
    "investigation": -1.4,
    "lawsuit": -1.5,
    "ban": -1.5,
    "export controls": -1.7,
    "restrictions": -1.4,
    "tariff": -1.0,
    "slump": -1.0,
    "falls": -0.8,
    "sell rating": -1.3,
    "price target cut": -1.8,
    "competition": -0.5,
}


EVENT_TERMS = {
    "earnings": ["earnings", "revenue", "guidance", "margin", "eps", "quarter"],
    "analyst": ["upgrade", "downgrade", "price target", "rating", "outperform", "buy rating", "sell rating"],
    "regulatory": ["export controls", "china", "ban", "restriction", "regulation", "probe", "investigation", "tariff"],
    "product": ["blackwell", "gpu", "chip", "data center", "cuda", "ai server", "supply"],
    "competition": ["amd", "broadcom", "avgo", "tsmc", "intel", "asic", "competition"],
}


SOURCE_QUALITY = {
    "Reuters": 1.15,
    "Bloomberg": 1.15,
    "CNBC": 1.05,
    "Wall Street Journal": 1.15,
    "The Wall Street Journal": 1.15,
    "Barron's": 1.05,
    "MarketWatch": 0.95,
    "Yahoo Finance": 0.95,
    "Investing.com": 0.90,
    "Seeking Alpha": 0.85,
}


@dataclass(frozen=True)
class NewsArticle:
    id: str
    title: str
    link: str
    source: str
    published_at: str
    summary: str
    query: str
    sentiment_score: float
    event_scores: dict[str, float]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _normalize_text(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _article_id(title: str, link: str) -> str:
    normalized = re.sub(r"\W+", " ", title.lower()).strip()
    return hashlib.sha1(f"{normalized}|{link}".encode("utf-8")).hexdigest()[:16]


def google_news_rss_url(query: str, days: int = 7) -> str:
    q = f"({query}) when:{days}d"
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {
            "q": q,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
    )


def parse_google_source(item: ET.Element) -> tuple[str, str]:
    source = item.find("source")
    if source is None:
        return "", ""
    return source.text or "", source.attrib.get("url", "")


def parse_published_at(text: str | None) -> pd.Timestamp:
    if not text:
        return pd.Timestamp.utcnow().tz_localize(None)
    try:
        return pd.Timestamp(parsedate_to_datetime(text)).tz_convert(None)
    except Exception:
        return pd.Timestamp.utcnow().tz_localize(None)


def score_text(title: str, summary: str, source: str = "") -> tuple[float, dict[str, float]]:
    text = f"{title} {summary}".lower()
    score = 0.0
    for term, weight in POSITIVE_TERMS.items():
        if term in text:
            score += weight
    for term, weight in NEGATIVE_TERMS.items():
        if term in text:
            score += weight
    quality = SOURCE_QUALITY.get(source, 1.0)
    score *= quality
    score = float(np.tanh(score / 4.0))

    event_scores: dict[str, float] = {}
    for event, terms in EVENT_TERMS.items():
        raw = sum(1.0 for term in terms if term in text)
        event_scores[event] = float(min(raw / 3.0, 1.0))
    event_scores["bullish"] = max(score, 0.0)
    event_scores["bearish"] = max(-score, 0.0)
    return score, event_scores


def fetch_google_news(query: str, days: int = 7, timeout: int = 20) -> list[NewsArticle]:
    url = google_news_rss_url(query, days)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    articles: list[NewsArticle] = []
    for item in root.findall(".//item"):
        title = _normalize_text(item.findtext("title", default=""))
        link = item.findtext("link", default="")
        summary = _normalize_text(item.findtext("description", default=""))
        source, _ = parse_google_source(item)
        published_at = parse_published_at(item.findtext("pubDate"))
        score, event_scores = score_text(title, summary, source)
        articles.append(
            NewsArticle(
                id=_article_id(title, link),
                title=title,
                link=link,
                source=source,
                published_at=published_at.isoformat(),
                summary=summary,
                query=query,
                sentiment_score=score,
                event_scores=event_scores,
            )
        )
    return articles


def fetch_live_news(
    queries: list[str] | None = None,
    days: int = 7,
    sleep_seconds: float = 0.2,
) -> list[NewsArticle]:
    queries = queries or DEFAULT_QUERIES
    seen: set[str] = set()
    articles: list[NewsArticle] = []
    for query in queries:
        try:
            for article in fetch_google_news(query, days=days):
                if article.id in seen:
                    continue
                seen.add(article.id)
                articles.append(article)
        except Exception as exc:
            print(f"news fetch failed for {query!r}: {exc}", flush=True)
        time.sleep(sleep_seconds)
    return sorted(articles, key=lambda item: item.published_at, reverse=True)


def articles_to_frame(articles: list[NewsArticle]) -> pd.DataFrame:
    rows = []
    for article in articles:
        row = asdict(article)
        for event, value in article.event_scores.items():
            row[f"event_{event}"] = value
        row.pop("event_scores", None)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["published_at"] = pd.to_datetime(frame["published_at"]).dt.tz_localize(None)
    return frame.sort_values("published_at")


def build_daily_news_features(
    articles: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    windows: tuple[int, ...] = (1, 3, 7),
) -> pd.DataFrame:
    out = pd.DataFrame(index=price_index)
    if articles.empty:
        for window in windows:
            out[f"news_count_{window}d"] = 0.0
            out[f"news_sentiment_mean_{window}d"] = 0.0
        out["news_recency_score"] = 0.0
        return out

    articles = articles.copy()
    articles["date"] = pd.to_datetime(articles["published_at"]).dt.normalize()
    daily = articles.groupby("date").agg(
        news_count=("id", "count"),
        news_sentiment_sum=("sentiment_score", "sum"),
        news_sentiment_mean=("sentiment_score", "mean"),
        news_bullish=("event_bullish", "sum"),
        news_bearish=("event_bearish", "sum"),
        news_earnings=("event_earnings", "sum"),
        news_analyst=("event_analyst", "sum"),
        news_regulatory=("event_regulatory", "sum"),
        news_product=("event_product", "sum"),
        news_competition=("event_competition", "sum"),
    )
    daily = daily.reindex(price_index.normalize()).fillna(0.0)
    daily.index = price_index
    for window in windows:
        out[f"news_count_{window}d"] = daily["news_count"].rolling(window, min_periods=1).sum()
        out[f"news_sentiment_sum_{window}d"] = daily["news_sentiment_sum"].rolling(window, min_periods=1).sum()
        out[f"news_sentiment_mean_{window}d"] = (
            out[f"news_sentiment_sum_{window}d"] / out[f"news_count_{window}d"].replace(0, np.nan)
        ).fillna(0.0)
        out[f"news_bull_bear_{window}d"] = (
            daily["news_bullish"].rolling(window, min_periods=1).sum()
            - daily["news_bearish"].rolling(window, min_periods=1).sum()
        )
        out[f"news_regulatory_{window}d"] = daily["news_regulatory"].rolling(window, min_periods=1).sum()
        out[f"news_earnings_{window}d"] = daily["news_earnings"].rolling(window, min_periods=1).sum()
        out[f"news_product_{window}d"] = daily["news_product"].rolling(window, min_periods=1).sum()
        out[f"news_competition_{window}d"] = daily["news_competition"].rolling(window, min_periods=1).sum()
    decay = np.array([0.55, 0.30, 0.15])
    sentiment = daily["news_sentiment_sum"]
    out["news_recency_score"] = sum(sentiment.shift(i).fillna(0.0) * decay[i] for i in range(len(decay)))
    out["news_risk_score"] = out["news_regulatory_3d"] + out["news_competition_3d"] + daily["news_bearish"].rolling(3, min_periods=1).sum()
    return out.fillna(0.0)


def live_news_overlay(articles: pd.DataFrame) -> dict[str, Any]:
    if articles.empty:
        return {
            "article_count": 0,
            "sentiment_score": 0.0,
            "signal": 0,
            "confidence_adjustment": 0.0,
            "risk_flags": [],
            "top_articles": [],
        }
    now = pd.Timestamp.utcnow().tz_localize(None)
    scoped = articles.copy()
    age_hours = ((now - scoped["published_at"]).dt.total_seconds() / 3600.0).clip(lower=0.0)
    weights = np.exp(-age_hours / 72.0)
    weighted_sentiment = float(np.average(scoped["sentiment_score"], weights=weights)) if weights.sum() else 0.0
    event_cols = [col for col in scoped.columns if col.startswith("event_")]
    event_totals = {col.replace("event_", ""): float(np.average(scoped[col], weights=weights)) for col in event_cols}
    risk_flags = []
    if event_totals.get("regulatory", 0.0) > 0.20 and weighted_sentiment < 0.05:
        risk_flags.append("regulatory_risk")
    if event_totals.get("competition", 0.0) > 0.25 and weighted_sentiment < 0.05:
        risk_flags.append("competition_pressure")
    if event_totals.get("earnings", 0.0) > 0.25:
        risk_flags.append("earnings_sensitive")
    signal = int(1 if weighted_sentiment >= 0.18 and not risk_flags else -1 if weighted_sentiment <= -0.18 or "regulatory_risk" in risk_flags else 0)
    top = scoped.sort_values("published_at", ascending=False).head(10)
    return {
        "article_count": int(len(scoped)),
        "sentiment_score": weighted_sentiment,
        "signal": signal,
        "confidence_adjustment": float(np.clip(abs(weighted_sentiment) * 0.20, 0.0, 0.08)),
        "event_scores": event_totals,
        "risk_flags": risk_flags,
        "top_articles": top[["published_at", "source", "title", "sentiment_score", "link"]].to_dict("records"),
    }


def save_news_outputs(articles: list[NewsArticle], output_dir: Path, price_index: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = articles_to_frame(articles)
    raw_path = output_dir / "live_news_articles.json"
    csv_path = output_dir / "live_news_articles.csv"
    overlay_path = output_dir / "live_news_overlay.json"
    raw_path.write_text(json.dumps([asdict(article) for article in articles], indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    frame.to_csv(csv_path, index=False)
    overlay = live_news_overlay(frame)
    overlay_path.write_text(json.dumps(overlay, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    feature_path = None
    if price_index is not None:
        features = build_daily_news_features(frame, price_index)
        feature_path = output_dir / "live_news_features.csv"
        features.to_csv(feature_path)
    return {
        "articles_json": str(raw_path),
        "articles_csv": str(csv_path),
        "overlay_json": str(overlay_path),
        "features_csv": str(feature_path) if feature_path else None,
        "overlay": overlay,
    }


def load_news_feature_cache(path: Path, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    data = pd.read_csv(path)
    if {"published_at", "title"}.issubset(data.columns):
        data["published_at"] = pd.to_datetime(data["published_at"]).dt.tz_localize(None)
        if "id" not in data:
            data["id"] = [
                _article_id(str(row.get("title", "")), str(row.get("link", "")))
                for _, row in data.iterrows()
            ]
        if "sentiment_score" not in data:
            scored = data.apply(lambda row: score_text(str(row.get("title", "")), str(row.get("summary", "")), str(row.get("source", ""))), axis=1)
            data["sentiment_score"] = [item[0] for item in scored]
            for event in EVENT_TERMS:
                data[f"event_{event}"] = [item[1][event] for item in scored]
            data["event_bullish"] = [item[1]["bullish"] for item in scored]
            data["event_bearish"] = [item[1]["bearish"] for item in scored]
        return build_daily_news_features(data, price_index)
    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return data.sort_index().reindex(price_index).ffill().fillna(0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live NVDA news and build point-in-time news features")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--queries", default="", help="Pipe-separated custom queries")
    parser.add_argument(
        "--input-json",
        default="",
        help="Optional path to a JSON list of article dicts (offline mode; skips network fetch).",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "news_live"))
    parser.add_argument("--price-index-csv", default="", help="Optional CSV with Date column to align daily features")
    return parser.parse_args()


def _articles_from_input_json(path: Path) -> list[NewsArticle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--input-json must be a JSON list of article objects")
    articles: list[NewsArticle] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        link = str(raw.get("link", "")).strip()
        if not title or not link:
            continue
        source = str(raw.get("source", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        query = str(raw.get("query", "")).strip() or "offline"
        published_at_raw = str(raw.get("published_at", "")).strip()
        try:
            published_at = pd.Timestamp(published_at_raw).tz_localize(None)
        except Exception:
            published_at = pd.Timestamp.utcnow().tz_localize(None)
        score, event_scores = score_text(title, summary, source)
        override_event_scores = raw.get("event_scores")
        articles.append(
            NewsArticle(
                id=str(raw.get("id", "")).strip() or _article_id(title, link),
                title=title,
                link=link,
                source=source,
                published_at=published_at.isoformat(),
                summary=summary,
                query=query,
                sentiment_score=float(raw.get("sentiment_score", score)),
                event_scores=override_event_scores if isinstance(override_event_scores, dict) else event_scores,
            )
        )
    return sorted(articles, key=lambda item: item.published_at, reverse=True)


def main() -> None:
    args = parse_args()
    queries = [q.strip() for q in args.queries.split("|") if q.strip()] if args.queries else DEFAULT_QUERIES
    price_index = None
    if args.price_index_csv:
        price_index = pd.read_csv(args.price_index_csv, parse_dates=["Date"], index_col="Date").index
    if args.input_json:
        articles = _articles_from_input_json(Path(args.input_json))
    else:
        articles = fetch_live_news(queries=queries, days=args.days)
    payload = save_news_outputs(articles, Path(args.output_dir), price_index=price_index)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
