from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nvda_quant_model.event_overlay.event_classifier import (
    classify_event_type,
    event_importance as infer_event_importance,
    nvda_relevance as infer_nvda_relevance,
    source_quality as infer_source_quality,
)
from nvda_quant_model.event_overlay.labels import build_event_labels
from nvda_quant_model.event_overlay.schema import (
    LABEL_COLUMNS,
    LEAKAGE_EXACT_FIELDS,
    LEAKAGE_PREFIXES,
)
from nvda_quant_model.event_overlay.validation import walk_forward_event_overlay_validation


LOGGER = logging.getLogger(__name__)

NVDA_CIK = "0001045810"
DEFAULT_USER_AGENT = "nvda-quant-model/1.0 research-contact@example.com"
MIN_SERIOUS_EVENT_COUNT = 250
MIN_SERIOUS_MONTHS = 24
MAX_SINGLE_SOURCE_SHARE = 0.75

HISTORICAL_EVENT_COLUMNS = (
    "event_id",
    "published_at_utc",
    "available_at_utc",
    "availability_basis",
    "cluster_id",
    "source",
    "url",
    "headline",
    "body_excerpt",
    "ticker_scope",
    "event_type",
    "source_quality",
    "nvda_relevance",
    "event_importance",
    "ingested_at_utc",
    "canonical_hash",
)

ARTICLE_COMPAT_COLUMNS = (
    "id",
    "published_at",
    "source",
    "title",
    "summary",
    "link",
)

EXTRA_HISTORY_LEAKAGE_EXACT = {
    *LEAKAGE_EXACT_FIELDS,
    *LABEL_COLUMNS,
    "label",
    "labels",
    "target",
    "target_return",
    "target_direction",
    "return_after_event",
    "post_event_return",
    "realized_return",
    "realized_direction",
}


@dataclass(frozen=True)
class NewsHistoryBuildResult:
    events: pd.DataFrame
    audit: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def assert_no_historical_event_leakage(columns: Iterable[str]) -> None:
    """Reject fields that are labels or future/post-event observations.

    Historical news events are model inputs. Future returns, output windows,
    labels, and any post-event technical indicators belong only in a separate
    labels table and must never travel with the event input store.
    """

    leaks: list[str] = []
    for column in columns:
        lowered = str(column).strip().lower()
        if lowered in EXTRA_HISTORY_LEAKAGE_EXACT or any(lowered.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            leaks.append(str(column))
        elif "future" in lowered or "post_event" in lowered or "after_event" in lowered:
            leaks.append(str(column))
    if leaks:
        raise ValueError("Historical event store contains leakage fields: " + ", ".join(sorted(set(leaks))))


def normalize_historical_event_frame(
    raw_events: pd.DataFrame,
    *,
    ticker: str = "NVDA",
    ingested_at: pd.Timestamp | str | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Normalize raw point-in-time news/events into the historical store schema."""

    if raw_events is None or raw_events.empty:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    if strict:
        assert_no_historical_event_leakage(raw_events.columns)

    ingested = _to_utc(ingested_at or pd.Timestamp.utcnow())
    rows: list[dict[str, Any]] = []
    for idx, row in raw_events.iterrows():
        published = _first_valid_timestamp(
            row,
            ("published_at_utc", "published_at", "published", "datetime", "date", "filingDate", "acceptanceDateTime"),
        )
        if pd.isna(published):
            continue
        source = _clean_text(row.get("source", row.get("domain", row.get("provider", "unknown")))) or "unknown"
        url = _clean_text(row.get("url", row.get("link", "")))
        headline = _clean_text(row.get("headline", row.get("title", row.get("name", ""))))
        body = _clean_text(row.get("body_excerpt", row.get("summary", row.get("body", row.get("content", "")))))
        if not headline and not body:
            continue
        ticker_scope = _clean_ticker_scope(row.get("ticker_scope", row.get("ticker", ticker)))
        text = f"{headline} {body}".strip()
        source_quality = _bounded_float(row.get("source_quality"), infer_source_quality(source))
        relevance = _bounded_float(row.get("nvda_relevance"), infer_nvda_relevance(text, ticker_scope))
        event_type = _clean_text(row.get("event_type", "")) or classify_event_type(text)
        importance = _bounded_float(
            row.get("event_importance"),
            infer_event_importance(text, event_type, relevance, source_quality),
        )
        available = _first_valid_timestamp(row, ("available_at_utc", "available_at"))
        if pd.notna(available) and available < published:
            raise ValueError("Historical available_at cannot precede published_at")
        original_ingested = _first_valid_timestamp(row, ("ingested_at_utc", "ingested_at"))
        canonical_hash = _canonical_hash(source, url, headline, published)
        event_id = _clean_text(row.get("event_id", row.get("id", ""))) or f"nvda_event_{canonical_hash[:16]}"
        rows.append(
            {
                "event_id": event_id,
                "published_at_utc": published,
                "available_at_utc": available,
                "availability_basis": _clean_text(row.get("availability_basis", "supplied_unverified" if pd.notna(available) else "unknown")),
                "cluster_id": _clean_text(row.get("cluster_id", "")) or hashlib.sha256((str(published.date()) + "|" + headline.lower()).encode()).hexdigest()[:20],
                "source": source,
                "url": url,
                "headline": headline,
                "body_excerpt": body[:1000],
                "ticker_scope": ticker_scope,
                "event_type": event_type,
                "source_quality": source_quality,
                "nvda_relevance": relevance,
                "event_importance": importance,
                "ingested_at_utc": original_ingested if pd.notna(original_ingested) else ingested,
                "canonical_hash": canonical_hash,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))

    frame = pd.DataFrame(rows)
    frame["published_at_utc"] = pd.to_datetime(frame["published_at_utc"], utc=True)
    frame["available_at_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True)
    frame["ingested_at_utc"] = pd.to_datetime(frame["ingested_at_utc"], utc=True)
    frame = frame.drop_duplicates("canonical_hash", keep="first")
    frame = frame.sort_values(["published_at_utc", "source", "headline"], kind="mergesort").reset_index(drop=True)
    return frame.loc[:, list(HISTORICAL_EVENT_COLUMNS)]


def to_event_overlay_frame(history_events: pd.DataFrame) -> pd.DataFrame:
    """Convert the historical store into the existing event overlay schema."""

    history = normalize_historical_event_frame(history_events, strict=False)
    if history.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    return pd.DataFrame(
        {
            "event_id": history["event_id"],
            "published_at": pd.to_datetime(history["published_at_utc"], utc=True),
            "available_at": pd.to_datetime(history["available_at_utc"], utc=True),
            "availability_basis": history["availability_basis"],
            "cluster_id": history["cluster_id"],
            "source": history["source"],
            "headline": history["headline"],
            "body": history["body_excerpt"],
            "ticker_scope": history["ticker_scope"],
            "event_type": history["event_type"],
            "source_quality": history["source_quality"],
            "nvda_relevance": history["nvda_relevance"],
            "event_importance": history["event_importance"],
        }
    )


def to_legacy_news_article_frame(history_events: pd.DataFrame) -> pd.DataFrame:
    """Export event rows in the article CSV shape accepted by --news-history-csv."""

    history = normalize_historical_event_frame(history_events, strict=False)
    if history.empty:
        return pd.DataFrame(columns=list(ARTICLE_COMPAT_COLUMNS))
    return pd.DataFrame(
        {
            "id": history["event_id"],
            "published_at": pd.to_datetime(history["published_at_utc"], utc=True),
            "available_at": pd.to_datetime(history["available_at_utc"], utc=True),
            "availability_basis": history["availability_basis"],
            "cluster_id": history["cluster_id"],
            "source": history["source"],
            "title": history["headline"],
            "summary": history["body_excerpt"],
            "link": history["url"],
        }
    )


def build_historical_event_labels(
    history_events: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Build a separate future-label table for event-model training/evaluation."""

    overlay_events = to_event_overlay_frame(history_events)
    labels = build_event_labels(overlay_events, market_data, horizons=horizons)
    metadata = normalize_historical_event_frame(history_events, strict=False)[["event_id", "published_at_utc", "source", "event_type"]]
    return metadata.merge(labels, on="event_id", how="left")


def audit_event_store_for_backtest(
    history_events: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    min_events: int = MIN_SERIOUS_EVENT_COUNT,
    min_months: int = MIN_SERIOUS_MONTHS,
    max_single_source_share: float = MAX_SINGLE_SOURCE_SHARE,
) -> dict[str, Any]:
    """Report whether the historical event store is large/diverse enough."""

    history = normalize_historical_event_frame(history_events, strict=False)
    if history.empty:
        return {
            "status": "insufficient_point_in_time_sample",
            "total_events": 0,
            "labeled_events": 0,
            "coverage_months": 0,
            "max_single_source_share": 0.0,
            "serious_backtest_ready": False,
            "reasons": ["no_point_in_time_events"],
        }

    month_count = int(history["published_at_utc"].dt.tz_convert(None).dt.to_period("M").nunique())
    source_share = float(history["source"].value_counts(normalize=True).max()) if not history.empty else 0.0
    labeled_events = _labeled_event_count(labels) if labels is not None else int(len(history))
    reasons = []
    if history["available_at_utc"].isna().any():
        reasons.append("historical_available_at_missing")
    if not history["availability_basis"].isin(["verified_ingestion_log", "verified_vendor_release"]).all():
        reasons.append("historical_availability_provenance_unverified")
    if labeled_events < min_events:
        reasons.append(f"labeled_events<{min_events}")
    if month_count < min_months:
        reasons.append(f"coverage_months<{min_months}")
    if source_share > max_single_source_share:
        reasons.append(f"single_source_share>{max_single_source_share:.2f}")
    ready = not reasons
    return {
        "status": "ready_for_walk_forward" if ready else "insufficient_point_in_time_sample",
        "total_events": int(len(history)),
        "labeled_events": int(labeled_events),
        "coverage_months": month_count,
        "start": history["published_at_utc"].min().isoformat(),
        "end": history["published_at_utc"].max().isoformat(),
        "source_counts": {str(k): int(v) for k, v in history["source"].value_counts().to_dict().items()},
        "max_single_source_share": source_share,
        "serious_backtest_ready": ready,
        "reasons": reasons,
    }


def run_historical_event_overlay_backtest(
    history_events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    *,
    min_train_size: int = 120,
    test_size: int = 30,
    step: int = 30,
    target_horizon: int = 1,
    min_events: int = MIN_SERIOUS_EVENT_COUNT,
) -> dict[str, Any]:
    """Run event-overlay validation only when the event store passes sample gates."""

    labels = build_historical_event_labels(history_events, market_data, horizons=(1, 3, 5))
    audit = audit_event_store_for_backtest(history_events, labels, min_events=min_events)
    if not audit["serious_backtest_ready"]:
        return {
            "status": "insufficient_point_in_time_sample",
            "audit": audit,
            "summary": pd.DataFrame(),
            "windows": pd.DataFrame(),
            "group_summary": pd.DataFrame(),
            "principle": "Event overlay adjusts distribution, not trading decision.",
        }
    validation = walk_forward_event_overlay_validation(
        to_event_overlay_frame(history_events),
        market_data,
        external_data,
        min_train_size=min_train_size,
        test_size=test_size,
        step=step,
        target_horizon=target_horizon,
    )
    return {"status": "validated", "audit": audit, **validation}


def load_vendor_event_csv(path: str | Path, *, ticker: str = "NVDA") -> pd.DataFrame:
    """Load a vendor-supplied point-in-time event CSV with strict leakage checks."""

    data = pd.read_csv(path)
    return normalize_historical_event_frame(data, ticker=ticker, strict=True)


def fetch_sec_filing_events(
    *,
    cik: str = NVDA_CIK,
    start: str | pd.Timestamp = "2019-01-01",
    end: str | pd.Timestamp | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch SEC EDGAR submission events for NVIDIA.

    SEC filings are point-in-time company events. They are not a replacement
    for a full historical news vendor, but they provide clean dated events.
    """

    cik_digits = str(cik).lstrip("0")
    payload = _read_json_url(
        f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
        user_agent=user_agent,
        timeout=timeout,
    )
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    frame = pd.DataFrame(recent)
    if frame.empty:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    keep_forms = {"8-K", "10-Q", "10-K"}
    if "form" in frame:
        frame = frame.loc[frame["form"].isin(keep_forms)].copy()
    rows = []
    for _, row in frame.iterrows():
        accession = str(row.get("accessionNumber", "")).strip()
        primary_doc = str(row.get("primaryDocument", "")).strip()
        accession_path = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_digits}/{accession_path}/{primary_doc}"
            if accession and primary_doc
            else ""
        )
        form = str(row.get("form", "")).strip()
        headline = f"NVIDIA SEC {form} filing"
        if form in {"10-Q", "10-K"}:
            headline = f"NVIDIA {form} quarterly/annual earnings filing"
        rows.append(
            {
                "event_id": f"sec_{accession_path or hashlib.sha1(str(row.to_dict()).encode()).hexdigest()[:12]}",
                "published_at_utc": row.get("acceptanceDateTime") or row.get("filingDate"),
                "source": "SEC EDGAR",
                "url": url,
                "headline": headline,
                "body_excerpt": f"Form {form} filed by NVIDIA. Accession {accession}.",
                "ticker_scope": "NVDA",
                "event_type": "earnings" if form in {"10-Q", "10-K"} else "legal_regulatory",
            }
        )
    out = normalize_historical_event_frame(pd.DataFrame(rows), strict=False)
    return _filter_date_range(out, start, end)


def fetch_gdelt_doc_events(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    query: str = '(Nvidia OR NVDA OR "NVIDIA Corp")',
    max_records_per_window: int = 250,
    window_days: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 30,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    """Fetch recent GDELT DOC article metadata for NVDA.

    GDELT's DOC endpoint is useful as a free recent-news source, but coverage
    varies by date and should be audited before any backtest is trusted.
    """

    start_ts = _to_utc(start)
    end_ts = _to_utc(end or pd.Timestamp.utcnow())
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts >= end_ts:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    chunks: list[pd.DataFrame] = []
    failed_windows = 0
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=window_days), end_ts)
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max_records_per_window),
            "sort": "hybridrel",
            "startdatetime": cursor.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
        }
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
        try:
            payload = _read_json_url(url, user_agent=user_agent, timeout=timeout)
        except RuntimeError as exc:  # pragma: no cover - network dependent
            failed_windows += 1
            LOGGER.warning("GDELT fetch failed for %s to %s: %s", cursor, chunk_end, exc)
            cursor = chunk_end
            continue
        articles = payload.get("articles", [])
        rows = [
            {
                "published_at_utc": item.get("seendate") or item.get("date"),
                "source": item.get("sourceCommonName") or item.get("domain") or "GDELT",
                "url": item.get("url", ""),
                "headline": item.get("title", ""),
                "body_excerpt": item.get("title", ""),
                "ticker_scope": "NVDA",
            }
            for item in articles
            if isinstance(item, dict)
        ]
        if rows:
            chunks.append(normalize_historical_event_frame(pd.DataFrame(rows), strict=False))
        cursor = chunk_end
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    if not chunks and failed_windows:
        raise RuntimeError(f"GDELT fetch failed for all {failed_windows} requested windows")
    if not chunks:
        return pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    return _filter_date_range(pd.concat(chunks, ignore_index=True), start_ts, end_ts)


def build_historical_event_store(
    *,
    start: str | pd.Timestamp = "2019-01-01",
    end: str | pd.Timestamp | None = None,
    sources: tuple[str, ...] = ("sec", "gdelt"),
    vendor_csvs: tuple[str | Path, ...] = (),
    ticker: str = "NVDA",
    user_agent: str = DEFAULT_USER_AGENT,
) -> NewsHistoryBuildResult:
    """Build a point-in-time historical event store from configured sources."""

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    source_set = {item.strip().lower() for item in sources}
    if "sec" in source_set:
        try:
            frames.append(fetch_sec_filing_events(start=start, end=end, user_agent=user_agent))
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"sec_fetch_failed:{exc}")
    if "gdelt" in source_set:
        try:
            frames.append(fetch_gdelt_doc_events(start=start, end=end, user_agent=user_agent))
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"gdelt_fetch_failed:{exc}")
    for path in vendor_csvs:
        try:
            frames.append(load_vendor_event_csv(path, ticker=ticker))
        except Exception as exc:
            warnings.append(f"vendor_csv_failed:{Path(path).name}:{exc}")
    combined = (
        normalize_historical_event_frame(pd.concat(frames, ignore_index=True), ticker=ticker, strict=False)
        if frames
        else pd.DataFrame(columns=list(HISTORICAL_EVENT_COLUMNS))
    )
    audit = audit_event_store_for_backtest(combined)
    audit["source_warnings"] = warnings
    return NewsHistoryBuildResult(events=combined, audit=audit, warnings=warnings)


def save_event_store_outputs(
    history_events: pd.DataFrame,
    output_dir: str | Path,
    *,
    market_data: pd.DataFrame | None = None,
) -> dict[str, str | None]:
    """Write non-secret event-store artifacts for local analysis."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = normalize_historical_event_frame(history_events, strict=False)
    events_path = out_dir / "nvda_historical_events.csv"
    article_path = out_dir / "nvda_news_history_articles.csv"
    audit_path = out_dir / "event_store_audit.json"
    history.to_csv(events_path, index=False)
    to_legacy_news_article_frame(history).to_csv(article_path, index=False)
    labels_path: Path | None = None
    audit = audit_event_store_for_backtest(history)
    if market_data is not None:
        labels = build_historical_event_labels(history, market_data)
        labels_path = out_dir / "nvda_event_labels.csv"
        labels.to_csv(labels_path, index=False)
        audit = audit_event_store_for_backtest(history, labels)
    audit_path.write_text(json.dumps(_json_safe(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "events_csv": str(events_path),
        "legacy_articles_csv": str(article_path),
        "labels_csv": str(labels_path) if labels_path else None,
        "audit_json": str(audit_path),
    }


def _read_json_url(url: str, *, user_agent: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(str(exc)) from exc


def _filter_date_range(
    events: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if events.empty:
        return events
    start_ts = _to_utc(start)
    end_ts = _to_utc(end or pd.Timestamp.utcnow())
    mask = (events["published_at_utc"] >= start_ts) & (events["published_at_utc"] <= end_ts)
    return events.loc[mask].reset_index(drop=True)


def _to_utc(value: Any) -> pd.Timestamp:
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 14 and stripped.isdigit():
            value = pd.to_datetime(stripped, format="%Y%m%d%H%M%S", utc=True)
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _first_valid_timestamp(row: pd.Series, columns: tuple[str, ...]) -> pd.Timestamp:
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            try:
                return _to_utc(row[column])
            except (ValueError, TypeError):
                continue
    return pd.NaT


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _clean_ticker_scope(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "NVDA").replace("|", ",").replace(";", ",").split(",")
    tickers = sorted({str(item).upper().strip() for item in values if str(item).strip()})
    return ",".join(tickers or ["NVDA"])


def _bounded_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not np.isfinite(parsed):
        parsed = float(default)
    return float(np.clip(parsed, 0.0, 1.0))


def _canonical_hash(source: str, url: str, headline: str, published_at: pd.Timestamp) -> str:
    normalized_url = url.strip().lower()
    normalized_title = " ".join(headline.lower().split())
    ts = _to_utc(published_at).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = "|".join([source.strip().lower(), normalized_url, normalized_title, ts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _labeled_event_count(labels: pd.DataFrame | None) -> int:
    if labels is None or labels.empty:
        return 0
    direction_cols = [col for col in labels.columns if col.startswith("future_") and col.endswith("_direction")]
    if not direction_cols:
        return int(len(labels))
    return int((labels[direction_cols[0]].astype(str) != "unknown").sum())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
