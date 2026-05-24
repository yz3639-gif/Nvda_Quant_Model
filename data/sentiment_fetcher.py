"""Automatic sentiment and positioning fetchers with short-lived caching.

Each fetcher is intentionally independent: one broken upstream endpoint should
never prevent the rest of the sentiment snapshot from loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json
import logging
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class SentimentFetcher:
    cache_dir: Path
    max_cache_age_hours: int = 6
    timeout: int = 20
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self) -> dict:
        """Return every metric that can be fetched automatically."""

        result = {}
        for name, fetcher in [
            ("aaii", self._fetch_aaii),
            ("naaim", self._fetch_naaim),
            ("cftc_cot", self._fetch_cftc_cot),
            ("finra_margin", self._fetch_finra_margin),
            ("ici_flows", self._fetch_ici_flows),
            ("cboe_pc", self._fetch_cboe_pc),
        ]:
            try:
                data = fetcher()
                if data:
                    result.update(data)
            except Exception as exc:  # pragma: no cover - network failure path
                warning = f"sentiment_fetcher: {name} failed: {exc}"
                self.warnings.append(warning)
                LOGGER.warning(warning)
        return {k: v for k, v in result.items() if v is not None and not _is_nan(v)}

    def _fetch_aaii(self) -> dict:
        cache_name = "aaii_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        try:
            raw = self._read_url(url)
            frame = _read_excel_bytes(raw)
        except Exception:
            page = self._read_url("https://www.aaii.com/sentimentsurvey/sent_results")
            tables = pd.read_html(BytesIO(page))
            frame = max(tables, key=len)
            url = "https://www.aaii.com/sentimentsurvey/sent_results"
        frame = _normalize_columns(frame)
        date_col = _find_col(frame, ["date", "week"]) or _find_col(frame, ["date"])
        bull_col = _find_col(frame, ["bullish"])
        bear_col = _find_col(frame, ["bearish"])
        if not date_col or not bull_col or not bear_col:
            raise ValueError("AAII columns not found")
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_col], errors="coerce"),
                "bullish": _percent_series(frame[bull_col]),
                "bearish": _percent_series(frame[bear_col]),
            }
        ).dropna()
        history["bull_bear_spread"] = history["bullish"] - history["bearish"]
        history = history.sort_values("date")
        latest = history.iloc[-1]
        tail = history["bull_bear_spread"].tail(260)
        data = {
            "AAII_bull_bear_spread": float(latest["bull_bear_spread"]),
            "AAII_date": pd.Timestamp(latest["date"]).date().isoformat(),
            "AAII_bull_bear_spread_percentile": _percentile(tail, latest["bull_bear_spread"]),
            "AAII_bull_bear_spread_z_score": _zscore(tail, latest["bull_bear_spread"]),
        }
        self.histories["AAII_bull_bear_spread"] = history[["date", "bull_bear_spread"]].rename(columns={"bull_bear_spread": "AAII_bull_bear_spread"})
        self._cache_set(cache_name, data, history)
        self.sources["AAII_bull_bear_spread"] = f"auto:{url}"
        return data

    def _fetch_naaim(self) -> dict:
        cache_name = "naaim_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        url = "https://www.naaim.org/wp-content/uploads/2013/05/NAAIM-Exposure-Index.csv"
        try:
            raw = self._read_url(url)
            frame = pd.read_csv(BytesIO(raw))
        except Exception:
            url = "https://www.naaim.org/programs/naaim-exposure-index/"
            raw = self._read_url(url)
            frame = _parse_naaim_google_chart(raw)
        frame = _normalize_columns(frame)
        date_col = _find_col(frame, ["date"])
        value_col = _find_col(frame, ["naaim", "mean"]) or _find_col(frame, ["naaim number"])
        if not date_col or not value_col:
            raise ValueError("NAAIM columns not found")
        values = _num_series(frame[value_col])
        if values.dropna().abs().median() > 200:
            values = values / 100.0
        history = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "NAAIM_exposure": values}).dropna()
        history = history.sort_values("date")
        latest = history.iloc[-1]
        tail = history["NAAIM_exposure"].tail(260)
        data = {
            "NAAIM_exposure": float(latest["NAAIM_exposure"]),
            "NAAIM_date": pd.Timestamp(latest["date"]).date().isoformat(),
            "NAAIM_exposure_percentile": _percentile(tail, latest["NAAIM_exposure"]),
            "NAAIM_exposure_z_score": _zscore(tail, latest["NAAIM_exposure"]),
        }
        self.histories["NAAIM_exposure"] = history
        self._cache_set(cache_name, data, history)
        self.sources["NAAIM_exposure"] = f"auto:{url}"
        return data

    def _fetch_cftc_cot(self) -> dict:
        cache_name = "cftc_cot_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        frames = []
        current_year = pd.Timestamp.today().year
        for year in range(current_year, current_year - 6, -1):
            url = f"https://www.cftc.gov/files/dea/history/dea_fut_xls_{year}.zip"
            try:
                raw = self._read_url(url, timeout=30)
                with zipfile.ZipFile(BytesIO(raw)) as archive:
                    excel_names = [n for n in archive.namelist() if n.lower().endswith((".xls", ".xlsx"))]
                    if not excel_names:
                        continue
                    with archive.open(excel_names[0]) as fh:
                        frames.append(_read_excel_bytes(fh.read()))
            except Exception as exc:
                self.warnings.append(f"CFTC {year} fetch failed: {exc}")
        if not frames:
            raise ValueError("No CFTC annual files parsed")
        frame = _normalize_columns(pd.concat(frames, ignore_index=True, sort=False))
        name_col = _find_col(frame, ["market", "exchange"]) or _find_col(frame, ["commodity"])
        date_col = _find_col(frame, ["as_of", "date"]) or _find_col(frame, ["as of", "date"]) or _find_col(frame, ["report", "date"])
        long_col = _find_col(frame, ["noncommercial", "long", "all"]) or _find_col(frame, ["noncomm", "long", "all"])
        short_col = _find_col(frame, ["noncommercial", "short", "all"]) or _find_col(frame, ["noncomm", "short", "all"])
        if not name_col or not date_col or not long_col or not short_col:
            raise ValueError("CFTC required columns not found")
        raw_dates = frame[date_col]
        if pd.to_numeric(raw_dates, errors="coerce").dropna().between(100000, 999999).mean() > 0.8:
            frame["date"] = pd.to_datetime(raw_dates.astype(str).str.zfill(6), format="%y%m%d", errors="coerce")
        else:
            frame["date"] = pd.to_datetime(raw_dates, errors="coerce")
        frame["net"] = _num_series(frame[long_col]) - _num_series(frame[short_col])
        data: dict[str, object] = {}
        histories = []
        for label, patterns, exclude_patterns, value_key, pct_key in [
            ("E-MINI S&P 500", ["E-MINI S&P 500"], ["MICRO"], "CFTC_emini_spx_net", "CFTC_emini_spx_net_percentile"),
            ("10Y UST", ["UST 10Y NOTE", "10-YEAR U.S. TREASURY", "ULTRA UST 10Y"], [], "CFTC_10y_ust_net", "CFTC_10y_ust_net_percentile"),
        ]:
            names = frame[name_col].astype(str).str.upper()
            mask = pd.Series(False, index=frame.index)
            for pattern in patterns:
                mask = mask | names.str.contains(pattern, regex=False, na=False)
            for pattern in exclude_patterns:
                mask = mask & ~names.str.contains(pattern, regex=False, na=False)
            subset = frame[mask][["date", "net"]].dropna().sort_values("date")
            if subset.empty:
                continue
            latest = subset.iloc[-1]
            tail = subset["net"].tail(260)
            data[value_key] = float(latest["net"])
            data[pct_key] = _percentile(tail, latest["net"])
            data["CFTC_date"] = pd.Timestamp(latest["date"]).date().isoformat()
            hist_name = pct_key
            hist = subset[["date", "net"]].rename(columns={"net": hist_name})
            hist[hist_name] = hist[hist_name].expanding(min_periods=20).apply(lambda s: _percentile(pd.Series(s), s.iloc[-1]), raw=False)
            self.histories[hist_name] = hist
            histories.append(hist)
        if not data:
            raise ValueError("CFTC target markets not found")
        self._cache_set(cache_name, data, pd.concat(histories, axis=0, ignore_index=True, sort=False) if histories else None)
        for key in data:
            if key.startswith("CFTC_"):
                self.sources[key] = "auto:CFTC annual futures-only XLS"
        return data

    def _fetch_finra_margin(self) -> dict:
        cache_name = "finra_margin_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        urls = [
            "https://www.finra.org/sites/default/files/MarginStatistics.xlsx",
            "https://www.finra.org/sites/default/files/2024-03/Margin_Statistics.xlsx",
        ]
        last_error = None
        for url in urls:
            try:
                raw = self._read_url(url)
                frame = _read_excel_bytes(raw)
                frame = _normalize_columns(frame)
                date_col = _find_col(frame, ["date"]) or frame.columns[0]
                debt_col = _find_col(frame, ["debit", "margin"]) or _find_col(frame, ["debit balances"])
                if not debt_col:
                    raise ValueError("FINRA debit balance column not found")
                history = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "margin_debt": _num_series(frame[debt_col])}).dropna().sort_values("date")
                if history.shape[0] < 13:
                    raise ValueError("FINRA history too short")
                latest = history.iloc[-1]
                yoy = float(latest["margin_debt"] / history.iloc[-13]["margin_debt"] - 1.0)
                data = {"FINRA_margin_debt_yoy": yoy, "FINRA_date": pd.Timestamp(latest["date"]).date().isoformat()}
                self.histories["FINRA_margin_debt_yoy"] = history.assign(FINRA_margin_debt_yoy=history["margin_debt"] / history["margin_debt"].shift(12) - 1.0)[["date", "FINRA_margin_debt_yoy"]]
                self._cache_set(cache_name, data, self.histories["FINRA_margin_debt_yoy"])
                self.sources["FINRA_margin_debt_yoy"] = f"auto:{url}"
                return data
            except Exception as exc:
                last_error = exc
        raise ValueError(f"FINRA margin fetch failed: {last_error}")

    def _fetch_ici_flows(self) -> dict:
        cache_name = "ici_flows_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        url = "https://www.ici.org/research/stats/flows/data"
        raw = self._read_url(url)
        tables = pd.read_html(BytesIO(raw))
        for table in tables:
            frame = _normalize_columns(table)
            date_col = _find_col(frame, ["date"])
            equity_cols = [c for c in frame.columns if "equity" in c.lower()]
            if not date_col or not equity_cols:
                continue
            value = _num_series(frame[equity_cols]).sum(axis=1) if isinstance(frame[equity_cols], pd.DataFrame) else _num_series(frame[equity_cols[0]])
            history = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), "equity_flow": value}).dropna().sort_values("date")
            if history.shape[0] >= 4:
                latest = history.iloc[-1]
                data = {"ICI_4w_equity_flow_b": float(history["equity_flow"].tail(4).sum()), "ICI_date": pd.Timestamp(latest["date"]).date().isoformat()}
                self.histories["ICI_4w_equity_flow_b"] = history.assign(ICI_4w_equity_flow_b=history["equity_flow"].rolling(4).sum())[["date", "ICI_4w_equity_flow_b"]]
                self._cache_set(cache_name, data, self.histories["ICI_4w_equity_flow_b"])
                self.sources["ICI_4w_equity_flow_b"] = f"auto:{url}"
                return data
        raise ValueError("ICI equity flow table not found")

    def _fetch_cboe_pc(self) -> dict:
        cache_name = "cboe_pc_latest.json"
        cached = self._cache_get(cache_name)
        if cached is not None:
            return cached
        specs = {
            "CBOE_equity_pc_10dma": "https://cdn.cboe.com/api/global/us_indices/daily_prices/CPCE_History.csv",
            "CBOE_total_pc_10dma": "https://cdn.cboe.com/api/global/us_indices/daily_prices/CPC_History.csv",
        }
        data = {}
        histories = []
        latest_date = None
        for key, url in specs.items():
            raw = self._read_url(url)
            frame = pd.read_csv(BytesIO(raw))
            frame = _normalize_columns(frame)
            date_col = _find_col(frame, ["date"])
            value_col = _find_col(frame, ["close"]) or _find_col(frame, ["value"])
            if not date_col or not value_col:
                raise ValueError(f"CBOE columns not found for {key}")
            history = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"), key: _num_series(frame[value_col])}).dropna().sort_values("date")
            data[key] = float(history[key].tail(10).mean())
            latest_date = pd.Timestamp(history["date"].iloc[-1]).date().isoformat()
            histories.append(history)
            self.histories[key] = history
            self.sources[key] = f"auto:{url}"
        if latest_date:
            data["CBOE_date"] = latest_date
        self._cache_set(cache_name, data, _outer_merge_histories(histories))
        return data

    def _read_url(self, url: str, timeout: int | None = None) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 sentiment-fetcher"})
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return response.read()

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _cache_get(self, name: str) -> dict | None:
        path = self._cache_path(name)
        if not path.exists():
            return None
        age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
        if age_hours > self.max_cache_age_hours:
            return None
        payload = json.loads(path.read_text())
        for key, source in payload.get("sources", {}).items():
            self.sources[key] = source
        for metric, records in payload.get("histories", {}).items():
            if records:
                hist = pd.DataFrame(records)
                if "date" in hist:
                    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
                self.histories[metric] = hist
        return payload.get("data", {})

    def _cache_set(self, name: str, data: dict, history: pd.DataFrame | None = None) -> None:
        histories = {}
        if history is not None and not history.empty:
            for col in history.columns:
                if col == "date":
                    continue
                histories[col] = history[["date", col]].dropna().assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")).to_dict("records")
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "sources": {key: self.sources.get(key, "auto") for key in data},
            "histories": histories,
        }
        self._cache_path(name).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out.dropna(how="all")


def _read_excel_bytes(raw: bytes, **kwargs) -> pd.DataFrame:
    for engine in [None, "xlrd", "openpyxl"]:
        try:
            options = dict(kwargs)
            if engine is not None:
                options["engine"] = engine
            return pd.read_excel(BytesIO(raw), **options)
        except Exception as exc:
            last_error = exc
    raise last_error


def _parse_naaim_google_chart(raw: bytes) -> pd.DataFrame:
    html = raw.decode("utf-8", errors="ignore")
    matches = re.findall(r"new Date\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\),\s*([+-]?\d+(?:\.\d+)?)", html)
    if not matches:
        raise ValueError("NAAIM chart data not found")
    rows = []
    for year, month0, day, value in matches:
        rows.append(
            {
                "Date": pd.Timestamp(year=int(year), month=int(month0) + 1, day=int(day)),
                "NAAIM Number Mean": float(value),
            }
        )
    return pd.DataFrame(rows)


def _find_col(frame: pd.DataFrame, keywords: list[str]) -> str | None:
    lowered = {col: col.lower() for col in frame.columns}
    for col, low in lowered.items():
        if all(keyword.lower() in low for keyword in keywords):
            return col
    return None


def _num_series(values) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        return values.apply(_num_series)
    return pd.to_numeric(pd.Series(values).astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def _percent_series(values) -> pd.Series:
    out = _num_series(values)
    if out.dropna().abs().median() <= 1.0:
        out = out * 100.0
    return out


def _percentile(history: pd.Series, value: float) -> float:
    clean = pd.to_numeric(history, errors="coerce").dropna()
    if clean.empty or _is_nan(value):
        return np.nan
    return float((clean <= float(value)).mean() * 100.0)


def _zscore(history: pd.Series, value: float) -> float:
    clean = pd.to_numeric(history, errors="coerce").dropna()
    if clean.shape[0] < 2 or clean.std(ddof=1) == 0 or _is_nan(value):
        return np.nan
    return float((float(value) - clean.mean()) / clean.std(ddof=1))


def _is_nan(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _outer_merge_histories(histories: list[pd.DataFrame]) -> pd.DataFrame:
    out = None
    for hist in histories:
        out = hist if out is None else out.merge(hist, on="date", how="outer")
    return out if out is not None else pd.DataFrame()
