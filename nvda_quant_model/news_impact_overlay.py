from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


DATASET_NAME = "GGLabYale/MTBench_finance_aligned_pairs_short"
DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/GGLabYale/MTBench_finance_aligned_pairs_short/"
    "resolve/main/data/train-00000-of-00001.parquet"
)

NEWS_TEXT_COLUMNS = ("text",)
INPUT_PRICE_COLUMNS = ("input_timestamps", "input_window")
LABEL_COLUMNS = ("trend", "alignment")
RAW_LEAKAGE_COLUMNS = ("output_timestamps", "output_window")
MIXED_TECHNICAL_COLUMN = "technical"
LEAKAGE_PREFIXES = ("out_", "overall_", "output_")


@dataclass(frozen=True)
class ColumnAudit:
    """Dataset columns grouped by how they are allowed to be used.

    The raw ``technical`` column in MTBench is a mixed JSON payload. Keys prefixed
    with ``in_`` are point-in-time inputs, while ``out_`` and ``overall_`` use
    future prices. The raw column is therefore excluded from model input and only
    parsed through ``extract_safe_technical_features``.
    """

    news_text_features: list[str]
    pre_news_price_features: list[str]
    label_fields: list[str]
    excluded_leakage_fields: dict[str, str]


@dataclass
class NewsImpactFeaturePipeline:
    """Feature builder that never fits on output-window or label columns."""

    text_method: Literal["tfidf", "finbert", "none"] = "tfidf"
    max_text_features: int = 256
    finbert_model_name: str = "ProsusAI/finbert"
    vectorizer: TfidfVectorizer | None = None
    scaler: StandardScaler | None = None
    price_feature_columns: list[str] | None = None
    text_feature_columns: list[str] | None = None

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        price_features = build_price_feature_frame(frame)
        self.price_feature_columns = list(price_features.columns)
        assert_no_data_leakage(self.price_feature_columns)
        self.scaler = StandardScaler(with_mean=False)
        price_matrix = sparse.csr_matrix(self.scaler.fit_transform(price_features))
        text_matrix = self._fit_transform_text(extract_news_texts(frame))
        return sparse.hstack([text_matrix, price_matrix], format="csr")

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if self.scaler is None or self.price_feature_columns is None:
            raise ValueError("NewsImpactFeaturePipeline must be fitted before transform().")
        price_features = build_price_feature_frame(frame).reindex(columns=self.price_feature_columns, fill_value=0.0)
        assert_no_data_leakage(list(price_features.columns))
        price_matrix = sparse.csr_matrix(self.scaler.transform(price_features))
        text_matrix = self._transform_text(extract_news_texts(frame))
        return sparse.hstack([text_matrix, price_matrix], format="csr")

    @property
    def feature_names(self) -> list[str]:
        return list(self.text_feature_columns or []) + list(self.price_feature_columns or [])

    def _fit_transform_text(self, texts: list[str]) -> sparse.csr_matrix:
        if self.text_method == "none":
            self.text_feature_columns = []
            return sparse.csr_matrix((len(texts), 0), dtype=float)
        if self.text_method == "finbert":
            matrix = _finbert_embeddings(texts, self.finbert_model_name)
            self.text_feature_columns = [f"finbert_embedding_{idx}" for idx in range(matrix.shape[1])]
            return sparse.csr_matrix(matrix)
        self.vectorizer = TfidfVectorizer(max_features=self.max_text_features, ngram_range=(1, 2), min_df=1)
        matrix = self.vectorizer.fit_transform(texts)
        self.text_feature_columns = [f"text_tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        return matrix.tocsr()

    def _transform_text(self, texts: list[str]) -> sparse.csr_matrix:
        if self.text_method == "none":
            return sparse.csr_matrix((len(texts), 0), dtype=float)
        if self.text_method == "finbert":
            matrix = _finbert_embeddings(texts, self.finbert_model_name)
            return sparse.csr_matrix(matrix)
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer is not fitted.")
        return self.vectorizer.transform(texts).tocsr()


@dataclass
class NewsImpactModel:
    """Lightweight classifier for the news event awareness layer.

    This model is deliberately not a trading model. It converts point-in-time
    news and pre-news price context into distribution-adjustment scores.
    """

    target: Literal["direction", "alignment"]
    model_type: str
    feature_pipeline: NewsImpactFeaturePipeline
    classifier: Any
    classes_: list[str]
    column_audit: ColumnAudit


@dataclass
class NewsImpactBundle:
    direction_model: NewsImpactModel | None
    alignment_model: NewsImpactModel | None
    column_audit: ColumnAudit


def load_mtbench_finance_dataset(
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    parquet_url: str = DEFAULT_PARQUET_URL,
) -> pd.DataFrame:
    """Load the MTBench finance aligned-pairs dataset from Hugging Face.

    The preferred path uses ``datasets.load_dataset`` when the optional
    dependency is installed. A pandas/parquet fallback is provided so the module
    remains usable in lean research environments.
    """

    try:
        from datasets import load_dataset  # type: ignore

        loaded = load_dataset(dataset_name, split=split)
        return loaded.to_pandas()
    except ModuleNotFoundError:
        if dataset_name != DATASET_NAME or split != "train":
            raise ImportError(
                "Install the optional 'datasets' package to load non-default MTBench splits."
            )
        return pd.read_parquet(parquet_url)


def inspect_mtbench_columns(columns: list[str] | pd.Index) -> ColumnAudit:
    names = list(columns)
    leakage = {
        "output_timestamps": "Future timestamps after the news event.",
        "output_window": "Future prices after the news event; allowed only for label construction/evaluation.",
    }
    if MIXED_TECHNICAL_COLUMN in names:
        leakage[MIXED_TECHNICAL_COLUMN] = (
            "Raw technical JSON mixes safe in_* indicators with leaking out_* and overall_* indicators; "
            "only parsed in_* keys are allowed."
        )
    return ColumnAudit(
        news_text_features=[col for col in NEWS_TEXT_COLUMNS if col in names],
        pre_news_price_features=[col for col in INPUT_PRICE_COLUMNS if col in names]
        + (["technical[in_* parsed only]"] if MIXED_TECHNICAL_COLUMN in names else []),
        label_fields=[col for col in LABEL_COLUMNS if col in names],
        excluded_leakage_fields={key: value for key, value in leakage.items() if key in names},
    )


def assert_no_data_leakage(feature_columns: list[str]) -> None:
    """Fail fast if a model input column contains future or label information."""

    forbidden_exact = set(RAW_LEAKAGE_COLUMNS) | set(LABEL_COLUMNS) | {MIXED_TECHNICAL_COLUMN}
    leaks: list[str] = []
    for column in feature_columns:
        lowered = column.lower()
        if lowered in forbidden_exact or any(lowered.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            leaks.append(column)
    if leaks:
        raise ValueError(
            "News Impact Overlay feature matrix contains data-leakage fields: "
            + ", ".join(sorted(leaks))
        )


def parse_json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_news_texts(frame: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for raw in frame.get("text", pd.Series([""] * len(frame))).tolist():
        parsed = parse_json_cell(raw)
        if parsed:
            content = str(parsed.get("content", "") or "")
            title = str(parsed.get("title", "") or parsed.get("headline", "") or "")
            url = str(parsed.get("article_url", "") or parsed.get("url", "") or "")
            texts.append(" ".join(part for part in [title, content, url] if part).strip())
        else:
            texts.append(str(raw or ""))
    return texts


def event_times(frame: pd.DataFrame) -> pd.Series:
    """Return point-in-time event timestamps for time-based splitting."""

    out: list[pd.Timestamp] = []
    for _, row in frame.iterrows():
        parsed = parse_json_cell(row.get("text"))
        candidate = parsed.get("published_at") or parsed.get("published_on") or parsed.get("date")
        content = str(parsed.get("content", ""))
        if not candidate:
            match = re.search(r"Published on:\s*([0-9T:Z+\-]+)", content)
            candidate = match.group(1) if match else None
        if candidate:
            try:
                out.append(pd.Timestamp(candidate).tz_localize(None))
                continue
            except Exception:
                pass
        timestamps = _as_float_array(row.get("input_timestamps"))
        if timestamps.size:
            out.append(pd.to_datetime(float(timestamps[-1]), unit="s", utc=True).tz_localize(None))
        else:
            out.append(pd.NaT)
    return pd.Series(out, index=frame.index, name="event_time")


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)] if arr.ndim == 1 else arr


def _window_to_frame(value: Any) -> pd.DataFrame:
    """Normalize input_window into a pre-event price frame.

    The current MTBench finance parquet stores ``input_window`` as a close-price
    vector. If future versions provide OHLCV arrays, this function can consume
    [open, high, low, close, volume] columns without changing the leakage rules.
    """

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        close = pd.Series(arr[np.isfinite(arr)], dtype=float)
        return pd.DataFrame({"close": close})
    if arr.ndim == 2 and arr.shape[1] >= 5:
        frame = pd.DataFrame(arr[:, :5], columns=["open", "high", "low", "close", "volume"])
        return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if arr.ndim == 2 and arr.shape[1] >= 1:
        close = pd.Series(arr[:, -1], dtype=float)
        return pd.DataFrame({"close": close.replace([np.inf, -np.inf], np.nan).dropna()})
    return pd.DataFrame({"close": pd.Series(dtype=float)})


def _pct_change(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return 0.0
    base = float(clean.iloc[-periods - 1])
    if base == 0.0:
        return 0.0
    return float(clean.iloc[-1] / base - 1.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _safe_last_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    if numerator.empty or denominator.empty:
        return 0.0
    den = float(denominator.iloc[-1])
    if not np.isfinite(den) or den == 0.0:
        return 0.0
    return float(numerator.iloc[-1] / den - 1.0)


def extract_safe_technical_features(raw_technical: Any, close: pd.Series) -> dict[str, float]:
    """Parse only in_* technical indicators.

    ``out_*`` and ``overall_*`` keys are deliberately ignored because they are
    computed with post-news prices and would leak the answer into training.
    """

    parsed = parse_json_cell(raw_technical)
    features: dict[str, float] = {}
    last_close = float(close.dropna().iloc[-1]) if close.dropna().size else 0.0
    for key, values in parsed.items():
        if not str(key).startswith("in_"):
            continue
        safe_key = re.sub(r"[^0-9a-zA-Z_]+", "_", str(key).lower())
        arr = _as_float_array(values)
        if arr.size == 0:
            continue
        series = pd.Series(arr, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        last = float(series.iloc[-1])
        first = float(series.iloc[0])
        features[f"tech_{safe_key}_last"] = last
        if last_close:
            features[f"tech_{safe_key}_last_to_close"] = float(last / last_close - 1.0)
        features[f"tech_{safe_key}_change"] = float(last / first - 1.0) if first else 0.0
    return features


def extract_input_price_features(row: pd.Series) -> dict[str, float]:
    frame = _window_to_frame(row.get("input_window"))
    close = frame["close"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    features: dict[str, float] = {
        "input_length": float(len(close)),
        "has_volume": 0.0,
        "has_vwap": 0.0,
    }
    if close.size < 2:
        return features

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    features.update(
        {
            "pre_return_1": _pct_change(close, 1),
            "pre_return_5": _pct_change(close, min(5, max(len(close) - 1, 1))),
            "pre_return_20": _pct_change(close, min(20, max(len(close) - 1, 1))),
            "pre_return_full": float(close.iloc[-1] / close.iloc[0] - 1.0) if close.iloc[0] else 0.0,
            "pre_realized_vol_full": float(returns.std(ddof=0)) if len(returns) else 0.0,
            "pre_realized_vol_20": float(returns.tail(20).std(ddof=0)) if len(returns) else 0.0,
            "pre_return_mean_20": float(returns.tail(20).mean()) if len(returns) else 0.0,
            "pre_return_min_20": float(returns.tail(20).min()) if len(returns) else 0.0,
            "pre_return_max_20": float(returns.tail(20).max()) if len(returns) else 0.0,
        }
    )

    sma_10 = close.rolling(10, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()
    ema_10 = _ema(close, 10)
    ema_50 = _ema(close, 50)
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, 9)
    rolling_mean = close.rolling(20, min_periods=2).mean()
    rolling_std = close.rolling(20, min_periods=2).std(ddof=0).replace(0.0, np.nan)
    upper = rolling_mean + 2.0 * rolling_std
    lower = rolling_mean - 2.0 * rolling_std
    last_close = float(close.iloc[-1])
    band_width = float((upper.iloc[-1] - lower.iloc[-1]) / last_close) if last_close and pd.notna(upper.iloc[-1]) else 0.0
    percent_b_den = upper.iloc[-1] - lower.iloc[-1] if pd.notna(upper.iloc[-1]) else np.nan
    percent_b = float((last_close - lower.iloc[-1]) / percent_b_den) if percent_b_den and pd.notna(percent_b_den) else 0.5
    features.update(
        {
            "sma_10_ratio": _safe_last_ratio(close, sma_10),
            "sma_50_ratio": _safe_last_ratio(close, sma_50),
            "ema_10_ratio": _safe_last_ratio(close, ema_10),
            "ema_50_ratio": _safe_last_ratio(close, ema_50),
            "ema_10_50_spread": float(ema_10.iloc[-1] / ema_50.iloc[-1] - 1.0) if ema_50.iloc[-1] else 0.0,
            "macd_last": float(macd.iloc[-1]),
            "macd_signal_last": float(macd_signal.iloc[-1]),
            "macd_hist_last": float(macd.iloc[-1] - macd_signal.iloc[-1]),
            "boll_bandwidth_20": band_width,
            "boll_percent_b_20": float(np.clip(percent_b, -1.0, 2.0)),
            "boll_zscore_20": float((last_close - rolling_mean.iloc[-1]) / rolling_std.iloc[-1])
            if pd.notna(rolling_std.iloc[-1]) and rolling_std.iloc[-1]
            else 0.0,
        }
    )

    if "volume" in frame and frame["volume"].notna().sum() >= 5:
        volume = frame["volume"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        vol_base = float(volume.tail(20).mean())
        features["has_volume"] = 1.0
        features["volume_anomaly"] = float(volume.iloc[-1] / vol_base - 1.0) if vol_base else 0.0
        if vol_base and len(volume) == len(close):
            vwap = float((close.tail(len(volume)) * volume).sum() / volume.sum()) if volume.sum() else last_close
            features["has_vwap"] = 1.0
            features["vwap_deviation"] = float(last_close / vwap - 1.0) if vwap else 0.0
    else:
        features["volume_anomaly"] = 0.0
        features["vwap_deviation"] = 0.0

    features.update(extract_safe_technical_features(row.get("technical"), close))
    return features


def build_price_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_input_price_features(row) for _, row in frame.iterrows()]
    out = pd.DataFrame(rows, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert_no_data_leakage(list(out.columns))
    return out


def _trend_payload(value: Any) -> dict[str, Any]:
    return parse_json_cell(value)


def output_percentage_change(frame: pd.DataFrame) -> pd.Series:
    changes: list[float] = []
    for _, row in frame.iterrows():
        trend = _trend_payload(row.get("trend"))
        if "output_percentage_change" in trend:
            changes.append(float(trend["output_percentage_change"]))
            continue
        output = _as_float_array(row.get("output_window"))
        if output.size >= 2 and output[0] != 0:
            changes.append(float((output[-1] / output[0] - 1.0) * 100.0))
        else:
            changes.append(0.0)
    return pd.Series(changes, index=frame.index, name="output_percentage_change")


def build_direction_target(frame: pd.DataFrame, neutral_threshold_pct: float = 0.25) -> pd.Series:
    change = output_percentage_change(frame)
    labels = np.where(change > neutral_threshold_pct, "bullish", np.where(change < -neutral_threshold_pct, "bearish", "neutral"))
    return pd.Series(labels, index=frame.index, name="direction_label")


def build_alignment_target(frame: pd.DataFrame) -> pd.Series:
    labels = []
    for raw in frame.get("alignment", pd.Series([""] * len(frame))).tolist():
        value = str(raw).strip().strip('"').lower()
        labels.append("aligned" if value in {"consistent", "aligned", "true", "1"} else "not_aligned")
    return pd.Series(labels, index=frame.index, name="alignment_label")


def prepare_news_impact_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["event_time"] = event_times(prepared)
    prepared["direction_label"] = build_direction_target(prepared)
    prepared["alignment_label"] = build_alignment_target(prepared)
    return prepared.sort_values("event_time", kind="mergesort")


def _make_classifier(model_type: str, random_state: int) -> Any:
    if model_type == "logistic":
        base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        return OneVsRestClassifier(base)
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("Install optional xgboost to use model_type='xgboost'.") from exc
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_news_impact_model(
    frame: pd.DataFrame,
    target: Literal["direction", "alignment"] = "direction",
    model_type: str = "logistic",
    text_method: Literal["tfidf", "finbert", "none"] = "tfidf",
    train_fraction: float = 0.80,
    random_state: int = 7,
) -> tuple[NewsImpactModel, dict[str, float]]:
    """Train a time-split news overlay classifier.

    ``trend`` and ``alignment`` are consumed only to build labels. They are never
    available to the feature pipeline.
    """

    prepared = prepare_news_impact_dataset(frame)
    if prepared.empty:
        raise ValueError("Cannot train News Impact Overlay on an empty frame.")
    split_idx = int(len(prepared) * train_fraction)
    split_idx = min(max(split_idx, 1), len(prepared) - 1)
    train = prepared.iloc[:split_idx]
    test = prepared.iloc[split_idx:]
    y_train = train["direction_label"] if target == "direction" else train["alignment_label"]
    y_test = test["direction_label"] if target == "direction" else test["alignment_label"]
    if y_train.nunique() < 2:
        raise ValueError(f"Training target {target!r} needs at least two classes.")

    pipeline = NewsImpactFeaturePipeline(text_method=text_method)
    x_train = pipeline.fit_transform(train)
    assert_no_data_leakage(pipeline.feature_names)
    classifier = _make_classifier(model_type, random_state=random_state)
    classifier.fit(x_train, y_train)

    model = NewsImpactModel(
        target=target,
        model_type=model_type,
        feature_pipeline=pipeline,
        classifier=classifier,
        classes_=[str(item) for item in classifier.classes_],
        column_audit=inspect_mtbench_columns(frame.columns),
    )
    metrics = evaluate_news_impact_model(model, test, y_true=y_test)
    return model, metrics


def train_news_impact_bundle(
    frame: pd.DataFrame,
    model_type: str = "logistic",
    text_method: Literal["tfidf", "finbert", "none"] = "tfidf",
    train_fraction: float = 0.80,
    random_state: int = 7,
) -> tuple[NewsImpactBundle, dict[str, dict[str, float]]]:
    direction_model, direction_metrics = train_news_impact_model(
        frame,
        target="direction",
        model_type=model_type,
        text_method=text_method,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    alignment_model: NewsImpactModel | None = None
    alignment_metrics: dict[str, float] = {}
    try:
        alignment_model, alignment_metrics = train_news_impact_model(
            frame,
            target="alignment",
            model_type=model_type,
            text_method=text_method,
            train_fraction=train_fraction,
            random_state=random_state + 1,
        )
    except ValueError:
        alignment_model = None
    bundle = NewsImpactBundle(
        direction_model=direction_model,
        alignment_model=alignment_model,
        column_audit=inspect_mtbench_columns(frame.columns),
    )
    return bundle, {"direction": direction_metrics, "alignment": alignment_metrics}


def _predict_proba_frame(model: NewsImpactModel, frame: pd.DataFrame) -> pd.DataFrame:
    matrix = model.feature_pipeline.transform(prepare_news_impact_dataset(frame))
    probabilities = model.classifier.predict_proba(matrix)
    return pd.DataFrame(probabilities, columns=model.classes_, index=frame.index)


def evaluate_news_impact_model(
    model: NewsImpactModel,
    frame: pd.DataFrame,
    y_true: pd.Series | None = None,
) -> dict[str, float]:
    prepared = prepare_news_impact_dataset(frame)
    if y_true is None:
        y_true = prepared["direction_label"] if model.target == "direction" else prepared["alignment_label"]
    if prepared.empty:
        return {}
    proba = _predict_proba_frame(model, prepared)
    pred = proba.idxmax(axis=1)
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(y_true, pred))}
    labels = list(model.classes_)
    try:
        metrics["log_loss"] = float(log_loss(y_true, proba.reindex(columns=labels), labels=labels))
    except ValueError:
        metrics["log_loss"] = float("nan")

    if model.target == "direction":
        y_up = (pd.Series(y_true, index=prepared.index) == "bullish").astype(int)
        prob_up = proba.get("bullish", pd.Series(0.0, index=prepared.index)).astype(float)
    else:
        y_up = (pd.Series(y_true, index=prepared.index) == "aligned").astype(int)
        prob_up = proba.get("aligned", pd.Series(0.5, index=prepared.index)).astype(float)
    try:
        metrics["auc"] = float(roc_auc_score(y_up, prob_up)) if y_up.nunique() > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["brier_score"] = float(brier_score_loss(y_up, prob_up))
    calibration = calibration_curve_frame(y_up, prob_up)
    metrics["calibration_mean_abs_error"] = float(calibration["abs_error"].mean()) if not calibration.empty else float("nan")
    return metrics


def calibration_curve_frame(y_true_binary: pd.Series, prob_up: pd.Series, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y_true_binary.astype(int), "p": prob_up.astype(float).clip(0.0, 1.0)})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        count=("y", "size"),
        mean_predicted_probability=("p", "mean"),
        realized_frequency=("y", "mean"),
    )
    grouped["abs_error"] = (grouped["mean_predicted_probability"] - grouped["realized_frequency"]).abs()
    return grouped.reset_index(drop=True)


def predict_news_impact_scores(bundle: NewsImpactBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Convert classifier probabilities into conservative overlay scores."""

    if bundle.direction_model is None:
        raise ValueError("A direction model is required to score news impact.")
    direction = _predict_proba_frame(bundle.direction_model, frame)
    bullish = direction.get("bullish", pd.Series(0.0, index=direction.index)).astype(float)
    bearish = direction.get("bearish", pd.Series(0.0, index=direction.index)).astype(float)
    neutral = direction.get("neutral", pd.Series(0.0, index=direction.index)).astype(float)
    if bundle.alignment_model is not None:
        alignment = _predict_proba_frame(bundle.alignment_model, frame).get(
            "aligned", pd.Series(0.5, index=direction.index)
        )
    else:
        alignment = pd.Series(0.5, index=direction.index)

    directional_strength = (bullish - bearish).abs()
    non_neutral = (1.0 - neutral).clip(0.0, 1.0)
    shock = (0.50 * non_neutral + 0.35 * directional_strength + 0.15 * (1.0 - alignment)).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "news_bullish_score": bullish.clip(0.0, 1.0),
            "news_bearish_score": bearish.clip(0.0, 1.0),
            "news_alignment_probability": alignment.astype(float).clip(0.0, 1.0),
            "news_shock_strength": shock.astype(float).clip(0.0, 1.0),
            "news_volatility_multiplier": 1.0 + 0.25 * shock.astype(float).clip(0.0, 1.0),
        },
        index=frame.index,
    )


def adjust_monte_carlo_distribution(
    base_prob_up: float,
    base_mu: float,
    base_sigma: float,
    option_implied_volatility: float | None,
    news_scores: dict[str, float] | pd.Series,
    lambda_mu: float = 0.05,
    lambda_vol: float = 0.25,
) -> dict[str, float | str | None]:
    """Apply the News Impact Overlay as a distribution adjustment only.

    This function does not emit buy/sell/hold decisions. It only nudges the
    drift and volatility used by a downstream Monte Carlo engine:

    adjusted_mu = base_mu + lambda_mu * (bullish_score - bearish_score)
    adjusted_sigma = base_sigma * (1 + lambda_vol * news_shock_strength)
    """

    bullish = float(news_scores.get("news_bullish_score", 0.0))
    bearish = float(news_scores.get("news_bearish_score", 0.0))
    shock = float(news_scores.get("news_shock_strength", 0.0))
    news_signal = float(np.clip(bullish - bearish, -1.0, 1.0))
    adjusted_mu = float(base_mu + lambda_mu * news_signal)
    adjusted_sigma = float(base_sigma * (1.0 + lambda_vol * np.clip(shock, 0.0, 1.0)))
    return {
        "base_prob_up": float(base_prob_up),
        "base_mu": float(base_mu),
        "base_sigma": float(base_sigma),
        "option_implied_volatility": None if option_implied_volatility is None else float(option_implied_volatility),
        "news_signal": news_signal,
        "news_shock_strength": float(np.clip(shock, 0.0, 1.0)),
        "lambda_mu": float(lambda_mu),
        "lambda_vol": float(lambda_vol),
        "adjusted_mu": adjusted_mu,
        "adjusted_sigma": adjusted_sigma,
        "distribution_role": "News event awareness layer; adjusts distribution, not trading decision.",
    }


def walk_forward_news_overlay_validation(
    frame: pd.DataFrame,
    min_train_size: int = 250,
    test_size: int = 50,
    step: int = 50,
    model_type: str = "logistic",
    text_method: Literal["tfidf", "finbert"] = "tfidf",
    random_state: int = 7,
) -> dict[str, Any]:
    """Compare price-only baseline vs price+news overlay with time splits."""

    prepared = prepare_news_impact_dataset(frame).reset_index(drop=True)
    if len(prepared) <= min_train_size + test_size:
        raise ValueError("Not enough samples for walk-forward validation.")
    rows: list[dict[str, float | int | str]] = []
    split_id = 0
    for start in range(min_train_size, len(prepared) - test_size + 1, step):
        train = prepared.iloc[:start]
        test = prepared.iloc[start : start + test_size]
        for name, method in [("base_price_only", "none"), ("news_overlay", text_method)]:
            model, _ = train_news_impact_model(
                train,
                target="direction",
                model_type=model_type,
                text_method=method,  # type: ignore[arg-type]
                train_fraction=0.999,
                random_state=random_state + split_id,
            )
            metrics = evaluate_news_impact_model(model, test)
            rows.append(
                {
                    "split": split_id,
                    "model": name,
                    "train_size": len(train),
                    "test_size": len(test),
                    **metrics,
                }
            )
        split_id += 1
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model")[["accuracy", "auc", "brier_score", "log_loss", "calibration_mean_abs_error"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    return {
        "summary": summary,
        "windows": results,
        "principle": "This module adjusts the distribution, not the trading decision.",
    }


def _finbert_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Install optional 'transformers' and 'torch' to use FinBERT embeddings, "
            "or call with text_method='tfidf'."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            encoded = tokenizer(
                texts[start : start + 16],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.cpu().numpy())
    return np.vstack(batches) if batches else np.empty((0, 0), dtype=float)
