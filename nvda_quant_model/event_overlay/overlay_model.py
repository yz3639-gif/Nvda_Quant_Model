from __future__ import annotations

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

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame
from nvda_quant_model.event_overlay.labels import build_event_labels
from nvda_quant_model.event_overlay.schema import assert_no_event_feature_leakage, event_text


@dataclass
class EventOverlayFeaturePipeline:
    text_method: Literal["tfidf", "finbert", "none"] = "tfidf"
    include_event_context: bool = True
    max_text_features: int = 256
    finbert_model_name: str = "ProsusAI/finbert"
    vectorizer: TfidfVectorizer | None = None
    scaler: StandardScaler | None = None
    numeric_columns: list[str] | None = None
    text_columns: list[str] | None = None

    def fit_transform(
        self,
        events: pd.DataFrame,
        market_data: pd.DataFrame,
        external_data: pd.DataFrame | None = None,
    ) -> sparse.csr_matrix:
        numeric = build_event_feature_frame(events, market_data, external_data, self.include_event_context)
        self.numeric_columns = list(numeric.columns)
        assert_no_event_feature_leakage(self.numeric_columns)
        self.scaler = StandardScaler(with_mean=False)
        numeric_matrix = sparse.csr_matrix(self.scaler.fit_transform(numeric))
        text_matrix = self._fit_text(event_text(classify_events(events)))
        return sparse.hstack([text_matrix, numeric_matrix], format="csr")

    def transform(
        self,
        events: pd.DataFrame,
        market_data: pd.DataFrame,
        external_data: pd.DataFrame | None = None,
    ) -> sparse.csr_matrix:
        if self.scaler is None or self.numeric_columns is None:
            raise ValueError("EventOverlayFeaturePipeline must be fitted first.")
        numeric = build_event_feature_frame(events, market_data, external_data, self.include_event_context)
        numeric = numeric.reindex(columns=self.numeric_columns, fill_value=0.0)
        assert_no_event_feature_leakage(list(numeric.columns))
        numeric_matrix = sparse.csr_matrix(self.scaler.transform(numeric))
        text_matrix = self._transform_text(event_text(classify_events(events)))
        return sparse.hstack([text_matrix, numeric_matrix], format="csr")

    @property
    def feature_names(self) -> list[str]:
        return list(self.text_columns or []) + list(self.numeric_columns or [])

    def _fit_text(self, texts: list[str]) -> sparse.csr_matrix:
        if self.text_method == "none":
            self.text_columns = []
            return sparse.csr_matrix((len(texts), 0), dtype=float)
        if self.text_method == "finbert":
            matrix = _finbert_embeddings(texts, self.finbert_model_name)
            self.text_columns = [f"finbert_{i}" for i in range(matrix.shape[1])]
            return sparse.csr_matrix(matrix)
        self.vectorizer = TfidfVectorizer(max_features=self.max_text_features, ngram_range=(1, 2), min_df=1)
        matrix = self.vectorizer.fit_transform(texts)
        self.text_columns = [f"text_tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        return matrix.tocsr()

    def _transform_text(self, texts: list[str]) -> sparse.csr_matrix:
        if self.text_method == "none":
            return sparse.csr_matrix((len(texts), 0), dtype=float)
        if self.text_method == "finbert":
            return sparse.csr_matrix(_finbert_embeddings(texts, self.finbert_model_name))
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer must be fitted first.")
        return self.vectorizer.transform(texts).tocsr()


@dataclass
class EventImpactOverlayModel:
    target_horizon: int
    model_type: str
    feature_pipeline: EventOverlayFeaturePipeline
    direction_classifier: Any
    volatility_classifier: Any
    tail_classifier: Any
    direction_classes: list[str]


def _classifier(model_type: str, random_state: int) -> Any:
    if model_type == "logistic":
        return OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


class ConstantClassifier:
    def __init__(self, value: int | str):
        self.value = value
        self.classes_ = np.array([value])

    def fit(self, x: Any, y: Any) -> "ConstantClassifier":
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        return np.ones((x.shape[0], 1), dtype=float)

    def predict(self, x: Any) -> np.ndarray:
        return np.array([self.value] * x.shape[0])


def _fit_or_constant(classifier: Any, x: sparse.csr_matrix, y: pd.Series) -> Any:
    if y.nunique(dropna=True) < 2:
        return ConstantClassifier(y.dropna().iloc[0] if len(y.dropna()) else 0)
    classifier.fit(x, y)
    return classifier


def train_event_overlay_model(
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    target_horizon: int = 1,
    model_type: str = "logistic",
    text_method: Literal["tfidf", "finbert", "none"] = "tfidf",
    include_event_context: bool = True,
    random_state: int = 7,
) -> tuple[EventImpactOverlayModel, dict[str, float]]:
    classified = classify_events(events)
    labels = build_event_labels(classified, market_data, horizons=(1, 3, 5))
    target_col = f"future_{target_horizon}d_direction"
    training = classified.join(labels.set_index("event_id"), on="event_id")
    training = training.loc[training[target_col] != "unknown"].reset_index(drop=True)
    if len(training) < 5:
        raise ValueError("Not enough labeled events to train Event Impact Overlay.")

    pipeline = EventOverlayFeaturePipeline(text_method=text_method, include_event_context=include_event_context)
    x = pipeline.fit_transform(training, market_data, external_data)
    assert_no_event_feature_leakage(pipeline.feature_names)
    y_direction = training[target_col].astype(str)
    y_vol = training["volatility_shock"].astype(int)
    y_tail = training["downside_tail_risk"].astype(int)
    direction_classifier = _fit_or_constant(_classifier(model_type, random_state), x, y_direction)
    volatility_classifier = _fit_or_constant(_classifier(model_type, random_state + 1), x, y_vol)
    tail_classifier = _fit_or_constant(_classifier(model_type, random_state + 2), x, y_tail)
    model = EventImpactOverlayModel(
        target_horizon=target_horizon,
        model_type=model_type,
        feature_pipeline=pipeline,
        direction_classifier=direction_classifier,
        volatility_classifier=volatility_classifier,
        tail_classifier=tail_classifier,
        direction_classes=[str(item) for item in direction_classifier.classes_],
    )
    metrics = evaluate_event_overlay_model(model, training, market_data, external_data, target_horizon)
    return model, metrics


def _probability_frame(classifier: Any, x: sparse.csr_matrix) -> pd.DataFrame:
    return pd.DataFrame(classifier.predict_proba(x), columns=[str(item) for item in classifier.classes_])


def _binary_probability(classifier: Any, x: sparse.csr_matrix) -> pd.Series:
    proba = _probability_frame(classifier, x)
    return proba.get("1", pd.Series(0.0, index=proba.index)).astype(float)


def score_event_overlay(
    model: EventImpactOverlayModel,
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    classified = classify_events(events).reset_index(drop=True)
    x = model.feature_pipeline.transform(classified, market_data, external_data)
    direction = _probability_frame(model.direction_classifier, x)
    bullish = direction.get("bullish", pd.Series(0.0, index=classified.index)).astype(float)
    bearish = direction.get("bearish", pd.Series(0.0, index=classified.index)).astype(float)
    vol_prob = _binary_probability(model.volatility_classifier, x)
    tail_prob = _binary_probability(model.tail_classifier, x)
    importance = classified["event_importance"].astype(float).clip(0.0, 1.0)
    relevance = classified["nvda_relevance"].astype(float).clip(0.0, 1.0)
    quality = classified["source_quality"].astype(float).clip(0.0, 1.0)
    reliability_weight = (0.50 * importance + 0.30 * relevance + 0.20 * quality).clip(0.0, 1.0)
    raw_direction = (bullish - bearish).clip(-1.0, 1.0)
    mu_adjustment = (raw_direction * reliability_weight).clip(-1.0, 1.0)
    sigma_multiplier = (1.0 + 0.30 * vol_prob * reliability_weight).clip(1.0, 1.30)
    tail_multiplier = (1.0 + 0.50 * tail_prob * reliability_weight).clip(1.0, 1.50)
    rows = []
    for idx, row in classified.iterrows():
        reasons = list(row.get("reason_codes", []))
        if vol_prob.iloc[idx] >= 0.55:
            reasons.append("volatility_shock_risk")
        if tail_prob.iloc[idx] >= 0.55:
            reasons.append("downside_tail_risk")
        rows.append(
            {
                "event_id": row["event_id"],
                "published_at": row["published_at"],
                "event_type": row["event_type"],
                "event_bullish_score": float(bullish.iloc[idx]),
                "event_bearish_score": float(bearish.iloc[idx]),
                "event_importance": float(importance.iloc[idx]),
                "event_mu_adjustment": float(mu_adjustment.iloc[idx]),
                "event_sigma_multiplier": float(sigma_multiplier.iloc[idx]),
                "event_tail_risk_multiplier": float(tail_multiplier.iloc[idx]),
                "volatility_shock_probability": float(vol_prob.iloc[idx]),
                "downside_tail_probability": float(tail_prob.iloc[idx]),
                "reason_codes": reasons,
            }
        )
    return pd.DataFrame(rows)


def evaluate_event_overlay_model(
    model: EventImpactOverlayModel,
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    target_horizon: int | None = None,
) -> dict[str, float]:
    horizon = target_horizon or model.target_horizon
    labels = build_event_labels(events, market_data, horizons=(1, 3, 5))
    target_col = f"future_{horizon}d_direction"
    frame = classify_events(events).join(labels.set_index("event_id"), on="event_id")
    frame = frame.loc[frame[target_col] != "unknown"].reset_index(drop=True)
    if frame.empty:
        return {}
    x = model.feature_pipeline.transform(frame, market_data, external_data)
    direction = _probability_frame(model.direction_classifier, x)
    pred = direction.idxmax(axis=1)
    y = frame[target_col].astype(str)
    y_up = (y == "bullish").astype(int)
    p_up = direction.get("bullish", pd.Series(0.0, index=direction.index)).astype(float)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y_up, p_up)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_up, p_up)) if y_up.nunique() > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    try:
        metrics["log_loss"] = float(log_loss(y, direction, labels=list(direction.columns)))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def _finbert_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise ImportError("Install optional transformers and torch, or use text_method='tfidf'.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            encoded = tokenizer(texts[start : start + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.cpu().numpy())
    return np.vstack(batches) if batches else np.empty((0, 0), dtype=float)
