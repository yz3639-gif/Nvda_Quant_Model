from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ValidationScore:
    name: str
    accuracy: float
    log_loss: float
    weight: float


class FeatureSelector:
    def __init__(self, top_k: int = 12, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.scores_: pd.Series = pd.Series(dtype=float)

    def fit(self, data: pd.DataFrame, feature_columns: list[str]) -> "FeatureSelector":
        X = data[feature_columns].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = data["target_direction"].astype(int)
        if len(feature_columns) <= self.top_k:
            self.selected_features_ = list(feature_columns)
            self.scores_ = pd.Series(1.0, index=feature_columns)
            return self

        scores = mutual_info_classif(X, y, random_state=self.random_state, discrete_features=False)
        ranking = pd.Series(scores, index=feature_columns).sort_values(ascending=False)

        selected: list[str] = []
        corr = X[ranking.index].corr().abs()
        for col in ranking.index:
            if len(selected) >= self.top_k:
                break
            if selected and corr.loc[col, selected].max() > 0.92:
                continue
            selected.append(col)
        self.selected_features_ = selected or list(ranking.head(self.top_k).index)
        self.scores_ = ranking
        return self


class SklearnDirectionReturnModel:
    def __init__(
        self,
        name: str,
        classifier: BaseEstimator,
        regressor: BaseEstimator,
        top_k: int = 12,
        random_state: int = 42,
        scale: bool = False,
    ):
        self.name = name
        self.classifier = classifier
        self.regressor = regressor
        self.selector = FeatureSelector(top_k=top_k, random_state=random_state)
        self.scale = scale
        self.selected_features_: list[str] = []

    def _build_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def fit(self, data: pd.DataFrame, feature_columns: list[str]) -> "SklearnDirectionReturnModel":
        self.selector.fit(data, feature_columns)
        self.selected_features_ = self.selector.selected_features_
        X = data[self.selected_features_]
        y_dir = data["target_direction"].astype(int)
        y_ret = data["target_return"].astype(float)
        self.clf_ = self._build_pipeline(clone(self.classifier))
        self.reg_ = self._build_pipeline(clone(self.regressor))
        self.clf_.fit(X, y_dir)
        self.reg_.fit(X, y_ret)
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        X = data[self.selected_features_]
        if hasattr(self.clf_[-1], "predict_proba"):
            prob_up = self.clf_.predict_proba(X)[:, 1]
        else:
            decision = self.clf_.decision_function(X)
            prob_up = 1 / (1 + np.exp(-decision))
        expected_return = self.reg_.predict(X)
        return pd.DataFrame(
            {
                "prob_up": np.clip(prob_up, 0.02, 0.98),
                "expected_return": expected_return,
            },
            index=data.index,
        )

    def validation_score(self, train: pd.DataFrame, val: pd.DataFrame, feature_columns: list[str]) -> ValidationScore:
        model = clone_model(self)
        model.fit(train, feature_columns)
        pred = model.predict(val)
        y = val["target_direction"].astype(int)
        pred_label = (pred["prob_up"] >= 0.5).astype(int)
        acc = accuracy_score(y, pred_label)
        try:
            ll = log_loss(y, pred["prob_up"].clip(0.01, 0.99))
        except ValueError:
            ll = 1.0
        edge = max(0.0, acc - 0.5)
        weight = max(0.05, edge * 10 + max(0.0, 0.75 - ll))
        return ValidationScore(self.name, float(acc), float(ll), float(weight))


def clone_model(model: SklearnDirectionReturnModel) -> SklearnDirectionReturnModel:
    return SklearnDirectionReturnModel(
        name=model.name,
        classifier=clone(model.classifier),
        regressor=clone(model.regressor),
        top_k=model.selector.top_k,
        random_state=model.selector.random_state,
        scale=model.scale,
    )


def build_candidate_models(top_k: int = 12, random_state: int = 42) -> list[SklearnDirectionReturnModel]:
    models: list[SklearnDirectionReturnModel] = [
        SklearnDirectionReturnModel(
            "ElasticNet_Logit",
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=random_state, max_iter=1000),
            ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=random_state, max_iter=5000),
            top_k=top_k,
            random_state=random_state,
            scale=True,
        ),
        SklearnDirectionReturnModel(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=12,
                random_state=random_state,
                class_weight="balanced_subsample",
            ),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=12,
                random_state=random_state,
            ),
            top_k=top_k,
            random_state=random_state,
        ),
        SklearnDirectionReturnModel(
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.035,
                max_leaf_nodes=8,
                l2_regularization=0.2,
                random_state=random_state,
            ),
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.035,
                max_leaf_nodes=8,
                l2_regularization=0.2,
                random_state=random_state,
            ),
            top_k=top_k,
            random_state=random_state,
        ),
    ]

    try:  # pragma: no cover - optional dependency
        from xgboost import XGBClassifier, XGBRegressor

        models.append(
            SklearnDirectionReturnModel(
                "XGBoost",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=0.5,
                    reg_alpha=0.1,
                    eval_metric="logloss",
                    random_state=random_state,
                ),
                XGBRegressor(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=0.5,
                    reg_alpha=0.1,
                    random_state=random_state,
                ),
                top_k=top_k,
                random_state=random_state,
            )
        )
    except Exception:
        pass

    return models

