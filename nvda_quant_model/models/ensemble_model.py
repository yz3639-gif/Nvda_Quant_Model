from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.models.baseline_model import SMARsiBaseline, VolumeMomentumRule
from nvda_quant_model.models.ml_model import ValidationScore, build_candidate_models


class EnsembleModel:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.models = [
            VolumeMomentumRule(config.rule_quantile, config.rule_max_filters),
            SMARsiBaseline(),
            *build_candidate_models(config.top_k_features, config.random_state),
        ]
        self.weights_: dict[str, float] = {}
        self.validation_scores_: list[ValidationScore] = []
        self.feature_importance_: pd.Series = pd.Series(dtype=float)

    def fit(self, data: pd.DataFrame, feature_columns: list[str]) -> "EnsembleModel":
        data = data.dropna(subset=["target_return", "target_direction"]).copy()
        split = max(int(len(data) * 0.78), min(len(data) - 20, 120))
        if split <= 20 or split >= len(data):
            train, val = data, data.tail(min(len(data), 63))
        else:
            train, val = data.iloc[:split], data.iloc[split:]

        weights: dict[str, float] = {}
        scores: list[ValidationScore] = []
        for model in self.models:
            if hasattr(model, "validation_score"):
                score = model.validation_score(train, val, feature_columns)
                weights[model.name] = score.weight
                scores.append(score)
            else:
                pred = model.fit(train, feature_columns).predict(val)
                acc = float(((pred["prob_up"] >= 0.5).astype(int) == val["target_direction"].astype(int)).mean())
                weight = max(0.05, (acc - 0.5) * 10)
                weights[model.name] = weight
                scores.append(ValidationScore(model.name, acc, 1.0, weight))

        total = sum(weights.values()) or 1.0
        self.weights_ = {name: weight / total for name, weight in weights.items()}
        self.validation_scores_ = scores

        for model in self.models:
            model.fit(data, feature_columns)

        self.feature_importance_ = self._aggregate_feature_importance(feature_columns)
        return self

    def _aggregate_feature_importance(self, feature_columns: list[str]) -> pd.Series:
        importance = pd.Series(0.0, index=feature_columns)
        for model in self.models:
            weight = self.weights_.get(model.name, 0.0)
            selector = getattr(model, "selector", None)
            if selector is not None and not selector.scores_.empty:
                scores = selector.scores_.reindex(feature_columns).fillna(0.0)
                if scores.max() > 0:
                    scores = scores / scores.sum()
                importance = importance.add(scores * weight, fill_value=0.0)
            elif model.name == "Adaptive_Volume_Momentum_Rule":
                for feature in ["20d_return", "volume_sma_ratio"]:
                    if feature in importance.index:
                        importance.loc[feature] += 0.5 * weight
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance.sort_values(ascending=False)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        rule_model = next((model for model in self.models if model.name == "Adaptive_Volume_Momentum_Rule"), None)
        if rule_model is not None:
            pred = rule_model.predict(data)
            prob_up = pred["prob_up"].clip(0.01, 0.99)
            expected_return = pred["expected_return"]
            confidence = np.maximum(prob_up, 1.0 - prob_up)
            direction = pd.Series(0, index=data.index, dtype=int)
            long_mask = (prob_up >= self.config.signal_threshold) & (expected_return >= self.config.min_expected_return)
            direction.loc[long_mask] = 1
            exposure = pd.Series(np.where(direction != 0, self.config.max_exposure, 0.0), index=data.index)
            return pd.DataFrame(
                {
                    "direction": direction,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "prob_up": prob_up,
                    "prob_down": 1.0 - prob_up,
                    "position": direction * exposure,
                },
                index=data.index,
            )

        if self.weights_:
            dominant_name, dominant_weight = max(self.weights_.items(), key=lambda item: item[1])
            if dominant_weight >= 0.65:
                dominant = next(model for model in self.models if model.name == dominant_name)
                pred = dominant.predict(data)
                prob_up = pred["prob_up"].clip(0.01, 0.99)
                expected_return = pred["expected_return"]
                confidence = np.maximum(prob_up, 1.0 - prob_up)
                direction = pd.Series(0, index=data.index, dtype=int)
                long_mask = (prob_up >= self.config.signal_threshold) & (expected_return >= self.config.min_expected_return)
                direction.loc[long_mask] = 1
                exposure = pd.Series(np.where(direction != 0, self.config.max_exposure, 0.0), index=data.index)
                return pd.DataFrame(
                    {
                        "direction": direction,
                        "confidence": confidence,
                        "expected_return": expected_return,
                        "prob_up": prob_up,
                        "prob_down": 1.0 - prob_up,
                        "position": direction * exposure,
                    },
                    index=data.index,
                )

        weighted_prob = pd.Series(0.0, index=data.index)
        weighted_return = pd.Series(0.0, index=data.index)
        for model in self.models:
            pred = model.predict(data)
            weight = self.weights_.get(model.name, 0.0)
            weighted_prob += pred["prob_up"] * weight
            weighted_return += pred["expected_return"] * weight

        prob_up = weighted_prob.clip(0.01, 0.99)
        confidence = np.maximum(prob_up, 1.0 - prob_up)
        direction = pd.Series(0, index=data.index, dtype=int)
        long_mask = (prob_up >= self.config.signal_threshold) & (weighted_return >= self.config.min_expected_return)
        direction.loc[long_mask] = 1

        exposure = pd.Series(np.where(direction != 0, self.config.max_exposure, 0.0), index=data.index)
        position = direction * exposure
        return pd.DataFrame(
            {
                "direction": direction,
                "confidence": confidence,
                "expected_return": weighted_return,
                "prob_up": prob_up,
                "prob_down": 1.0 - prob_up,
                "position": position,
            },
            index=data.index,
        )

    def model_comparison_frame(self) -> pd.DataFrame:
        rows = []
        for score in self.validation_scores_:
            rows.append(
                {
                    "model": score.name,
                    "validation_accuracy": score.accuracy,
                    "validation_log_loss": score.log_loss,
                    "ensemble_weight": self.weights_.get(score.name, 0.0),
                    "selected_rule": getattr(score, "selected_rule", ""),
                    "coverage": getattr(score, "coverage", np.nan),
                }
            )
        return pd.DataFrame(rows).sort_values("ensemble_weight", ascending=False)
