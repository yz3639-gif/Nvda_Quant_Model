from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.data.data_validation import TrainingPreprocessor, purge_immature_labels
from nvda_quant_model.models.baseline_model import SMARsiBaseline, VolumeMomentumRule
from nvda_quant_model.models.ml_model import ValidationScore, build_candidate_models, load_model_parameter_overrides


class EnsembleModel:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.prediction_mode = config.effective_prediction_mode
        if self.prediction_mode == "precision_rule":
            raise ValueError("precision_rule uses the dedicated main.run precision-rule bypass")
        rule_model = VolumeMomentumRule(config.rule_quantile, config.rule_max_filters)
        parameter_overrides = load_model_parameter_overrides(config.model_params_path)
        if self.prediction_mode == "ml_ensemble":
            self.models = build_candidate_models(config.top_k_features, config.random_state, parameter_overrides)
        else:
            self.models = [rule_model] if config.fast_rule_only else [
                rule_model, SMARsiBaseline(),
                *build_candidate_models(config.top_k_features, config.random_state, parameter_overrides),
            ]
        self.weights_: dict[str, float] = {}
        self.validation_scores_: list[ValidationScore] = []
        self.feature_importance_: pd.Series = pd.Series(dtype=float)

    def fit(self, data: pd.DataFrame, feature_columns: list[str]) -> "EnsembleModel":
        data = data.dropna(subset=["target_return", "target_direction"]).sort_index().copy()
        if len(data) < 30:
            raise ValueError("At least 30 labeled rows are required for disjoint temporal validation")
        split = max(int(len(data) * 0.78), min(len(data) - 20, 120))
        train, val = data.iloc[:split], data.iloc[split:]
        self.temporal_validation_ = "legacy_missing_availability"
        inner_fit_at = None
        if "label_end_at" in data or "available_at" in data:
            if "available_at" not in data:
                raise ValueError("Training requires explicit available_at")
            inner_fit_at = pd.to_datetime(val["available_at"], utc=True).min()
            train = purge_immature_labels(train, inner_fit_at)
            self.temporal_validation_ = "purged_by_label_end_at"
        if train.empty or val.empty:
            raise ValueError("No disjoint mature train/validation samples")
        inner_preprocessor = TrainingPreprocessor(feature_columns).fit(train, fit_at=inner_fit_at)
        self.inner_preprocessor_ = inner_preprocessor
        train = inner_preprocessor.transform(train)
        val = inner_preprocessor.transform(val)

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

        final_fit_at = pd.to_datetime(data["fit_at"], utc=True).max() if "fit_at" in data else None
        self.preprocessor_ = TrainingPreprocessor(feature_columns).fit(data, fit_at=final_fit_at)
        fitted_data = self.preprocessor_.transform(data)
        for model in self.models:
            model.fit(fitted_data, feature_columns)

        self.diagnostic_feature_importance_ = self._aggregate_feature_importance(feature_columns, signal_only=False)
        self.feature_importance_ = self._aggregate_feature_importance(feature_columns, signal_only=True)
        return self

    def _aggregate_feature_importance(self, feature_columns: list[str], *, signal_only: bool = True) -> pd.Series:
        importance = pd.Series(0.0, index=feature_columns)
        weights = self.prediction_metadata()["signal_weights"] if signal_only else self.weights_
        for model in self.models:
            weight = weights.get(model.name, 0.0)
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

    def prediction_metadata(self) -> dict:
        rule_name = "Adaptive_Volume_Momentum_Rule"
        return {
            "prediction_mode": self.prediction_mode,
            "signal_model": rule_name if self.prediction_mode == "rule" else "ML_Weighted_Ensemble",
            "signal_models": [rule_name] if self.prediction_mode == "rule" else [m.name for m in self.models],
            "diagnostic_models": [m.name for m in self.models if m.name != rule_name] if self.prediction_mode == "rule" else [],
            "signal_weights": {rule_name: 1.0} if self.prediction_mode == "rule" else dict(self.weights_),
            "probability_status": "uncalibrated_score",
            "feature_importance_method": "rule_input_proxy" if self.prediction_mode == "rule" else "weighted_feature_selection_scores",
            "temporal_validation": getattr(self, "temporal_validation_", "not_fitted"),
        }

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self, "preprocessor_"):
            data = self.preprocessor_.transform(data)
        if self.prediction_mode == "rule":
            rule_model = next(model for model in self.models if model.name == "Adaptive_Volume_Momentum_Rule")
            pred = rule_model.predict(data)
            prob_up = pred["prob_up"].clip(0.01, 0.99)
            expected_return = pred["expected_return"]
        else:
            if not self.weights_:
                raise RuntimeError("Fit ml_ensemble before predicting")
            prob_up = pd.Series(0.0, index=data.index)
            expected_return = pd.Series(0.0, index=data.index)
            for model in self.models:
                pred = model.predict(data)
                weight = self.weights_.get(model.name, 0.0)
                prob_up += pred["prob_up"] * weight
                expected_return += pred["expected_return"] * weight
            prob_up = prob_up.clip(0.01, 0.99)
        confidence = np.maximum(prob_up, 1.0 - prob_up)
        direction = pd.Series(0, index=data.index, dtype=int)
        long_mask = (prob_up >= self.config.signal_threshold) & (expected_return >= self.config.min_expected_return)
        direction.loc[long_mask] = 1
        exposure = pd.Series(np.where(direction != 0, self.config.max_exposure, 0.0), index=data.index)
        out = pd.DataFrame(
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
        out.attrs.update(self.prediction_metadata())
        return out

    def model_comparison_frame(self) -> pd.DataFrame:
        rows = []
        for score in self.validation_scores_:
            rows.append(
                {
                    "model": score.name,
                    "validation_accuracy": score.accuracy,
                    "validation_log_loss": score.log_loss,
                    "ensemble_weight": self.weights_.get(score.name, 0.0),
                    "signal_weight": self.prediction_metadata()["signal_weights"].get(score.name, 0.0),
                    "role": "diagnostic" if score.name in self.prediction_metadata()["diagnostic_models"] else "signal",
                    "prediction_mode": self.prediction_mode,
                    "selected_rule": getattr(score, "selected_rule", ""),
                    "coverage": getattr(score, "coverage", np.nan),
                }
            )
        return pd.DataFrame(rows).sort_values("ensemble_weight", ascending=False) if rows else pd.DataFrame()
