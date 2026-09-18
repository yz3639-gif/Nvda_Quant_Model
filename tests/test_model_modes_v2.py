from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.models.baseline_model import VolumeMomentumRule
from nvda_quant_model.models.ensemble_model import EnsembleModel
from nvda_quant_model.models.ml_model import ValidationScore


class FixedModel:
    def __init__(self, name, probability, weight):
        self.name, self.probability, self.weight = name, probability, weight
        self.fits = []

    def fit(self, data, features):
        self.fits.append(data.copy())
        return self

    def predict(self, data):
        return pd.DataFrame({'prob_up': self.probability, 'expected_return': 0.01}, index=data.index)

    def validation_score(self, train, val, features):
        self.validation_train = train.copy()
        self.validation_data = val.copy()
        return ValidationScore(self.name, 0.6, 0.7, self.weight)


def frame(n=160):
    index = pd.bdate_range('2024-01-02', periods=n)
    return pd.DataFrame({'20d_return': np.sin(np.arange(n) / 6) * .1,
                         'volume_sma_ratio': 1 + np.cos(np.arange(n) / 5) * .5,
                         'target_return': np.sin(np.arange(n) / 6 + .2) * .02,
                         'target_direction': (np.sin(np.arange(n) / 6 + .2) > 0).astype(int)}, index=index)


def test_default_rule_matches_direct_rule_and_diagnostics_do_not_drive_signal(monkeypatch):
    monkeypatch.setattr('nvda_quant_model.models.ensemble_model.build_candidate_models', lambda *args: [FixedModel('always_bearish', .05, 10)])
    data = frame()
    features = ['20d_return', 'volume_sma_ratio']
    model = EnsembleModel(StrategyConfig()).fit(data, features)
    fast = EnsembleModel(StrategyConfig(fast_rule_only=True)).fit(data, features)
    pd.testing.assert_frame_equal(model.predict(data), fast.predict(data), check_flags=False, check_freq=False)
    direct = VolumeMomentumRule(.55, 2).fit(model.preprocessor_.transform(data), features)
    pd.testing.assert_series_equal(model.predict(data)['prob_up'], direct.predict(model.preprocessor_.transform(data))['prob_up'].clip(.01, .99))
    meta = model.prediction_metadata()
    assert meta['prediction_mode'] == 'rule'
    assert meta['signal_weights'] == {'Adaptive_Volume_Momentum_Rule': 1.0}
    assert 'always_bearish' in meta['diagnostic_models']
    comparison = model.model_comparison_frame().set_index('model')
    assert comparison.loc['always_bearish', 'signal_weight'] == 0
    assert comparison.loc['always_bearish', 'role'] == 'diagnostic'


def test_ml_mode_uses_real_weighted_candidates_even_with_dominant_weight(monkeypatch):
    candidates = [FixedModel('ml_a', .8, 4), FixedModel('ml_b', .2, 1)]
    monkeypatch.setattr('nvda_quant_model.models.ensemble_model.build_candidate_models', lambda *args: candidates)
    model = EnsembleModel(StrategyConfig(prediction_mode='ml_ensemble')).fit(frame(), ['20d_return'])
    result = model.predict(frame(3))
    assert np.allclose(result.prob_up, .68)
    assert result.attrs['signal_model'] == 'ML_Weighted_Ensemble'
    assert result.attrs['diagnostic_models'] == []
    assert result.attrs['signal_weights'] == {'ml_a': .8, 'ml_b': .2}
    assert result.attrs['probability_status'] == 'uncalibrated_score'


@pytest.mark.parametrize('config', [StrategyConfig(prediction_mode='unknown'),
    StrategyConfig(prediction_mode='precision_rule'),
    StrategyConfig(prediction_mode='ml_ensemble', fast_rule_only=True),
    StrategyConfig(prediction_mode='ml_ensemble', precision_rule_path='x.json')])
def test_invalid_or_conflicting_modes_fail_explicitly(config):
    with pytest.raises(ValueError):
        EnsembleModel(config)


def test_precision_path_preserves_old_bypass_and_names_actual_mode():
    config = StrategyConfig(precision_rule_path='rule.json')
    assert config.effective_prediction_mode == 'precision_rule'
    assert replace(config, prediction_mode='precision_rule').effective_prediction_mode == 'precision_rule'
    with pytest.raises(ValueError, match='dedicated main.run'):
        EnsembleModel(config)


def test_inner_validation_preprocessor_does_not_see_validation_outlier_and_purges(monkeypatch):
    candidate = FixedModel('ml_a', .6, 1)
    monkeypatch.setattr('nvda_quant_model.models.ensemble_model.build_candidate_models', lambda *args: [candidate])
    data = frame(100)
    data['available_at'] = pd.to_datetime(data.index, utc=True) + pd.Timedelta(hours=21)
    data['label_end_at'] = data['available_at'].shift(-3)
    # Use mature dummy endpoints for the final rows; outer caller owns final fit cutoff.
    data['label_end_at'] = data['label_end_at'].fillna(data['available_at'].iloc[-1])
    split = 80
    data.iloc[split:, data.columns.get_loc('20d_return')] = 10000
    model = EnsembleModel(StrategyConfig(prediction_mode='ml_ensemble')).fit(data, ['20d_return'])
    assert model.inner_preprocessor_.upper_['20d_return'] < 1
    assert candidate.validation_train.label_end_at.max() <= candidate.validation_data.available_at.min()
    assert len(candidate.validation_train) < split
    assert candidate.validation_data['20d_return'].max() < 1
    assert model.prediction_metadata()['temporal_validation'] == 'purged_by_label_end_at'


def test_tiny_training_does_not_fall_back_to_resubstitution_validation():
    with pytest.raises(ValueError, match='disjoint'):
        EnsembleModel(StrategyConfig(fast_rule_only=True)).fit(frame(20), ['20d_return'])


def test_real_sklearn_ensemble_fits_and_predicts_bounded_values(monkeypatch):
    from nvda_quant_model.models.ml_model import build_candidate_models
    candidates = build_candidate_models(top_k=2, parameter_overrides={
        'HistGradientBoosting': {'classifier': {'max_iter': 5}, 'regressor': {'max_iter': 5}}})
    candidates = [m for m in candidates if m.name in {'ElasticNet_Logit', 'HistGradientBoosting'}]
    monkeypatch.setattr('nvda_quant_model.models.ensemble_model.build_candidate_models', lambda *args: candidates)
    model = EnsembleModel(StrategyConfig(prediction_mode='ml_ensemble')).fit(frame(), ['20d_return', 'volume_sma_ratio'])
    result = model.predict(frame(10))
    assert result.prob_up.between(.01, .99).all()
    assert np.isfinite(result.expected_return).all()
    assert sum(model.weights_.values()) == pytest.approx(1)
