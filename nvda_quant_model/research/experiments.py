"""Nested chronological prediction experiments; no test-driven threshold search."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nvda_quant_model.data.data_validation import TrainingPreprocessor, purge_immature_labels

FEATURES = ['return_1', 'return_5', 'return_20', 'return_60', 'vol_20', 'vol_60',
            'range', 'overnight', 'volume_ratio', 'price_ma20', 'price_ma60', 'volume_change']
MODEL_GRID = {'rule': [None], 'logistic': [.2, 1.0], 'boosting': [2, 3]}


def research_frame(prices: pd.DataFrame) -> pd.DataFrame:
    from nvda_quant_model.data.session_calendar import session_schedule
    out = pd.DataFrame(index=prices.index)
    close = prices.Close
    ret = close.pct_change()
    for n in [1, 5, 20, 60]:
        out[f'return_{n}'] = close.pct_change(n)
    for n in [20, 60]:
        out[f'vol_{n}'] = ret.rolling(n).std()
    out['range'] = (prices.High - prices.Low) / close
    out['overnight'] = prices.Open / close.shift() - 1
    out['volume_ratio'] = prices.Volume / prices.Volume.rolling(20).mean()
    out['price_ma20'] = close / close.rolling(20).mean() - 1
    out['price_ma60'] = close / close.rolling(60).mean() - 1
    out['volume_change'] = prices.Volume.pct_change().clip(-.99, 10)
    # Decision after session t close. Target covers open(t+1) to open(t+2).
    out['target_return'] = prices.Open.shift(-2) / prices.Open.shift(-1) - 1
    out['target_direction'] = (out.target_return > 0).astype(float).where(out.target_return.notna())
    times = session_schedule(prices.index)
    out['available_at'] = times['market_close'].to_numpy()
    out['decision_at'] = out['available_at']
    out['label_end_at'] = pd.Series(times['market_open'].to_numpy(), index=prices.index).shift(-2)
    return out.iloc[60:].copy()


def _mature(data, fit_at):
    return purge_immature_labels(data, fit_at=fit_at).dropna(subset=['target_direction', 'target_return'])


class FittedPredictor:
    def __init__(self, family, parameter, seed):
        self.family, self.parameter, self.seed = family, parameter, seed

    def fit(self, train, fit_at):
        train = _mature(train, fit_at)
        if len(train) < 30:
            raise ValueError('Insufficient mature training rows')
        self.preprocessor = TrainingPreprocessor(FEATURES).fit(train, fit_at=fit_at)
        x = self.preprocessor.transform(train)[FEATURES]
        y = train.target_direction.astype(int)
        self.base_rate = (y.sum() + 1) / (len(y) + 2)
        self.train_abs_return = float(train.target_return.abs().median())
        self.model = None
        if self.family == 'rule':
            self.momentum = float(x.return_20.quantile(.55))
            self.volume = float(x.volume_ratio.quantile(.55))
            active = (x.return_20 > self.momentum) & (x.volume_ratio > self.volume)
            self.active_rate = (y[active].sum() + 1) / (active.sum() + 2)
            self.inactive_rate = (y[~active].sum() + 1) / ((~active).sum() + 2)
        elif y.nunique() > 1:
            if self.family == 'logistic':
                self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.parameter, max_iter=500, random_state=self.seed))
            else:
                self.model = HistGradientBoostingClassifier(max_iter=80, max_depth=self.parameter,
                    max_leaf_nodes=2**self.parameter, min_samples_leaf=20, learning_rate=.04,
                    l2_regularization=1, random_state=self.seed)
            self.model.fit(x, y)
        return self

    def predict(self, frame):
        x = self.preprocessor.transform(frame)[FEATURES]
        if self.family == 'rule':
            active = (x.return_20 > self.momentum) & (x.volume_ratio > self.volume)
            return np.where(active, self.active_rate, self.inactive_rate).clip(.001, .999)
        if self.model is None:
            return np.full(len(frame), self.base_rate)
        return self.model.predict_proba(x)[:, 1].clip(.001, .999)


def _logit(p):
    p = np.clip(p, .001, .999)
    return np.log(p / (1-p)).reshape(-1, 1)


def probability_scores(y, p):
    return {'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, np.clip(p, .001, .999), labels=[0, 1])), 'n': len(y)}


def fit_fold(train, test, family, seed, threshold, target_volatility=.25, max_exposure=1.):
    """Inner tune then calibrate on disjoint forward blocks; freeze at outer decision."""
    fit_at = test.decision_at.iloc[0]
    train = _mature(train, fit_at)
    n = len(train)
    a, b = int(n*.60), int(n*.80)
    tuning, calibration = train.iloc[a:b], train.iloc[b:]
    tuning = _mature(tuning, calibration.decision_at.iloc[0])
    if min(a, len(tuning), len(calibration)) < 20:
        raise ValueError('Fold too short for train / tune / calibrate chronology')
    trials = []
    for parameter in MODEL_GRID[family]:
        model = FittedPredictor(family, parameter, seed).fit(train.iloc[:a], tuning.decision_at.iloc[0])
        p = model.predict(tuning)
        trials.append({'parameter': parameter, **probability_scores(tuning.target_direction.astype(int), p)})
    parameter = min(trials, key=lambda r: r['brier'])['parameter']
    calibration_model = FittedPredictor(family, parameter, seed).fit(train.iloc[:b], calibration.decision_at.iloc[0])
    cp = calibration_model.predict(calibration)
    cy = calibration.target_direction.astype(int).to_numpy()
    # Calibrator selection sees an earlier part of the calibration block only.
    split = max(10, len(cp)//2)
    mappings = {'raw': lambda p: p, 'shrink': lambda p: .5*p + .5*calibration_model.base_rate}
    mature_cal = (calibration.label_end_at.iloc[:split] <= calibration.decision_at.iloc[split]).to_numpy()
    if len(np.unique(cy[:split][mature_cal])) == 2:
        sigmoid = LogisticRegression(C=1, random_state=seed).fit(_logit(cp[:split][mature_cal]), cy[:split][mature_cal])
        mappings['sigmoid'] = lambda p: sigmoid.predict_proba(_logit(p))[:, 1]
    cal_scores = {name: probability_scores(cy[split:], fn(cp[split:])) for name, fn in mappings.items()}
    selected_calibrator = min(cal_scores, key=lambda name: cal_scores[name]['brier'])
    if len(np.unique(cy)) == 2:
        sigmoid_all = LogisticRegression(C=1, random_state=seed).fit(_logit(cp), cy)
    else:
        sigmoid_all = None
    # Final refit uses only labels mature by outer fit time. Hyperparameters and mapping already chosen.
    final = FittedPredictor(family, parameter, seed).fit(train, fit_at)
    raw = final.predict(test)
    pred = pd.DataFrame(index=test.index)
    pred['raw'] = raw
    pred['shrink'] = .5*raw + .5*calibration_model.base_rate
    pred['sigmoid'] = sigmoid_all.predict_proba(_logit(raw))[:, 1] if sigmoid_all is not None else calibration_model.base_rate
    pred['calibrated'] = pred[selected_calibrator]
    pred['base_rate'] = final.base_rate
    pred['target_direction'] = test.target_direction
    pred['target_return'] = test.target_return
    pred['vol_20'] = test.vol_20
    pred['decision_at'] = test.decision_at
    pred['label_end_at'] = test.label_end_at
    pred['fit_at'] = fit_at
    pred['selected_parameter'] = 'fixed' if parameter is None else str(parameter)
    pred['selected_calibrator'] = selected_calibrator
    # Exposure is estimated on the forward calibration holdout using the chosen
    # earlier-fitted map. It is not an exact match of future realized risk.
    active_train = mappings[selected_calibrator](cp[split:]) >= threshold
    pred['training_exposure'] = float(active_train.mean()) * max_exposure
    scaler = (target_volatility/(calibration.vol_20.iloc[split:]*np.sqrt(252))).clip(upper=max_exposure).fillna(0)
    pred['training_vol_exposure'] = float((active_train*scaler).mean())
    pred['training_conditional_size'] = float(scaler.loc[active_train].mean()) if active_train.any() else 0.
    pred['abs_return_train'] = final.train_abs_return
    record = {'family': family, 'fit_at': fit_at, 'train_first': train.index[0], 'train_last': train.index[-1],
              'max_training_label_end_at': train.label_end_at.max(), 'train_n': len(train),
              'test_first': test.index[0], 'test_last': test.index[-1], 'selected_parameter': parameter,
              'selected_calibrator': selected_calibrator, 'candidate_trials': trials, 'calibration_trials': cal_scores,
              'tuning_fit_at': tuning.decision_at.iloc[0],
              'max_inner_training_label_end': _mature(train.iloc[:a], tuning.decision_at.iloc[0]).label_end_at.max(),
              'calibration_fit_at': calibration.decision_at.iloc[0],
              'max_calibration_model_training_label_end': _mature(train.iloc[:b], calibration.decision_at.iloc[0]).label_end_at.max(),
              'calibrator_selection_fit_at': calibration.decision_at.iloc[split],
              'max_sigmoid_selection_label_end': calibration.label_end_at.iloc[:split].loc[mature_cal].max(),
              'calibration_n': len(calibration), 'calibration_status': 'forward_inner_block_refit_transfer_requires_outer_validation'}
    return pred, record


def run_predictions(frame, config):
    start = config['train_sessions']
    outputs, records = [], []
    for fold, pos in enumerate(range(start, len(frame)-2, config['test_sessions'])):
        if config.get('max_folds') is not None and fold >= config['max_folds']:
            break
        train = frame.iloc[max(0, pos-config['train_sessions']):pos]
        test = frame.iloc[pos:min(pos+config['test_sessions'], len(frame)-2)]
        if len(train) < config['min_train_sessions'] or test.empty:
            continue
        for family in MODEL_GRID:
            p, record = fit_fold(train, test, family, config['seed'], config['probability_threshold'], config['target_volatility'], config['max_exposure'])
            p['model'] = family
            p['fold'] = fold
            p.index.name = 'Date'
            outputs.append(p.reset_index())
            records.append({'fold': fold, **record})
    if not outputs:
        raise ValueError('No valid evaluation folds; supply more data or a smaller registered training window')
    return pd.concat(outputs, ignore_index=True), records
