from __future__ import annotations

import numpy as np
import pandas as pd

from methods.historical import simulate_block_bootstrap_paths
from methods.monte_carlo import fit_student_t_distribution, simulate_gbm_paths, simulate_student_t_gbm_paths
from nvda_quant_model.calibration import _newey_west_mean_se


def interval_score(actual, low, high, level):
    alpha = 1-level
    return high-low + 2/alpha*max(low-actual, 0) + 2/alpha*max(actual-high, 0)


def distribution_experiment(close, config):
    """Terminal-return forecasts only; no path-risk or Student-t price-mean claims."""
    train_n = config['distribution_train_sessions']
    n_sims = config['distribution_simulations']
    returns = np.log(close/close.shift()).dropna()
    rows = []
    fitted = {}
    for horizon in config['distribution_horizons']:
        designs = [('nonoverlap', horizon)]
        if horizon > 5:
            designs.append(('overlap_supplement', 5))
        for design, step in designs:
            for pos in range(train_n, len(close)-horizon, step):
                history = returns.loc[:close.index[pos]].tail(train_n)
                if len(history) < train_n:
                    continue
                if pos not in fitted:
                    x = history.to_numpy()
                    sigma = float(x.std(ddof=1))
                    ewma = float(history.ewm(span=60, adjust=False).std().iloc[-1])
                    fitted[pos] = (x, sigma, ewma, fit_student_t_distribution(x).df)
                x, sigma, ewma, df = fitted[pos]
                mu = float(x.mean())
                actual = float(close.iloc[pos+horizon]/close.iloc[pos]-1)
                for vol_id, vol in [('rolling', sigma), ('ewma', ewma)]:
                    if not np.isfinite(vol) or vol <= 0:
                        raise ValueError('Invalid fitted volatility')
                    adjusted = (x-mu)*vol/sigma+mu
                    for model_id, method in enumerate(['bootstrap', 'normal', 'student_t']):
                        # Stable local seed; adding another candidate does not change old draws.
                        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], pos, horizon, model_id]))
                        if method == 'bootstrap':
                            paths = simulate_block_bootstrap_paths(adjusted, 1, horizon, n_sims, 5, rng)
                        elif method == 'normal':
                            paths = simulate_gbm_paths(1, horizon, n_sims, (mu+.5*vol**2)*252, vol*np.sqrt(252), rng)
                        else:
                            paths = simulate_student_t_gbm_paths(1, horizon, n_sims, (mu+.5*vol**2)*252, vol*np.sqrt(252), df, rng)
                        terminal = paths[:, -1]-1
                        if not np.isfinite(terminal).all():
                            raise ValueError('Nonfinite simulated terminal returns')
                        row = {'model': f'{method}_{vol_id}', 'method': method, 'volatility': vol_id,
                            'design': design, 'horizon': horizon, 'step': step, 'as_of': close.index[pos],
                            'label_end': close.index[pos+horizon], 'actual_return': actual,
                            'pit': float((terminal <= actual).mean()), 'train_n': train_n, 'n_sims': n_sims,
                            'sigma': vol, 'log_drift': mu, 'student_t_df': df if method == 'student_t' else None}
                        for level in [.50, .80, .95]:
                            low, high = np.quantile(terminal, [(1-level)/2, 1-(1-level)/2])
                            tag = int(level*100)
                            row.update({f'low_{tag}': low, f'high_{tag}': high, f'width_{tag}': high-low,
                                        f'hit_{tag}': low <= actual <= high, f'below_{tag}': actual < low,
                                        f'above_{tag}': actual > high, f'score_{tag}': interval_score(actual, low, high, level)})
                        rows.append(row)
    windows = pd.DataFrame(rows)
    summary = []
    if windows.empty:
        raise ValueError('No distribution evaluation windows')
    for (model, design, horizon, step), group in windows.groupby(['model', 'design', 'horizon', 'step']):
        # Nonoverlapping outcomes still share fitted histories; HAC is descriptive.
        lags = max(1, int(np.ceil(horizon/step)))
        for level in [.5, .8, .95]:
            tag = int(level*100)
            hit = group[f'hit_{tag}'].astype(float)
            se = _newey_west_mean_se(hit, lags)
            coverage = float(hit.mean())
            summary.append({'model': model, 'design': design, 'horizon': horizon, 'level': level,
                'coverage': coverage, 'coverage_hac_se': se, 'coverage_z': (coverage-level)/se if se > 0 else None,
                'below_rate': group[f'below_{tag}'].mean(), 'above_rate': group[f'above_{tag}'].mean(),
                'interval_width': group[f'width_{tag}'].mean(), 'interval_score': group[f'score_{tag}'].mean(),
                'pit_mean': group.pit.mean(), 'n': len(group), 'hac_lags': lags,
                'status': 'development_evaluation_no_automatic_promotion'})
    return windows, pd.DataFrame(summary)
