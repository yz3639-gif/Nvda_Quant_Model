"""One offline entry point for preregistered daily research and honest reporting."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.backtest.execution import ExecutionConfig
from nvda_quant_model.config import StrategyConfig
from nvda_quant_model.data.data_validation import validate_ohlcv
from nvda_quant_model.research.distributions import distribution_experiment
from nvda_quant_model.research.experiments import probability_scores, research_frame, run_predictions
from nvda_quant_model.research.provenance import block_mean_interval, code_state, dependencies, directory_hash, sha256, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]


def validate_config(config):
    positive = ['train_sessions', 'test_sessions', 'min_train_sessions', 'distribution_train_sessions',
                'distribution_simulations', 'bootstrap_repetitions', 'bootstrap_block_sessions']
    for key in positive:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['train_sessions'] < config['min_train_sessions']:
        raise ValueError('Training window shorter than minimum')
    if config['candidate_budget'] > 12 or config['candidate_budget'] < 11:
        raise ValueError('This registered design has 11 candidates; budget must be 11 or 12')
    if not config['distribution_horizons'] or any(h <= 0 for h in config['distribution_horizons']):
        raise ValueError('Distribution horizons must be positive')
    if not .5 <= config['probability_threshold'] < 1:
        raise ValueError('Threshold must be prespecified in [.5,1)')


def signal_frame(position, probability=None):
    out = pd.DataFrame({'position': position})
    p = pd.Series(.5, index=out.index) if probability is None else probability.reindex(out.index).fillna(.5)
    out['direction'] = np.sign(out.position)
    out['prob_up'] = p
    out['prob_down'] = 1-p
    out['confidence'] = np.maximum(p, 1-p)
    out['expected_return'] = 0.0
    return out


def save_backtest(result, path):
    path.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(path/'equity.csv', index_label='Date')
    result.trades.to_csv(path/'trades.csv', index=False)
    result.fills.to_csv(path/'fills.csv', index=False)
    write_json(path/'account.json', result.account)
    write_json(path/'metrics.json', result.metrics)


def trading_experiment(prices, frame, predictions, config, output):
    strategy = StrategyConfig(commission=config['commission'], slippage=config['slippage'],
        stop_loss_pct=config['stop_loss_pct'], take_profit_pct=config['take_profit_pct'],
        max_exposure=config['max_exposure'], max_drawdown_limit=config['max_drawdown_limit'])
    dates = pd.DatetimeIndex(sorted(predictions.Date.unique()))
    # Include one final bar so the final dated close decision can execute.
    end_pos = prices.index.get_loc(dates[-1])+1
    bars = prices.loc[dates[0]:prices.index[min(end_pos, len(prices)-1)]]
    definitions = {}
    for model, group in predictions.groupby('model'):
        p = group.set_index('Date').reindex(bars.index)
        for mapping in ['raw', 'calibrated']:
            active = (p[mapping] >= config['probability_threshold']).astype(float)
            definitions[f'{model}_{mapping}'] = (signal_frame(active*config['max_exposure'], p[mapping]), strategy)
        multiplier = (config['target_volatility']/(p.vol_20*np.sqrt(252))).clip(upper=config['max_exposure']).fillna(0)
        active = (p.calibrated >= config['probability_threshold']).astype(float)
        definitions[f'{model}_vol_target'] = (signal_frame(active*multiplier, p.calibrated), strategy)
        fixed_size = p.training_conditional_size.fillna(0).clip(0, config['max_exposure'])
        definitions[f'{model}_fixed_risk_control'] = (signal_frame(active*fixed_size, p.calibrated), strategy)
        # Match *training-estimated* exposure, never match using future test exposure.
        no_brackets = replace(strategy, stop_loss_pct=.999999, take_profit_pct=1e9, max_drawdown_limit=1.)
        definitions[f'{model}_exposure_control'] = (signal_frame(p.training_vol_exposure.fillna(0).clip(0, config['max_exposure'])), no_brackets)
    no_brackets = replace(strategy, stop_loss_pct=.999999, take_profit_pct=1e9, max_drawdown_limit=1.)
    definitions['buy_hold'] = (signal_frame(pd.Series(1., index=bars.index)*config['max_exposure']), no_brackets)
    definitions['trend'] = (signal_frame((frame.price_ma60.reindex(bars.index)>0).astype(float)*config['max_exposure']), strategy)
    summaries, stresses, neighborhoods, strata = [], [], [], []
    result_returns = {}
    for name, (signals, cfg) in definitions.items():
        result = BacktestEngine(cfg, ExecutionConfig()).backtest(signals, bars)
        save_backtest(result, output/'backtests'/name)
        signals.to_csv(output/'backtests'/name/'signals.csv', index_label='Date')
        result_returns[name] = result.daily_returns
        summaries.append({'model': name, **result.metrics,
            'mean_abs_exposure': result.equity_curve.position.abs().mean(),
            'halted': result.account['halted'], 'execution_mode': 'next_open', 'stop_reference': 'entry',
            'evaluation_first': bars.index[0], 'evaluation_last': bars.index[-1]})
        base_positions = result.equity_curve.position.abs()
        for multiple in [1, 2, 4]:
            stress_cfg = replace(cfg, commission=cfg.commission*multiple, slippage=cfg.slippage*multiple)
            stressed = result if multiple == 1 else BacktestEngine(stress_cfg, ExecutionConfig()).backtest(signals, bars)
            stresses.append({'model': name, 'cost_multiplier': multiple, **stressed.metrics})
        if name.endswith('_calibrated') or name == 'trend':
            for stop_m in [.8, 1., 1.2]:
                for take_m in [.8, 1., 1.2]:
                    neighbor = BacktestEngine(replace(cfg, stop_loss_pct=cfg.stop_loss_pct*stop_m,
                        take_profit_pct=cfg.take_profit_pct*take_m), ExecutionConfig()).backtest(signals, bars)
                    neighborhoods.append({'model': name, 'stop_multiplier': stop_m, 'take_multiplier': take_m, **neighbor.metrics})
        for year, subset in result.daily_returns.groupby(result.daily_returns.index.year):
            strata.append({'model': name, 'stratum': str(year), 'n': len(subset),
                'compounded_return': (1+subset).prod()-1, 'mean_daily_return': subset.mean(),
                'mean_abs_exposure': base_positions.loc[subset.index].mean()})
        # Causal volatility states use each day's prior 252-session median only.
        vol = frame.vol_20.shift().reindex(bars.index)
        prior = frame.vol_20.rolling(252, min_periods=60).median().shift(2).reindex(bars.index)
        for label, mask in [('high_vol', vol>prior), ('low_vol', vol<=prior)]:
            subset = result.daily_returns.loc[mask.fillna(False)]
            strata.append({'model': name, 'stratum': label, 'n': len(subset),
                'compounded_return': None, 'mean_daily_return': subset.mean(),
                'mean_abs_exposure': base_positions.loc[subset.index].mean()})
    pd.DataFrame(summaries).to_csv(output/'strategy_summary.csv', index=False)
    pd.DataFrame(stresses).to_csv(output/'cost_stress.csv', index=False)
    pd.DataFrame(neighborhoods).to_csv(output/'parameter_neighborhood.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'stratified_results.csv', index=False)
    returns = pd.DataFrame(result_returns)
    returns.to_csv(output/'daily_returns.csv', index_label='Date')
    intervals = []
    for name in returns:
        if name == 'buy_hold':
            continue
        delta = returns[name]-returns.buy_hold
        intervals.append({'model': name, 'comparison': 'daily_net_return_minus_buy_hold',
            **block_mean_interval(delta, config['bootstrap_repetitions'], config['bootstrap_block_sessions'], config['seed'])})
    pd.DataFrame(intervals).to_csv(output/'block_uncertainty.csv', index=False)


def calibration_report(predictions, config, output):
    rows, reliability, uncertainty = [], [], []
    for model, group in predictions.groupby('model'):
        valid = group.dropna(subset=['target_direction'])
        y = valid.target_direction.astype(int)
        for mapping in ['raw', 'shrink', 'sigmoid', 'calibrated', 'base_rate']:
            scores = probability_scores(y, valid[mapping])
            active = valid[mapping] >= config['probability_threshold']
            subset_scores = probability_scores(y[active], valid.loc[active, mapping]) if active.any() else {'brier': None, 'log_loss': None, 'n': 0}
            rows.append({'model': model, 'mapping': mapping, **scores, **{f'active_{k}':v for k,v in subset_scores.items()}})
            bins = pd.cut(valid[mapping], np.linspace(0, 1, 11), include_lowest=True)
            for bucket, index in valid.groupby(bins, observed=True).groups.items():
                subset = valid.loc[index]
                reliability.append({'model': model, 'mapping': mapping, 'bin': str(bucket),
                    'predicted': subset[mapping].mean(), 'observed': subset.target_direction.mean(), 'n': len(subset)})
            difference = (valid[mapping]-y)**2-(valid.base_rate-y)**2
            uncertainty.append({'model': model, 'mapping': mapping,
                **block_mean_interval(difference, config['bootstrap_repetitions'], config['bootstrap_block_sessions'], config['seed'])})
    pd.DataFrame(rows).to_csv(output/'probability_scores.csv', index=False)
    pd.DataFrame(reliability).to_csv(output/'reliability.csv', index=False)
    pd.DataFrame(uncertainty).to_csv(output/'probability_uncertainty.csv', index=False)


def news_eligibility(baseline_dir, output):
    path = baseline_dir/'event_overlay_backtest/classified_events_with_labels.csv' if baseline_dir else None
    if path is None or not path.exists():
        result = {'status': 'insufficient_point_in_time_sample', 'events': 0, 'serious_backtest_ready': False,
            'reason': 'No point-in-time historical event store supplied; synthetic examples do not invent news evidence.'}
    else:
        events = pd.read_csv(path)
        timestamp_col = next((c for c in ['published_at_utc', 'timestamp', 'published_at'] if c in events), None)
        dates = pd.to_datetime(events[timestamp_col], utc=True, errors='coerce') if timestamp_col else pd.Series(dtype='datetime64[ns, UTC]')
        months = (dates.max()-dates.min()).days/30.4375 if dates.notna().any() else 0
        available_col = next((c for c in ['available_at', 'available_at_utc'] if c in events), None)
        basis_col = next((c for c in ['availability_basis', 'available_at_basis'] if c in events), None)
        has_available = bool(available_col and basis_col
            and pd.to_datetime(events[available_col], utc=True, errors='coerce').notna().all()
            and events[basis_col].isin(['verified_ingestion_log', 'verified_vendor_release']).all())
        counts = events.source.value_counts(normalize=True) if 'source' in events else pd.Series(dtype=float)
        concentration = float(counts.max()) if len(counts) else None
        reasons = []
        if len(events) < 250: reasons.append('Fewer than 250 events; minimum gate is not a statistical-power guarantee.')
        if months < 24: reasons.append('Less than 24 months coverage.')
        if not has_available: reasons.append('Historical availability provenance is missing; publication date alone does not establish ingestion availability.')
        if concentration is None or concentration > .75: reasons.append('Source concentration missing or above 75%.')
        result = {'status': 'insufficient_point_in_time_sample' if reasons else 'eligible_requires_preregistered_ablation',
            'events': len(events), 'months': months, 'source_max_share': concentration,
            'available_at_verified': has_available, 'serious_backtest_ready': not reasons,
            'reasons': reasons, 'source_file': str(path), 'source_sha256': sha256(path),
            'conclusion': 'News remains explanatory context. No incremental alpha claim.'}
    write_json(output/'news_eligibility.json', result)
    return result


def run(config_path, prices_path, output, *, baseline_dir=None, cache_dir=None, resume=False, synthetic=False):
    config_path, prices_path, output = Path(config_path), Path(prices_path), Path(output)
    config = json.loads(config_path.read_text())
    validate_config(config)
    if synthetic and 'synthetic' not in config['evaluation_status']:
        raise ValueError('Synthetic prices require a synthetic-labelled experiment configuration')
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices = validate_ohlcv(prices, strict_calendar=True)
    state = code_state(ROOT)
    identity = {'source_sha256': state['source_sha256'], 'input_sha256': sha256(prices_path),
        'config_sha256': sha256(config_path), 'dependencies': dependencies(), 'synthetic': synthetic,
        'baseline_sha256': directory_hash(baseline_dir), 'cache_sha256': directory_hash(cache_dir, '*.csv')}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or any(manifest.get(k) != v for k,v in identity.items()):
            raise ValueError('Existing run is immutable; use a new output directory, or --resume with identical code/data/config')
        for status in manifest.get('stages', {}).values():
            if status.get('status') == 'complete':
                for rel, digest in status.get('artifacts', {}).items():
                    path = output/rel
                    if not path.is_file() or sha256(path) != digest:
                        raise ValueError(f'Resume artifact integrity failed: {rel}')
    else:
        manifest = {**identity, 'started_at': utc_now(), 'status': 'registered', 'config': config,
            'code': state, 'input_path': str(prices_path.resolve()),
            'data_first': prices.index[0], 'data_last': prices.index[-1], 'data_rows': len(prices),
            'data_quality': prices.attrs['data_quality'], 'stages': {},
            'synthetic': synthetic, 'prediction_target': 'next_session_open_to_following_session_open',
            'decision_clock': 'XNYS session close; theoretical immediate availability, no vendor SLA',
            'execution_clock': 'next observed XNYS session open',
            'price_adjustment': 'consistent adjusted OHLC assumed from input; cache vendor lineage not independently certified',
            'holdout_status': config['evaluation_status'], 'thread_limit': 1}
        write_json(manifest_path, manifest)  # Register before training or evaluating.
    lock = output/'.running'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Run lock exists; verify the recorded process before removing a stale lock') from exc
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    def stage(name, action):
        if manifest['stages'].get(name, {}).get('status') == 'complete':
            return
        manifest['stages'][name] = {'status': 'running', 'started_at': utc_now()}
        write_json(manifest_path, manifest)
        print(f'Running {name}', flush=True)
        action()
        after = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
            if p.is_file() and p not in {manifest_path, lock}}
        # Hash the full checkpoint, including deterministic rewrites after a
        # failed attempt and nested replay manifests, not only changed files.
        manifest['stages'][name].update(status='complete', completed_at=utc_now(), artifacts=after)
        write_json(manifest_path, manifest)
    try:
        with threadpool_limits(limits=1):
            frame = research_frame(prices)
            def predictions_stage():
                predictions, folds = run_predictions(frame, config)
                predictions.to_csv(output/'predictions.csv', index=False)
                write_json(output/'folds.json', folds)
            stage('predictions', predictions_stage)
            predictions = pd.read_csv(output/'predictions.csv', parse_dates=['Date'])
            stage('probability', lambda: calibration_report(predictions, config, output))
            stage('trading', lambda: trading_experiment(prices, frame, predictions, config, output))
            def distribution_stage():
                windows, summary = distribution_experiment(prices.Close, config)
                windows.to_csv(output/'distribution_windows.csv', index=False)
                summary.to_csv(output/'distribution_summary.csv', index=False)
            stage('distributions', distribution_stage)
            stage('news', lambda: news_eligibility(baseline_dir, output))
            if baseline_dir and cache_dir:
                from nvda_quant_model.research.replay import replay_historical
                stage('historical_replay', lambda: replay_historical(baseline_dir, cache_dir, output/'historical_replay'))
            from nvda_quant_model.research.reporting import build_report
            stage('report', lambda: build_report(output))
        manifest.update(status='complete', completed_at=utc_now())
        manifest['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p not in {manifest_path, lock}}
        write_json(manifest_path, manifest)
        index = output.parent/'run_index.jsonl'
        with index.open('a') as f:
            f.write(json.dumps({'run': output.name, 'status': 'complete', **identity, 'manifest_sha256': sha256(manifest_path)})+'\n')
        print(f'Complete: {output}', flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', failed_at=utc_now())
        write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'research/configs/smoke.json')
    parser.add_argument('--prices', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        from nvda_quant_model.research.sample import write_sample
        path = args.prices or ROOT/'research/sample/synthetic_ohlcv.csv'
        if not path.exists(): write_sample(path)
    elif args.prices is None:
        parser.error('--prices is required unless --synthetic is set')
    else:
        path = args.prices
    run(args.config, path, args.output, baseline_dir=args.baseline_dir, cache_dir=args.cache_dir,
        resume=args.resume, synthetic=args.synthetic)


if __name__ == '__main__':
    main()
