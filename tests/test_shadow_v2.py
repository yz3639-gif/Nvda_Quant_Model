"""Clock and immutability tests are simulations, never prospective evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.data.session_calendar import session_dates, session_schedule
from nvda_quant_model.research.provenance import sha256
from nvda_quant_model.research.shadow import ShadowConfig, attach_outcome, main, record_forecast, register_model


DECISION = pd.Timestamp('2025-02-10T21:05:00Z')


def write_prices(directory, prices, available_at, data_kind='synthetic', name='input'):
    path = directory / f'{name}.csv'
    prices.to_csv(path, index_label='Date')
    available = pd.Timestamp(available_at)
    provenance = {'prices_sha256': sha256(path), 'data_kind': data_kind, 'source': 'offline synthetic test fixture',
                  'price_adjustment': 'synthetic_consistent_ohlc', 'available_at': available.isoformat(),
                  'retrieved_at': (available + pd.Timedelta(seconds=5)).isoformat()}
    metadata = directory / f'{name}.json'
    metadata.write_text(json.dumps(provenance))
    return path, metadata


@pytest.fixture
def setup(tmp_path):
    dates = session_dates('2024-05-01', '2025-02-10')
    rng = np.random.default_rng(41)
    opening = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, len(dates))))
    closing = opening * np.exp(rng.normal(0, .005, len(dates)))
    prices = pd.DataFrame({'Open': opening, 'High': np.maximum(opening, closing)*1.01,
                           'Low': np.minimum(opening, closing)*.99, 'Close': closing,
                           'Volume': rng.integers(100_000, 1_000_000, len(dates))}, index=dates)
    config = ShadowConfig('fixed_rule', 'rule', None, train_sessions=60)
    registration = register_model(config, tmp_path/'store', simulation=True, now=DECISION-pd.Timedelta(days=1))
    path, provenance = write_prices(tmp_path, prices, '2025-02-10T21:01:00Z')
    return tmp_path, prices, config, registration, path, provenance


def test_shadow_forecast_uses_mature_labels_and_creates_immutable_record(setup):
    directory, prices, config, registration, path, provenance = setup
    record_path = record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    record = json.loads(record_path.read_text())
    assert record['evidence_status'] == 'simulation_not_prospective'
    assert record['training']['n'] == 60
    assert pd.Timestamp(record['training']['max_label_end_at']) <= DECISION
    assert pd.Timestamp(record['available_at']) <= pd.Timestamp(record['fit_at']) <= pd.Timestamp(record['created_at'])
    assert record['fill_at'] == '2025-02-11T14:30:00+00:00'
    assert record['label_end_at'] == '2025-02-12T14:30:00+00:00'
    assert record['orders_sent'] is False
    assert 0 < record['prob_up'] < 1
    before = record_path.read_bytes()
    assert record_forecast(path, registration, provenance, simulation=True, now=DECISION+pd.Timedelta(minutes=1)) == record_path
    assert record_path.read_bytes() == before
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc('Volume')] += 1
    path, provenance = write_prices(directory, changed, '2025-02-10T21:01:00Z')
    with pytest.raises(FileExistsError, match='historical records cannot be overwritten'):
        record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    assert record_path.read_bytes() == before


def test_shadow_rejects_stale_and_future_daily_prices(setup):
    directory, prices, config, registration, path, provenance = setup
    stale, metadata = write_prices(directory, prices.iloc[:-1], '2025-02-10T21:01:00Z', name='stale')
    with pytest.raises(ValueError, match='Stale or future'):
        record_forecast(stale, registration, metadata, simulation=True, now=DECISION)
    # A completed next-session bar cannot have been supplied at yesterday's decision.
    future = pd.concat([prices, prices.iloc[-1:].set_axis(pd.DatetimeIndex(['2025-02-11']))])
    future_path, metadata = write_prices(directory, future, '2025-02-10T21:01:00Z', name='future')
    with pytest.raises(ValueError, match='before its final session close'):
        record_forecast(future_path, registration, metadata, simulation=True, now=DECISION)
    assert not (registration.parent/'forecasts').exists()


def test_shadow_rejects_recording_after_intended_next_open(setup):
    _, _, _, registration, path, provenance = setup
    with pytest.raises(ValueError, match='open has passed'):
        record_forecast(path, registration, provenance, simulation=True, now='2025-02-11T14:30:00Z')
    assert not (registration.parent/'forecasts').exists()


def test_shadow_rechecks_deadline_after_fit(setup, monkeypatch):
    _, _, _, registration, path, provenance = setup
    import nvda_quant_model.research.shadow as module
    ticks = iter([DECISION, pd.Timestamp('2025-02-11T14:30:01Z')])
    monkeypatch.setattr(module, '_clock', lambda now, simulation: next(ticks))
    with pytest.raises(ValueError, match='Fitting finished after'):
        record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    assert not (registration.parent/'forecasts').exists()


def test_injected_clock_and_synthetic_prices_cannot_claim_prospective_evidence(setup, monkeypatch):
    directory, _, config, _, path, provenance = setup
    with pytest.raises(ValueError, match='Injected now requires simulation'):
        register_model(config, directory/'forbidden', now=DECISION)
    import nvda_quant_model.research.shadow as module
    # Controlled unit-test clock, not a CLI option or actual evidence record.
    monkeypatch.setattr(module, '_clock', lambda now, simulation: DECISION)
    registration = register_model(config, directory/'synthetic_rejection')
    with pytest.raises(ValueError, match='Synthetic data cannot enter prospective'):
        record_forecast(path, registration, provenance)
    assert not (registration.parent/'forecasts').exists()


def test_registration_and_input_hashes_are_immutable(setup):
    directory, _, config, registration, path, provenance = setup
    with pytest.raises(FileExistsError, match='Model registration is immutable'):
        register_model(ShadowConfig(config.model_id, 'logistic', 1.0, train_sessions=60),
                       directory/'store', simulation=True, now=DECISION)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='prices_sha256'):
        record_forecast(path, registration, provenance, simulation=True, now=DECISION)


def test_outcome_only_after_maturity_and_never_changes_forecast(setup):
    directory, prices, _, registration, path, provenance = setup
    forecast = record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    before = forecast.read_bytes()
    with pytest.raises(ValueError, match='has not matured'):
        attach_outcome(forecast, path, provenance, simulation=True, now='2025-02-12T14:29:59Z')
    rows = pd.DataFrame({'Open': [100., 102.], 'High': [103., 104.], 'Low': [99., 101.],
                         'Close': [102., 103.], 'Volume': [100_000, 100_000]},
                        index=pd.DatetimeIndex(['2025-02-11', '2025-02-12']))
    mature, metadata = write_prices(directory, rows, '2025-02-12T21:01:00Z', name='outcomes')
    outcome_path = attach_outcome(forecast, mature, metadata, simulation=True, now='2025-02-12T21:05:00Z')
    outcome = json.loads(outcome_path.read_text())
    assert outcome['target_return'] == pytest.approx(.02)
    assert outcome['target_direction'] == 1
    assert outcome['identity']['forecast_sha256'] == json.loads(before)['record_sha256']
    assert forecast.read_bytes() == before
    assert attach_outcome(forecast, mature, metadata, simulation=True, now='2025-02-12T21:06:00Z') == outcome_path
    rows.loc[pd.Timestamp('2025-02-12'), 'Open'] = 103
    mature, metadata = write_prices(directory, rows, '2025-02-12T21:01:00Z', name='outcomes')
    with pytest.raises(FileExistsError, match='Outcome is immutable'):
        attach_outcome(forecast, mature, metadata, simulation=True, now='2025-02-12T21:07:00Z')
    assert forecast.read_bytes() == before


def test_calendar_clock_handles_early_close_weekend_and_dst():
    from nvda_quant_model.research.shadow import _market_clock
    session, closed, entry, exit_session = _market_clock(pd.Timestamp('2025-11-28T18:05:00Z'))
    assert str(session.date()) == '2025-11-28'
    assert closed.market_close.hour == 18  # Friday after Thanksgiving early close.
    assert entry.market_open == pd.Timestamp('2025-12-01T14:30:00Z')
    _, _, entry, _ = _market_clock(pd.Timestamp('2025-03-07T22:00:00Z'))
    assert entry.market_open == pd.Timestamp('2025-03-10T13:30:00Z')


def test_cli_does_not_expose_backdated_now_option():
    with pytest.raises(SystemExit) as exc:
        main(['register', '--config', 'unused.json', '--store', 'unused', '--now', '2025-02-10T21:05:00Z'])
    assert exc.value.code == 2


def test_shadow_rejects_tampered_snapshot_without_rewriting_forecast(setup):
    _, _, _, registration, path, provenance = setup
    forecast = record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    original = forecast.read_bytes()
    snapshot = Path(json.loads(original)['prices_snapshot'])
    snapshot.write_text(snapshot.read_text()+'\n')
    with pytest.raises(ValueError, match='snapshot was modified'):
        record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    assert forecast.read_bytes() == original


def test_logistic_model_is_fixed_and_produces_finite_shadow_probability(setup):
    directory, _, _, _, path, provenance = setup
    registration = register_model(ShadowConfig('fixed_logistic', 'logistic', .2, train_sessions=60),
                                  directory/'store', simulation=True, now=DECISION-pd.Timedelta(days=1))
    forecast = record_forecast(path, registration, provenance, simulation=True, now=DECISION)
    result = json.loads(forecast.read_text())
    assert result['model']['parameter'] == .2
    assert np.isfinite(result['prob_up'])
    assert result['probability_status'] == 'raw_uncalibrated_fixed_model'
