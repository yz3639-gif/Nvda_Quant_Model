import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nvda_quant_model.research.provenance import block_mean_interval, write_json
from nvda_quant_model.research.runner import news_eligibility, validate_config
from nvda_quant_model.research.distributions import interval_score


def test_invalid_registered_budget_rejected_before_training():
    config = json.loads((Path(__file__).resolve().parents[1]/'research/configs/smoke.json').read_text())
    config['candidate_budget'] = 100
    with pytest.raises(ValueError, match='budget'):
        validate_config(config)


def test_interval_score_penalizes_misses_and_unnecessary_width():
    assert interval_score(.2, -.1, .1, .8) > interval_score(.05, -.1, .1, .8)
    assert interval_score(0, -.5, .5, .8) > interval_score(0, -.1, .1, .8)


def test_strict_json_and_deterministic_block_bootstrap(tmp_path):
    write_json(tmp_path/'data.json', {'missing': np.nan, 'infinite': np.inf, 'n': np.int64(3)})
    assert json.loads((tmp_path/'data.json').read_text()) == {'missing': None, 'infinite': None, 'n': 3}
    x = np.sin(np.arange(120))
    assert block_mean_interval(x, seed=5) == block_mean_interval(x, seed=5)


def test_news_publication_assumption_is_not_verified_history(tmp_path):
    directory = tmp_path/'event_overlay_backtest'; directory.mkdir()
    timestamps = pd.date_range('2019-01-01', periods=300, freq='4D', tz='UTC')
    pd.DataFrame({'timestamp': timestamps, 'available_at': timestamps,
        'availability_basis': 'publication_time_assumption', 'source': ['a', 'b']*150}).to_csv(directory/'classified_events_with_labels.csv', index=False)
    result = news_eligibility(tmp_path, tmp_path)
    assert result['serious_backtest_ready'] is False
    assert result['available_at_verified'] is False
