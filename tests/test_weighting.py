import pandas as pd

from analysis.weighting import assert_small_sample_weight_bound, build_weight_sets


def test_small_sample_event_cannot_dominate_large_sample_event():
    events = pd.DataFrame(
        [
            {"event_id": "small", "event_name": "small", "clean_samples": 6, "importance": 5, "macro_no_surprise": False},
            {"event_id": "large", "event_name": "large", "clean_samples": 150, "importance": 5, "macro_no_surprise": False},
        ]
    )
    post = pd.DataFrame(
        [
            {"event_id": "small", "posterior_mean": 0.05, "posterior_sd": 0.02},
            {"event_id": "large", "posterior_mean": 0.01, "posterior_sd": 0.02},
        ]
    )
    signals = pd.DataFrame(
        [
            {"event_id": "small", "asset": "SPY", "subset": "clean", "window": "post_1_5", "p_value_holm": 0.01},
            {"event_id": "large", "asset": "SPY", "subset": "clean", "window": "post_1_5", "p_value_holm": 0.01},
        ]
    )
    weights = build_weight_sets(events, post, signals, {"weight_formula": {"economic_signal_cap": 0.05, "small_sample_penalty": {"n_lt_10": 0.3, "n_lt_30": 0.6}}})
    assert assert_small_sample_weight_bound(weights, bound=1.5)

