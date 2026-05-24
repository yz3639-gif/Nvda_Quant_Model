import numpy as np
import pandas as pd

from analysis.sentiment_filter import attach_environment_distances, build_environment_vector, match_historical_environment


def _snapshot(metrics: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for metric, (value, z_score) in metrics.items():
        rows.append(
            {
                "metric": metric,
                "value": value,
                "percentile": np.nan,
                "z_score": z_score,
                "status": "available",
                "source": "test",
                "extreme_flag": "",
            }
        )
    return pd.DataFrame(rows)


def test_three_available_dimensions_enable_matching():
    current = build_environment_vector(_snapshot({"VIX": (18, 0.2), "HY_OAS": (3.2, -0.1), "AAII_bull_bear_spread": (7, 0.4)}))
    historical = pd.DataFrame(
        {
            "VIX": [0.1, 2.0, -1.0],
            "HY_OAS": [-0.2, 1.0, -1.5],
            "AAII_bull_bear_spread": [0.3, -2.0, 1.0],
        },
        index=["sample_a", "sample_b", "sample_c"],
    )

    matched, meta = match_historical_environment(current, historical, min_dims_required=3, top_k_pct=0.5)

    assert meta["status"] == "success"
    assert matched
    assert meta["n_dimensions"] == 3


def test_two_available_dimensions_disable_matching():
    current = build_environment_vector(_snapshot({"VIX": (18, 0.2), "HY_OAS": (3.2, -0.1)}))
    matched, meta = match_historical_environment(current, pd.DataFrame({"VIX": [0.0], "HY_OAS": [0.0]}), min_dims_required=3)

    assert matched == []
    assert meta["status"] == "insufficient_dimensions"
    assert meta["min_required"] == 3


def test_nan_historical_dimensions_are_ignored_per_sample():
    current = build_environment_vector(_snapshot({"VIX": (18, 0.0), "HY_OAS": (3.2, 0.0), "NAAIM_exposure": (80, 0.0)}))
    historical = pd.DataFrame(
        {
            "VIX": [0.1, 1.5],
            "HY_OAS": [0.2, np.nan],
            "NAAIM_exposure": [0.3, 0.1],
        },
        index=["enough_dims", "too_many_nans"],
    )

    matched, meta = match_historical_environment(current, historical, min_dims_required=3)

    assert meta["status"] == "success"
    assert matched == ["enough_dims"]


def test_1972_sample_with_only_two_dimensions_is_handled_without_error():
    dates = pd.to_datetime(["1972-02-18", "2026-05-08"])
    sentiment_history = pd.DataFrame(
        {
            "VIX": [18.0, 20.0],
            "HY_OAS": [3.0, 4.0],
            "AAII_bull_bear_spread": [np.nan, 5.0],
        },
        index=dates,
    )
    samples = pd.DataFrame(
        [{"sample_id": "nixon_1972", "date": "1972-02-21", "excluded_noise": False}]
    )
    snapshot = _snapshot({"VIX": (20.0, 1.0), "HY_OAS": (4.0, 1.0), "AAII_bull_bear_spread": (5.0, 1.0)})

    out = attach_environment_distances(samples, sentiment_history, pd.Timestamp("2026-05-10"), snapshot_df=snapshot, min_dims_required=3)

    assert "environment_match" in out
    assert not bool(out["environment_match"].iloc[0])
    assert out.attrs["env_match_metadata"]["status"] in {"insufficient_historical_overlap", "insufficient_dimensions"}
