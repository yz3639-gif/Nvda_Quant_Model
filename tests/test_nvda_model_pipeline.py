from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nvda_quant_model.backtest.backtest_engine import BacktestEngine
from nvda_quant_model.calibration import engine_rolling_calibration, probability_calibration, rolling_interval_calibration
from nvda_quant_model.candidate_promotion import classify_candidate, hard_gate_failures, select_promotion_candidates
from nvda_quant_model.config import MACRO_TICKERS, PROJECT_ROOT, SECTOR_TICKERS, StrategyConfig
from nvda_quant_model.data.feature_engineering import build_model_frame
from nvda_quant_model.data.load_data import load_ohlcv
from nvda_quant_model.high_sample_optimizer import SEARCH_SPACE as HIGH_SAMPLE_SEARCH_SPACE
from nvda_quant_model.high_sample_optimizer import high_sample_quality, high_sample_score
from nvda_quant_model.high_sample_validation import read_live_optimizer_csv, select_high_sample_candidates
from nvda_quant_model.live_order_flow import analyze_order_flow, bars_frame
from nvda_quant_model.long_run_optimizer import SEARCH_SPACE as STRICT_SEARCH_SPACE
from nvda_quant_model.meta_decision_layer import MetaPolicy, build_meta_signals
from nvda_quant_model.models.ml_model import build_candidate_models, load_model_parameter_overrides
from nvda_quant_model.options_volatility import analyze_options
from nvda_quant_model.order_flow_backtest import backtest_intraday_signals
from nvda_quant_model.optimizer_csv import append_rows_schema_safe
from nvda_quant_model.optimizer_monitor import (
    OptimizerSpec,
    is_optimizer_running,
    needs_high_sample_validation,
    optimizer_process_count,
    optimizer_specs,
    validation_contains_label,
)
from nvda_quant_model.precision_search import MOMENTUM_FEATURES
from nvda_quant_model.pipeline import prepare_model_inputs
from nvda_quant_model.reaction_pool import build_reaction_signals, combine_baseline_and_reaction
from nvda_quant_model.reaction_pool_validation import _passes_validation
from nvda_quant_model.risk_off_guard import (
    _passes_production_gate,
    apply_risk_off_guard,
    blocked_trade_diagnostics,
)
from nvda_quant_model.production_model import BASELINE_PRODUCTION_LABEL
from nvda_quant_model.strict_model_selection import rule_from_row
from nvda_quant_model.stable_candidate_optimizer import (
    select_stable_candidates,
    stable_score,
    summarize_stable_validation,
)
from nvda_quant_model.stable_repair_optimizer import (
    baseline_constraint_failures,
    combine_core_and_repair_signals,
    sixty_month_first_score,
    sixty_month_improvement_failures,
)


def _synthetic_prices(periods: int = 520) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    drift = np.linspace(0.0, 0.35, periods)
    cycle = 0.04 * np.sin(np.arange(periods) / 9.0)
    close = pd.Series(100.0 * (1.0 + drift + cycle), index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]) * 1.001,
            "High": close * 1.018,
            "Low": close * 0.982,
            "Close": close,
            "Volume": 20_000_000 + (np.arange(periods) % 17) * 250_000,
        },
        index=index,
    )


def _synthetic_external(index: pd.DatetimeIndex) -> pd.DataFrame:
    trend = np.linspace(0.0, 0.18, len(index))
    cycle = np.sin(np.arange(len(index)) / 11.0)
    columns = {
        MACRO_TICKERS["VIX"]: 17.0 + 3.5 * cycle,
        MACRO_TICKERS["SP500"]: 4500.0 * (1.0 + trend + 0.01 * cycle),
        MACRO_TICKERS["NASDAQ"]: 14500.0 * (1.0 + 1.25 * trend + 0.015 * cycle),
        MACRO_TICKERS["USD"]: 103.0 + 0.4 * cycle,
        MACRO_TICKERS["TREASURY_10Y"]: 42.0 + 1.2 * cycle,
        MACRO_TICKERS["HYG"]: 75.0 * (1.0 + 0.03 * trend),
        MACRO_TICKERS["LQD"]: 108.0 * (1.0 + 0.01 * trend),
    }
    for ticker in SECTOR_TICKERS.values():
        columns[ticker] = 100.0 * (1.0 + 1.4 * trend + 0.012 * cycle)
    return pd.DataFrame(columns, index=index)


def test_backtest_engine_records_take_profit_trade_with_costs() -> None:
    index = pd.bdate_range("2025-01-02", periods=3)
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.5, 105.0],
            "High": [101.0, 106.0, 105.5],
            "Low": [99.0, 100.0, 104.0],
            "Close": [100.0, 105.0, 105.0],
            "Volume": [1_000_000, 1_100_000, 1_050_000],
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "position": [1.0, 1.0, 0.0],
            "direction": [1, 1, 0],
            "confidence": [0.7, 0.7, 0.0],
            "expected_return": [0.01, 0.01, 0.0],
            "prob_up": [0.7, 0.7, 0.0],
            "prob_down": [0.3, 0.3, 0.0],
        },
        index=index,
    )
    result = BacktestEngine(StrategyConfig(initial_capital=100_000.0)).backtest(signals, prices)

    assert result.metrics["final_equity"] > 100_000.0
    assert result.metrics["num_trades"] == 1
    assert result.trades.iloc[0]["exit_reason"] == "take_profit"
    # Exit fee is paid on the actual 105 notional; the initial entry also pays a fee.
    assert np.isclose(result.daily_returns.iloc[1], 0.048425)
    assert result.daily_returns.iloc[0] < 0
    assert np.isclose(result.trades.pnl.sum(), result.metrics['final_equity']-100_000)
    assert result.daily_returns.iloc[2] == 0.0


def test_rule_from_row_applies_audited_nvda_production_override() -> None:
    row = pd.Series(
        {
            "label": BASELINE_PRODUCTION_LABEL,
            "train_window": 147,
            "test_window": 63,
            "momentum_feature": "10d_return",
            "momentum_quantile": 0.60,
            "volume_quantile": 0.45,
            "max_rsi": 75.0,
            "min_price_60ma": 0.96,
            "require_smh_positive": True,
            "require_qqq_positive": False,
            "require_sp500_positive": False,
            "require_macd_positive": False,
            "require_obv_positive": True,
            "require_vol_calm": False,
            "require_peer_breadth_positive": False,
            "require_peer_mean_positive": False,
            "require_peer_event_net_positive": False,
            "require_no_peer_business_stress": False,
            "vix_quantile_cap": 0.90,
            "exclude_negative_pre_holiday": False,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.045,
            "max_exposure": 1.0,
        }
    )

    rule = rule_from_row(row)

    assert rule.stop_loss_pct == 0.025
    assert rule.take_profit_pct == 0.040


def test_backtest_engine_halts_without_fake_reentry_after_drawdown_limit() -> None:
    index = pd.bdate_range("2025-01-02", periods=5)
    prices = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 80.0, 81.0, 82.0],
            "High": [101.0, 100.0, 82.0, 83.0, 84.0],
            "Low": [99.0, 79.0, 79.0, 80.0, 81.0],
            "Close": [100.0, 80.0, 81.0, 82.0, 83.0],
            "Volume": [1_000_000] * 5,
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "position": [1.0, 1.0, 1.0, 1.0, 1.0],
            "direction": [1, 1, 1, 1, 1],
            "confidence": [0.7] * 5,
            "expected_return": [0.01] * 5,
            "prob_up": [0.7] * 5,
            "prob_down": [0.3] * 5,
        },
        index=index,
    )
    cfg = StrategyConfig(
        initial_capital=100_000.0,
        max_drawdown_limit=0.05,
        stop_loss_pct=0.50,
        take_profit_pct=0.50,
    )

    result = BacktestEngine(cfg).backtest(signals, prices)

    assert result.equity_curve["position"].iloc[1:].eq(0.0).all()
    assert result.metrics["num_trades"] == 1
    assert result.trades.iloc[0]["exit_reason"] == "drawdown_stop"


def test_build_model_frame_adds_adaptive_regime_features_without_target_leakage() -> None:
    prices = _synthetic_prices()
    external = _synthetic_external(prices.index)
    start = prices.index[320].strftime("%Y-%m-%d")
    end = prices.index[-2].strftime("%Y-%m-%d")

    frame, feature_columns = build_model_frame(
        prices,
        external,
        start,
        end,
        include_fundamentals=False,
        peer_ohlcv={},
    )

    for column in [
        "regime_high_vol",
        "regime_low_vol",
        "regime_uptrend",
        "adaptive_momentum_score",
        "adaptive_rsi_signal",
        "adaptive_risk_budget",
    ]:
        assert column in frame.columns
        assert column in feature_columns
        # Raw feature construction must not fit medians across the full frame.
        from nvda_quant_model.data.data_validation import TrainingPreprocessor
        fitted = TrainingPreprocessor(feature_columns).fit(frame.iloc[:30])
        assert fitted.transform(frame)[column].notna().all()

    sample_date = prices.index[360]
    expected_next_return = prices["Close"].pct_change().shift(-1).loc[sample_date]
    assert frame.loc[sample_date, "target_return"] == expected_next_return


def test_build_model_frame_adds_bollinger_and_ema_features() -> None:
    prices = _synthetic_prices()
    external = _synthetic_external(prices.index)
    start = prices.index[320].strftime("%Y-%m-%d")
    end = prices.index[-2].strftime("%Y-%m-%d")

    frame, feature_columns = build_model_frame(
        prices,
        external,
        start,
        end,
        include_fundamentals=False,
        peer_ohlcv={},
    )

    expected_columns = [
        "ema_10_ratio",
        "ema_20_ratio",
        "ema_50_ratio",
        "ema_10_20_spread",
        "ema_20_50_spread",
        "boll_upper_distance_20",
        "boll_lower_distance_20",
        "boll_bandwidth_20",
        "boll_percent_b_20",
        "boll_zscore_20",
    ]
    for column in expected_columns:
        assert column in frame.columns
        assert column in feature_columns
        assert np.isfinite(frame[column]).all()

    assert frame["boll_bandwidth_20"].gt(0).all()


def test_optimizers_share_bollinger_and_ema_momentum_search_space() -> None:
    expected = {
        "ema_10_20_spread",
        "ema_20_50_spread",
        "boll_percent_b_20",
        "boll_zscore_20",
    }

    assert expected.issubset(MOMENTUM_FEATURES)
    assert STRICT_SEARCH_SPACE["momentum_feature"] == MOMENTUM_FEATURES
    assert HIGH_SAMPLE_SEARCH_SPACE["momentum_feature"] == MOMENTUM_FEATURES


def test_load_ohlcv_recovers_from_corrupt_cache(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "NVDA_2025-01-01_2025-01-04.csv"
    cache_file.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2025-01-01,1,2,0.5,1.5,100\n"
        "2025-01-02,1,2,0.5,1.5,100,extra\n",
        encoding="utf-8",
    )
    downloaded = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [1000, 1100],
        },
        index=pd.DatetimeIndex(["2025-01-01", "2025-01-02"]),
    )

    monkeypatch.setattr("nvda_quant_model.data.load_data.yf.download", lambda *args, **kwargs: downloaded)

    data = load_ohlcv("NVDA", "2025-01-01", "2025-01-03", cache_dir=tmp_path)

    assert data["Close"].tolist() == [10.5, 11.5]
    assert cache_file.exists()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_prepare_model_inputs_delegates_data_loading_and_builds_status(monkeypatch) -> None:
    prices = _synthetic_prices(180)
    external = _synthetic_external(prices.index)

    def fake_resolve_data_window(*args, **kwargs):
        return "2024-03-01", "2024-09-06", {
            "requested_start": "auto",
            "requested_end": "latest",
            "start_source": "test",
            "end_source": "test",
            "latest_daily_close": float(prices["Close"].iloc[-1]),
        }

    monkeypatch.setattr("nvda_quant_model.pipeline.resolve_data_window", fake_resolve_data_window)
    monkeypatch.setattr("nvda_quant_model.pipeline.load_market_data", lambda *args, **kwargs: (prices, external))
    monkeypatch.setattr("nvda_quant_model.pipeline.load_peer_ohlcv_panel", lambda *args, **kwargs: {})

    inputs = prepare_model_inputs(
        StrategyConfig(start_date="auto", end_date="latest", include_peer_events=False, include_fundamentals=False)
    )

    assert inputs.config.start_date == "2024-03-01"
    assert inputs.config.end_date == "2024-09-06"
    assert inputs.report_prices.index.min() >= pd.Timestamp("2024-03-01")
    assert "adaptive_risk_budget" in inputs.feature_columns
    assert inputs.data_status["status"] == "FRESH"


def test_model_parameter_overrides_apply_estimator_params_and_top_k(tmp_path) -> None:
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps(
            {
                "model_params": {
                    "RandomForest": {
                        "classifier": {"n_estimators": 12, "max_depth": 2},
                        "regressor": {"n_estimators": 14, "max_depth": 3},
                        "top_k": {"value": 7},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_model_parameter_overrides(str(path))
    random_forest = next(
        model for model in build_candidate_models(top_k=12, random_state=123, parameter_overrides=overrides)
        if model.name == "RandomForest"
    )

    assert random_forest.classifier.n_estimators == 12
    assert random_forest.classifier.max_depth == 2
    assert random_forest.regressor.n_estimators == 14
    assert random_forest.regressor.max_depth == 3
    assert random_forest.selector.top_k == 7


def test_candidate_promotion_selects_baseline_and_strict_buckets() -> None:
    rows = pd.DataFrame(
        [
            {
                "label": "baseline",
                "long_score": 8.0,
                "bt_annualized_return": 0.22,
                "bt_sharpe_ratio": 1.8,
                "bt_max_drawdown": -0.06,
                "bt_win_rate": 0.66,
                "bt_profit_factor": 3.0,
                "bt_num_trades": 31,
                "dir_precision": 0.68,
                "dir_active_days": 45,
                "dir_worst_year_precision": 0.62,
            },
            {
                "label": "high_score",
                "long_score": 9.0,
                "bt_annualized_return": 0.24,
                "bt_sharpe_ratio": 2.0,
                "bt_max_drawdown": -0.05,
                "bt_win_rate": 0.70,
                "bt_profit_factor": 4.0,
                "bt_num_trades": 32,
                "dir_precision": 0.72,
                "dir_active_days": 50,
                "dir_worst_year_precision": 0.66,
            },
        ]
    )

    selected, metadata = select_promotion_candidates(rows, baseline_label="baseline", top_per_bucket=1)

    assert metadata["baseline_found"] is True
    assert metadata["strict30_count"] == 2
    assert set(selected["label"]) == {"baseline", "high_score"}
    assert "current_baseline" in selected.set_index("label").loc["baseline", "roles"]


def test_candidate_promotion_classifies_stable_improvement() -> None:
    baseline = pd.DataFrame(
        [
            {
                "label": "baseline",
                "lookback_months": 24,
                "annualized_return": 0.20,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "win_rate": 0.62,
                "profit_factor": 2.5,
                "num_trades": 35,
            }
        ]
    )
    candidate = pd.DataFrame(
        [
            {
                "label": "candidate",
                "lookback_months": 24,
                "annualized_return": 0.24,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.07,
                "win_rate": 0.66,
                "profit_factor": 2.6,
                "num_trades": 36,
            },
            {
                "label": "candidate",
                "lookback_months": 36,
                "annualized_return": 0.16,
                "sharpe_ratio": 1.3,
                "max_drawdown": -0.10,
                "win_rate": 0.60,
                "profit_factor": 2.1,
                "num_trades": 50,
            },
            {
                "label": "candidate",
                "lookback_months": 60,
                "annualized_return": 0.16,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.18,
                "win_rate": 0.56,
                "profit_factor": 1.7,
                "num_trades": 90,
                "dir_active_days": 130,
            },
        ]
    )

    assert classify_candidate(candidate, baseline) == "promote_candidate"


def test_candidate_promotion_rejects_weak_60m_window() -> None:
    candidate = pd.DataFrame(
        [
            {
                "label": "candidate",
                "lookback_months": 24,
                "annualized_return": 0.24,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.07,
                "win_rate": 0.66,
                "profit_factor": 2.6,
                "num_trades": 36,
            },
            {
                "label": "candidate",
                "lookback_months": 36,
                "annualized_return": 0.18,
                "sharpe_ratio": 1.3,
                "max_drawdown": -0.10,
                "win_rate": 0.60,
                "profit_factor": 2.1,
                "num_trades": 50,
            },
            {
                "label": "candidate",
                "lookback_months": 60,
                "annualized_return": 0.03,
                "sharpe_ratio": 0.4,
                "max_drawdown": -0.21,
                "win_rate": 0.53,
                "profit_factor": 1.4,
                "num_trades": 30,
                "dir_active_days": 70,
            },
        ]
    )

    assert hard_gate_failures(candidate.iloc[-1], enforce_long_sample=True)
    assert classify_candidate(candidate, pd.DataFrame()) == "reject_long_window_failure"


def test_probability_calibration_scores_active_nvda_signals() -> None:
    index = pd.bdate_range("2026-01-02", periods=6)
    signals = pd.DataFrame(
        {
            "position": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "prob_up": [0.70, 0.70, 0.50, 0.60, 0.60, 0.50],
        },
        index=index,
    )
    frame = pd.DataFrame({"target_direction": [1, 0, 1, 1, 1, 0]}, index=index)

    calibration = probability_calibration(signals, frame, bins=4)

    assert calibration["samples"] == 4
    assert 0.0 <= calibration["brier_score"] <= 1.0
    assert calibration["realized_up"] == 0.75
    assert calibration["bins"]


def test_rolling_interval_calibration_reports_nvda_coverage() -> None:
    index = pd.bdate_range("2024-01-02", periods=340)
    returns = 0.001 + 0.012 * np.sin(np.arange(len(index)) / 12.0)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index)

    summary, windows = rolling_interval_calibration(close, horizon_days=5, train_window=120, step=20)

    assert not windows.empty
    assert set(summary["confidence_level"]) == {0.80, 0.95}
    assert summary["coverage"].between(0.0, 1.0).all()


def test_engine_rolling_calibration_reports_pit_and_interval_coverage() -> None:
    index = pd.bdate_range("2020-01-02", periods=420)
    returns = 0.0006 + 0.018 * np.sin(np.arange(len(index)) / 15.0)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index)

    summary, windows = engine_rolling_calibration(
        close,
        horizon_days=10,
        train_window_years=1.0,
        step=20,
        n_sims=750,
        seed=7,
    )

    assert not windows.empty
    assert summary["pit_mean"].between(0.0, 1.0).all()
    assert set(summary["confidence_level"]) == {0.50, 0.80, 0.95}
    assert set(summary["method"]) == {"historical_bootstrap", "gbm_normal", "gbm_student_t"}
    assert set(windows["method"]) == {"historical_bootstrap", "gbm_normal", "gbm_student_t"}
    assert {"actual_simple_return", "simulated_mean_return", "engine_source"}.issubset(windows.columns)
    assert summary["actual_coverage"].between(0.0, 1.0).all()
    assert {"coverage_hac_se", "coverage_z_hac", "pit_ks_pvalue", "hac_lags"}.issubset(summary.columns)
    assert summary["coverage_hac_se"].notna().all()


def test_high_sample_optimizer_requires_trade_count_and_quality_gates() -> None:
    class Args:
        min_trades = 60
        target_trades = 90
        min_active_days = 90
        target_active_days = 130
        min_precision = 0.58
        min_worst_year_precision = 0.52
        min_sharpe = 1.0
        min_annualized = 0.10
        max_drawdown = 0.20
        min_win_rate = 0.55
        min_profit_factor = 1.5

    row = {
        "bt_num_trades": 64,
        "dir_active_days": 103,
        "dir_precision": 0.592,
        "dir_worst_year_precision": 0.55,
        "bt_sharpe_ratio": 1.12,
        "bt_annualized_return": 0.13,
        "bt_max_drawdown": -0.09,
        "bt_win_rate": 0.57,
        "bt_profit_factor": 1.8,
    }
    periods = pd.DataFrame({"precision": [0.60, 0.55, 0.62, 0.52]})

    quality = high_sample_quality(row, Args)

    assert quality["sample_gate"] is True
    assert quality["quality_gate"] is True
    assert high_sample_score(row, periods, Args) > 0

    row["bt_num_trades"] = 21
    quality = high_sample_quality(row, Args)

    assert quality["sample_gate"] is False
    assert quality["quality_gate"] is True


def test_high_sample_validation_uses_fallback_baseline_label() -> None:
    high_sample = pd.DataFrame(
        [
            {
                "label": "high_sample",
                "high_sample_score": 1.0,
                "sample_gate": True,
                "quality_gate": True,
            }
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                "label": "fresh_repaired_baseline",
                "long_score": 2.0,
            }
        ]
    )

    candidates, metadata = select_high_sample_candidates(high_sample, baseline, baseline_label="missing_baseline")

    assert metadata["baseline_found"] is False
    assert metadata["effective_baseline_label"] == "fresh_repaired_baseline"
    assert candidates.iloc[0]["roles"] == "current_baseline"


def test_high_sample_validation_skips_partially_written_optimizer_rows(tmp_path) -> None:
    path = tmp_path / "live_results.csv"
    path.write_text(
        "label,score,bt_num_trades\n"
        "good_a,1.0,60\n"
        "partial,2.0,61,unfinished_extra_field\n"
        "good_b,3.0,62\n",
        encoding="utf-8",
    )

    frame = read_live_optimizer_csv(path, attempts=1, delay_seconds=0.0)

    assert frame["label"].tolist() == ["good_a", "good_b"]
    assert frame["bt_num_trades"].tolist() == [60, 62]


def test_optimizer_csv_append_migrates_schema_without_losing_rows(tmp_path) -> None:
    path = tmp_path / "results.csv"
    append_rows_schema_safe(path, [{"label": "old_rule", "score": 1.0}])
    append_rows_schema_safe(path, [{"label": "new_rule", "score": 2.0, "calibration_bins": "[]"}])

    frame = pd.read_csv(path)

    assert frame["label"].tolist() == ["old_rule", "new_rule"]
    assert frame["score"].tolist() == [1.0, 2.0]
    assert pd.isna(frame.loc[0, "calibration_bins"])
    assert frame.loc[1, "calibration_bins"] == "[]"


def test_optimizer_monitor_detects_new_high_sample_validation_need() -> None:
    validation = {"top_rows": [{"label": "old_rule", "status": "reject_weak_metrics"}]}
    high_state = {"best_qualified_label": "new_rule"}

    assert validation_contains_label(validation, "old_rule")
    assert not validation_contains_label(validation, "new_rule")
    assert needs_high_sample_validation(high_state, validation)

    validation["top_rows"].append({"label": "new_rule", "status": "reject_weak_metrics"})

    assert not needs_high_sample_validation(high_state, validation)


def test_optimizer_monitor_process_matching_uses_module_and_output_dir(tmp_path) -> None:
    spec = OptimizerSpec(
        name="strict",
        module="nvda_quant_model.long_run_optimizer",
        screen_name="screen_name",
        output_dir=tmp_path / "strict",
        log_name="optimizer.log",
        extra_args=(),
    )
    ps_output = f"python3.13 -m nvda_quant_model.long_run_optimizer --output-dir {spec.output_dir}"

    assert is_optimizer_running(spec, ps_output)
    assert not is_optimizer_running(spec, "python3.13 -m nvda_quant_model.long_run_optimizer --output-dir other")

    relative_output = "nvda_quant_model/outputs/strict"
    relative_spec = OptimizerSpec(
        name="strict",
        module="nvda_quant_model.long_run_optimizer",
        screen_name="screen_name",
        output_dir=PROJECT_ROOT / "outputs" / "strict",
        log_name="optimizer.log",
        extra_args=(),
    )
    ps_output = "\n".join(
        [
            f"python3.13 -m nvda_quant_model.long_run_optimizer --output-dir {relative_output}",
            f"python3.13 -m nvda_quant_model.long_run_optimizer --output-dir {relative_output}",
        ]
    )

    assert is_optimizer_running(relative_spec, ps_output)
    assert optimizer_process_count(relative_spec, ps_output) == 2


def test_optimizer_monitor_can_include_stable_optimizer(tmp_path) -> None:
    class Args:
        strict_output_dir = str(tmp_path / "strict")
        high_sample_output_dir = str(tmp_path / "high")
        stable_output_dir = str(tmp_path / "stable")
        hours = 1.0
        price_override = 215.34
        include_stable = True

    specs = optimizer_specs(Args)
    stable = [spec for spec in specs if spec.name == "stable"]

    assert len(stable) == 1
    assert stable[0].state_path.name == "stable_state.json"
    assert stable[0].module == "nvda_quant_model.stable_candidate_optimizer"


def test_optimizer_monitor_can_include_stable_repair_optimizer(tmp_path) -> None:
    class Args:
        strict_output_dir = str(tmp_path / "strict")
        high_sample_output_dir = str(tmp_path / "high")
        stable_output_dir = str(tmp_path / "stable")
        repair_output_dir = str(tmp_path / "repair")
        hours = 1.0
        price_override = 215.34
        include_stable = False
        include_repair = True

    specs = optimizer_specs(Args)
    repair = [spec for spec in specs if spec.name == "stable_repair"]

    assert len(repair) == 1
    assert repair[0].state_path.name == "stable_repair_state.json"
    assert repair[0].module == "nvda_quant_model.stable_repair_optimizer"


def test_repair_overlay_only_adds_exposure_when_core_is_flat() -> None:
    index = pd.bdate_range("2026-01-02", periods=4)
    core = pd.DataFrame(
        {
            "position": [1.0, 0.0, 0.0, 1.0],
            "direction": [1, 0, 0, 1],
            "confidence": [0.7, 0.0, 0.0, 0.7],
            "expected_return": [0.01, 0.0, 0.0, 0.01],
            "prob_up": [0.7, 0.5, 0.5, 0.7],
            "prob_down": [0.3, 0.5, 0.5, 0.3],
        },
        index=index,
    )
    repair = pd.DataFrame(
        {
            "position": [1.0, 1.0, 0.0, 1.0],
            "direction": [1, 1, 0, 1],
            "confidence": [0.6, 0.61, 0.0, 0.62],
            "expected_return": [0.008, 0.009, 0.0, 0.01],
            "prob_up": [0.6, 0.61, 0.5, 0.62],
            "prob_down": [0.4, 0.39, 0.5, 0.38],
        },
        index=index,
    )

    combined = combine_core_and_repair_signals(core, repair, index, repair_exposure=0.35)

    assert combined.loc[index[0], "position"] == 1.0
    assert combined.loc[index[3], "position"] == 1.0
    assert combined.loc[index[1], "position"] == 0.35
    assert combined["repair_overlay_active"].tolist() == [0, 1, 0, 0]
    assert combined.loc[index[1], "prob_up"] == 0.61


def test_repair_overlay_respects_min_probability_filter() -> None:
    index = pd.bdate_range("2026-01-02", periods=2)
    core = pd.DataFrame(
        {
            "position": [0.0, 0.0],
            "direction": [0, 0],
            "confidence": [0.0, 0.0],
            "expected_return": [0.0, 0.0],
            "prob_up": [0.5, 0.5],
            "prob_down": [0.5, 0.5],
        },
        index=index,
    )
    repair = pd.DataFrame(
        {
            "position": [1.0, 1.0],
            "direction": [1, 1],
            "confidence": [0.61, 0.67],
            "expected_return": [0.01, 0.01],
            "prob_up": [0.61, 0.67],
            "prob_down": [0.39, 0.33],
        },
        index=index,
    )

    combined = combine_core_and_repair_signals(core, repair, index, repair_exposure=0.20, min_repair_prob=0.65)

    assert combined["repair_overlay_active"].tolist() == [0, 1]
    assert combined.loc[index[0], "position"] == 0.0
    assert combined.loc[index[1], "position"] == 0.20


def test_repair_constraints_block_recent_or_mid_degradation() -> None:
    baseline = pd.Series(
        {
            "annualized_return": 0.27,
            "sharpe_ratio": 2.4,
            "max_drawdown": -0.03,
            "win_rate": 0.75,
            "profit_factor": 6.6,
        }
    )
    candidate = pd.Series(
        {
            "annualized_return": 0.28,
            "sharpe_ratio": 2.5,
            "max_drawdown": -0.0315,
            "win_rate": 0.74,
            "profit_factor": 6.7,
        }
    )

    failures = baseline_constraint_failures(candidate, baseline, months=24)

    assert "24m_win_rate" in failures
    assert "24m_max_drawdown" in failures


def test_repair_improvement_requires_60m_to_beat_baseline_and_targets() -> None:
    baseline = pd.Series(
        {
            "annualized_return": 0.154,
            "sharpe_ratio": 1.24,
            "max_drawdown": -0.133,
            "win_rate": 0.57,
            "profit_factor": 2.32,
            "num_trades": 93,
        }
    )
    candidate = pd.Series(
        {
            "annualized_return": 0.17,
            "sharpe_ratio": 1.32,
            "max_drawdown": -0.12,
            "win_rate": 0.58,
            "profit_factor": 2.5,
            "num_trades": 110,
        }
    )

    failures = sixty_month_improvement_failures(candidate, baseline, min_annualized_target=0.18, min_sharpe_target=1.40)

    assert "60m_annualized_not_improved" not in failures
    assert "60m_annualized_target" in failures
    assert "60m_sharpe" in failures


def test_repair_score_rewards_60m_improvement_after_constraints_pass() -> None:
    stress = pd.DataFrame(
        [
            {"lookback_months": 24, "annualized_return": 0.28, "sharpe_ratio": 2.5, "max_drawdown": -0.02, "win_rate": 0.76, "profit_factor": 6.8},
            {"lookback_months": 36, "annualized_return": 0.19, "sharpe_ratio": 1.7, "max_drawdown": -0.12, "win_rate": 0.64, "profit_factor": 3.3},
            {
                "lookback_months": 60,
                "annualized_return": 0.20,
                "sharpe_ratio": 1.45,
                "max_drawdown": -0.12,
                "win_rate": 0.59,
                "profit_factor": 2.7,
                "num_trades": 120,
                "dir_active_days": 190,
            },
        ]
    )
    baseline = pd.DataFrame(
        [
            {"lookback_months": 24, "annualized_return": 0.27, "sharpe_ratio": 2.4, "max_drawdown": -0.03, "win_rate": 0.75, "profit_factor": 6.6},
            {"lookback_months": 36, "annualized_return": 0.187, "sharpe_ratio": 1.59, "max_drawdown": -0.133, "win_rate": 0.63, "profit_factor": 3.17},
            {
                "lookback_months": 60,
                "annualized_return": 0.154,
                "sharpe_ratio": 1.24,
                "max_drawdown": -0.133,
                "win_rate": 0.57,
                "profit_factor": 2.32,
                "num_trades": 93,
                "dir_active_days": 159,
            },
        ]
    )
    oos = pd.DataFrame({"sharpe_degradation": [0.05, 0.02]})

    score = sixty_month_first_score(stress, baseline, oos, constraint_failures=[], improvement_failures=[])

    assert score > 0


def test_stable_candidate_summary_rejects_oos_degradation() -> None:
    stress = pd.DataFrame(
        [
            {
                "label": "candidate",
                "roles": "high_sample_stability_probe",
                "lookback_months": months,
                "annualized_return": 0.20,
                "sharpe_ratio": 1.4,
                "max_drawdown": -0.10,
                "win_rate": 0.60,
                "profit_factor": 2.0,
                "num_trades": 90 if months == 60 else 45,
                "dir_active_days": 130 if months == 60 else 70,
                "dir_precision": 0.61,
            }
            for months in (24, 36, 60)
        ]
    )
    oos = pd.DataFrame(
        [
            {
                "label": "candidate",
                "window_days": 63,
                "sharpe_degradation": 0.35,
                "oos_num_trades": 5,
                "oos_status": "reject_oos_degradation",
            }
        ]
    )

    summary = summarize_stable_validation(stress, oos, pd.DataFrame(), baseline_label="baseline")

    assert summary.loc[0, "status"] == "reject_oos_degradation"


def test_stable_candidate_summary_blocks_regime_weakness() -> None:
    stress = pd.DataFrame(
        [
            {
                "label": "candidate",
                "roles": "high_sample_stability_probe",
                "lookback_months": months,
                "annualized_return": 0.20,
                "sharpe_ratio": 1.4,
                "max_drawdown": -0.10,
                "win_rate": 0.60,
                "profit_factor": 2.0,
                "num_trades": 90 if months == 60 else 45,
                "dir_active_days": 130 if months == 60 else 70,
                "dir_precision": 0.61,
            }
            for months in (24, 36, 60)
        ]
    )
    oos = pd.DataFrame(
        [
            {
                "label": "candidate",
                "window_days": days,
                "sharpe_degradation": 0.05,
                "oos_num_trades": 4,
                "oos_status": "pass",
            }
            for days in (45, 63, 84, 126)
        ]
    )
    regimes = pd.DataFrame(
        [
            {
                "label": "candidate",
                "regime": "high_vol_downtrend",
                "active_days": 6,
                "precision": 0.33,
                "regime_status": "watchlist_regime_weakness",
            }
        ]
    )

    summary = summarize_stable_validation(stress, oos, regimes, baseline_label="baseline")

    assert summary.loc[0, "status"] == "watchlist_regime_weakness"
    assert stable_score(stress, oos, regimes) > 0


def test_stable_candidate_selection_keeps_current_baseline() -> None:
    high_sample = pd.DataFrame(
        [
            {
                "label": "challenger",
                "sample_gate": True,
                "quality_gate": True,
                "high_sample_score": 5.0,
            }
        ]
    )
    strict = pd.DataFrame(
        [
            {
                "label": BASELINE_PRODUCTION_LABEL,
                "long_score": 3.0,
                "bt_num_trades": 32,
            }
        ]
    )

    selected, metadata = select_stable_candidates(high_sample, strict, baseline_label=BASELINE_PRODUCTION_LABEL, top_n=1)

    assert metadata["baseline_found"] is True
    assert BASELINE_PRODUCTION_LABEL in set(selected["label"])
    assert selected.set_index("label").loc[BASELINE_PRODUCTION_LABEL, "roles"] == "current_baseline"


def test_reaction_pool_alerts_only_when_baseline_is_flat() -> None:
    index = pd.bdate_range("2026-01-02", periods=3)
    baseline = pd.DataFrame(
        {
            "position": [0.0, 1.0, 0.0],
            "direction": [0, 1, 0],
            "confidence": [0.0, 0.7, 0.0],
            "expected_return": [0.0, 0.01, 0.0],
            "prob_up": [0.5, 0.7, 0.5],
            "prob_down": [0.5, 0.3, 0.5],
        },
        index=index,
    )
    panel = pd.concat(
        {
            "active": pd.DataFrame({"watch_01": [1.0, 1.0, 1.0], "watch_02": [1.0, 1.0, 0.0]}, index=index),
            "prob": pd.DataFrame({"watch_01": [0.62, 0.64, 0.61], "watch_02": [0.60, 0.63, 0.50]}, index=index),
            "expected": pd.DataFrame({"watch_01": [0.01, 0.01, 0.01], "watch_02": [0.008, 0.008, 0.0]}, index=index),
        },
        axis=1,
    )
    meta = pd.DataFrame({"name": ["watch_01", "watch_02"], "weight": [0.6, 0.4]})

    reaction, diagnostics = build_reaction_signals(
        baseline,
        panel,
        meta,
        index,
        min_vote_count=2,
        min_vote_weight=0.20,
        prob_threshold=0.54,
        max_shadow_exposure=0.25,
    )
    combined = combine_baseline_and_reaction(baseline, reaction, index)

    assert reaction.loc[index[0], "position"] > 0
    assert reaction.loc[index[1], "position"] == 0.0
    assert reaction.loc[index[2], "position"] == 0.0
    assert diagnostics.loc[index[0], "alert"] == 1
    assert combined.loc[index[0], "reaction_overlay"] == 1
    assert combined.loc[index[1], "reaction_overlay"] == 0


def test_reaction_pool_validation_requires_out_of_sample_improvement() -> None:
    metrics = pd.DataFrame(
        [
            {
                "layer": "baseline",
                "annualized_return": 0.12,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.10,
                "direction_precision": 0.58,
                "active_days": 40,
            },
            {
                "layer": "reaction_shadow",
                "annualized_return": 0.01,
                "sharpe_ratio": 0.8,
                "max_drawdown": -0.02,
                "direction_precision": 0.65,
                "active_days": 9,
            },
            {
                "layer": "combined_shadow",
                "annualized_return": 0.13,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.11,
                "direction_precision": 0.60,
                "active_days": 49,
            },
        ]
    )

    assert _passes_validation(metrics, min_reaction_days=8)
    metrics.loc[metrics["layer"] == "combined_shadow", "sharpe_ratio"] = 1.0
    assert not _passes_validation(metrics, min_reaction_days=8)


def test_risk_off_guard_blocks_only_existing_baseline_longs() -> None:
    index = pd.bdate_range("2026-01-02", periods=4)
    baseline = pd.DataFrame(
        {
            "position": [1.0, 0.0, 1.0, 1.0],
            "direction": [1, 0, 1, 1],
            "confidence": [0.7, 0.0, 0.7, 0.7],
            "expected_return": [0.01, 0.0, 0.01, 0.01],
            "prob_up": [0.7, 0.5, 0.7, 0.7],
            "prob_down": [0.3, 0.5, 0.3, 0.3],
        },
        index=index,
    )
    risk_mask = pd.Series([True, True, False, True], index=index)
    guarded = apply_risk_off_guard(baseline, risk_mask, index)

    assert guarded.loc[index[0], "position"] == 0.0
    assert guarded.loc[index[1], "blocked_long"] == 0
    assert guarded.loc[index[2], "position"] == 1.0
    assert guarded.loc[index[3], "prob_up"] == 0.5
    assert guarded["blocked_long"].sum() == 2


def test_risk_off_guard_production_gate_requires_no_metric_degradation() -> None:
    metrics = pd.DataFrame(
        [
            {
                "layer": "baseline",
                "annualized_return": 0.18,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.10,
                "direction_precision": 0.60,
            },
            {
                "layer": "risk_guarded",
                "annualized_return": 0.19,
                "sharpe_ratio": 1.3,
                "max_drawdown": -0.09,
                "direction_precision": 0.63,
                "blocked_days": 3,
                "blocked_loss_rate": 0.67,
            },
        ]
    )

    assert _passes_production_gate(metrics, min_blocked_days=2)
    metrics.loc[metrics["layer"] == "risk_guarded", "annualized_return"] = 0.17
    assert not _passes_production_gate(metrics, min_blocked_days=2)


def test_blocked_trade_diagnostics_counts_avoided_losses() -> None:
    index = pd.bdate_range("2026-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "target_return": [-0.02, 0.01, -0.01, 0.03],
            "target_direction": [0, 1, 0, 1],
        },
        index=index,
    )
    baseline = pd.DataFrame({"position": [1.0, 1.0, 1.0, 0.0], "prob_up": [0.6] * 4}, index=index)
    guarded = pd.DataFrame({"blocked_long": [1, 0, 1, 0], "risk_off": [1, 0, 1, 0]}, index=index)

    diagnostics = blocked_trade_diagnostics(baseline, guarded, frame)

    assert diagnostics["blocked_days"] == 2
    assert diagnostics["blocked_loss_rate"] == 1.0
    assert diagnostics["blocked_loss_capture"] == 1.0


def test_meta_decision_layer_prioritizes_validated_blocks_and_tactical_longs() -> None:
    index = pd.bdate_range("2026-01-02", periods=4)
    baseline = pd.DataFrame(
        {
            "position": [1.0, 0.0, 0.0, 1.0],
            "direction": [1, 0, 0, 1],
            "confidence": [0.7, 0.0, 0.0, 0.7],
            "expected_return": [0.01, 0.0, 0.0, 0.01],
            "prob_up": [0.7, 0.5, 0.5, 0.7],
            "prob_down": [0.3, 0.5, 0.5, 0.3],
        },
        index=index,
    )
    reaction = pd.DataFrame(
        {
            "position": [0.0, 0.25, 0.25, 0.0],
            "direction": [0, 1, 1, 0],
            "confidence": [0.0, 0.62, 0.63, 0.0],
            "expected_return": [0.0, 0.004, 0.005, 0.0],
            "prob_up": [0.5, 0.62, 0.63, 0.5],
            "prob_down": [0.5, 0.38, 0.37, 0.5],
        },
        index=index,
    )
    risk_mask = pd.Series([True, False, True, False], index=index)

    strict, strict_diag = build_meta_signals(
        baseline,
        reaction,
        risk_mask,
        index,
        MetaPolicy("strict", allow_reaction_trades=False, allow_risk_blocks=True),
    )

    assert strict_diag.loc[index[0], "decision"] == "BLOCK_LONG"
    assert strict.loc[index[0], "position"] == 0.0
    assert strict_diag.loc[index[1], "decision"] == "WATCH"
    assert strict_diag.loc[index[2], "decision"] == "RISK_OFF"
    assert strict_diag.loc[index[3], "decision"] == "BASELINE_LONG"

    research, research_diag = build_meta_signals(
        baseline,
        reaction,
        risk_mask,
        index,
        MetaPolicy("research", allow_reaction_trades=True, allow_risk_blocks=False),
    )

    assert research_diag.loc[index[0], "decision"] == "BASELINE_LONG"
    assert research_diag.loc[index[1], "decision"] == "TACTICAL_LONG"
    assert research.loc[index[1], "position"] == 0.25
    assert research_diag.loc[index[2], "decision"] == "RISK_OFF"


def test_live_order_flow_overlay_scores_top_of_book_and_minute_trend() -> None:
    snapshot = {
        "source": "alpaca_stock_snapshot",
        "feed": "iex",
        "snapshots": {
            "NVDA": {
                "latest_trade": {"timestamp": "2026-05-22T20:00:00Z", "price": 215.4, "size": 200},
                "latest_quote": {
                    "timestamp": "2026-05-22T20:00:00Z",
                    "bid_price": 215.35,
                    "bid_size": 900,
                    "ask_price": 215.45,
                    "ask_size": 300,
                },
                "minute_bar": {
                    "timestamp": "2026-05-22T19:59:00Z",
                    "open": 215.1,
                    "high": 215.45,
                    "low": 215.0,
                    "close": 215.4,
                    "volume": 50_000,
                    "trade_count": 700,
                    "vwap": 215.2,
                },
                "daily_bar": {
                    "timestamp": "2026-05-22T04:00:00Z",
                    "open": 213.0,
                    "high": 216.0,
                    "low": 212.0,
                    "close": 215.4,
                    "volume": 8_000_000,
                },
                "previous_daily_bar": {"timestamp": "2026-05-21T04:00:00Z", "close": 213.0},
            }
        },
    }
    bars = bars_frame(
        [
            {
                "timestamp": f"2026-05-22T19:{minute:02d}:00Z",
                "open": 214.0 + minute * 0.02,
                "high": 214.1 + minute * 0.02,
                "low": 213.9 + minute * 0.02,
                "close": 214.0 + minute * 0.025,
                "volume": 10_000 + minute * 400,
                "trade_count": 200 + minute,
                "vwap": 214.0 + minute * 0.02,
            }
            for minute in range(30, 60)
        ]
    )

    overlay = analyze_order_flow(snapshot, "NVDA", bars)

    assert overlay["top_of_book"]["quote_imbalance"] == 0.5
    assert overlay["minute_trend"]["return_5m"] > 0
    assert overlay["micro_signal"]["signal"] == 1
    assert overlay["micro_signal"]["label"] == "bullish"
    assert overlay["execution_filter"]["use_as_entry_signal"] is False


def test_options_overlay_marks_path_dependent_metrics_unavailable(tmp_path, monkeypatch) -> None:
    snapshot = {
        "as_of_date": "2026-05-25",
        "source": "test_option_snapshot",
        "feed": "indicative",
        "snapshots": {
            "NVDA260605C00215000": {
                "latest_quote": {"bid_price": 10.0, "ask_price": 10.4},
                "latest_trade": {"price": 10.2},
                "implied_volatility": 0.55,
                "greeks": {"delta": 0.50},
            },
            "NVDA260605P00215000": {
                "latest_quote": {"bid_price": 9.5, "ask_price": 9.9},
                "latest_trade": {"price": 9.7},
                "implied_volatility": 0.58,
                "greeks": {"delta": -0.50},
            },
            "NVDA260612C00215000": {
                "latest_quote": {"bid_price": 13.0, "ask_price": 13.4},
                "latest_trade": {"price": 13.2},
                "implied_volatility": 0.56,
                "greeks": {"delta": 0.50},
            },
            "NVDA260612P00215000": {
                "latest_quote": {"bid_price": 12.5, "ask_price": 12.9},
                "latest_trade": {"price": 12.7},
                "implied_volatility": 0.62,
                "greeks": {"delta": -0.50},
            },
        },
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(
        "nvda_quant_model.options_volatility.realized_volatility",
        lambda ticker="NVDA": {"hv_10d": 0.45, "hv_20d": 0.50, "hv_60d": 0.48},
    )

    report = analyze_options(path, underlying_price=215.0)

    assessment = report["distribution_assessment"]
    assert assessment["path_dependent_metrics_status"] == "unavailable"
    assert assessment["barrier_probabilities"] is None
    assert "must not be interpreted" in assessment["warning"]


def test_order_flow_backtest_charges_round_trip_cost_once() -> None:
    index = pd.date_range("2026-05-22T14:00:00Z", periods=20, freq="min")
    scored = pd.DataFrame(
        {
            "open": np.linspace(100.0, 101.9, len(index)),
            "close": np.linspace(100.1, 102.0, len(index)),
            "signal": [1] + [0] * (len(index) - 1),
            "micro_score": [3.0] + [0.0] * (len(index) - 1),
            "session": ["2026-05-22"] * len(index),
        },
        index=index,
    )

    trades, metrics = backtest_intraday_signals(scored, horizon_minutes=5, cost_per_trade=0.001)

    expected_gross = scored["close"].iloc[5] / scored["open"].iloc[1] - 1.0
    assert len(trades) == 1
    assert np.isclose(trades.iloc[0]["net_return"], expected_gross - 0.001)
    assert metrics["cost_per_trade"] == 0.001
