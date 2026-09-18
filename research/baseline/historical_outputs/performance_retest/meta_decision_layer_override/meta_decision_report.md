# NVDA Meta Decision Layer Backtest

## Purpose

This layer is the final decision outlet. It combines the baseline rule, reaction pool, risk-off guard, and live event overlays while keeping historical backtests point-in-time clean.

## Production Rules

- Baseline can trade when active.
- Reaction can only create TACTICAL_LONG if its time-split validation passed.
- Risk-off can only create BLOCK_LONG if the guard is production-approved across windows.
- Live news/options/order-flow are latest overlays unless a point-in-time history is provided.

## Decision

- Strict production pass windows: 3 / 3
- Research watchlist pass windows: 3 / 3
- Production decision: meta_orchestrator_ready_baseline_only
- Reaction validation pass: False
- Risk global decision: watchlist_only

## Latest Meta Decision

- Date: 2026-05-26 (Tuesday)
- Decision: NO_TRADE
- Position: 0.0
- Prob up: 50.00%
- Expected return: 0.00%
- Source layer: none
- Baseline signal: 0
- Risk off: 0
- Reaction monitor: 0
- Live overlay flags: none

## Metrics

|   lookback_months | segment    | layer                   | annualized_return   |   sharpe_ratio | max_drawdown   | win_rate   |   profit_factor |   num_trades | direction_precision   |   active_days |   decision_baseline_long |   decision_tactical_long |   decision_block_long |   decision_risk_off |   decision_watch |
|------------------:|:-----------|:------------------------|:--------------------|---------------:|:---------------|:-----------|----------------:|-------------:|:----------------------|--------------:|-------------------------:|-------------------------:|----------------------:|--------------------:|-----------------:|
|                24 | train      | baseline                | 25.49%              |           2.57 | -1.18%         | 81.25%     |           19.95 |           16 | 77.78%                |            27 |                      nan |                      nan |                   nan |                 nan |              nan |
|                24 | train      | meta_strict_production  | 25.49%              |           2.57 | -1.18%         | 81.25%     |           19.95 |           16 | 77.78%                |            27 |                       27 |                        0 |                     0 |                 149 |                0 |
|                24 | train      | meta_watchlist_research | 25.49%              |           2.57 | -1.18%         | 81.25%     |           19.95 |           16 | 77.78%                |            27 |                       27 |                        0 |                     0 |                 149 |                0 |
|                24 | validation | baseline                | 30.13%              |           2.24 | -2.94%         | 68.75%     |            4.05 |           16 | 60.00%                |            25 |                      nan |                      nan |                   nan |                 nan |              nan |
|                24 | validation | meta_strict_production  | 30.13%              |           2.24 | -2.94%         | 68.75%     |            4.05 |           16 | 60.00%                |            25 |                       25 |                        0 |                     0 |                  71 |                0 |
|                24 | validation | meta_watchlist_research | 30.13%              |           2.24 | -2.94%         | 68.75%     |            4.05 |           16 | 60.00%                |            25 |                       25 |                        0 |                     0 |                  71 |                0 |
|                24 | full       | baseline                | 27.10%              |           2.41 | -2.94%         | 75.00%     |            6.64 |           32 | 69.23%                |            52 |                      nan |                      nan |                   nan |                 nan |              nan |
|                24 | full       | meta_strict_production  | 27.10%              |           2.41 | -2.94%         | 75.00%     |            6.64 |           32 | 69.23%                |            52 |                       52 |                        0 |                     0 |                 220 |                0 |
|                24 | full       | meta_watchlist_research | 27.10%              |           2.41 | -2.94%         | 75.00%     |            6.64 |           32 | 69.23%                |            52 |                       52 |                        0 |                     0 |                 220 |                0 |
|                36 | train      | baseline                | 9.90%               |           0.93 | -13.32%        | 51.85%     |            2.04 |           27 | 54.55%                |            44 |                      nan |                      nan |                   nan |                 nan |              nan |
|                36 | train      | meta_strict_production  | 9.90%               |           0.93 | -13.32%        | 51.85%     |            2.04 |           27 | 54.55%                |            44 |                       44 |                        0 |                     0 |                 237 |                0 |
|                36 | train      | meta_watchlist_research | 9.90%               |           0.93 | -13.32%        | 51.85%     |            2.04 |           27 | 54.55%                |            44 |                       44 |                        0 |                     0 |                 237 |                0 |
|                36 | validation | baseline                | 36.97%              |           2.68 | -2.94%         | 74.07%     |            5.38 |           27 | 68.89%                |            45 |                      nan |                      nan |                   nan |                 nan |              nan |
|                36 | validation | meta_strict_production  | 36.97%              |           2.68 | -2.94%         | 74.07%     |            5.38 |           27 | 68.89%                |            45 |                       45 |                        0 |                     0 |                 111 |                0 |
|                36 | validation | meta_watchlist_research | 36.97%              |           2.68 | -2.94%         | 74.07%     |            5.38 |           27 | 68.89%                |            45 |                       45 |                        0 |                     0 |                 111 |                0 |
|                36 | full       | baseline                | 18.71%              |           1.59 | -13.32%        | 62.96%     |            3.17 |           54 | 61.80%                |            89 |                      nan |                      nan |                   nan |                 nan |              nan |
|                36 | full       | meta_strict_production  | 18.71%              |           1.59 | -13.32%        | 62.96%     |            3.17 |           54 | 61.80%                |            89 |                       89 |                        0 |                     0 |                 348 |                0 |
|                36 | full       | meta_watchlist_research | 18.71%              |           1.59 | -13.32%        | 62.96%     |            3.17 |           54 | 61.80%                |            89 |                       89 |                        0 |                     0 |                 348 |                0 |
|                60 | train      | baseline                | 10.87%              |           0.85 | -13.32%        | 49.21%     |            1.65 |           63 | 57.27%                |           110 |                      nan |                      nan |                   nan |                 nan |              nan |
|                60 | train      | meta_strict_production  | 10.87%              |           0.85 | -13.32%        | 49.21%     |            1.65 |           63 | 57.27%                |           110 |                      110 |                        0 |                     0 |                 389 |                2 |
|                60 | train      | meta_watchlist_research | 10.87%              |           0.85 | -13.32%        | 49.21%     |            1.65 |           63 | 57.27%                |           110 |                      110 |                        0 |                     0 |                 389 |                2 |
|                60 | validation | baseline                | 24.29%              |           2.25 | -2.94%         | 73.33%     |            5.76 |           30 | 67.35%                |            49 |                      nan |                      nan |                   nan |                 nan |              nan |
|                60 | validation | meta_strict_production  | 24.29%              |           2.25 | -2.94%         | 73.33%     |            5.76 |           30 | 67.35%                |            49 |                       49 |                        0 |                     0 |                 202 |                0 |
|                60 | validation | meta_watchlist_research | 24.29%              |           2.25 | -2.94%         | 73.33%     |            5.76 |           30 | 67.35%                |            49 |                       49 |                        0 |                     0 |                 202 |                0 |
|                60 | full       | baseline                | 15.40%              |           1.24 | -13.32%        | 56.99%     |            2.32 |           93 | 60.38%                |           159 |                      nan |                      nan |                   nan |                 nan |              nan |
|                60 | full       | meta_strict_production  | 15.40%              |           1.24 | -13.32%        | 56.99%     |            2.32 |           93 | 60.38%                |           159 |                      159 |                        0 |                     0 |                 591 |                2 |
|                60 | full       | meta_watchlist_research | 15.40%              |           1.24 | -13.32%        | 56.99%     |            2.32 |           93 | 60.38%                |           159 |                      159 |                        0 |                     0 |                 591 |                2 |

## Files

- Metrics: nvda_quant_model/outputs/performance_retest/meta_decision_layer_override/meta_decision_metrics.csv
- Latest: nvda_quant_model/outputs/performance_retest/meta_decision_layer_override/meta_decision_latest.json
- Summary: nvda_quant_model/outputs/performance_retest/meta_decision_layer_override/meta_decision_summary.json