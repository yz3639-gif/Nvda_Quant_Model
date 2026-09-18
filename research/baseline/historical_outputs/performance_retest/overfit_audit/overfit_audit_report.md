# NVDA Overfit Audit

- Label: `tw147_sw63_10d_return_mq0.60_vq0.45_smh++obv++rsi<75+p60>0.96+vix<0.9`
- Overall status: `PASS_BUT_FRAGILE`
- Hard gates pass: True
- Minimum annualized margin: 0.40%
- Minimum trade count: 32

## Warnings
- Annualized return passes, but the minimum margin is below 1 percentage point.
- Only one nearby stop/take parameter set passes all hard gates; parameter fragility is elevated.
- Shorter windows still have fewer than 60 trades; use 24M/36M as supporting evidence, not sole proof.
- At least one yearly regime has precision below 50%; 2022-like regimes need separate handling.
- Probability calibration is weak in at least one window; direction signal is better than prob_up scale.
- Terminal-return engine 95% interval remains under-covered; tail risk is still understated.

## Controls
- Keep the audited production override, but do not promote any new rule without 24/36/60 stress validation.
- Treat 2022-like/high-volatility regimes as a separate model bucket before increasing exposure.
- Do not use prob_up as a sizing scalar until calibration error is reduced.
- Require parameter-neighborhood evidence before future stop/take changes become production.

## Window Thresholds

|   lookback_months |   annualized_return |   annualized_margin |   sharpe_ratio |   max_drawdown |   win_rate |   profit_factor |   num_trades | passes_hard_gates   |
|------------------:|--------------------:|--------------------:|---------------:|---------------:|-----------:|----------------:|-------------:|:--------------------|
|                24 |            0.270954 |          0.120954   |        2.40839 |     -0.0294183 |   0.75     |         6.64223 |           32 | True                |
|                36 |            0.187073 |          0.0370727  |        1.58573 |     -0.133206  |   0.62963  |         3.16691 |           54 | True                |
|                60 |            0.154015 |          0.00401453 |        1.24035 |     -0.133206  |   0.569892 |         2.31926 |           93 | True                |

## Train Vs Validation

|   lookback_months |   train_annualized_return |   validation_annualized_return |   train_sharpe_ratio |   validation_sharpe_ratio |   train_num_trades |   validation_num_trades |   full_worst_year |   full_worst_year_precision |
|------------------:|--------------------------:|-------------------------------:|---------------------:|--------------------------:|-------------------:|------------------------:|------------------:|----------------------------:|
|                24 |                 0.254857  |                       0.301318 |             2.57378  |                   2.24227 |                 16 |                      16 |              2026 |                    0.666667 |
|                36 |                 0.0989833 |                       0.369714 |             0.93351  |                   2.68049 |                 27 |                      27 |              2024 |                    0.5      |
|                60 |                 0.108749  |                       0.242909 |             0.848439 |                   2.24633 |                 63 |                      30 |              2022 |                    0.434783 |

## Parameter Neighborhood

|   stop_loss_pct |   take_profit_pct | pass_all_thresholds   |   min_annualized_return |   min_sharpe_ratio |   max_abs_drawdown |   min_win_rate |   min_profit_factor |   min_num_trades |
|----------------:|------------------:|:----------------------|------------------------:|-------------------:|-------------------:|---------------:|--------------------:|-----------------:|
|           0.02  |             0.035 | False                 |                0.102392 |           0.953014 |           0.167174 |       0.557895 |             1.92373 |               33 |
|           0.02  |             0.04  | False                 |                0.13269  |           1.12067  |           0.152852 |       0.557895 |             2.12657 |               33 |
|           0.02  |             0.045 | False                 |                0.128584 |           1.04919  |           0.156572 |       0.532609 |             2.08826 |               32 |
|           0.025 |             0.035 | False                 |                0.122059 |           1.07908  |           0.13738  |       0.569892 |             2.09275 |               32 |
|           0.025 |             0.04  | True                  |                0.154015 |           1.24035  |           0.133206 |       0.569892 |             2.31926 |               32 |
|           0.025 |             0.045 | False                 |                0.146688 |           1.13977  |           0.13318  |       0.550562 |             2.25735 |               31 |
|           0.03  |             0.035 | False                 |                0.104314 |           0.903944 |           0.1534   |       0.569892 |             1.88229 |               32 |
|           0.03  |             0.04  | False                 |                0.135765 |           1.07405  |           0.132762 |       0.569892 |             2.09062 |               32 |
|           0.03  |             0.045 | False                 |                0.1239   |           0.948608 |           0.154455 |       0.539326 |             1.98873 |               31 |
