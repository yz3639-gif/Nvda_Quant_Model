# NVDA daily research: evidence and limits

**Historical development evaluation. Previously researched data are not an untouched holdout.**

Input: 2019-11-11 — 2026-05-22  |  Frozen historical inputs · development evaluation.

This run compares a volume/momentum rule, logistic regression and shallow boosting under one daily research design. Probability mappings use chronological inner blocks; net outcomes use the saved execution and cost configuration. The charts show prespecified model identities, not the best-performing subset. Full variants, including losses and controls, remain in the CSV tables.

Input dates: 2019-11-11 to 2026-05-22; rows: 1641. Evaluation status: `historically_researched_development_evaluation`. Prediction target: `next_session_open_to_following_session_open`.

## Net strategy outcomes

| Strategy | Net total return | Annualized return | Sharpe | Max drawdown | Mean exposure |
| --- | --- | --- | --- | --- | --- |
| boosting_raw | -17.58% | -4.43% | -0.654 | -20.21% | 0.56% |
| boosting_calibrated | -17.18% | -4.32% | -0.423 | -20.27% | 1.12% |
| boosting_vol_target | -18.87% | -4.78% | -0.706 | -20.09% | 2.85% |
| boosting_fixed_risk_control | -19.43% | -4.93% | -0.766 | -20.82% | 2.36% |
| boosting_exposure_control | 47.08% | 9.46% | 0.545 | -18.91% | 29.48% |
| logistic_raw | -15.65% | -3.91% | -0.402 | -20.38% | 0.93% |
| logistic_calibrated | -15.45% | -3.85% | -0.506 | -20.20% | 1.12% |
| logistic_vol_target | -6.98% | -1.68% | -0.121 | -21.17% | 7.11% |
| logistic_fixed_risk_control | 1.09% | 0.25% | 0.077 | -20.30% | 7.07% |
| logistic_exposure_control | 68.26% | 12.96% | 0.817 | -18.23% | 26.33% |
| rule_raw | -16.31% | -4.08% | -0.442 | -21.00% | 2.04% |
| rule_calibrated | -16.31% | -4.08% | -0.442 | -21.00% | 2.04% |
| rule_vol_target | -20.84% | -5.33% | -0.422 | -23.32% | 8.00% |
| rule_fixed_risk_control | -17.93% | -4.52% | -0.645 | -20.01% | 3.10% |
| rule_exposure_control | 44.29% | 8.97% | 0.528 | -19.91% | 29.11% |
| buy_hold | 801.59% | 67.36% | 1.242 | -60.80% | 99.91% |
| trend | -18.36% | -4.64% | -0.371 | -21.06% | 3.07% |

The drawdown guard halted these variants during evaluation: `boosting_raw`, `boosting_calibrated`, `boosting_vol_target`, `boosting_fixed_risk_control`, `logistic_raw`, `logistic_calibrated`, `logistic_vol_target`, `logistic_fixed_risk_control`, `rule_raw`, `rule_calibrated`, `rule_vol_target`, `rule_fixed_risk_control`, `trend`. Their later cash exposure is part of the reported outcome.

5 of 17 saved strategy variants have positive net total return at baseline costs. The variants share data and are not independent replications.

At 4× recorded costs, 4 of 17 variants remain net positive. Cost sensitivity is evaluated on the same frozen signals; it does not include an order-book capacity model.

Of 16 recorded return-difference intervals versus buy-and-hold, 0 include zero. These are descriptive circular-block intervals for mean daily net-return differences, not selection-adjusted tests. Annualization and a short favorable window do not establish a durable edge.

The exposure controls use training-estimated exposure; they do not guarantee identical realized risk. Use `cost_stress.csv`, `parameter_neighborhood.csv` and `stratified_results.csv` to inspect costs, nearby risk settings and annual/volatility-state variation.

## Probability evidence

| Model | Mapping | Brier | Log loss | Predictions |
| --- | --- | --- | --- | --- |
| boosting | raw | 0.257 | 0.707 | 1075 |
| boosting | shrink | 0.252 | 0.697 | 1075 |
| boosting | sigmoid | 0.251 | 0.696 | 1075 |
| boosting | calibrated | 0.252 | 0.697 | 1075 |
| boosting | base_rate | 0.250 | 0.692 | 1075 |
| logistic | raw | 0.253 | 0.701 | 1075 |
| logistic | shrink | 0.250 | 0.693 | 1075 |
| logistic | sigmoid | 0.250 | 0.693 | 1075 |
| logistic | calibrated | 0.250 | 0.692 | 1075 |
| logistic | base_rate | 0.250 | 0.692 | 1075 |
| rule | raw | 0.250 | 0.692 | 1075 |
| rule | shrink | 0.250 | 0.693 | 1075 |
| rule | sigmoid | 0.249 | 0.691 | 1075 |
| rule | calibrated | 0.250 | 0.693 | 1075 |
| rule | base_rate | 0.250 | 0.692 | 1075 |

- Shallow boosting: selected calibration Brier 0.2519; raw 0.2567; training base rate 0.2497. Calibration reduced observed error versus raw; it did not beat the base-rate score in this run.
- Logistic: selected calibration Brier 0.2496; raw 0.2532; training base rate 0.2497. Calibration reduced observed error versus raw; it beat the base-rate score in this run.
- Volume / momentum: selected calibration Brier 0.2499; raw 0.2496; training base rate 0.2497. Calibration did not reduce observed error versus raw; it did not beat the base-rate score in this run.

Reliability markers show bin averages and counts. Sparse bins and the forward calibration/refit transfer limit inference. The uncertainty table is descriptive; calibrated labels alone do not justify greater exposure.

0 of 3 recorded calibrated-minus-base-rate Brier intervals lie entirely below zero. This is a descriptive block-bootstrap comparison; shared histories and model selection prevent treating these as independent confirmatory tests.

## Distribution and tail evidence

| Distribution | Sessions | Nominal level | Coverage | Interval score | Windows |
| --- | --- | --- | --- | --- | --- |
| bootstrap_ewma | 21 | 95.00% | 92.59% | 0.826 | 54 |
| bootstrap_rolling | 21 | 95.00% | 92.59% | 0.810 | 54 |
| normal_ewma | 21 | 95.00% | 94.44% | 0.823 | 54 |
| normal_rolling | 21 | 95.00% | 92.59% | 0.793 | 54 |
| student_t_ewma | 21 | 95.00% | 94.44% | 0.813 | 54 |
| student_t_rolling | 21 | 95.00% | 92.59% | 0.793 | 54 |

The primary panels use nonoverlapping terminal-return outcomes; overlapping windows are a supplement in the saved tables. Coverage whiskers use 1.96 times the recorded HAC standard error. They are descriptive and are not a proof of correct calibration. Interval score penalizes both wide intervals and missed outcomes; inspect width, left/right misses and PIT together. Terminal-return coverage is not a stop-loss-hit or maximum-drawdown forecast.

## News eligibility

Recorded status: `insufficient_point_in_time_sample`; events: 26. No news alpha claim is made. Eligibility is a minimum provenance/sample gate, not proof of statistical power.

- Fewer than 250 events; minimum gate is not a statistical-power guarantee.
- Less than 24 months coverage.
- Historical availability provenance is missing; publication date alone does not establish ingestion availability.

## Reproducibility and decision

Input SHA-256: `e4e4a1b1b4da6a036f2c423ec415f23c91f948114fbaf43be8f676537565afc0`. Source digest: `ef05a1fc6ff25c2968212205b77c9c0f1445ec5213584eb4b55b307de2c09e83`. Where produced, the saved configuration, folds, predictions, trade/fill ledgers, account reconciliation and dependency records support this report.

Price basis: consistent adjusted OHLC assumed from input; cache vendor lineage not independently certified. Decision clock: XNYS session close; theoretical immediate availability, no vendor SLA. Execution clock: next observed XNYS session open.

The historical model search makes reuse of these periods development evidence. There is no automatic strategy promotion. Negative results remain part of the research record; any future deployment decision requires prospective predictions, stable economic evidence and a separate review.

Historical replay artifacts are retained in `historical_replay/`. Legacy, corrected-close and next-open/entry-reference cases must retain separate identities; execution and strategy-semantic changes must not be presented as a single pure bug fix.

Saved comparison: `historical_replay/historical_replay.csv`.

| Family | Lookback Months | Variant | Legacy Reference Status | Total Return | Annualized Return | Max Drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| strict_baseline | 24 | legacy_close_daily_reset | near_match_not_exact | 61.23% | 27.10% | -2.94% |
| strict_baseline | 24 | corrected_close_daily_reset | near_match_not_exact | 61.14% | 27.06% | -2.94% |
| strict_baseline | 24 | corrected_next_open_entry | near_match_not_exact | 11.98% | 5.85% | -15.75% |
| strict_baseline | 36 | legacy_close_daily_reset | matched_within_1e-10 | 67.05% | 18.71% | -13.32% |
| strict_baseline | 36 | corrected_close_daily_reset | matched_within_1e-10 | 71.11% | 19.66% | -13.30% |
| strict_baseline | 36 | corrected_next_open_entry | matched_within_1e-10 | -2.88% | -0.97% | -17.78% |
| strict_baseline | 60 | legacy_close_daily_reset | matched_within_1e-10 | 104.21% | 15.40% | -13.32% |
| strict_baseline | 60 | corrected_close_daily_reset | matched_within_1e-10 | 107.76% | 15.80% | -13.30% |
| strict_baseline | 60 | corrected_next_open_entry | matched_within_1e-10 | -3.19% | -0.65% | -20.84% |
| default_model | 24 | legacy_close_daily_reset | near_match_not_exact | 6.46% | 3.19% | -17.47% |
| default_model | 24 | corrected_close_daily_reset | near_match_not_exact | 6.99% | 3.45% | -17.45% |
| default_model | 24 | corrected_next_open_entry | near_match_not_exact | -2.90% | -1.47% | -15.56% |
