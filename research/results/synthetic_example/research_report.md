# NVDA daily research: evidence and limits

**Synthetic correctness example. These prices are simulated and do not demonstrate investment performance.**

Input: 2019-01-02 — 2022-05-16  |  Simulated prices · correctness demonstration.

This run compares a volume/momentum rule, logistic regression and shallow boosting under one daily research design. Probability mappings use chronological inner blocks; net outcomes use the saved execution and cost configuration. The charts show prespecified model identities, not the best-performing subset. Full variants, including losses and controls, remain in the CSV tables.

Input dates: 2019-01-02 to 2022-05-16; rows: 850. Evaluation status: `synthetic_correctness_example_not_investment_evidence`. Prediction target: `next_session_open_to_following_session_open`.

## Net strategy outcomes

| Strategy | Net total return | Annualized return | Sharpe | Max drawdown | Mean exposure |
| --- | --- | --- | --- | --- | --- |
| boosting_raw | 16.32% | 34.97% | 1.570 | -10.86% | 48.03% |
| boosting_calibrated | 21.72% | 47.69% | 1.596 | -12.51% | 69.29% |
| boosting_vol_target | 19.69% | 42.85% | 1.616 | -12.20% | 64.33% |
| boosting_fixed_risk_control | 17.17% | 36.93% | 1.427 | -11.93% | 61.77% |
| boosting_exposure_control | 38.01% | 89.51% | 3.765 | -4.98% | 68.49% |
| logistic_raw | 7.22% | 14.84% | 0.782 | -10.93% | 49.61% |
| logistic_calibrated | 29.56% | 67.17% | 2.325 | -10.86% | 62.99% |
| logistic_vol_target | 25.49% | 56.92% | 2.183 | -10.86% | 58.69% |
| logistic_fixed_risk_control | 26.05% | 58.32% | 2.332 | -10.64% | 57.12% |
| logistic_exposure_control | 35.82% | 83.59% | 4.218 | -3.10% | 56.36% |
| rule_raw | 4.07% | 8.24% | 0.457 | -12.51% | 48.03% |
| rule_calibrated | 21.72% | 47.69% | 1.596 | -12.51% | 69.29% |
| rule_vol_target | 19.69% | 42.85% | 1.616 | -12.20% | 64.33% |
| rule_fixed_risk_control | 18.44% | 39.91% | 1.555 | -11.73% | 61.20% |
| rule_exposure_control | 22.32% | 49.14% | 2.084 | -8.35% | 71.20% |
| buy_hold | 47.56% | 116.41% | 3.143 | -8.93% | 99.21% |
| trend | 20.94% | 45.84% | 1.572 | -12.51% | 64.57% |

17 of 17 saved strategy variants have positive net total return at baseline costs. The variants share data and are not independent replications.

At 4× recorded costs, 4 of 17 variants remain net positive. Cost sensitivity is evaluated on the same frozen signals; it does not include an order-book capacity model.

Of 16 recorded return-difference intervals versus buy-and-hold, 4 include zero. These are descriptive circular-block intervals for mean daily net-return differences, not selection-adjusted tests. Annualization and a short favorable window do not establish a durable edge.

The exposure controls use training-estimated exposure; they do not guarantee identical realized risk. Use `cost_stress.csv`, `parameter_neighborhood.csv` and `stratified_results.csv` to inspect costs, nearby risk settings and annual/volatility-state variation.

## Probability evidence

| Model | Mapping | Brier | Log loss | Predictions |
| --- | --- | --- | --- | --- |
| boosting | raw | 0.248 | 0.691 | 126 |
| boosting | shrink | 0.245 | 0.682 | 126 |
| boosting | sigmoid | 0.248 | 0.695 | 126 |
| boosting | calibrated | 0.248 | 0.695 | 126 |
| boosting | base_rate | 0.246 | 0.685 | 126 |
| logistic | raw | 0.252 | 0.701 | 126 |
| logistic | shrink | 0.246 | 0.685 | 126 |
| logistic | sigmoid | 0.246 | 0.690 | 126 |
| logistic | calibrated | 0.229 | 0.651 | 126 |
| logistic | base_rate | 0.246 | 0.685 | 126 |
| rule | raw | 0.246 | 0.684 | 126 |
| rule | shrink | 0.247 | 0.688 | 126 |
| rule | sigmoid | 0.246 | 0.688 | 126 |
| rule | calibrated | 0.246 | 0.688 | 126 |
| rule | base_rate | 0.246 | 0.685 | 126 |

- Shallow boosting: selected calibration Brier 0.2479; raw 0.2482; training base rate 0.2458. Calibration reduced observed error versus raw; it did not beat the base-rate score in this run.
- Logistic: selected calibration Brier 0.2290; raw 0.2521; training base rate 0.2458. Calibration reduced observed error versus raw; it beat the base-rate score in this run.
- Volume / momentum: selected calibration Brier 0.2457; raw 0.2456; training base rate 0.2458. Calibration did not reduce observed error versus raw; it beat the base-rate score in this run.

Reliability markers show bin averages and counts. Sparse bins and the forward calibration/refit transfer limit inference. The uncertainty table is descriptive; calibrated labels alone do not justify greater exposure.

1 of 3 recorded calibrated-minus-base-rate Brier intervals lie entirely below zero. This is a descriptive block-bootstrap comparison; shared histories and model selection prevent treating these as independent confirmatory tests.

## Distribution and tail evidence

| Distribution | Sessions | Nominal level | Coverage | Interval score | Windows |
| --- | --- | --- | --- | --- | --- |
| bootstrap_ewma | 21 | 95.00% | 78.57% | 0.593 | 28 |
| bootstrap_rolling | 21 | 95.00% | 82.14% | 0.544 | 28 |
| normal_ewma | 21 | 95.00% | 85.71% | 0.595 | 28 |
| normal_rolling | 21 | 95.00% | 89.29% | 0.567 | 28 |
| student_t_ewma | 21 | 95.00% | 85.71% | 0.583 | 28 |
| student_t_rolling | 21 | 95.00% | 85.71% | 0.562 | 28 |

The primary panels use nonoverlapping terminal-return outcomes; overlapping windows are a supplement in the saved tables. Coverage whiskers use 1.96 times the recorded HAC standard error. They are descriptive and are not a proof of correct calibration. Interval score penalizes both wide intervals and missed outcomes; inspect width, left/right misses and PIT together. Terminal-return coverage is not a stop-loss-hit or maximum-drawdown forecast.

## News eligibility

Recorded status: `insufficient_point_in_time_sample`; events: 0. No news alpha claim is made. Eligibility is a minimum provenance/sample gate, not proof of statistical power.

- No point-in-time historical event store supplied; synthetic examples do not invent news evidence.

## Reproducibility and decision

Input SHA-256: `710f4516c9a8942310384a25c3bb99a85fa65dd075121f08c28365dd1f9c7467`. Source digest: `ef05a1fc6ff25c2968212205b77c9c0f1445ec5213584eb4b55b307de2c09e83`. Where produced, the saved configuration, folds, predictions, trade/fill ledgers, account reconciliation and dependency records support this report.

Price basis: consistent adjusted OHLC assumed from input; cache vendor lineage not independently certified. Decision clock: XNYS session close; theoretical immediate availability, no vendor SLA. Execution clock: next observed XNYS session open.

The historical model search makes reuse of these periods development evidence. There is no automatic strategy promotion. Negative results remain part of the research record; any future deployment decision requires prospective predictions, stable economic evidence and a separate review.
