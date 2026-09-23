# Implementation status — NVDA Research Platform v2

The reproducible daily research workflow is implemented. The frozen reference experiment was generated on **2026-09-18**, using primary market data through **2026-05-22**. Documentation and publication checks were updated on **2026-09-23**; that update did not fetch new prices or create a live trading record.

The reference numerical source identity is commit `11c9cd1e40044a03169598fe257698aec6aee9d2`. Saved run metadata identify that original implementation independently of later documentation and publication changes. The exact source digest is retained in the [reference manifest](results/frozen_20260918/manifest.json).

## Implemented scope

| Area | Delivered capability | Reviewable evidence |
|---|---|---|
| Data and timing | Training-only preprocessing, label-maturity checks, session-aware decision clocks and future-data mutation tests | [Data contract](DATA_CONTRACT.md), regression suite |
| Prediction | Three directional families, five registered settings, chronological inner selection and 18 outer folds | [Configuration](configs/full.json), [fold records](results/frozen_20260918/folds.json) |
| Probability and sizing | Raw/calibrated/base-rate scores; fixed, volatility-targeted and training-estimated exposure controls | [Probability results](results/frozen_20260918/probability_scores.csv), [strategy results](results/frozen_20260918/strategy_summary.csv) |
| Costs and robustness | 17 strategy/control variants; 51 cost cases at 1×/2×/4×; 36 nearby stop/take settings | [Cost stress](results/frozen_20260918/cost_stress.csv), [parameter tests](results/frozen_20260918/parameter_neighborhood.csv) |
| Distribution forecasts | Six variants: bootstrap, Normal and Student-t, each with rolling/EWMA volatility; 5- and 21-session horizons | [Distribution results](results/frozen_20260918/distribution_summary.csv) |
| Execution and accounting | Fill-level fees, partial/reversal trades, gap handling, explicit stop policies, cash/share/P&L reconciliation | [Backtest implementation](../nvda_quant_model/backtest/backtest_engine.py), saved ledgers |
| Historical sensitivity | 12 replay cases using frozen signals, separating legacy accounting, corrected accounting and changed execution rules | [Replay table](results/frozen_20260918/historical_replay/historical_replay.csv) |
| Reproducibility | Registered input/configuration/source identities, stage checkpoints, immutable completed runs and atomic replay publication | [Runbook](RUNBOOK.md), [reproduction check](REPRODUCTION_CHECK.json) |
| Prospective infrastructure | Register fixed models, record time-stamped forecasts and attach mature outcomes | [Shadow workflow](RUNBOOK.md#prospective-shadow-evidence) |

The candidate budget is **11 = five directional settings + six distribution variants**. The 17 trading variants and 51 cost cases are shared-data comparisons, not 17 or 51 independent model discoveries.

## Recorded verification

The following reference-run checks are recorded in [REPRODUCTION_CHECK.json](REPRODUCTION_CHECK.json):

- Isolated Python 3.13.9 environment with pinned dependencies; **196 offline tests passed**, with sockets disabled.
- Independently extracted compact package: **103 included regression tests passed**, plus a successful synthetic CLI run without the original project or cache.
- Independent full frozen-data rerun: **12 principal result tables were byte-identical**, with a recorded numerical tolerance of `1e-12`.
- Read-only audit: **242 artifact hashes**, **807 checkpoint comparisons** and **216 maturity checks** passed.
- **25 corrected ledgers** reconciled **24,320 marked bars**, **5,435 fills** and **1,347 closed trades**. The largest recorded account-identity residual was **$2.63e-10**; drawdown recomputation included initial equity.

For the **2026-09-23 publication review**, the unchanged engine's full offline suite was rerun locally: **196 passed**. Historical reproduction counts above retain their original scope; they are not claims that a new market experiment or prospective test was run on that date. CI configuration alone is not evidence of remote workflow execution.

## Research decisions

The fixed volume/momentum comparator is a new research implementation. Historical strict/default cases separately replay frozen signals; original feature construction and model-selection history are not regenerated. Strict 36/60-month legacy references match within `1e-10`; strict 24-month/default references remain near matches.

For the 60-month frozen-signal case, annualized return was **15.40%** in legacy mode, **15.80%** with corrected close/daily-reset accounting and **−0.65%** with next-open/entry-reference rules. The final comparison changes the trading definition as well as execution timing. The old Sharpe of 1.24 retains its historical identity and is not a result for the new model comparison.

Selected logistic calibration produced a Brier score near **0.24957**, versus **0.25322** raw and **0.24965** for the training base rate. Its descriptive block interval for improvement over the base rate crosses zero. The logistic volatility-targeted strategy returned **−6.98% net** in its saved simulated account. Results do not support automatic model promotion.

News validation stopped at the insufficient-data gate: **26 records**, roughly **3.25 months**, and unverified historical availability. The source history and all new chronological folds are historical development evidence. This implementation does not supply an untouched final holdout or a prospective track record.

## Open validation work

| Item | Current boundary | Required next evidence |
|---|---|---|
| Market-data freshness | Main input ends 2026-05-22; separate default replay ends 2026-07-24 | New documented data vintage and a separately identified run |
| Prospective performance | Logging infrastructure is tested; no forward record is claimed | Immutable forecasts recorded before outcomes, then matured and scored |
| Forecast-to-trade alignment | Open-to-open target; stop/take policies can shorten holding periods | Prespecified horizon-aligned and stopped-policy comparisons |
| Cost realism | Fixed per-side simulation assumptions and multipliers | Appropriate spread/liquidity evidence and explicit capacity assumptions |
| News contribution | Historical sample/provenance gates fail | Sufficient availability-verified events before an incremental-alpha test |
| Production execution | Daily research engine; no production order integration | Separate connectivity, order-state, recovery and operating-risk validation |

Market reference: [frozen results](results/frozen_20260918/research_report.md). Offline example: [synthetic results](results/synthetic_example/research_report.md). Engineering correctness and reproducibility make these conclusions inspectable; they do not establish profitable live trading.
