# Research runbook

## Choose the evidence you want to reproduce

| Path | What it establishes | Data boundary |
|---|---|---|
| Synthetic example | The complete pipeline runs without a market-data subscription | Deterministic generated fixture; not market performance |
| Frozen market experiment | The saved v2 historical development evaluation | 1,641 rows ending 2026-05-22; run generated 2026-09-18 |
| Historical replay | Sensitivity of already selected signals to accounting and trading rules | Strict cases end 2026-05-22; separate default case ends 2026-07-24 |
| Prospective shadow workflow | Immutable model registration, forecast recording and later outcomes | Requires fresh, provenance-bound inputs; no prospective track record is supplied |

The publication/documentation date is separate from these input dates. None of the commands below silently refreshes the frozen data or turns previously researched history into an untouched holdout.

## Environment and offline checks

Create a separate Python 3.13 environment and install `research/environment.lock.txt`. The top-level `requirements-research.txt` declares direct dependencies; the lock also pins transitive dependencies. The reference run uses one numerical thread. Cross-platform BLAS and font differences can change low-order floats or image bytes; compare numerical results within explicit tolerances, not PNG hashes across operating systems.

```bash
python3.13 -m venv .venv-research
.venv-research/bin/python -m pip install -r research/environment.lock.txt
NVDA_OFFLINE_TESTS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-research/bin/python -m pytest -q -p no:cacheprovider
```

`NVDA_OFFLINE_TESTS=1` blocks socket connections inside pytest. The offline example itself has no fetching path. Historical network adapters remain separate explicit commands.

## Synthetic example

```bash
MPLCONFIGDIR=.mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-research/bin/python -m nvda_quant_model.research.runner \
  --synthetic --config research/configs/smoke.json --output research/runs/example
```

This creates a deterministic, license-free OHLCV fixture, two outer folds, three directional model families, probability/sizing diagnostics, distribution forecasts and plots. It validates the pipeline, not a market hypothesis. Use a new output directory if that example has already completed.

## Frozen market experiment

On the original local research machine, the baseline cache is copied into `nvda_quant_model/cache/` and excluded from git. Its hash is recorded in `research/baseline/manifest.json`. To rerun the reference experiment:

```bash
MPLCONFIGDIR=.mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-research/bin/python -m nvda_quant_model.research.runner \
  --config research/configs/full.json \
  --prices nvda_quant_model/cache/NVDA_2019-11-09_2026-05-23.csv \
  --baseline-dir research/baseline/historical_outputs \
  --cache-dir nvda_quant_model/cache \
  --output research/runs/nvda-frozen-reproduction
```

For other machines, supply your own licensed OHLCV snapshot through `--prices` and omit the optional historical replay arguments if the original signal/cache snapshots are unavailable. A fresh vendor download is a new data vintage and is not automatically an exact reproduction. The public sample is synthetic; no broad market-data redistribution license is implied.

The replay pack also contains scoped frozen price/signal slices and hashes for auditing its 12 variants. These reconstruct execution, not the original feature/model search. Strict 36/60-month references reproduce within the specified numerical tolerance; strict 24-month/default references are explicitly near matches rather than exact matches.

The full configuration registers **11 candidates: five settings across three directional model families and six predictive-distribution variants**. The saved market run contains 18 chronological outer folds, 1,075 predictions per family, 17 strategy/control variants, 51 cost cases (17 × 3) and 36 stop/take neighborhoods. These totals refer to the frozen full run, not the two-fold smoke example. Registered comparisons constrain this run; they do not reverse earlier historical model selection.

Baseline single-side costs are 10 basis points of commission plus 5 basis points of slippage on traded notional, stressed at 1×, 2× and 4×. These are explicit simulation assumptions, not a broker quote or measured market impact. The default execution is next-session open with entry-referenced stops, gap-at-open fills and stop-first ambiguity resolution. Inspect each saved `account.json` alongside its results rather than applying those defaults to legacy replay cases.

## Run identities, failures and resume

The configuration and input/source/dependency identities are registered before model fitting. Each stage has an integrity checkpoint. A failure leaves a failed manifest; completed stages can be resumed with the *same* code, prices, configuration, auxiliary input hashes and dependencies:

```bash
.venv-research/bin/python -m nvda_quant_model.research.runner \
  --synthetic --config research/configs/smoke.json \
  --output research/runs/example --resume
```

Supply the same optional baseline/cache arguments on resume. Edited or missing completed artifacts are rejected; they are not silently rehashed and accepted. Historical replay publishes atomically, retaining failed staging attempts for diagnosis outside the successful run directory. A completed output directory is immutable to a new run. There is no ambiguous `latest` alias; `run_index.jsonl` records explicit run directories and manifest hashes.

A `.running` file records the process ID. After an interrupted operating-system process, verify that exact process is no longer running before removing its stale lock. The runner does not infer that a timeout means completion. Source changes require a new run directory. Research does not automatically delete old artifacts or start unbounded searches.

## Prospective shadow evidence

The saved historical windows cannot become an untouched holdout by being split again. A separate `research.shadow` CLI can register a fixed model, record a genuinely forward forecast from fresh inputs, and attach a later mature outcome. It does not download prices, place orders or run on a schedule.

Example model configuration:

```json
{"model_id":"logistic_shadow_v1","family":"logistic","parameter":0.2,"seed":42,"train_sessions":504,"probability_threshold":0.55,"max_exposure":1.0}
```

```bash
.venv-research/bin/python -m nvda_quant_model.research.shadow register --config shadow_config.json --store research/shadow
.venv-research/bin/python -m nvda_quant_model.research.shadow forecast --registration REGISTRATION_JSON --prices FRESH_OHLCV_CSV --provenance PROVENANCE_JSON
.venv-research/bin/python -m nvda_quant_model.research.shadow outcome --forecast FORECAST_JSON --prices MATURE_OHLCV_CSV --provenance PROVENANCE_JSON
```

Provenance binds the exact `prices_sha256`, `data_kind` (`observed_market` or `synthetic`), `source`, `price_adjustment`, and timezone-aware `available_at`/`retrieved_at`. Authenticity is caller-attested. Prospective recording requires the latest closed XNYS session and must occur before the next open; registration locks model code, configuration and dependencies. Injected test clocks require a separate simulation namespace. Forecasts and outcomes are create-only and hash-linked. No actual prospective track record has been created by this upgrade.

## Files to review

Start with [the model card](MODEL_CARD.md) and [frozen research report](results/frozen_20260918/research_report.md), then inspect:

- `manifest.json`: source/configuration identities, input dates, dependencies and artifact hashes.
- `folds.json` and `predictions.csv`: chronological splits, fitted settings, information clocks and all scored predictions.
- `probability_scores.csv` and `probability_uncertainty.csv`: raw/calibrated/base-rate comparisons and descriptive block intervals.
- `distribution_windows.csv` and `distribution_summary.csv`: coverage, interval score, width, tail misses and horizon definitions.
- `strategy_summary.csv`, `cost_stress.csv`, `parameter_neighborhood.csv` and `stratified_results.csv`: net outcomes, exposure, turnover-sensitive costs and adverse regimes.
- `backtests/*/fills.csv`, `trades.csv`, `equity.csv` and `account.json`: traded quantities, fees, realized/unrealized P&L, balances and halt state.
- `historical_replay/historical_replay.csv`: legacy reference status and the distinct accounting/execution variants.

The charts read saved result files. The primary forecast target is next-open to following-open; bracket exits and risk halts can change actual holding periods. Do not interpret a probability-score improvement as an identical improvement in realized trading performance. The [technical brief](TECHNICAL_BRIEF.md) explains the practical consequences.

## Build the compact package

Run `python scripts/build_nvda_repro_pack.py --output nvda-research-v2.zip`. The archive includes runnable source, pinned dependencies, a synthetic fixture, focused regression tests and compact market result evidence. Full market-run manifests refer to additional ledgers retained in the source repository; the compact archive does not contain the personal historical cache.
