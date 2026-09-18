# Research runbook

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

This creates a deterministic, license-free OHLCV fixture, two outer folds, three model families, probability/sizing diagnostics, distribution forecasts and plots. It validates the pipeline, not a market hypothesis.

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

The replay pack also contains scoped frozen price/signal slices and hashes for auditing its 12 variants. These reconstruct execution, not the original feature/model search. Strict 36/60-month references reproduce within the specified numerical tolerance; strict24/default are explicitly near matches rather than exact matches.

## Run identities, failures and resume

The configuration and input/source/dependency identities are registered before model fitting. Each stage has an integrity checkpoint. A failure leaves a failed manifest; completed stages can be resumed with the *same* code, prices, configuration, auxiliary input hashes and dependencies:

```bash
.venv-research/bin/python -m nvda_quant_model.research.runner \
  --synthetic --output research/runs/example --resume
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

`manifest.json`, `folds.json`, `predictions.csv`, probability scores/reliability, distribution windows/summary, strategy summary, cost/parameter stresses, yearly/regime strata, block uncertainty, and each strategy's fill/account/equity files. Figures read these files directly. The model card and technical brief explain assumptions and unsuccessful results; use them alongside the LinkedIn image.
