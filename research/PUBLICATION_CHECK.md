# Publication verification — September 23, 2026

This check publishes the implemented v2 research workflow and explains its saved results. It does not fetch new market prices, rerun model selection on fresh data or create a prospective trading record.

## Checks performed for this publication

| Check | Observed result |
|---|---|
| Full offline regression suite | **196 passed in 21.46 seconds**, with test socket connections disabled |
| Synthetic end-to-end smoke run | Completed predictions, probability, trading, distributions, news eligibility and reporting |
| Research figure | Generated from the saved CSVs; all 17 strategy/control rows retained; PNG and SVG visually checked and reproduced with identical hashes |
| Documentation figures and counts | Data dates, 1,641 rows, 18 distinct folds, 3 directional families, 11 candidate specifications, 17 strategies/controls and 51 cost cases checked against saved artifacts |
| Exported evidence integrity | 373 JSON/CSV files, **970,547 numeric values unchanged**, 1,766 artifact/checkpoint hash references, 42 baseline records and 12 replay manifests verified |
| Local Markdown navigation | All linked file paths resolve across the current README, research documentation and archived README |
| Public scope | Explicit primary data cutoff of May 22, 2026; separate default replay cutoff of July 24, 2026; original experiment date September 18, 2026 |

Reference environment: macOS, Apple Silicon, Python 3.13.9 and the pinned research environment. The test timing is an observed local run, not a performance target or a cross-machine benchmark.

## Commands

From the repository root:

```bash
NVDA_OFFLINE_TESTS=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-research/bin/python -B -m pytest -q -p no:cacheprovider

MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-research/bin/python -m nvda_quant_model.research.runner \
  --synthetic --config research/configs/smoke.json --output research/runs/publication_smoke_20260923

MPLCONFIGDIR=.mplconfig .venv-research/bin/python scripts/build_readme_figures.py
```

Use a new output directory on another run; completed experiment directories are immutable. The output directory above is local and ignored by Git. It contains synthetic inputs and is not a replacement for the dated market experiment.

## Historical evidence remains separately identified

[REPRODUCTION_CHECK.json](REPRODUCTION_CHECK.json) records the September 18 numerical reproduction and package checks. Its 12 byte-identical principal tables, 103 extracted-package tests and ledger audit belong to that original verification. The publication's 196-test rerun is a new observed check; it does not re-label the original market inputs as September data.

Local filesystem and branch metadata are normalized for publication with a separate provenance record. Numerical outcomes and original source identities remain distinct from the published file bytes. Current hash maps must describe the exported files, while original hashes remain available for comparison.

See [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md) and [PUBLICATION_PROVENANCE.json](PUBLICATION_PROVENANCE.json). Run `python scripts/verify_publication.py` to verify the published export using the standard library. Unshipped original cache inputs are explicitly counted on a fresh checkout; they are not silently assumed to be available.

## Remote verification

The command-line GitHub credential cannot create Actions workflows. The [workflow template](ci/research.yml.example) is retained for activation through a connection with workflow permission. Local checks above passed; no remote CI pass is claimed. The repository's Actions history is the authoritative record of remote run status for a given published commit.
