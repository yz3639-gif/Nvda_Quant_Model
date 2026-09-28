# Portfolio verification — September 28, 2026

This update brings the saved evidence to the README's first screen and adds an executable GitHub Actions workflow. It does not revise research code, frozen results, source identities or baseline records. The main market-data cutoff remains May 22, 2026.

## Observed local checks

| Check | Result |
|---|---|
| Environment | Python 3.13.9; every installed version matches `environment.lock.txt`; `pip check` passes |
| Offline regression suite | **196 passed in 21.33 seconds**, with test socket connections blocked |
| Synthetic smoke example | Completed predictions, probability, trading, distributions, news eligibility and report stages using `configs/smoke.json` |
| Published evidence integrity | 373 JSON/CSV files and **970,547 numeric values unchanged**; 1,766 artifact references, 42 baseline artifacts, 12 replay manifests and 162 source-file hashes verified |
| README evidence | Existing figure retained, with links to source, replay and full strategy table; no new financial results generated |
| Workflow configuration | YAML parsed; push, pull-request and manual triggers; read-only repository permissions; unique synthetic output directory per run and attempt |

The observed run used macOS on Apple Silicon. Its runtime is a local measurement, not a cross-machine benchmark. The synthetic example is a pipeline check and cannot establish market performance.

```bash
NVDA_OFFLINE_TESTS=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  .venv-research/bin/python -B -m pytest -q -p no:cacheprovider

MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
  .venv-research/bin/python -B -m nvda_quant_model.research.runner \
  --synthetic --config research/configs/smoke.json \
  --output research/runs/portfolio_smoke_20260928

.venv-research/bin/python -B scripts/verify_publication.py
```

Use a fresh output directory when repeating the synthetic run. The directory above is ignored by Git; it does not overwrite any published result.

## Remote verification

The [workflow](../.github/workflows/research.yml) installs the same lock on Ubuntu with Python 3.13.9, verifies the published evidence, runs the offline suite and generates the two-fold synthetic report. Synthetic outputs are uploaded for 14 days. Cross-platform image bytes are not compared to the macOS reference.

The [remote run for commit `69bebe0`](https://github.com/yz3639-gif/Nvda_Quant_Model/actions/runs/36469543095) passed on September 28, 2026: pinned-environment installation, syntax checks, published-evidence integrity, offline regression tests, synthetic smoke generation and report upload all completed successfully. The synthetic artifact is named `synthetic-research-36469543095-1`; its scheduled expiry is October 12, 2026.

That observation belongs to this exact branch commit. The README badge follows push runs on `main`, and the [Actions history](https://github.com/yz3639-gif/Nvda_Quant_Model/actions/workflows/research.yml) is the status record for each later commit. Frozen historical reproduction remains documented in [REPRODUCTION_CHECK.json](REPRODUCTION_CHECK.json), separately from these checks.
