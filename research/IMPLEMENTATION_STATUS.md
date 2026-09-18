# Implementation status — complete within the approved research scope

The user approved stages0–6 on2026-09-18, prioritizing research and recruiting. Source implementation is frozen at commit `11c9cd1`; result/documentation commits retain that auditable source identity. No remote push, live order or deployment was performed.

| Milestone | Evidence |
|---|---|
| M1: trustworthy foundations | Isolated `codex/nvda-research-v2` clone; six original local files preserved byte-for-byte; frozen618CSV input hashes and42 historical artifact files; corrected fills/accounting; training-only preprocessing, label maturity and calendar-aware news tests |
| M2: unified research system | `research.runner` completed predictions, probability, trading, distributions, news eligibility, historical replay and reporting;242 saved artifact hashes verified; explicit execution/config/data/source/dependency identities; immutable checkpoints and atomic replay |
| M3: research conclusions | 11 registered candidates;18outerfolds;1,075predictions/model;17 strategy/control variants;51 cost stresses;36 risk-parameter neighbors; six distribution variants at21/5sessions; news experiment stopped at the documented insufficient-data gate |
| M4: reviewable deliverables | README, model card, data contract, runbook, technical brief, generated report, PNG/PDF/SVG charts, LinkedIn figure/description and an independently extracted offline package |

## Verification

- Clean isolated Python3.13.9 environment installed from pinned dependencies.
- Full offline suite: **196 passed**, with socket connections disabled.
- Independent extracted package: **103 included regression tests passed**, followed by a successful synthetic CLI run without the original project/cache.
- Independent full frozen-data rerun:12 principal result tables are byte-identical; numeric tolerance checked at1e−12. See `REPRODUCTION_CHECK.json`.
- Final read-only audit:242 artifact hashes,807 checkpoint comparisons and216 inner/outer maturity checks pass.
- 25 corrected ledgers reconcile24,320 marked bars,5,435fills and1,347closed trades; largest account-identity residual is$2.63e−10. Drawdown recomputation includes initial equity.
- Syntax compilation and `git diff --check` pass. GitHub Actions is configured; no remote CI execution is asserted.
- Original workspace user files and the original24-line news fallback diff remain unchanged.

## Research conclusions and boundaries

The fixed volume/momentum comparator is a new research model. Historical strict/default cases replay frozen signals; missing exact original model/data-vintage provenance is not fabricated. Strict36/60-month legacy references match within1e−10, while strict24/default are near rather than exact matches.

The strict60-month frozen-signal replay gives approximately15.40%annualized in legacy mode,15.80%with corrected close/daily-reset accounting, and−0.65%with next-open/entry assumptions. The last change is a different trading definition, not pure bug-fix attribution.

Calibrated Logistic Brier is approximately0.24957 versus0.25323raw and0.24965training base rate. Its block interval for incremental Brier crosses zero. No model earns automatic promotion. News has26records over3.25months with unverified historical availability and remains explanatory only. Prospective shadow logging is implemented and tested, but no prospective track record has been manufactured.

Market experiment: `results/frozen_20260918/`. Synthetic example: `results/synthetic_example/`. The source data and model-selection history are historical development evidence, not a new untouched holdout. Remaining empirical limitations are explicit outcomes of the approved plan, not claims that the strategy is profitable.
