# Implementation status

User approved the complete stages 0–6 plan on 2026-09-18, with research and recruiting as the priority.

## M1 — trustworthy foundations (in progress)

- [x] Isolated local clone on `codex/nvda-research-v2`; original repository untouched.
- [x] Preserve the existing news fallback patch, night-price/VIX modules and local tests.
- [x] Freeze 618 input CSV files and 38 relevant historical result files with SHA-256 manifest.
- [ ] Repair execution and account reconciliation with regression tests.
- [ ] Enforce training-only preprocessing, mature labels and session-aware news timing.
- [ ] Make signal model identities and news freshness explicit.

## M2 — unified research system

- [ ] Configuration-driven offline runner, chronological inner/outer validation and manifests.
- [ ] Strict/default historical replay and corrected execution comparison.
- [ ] Common benchmarks, cost stress, parameter neighborhoods and uncertainty estimates.

## M3 — research conclusions

- [ ] Rule / logistic / shallow boosting comparison.
- [ ] Probability calibration and risk-controlled sizing ablations.
- [ ] Bootstrap / Normal / Student-t distribution validation at 21 and 5 sessions.
- [ ] News data eligibility and incremental-signal ablation, or evidence-backed insufficient-data outcome.

## M4 — reviewable deliverables

- [ ] Offline CI and clean-environment reproducible synthetic example.
- [ ] README, model card, research report and technical explanation.
- [ ] Result-generated charts, LinkedIn figure and short English description.

No live orders, remote pushes or deployment are part of this implementation run. Historical data already used in model development are explicitly development evaluation, not an untouched holdout.
