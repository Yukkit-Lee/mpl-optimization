# Unified Experiment Branch

`unified-experiments` is the canonical branch for continued development.

It is based on commit `9da6cdc` from `new-stage-b-mtt-anchor-postagg`. Git ancestry
and tree audits confirm that this commit already contains the histories and files
from the following branches:

- `autodl-migration`
- `cross-stage-refactor`
- `new-secure-feature-pipeline`
- `new-true-ours-pea-figure2`
- `new-stage-b-mtt-anchor-kd`
- `new-stage-b-secure-anchor-refinement`
- `new-stage-b-secure-heterogeneity-trigger`
- `new-stage-b-mtt-anchor-postagg`

The older remote branches are retained as read-only experiment checkpoints. New
code, reports, figures, tables, and progress-log updates should be committed only
to `unified-experiments` unless a new isolated experiment is explicitly required.

The latest positive Stage-B result is Secure Average followed by Official-MTT
anchor cross-entropy refinement. KD/consensus causality, disagreement-guided
anchor selection, and heterogeneity-triggered routing remain documented negative
results and must not be presented as positive contributions.

## Canonical status and reports

- Authoritative human-readable result report:
  `reportToMe/unified_experiment_report.md`.
- Authoritative machine-readable status:
  `new_log/unified_experiments/unified_artifact_status.json`.
- `reportToMe/paper_readiness_assessment.md` is the current concise readiness
  view; its pre-unification content has been superseded.
- Historical `RUNNING`, `FAIL`, `NOT_READY`, partial JSON, and legacy mixed-schema
  CSV files are retained as provenance only. They do not override a later
  completion audit or the unified status index.

The GitHub repository default branch should be `unified-experiments` so that new
clones do not silently start from the obsolete `main` branch.
