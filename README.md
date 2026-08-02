# MPL Optimization — Unified Experiments

This repository's canonical development branch is `unified-experiments`. It
contains the complete, linearly inherited histories of the migration, secure
feature, True Ours, and Stage-B ablation branches listed in
[`BRANCH_CONSOLIDATION.md`](BRANCH_CONSOLIDATION.md).

## Start here

1. Read [`AGENTS.md`](AGENTS.md) before modifying or running anything.
2. Read the canonical experiment report:
   [`reportToMe/unified_experiment_report.md`](reportToMe/unified_experiment_report.md).
3. Read the machine-readable current/superseded artifact index under
   `new_log/unified_experiments/`.
4. Consult [`progressLog/project_progress.md`](progressLog/project_progress.md)
   for the chronological record; it is not a replacement for the canonical
   result report.

## Evidence boundaries

- The current True Ours comparison is development-validation-only.
- The current positive Stage-B extension is Secure Average followed by
  supervised Official-MTT anchor cross-entropy refinement, evaluated at seed42.
- Stage-C privacy claims are conditional on the Stage-C mechanism; the
  data-dependent Stage-B initializer is not record-level DP.
- Historical official-test, partial, failed, superseded, and quarantined files
  remain for provenance but are not automatically paper-ready.
- Local checkpoints belong in `new_checkpoints/`, which is intentionally ignored.

Modern dependencies are specified in `requirements.txt` / `environment.yml`;
legacy TF-Encrypted dependencies are isolated in
`requirements-tfe-legacy.txt` / `environment-tfe-legacy.yml`.
