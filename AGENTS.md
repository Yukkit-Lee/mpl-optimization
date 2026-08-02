# Unified experiment agent rules

The canonical development branch is `unified-experiments`.

- Before starting any experiment, use an anti-disconnect mechanism such as
  `tmux`, verify resources and disk space, and record the exact command.
- Do not constrain CPU/BLAS threads to one unless a specific correctness test
  requires it. Use the available hardware for approved experiments.
- Every repository modification must also update
  `progressLog/project_progress.md`, then be committed and pushed to GitHub.
- Never commit `new_checkpoints/` or private runtime state. Frozen, audited TFE
  input assets may be committed only when they are required for reproducibility
  and are explicitly distinguished from checkpoints.
- Do not treat historical `FAIL`, `RUNNING`, partial, quarantined, or superseded
  artifacts as current state. Read `new_log/unified_experiments/` and
  `reportToMe/unified_experiment_report.md` first.
- Do not claim official-test, locked-holdout, end-to-end record-level DP, or
  communication results unless the cited artifact explicitly audits that scope.
- Preserve negative results. KD/consensus causality, disagreement-guided anchor
  selection, and heterogeneity-triggered routing are not positive modules under
  their evaluated protocols.
