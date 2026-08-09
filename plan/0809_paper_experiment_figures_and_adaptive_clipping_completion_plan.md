# 0809 paper experiment figures and Adaptive Clipping completion plan

Status: `IN_PROGRESS`

Target branch: GitHub default branch `unified-experiments`.

## 1. Scope

This task updates the experimental section of `初稿_真实密态环境.md` and its
supporting artifacts. It must preserve all previously audited positive and
negative results. Existing TFE/SPU runs are read-only unless a new Adaptive
Clipping experiment is explicitly launched under the controls below.

Required deliverables:

1. Rebuild Figure 2 as a 2x3 dual-backend plot: TFE row first, SPU row second;
   remove summary boxes; use PEA-like colors, compact legends, and I-shaped
   sample-standard-deviation error bars where `n >= 2`.
2. Rebuild Figure 3 from frozen Module-A histories. Replace uncertainty bands
   with I-shaped error bars and stop the Official-MTT curve at the declared
   frozen target-release boundary to expose its convergence advantage without
   extrapolation.
3. Redesign Figure 4 so the initializer-source ablation is readable without a
   dense grouped bar chart. Keep Table 4 until the user reviews the replacement.
4. Rebuild Figure 5 with a shared bottom legend and I-shaped error bars only
   when repeated-seed evidence exists. A single-seed panel must say `n=1` and
   must not display invented uncertainty.
5. Diagnose and run a matched Fixed-vs-Adaptive clipping experiment using the
   same latest initializer, data, privacy budget, sampling, update budget, and
   evaluation schedule. Development validation alone selects configurations;
   official test and locked holdout remain unread. The historical negative
   ablation remains visible. Adaptive superiority may be claimed only against
   the best validation-selected Fixed bound and only if the predeclared
   multi-seed criterion is met.
6. Build Figure 6 and complete Table 5 from the matched experiment, including
   accuracy, clipping trajectory/statistics, epsilon, and runtime boundaries.
7. Evaluate Figure 7 strictly by cost and evidentiary value. Skip it if the
   existing artifacts cannot support a scientifically useful plot without a
   costly new scalability matrix.
8. Complete Table 6 as a contextual published-results table, not a falsely
   matched leaderboard. Every external row must identify task/model, privacy
   scope, party count, accuracy/runtime availability, and comparability limits.
9. Reduce Table 7 to Dataset, Initializer, Plaintext Accuracy, TFE Accuracy, and
   Absolute Gap.
10. Update both English and Chinese experimental text, then audit every number,
    caption, path, seed count, privacy boundary, and test/holdout statement.

## 2. Figure style contract

- Full Data: blue `#2C7BB6`; proposed method: red `#D7191C`.
- No translucent uncertainty bands.
- Repeated-seed uncertainty: I-shaped error bars with visible caps and sample
  standard deviation. Error bars are placed at a sparse, deterministic subset
  of curve points so they do not obscure the trajectories.
- `n=1`: no uncertainty mark; state `n=1` in the caption or legend.
- Legends are shared and placed below the panels when practical.
- Development-validation accuracy is never relabeled as official test accuracy.
- TFE/SPU times are backend-specific measurements, not hardware-normalized
  cross-backend speedups.

## 3. Experiment safety and reproducibility

Before any new run:

1. Record cgroup memory limit/current usage, CPU count/cpuset, disk space, Git
   commit, input hashes, and exact command.
2. Launch through a named `tmux` session and write a run-local log plus a machine
   readable preflight manifest.
3. Do not constrain CPU/BLAS threads to one; use the available CPU capacity while
   keeping peak memory below the measured cgroup limit.
4. Run a bounded plaintext diagnostic screen first. Freeze the candidate grid
   and selection rule before reading its results; change one factor at a time.
5. Run matched formal seeds only after the screen and a freeze artifact pass.
6. Never commit `new_checkpoints/`, private runtime state, secrets, or `spu.log`.
7. Update `progressLog/project_progress.md`, commit, and push each completed
   mutation phase.

## 4. Adaptive decision rule

The primary comparison uses the latest Stage-B initializer available to both
arms and varies only the clipping mechanism. Fixed clipping must be tuned over a
predeclared bound grid; Adaptive must be tuned over a predeclared initial-bound,
target-fraction, and update-rate grid. Both arms use matched privacy accounting,
including Adaptive count releases.

A positive Adaptive contribution requires all of the following:

- final development accuracy exceeds the best tuned Fixed result by at least
  1.0 percentage point on the primary dataset;
- the gain has the same sign across all formal seeds;
- early AUC is not worse;
- no test/holdout read, privacy-budget mismatch, or initializer mismatch;
- the full grid and negative outcomes remain archived.

If the rule is not met, Figure 6 and Table 5 report Adaptive as a functional but
non-superior mechanism. No additional search is allowed merely to reverse the
sign of the result.

## 5. Phases

### Phase A — branch, draft, and artifact audit

- [x] Verify GitHub default branch (`unified-experiments`).
- [x] Fast-forward local branch without losing the concurrently edited draft.
- [x] Verify resource gate: 32 CPUs, 60 GiB cgroup memory, 29 GiB free disk,
  no GPU, no active experiment/tmux session.
- [ ] Produce the exact figure/table source map and unsupported-claim list.

### Phase B — existing-data figure rebuild

- [ ] Implement a manuscript figure builder with hard source assertions.
- [ ] Generate and visually inspect Figures 2--5 in PNG and PDF.
- [ ] Update English/Chinese captions and paths.
- [ ] Ask the user whether Table 4 should be removed after showing the new
  Figure 4 and explaining retained information.

### Phase C — Adaptive diagnostic and formal run

- [ ] Audit the latest initializer-to-Stage-C code boundary.
- [ ] Freeze the bounded screen, metrics, seeds, and acceptance criterion.
- [ ] Run the plaintext screen in tmux at maximum safe CPU utilization.
- [ ] Freeze finalists; run the smallest matched formal matrix needed for the
  stated criterion, with interruption recovery and complete provenance.
- [ ] Generate Figure 6, Table 5, and an honest result report.

### Phase D — external and systems evidence

- [ ] Verify Table 6 against primary publications and the local source registry.
- [ ] Simplify Table 7 and audit its six numerical rows.
- [ ] Decide Figure 7: build from sufficient existing evidence or document an
  explicit skip with estimated cost and missing measurements.

### Phase E — manuscript and repository completion

- [ ] Update both language versions of Section 5.
- [ ] Run artifact/hash/path/numerical-boundary audits and inspect every PNG.
- [ ] Update progress and final report, commit, and push to
  `unified-experiments`.

## 6. Stop conditions

Stop and report rather than inventing evidence if a required initializer or
feature asset is missing, the cgroup memory/disk gate fails, the secure runner
cannot preserve the matched protocol, or the Adaptive acceptance rule is not
met. Figure 7 is optional by explicit user instruction and is the first item to
skip when its cost is disproportionate.
