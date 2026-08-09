# Claude Code guide: legitimate Adaptive Clipping formal experiment

Date: 2026-08-09 UTC  
Canonical branch: `unified-experiments`  
Purpose: obtain a scientifically defensible Adaptive-Clipping contribution, or
preserve a completed negative result, without any official-test/holdout use.

## 0. Read this first

This guide replaces the abandoned idea of inserting test records into training.
Do **not** load, encode, inspect, select, train on, or evaluate against official
test or locked holdout data. Do not create a "diagnostic" exception. The user
has explicitly cancelled the contamination task.

Before touching code, read completely:

1. `AGENTS.md`
2. `plan/0809_paper_experiment_figures_and_adaptive_clipping_completion_plan.md`
3. `reportToMe/new_adaptive_latest_initializer_formal_assessment_0809.md`
4. `new_configs/new_paper_adaptive/new_adaptive_latest_initializer_screen_freeze.json`
5. `new_configs/new_paper_adaptive/new_adaptive_latest_initializer_formal_freeze.json`
6. `new_scripts/new_paper_adaptive/new_screen_latest_initializer_fixed_vs_adaptive.py`
7. `new_log/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.json`
8. `初稿_真实密态环境.md`, Section 5.5 in both languages.

Never rewrite or delete the existing negative result. New work must use a new
namespace such as `new_adaptive_transfer_robustness_v1`.

## 1. Current authoritative result

The latest-initializer CIFAR-10 comparison is complete:

| Mode | Final development accuracy | Early AUC | Final C | epsilon |
|---|---:|---:|---:|---:|
| Fixed DP | 58.673% +/- 0.214pp | 58.321% +/- 0.149pp | 2.000 | 3.0 |
| Adaptive DP | 58.807% +/- 0.023pp | 58.328% +/- 0.145pp | 1.388 +/- 0.091 | 3.0 |

Adaptive minus Fixed is `+0.133 +/- 0.192pp`; seed42 is negative. It failed
the frozen `+1pp / all seeds positive / AUC non-worse` gate, so no TFE run was
authorized. The main reason is mechanistic: the selection-optimal Fixed bound
is C=2.0, while observed gradient norms are roughly 1.25--1.42. Fixed therefore
clips almost nothing and already behaves like an oracle; Adaptive has little
clipping bias to repair and spends privacy on noisy count releases.

Do not rerun the same grid hoping for a different random outcome.

## 2. Defensible new research question

The next experiment must test **transfer robustness**, not claim that Adaptive
beats a separately oracle-tuned Fixed bound in every cell.

Primary question:

> If one clipping policy is selected once and transferred across dataset and
> privacy settings without per-cell retuning, does Adaptive reduce sensitivity
> to clipping-bound misspecification while preserving the same composed privacy
> budget?

This is a legitimate contribution because per-deployment oracle tuning is
expensive and often unavailable. Keep a selection-tuned Fixed oracle as a
secondary upper-bound reference, but do not use it as the only primary control.

Allowed conclusion if positive: "Adaptive clipping improves robustness to a
transferred clipping policy." Do **not** generalize this to "Adaptive always
improves accuracy" or attribute the whole end-to-end gain to Adaptive.

## 3. Frozen source assets

Use only these latest C5 initializers and matching development caches:

### CIFAR-10

- initializer:
  `new_output/new_stage_b_anchor_postagg/tfe_validation/cifar10/seed42/new_stage_b_anchor_postagg_tfe_model_cifar10_seed42.npz`
- SHA256:
  `c7156719beb18f58dc5d539afca061627e91b7c722c7415bf415a710843ab3ec`
- Stage-C cache:
  `new_cache/new_true_ours/official_mtt/stage_c/cifar10/new_true_ours_stage_c_cifar10_seed42.npz`
- SHA256:
  `3a36afc71b800fee5069bf158c427efcc0f451dadfc8aca9d469cd7e707d719c`

### CIFAR-100

- initializer:
  `new_output/new_stage_b_anchor_postagg/tfe_validation/cifar100/seed42/new_stage_b_anchor_postagg_tfe_model_cifar100_seed42.npz`
- SHA256:
  `c93a765f69a09960eeea3617676a3696b5c3a82c645ef383d069cbbebeef659c`
- Stage-C cache:
  `new_cache/new_true_ours/official_mtt/stage_c/cifar100/new_true_ours_stage_c_cifar100_seed42.npz`
- SHA256:
  `20a4542844b94846d6ab5c576d0a09193bff779a63fe7232f63b8ca7fbd8c8fb`

Hard-assert for every cache:

- `test_loaded == 0`
- `initializer_source == official_mtt_only`
- `full_data_initializer_dependency == 0`
- `x_selection/y_selection` are used for tuning only
- `x_reference/y_reference` are not read until finalists are frozen
- official test and locked holdout are never accessed.

The C5 initializer stores `c5_weight/c5_bias`. Plaintext code may read these
members directly. Before TFE, create a small audited adapter NPZ containing
only `weight=c5_weight` and `bias=c5_bias`, plus a JSON manifest with both source
and adapter SHA256. This is a frozen input asset, not a checkpoint.

## 4. Required implementation semantics

Reuse the corrected semantics in
`new_screen_latest_initializer_fixed_vs_adaptive.py`:

- independent Poisson sampling;
- denominator equals `expected_batch_size`, not realized random batch size;
- per-example global gradient norm;
- gradient-noise common random numbers between matched Fixed and Adaptive;
- Adaptive count statistic normalized by expected batch size;
- count noise included in RDP composition;
- Fixed has zero count releases;
- Adaptive count-release count is derived from warmup/update interval;
- epsilon is recalibrated separately for every dataset, step budget and mode;
- no driver seed/share is used in a secure run;
- no post-result change to gamma direction or accounting boundary.

Add hard assertions that the plaintext update and TFE graph use the same
direction:

`C_{t+1} = clip(C_t * exp(-eta * (noisy_fraction_below - gamma)))`.

## 5. Phase A: code audit and selection-only diagnostics

Do not start a candidate grid until all of the following are written to a
preflight JSON:

1. branch and pushed Git commit;
2. CPU count, cpuset, cgroup memory limit/current use, free disk;
3. initializer/cache hashes;
4. evaluation fields proving selection-only use;
5. exact candidate count and expected maximum memory;
6. exact tmux command and log path;
7. official-test/locked-holdout read flags fixed to false.

On `x_selection/y_selection` only, record for both datasets:

- initial accuracy;
- per-example gradient-norm percentiles p10/p25/p50/p75/p90/p99;
- fraction clipped at C in `{0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`;
- how the norm distribution changes at 0%, 25%, 50%, 75%, 100% of the update
  budget for one diagnostic seed.

This phase determines whether there is a real nonstationary/misspecified-bound
regime. If every transferred bound clips almost nothing throughout, stop and
report that the data do not support an Adaptive contribution.

## 6. Phase B: bounded selection screen

Use selection data only. Freeze the following grid before running it; if code
audits show a value is nonsensical, revise once **before** any candidate result
exists and record the reason.

Shared:

- datasets: CIFAR-10 and CIFAR-100;
- epsilon: `{1, 3, 8}`;
- delta: `1e-5`;
- expected Poisson batch: `16`;
- steps: `{160, 320}`;
- learning rate: `{0.001, 0.002, 0.005}`;
- screen RNG: `817`;
- warmup: `5`;
- update interval: `{5, 10}`;
- count noise: `{8, 12}` (account both exactly);
- C range: `[0.05, 4.0]`.

Fixed candidates:

- C in `{0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`.

Adaptive candidates:

- initial C in `{0.75, 1.0, 1.25, 1.5}`;
- gamma in `{0.3, 0.5, 0.7}`;
- eta in `{0.01, 0.02, 0.05}`.

Selection must be hierarchical to avoid a Cartesian explosion:

1. choose steps/LR on CIFAR-10 at epsilon=3 with the median Fixed C=1.0;
2. choose **one transferred Fixed C** on CIFAR-10 selection and freeze it for
   every dataset/epsilon formal cell;
3. choose one Adaptive tuple on CIFAR-10 selection and freeze it unchanged for
   every dataset/epsilon formal cell;
4. independently record a per-dataset selection-tuned Fixed oracle, but use it
   only as a secondary ceiling, not as the primary transferred control.

Selection order: mean normalized final utility across CIFAR-10 epsilon cells,
then mean early AUC, then fewer count releases, then lower runtime. Preserve all
candidates, including negative ones.

## 7. Phase C: freeze before development evaluation

After selection completes, create and push a formal freeze JSON containing:

- exact selected transferred Fixed config;
- exact selected Adaptive config;
- optional selection-only Fixed oracle configs;
- datasets `{cifar10,cifar100}`;
- epsilons `{1,3,8}`;
- formal RNG seeds `{42,43,44}`;
- all input and runner hashes;
- the acceptance rule below;
- explicit `official_test_read=false` and `locked_holdout_read=false`;
- output directories and expected cardinality.

Do not read `x_reference/y_reference` before this freeze is committed and
pushed.

Primary formal matrix: 2 datasets x 3 epsilons x 3 RNG seeds x 2 modes = 36
plaintext runs. Run in tmux with 8 process workers and up to 4 BLAS threads per
worker. Do not set all BLAS/OMP threads to one.

Each run must save:

- final development-validation accuracy;
- best accuracy and early AUC;
- full accuracy trajectory;
- C trajectory;
- clipped fraction diagnostic;
- noise multiplier, count-noise multiplier and count releases;
- exact accounted epsilon/delta;
- sample rate and train size;
- runtime and RNG seed;
- input/config hashes;
- official-test/holdout flags.

## 8. Frozen positive gate

Use a robustness-oriented gate; do not reuse the impossible oracle-Fixed +1pp
criterion as the only definition.

Adaptive may be called a positive **transfer-robustness** contribution only if:

1. its mean final gain over the single transferred Fixed policy is at least
   `+0.5pp` across the six dataset/epsilon cells;
2. at least 4 of 6 cell means are nonnegative;
3. mean early AUC is non-worse;
4. no dataset mean loses more than `0.5pp`;
5. all 36 epsilon values match their targets within numerical tolerance;
6. every Adaptive run has the expected nonzero count releases;
7. no official test/locked holdout is read.

Report the selection-tuned Fixed oracle separately. If Adaptive beats the
transferred Fixed but not the oracle, say exactly that. Never relabel the
transferred control as an oracle-tuned Fixed baseline.

If the gate fails, stop. Do not expand the grid after reading development
results, do not swap seeds, and do not start TFE.

## 9. Phase D: smallest secure TFE confirmation

Only if every primary gate passes, choose the two predeclared epsilon=3 cells:

- CIFAR-10 transferred Fixed vs Adaptive, seed42;
- CIFAR-100 transferred Fixed vs Adaptive, seed42.

That is four TFE runs. Use `/root/autodl-tmp/new_envs/tfe/bin/python` and
`new_stage_c_secure_dpsgd.py`. For each mode calibrate its noise multiplier
with the exact sample rate, steps, count noise and count releases. Use:

- `--sampling_mode poisson`
- `--expected_batch_size 16`
- `--clipping_scope global`
- `--initializer_privacy_status data_dependent_non_dp`
- `--input_lineage official_mtt_only`
- `--evaluation_split development_validation`
- `--precision_fractional 16`
- checkpoint interval no larger than 10.

Every command must run in its own named tmux session and write an exact command
line to a tracked log. Verify provider noise commitments, checkpoint recovery,
no client-local reveal, epsilon, trace cardinality and plaintext/TFE numerical
gap before accepting the cell.

Do not run SPU for this follow-up unless the user explicitly requests it after
the TFE results are audited.

## 10. tmux and Git protocol

Before every experiment batch:

```bash
free -h
nproc
df -h /root/autodl-tmp
tmux list-sessions
git status --short
git branch --show-current
git rev-parse HEAD
```

Example plaintext launch shape (replace the runner with the committed one):

```bash
tmux new-session -d -s adaptive_transfer_screen_0809 \
  'cd /root/autodl-tmp/tfe_experiments && \
   env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
   NUMEXPR_NUM_THREADS=4 \
   /root/autodl-tmp/new_envs/distillation/bin/python \
   new_scripts/new_paper_adaptive/<committed_runner>.py --workers 8 \
   2>&1 | tee new_log/new_paper_adaptive/txt/<tracked_log>.log'
```

For every repository modification:

1. update `progressLog/project_progress.md`;
2. run `git diff --check` and syntax/JSON checks;
3. commit on `unified-experiments`;
4. push before starting the next experiment batch;
5. verify local HEAD equals `origin/unified-experiments`;
6. never commit `new_checkpoints/`, private runtime state, tokens or `spu.log`.

## 11. Figure and table deliverables

If the gate passes, generate a new candidate Figure 6 without overwriting the
current honest negative figure until manuscript review. Use PEA styling:

- blue transferred Fixed, red Adaptive;
- I-shaped sample-SD bars over RNG seeds 42/43/44;
- no uncertainty shading;
- bottom shared legend;
- one row for accuracy trajectories across dataset/epsilon cells;
- a compact second row or companion panel for C trajectory and clipped fraction;
- caption states development validation, conditional Stage-C epsilon and
  transferred-policy scope.

Table 5 must include transferred Fixed, Adaptive, and selection-tuned Fixed
oracle as separate rows. It must report mean +/- sample SD, early AUC, count
releases, epsilon and whether the transfer gate passed.

If the gate fails, preserve the existing Figure 6/Table 5 and write a concise
negative addendum instead of manufacturing a favorable plot.

## 12. Completion checklist

The task is complete only when all applicable items hold:

- [ ] no official test or locked holdout access;
- [ ] selection uses only `x_selection/y_selection`;
- [ ] finalists frozen and pushed before development read;
- [ ] 36/36 plaintext formal rows complete with exact epsilon;
- [ ] all candidates and negative cells retained;
- [ ] acceptance gate evaluated mechanically;
- [ ] TFE skipped on gate failure, or 4/4 TFE cells audited on gate pass;
- [ ] Figure/Table explicitly distinguish transferred Fixed from oracle Fixed;
- [ ] progress, report, freeze, configs, trace, logs and hash manifest pushed;
- [ ] worktree clean except known ignored/untracked runtime files;
- [ ] no checkpoint, token, private JSONL or test-contaminated artifact committed.

## 13. Prohibited shortcuts

- test/holdout in training, selection, tuning or plotting;
- selecting the best formal RNG seed;
- changing the positive gate after seeing results;
- hiding failed candidates;
- comparing Adaptive to a deliberately poor Fixed bound while calling it the
  best Fixed baseline;
- claiming end-to-end record-level DP;
- claiming official-test accuracy;
- interpreting plaintext runtime as secure runtime;
- starting long TFE/SPU runs before the plaintext gate passes.

