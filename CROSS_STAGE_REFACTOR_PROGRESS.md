# Cross-Stage Refactor Progress

## Checklist

- [x] Phase 0: Baseline snapshot and branch
- [ ] Phase 1: Remove 2048-D feature path from main A->B->C
- [ ] Phase 2: Clarify and implement plaintext/MPC boundary
- [ ] Phase 3: Add TFE secure-system Stage-C validation
- [ ] Phase 4: Repair privacy accounting and test-set boundary
- [ ] Phase 5: Final end-to-end regression and documentation

## Recovery Information

- Branch: `cross-stage-refactor`
- Baseline commit: `1391ffe97a00476c53e45c14ba2336e0b744062b`
- User-provided instructions:
  - `plan/codex_cross_stage_refactor_plan.md`
  - `plan/codex_continuous_refactor_instruction.md`
- Resource policy:
  - CPU/BLAS threads: `1`
  - GPU jobs: single process, single visible GPU
  - Datasets: serial execution only
  - Long jobs: `nohup`, PID file, dedicated log, no concurrent duplicate
  - No plaintext fallback for requested MPC backends

## Phase 0

- Status: `COMPLETED`
- Modified/added:
  - `CROSS_STAGE_REFACTOR_PROGRESS.md`
  - `plan/codex_cross_stage_refactor_plan.md`
  - `plan/codex_continuous_refactor_instruction.md`
- Command:
  - Stage A baseline sanity through direct Python entry with `train_steps=2`, `eval_interval=1`.
  - Runtime environment fixed to one visible GPU and one CPU/BLAS thread.
- Result:
  - CIFAR-10 Full/Random/Distilled image branches all completed two optimizer steps.
  - `feature_extractor=False` and `shallow_classifier=False` remained unchanged.
- Output:
  - `log/json/stageA_directImage_cifar10_20260725_104432.json`
  - `log/txt/stageA_directImage_cifar10_20260725_104432.log`
  - `expFig/cross_stage_refactor_baseline/phase0_stageA_cifar10_2steps_20260725_104432.png`
- Protection:
  - The sanity run temporarily targeted the fixed Figure 4 filename. Its 2-step image was copied to the baseline directory and the committed formal Figure 4 PNG was restored immediately.
- Known limitation:
  - The legacy Stage A plotting entry uses a fixed figure filename; new cross-stage outputs must use timestamped names.
- Git commit: `2203070` (`Snapshot cross-stage refactor baseline`).
- Baseline key outputs:
  - `log/json/stageA_directImage_cifar10_20260725_100011.json`
  - `log/json/stageA_directImage_cifar100_20260725_100048.json`
  - `log/json/stageA_directImage_tiny_20260725_100121.json`
  - `log/json/end2end_ABC_framework_cifar100_20260714_200537.json`
  - `log/json/moduleC_dp_optimization_cifar100_20260714_120124.json`
- Existing 2048-D feature path and caches are retained for reproducibility and will be marked legacy.
- New main-pipeline outputs will use the `crossStageConvNet_` prefix.

## Phase 1

- Status: `PENDING`

## Phase 2

- Status: `PENDING`

## Phase 3

- Status: `PENDING`

## Phase 4

- Status: `PENDING`

## Phase 5

- Status: `PENDING`
