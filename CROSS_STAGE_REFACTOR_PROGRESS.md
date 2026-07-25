# Cross-Stage Refactor Progress

## Checklist

- [x] Phase 0: Baseline snapshot and branch
- [x] Phase 1: Remove 2048-D feature path from main A->B->C
- [x] Phase 2: Clarify and implement plaintext/MPC boundary
- [x] Phase 3: Add TFE secure-system Stage-C validation
- [x] Phase 4: Repair privacy accounting and test-set boundary
- [x] Phase 5: Final end-to-end regression and documentation

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

- Status: `COMPLETED`
- Main pipeline:
  - official MTT distilled NCHW images
  - local full ConvNets
  - full-ConvNet KD
  - KD-Avg / KD-ReferenceWeighted / KD-Best full-state candidates
  - heldout-reference-guided chosen full ConvNet
  - same full ConvNet in Stage C Plain / Fixed-DP / Adaptive-DP
- Added:
  - `official_distilled_convnet_pipeline.py`
  - `legacy_feature_pipeline/README.md`
- Reworked:
  - `exp_module_b_kd_noniid_official.py`
  - `exp_official_distilled_figure5_init.py`
  - `exp_moduleC_dp_optimization.py`
  - `exp_end2end_framework.py`
- Legacy isolation:
  - Previous 2048-D scripts copied under `legacy_feature_pipeline/`.
  - Current entry points do not import feature extraction APIs.
- Safe validation:
  - Environment: distillation, one visible GPU, CPU/BLAS threads all `1`.
  - Dataset: CIFAR-10.
  - Clients: `2`; local steps: `2`; KD steps: `2`.
  - Stage C: Plain / Fixed-DP / Adaptive-DP, `2` steps each, batch size `2`.
- Assertions passed:
  - train/reference/test source identifiers are pairwise disjoint;
  - all local full ConvNets changed after KD;
  - all candidates share the complete state-dict schema;
  - no old feature extraction API is referenced by current entry points;
  - checkpoint save/load state signatures match;
  - all Stage C modes start from the loaded chosen signature;
  - history records per-layer gradient norms, total norm, clipped fraction and `C_t`.
- Outputs:
  - `log/json/crossStageConvNet_stageB_cifar10_20260725_105336.json`
  - `log/json/crossStageConvNet_stageC_cifar10_20260725_105347.json`
  - `log/json/crossStageConvNet_end2end_cifar10_20260725_105450.json`
  - `expTab/crossStageConvNet_end2end_cifar10_20260725_105450.csv`
- Known limitation:
  - The two-step smoke run is a structural validation, not an accuracy experiment.

## Phase 2

- Status: `COMPLETED`
- `secure_aggregation.py` now exposes explicit, non-interchangeable backends:
  - `plaintext_aggregate_and_select`
  - `tfe_mpc_aggregate_and_select`
  - `aggregate_and_select(backend=..., allow_plaintext_fallback=False)`
- Silent plaintext fallback was removed from `feature_kd_init.py`.
- Existing non-legacy callers explicitly request `backend="tfe_mpc"`.
- TFE scope is truthful:
  - ABY3 computes linear-parameter averaging and candidate logits.
  - Averaged parameters and candidate logits are revealed.
  - Accuracy and candidate argmax happen after reveal.
  - Scope name: `secure_score_computation_with_revealed_selection`.
- Required metadata is printed and returned:
  - `backend`
  - `protocol`
  - `secret_shared_inputs`
  - `revealed_values`
  - `fallback_used=False`
  - `selection_security_scope`
- Safe validation:
  - distillation environment plaintext API test passed.
  - missing explicit backend guard passed.
  - old TFE environment ABY3 two-client, four-feature micro-test passed with one CPU thread.
- Full ConvNet KD/aggregation remains in the algorithmic PyTorch plaintext pipeline and is not mislabeled MPC.

## Phase 3

- Status: `COMPLETED_WITH_DOCUMENTED_BLOCKERS`
- Added `exp_stageC_tfe_secure_system.py`.
- Track: `secure_system_tfe_lightweight`; it does not replace the full-ConvNet algorithmic pipeline.
- PASS:
  - CIFAR-10 official distilled images are flattened directly.
  - 20 image/label records and Dense parameters are secret-shared under ABY3.
  - Secure forward and two SGD updates execute in TFE.
  - Revealed MSE is bounded and decreases: `0.3895 -> 0.3555 -> 0.3297`.
  - Runtime: approximately `0.57s` for the measured TFE section.
- Reveal boundary:
  - loss;
  - prediction logits;
  - final lightweight model parameters.
- System metrics:
  - input bytes and model parameter bytes are recorded;
  - communication bytes/rounds are `N/A` because local TFE ABY3 exposes no transport counters;
  - LAN/WAN is not claimed from single-process simulation.
- BLOCKED with evidence:
  - exact secure per-example clipping;
  - adaptive secure clipping update;
  - exact MTT ConvNet because TFE lacks its GroupNorm;
  - MPC secure noise because TF 1.15.5 secure-random custom op is missing.
- Found randomness module targets TF 1.15.2 only; NumPy binomial simulation is explicitly not MPC.
- Output:
  - `log/json/stageC_tfe_secure_system_cifar10_20260725_105947.json`

## Phase 4

- Status: `COMPLETED`
- Installed and pinned `opacus==1.5.4`.
- Added:
  - `stage_c_privacy_accounting.py`
  - `tests/test_stage_c_privacy.py`
- Gaussian Fixed-DP and Adaptive-DP now use Opacus `RDPAccountant`.
- Every accounting record includes:
  - sample rate;
  - steps;
  - noise multiplier;
  - delta;
  - `privacy_scope=stage_c_only`;
  - `stage_c_epsilon`;
  - `accountant_type=opacus_rdp`;
  - `noise_mechanism=gaussian`.
- Non-Gaussian binomial simulation returns `stage_c_epsilon=N/A`; Gaussian accounting is not reused.
- Target construction is centralized in `build_reference_target`, whose API accepts no test argument.
- Test data is used only for fixed-interval readonly history and final reporting.
- Required limitation text is emitted automatically.
- Unit tests passed:
  - epsilon increases with steps;
  - epsilon decreases with noise multiplier;
  - Fixed/Adaptive use the same accounting;
  - binomial simulation is not Gaussian-accounted;
  - reference target API has no test input.
- CIFAR-10 two-step validation:
  - Fixed/Adaptive epsilon: `0.5194330147`;
  - sample rate: `0.004`;
  - noise multiplier: `1.2`;
  - delta: `1e-5`.
- Output:
  - `log/json/crossStageConvNet_stageC_cifar10_20260725_110222.json`

## Phase 5

- Status: `COMPLETED`
- Added:
  - `CROSS_STAGE_PIPELINE.md`
  - `exp_cross_stage_report.py`
  - `scripts/run_safe_experiment.sh`
- Safety:
  - one repository-level `flock`;
  - one visible GPU;
  - CPU/BLAS threads fixed to `1`;
  - datasets executed serially;
  - PTY session used because this execution environment reaps detached `nohup` children.
- Three-dataset end-to-end smoke passed:
  - CIFAR-10: `log/json/crossStageConvNet_end2end_cifar10_20260725_110322.json`
  - CIFAR-100: `log/json/crossStageConvNet_end2end_cifar100_20260725_110327.json`
  - Tiny ImageNet: `log/json/crossStageConvNet_end2end_tiny_20260725_110344.json`
- CIFAR-10 complete feasible run:
  - 5 clients;
  - 45 local steps/client;
  - 20 KD steps/client;
  - 60 Stage C steps/mode;
  - Stage B chosen: `KD-Best`;
  - checkpoint signature verified;
  - Fixed/Adaptive RDP epsilon: `0.6958885645`;
  - output: `log/json/crossStageConvNet_end2end_cifar10_20260725_110527.json`.
- Observed limitation:
  - With the current untuned Adaptive configuration, `C_t` grows to about `57.4`
    and accuracy degrades. This result is retained rather than cosmetically tuned;
    future tuning must use the heldout reference set only.
- Figure5 full-ConvNet two-step smoke and PNG generation passed:
  - `expFig/figure5_crossStageConvNet/crossStageConvNet_figure5_cifar10_20260725_110712.png`
- Unified outputs:
  - `log/json/crossStageConvNet_unified_20260725_110656.json`
  - `expTab/crossStageConvNet_unified_20260725_110656.csv`
- Final boundary documentation distinguishes:
  - algorithmic full-ConvNet PyTorch pipeline;
  - representative TFE lightweight system validation;
  - Stage-C-only DP accounting.
