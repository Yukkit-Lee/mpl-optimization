# Formal Experiment Progress

## Execution Contract

- Completed: 2026-07-25 12:30 CST
- Branch: `cross-stage-refactor`
- Final run ID: `20260725_121001`
- Formal experiments are separate from every `--quick` / smoke output.
- Device policy: one visible GPU (`CUDA_VISIBLE_DEVICES=0`).
- CPU policy: OMP/MKL/OpenBLAS/NumExpr/PyTorch threads fixed to `1`.
- Scheduling: datasets, splits, methods and seeds serial under repository `flock`.
- Adaptive tuning boundary: heldout reference set only; every pilot records `test_evaluation_enabled=false`.
- Test policy: readonly reporting at predetermined intervals and final step only.
- Result integrity: all outcomes are retained; no test-guided tuning, seed deletion or fabricated advantage.

## Formal Configuration

- Datasets: CIFAR-10, CIFAR-100, Tiny ImageNet.
- Seeds: 42, 43, 44.
- Module B splits: IID, Mild Non-IID, Severe Non-IID.
- Module B methods: NoKD, KD-Avg, KD-ReferenceWeighted, KD-Best.
- Module B curve length: 800 optimizer steps, 10 recorded evaluation points.
- Stage B local/KD training: 300/100 optimizer steps; all client ConvNets share one seed-specific base initialization.
- Module C methods: Plain, Fixed-DP, Adaptive-DP.
- Module C: 300 steps, DP batch 16, learning rate 0.001, noise multiplier 1.2, delta 1e-5.
- Adaptive search: gamma in [0.1, 0.2, 0.3], eta in [0.03, 0.05], 40-step reference-only pilots.
- End-to-End methods: Random subset, Distilled only, Distilled+Stage B, +Fixed-DP, +Adaptive-DP.
- Exact run count: 237 serial sub-runs (108 B curves, 27 C curves, 45 E2E curves, 54 reference pilots, 3 TFE validations).

## Completion Checklist

- [x] Resolved configuration, estimated run count and output paths printed before training
- [x] Module B: 3 datasets x 3 splits x 3 seeds x 4 methods
- [x] Module B: complete 800-step Figure 5 curves
- [x] Module C: 3 datasets x 3 seeds x 3 methods
- [x] Fixed/Adaptive privacy settings equality audited
- [x] Adaptive tuning performed on heldout reference only
- [x] `C_t` divergence diagnosed and bounded by reference gradient quantiles
- [x] End-to-End: 3 datasets x 3 seeds x 5 methods
- [x] TFE lightweight secure-system validation: 3 datasets
- [x] JSON / CSV / TXT / 9 PNG outputs verified
- [x] Mean +/- std summaries generated
- [x] Project progress updated
- [ ] Git commit and push completed

## Main Formal Findings

### Module B

- CIFAR-10 IID: KD-Avg=0.5624, KD-RW=0.5624, NoKD=0.5611; KD reaches the reference target in 80 steps versus NoKD 320 steps.
- CIFAR-10 Severe: NoKD final accuracy remains higher (0.5489) than KD-RW (0.4749), so no universal final-accuracy gain is claimed.
- CIFAR-100 IID: KD-Avg=0.3066 versus NoKD=0.3060; Severe NoKD remains higher (0.2647 versus KD-RW=0.2575).
- Tiny IID/Mild/Severe: KD-Avg or KD-RW is slightly higher than NoKD (0.0611/0.0617/0.0544 versus 0.0593/0.0605/0.0509).
- Interpretation: the formal evidence supports faster initialization convergence and dataset-dependent robustness, not a universal final-accuracy dominance claim.

### Module C

- CIFAR-10: Adaptive-DP=0.2608 +/- 0.0056, Fixed-DP=0.2553 +/- 0.0044, both epsilon=2.8391.
- CIFAR-100: Adaptive-DP=0.0388 +/- 0.0059, Fixed-DP=0.0386 +/- 0.0019, both epsilon=0.5488.
- Tiny: Adaptive-DP=0.0131 +/- 0.0014, Fixed-DP=0.0162 +/- 0.0018, both epsilon=0.4585.
- `C_t` no longer reaches the old unbounded value 57.4, but it still grows during the 300-step formal training: CIFAR-10 about 3.03 to 6.78, CIFAR-100 about 34.14 to 42.11, and Tiny about 29.60 to 37.73.
- The reference pilot often decreases `C_t`, while the later formal trajectory increases as training-gradient norms shift. This diagnosed non-stationarity and the Tiny utility result remain limitations; no universal Adaptive improvement is claimed.

### End-to-End

- CIFAR-100 and Tiny show a clear Stage-B initialization benefit over Distilled-only under the fixed 300-step budget (0.1797 versus 0.0591; 0.0427 versus 0.0104).
- CIFAR-10 Distilled-only remains higher than Distilled+Stage B (0.3658 versus 0.3257).
- DP methods report Stage-C epsilon only; Stage A and Stage B are excluded from epsilon.

## TFE Secure-System Boundary

- Track label: `secure_system_tfe_lightweight`.
- CIFAR-10/CIFAR-100/Tiny use exactly 50/100/200 samples and 10 secure steps.
- Model: ABY3 `Flatten+Dense`, not the full MTT ConvNet.
- Reveals: loss, prediction logits and final model parameters.
- Communication bytes/rounds: unavailable from the local TFE runtime and reported as N/A.
- Full GroupNorm ConvNet, exact secure per-example clipping, Adaptive-DP and secure randomness are BLOCKED, not replaced by plaintext claims.

## Outputs

- Main JSON: `log/json/formalCrossStage_20260725_121001.json`
- Raw CSV: `expTab/formalCrossStage_20260725_121001.csv`
- TXT: `log/txt/formalCrossStage_20260725_121001.log`
- Paper tables: `expPaperTab/formalModuleB_20260725_121001.csv`, `formalModuleC_20260725_121001.csv`, `formalEnd2End_20260725_121001.csv`
- Figures: 9 PNG files under `expFig/formal_cross_stage/` with run ID `20260725_121001`.
- TFE JSON: `stageC_tfe_secure_system_{cifar10,cifar100,tiny}_*.json` under `log/json/`.

## Reliability Notes

- Detached jobs are not reliable in the current executor, so the run used a persistent PTY plus repository `flock`.
- Partial JSON was written after each seed and used to resume without duplicate rows.
- Matplotlib was moved to a standalone CPU-only process that imports no PyTorch/CUDA modules; all 9 plots then completed.
- Earlier diagnostic run `20260725_113359` and interrupted calibration runs are retained but are not the final paper run.
