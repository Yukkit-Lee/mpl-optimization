# Follow-up Plan: Finalize End-to-End Accuracy and Establish Adaptive DP Utility

## 0. Current State

The latest diagnosis has established:

- The previous accuracy gap was mainly caused by the original Secure Consensus initializer.
- Official MTT features themselves are not the main bottleneck.
- Adaptive DP noise is not the main source of the observed accuracy loss.
- Replacing pure Consensus initialization with Average initialization raises validation accuracy to approximately the Full Data Feature Model level.
- The improved configuration has not yet completed a fully frozen three-seed final-test closure for all datasets.
- Adaptive DP-SGD has not yet been shown to significantly outperform a properly tuned Fixed DP-SGD baseline.

This plan has two goals:

1. Complete a trustworthy End-to-End final result with high test accuracy.
2. Test whether Adaptive DP-SGD provides a real, statistically consistent utility advantage over Fixed DP-SGD under a meaningful, non-floor training regime.

Do not change the test set or use test results for tuning.

---

# 1. Preserve Existing Work

Do not delete, overwrite, or unnecessarily rerun:

- Existing Stage A experiments
- Existing Stage B convergence and Non-IID experiments
- Existing Module C 54/54 Fixed/Adaptive grid
- Existing old End-to-End experiments
- Existing Full Data Feature Model runs
- Existing Ours runs
- Existing TFE system measurements
- Existing checkpoints, caches, JSON, CSV, logs, and figures

All new work must be stored separately:

```text
new_log/new_end2end_adaptive_followup/
new_expTab/new_end2end_adaptive_followup/
new_expFig/new_end2end_adaptive_followup/
new_checkpoints/new_end2end_adaptive_followup/
new_cache/new_end2end_adaptive_followup/
```

---

# 2. Evaluation Protocol

Use three strictly separated metrics:

## Reference

May be used only for internal Stage B scoring and diagnostics.

## Independent Validation

Use only this split for:

- hyperparameter selection
- training-budget selection
- initializer selection
- clipping selection
- checkpoint selection
- early stopping

The validation split must not participate in:

- Stage B weighting
- KD scoring
- candidate selection
- consensus construction

## Test

Use only once after a configuration is frozen.

Forbidden:

- test-based tuning
- test-based seed selection
- test-based checkpoint selection
- returning to tuning after reading final test results

Generate a split-integrity audit before training:

```text
new_followup_split_audit.json
new_followup_split_audit.md
```

---

# 3. Goal A: Complete the Improved End-to-End Result

## 3.1 Freeze the Current Improved Candidate

Current candidate:

```text
Feature source: Official MTT
Initializer: Secure Average
Stage C: Adaptive DP-SGD
Target privacy budget: ε ≤ 3
δ = 1e-5
Batch size = 16
Learning rate = 0.002
Steps:
  CIFAR-10 = 10
  CIFAR-100 = 20
  Tiny ImageNet = 30
Seeds = 42, 43, 44
```

Before any further change:

1. Finish Tiny ImageNet seeds 43 and 44.
2. Confirm CIFAR-10 and CIFAR-100 have three complete frozen models.
3. Perform one final read-only test evaluation for all nine improved models.
4. Compute mean ± standard deviation.
5. Compare against the Full Data Feature Model using the same test pipeline.

Outputs:

```text
new_improved_ours_final_test.json
new_improved_ours_final_test.csv
new_improved_ours_vs_full_data.csv
```

Required table:

| Dataset | Full Data Test Acc | Improved Ours Test Acc | Gap | Retention |
|---|---:|---:|---:|---:|

Do not claim the End-to-End improvement is complete until all nine models have been evaluated.

---

# 4. Goal B: Rescue the Stage B Contribution

Pure Consensus currently harms independent validation accuracy. Do not simply rename Average initialization as Consensus.

The objective is to determine whether a regularized consensus mechanism can preserve generalization and provide value under IID or Non-IID conditions.

## 4.1 Baselines

Use:

```text
B0: Secure Average
B1: Original Secure Consensus
B2: Shrinkage Consensus
B3: Temperature-Smoothed Consensus
B4: Weight-Clipped Consensus
B5: Top-k Consensus
```

## 4.2 Shrinkage Consensus

Define:

```text
W_init = (1 - β) · W_average + β · W_consensus
```

Test:

```text
β = 0.00, 0.10, 0.25, 0.50, 0.75, 1.00
```

Here:

- `β=0` is pure Average
- `β=1` is pure Consensus

## 4.3 Temperature Smoothing

For reference-based client weights, test:

```text
T = 1, 2, 4, 8
```

Higher temperature should reduce overconfident client weighting.

## 4.4 Weight Clipping

Cap any single client weight at:

```text
0.30, 0.40, 0.50
```

Then renormalize.

## 4.5 Top-k Consensus

Test:

```text
top-k = 2, 3, all clients
```

Aggregate selected clients rather than choosing one dominant candidate.

## 4.6 Evaluation

Screen first on:

```text
CIFAR-10 seed 42
CIFAR-100 seed 42
Severe Non-IID split
```

Use independent validation only.

Advance at most two candidates per dataset.

A Stage B candidate is useful only if it satisfies at least one of:

1. IID validation accuracy is not lower than Average by more than 0.5 pp and Severe Non-IID accuracy is higher.
2. Validation accuracy is at least 1 pp above Average.
3. Time-to-target is lower without harming final validation accuracy.

If no candidate satisfies these rules, retain Secure Average as the main initializer and demote Consensus to an ablation/negative result.

Outputs:

```text
new_stage_b_rescue_screen.csv
new_stage_b_rescue_formal.csv
new_stage_b_weight_statistics.json
```

---

# 5. Goal C: Test Whether Adaptive DP Improves Utility

The old Module C comparison suffered from a floor effect caused by a data-independent random initializer and a very short secure-training regime.

The new comparison must use a strong initializer and a configuration where accuracy is not near random guessing.

## 5.1 Fair Comparison

Compare:

```text
Fixed DP-SGD
Adaptive DP-SGD
```

Keep identical:

- dataset
- feature source
- initializer
- classifier
- batch size
- learning rate
- steps
- privacy upper bound
- delta
- backend
- fixed-point precision
- seeds
- validation/test protocol

## 5.2 Fixed Clipping Grid

A single arbitrary fixed clipping norm is not enough.

Screen:

```text
C_fixed = 0.25, 0.50, 1.00, 2.00, 4.00
```

The Adaptive method must be compared against the best validation-selected Fixed clipping baseline, not only against `C=1`.

## 5.3 Adaptive Clipping Grid

Screen:

```text
initial C = 0.50, 1.00, 2.00
target clipped fraction = 0.10, 0.30, 0.50
EMA/update rate = conservative, current, aggressive
```

All configurations must satisfy the same privacy upper bound:

```text
ε ≤ 3
```

## 5.4 Privacy Budgets

First screen at:

```text
ε = 3
```

If Adaptive shows promise, run formal comparisons at:

```text
ε = 1
ε = 3
```

Do not expand to more budgets until the main effect is confirmed.

## 5.5 Datasets

Screen:

```text
CIFAR-10 seed 42
CIFAR-100 seed 42
```

Then run three seeds on the best settings.

Add Severe Non-IID only if Adaptive is hypothesized to help under heterogeneous gradient norms.

Tiny ImageNet is optional for this claim and should not delay the main comparison.

## 5.6 Required Diagnostics

Record per step:

- clipping norm
- unclipped gradient norm distribution
- clipped fraction
- noise standard deviation
- update norm
- loss
- independent validation accuracy
- fixed-point saturation/overflow
- accountant epsilon

Generate:

```text
new_fixed_vs_adaptive_diagnostics.csv
new_fixed_vs_adaptive_validation_curves.png
new_fixed_vs_adaptive_clipping_curves.png
```

## 5.7 Success Criterion

Adaptive DP may be claimed to improve utility only if:

1. It exceeds the best tuned Fixed DP result by at least 1 percentage point on mean validation/test accuracy; and
2. It improves at least two of three seeds; and
3. The privacy upper bound and training budget are matched; and
4. The gain is not caused by a different initializer or data split.

Preferred evidence:

```text
paired per-seed comparison
mean ± std
paired bootstrap confidence interval
```

If the gain is below 1 pp or inconsistent, report Adaptive as functional but not significantly better.

---

# 6. Goal D: Optimize End-to-End Accuracy Without Confounding Changes

After Stage B and Adaptive screening, optimize only one factor at a time.

## 6.1 Training Steps

Screen:

```text
CIFAR-10: 10, 20, 40
CIFAR-100: 20, 40, 80
Tiny: 30, 60, 120
```

Recompute the accountant for every setting.

## 6.2 Batch Size

Screen:

```text
16, 32, 64
```

Use 128 only if memory and accountant behavior remain reasonable.

## 6.3 Learning Rate

Screen:

```text
0.5×, 1×, 2× current LR
```

## 6.4 Selection Rule

A configuration advances only if:

- independent validation improves;
- ε remains within the same bound;
- there is no numerical instability;
- runtime remains acceptable;
- improvement persists across checkpoints.

Do not optimize reference accuracy alone.

---

# 7. Tiny ImageNet Recovery

Tiny ImageNet remains the highest-risk dataset.

After completing the missing seeds, diagnose in this order:

1. Secure Average vs best regularized Consensus
2. 30 vs 60 vs 120 steps
3. learning rate
4. batch size
5. MTT budget

For MTT budget, test only when more official public assets are available:

```text
8000
12000
16000
```

Do not introduce DP-MTT.

The Tiny target is:

```text
Improved Ours test accuracy close to the Full Data Feature Model,
or at minimum retention ≥ 85%
```

If this target cannot be reached without greatly increasing secure workload, report the tradeoff rather than hiding the result.

---

# 8. Numeric Correctness Ladder

If validation improvement does not transfer from plaintext to TFE, run:

```text
P1: PyTorch float, no DP
P2: PyTorch float, Fixed DP
P3: PyTorch float, Adaptive DP
P4: Plaintext fixed-point emulation
P5: TFE secure Fixed DP
P6: TFE secure Adaptive DP
```

Compare:

- logits
- loss
- gradients
- clipping
- noise statistics
- parameter updates
- validation predictions

Prioritize fixing:

- scaling
- truncation
- overflow
- secure division
- batch averaging
- duplicated noise variance

Do not hide numeric defects with learning-rate tuning.

---

# 9. Execution Strategy

## Screening

Use:

```text
seed = 42
```

No test reads.

## Formal Runs

Run:

```text
seeds = 42, 43, 44
```

Only after configuration freeze.

## Hardware

Use:

- NVIDIA RTX 5090
- 25 vCPU Intel Xeon Platinum 8470Q

Requirements:

- do not impose one-thread limits;
- use CPU multiprocessing where useful;
- use GPU proxy experiments for fast screening;
- reuse feature caches and memmaps;
- run long jobs in tmux;
- monitor GPU, CPU, RAM, and disk;
- avoid concurrent jobs that compete for the same memory-heavy cache.

---

# 10. Priority Order

```text
Priority 1: Complete improved End-to-End three-seed final test
Priority 2: Rescue or formally reject the Consensus contribution
Priority 3: Compare Adaptive against best tuned Fixed clipping
Priority 4: Optimize steps, batch size, and learning rate
Priority 5: Tiny ImageNet recovery
Priority 6: Numeric correctness ladder if needed
```

Do not start external-method reproduction or SPU work in this plan.

---

# 11. Required Outputs

```text
new_followup_split_audit.json
new_improved_ours_final_test.json
new_improved_ours_final_test.csv
new_improved_ours_vs_full_data.csv
new_stage_b_rescue_screen.csv
new_stage_b_rescue_formal.csv
new_stage_b_weight_statistics.json
new_fixed_vs_adaptive_screen.csv
new_fixed_vs_adaptive_formal.csv
new_fixed_vs_adaptive_diagnostics.csv
new_end2end_hyperparameter_screen.csv
new_tiny_recovery_screen.csv
new_end2end_adaptive_followup_report.md
```

Figures:

```text
new_improved_ours_vs_full_data_test.png/pdf
new_stage_b_rescue_validation.png/pdf
new_fixed_vs_adaptive_validation.png/pdf
new_fixed_vs_adaptive_clipping.png/pdf
new_end2end_accuracy_runtime_pareto.png/pdf
```

---

# 12. Final Report Questions

The final report must answer:

1. Did improved Ours match Full Data on final test, not only validation?
2. Did the result hold across all three seeds?
3. Was the original Consensus mechanism overfitting internal reference metrics?
4. Can a regularized Consensus outperform or match Average under Severe Non-IID?
5. Does Adaptive DP outperform the best tuned Fixed DP baseline?
6. Is any Adaptive gain consistent across seeds?
7. Is the gain obtained under the same privacy bound and training budget?
8. Which component now limits accuracy?
9. Did Tiny ImageNet recover?
10. Which claims are supported and which remain unsupported?

---

# 13. Forbidden Actions

- Do not use test data for tuning.
- Do not select seeds based on test accuracy.
- Do not claim Adaptive superiority against an untuned Fixed baseline.
- Do not call pure Average initialization a Consensus contribution.
- Do not hide the old Consensus negative result.
- Do not alter multiple major factors in one experiment.
- Do not exceed the privacy budget to obtain accuracy.
- Do not add DP-MTT.
- Do not rerun the entire existing mainline without a documented need.
