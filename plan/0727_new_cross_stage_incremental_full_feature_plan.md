# Incremental Formal Experiment Plan  
## Preserve Existing Results, Replace End-to-End Baseline, Add Limited External Comparison

> This plan is incremental.  
> Do **not** delete, invalidate, or unnecessarily rerun experiments that are already complete.  
> Reuse all verified results, histories, checkpoints, CSV/JSON reports, figures, and test evaluations from the existing `new-secure-feature-pipeline` branch unless a specific new experiment requires a different configuration.

---

# 1. Existing Completed Work Must Be Preserved

The following results already exist and must remain part of the paper evidence:

## Stage A

- Public MTT results for CIFAR-10, CIFAR-100, and Tiny ImageNet
- Official MTT vs equal-size Random Subset
- Shared 2048-D encoder/feature validation
- Existing Stage-A figures and CSV/JSON reports

## Stage B

- TFE/ABY3 secure implementation
- Convergence experiments
- IID / Mild / Severe Non-IID experiments
- Existing three-seed results
- Existing Stage-B figures and tables

## Stage C

- 54/54 formal Fixed/Adaptive DP runs
- Valid accountant outputs
- Existing privacy-utility results
- Existing negative result showing Adaptive did not consistently outperform Fixed
- Existing Module-C reports, figures, and checkpoints

## Existing End-to-End

- Current Random-Subset baseline vs Ours results
- 18/18 runs
- 72 read-only final test evaluations
- Existing end-to-end figures, tables, histories, and reports

## TFE System Evaluation

- 9/9 system runs
- Runtime, batch-size, and memory results
- Existing TFE reports and figures

These completed experiments must **not** be removed from the project.  
They may remain as:

- ablation evidence;
- supplementary material;
- internal comparison;
- system validation;
- negative-result analysis.

Do not rerun them unless:

1. a required artifact is missing;
2. the run cannot be reproduced;
3. the data format is incompatible with the new report;
4. a fairness issue makes the result unusable.

---

# 2. Current Hardware

Server:

- GPU: NVIDIA RTX 5090
- CPU: 25 vCPU Intel(R) Xeon(R) Platinum 8470Q

## Performance Policy

Use the available hardware aggressively and efficiently.

Requirements:

- use CUDA for all supported PyTorch workloads;
- maximize RTX 5090 utilization;
- increase batch size when memory allows;
- use mixed precision where numerically safe;
- use pinned memory, prefetching, caching, and parallel data loading when helpful;
- allow CPU multi-threading and multiprocessing;
- do not impose artificial one-thread limits;
- use all useful CPU cores without causing duplicated jobs or uncontrolled process creation;
- monitor GPU utilization, GPU memory, CPU load, RAM, and disk;
- use tmux for all long-running jobs;
- save checkpoints and resume interrupted jobs;
- do not reduce seeds, epochs, model size, or formal evaluation quality merely to save time.

Before launching formal jobs, record:

```bash
nvidia-smi
lscpu
free -h
df -h
```

---

# 3. File and Branch Rules

Continue on:

```text
new-secure-feature-pipeline
```

Do not overwrite existing formal artifacts.

All newly created incremental files should use the prefix:

```text
new_
```

Recommended directories:

```text
new_log/new_incremental_comparison/
new_expFig/new_incremental_comparison/
new_expTab/new_incremental_comparison/
new_cache/new_incremental_comparison/
new_checkpoints/new_incremental_comparison/
new_scripts/
```

---

# 4. Main Change: New End-to-End Figure

The main end-to-end figure must now compare only:

```text
Full Data Feature Model
vs
Ours
```

The old Random-Subset baseline must remain in the project as an ablation result, but it is no longer the primary end-to-end baseline.

---

# 5. New End-to-End Baseline

## 5.1 Full Data Feature Model

Pipeline:

```text
Full Training Dataset
→ Same Shared ConvNet Encoder
→ Full 2048-D Feature Set
→ Secure Average Initialization
→ TFE/ABY3 Secure Fixed DP-SGD
→ Final Model
```

This baseline represents a conventional feature-based secure MPL pipeline that uses the complete training feature set.

## 5.2 Ours

Pipeline:

```text
Official MTT Distilled Images
→ Same Shared ConvNet Encoder
→ Compact 2048-D Distilled Feature Set
→ TFE/ABY3 Secure Consensus Initialization
→ TFE/ABY3 Secure Fixed DP-SGD
→ Final Model
```

Use Secure Fixed DP-SGD in both methods for the main comparison.

Do not use Adaptive DP as the main distinguishing factor because the existing Stage-C ablation did not show a stable Adaptive advantage.

---

# 6. Fairness Requirements for the New End-to-End Experiment

Both methods must use:

- the same encoder architecture;
- the same encoder checkpoint policy;
- the same 2048-D feature definition;
- the same feature normalization rule;
- the same linear classifier;
- the same TFE/ABY3 backend;
- the same loss function;
- the same fixed-point precision;
- the same optimizer;
- the same target epsilon and delta;
- the same evaluation protocol;
- the same hardware;
- the same wall-clock timing boundary;
- at least 3 seeds.

The intended difference is:

```text
Full Data Feature Model:
complete feature set + secure average initialization

Ours:
MTT compact feature set + secure consensus initialization
```

Do not change multiple unrelated hyperparameters between the methods.

---

# 7. New End-to-End Metrics

For CIFAR-10, CIFAR-100, and Tiny ImageNet, report:

- final test accuracy;
- validation/reference accuracy;
- wall-clock secure training time;
- time to target accuracy;
- secure steps to target accuracy;
- secret-shared examples processed to target accuracy;
- total processed examples;
- peak GPU memory;
- peak process-tree RAM;
- epsilon and delta;
- mean ± standard deviation over 3 seeds.

## Target Accuracy Rule

For each dataset, choose a target accuracy using validation/reference results before the final test is read.

The target must be reachable by both methods.

Do not choose different targets for the two methods.

---

# 8. Main End-to-End Figure Design

Generate a paper-ready main figure with three subplots:

```text
CIFAR-10
CIFAR-100
Tiny ImageNet
```

Each subplot contains exactly two curves:

```text
Full Data Feature Model
Ours
```

Preferred axes:

```text
x-axis: Secure Training Time
y-axis: Accuracy
```

Also generate an auxiliary version:

```text
x-axis: Secret-Shared Examples Processed
y-axis: Accuracy
```

Output:

```text
new_figure2_full_feature_vs_ours.png
new_figure2_full_feature_vs_ours.pdf
new_figure2_processed_examples_vs_accuracy.png
new_figure2_processed_examples_vs_accuracy.pdf
```

The figure should emphasize:

> Ours uses a much smaller secure training set and a stronger initialization to reach useful accuracy with lower secure workload.

It is acceptable if Full Data Feature Model eventually achieves higher final accuracy.  
In that case, report:

- accuracy retention;
- time speedup at a common target accuracy;
- secure-workload reduction at the same target.

Do not claim that Ours has higher maximum accuracy unless the measured results support it.

---

# 9. Stage A, Stage B, Stage C, and TFE Existing Experiments

Do not delete or replace the existing experiments.

## Stage A

Reuse existing:

```text
Full / Random Subset / Official MTT
```

If the Full Dataset curve is missing from the current Stage-A figure, add only the missing Full Dataset runs or reporting step.

Do not rerun Official MTT and Random Subset when existing histories are complete and compatible.

## Stage B

Reuse:

- existing convergence experiments;
- existing Non-IID experiments;
- existing TFE correctness evidence.

Do not reduce them to a new single-dataset experiment.  
The already completed three-dataset evidence remains valid.

## Stage C

Reuse all existing Fixed/Adaptive results.

In the paper:

- Fixed DP-SGD is the primary Stage-C method;
- Adaptive is retained as an ablation/negative result;
- do not hide the negative result;
- do not rerun the 54-run grid unless a concrete implementation error is found.

## TFE System Evaluation

Reuse all 9 completed system runs.

Do not rerun unless the new Full Data Feature Model requires a separate system-scale measurement.

---

# 10. External Method Comparison

The external comparison should include only 3–4 methods that are relevant and can be reported responsibly.

Recommended set:

```text
PEA
CaPC
DDGauss
Ours
```

A system-oriented method such as CryptGPU may be discussed separately in text rather than inserted into the main accuracy table when the evaluation settings are not aligned.

## 10.1 Data Collection Priority

For each external method:

### First priority: use published results

Search the original paper, official supplementary material, and official repository for:

- test accuracy;
- dataset;
- epsilon/delta;
- model;
- number of parties;
- threat model;
- runtime;
- hardware;
- communication;
- whether the result is end-to-end DP.

Use the published value only when the setting is sufficiently close to the current experiment.

Label it clearly:

```text
Reported by original paper
```

### Second priority: derive from official released logs

Use official repository logs or released result files if the paper table is incomplete.

### Third priority: run the method

Only run an external method when:

1. no usable result is reported;
2. official code is available;
3. the dataset and privacy setting can be aligned;
4. the implementation effort is reasonable;
5. the result materially strengthens the paper.

Do not spend large amounts of time reproducing a method solely to fill a table cell.

---

# 11. External Comparison Table

Create one concise table.

Suggested columns:

| Method | Year | Dataset | Test Accuracy | ε | δ | MPC/Secure Aggregation | End-to-End Record DP | Non-IID | Source |
|---|---:|---|---:|---:|---:|---|---|---|---|

Rules:

- use 3–4 external methods only;
- do not mix incomparable numbers without footnotes;
- mark unavailable values as `N/A`;
- distinguish paper-reported results from locally reproduced results;
- do not imply strict superiority when hardware, model, privacy scope, or dataset differ;
- include a short comparability note for each external method.

Recommended final rows:

```text
PEA
CaPC
DDGauss
Ours
```

If CaPC or DDGauss has no sufficiently comparable result and reproduction is impractical, replace it with another closely related method with publicly reported, comparable data.

Do not include SecureML, SecureNN, CrypTen, and CryptGPU merely to make the table longer.  
Those systems can remain in Related Work or a separate capability table.

---

# 12. CryptGPU Efficiency Comparison

Do not reproduce CryptGPU in this round unless there is a strong, low-cost, directly comparable official result.

Preferred treatment:

- cite the CryptGPU result reported in its original paper or in PEA;
- explain in text that direct secure CNN training is computationally expensive;
- use it as motivation for compact-feature secure training;
- do not divide its reported runtime by the current local runtime unless the target accuracy, model, hardware, protocol, and timing scope are aligned.

A permitted textual statement format:

```text
Prior systems such as CryptGPU demonstrate the high cost of directly training deep neural networks under MPC. In contrast, our approach reduces the secure workload by replacing raw-image deep training with compact distilled features and a linear secure model. Because the published CryptGPU settings differ in model, hardware, and protocol, we use its reported runtime as contextual evidence rather than as a strict head-to-head speedup.
```

No new CryptGPU experiment is required by this plan.

---

# 13. Privacy Interpretation

Use the existing privacy interpretation.

## Stage A

```text
epsilon = N/A
```

Official MTT images are public benchmark artifacts.

## Stage B

```text
MPC confidentiality only
epsilon = N/A
```

## Standalone Module C with data-independent initialization

The released model may carry valid record-level DP according to the verified accountant.

## End-to-End

Because the Stage-B initializer is data-dependent and not DP:

- report Stage-C epsilon as conditional gradient-mechanism accounting;
- do not claim complete end-to-end record-level DP for the final selected-initializer model.

---

# 14. Incremental Execution Checklist

```text
[ ] Audit and reuse all existing completed artifacts
[ ] Implement Full Data Feature Model baseline
[ ] Run Full Data Feature Model for 3 datasets × 3 seeds
[ ] Reuse or rerun Ours only when timing/configuration must be aligned
[ ] Generate new two-curve End-to-End figures
[ ] Compute time-to-target and workload reduction
[ ] Complete 3–4 method external comparison table
[ ] Add CryptGPU contextual textual comparison
[ ] Preserve all previous Stage A/B/C/TFE results
[ ] Generate final incremental report
```

---

# 15. Avoid Unnecessary Reruns

Before launching any job, search the existing manifest and artifacts.

Reuse a run when all of the following match:

- dataset;
- seed;
- encoder;
- feature cache;
- model;
- initialization;
- backend;
- epsilon/delta;
- batch size;
- optimizer;
- training budget;
- evaluation schedule;
- timing scope.

Only rerun Ours if the previous Ours run is not directly comparable with the new Full Data Feature Model baseline.

If only timing boundaries differ, prefer a minimal aligned rerun rather than rebuilding every stage.

---

# 16. Experiment Quality Requirements

Forbidden:

- smoke-test results in final figures;
- one-seed conclusions;
- test-set tuning;
- cherry-picking seeds;
- overwriting negative results;
- silently changing privacy scope;
- presenting paper-reported external numbers as locally reproduced.

Required:

- at least 3 seeds for new main experiments;
- complete histories;
- checkpoints;
- CSV/JSON;
- publication-quality PNG/PDF;
- mean ± standard deviation;
- exact reproduction commands;
- explicit source for every external result.

---

# 17. Git and Long-Running Jobs

Use tmux for all long jobs.

Recommended session:

```text
new_incremental_full_feature
```

Commit after each major stage:

```text
Add full-data feature end-to-end baseline
Add aligned full-feature vs ours evaluation
Add external comparison table
Add final incremental paper figures
```

Push to the existing branch after tests pass.

---

# 18. Final Agent Report

At completion, report:

1. Existing results reused
2. New experiments actually run
3. Full Data Feature Model results
4. Ours aligned results
5. Time-to-target comparison
6. Secure workload reduction
7. Final test accuracy comparison
8. External method data sources
9. External methods locally reproduced, if any
10. CryptGPU textual comparison
11. New figure/table paths
12. New JSON/CSV paths
13. Exact commands
14. Supported paper claims
15. Claims that must still be avoided
16. Remaining limitations

Do not report the project as complete until the new Full Data Feature Model vs Ours comparison and the external comparison table are both finished.
