# Agent Experiment Task: MTT-Anchor KD/Consensus for Stage-B Non-IID Initialization

## Runtime override — 2026-07-29

The user's latest instruction overrides every three-seed expansion in this
document for the current task:

- every designed experiment in Phases A, B, and C runs seed 42 only;
- Phase-B confirmation uses the same single-seed Severe/IID gates as the smoke
  stage and must not report mean, standard deviation, paired-seed stability, or
  `2/3 seeds`;
- Phase-C TFE/ABY3 validation also runs seed 42 only;
- Tiny-ImageNet remains optional as specified below and is not added merely to
  create more runs;
- only after the complete single-seed task shows that the data/method is
  effective may the agent generate a separate plan for seeds 43 and 44;
- generating that extension plan does not authorize running seeds 43 and 44.

All other lineage, search-budget, evaluation, security, reporting, and forbidden
action requirements remain unchanged.

## 0. Mission

Design, diagnose, tune, and validate a **Stage-B-only** knowledge-distillation/consensus initialization method under client heterogeneity.

The target method is:

```text
Official MTT compact features
→ heterogeneous client-local lightweight heads
→ MTT-anchor logit aggregation
→ consensus teacher
→ distilled global student initializer
```

The experiment must answer:

1. Do heterogeneous clients contain complementary knowledge that Secure Average cannot use?
2. Can KD/Consensus fuse this complementary knowledge into a stronger Stage-B initializer?
3. Can the method improve initialization quality under Mild/Severe Non-IID without materially harming IID performance?
4. Can the selected KD/Consensus design be reproduced under real TFE/ABY3 secure computation?

This task is **not an End-to-End experiment**.

Do not run Stage C, Adaptive DP-SGD, Fixed DP-SGD, final test, locked holdout, Figure 2, T@Target, or any full pipeline evaluation.

---

# 1. Hard Constraints

## 1.1 Data lineage

All Stage-B candidates must use:

```text
initializer_source = official_mtt_only
full_data_initializer_dependency = false
```

Forbidden:

- loading any Full-Data initializer;
- loading hybrid Full-Data-init + MTT checkpoints;
- using official test or locked holdout;
- silently falling back to plaintext in the formal TFE stage;
- reusing development samples inside local training or anchor construction.

## 1.2 Evaluation scope

This task evaluates only:

- Stage-B initializer quality;
- client knowledge diversity;
- teacher quality;
- teacher-to-student retention;
- IID/Mild/Severe Non-IID robustness;
- Stage-B runtime and memory;
- plaintext-to-TFE consistency.

This task must not claim:

- End-to-End speedup;
- Stage-C convergence reduction;
- final model accuracy;
- complete record-level DP;
- Adaptive superiority.

A better initializer may motivate later convergence claims, but this task alone does not prove them.

## 1.3 Search discipline

Do not stop after the first negative smoke test.

Continue through the ordered diagnostic and tuning stages below until one of the following occurs:

1. a pre-defined success gate is passed; or
2. all pre-registered search families are exhausted and evidence shows saturation.

Do not search indefinitely on one development split. After the predefined search budget is exhausted, produce an honest negative result and stop.

---

# 2. Branch and Output Isolation

Create or use a dedicated branch:

```text
new-stage-b-mtt-anchor-kd
```

Write all new artifacts under:

```text
new_scripts/new_stage_b_kd/
new_configs/new_stage_b_kd/
new_log/new_stage_b_kd/
new_expTab/new_stage_b_kd/
new_expFig/new_stage_b_kd/
new_output/new_stage_b_kd/
new_checkpoints/new_stage_b_kd/
```

Do not overwrite existing True Ours, Full Data, Public-MTT, or previous Consensus experiments.

Before any run:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

Record the source commit in every result JSON.

---

# 3. Method Definition

## 3.1 Baseline B1: MTT-only Secure Average

Each client trains a lightweight classifier on its local Official-MTT feature subset.

The baseline initializer is:

```text
local heads
→ parameter-wise average
→ global initializer
```

Use the same local initialization, local optimizer, feature scaler, and evaluation split as the proposed method.

## 3.2 Diagnostic B2: Uniform Logit Ensemble

Each local head predicts logits on an MTT anchor set.

Construct:

```text
teacher probability
= uniform average of client soft predictions
```

This tests whether prediction-space fusion is better than parameter averaging.

## 3.3 Main B3: Class-Aware MTT-Anchor Consensus KD

For client \(i\), anchor example \(x\), and class \(c\):

\[
z_i(x)=f_i(x)
\]

Estimate client reliability per class:

\[
r_{i,c}
\]

Convert reliability to normalized class-aware weights:

\[
w_{i,c}
=
\frac{\exp(r_{i,c}/\tau_w)}
{\sum_j\exp(r_{j,c}/\tau_w)}
\]

Build the teacher:

\[
p_T(x)
=
\sum_i
w_{i,c(x)}
\operatorname{softmax}\left(\frac{z_i(x)}{T}\right)
\]

Train the global student from the Secure-Average initializer:

\[
\mathcal{L}
=
(1-\lambda)\mathcal{L}_{CE}
+
\lambda T^2\mathcal{L}_{KD}
+
\mu\|\theta-\theta_{avg}\|_2^2
\]

The output student is the Stage-B initializer.

## 3.4 Candidate reliability functions

Test in the following order:

1. uniform;
2. per-class anchor accuracy;
3. confidence;
4. negative entropy;
5. calibrated confidence;
6. class-coverage-aware reliability.

Do not add more reliability families until these are exhausted.

---

# 4. MTT Data and Anchor Split

Use the audited Official MTT assets.

For each class, create mutually exclusive subsets:

```text
local-client pool
anchor pool
development evaluation
```

Preferred initial split:

```text
40 samples/class: client-local training pool
8 samples/class: anchor pool
```

Requirements:

- anchor examples must not enter client-local training;
- development examples must not enter client-local training or anchor KD;
- locked holdout and official test remain unread;
- save sample IDs and SHA256 in a split manifest;
- verify pairwise intersection size is zero.

Generate:

```text
new_stage_b_kd_split_manifest.json
new_stage_b_kd_split_audit.md
```

---

# 5. Non-IID Protocol

Evaluate:

```text
IID
Mild Non-IID: Dirichlet alpha = 0.5
Severe Non-IID: Dirichlet alpha = 0.1
```

Use five clients.

For each split, record:

- samples per client;
- classes per client;
- minimum/maximum class coverage;
- class imbalance;
- empty classes;
- split hash.

Primary target setting:

```text
Severe Non-IID
```

IID is a safety constraint, not the main optimization target.

---

# 6. Phase A: Plaintext Smoke Test

## 6.1 Purpose

Use plaintext only to cheaply determine whether useful complementary knowledge exists and whether the KD design is viable.

Initial scope:

```text
datasets: CIFAR-10 and CIFAR-100
seed: 42
distribution: Severe Non-IID
backend: plaintext float
```

Do not start with Tiny-ImageNet.

## 6.2 Mandatory diagnostics before tuning

For every smoke configuration, record:

### Client diversity

- pairwise parameter cosine distance;
- pairwise prediction disagreement;
- mean Jensen-Shannon divergence;
- per-client per-class accuracy;
- per-client class coverage;
- local-head development accuracy.

### Teacher upper bound

Evaluate directly without student training:

- Secure Average model;
- uniform ensemble teacher;
- class-aware ensemble teacher;
- best local head;
- oracle class-wise teacher for diagnosis only.

### Student retention

After KD, record:

- student development accuracy;
- worst-class accuracy;
- macro-F1;
- teacher-to-student gain retention;
- distance from Secure Average initializer.

## 6.3 Mandatory diagnosis logic

### Case 1: clients are nearly identical

Trigger condition:

```text
prediction disagreement ≈ 0
or
JS divergence ≈ 0
```

Then do not tune teacher temperature first.

Tune client diversity:

```text
local epochs: 5, 10, 20, 40
local learning rate: 0.5x, 1x, 2x
local weight decay: 0, 1e-4, 1e-3
final synchronization: disabled
```

The goal is not maximum local accuracy. The goal is:

```text
usable local heads + measurable complementary knowledge
```

### Case 2: teacher is not better than Secure Average

Tune in this order:

1. client local epochs;
2. anchor size;
3. reliability function;
4. teacher temperature;
5. weight temperature;
6. clipping/capping of client weights.

Do not tune student learning rate while the teacher itself is worse than the baseline.

### Case 3: teacher is better but student loses the gain

Tune:

```text
KD temperature T: 1, 2, 4, 8
KD weight lambda: 0.25, 0.5, 0.75
average regularization mu: 0, 1e-4, 1e-3
student steps: 20, 50, 100
student LR: 0.5x, 1x, 2x
```

Target:

```text
student retains at least 80% of teacher gain over Secure Average
```

### Case 4: only one class or majority classes improve

Tune:

- class-balanced CE;
- class-balanced KD;
- reliability floor;
- maximum client weight;
- worst-class-aware selection criterion.

Do not accept a method that raises average accuracy by sacrificing minority classes severely.

---

# 7. Ordered Smoke-Test Search

Run one factor at a time.

## Stage S1: Local diversity screen

Fixed teacher: uniform.

Search:

```text
local epochs = 5, 10, 20, 40
local LR multiplier = 0.5, 1, 2
```

Keep at most the best two settings.

## Stage S2: Anchor screen

Search:

```text
anchor samples/class = 4, 8, 12
```

Keep the best anchor size using teacher accuracy and worst-class accuracy.

## Stage S3: Teacher screen

Search:

```text
reliability =
  uniform
  per-class accuracy
  confidence
  negative entropy
  calibrated confidence

teacher temperature T =
  1, 2, 4, 8

weight temperature tau_w =
  0.5, 1, 2, 4
```

Do not evaluate the full Cartesian product.

Use sequential tuning:

```text
reliability
→ T
→ tau_w
```

## Stage S4: Weight stabilization

Search:

```text
maximum client weight = 0.30, 0.40, 0.50, 1.00
reliability shrinkage to uniform = 0, 0.25, 0.50
```

## Stage S5: Student KD screen

Search sequentially:

```text
lambda
→ mu
→ student steps
→ student LR
```

Keep at most two finalist configurations per dataset.

## Stage S6: Mild Non-IID and IID safety check

Evaluate finalists on:

```text
IID
Mild Non-IID
Severe Non-IID
```

Reject configurations with clear IID degradation.

---

# 8. Smoke-Test Success Gates

A candidate passes the smoke stage only if all mandatory gates pass.

## 8.1 Primary effect gate

Under Severe Non-IID, at least one:

```text
initializer development accuracy
>= Secure Average + 1.0 percentage point
```

or:

```text
worst-class accuracy
>= Secure Average + 2.0 percentage points
```

## 8.2 IID safety gate

```text
IID development accuracy drop <= 0.5 percentage point
```

## 8.3 Teacher/student gate

```text
teacher > Secure Average
student retains >= 80% of teacher gain
```

## 8.4 Diversity validity gate

The selected setting must show non-zero and interpretable client diversity.

## 8.5 Lineage gate

```text
Official-MTT-only
Full-Data dependency = false
locked holdout/test unread
```

If no configuration passes after all registered smoke stages, produce a negative report. Do not silently invent more search families.

---

# 9. Phase B: Plaintext Confirmation

Only after smoke success.

Scope:

```text
datasets: CIFAR-10 and CIFAR-100
seeds: 42, 43, 44
distributions: IID, Mild, Severe
methods:
  Secure Average
  Uniform Logit KD
  Best Class-Aware Consensus KD
```

Tiny-ImageNet remains optional.

Report:

- mean ± sample standard deviation;
- paired seed differences;
- macro-F1;
- worst-class accuracy;
- teacher gain;
- student retention;
- Stage-B wall time.

Confirmation success:

```text
at least 2/3 seeds improve in Severe Non-IID
mean gain >= 1.0pp accuracy
or mean worst-class gain >= 2.0pp
IID loss <= 0.5pp
```

If confirmation fails, do not move to TFE.

---

# 10. Phase C: TFE/ABY3 Secure Validation

Only after plaintext confirmation passes.

This phase validates the secure implementation of the selected Stage-B method.

Scope:

```text
datasets: CIFAR-10 and CIFAR-100
seeds: start with 42
methods:
  Secure Average
  selected Consensus KD
backend: TFE/ABY3
```

Run seeds 43/44 only if seed42 passes correctness and utility gates.

## 10.1 Secure implementation ladder

Validate in order:

```text
P1 plaintext float
P2 plaintext fixed-point emulation
P3 TFE secure logit aggregation
P4 TFE secure teacher construction
P5 TFE secure or securely fed KD student
```

Record numerical differences at each level.

## 10.2 Secure correctness gates

- no plaintext fallback;
- TFE/ABY3 backend confirmed;
- model and split hashes match;
- logits/probabilities finite;
- weight normalization finite;
- no fixed-point overflow;
- no NaN;
- secure result direction matches plaintext;
- development-only evaluation;
- locked holdout/test unread.

## 10.3 Formal secure result

A secure finalist passes if:

```text
TFE Consensus KD
>= TFE Secure Average + 0.8pp
```

under Severe Non-IID, or preserves at least 80% of the confirmed plaintext gain.

Also report the extra Stage-B secure cost.

Do not run Stage C to compensate for a weak Stage-B result.

---

# 11. Runtime and Resource Logging

For every run, record:

- data loading time;
- local training time;
- anchor inference time;
- teacher aggregation time;
- student KD time;
- secure graph build time;
- secure session runtime;
- peak RSS;
- output model hash.

Generate:

```text
new_stage_b_kd_runtime_breakdown.csv
new_stage_b_kd_resource_summary.csv
```

Do not report communication bytes or rounds unless actually measured.

---

# 12. Required Figures and Tables

## Main Stage-B ablation table

```text
Dataset
Distribution
Method
Init Dev Accuracy
Macro-F1
Worst-Class Accuracy
Stage-B Runtime
```

## Figure B1: Non-IID initializer quality

Methods:

```text
Secure Average
Uniform Logit KD
Class-Aware Consensus KD
```

X-axis:

```text
IID / Mild / Severe
```

Y-axis:

```text
Stage-B development accuracy
```

## Figure B2: Client diversity and teacher gain

Show:

- prediction disagreement;
- JS divergence;
- teacher gain over Average;
- student retained gain.

## Figure B3: Runtime breakdown

Show:

```text
local training
anchor inference
secure aggregation
student KD
```

## Supplementary

- class-wise accuracy;
- weight distributions;
- temperature sweep;
- anchor-size sweep;
- plaintext vs TFE numerical differences.

---

# 13. Mandatory Reports

Generate:

```text
new_stage_b_kd_smoke_registry.csv
new_stage_b_kd_smoke_summary.json
new_stage_b_kd_diagnostic_report.md
new_stage_b_kd_plaintext_confirmation.csv
new_stage_b_kd_tfe_validation.csv
new_stage_b_kd_claim_audit.md
new_stage_b_kd_final_report.md
```

The final report must answer:

1. Did clients contain complementary knowledge?
2. Was the ensemble teacher better than Secure Average?
3. Did the student preserve teacher gains?
4. Which reliability rule worked best?
5. Did the method help Severe Non-IID?
6. Did it harm IID?
7. Was the gain stable across seeds?
8. Was the result reproduced in TFE/ABY3?
9. What extra Stage-B time did KD require?
10. Which claims are supported?
11. Which claims remain unsupported?
12. Should Consensus KD remain in the paper as a positive module, optional extension, or negative result?

---

# 14. Forbidden Actions

Do not:

- run any End-to-End experiment;
- run Stage C;
- run Adaptive or Fixed DP-SGD;
- regenerate Figure 2;
- read locked holdout or official test;
- load Full-Data initializers;
- tune on TFE before plaintext viability is shown;
- expand every smoke candidate to three seeds;
- keep searching indefinitely after the registered families are exhausted;
- select configurations by a single favorable class or seed;
- hide negative Consensus results;
- claim faster private convergence from Stage-B-only evidence;
- call Secure Average “Consensus KD”;
- modify existing formal True Ours artifacts.

---

# 15. Agent Execution Loop

For each iteration:

```text
1. Read existing COMPLETE artifacts.
2. Skip completed valid work.
3. Run the next one-factor smoke diagnostic.
4. Compare against Secure Average.
5. Determine whether the bottleneck is:
   client diversity,
   teacher quality,
   student retention,
   class imbalance,
   or secure numerics.
6. Change only the relevant factor.
7. Update the registry and diagnostic report.
8. Commit valid code/results.
9. Continue until a smoke success gate passes or the registered search is exhausted.
```

Every progress update must state:

- what changed;
- why it changed;
- baseline result;
- candidate result;
- whether the gain is average-class or worst-class;
- whether IID safety holds;
- whether test/holdout remained unread;
- whether the result is plaintext or TFE;
- the next single factor to change.

---

# 16. Final Paper Positioning

If successful, the supported Stage-B statement is:

> Under heterogeneous client distributions, MTT-anchor class-aware consensus distillation fuses complementary prediction-space knowledge and produces a stronger Stage-B initializer than direct secure parameter averaging.

Do not claim End-to-End convergence acceleration unless a separate future Stage-C experiment verifies it.

If unsuccessful, report:

> Direct Secure Average remains the most reliable initializer under the tested compact-data protocol, while the evaluated consensus-distillation variants do not provide stable gains under strict disjoint evaluation.
