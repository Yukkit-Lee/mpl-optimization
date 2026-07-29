# Codex Experiment Task: Heterogeneity-Aware Secure Anchor Refinement

## 0. Core Objective

Implement and evaluate a **Stage-B-only** method named:

> **Heterogeneity-Aware Secure Anchor Refinement**

The method is:

```text
Official MTT compact features
        ↓
5 clients independently train lightweight local heads
        ↓
TFE/ABY3 Secure Average
        ↓
securely compute client prediction disagreement on MTT anchors
        ↓
identify unstable classes / unstable anchor samples
        ↓
perform disagreement-weighted supervised CE refinement
        ↓
Global Stage-B Initializer
```

The key positioning is:

- Secure Average is only the basic aggregation operator;
- disagreement estimation is the diagnosis mechanism;
- labeled-anchor CE is the correction mechanism;
- Official MTT provides both compact client training data and compact calibration anchors;
- the entire method stays inside Stage B;
- this task does **not** run any End-to-End experiment.

The experiment must answer:

1. Does Severe Non-IID make direct Secure Average produce an unstable or biased initializer?
2. Can client disagreement identify the weak samples or weak classes of the averaged initializer?
3. Can disagreement-weighted labeled-anchor CE achieve better accuracy, better worst-class accuracy, or lower refinement cost than using all anchors uniformly?
4. Can the selected method be reproduced in real TFE/ABY3 without plaintext fallback?

---

# 1. Absolute Restrictions

Do not run:

- Stage C;
- Adaptive DP-SGD;
- Fixed DP-SGD;
- End-to-End training;
- Figure 2 regeneration;
- T@80/T@90/T@95;
- locked holdout;
- official test;
- Full-Data initialization;
- seeds 43 or 44;
- Tiny-ImageNet unless explicitly added in a later task.

This task uses only:

```text
datasets: CIFAR-10 and CIFAR-100
seed: 42
primary distribution: Severe Non-IID
secondary safety check: IID
stage: Stage B only
```

The current End-to-End True Ours results must remain untouched.

---

# 2. Data and Lineage Requirements

Every candidate must satisfy:

```text
initializer_source = official_mtt_only
full_data_initializer_dependency = false
locked_holdout_read = false
official_test_read = false
```

Use five clients.

Use only audited Official MTT features.

Create three mutually exclusive subsets:

```text
client-local training pool
labeled MTT anchor pool
development evaluation split
```

Preferred initial split:

```text
40 samples/class: client-local pool
8 samples/class: labeled anchor pool
```

Requirements:

- no anchor sample may appear in client-local training;
- no development sample may appear in client-local training or anchor refinement;
- save sample IDs and SHA256;
- verify every pairwise overlap is zero;
- never use development labels for training;
- never use locked holdout or official test.

Generate:

```text
new_stage_b_anchor_refinement_split_manifest.json
new_stage_b_anchor_refinement_split_audit.md
```

---

# 3. Branch and Output Isolation

Create a dedicated branch:

```text
new-stage-b-secure-anchor-refinement
```

Use isolated directories:

```text
new_scripts/new_stage_b_anchor_refinement/
new_configs/new_stage_b_anchor_refinement/
new_log/new_stage_b_anchor_refinement/
new_expTab/new_stage_b_anchor_refinement/
new_expFig/new_stage_b_anchor_refinement/
new_output/new_stage_b_anchor_refinement/
new_checkpoints/new_stage_b_anchor_refinement/
```

Do not overwrite:

- `new_true_ours`;
- `new_public_mtt_mainline`;
- previous KD/Consensus runs;
- existing formal Stage-B outputs;
- existing Figure 2 assets.

Before running:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

Record the source commit in every JSON and CSV.

---

# 4. Compared Methods

## B0: Secure Average

```text
client-local heads
→ TFE/ABY3 Secure Average
→ global initializer
```

This is the baseline.

## B1: Secure Average + Uniform Anchor CE

```text
Secure Average initializer
→ use all labeled anchors with equal weight
→ CE refinement
→ refined initializer
```

This measures the value of anchor supervision itself.

## B2: Secure Average + Disagreement-Weighted Anchor CE

This is the proposed method.

```text
Secure Average initializer
→ compute client disagreement on each anchor
→ assign higher refinement weight to unstable anchors
→ weighted CE refinement
→ refined initializer
```

## B3: Secure Average + Top-Disagreement Anchor CE

Use only the highest-disagreement anchors.

This tests whether the same utility can be achieved with fewer anchor examples or fewer refinement steps.

## B4: Secure Average + KD

Reuse the already implemented KD variant only as a negative control.

Do not tune KD again.

The primary comparison is:

```text
B2 vs B1
```

The method has independent value only if disagreement guidance provides a clear advantage over uniform anchor CE.

---

# 5. Disagreement Definition

For client \(i\), anchor \(x\), and predicted probability vector \(p_i(x)\):

\[
\bar p(x)=\frac{1}{K}\sum_{i=1}^{K}p_i(x)
\]

Use at least these disagreement metrics:

## D1: Prediction variance

\[
d_{\mathrm{var}}(x)
=
\frac{1}{K}\sum_{i=1}^{K}
\|p_i(x)-\bar p(x)\|_2^2
\]

## D2: Mean Jensen-Shannon divergence

\[
d_{\mathrm{JS}}(x)
=
\frac{1}{K}\sum_{i=1}^{K}
\mathrm{KL}
\left(
p_i(x)
\middle\|
\bar p(x)
\right)
\]

## D3: Vote entropy

Use the entropy of the client argmax-vote distribution.

Do not add more metrics until D1-D3 are tested.

---

# 6. Weighting Functions

Convert disagreement score \(d(x)\) into CE refinement weight \(g(x)\).

Test sequentially:

## W0: Uniform

\[
g(x)=1
\]

## W1: Linear normalized

\[
g(x)
=
\operatorname{clip}
\left(
\frac{d(x)-d_{\min}}
{d_{\max}-d_{\min}+\varepsilon},
0,
1
\right)
\]

## W2: Thresholded

\[
g(x)=
\mathbf{1}[d(x)\geq\tau]
\]

## W3: Soft threshold

\[
g(x)
=
\sigma
\left(
\frac{d(x)-\tau}{s}
\right)
\]

## W4: Class-balanced disagreement

\[
g'(x)
=
g(x)\cdot
\frac{1}{N_{y(x)}}
\]

This prevents high-disagreement majority classes from dominating refinement.

---

# 7. Refinement Objective

Starting from the Secure Average initializer \(\theta_{\mathrm{avg}}\):

\[
\mathcal{L}_{\mathrm{refine}}
=
\frac{
\sum_{x\in A}
g(x)\,
\mathcal{L}_{\mathrm{CE}}
(f_\theta(x),y_x)
}{
\sum_{x\in A}g(x)+\varepsilon
}
+
\mu\|\theta-\theta_{\mathrm{avg}}\|_2^2
\]

The regularization term limits destructive drift.

Record:

- refinement steps;
- learning rate;
- \(\mu\);
- active-anchor ratio;
- total weighted-anchor mass;
- final distance from \(\theta_{\mathrm{avg}}\).

---

# 8. Execution Strategy

The experiment has two phases only:

```text
Phase A: plaintext smoke test
Phase B: formal Stage-B TFE/ABY3 validation
```

No End-to-End phase exists.

---

# 9. Phase A: Low-Cost Plaintext Smoke Test

## 9.1 Initial scope

Run:

```text
CIFAR-10
CIFAR-100
seed42
Severe Non-IID
plaintext float
```

Use cached Official MTT features.

Do not start in TFE.

## 9.2 First diagnostic

Before refinement, measure:

- each client's local development accuracy;
- Secure Average development accuracy;
- macro-F1;
- worst-class accuracy;
- per-class accuracy;
- pairwise prediction disagreement;
- JS divergence;
- vote entropy;
- correlation between disagreement and Secure Average error;
- correlation between disagreement and class-wise weakness.

The most important diagnostic is:

```text
Does higher client disagreement predict Secure Average mistakes?
```

Record:

- AUROC for using disagreement to detect Average errors;
- precision@top-10%, top-20%, top-30% disagreement;
- error rate in low-disagreement vs high-disagreement anchors.

If disagreement does not correlate with Average errors, do not immediately tune CE learning rate. First compare D1, D2, and D3.

---

# 10. Low-Cost Sequential Tuning

Do not run a full Cartesian grid.

Tune one factor at a time.

## S1: Baseline anchor-refinement budget

Run B1 with:

```text
steps = 5, 10, 20, 40
LR = existing Stage-B LR × {0.25, 0.5, 1.0}
mu = 0, 1e-4, 1e-3
```

Use sequential selection:

```text
steps
→ LR
→ mu
```

Select the lowest-cost B1 setting that reaches near-best validation accuracy.

## S2: Disagreement metric

Fix the selected B1 budget.

Compare:

```text
D1 variance
D2 JS divergence
D3 vote entropy
```

Select by:

1. development accuracy;
2. worst-class accuracy;
3. macro-F1;
4. active-anchor cost.

## S3: Disagreement weighting

Compare sequentially:

```text
W1 linear normalized
W2 thresholded
W3 soft threshold
W4 class-balanced disagreement
```

For W2 and W3, test:

```text
top disagreement ratio = 10%, 20%, 30%, 50%
```

Do not test more thresholds unless all four fail clearly.

## S4: Cost-saving top-disagreement refinement

Compare:

```text
all anchors
top 50%
top 30%
top 20%
top 10%
```

Success means one of:

- same accuracy within 0.2pp using at least 30% fewer active anchors;
- same accuracy within 0.2pp using at least 30% fewer refinement steps;
- at least +1.0pp accuracy at the same refinement cost;
- at least +2.0pp worst-class accuracy at the same refinement cost.

## S5: IID safety smoke

Run only the best B2/B3 candidate on IID.

Reject if:

```text
IID accuracy drops by more than 0.5pp relative to B1
```

---

# 11. Smoke-Test Success Gates

Do not proceed to TFE unless at least one proposed candidate passes.

## Gate A: Utility gain

At equal refinement steps and equal anchor budget:

```text
B2 or B3 >= B1 + 1.0pp development accuracy
```

or:

```text
B2 or B3 >= B1 + 2.0pp worst-class accuracy
```

## Gate B: Cost reduction

At nearly equal utility:

```text
accuracy loss <= 0.2pp
```

and at least one:

```text
active anchors reduced by >= 30%
refinement steps reduced by >= 30%
estimated weighted CE operations reduced by >= 30%
```

## Gate C: Error targeting

The selected disagreement score must show:

```text
high-disagreement anchors have a clearly higher Average error rate
```

and preferably:

```text
error-detection AUROC > 0.60
```

## Gate D: IID safety

```text
IID drop <= 0.5pp
```

## Gate E: Lineage

```text
Official-MTT-only
Full-Data dependency = false
locked/test unread
```

A candidate may pass through either:

```text
utility improvement
```

or:

```text
equivalent utility with lower refinement cost
```

This is important: the method does not have to beat uniform CE in raw accuracy if it can reach the same result using substantially fewer anchors or steps.

---

# 12. Smoke-Test Iteration Logic

If no improvement appears, diagnose before changing parameters.

## Case 1: Disagreement does not predict errors

Try:

```text
D1 → D2 → D3
```

Then test class-level disagreement:

```text
mean disagreement per true class
class-wise Average error
```

If sample-level disagreement fails but class-level disagreement correlates with weak classes, switch to class-weighted CE.

## Case 2: Disagreement predicts errors, but weighted CE is worse

Possible causes:

- weights too concentrated;
- too few anchors selected;
- high-disagreement anchors include noisy/outlier samples;
- learning rate too high;
- refinement drifts too far from Average.

Tune:

```text
weight cap = 2x, 4x, 8x median weight
mu = 1e-4, 1e-3, 1e-2
top ratio = 20%, 30%, 50%
LR = 0.25x, 0.5x
```

## Case 3: Weighted CE equals uniform CE

Try to demonstrate cost reduction:

```text
same accuracy with fewer anchors
same accuracy with fewer steps
```

This is an acceptable positive result.

## Case 4: Only worst classes improve

Keep the candidate if:

```text
worst-class gain >= 2pp
macro-F1 improves
average accuracy does not drop by more than 0.5pp
```

## Case 5: CIFAR-10 works but CIFAR-100 fails

Try class-balanced disagreement weighting and class-level rather than sample-level weighting.

Do not redesign the entire method.

---

# 13. Phase B: Formal Stage-B TFE/ABY3 Validation

Proceed only after smoke success.

Run:

```text
datasets: CIFAR-10 and CIFAR-100
seed: 42 only
distribution: Severe Non-IID
methods:
  B0 Secure Average
  B1 Uniform Anchor CE
  best B2/B3 disagreement-guided refinement
backend: TFE/ABY3
```

Do not run seeds 43 or 44.

Do not run IID in formal TFE unless required for a correctness check.

## 13.1 Secure implementation ladder

Validate in order:

```text
P1 plaintext float
P2 plaintext fixed-point emulation
P3 TFE secure client prediction aggregation
P4 TFE secure disagreement computation
P5 TFE weighted CE refinement
P6 final TFE Stage-B initializer
```

Record numerical deviation at every level.

## 13.2 Secure correctness gates

All must pass:

- backend is TFE/ABY3;
- no plaintext fallback;
- Official-MTT-only lineage;
- Full-Data dependency false;
- finite logits;
- finite disagreement;
- finite weights;
- no overflow;
- no NaN;
- weight normalization valid;
- selected-anchor mask valid;
- development-only evaluation;
- locked holdout/test unread;
- model and split SHA match.

## 13.3 Formal success gate

The TFE result must preserve the smoke-test conclusion.

Pass if at least one:

```text
TFE B2/B3 >= TFE B1 + 0.8pp accuracy
```

or:

```text
TFE B2/B3 >= TFE B1 + 1.5pp worst-class accuracy
```

or:

```text
accuracy within 0.3pp of B1
and active anchor / refinement cost reduced by >= 30%
```

---

# 14. Runtime Accounting

Record Stage-B-only runtime:

- client local training;
- Secure Average;
- anchor inference;
- disagreement computation;
- anchor selection;
- weighted CE refinement;
- total Stage-B runtime;
- peak RSS.

Generate:

```text
new_stage_b_anchor_refinement_runtime_breakdown.csv
new_stage_b_anchor_refinement_resource_summary.csv
```

Do not report:

- Stage-C runtime;
- End-to-End runtime;
- communication bytes or rounds unless genuinely measured.

---

# 15. Required Tables

## Table B1: Main Stage-B comparison

Columns:

```text
Dataset
Distribution
Method
Dev Accuracy
Macro-F1
Worst-Class Accuracy
Active Anchors
Refinement Steps
Stage-B Runtime
Backend
```

Methods:

```text
Secure Average
Uniform Anchor CE
Disagreement-Weighted Anchor CE
Top-Disagreement Anchor CE
KD negative control
```

## Table B2: Disagreement diagnosis

Columns:

```text
Dataset
Metric
Error-Detection AUROC
Top-10% Error Precision
Top-20% Error Precision
High-vs-Low Disagreement Error Gap
```

## Table B3: Plaintext-to-TFE consistency

Columns:

```text
Dataset
Method
Plaintext Accuracy
Fixed-Point Accuracy
TFE Accuracy
Absolute Gap
```

---

# 16. Required Figures

## Figure B1: Stage-B initializer quality

X-axis:

```text
Average
Uniform Anchor CE
Disagreement-Weighted CE
Top-Disagreement CE
KD
```

Y-axis:

```text
development accuracy
```

Separate panels:

```text
CIFAR-10
CIFAR-100
```

## Figure B2: Disagreement vs Average error

Plot:

```text
anchor disagreement score
vs
Secure Average correctness / loss
```

Show whether high disagreement identifies weak anchors.

## Figure B3: Utility-cost tradeoff

X-axis:

```text
active anchor ratio or refinement steps
```

Y-axis:

```text
development accuracy
```

Compare:

```text
Uniform CE
Disagreement-guided CE
```

## Figure B4: Worst-class improvement

Compare class-wise accuracy before and after refinement.

---

# 17. Mandatory Output Files

Generate:

```text
new_stage_b_anchor_refinement_smoke_registry.csv
new_stage_b_anchor_refinement_smoke_summary.json
new_stage_b_anchor_refinement_diagnostic_report.md
new_stage_b_anchor_refinement_tfe_validation.csv
new_stage_b_anchor_refinement_main_table.csv
new_stage_b_anchor_refinement_claim_audit.md
new_stage_b_anchor_refinement_final_report.md
```

The final report must answer:

1. Does disagreement correlate with Secure Average errors?
2. Which disagreement metric works best?
3. Does disagreement-guided CE beat uniform CE?
4. If not more accurate, does it achieve the same accuracy with fewer anchors or fewer steps?
5. Does it improve worst-class accuracy?
6. Does it preserve IID behavior?
7. Does the result reproduce in TFE/ABY3?
8. What extra Stage-B runtime does it add?
9. Is the method a positive module, a cost-saving optimization, or a negative result?
10. Which claims are safe for the paper?

---

# 18. Agent Progress Loop

For every iteration:

```text
1. inspect existing COMPLETE artifacts;
2. skip valid completed runs;
3. run the next lowest-cost plaintext smoke test;
4. compare against Secure Average and Uniform Anchor CE;
5. diagnose whether the bottleneck is:
   disagreement quality,
   weighting concentration,
   class imbalance,
   refinement budget,
   or secure numerics;
6. change only one factor;
7. update the registry;
8. continue until a success gate passes or the registered search is exhausted;
9. only then run formal TFE seed42.
```

Each progress update must state:

- current dataset;
- backend;
- distribution;
- method;
- changed factor;
- baseline B0;
- baseline B1;
- candidate B2/B3;
- accuracy gain;
- worst-class gain;
- active-anchor reduction;
- runtime;
- whether holdout/test stayed unread;
- next single factor.

---

# 19. Stop Conditions

Stop positively when:

- one smoke candidate passes a predefined gate;
- the result is reproduced in TFE/ABY3 seed42;
- all reports and figures are generated.

Stop negatively when:

- D1-D3 all fail to identify Average errors;
- W1-W4 fail to improve utility or reduce cost;
- IID safety fails;
- TFE does not preserve the plaintext conclusion;
- all registered low-cost tuning families are exhausted.

Do not add an unlimited second search stage.

---

# 20. Paper Positioning

If successful, the supported statement is:

> Under severe client heterogeneity, prediction disagreement identifies unstable regions of the securely averaged initializer. By focusing labeled MTT-anchor refinement on these regions, the proposed method improves weak-class utility or reaches the same initializer quality with fewer refinement operations.

Do not claim:

- Secure Average is novel;
- the method improves End-to-End convergence;
- the method improves final test accuracy;
- the method provides End-to-End record-level DP;
- communication savings unless measured.

If the result is negative, report:

> Labeled MTT-anchor refinement remains effective, but client disagreement does not provide stable utility or cost benefits beyond uniform supervised refinement under the evaluated compact-data protocol.
