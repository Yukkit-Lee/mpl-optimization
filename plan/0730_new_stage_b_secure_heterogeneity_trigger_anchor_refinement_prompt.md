# Codex Experiment Task: Secure Heterogeneity-Triggered MTT Anchor Refinement

## 0. Task Summary

Implement and evaluate a **Stage-B-only** module named:

> **Secure Heterogeneity-Triggered MTT Anchor Refinement**

Target pipeline:

```text
Official MTT compact local features
        ↓
5 clients independently train lightweight local heads
        ↓
TFE/ABY3 Secure Average
        ↓
securely estimate client label-distribution heterogeneity
        ↓
securely compare the heterogeneity score with a frozen threshold
        ↓
reveal only a one-bit refine / no-refine decision
        ├── low heterogeneity: directly output Secure Average
        └── high heterogeneity: run class-balanced labeled MTT-anchor CE refinement
        ↓
Global Stage-B Initializer
```

Core positioning:

- Secure Average is only the basic aggregation primitive.
- The proposed contribution is the secure heterogeneity-triggered routing policy.
- Official MTT provides both compact client-local training data and compact class-balanced anchors.
- The module stays entirely inside Stage B.
- This task does not run any End-to-End experiment.

The experiment must answer:

1. Does Severe Non-IID make direct Secure Average unreliable?
2. Can a secure scalar label-heterogeneity score predict when full anchor CE is beneficial?
3. Can a frozen trigger retain Secure Average in IID while activating anchor CE in Severe Non-IID?
4. Is the trigger cheaper than always running anchor refinement?
5. Can the decision and selected branch be reproduced in real TFE/ABY3?

---

# 1. Hard Restrictions

Do not run or modify:

- Stage C;
- Adaptive DP-SGD;
- Fixed DP-SGD;
- End-to-End experiments;
- Figure 2;
- T@80 / T@90 / T@95;
- locked holdout;
- official test;
- Full-Data initializer experiments;
- seeds 43 or 44;
- Tiny-ImageNet;
- existing formal True Ours outputs.

Use only:

```text
datasets: CIFAR-10 and CIFAR-100
seed: 42
clients: 5
distributions:
  IID
  Mild Non-IID
  Severe Non-IID
stage: Stage B only
```

The experiment must first pass a low-cost plaintext smoke gate. Only then may it run formal TFE/ABY3 Stage-B validation.

---

# 2. Motivation and Required Causal Logic

Existing completed diagnostics show:

```text
Severe Non-IID:
Secure Average is weak.
Uniform labeled-anchor CE is strongly beneficial.

IID:
Uniform labeled-anchor CE is strongly harmful.

Prediction-disagreement weighting:
fails to identify Average errors and harms utility.
```

Therefore this experiment must not select individual anchors by client prediction disagreement.

The new question is:

> Can a secure scalar estimate of global client label heterogeneity decide when the full class-balanced anchor refinement should be activated?

Required causal chain:

```text
client label skew
        ↓
direct Secure Average becomes unreliable
        ↓
secure heterogeneity score crosses threshold
        ↓
anchor refinement is activated
        ↓
Severe Non-IID recovers while IID avoids destructive over-refinement
```

---

# 3. Branch and Output Isolation

Create or use:

```text
branch:
new-stage-b-secure-heterogeneity-trigger
```

All new artifacts must use isolated directories:

```text
new_scripts/new_stage_b_heterogeneity_trigger/
new_configs/new_stage_b_heterogeneity_trigger/
new_log/new_stage_b_heterogeneity_trigger/
new_expTab/new_stage_b_heterogeneity_trigger/
new_expFig/new_stage_b_heterogeneity_trigger/
new_output/new_stage_b_heterogeneity_trigger/
new_checkpoints/new_stage_b_heterogeneity_trigger/
```

Do not overwrite earlier KD, disagreement-weighted CE, True Ours, or Public-MTT artifacts.

Before work:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

Every JSON/CSV report must record:

```text
source_git_commit
dataset
seed
distribution
split_manifest_path
split_manifest_sha256
model/config SHA256
official_mtt_only = true
full_data_initializer_dependency = false
locked_holdout_read = false
official_test_read = false
```

---

# 4. Reuse Existing Valid Work

Before running anything:

1. inspect all COMPLETE artifacts from:
   - Secure Average;
   - Uniform Anchor CE;
   - previous KD/Consensus;
   - previous disagreement-guided refinement;
2. reuse exact-valid B0/B1 results when:
   - dataset, seed, split, distribution, local-head config, anchor split, evaluation split, and backend match;
3. do not rerun valid expensive work;
4. rerun only when an exact semantic or hash mismatch exists.

Generate:

```text
new_stage_b_heterogeneity_trigger_reuse_audit.json
```

---

# 5. Data and Split Requirements

Use audited Official MTT compact features.

Required mutually exclusive sets:

```text
client-local training pool
class-balanced labeled MTT anchor pool
development evaluation split
```

Preferred existing split:

```text
40 samples/class: client-local pool
8 samples/class: labeled anchor pool
```

Mandatory checks:

- client-local ∩ anchor = empty;
- client-local ∩ development = empty;
- anchor ∩ development = empty;
- locked holdout/test unread;
- sample IDs and SHA256 preserved.

Reuse the existing audited split manifest if valid. Otherwise create:

```text
new_stage_b_heterogeneity_trigger_split_manifest.json
new_stage_b_heterogeneity_trigger_split_audit.md
```

---

# 6. Client Distribution Protocol

Use the same five-client construction across methods.

Evaluate:

```text
IID

Mild Non-IID:
Dirichlet alpha = 0.5

Severe Non-IID:
Dirichlet alpha = 0.1
```

If existing registered splits use different exact alpha values, preserve the existing protocol and document it instead of silently changing it.

For each client \(i\) and class \(c\), define the private class count:

\[
n_{i,c}
\]

and local normalized label distribution:

\[
q_{i,c}
=
\frac{n_{i,c}}
{\sum_{k=1}^{C} n_{i,k}}
\]

The global distribution is:

\[
\bar q_c
=
\frac{\sum_i n_{i,c}}
{\sum_i\sum_k n_{i,k}}
\]

Raw \(n_{i,c}\), \(q_{i,c}\), or per-client histograms must not be revealed in the secure version.

---

# 7. Compared Methods

## B0: Secure Average

```text
local MTT heads
→ Secure Average
→ output averaged initializer
```

## B1: Always-On Uniform Anchor CE

```text
Secure Average
→ class-balanced CE on all labeled MTT anchors
→ output refined initializer
```

This is the positive Severe Non-IID reference and the harmful IID reference.

## B2: Proposed Secure Heterogeneity-Triggered Refinement

```text
Secure Average
→ secure heterogeneity score
→ secure threshold comparison
→ reveal one trigger bit
   0: output Secure Average
   1: run full Uniform Anchor CE
→ output selected initializer
```

## B3: Oracle Routing Upper Bound

For diagnosis only:

```text
choose max(B0, B1) using development accuracy
```

B3 is not a deployable method and must not enter the paper as a formal method.

It quantifies the maximum value of a perfect trigger.

The central comparison is:

```text
B2 vs B0
B2 vs B1
B2 vs B3 oracle regret
```

---

# 8. Secure Heterogeneity Metrics

Test only simple, interpretable metrics based on secret-shared class counts.

Do not use client prediction disagreement.

## H1: Mean Total Variation Distance

\[
H_{\mathrm{TV}}
=
\frac{1}{K}
\sum_{i=1}^{K}
\frac{1}{2}
\sum_{c=1}^{C}
|q_{i,c}-\bar q_c|
\]

Interpretation:

```text
0 = local distributions match the global distribution
larger value = stronger label skew
```

## H2: Mean Normalized Entropy Deficit

\[
H_{\mathrm{ent}}
=
1-
\frac{1}{K}
\sum_{i=1}^{K}
\frac{
-\sum_{c=1}^{C} q_{i,c}\log(q_{i,c}+\varepsilon)
}{
\log C
}
\]

Interpretation:

```text
small = clients cover many classes evenly
large = clients are class-concentrated
```

## H3: Missing-Class Ratio

\[
H_{\mathrm{miss}}
=
\frac{1}{KC}
\sum_{i=1}^{K}
\sum_{c=1}^{C}
\mathbf{1}[n_{i,c}=0]
\]

Interpretation:

```text
fraction of client-class pairs with no local support
```

## H4: Maximum Client TV Distance

\[
H_{\mathrm{maxTV}}
=
\max_i
\frac{1}{2}
\sum_c |q_{i,c}-\bar q_c|
\]

Use H4 only if H1-H3 cannot separate the regimes.

Do not create a learned neural or tree-based trigger. The current one-seed setting is too small for a learned gate.

---

# 9. Trigger Privacy and Execution Mode

The formal low-cost implementation must use:

## Revealed-Bit Trigger Mode

1. each client secret-shares its class counts;
2. TFE/ABY3 computes \(H\);
3. TFE/ABY3 computes:

\[
r = \mathbf{1}[H \ge \tau]
\]

4. reveal only \(r\), where:

```text
r = 0: no refinement
r = 1: run anchor refinement
```

5. do not reveal:
   - per-client counts;
   - per-class local distributions;
   - raw heterogeneity score in the formal secure run.

The paper/privacy report must state:

> The method reveals one coarse aggregate heterogeneity decision bit. It does not provide DP for that bit and does not claim that label-distribution metadata is fully hidden.

An optional secret-select mode may compute both branches and securely select a model, but it does not save runtime and is not required.

---

# 10. Phase A: Plaintext Smoke Test

## 10.1 Scope

Run:

```text
datasets:
  CIFAR-10
  CIFAR-100

seed:
  42

distributions:
  IID
  Mild Non-IID
  Severe Non-IID

backend:
  plaintext float
```

Use cached local heads and B0/B1 results whenever exact-valid.

## 10.2 Smoke Step S0: Reproduce the Crossover

For each dataset/distribution, evaluate or reuse:

```text
B0 Secure Average
B1 Always-On Uniform Anchor CE
```

Record:

- development accuracy;
- macro-F1;
- worst-class accuracy;
- per-class accuracy;
- Stage-B runtime;
- anchor-refinement runtime;
- peak RSS.

The smoke test must first confirm:

```text
IID: B0 is preferable or B1 is clearly harmful
Severe: B1 is clearly preferable
```

If this crossover is not reproduced under exact matched configs, stop and audit earlier results before designing the trigger.

---

# 11. Phase A1: Metric Diagnosis

For H1-H3, compute the scalar score for IID, Mild, and Severe.

Create a table:

```text
Dataset
Distribution
H_TV
H_ent
H_miss
B1_minus_B0_accuracy
B1_minus_B0_worst_class
```

Required diagnosis:

1. Does the metric increase monotonically from IID to Severe?
2. Does larger heterogeneity correspond to larger B1 benefit?
3. Can one threshold separate no-refine IID from refine Severe?
4. How should Mild be routed?

Rank metrics by:

1. correct IID/Severe separation;
2. monotonic relation with B1−B0;
3. simplest secure implementation;
4. numerical stability in fixed point.

---

# 12. Phase A2: Low-Cost Threshold Search

Do not run a large grid.

For each metric, candidate thresholds are only:

```text
midpoint(IID score, Mild score)
midpoint(Mild score, Severe score)
midpoint(IID score, Severe score)
```

For missing-class ratio, also test:

```text
10%
25%
50%
```

The proposed routed output is:

\[
\theta_{\mathrm{B2}}
=
\begin{cases}
\theta_{\mathrm{avg}}, & H < \tau \\
\theta_{\mathrm{anchor}}, & H \ge \tau
\end{cases}
\]

Since B0/B1 are already trained, plaintext threshold screening must reuse outputs rather than retrain models.

---

# 13. Phase A3: Optional Two-Metric Rule

Only if no single metric passes.

Test one simple rule:

```text
refine if:
H_miss >= tau_miss
OR
H_TV >= tau_TV
```

or:

```text
refine if:
H_miss >= tau_miss
AND
H_TV >= tau_TV
```

Do not fit arbitrary coefficients.

Do not add more than these two rules.

---

# 14. Smoke-Test Success Gates

The proposed trigger passes only if all mandatory gates pass.

## Gate 1: Correct Extreme Routing

For both CIFAR-10 and CIFAR-100:

```text
IID → no refinement
Severe Non-IID → refinement
```

## Gate 2: Low Oracle Regret

For every dataset and distribution:

\[
R =
\max(\mathrm{Acc}_{B0},\mathrm{Acc}_{B1})
-
\mathrm{Acc}_{B2}
\]

Require:

```text
mean oracle regret <= 0.5pp
maximum oracle regret <= 1.0pp
```

Mild Non-IID may route either way if the regret gate passes.

## Gate 3: Severe Non-IID Recovery

For both datasets:

```text
B2 severe accuracy
>= B0 severe accuracy + 5pp
```

## Gate 4: IID Safety

For both datasets:

```text
B2 IID accuracy
>= B0 IID accuracy - 0.3pp
```

## Gate 5: Cost Avoidance

On every no-refine distribution:

```text
B2 must skip anchor CE completely
```

Report:

```text
anchor-refinement operations saved
anchor examples not processed
refinement steps not executed
wall-clock time saved
```

Target:

```text
at least 90% of B1 refinement cost avoided
```

when the trigger chooses no refinement.

## Gate 6: Simplicity

Prefer:

```text
one metric + one threshold
```

A two-metric rule is accepted only if every single metric fails.

---

# 15. Phase A4: Confirmation Within Seed42

Because only seed42 is permitted, use a lightweight split-stability check without adding seeds.

Preferred option:

- keep the same Official MTT data and seed42;
- create one independent client-partition confirmation instance using deterministic `partition_id=1`;
- do not change model hyperparameters or threshold;
- apply the frozen threshold from `partition_id=0`.

Run only:

```text
IID
Severe Non-IID
```

for the selected trigger.

If creating another partition violates current split-freeze rules, skip this phase and explicitly label the result as a seed42 case study.

Do not tune on the confirmation partition.

---

# 16. Freeze Before TFE

After smoke success, generate:

```text
new_stage_b_heterogeneity_trigger_config_freeze.json
new_stage_b_heterogeneity_trigger_threshold_freeze.json
new_stage_b_heterogeneity_trigger_freeze_manifest.md
```

Freeze:

- selected metric;
- selected threshold;
- data/split hashes;
- B0/B1 configs;
- anchor CE steps/LR/regularization;
- fixed-point precision;
- seed42;
- trigger disclosure policy;
- TFE timing boundary;
- success gates.

Commit and push the freeze before formal TFE runs.

---

# 17. Phase B: Formal TFE/ABY3 Stage-B Validation

Run only after smoke gates pass and freeze is pushed.

## 17.1 Scope

```text
datasets:
  CIFAR-10
  CIFAR-100

seed:
  42 only

distributions:
  IID
  Severe Non-IID

methods:
  B0 Secure Average
  B1 Always-On Uniform Anchor CE
  B2 Secure Heterogeneity-Triggered Refinement

backend:
  TFE/ABY3
```

Mild is optional and should be omitted if time is limited.

## 17.2 Formal Execution

For B2:

```text
1. secret-share client class counts;
2. securely compute frozen heterogeneity metric;
3. securely compare with frozen threshold;
4. reveal only the trigger bit;
5. if bit=0:
      output Secure Average;
6. if bit=1:
      execute the exact frozen Uniform Anchor CE;
7. output Stage-B initializer;
```

Do not use plaintext histograms to make the formal trigger decision.

## 17.3 Secure Correctness Gates

All must pass:

- backend = TFE/ABY3;
- no plaintext fallback;
- Official-MTT-only lineage;
- Full-Data dependency = false;
- raw histograms not revealed;
- only one trigger bit revealed;
- metric finite;
- threshold comparison correct;
- trigger matches plaintext frozen decision;
- no NaN/overflow;
- model and split hashes pass;
- development-only evaluation;
- locked/test unread.

## 17.4 Formal Utility Gates

For both datasets:

```text
IID:
B2 accuracy within 0.5pp of B0

Severe:
B2 accuracy within 0.5pp of B1
```

## 17.5 Formal Cost Gate

For IID:

```text
B2 must avoid the full anchor-refinement secure runtime.
```

Report:

```text
B0 Stage-B seconds
B1 Stage-B seconds
B2 Stage-B seconds
secure trigger seconds
refinement seconds avoided
```

The trigger is useful only if:

```text
secure trigger cost
<
avoided anchor-refinement cost
```

---

# 18. Runtime and Privacy Accounting

Record:

- local-head training runtime;
- Secure Average runtime;
- secure histogram/count sharing runtime;
- secure heterogeneity-score runtime;
- secure threshold comparison runtime;
- one-bit reveal runtime;
- anchor CE runtime when activated;
- total Stage-B runtime;
- peak RSS.

Do not report communication bytes/rounds unless actually measured.

Privacy statement must distinguish:

```text
MPC hides raw counts and intermediate score.
One aggregate trigger bit is revealed.
No DP guarantee is claimed for that bit.
Stage B remains data-dependent and non-DP.
```

---

# 19. Required Tables

## Table B1: Crossover and Trigger Result

Columns:

```text
Dataset
Distribution
B0 Accuracy
B1 Accuracy
Selected Metric
Metric Value
Threshold
Trigger Bit
B2 Accuracy
Oracle Regret
```

## Table B2: Cost Comparison

Columns:

```text
Dataset
Distribution
Method
Secure Average Seconds
Trigger Seconds
Anchor CE Seconds
Total Stage-B Seconds
Refinement Activated
```

## Table B3: Utility and Class Robustness

Columns:

```text
Dataset
Distribution
Method
Accuracy
Macro-F1
Worst-Class Accuracy
Improved Classes / Total Classes
```

## Table B4: Plaintext vs TFE

Columns:

```text
Dataset
Distribution
Metric
Plaintext Trigger
TFE Trigger
Plaintext Accuracy
TFE Accuracy
Absolute Gap
```

---

# 20. Required Figures

## Figure B1: Heterogeneity Score vs Anchor Benefit

X-axis:

```text
heterogeneity score
```

Y-axis:

```text
B1 accuracy - B0 accuracy
```

Points:

```text
IID
Mild
Severe
```

Separate panels for CIFAR-10 and CIFAR-100.

## Figure B2: Routed Stage-B Accuracy

Compare:

```text
B0 Secure Average
B1 Always Refine
B2 Triggered Refine
B3 Oracle
```

across:

```text
IID / Mild / Severe
```

## Figure B3: Stage-B Runtime

Show:

```text
B0
B1
B2
```

with trigger and refinement breakdown.

## Figure B4: Trigger Decision Diagram

```text
secret-shared class counts
→ secure heterogeneity score
→ threshold comparison
→ one-bit decision
→ Average or Anchor CE
```

---

# 21. Mandatory Output Files

Generate:

```text
new_stage_b_heterogeneity_trigger_reuse_audit.json
new_stage_b_heterogeneity_metric_table.csv
new_stage_b_heterogeneity_threshold_screen.csv
new_stage_b_heterogeneity_smoke_summary.json
new_stage_b_heterogeneity_smoke_report.md
new_stage_b_heterogeneity_trigger_config_freeze.json
new_stage_b_heterogeneity_trigger_threshold_freeze.json
new_stage_b_heterogeneity_trigger_tfe_results.csv
new_stage_b_heterogeneity_trigger_runtime.csv
new_stage_b_heterogeneity_trigger_main_table.csv
new_stage_b_heterogeneity_trigger_claim_audit.md
new_stage_b_heterogeneity_trigger_final_report.md
```

The final report must answer:

1. Which heterogeneity metric best separates IID and Severe?
2. Does the metric predict when anchor CE is beneficial?
3. What threshold is selected?
4. Does B2 avoid IID degradation?
5. Does B2 preserve Severe Non-IID recovery?
6. What is the oracle regret?
7. How much refinement cost is avoided?
8. Does secure trigger overhead remain lower than the avoided cost?
9. Does the trigger reproduce in TFE/ABY3?
10. What information is revealed?
11. Which claims are safe for the paper?
12. Is the method a positive Stage-B module or only a seed42 case study?

---

# 22. Agent Iteration Protocol

For every iteration:

```text
1. inspect COMPLETE artifacts;
2. reuse exact-valid B0/B1 outputs;
3. compute H1-H3 in plaintext;
4. diagnose monotonicity and B1 benefit;
5. screen midpoint thresholds only;
6. evaluate routing regret;
7. change only the metric or threshold;
8. use two-metric rule only if every single metric fails;
9. freeze once smoke gates pass;
10. run TFE seed42 only after freeze;
11. stop after formal Stage-B reports are complete.
```

Every progress update must state:

- dataset;
- distribution;
- backend;
- B0 and B1 accuracy;
- heterogeneity metric;
- threshold;
- trigger decision;
- B2 accuracy;
- oracle regret;
- trigger cost;
- anchor cost avoided;
- whether holdout/test stayed unread;
- next single action.

---

# 23. Stop Conditions

## Positive Stop

Stop when:

- one simple trigger passes all smoke gates;
- the threshold is frozen;
- TFE/ABY3 reproduces IID no-refine and Severe refine;
- runtime and claim audits are complete.

## Negative Stop

Stop when:

- no H1-H3 threshold separates IID and Severe;
- trigger oracle regret exceeds the gate;
- secure trigger overhead exceeds avoided anchor cost;
- TFE trigger differs from plaintext due to numerical instability;
- all registered simple metrics and the two-metric fallback are exhausted.

Do not add prediction-disagreement weighting again.

Do not launch End-to-End experiments.

---

# 24. Paper-Safe Positioning

If successful:

> We introduce a secure heterogeneity-triggered initialization policy. Client class-support statistics are aggregated under secret sharing to derive a coarse heterogeneity decision. Direct Secure Average is retained for homogeneous clients, while a compact class-balanced MTT-anchor refinement is activated only under severe label skew. This avoids destructive over-refinement in IID settings while recovering the initializer under heterogeneous distributions.

Required limitation:

> The current evaluation uses seed42. The trigger reveals one coarse aggregate decision bit, and Stage B is not claimed to satisfy record-level differential privacy.

Do not claim:

- Secure Average is novel;
- the trigger is DP;
- no metadata is revealed;
- End-to-End convergence speedup;
- final test improvement;
- communication savings unless measured.
