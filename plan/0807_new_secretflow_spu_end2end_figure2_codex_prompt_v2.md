# Codex Task V2: Add SecretFlow-SPU End-to-End Main-Figure Experiments

## 0. Overall Goal

Add **SecretFlow-SPU** as the second MPC/MPL backend of the current paper.

The goal is deliberately narrow:

> Reproduce the paper's end-to-end Full-vs-Ours efficiency/utility experiment on SecretFlow-SPU, using the **best validated Stage-B version from the completed ablation experiments**, and generate an SPU version of the three-panel Figure 2 for CIFAR-10, CIFAR-100, and Tiny-ImageNet.

Do **not** repeat the paper's ablation experiments on SPU.

The SPU experiment is only for the end-to-end/main-figure evidence.

The preferred paper story is:

```text
TFE / ABY3:
existing Figure 2 and existing ablations

SecretFlow-SPU:
end-to-end Full vs Ours only
```

The best validated current Ours Stage-B semantics to port to SPU are:

```text
compact features
→ secure client-specific lightweight-head training
→ secure parameter aggregation
→ class-balanced labeled compact-anchor CE refinement
→ refined global initializer
→ Adaptive Stage C
```

Do NOT force SPU to imitate an older TFE Stage-B variant merely because an older Figure 2 may have been produced with that variant.

Do NOT modify the existing TFE implementation or its formal result artifacts.

---

# 1. Very First Action: Report Existing Figure-2 Semantics in Chat Only

Before any installation, code modification, smoke test, or formal experiment command is executed:

1. inspect the existing repository and Figure-2 source artifacts using read-only inspection;
2. determine what Stage-B semantics the **current TFE Figure 2** actually uses for:
   - Full;
   - Ours;
3. report the result **only in the Codex conversation/chat**.

The report should be concise:

```text
Current TFE Figure-2 audit:

CIFAR-10 Full Stage B:
...

CIFAR-10 Ours Stage B:
...

CIFAR-100 Full Stage B:
...

CIFAR-100 Ours Stage B:
...

Tiny-ImageNet Full Stage B:
...

Tiny-ImageNet Ours Stage B:
...

Does current Figure-2 Ours include the latest Secure Average + Anchor CE?
YES / NO

Important:
This audit is informational only.
SPU will use the best validated Stage-B version from the completed ablations regardless of the old Figure-2 semantics.
```

Do NOT create a semantic-audit file for this item unless later needed for debugging.

Do NOT alter the TFE code or TFE Figure-2 artifacts after this audit.

The purpose of the audit is only to tell me what the current TFE figure represents.

---

# 2. Formal SPU Experiment Scope

Run all three datasets:

```text
CIFAR-10
CIFAR-100
Tiny-ImageNet
```

For the first SPU version, use only:

```text
seed = 42
```

for **all three datasets**.

Tiny-ImageNet is also seed42 only.

Formal methods:

```text
Full
Ours
```

Therefore the formal SPU matrix is exactly six runs:

```text
1. CIFAR-10     Full  seed42
2. CIFAR-10     Ours  seed42
3. CIFAR-100    Full  seed42
4. CIFAR-100    Ours  seed42
5. Tiny-ImageNet Full seed42
6. Tiny-ImageNet Ours seed42
```

No extra seeds in this task.

No SPU ablation experiments.

No SPU KD/Consensus/disagreement/heterogeneity-trigger experiments.

No SPU Fixed-vs-Adaptive ablation.

No SPU precision ablation unless required strictly to make the formal SPU implementation numerically stable; if so, treat it as backend engineering, not a paper ablation.

---

# 3. Existing TFE Code Is Read-Only

The current TFE code, configs, logs, checkpoints, plots, and formal result artifacts are **read-only reference material**.

Never:

- edit existing TFE source files;
- change existing TFE configs;
- overwrite existing TFE checkpoints;
- rerun TFE unless explicitly requested later;
- overwrite current Figure 2;
- overwrite current Figure-2 plot data;
- rewrite existing TFE logs.

If shared utility code is needed, copy the minimum required logic into the new SPU namespace instead of modifying the TFE implementation.

All SPU work must be isolated.

Use:

```text
branch:
new-secretflow-spu-end2end

conda environment:
spu_e2e
```

New directories:

```text
new_scripts/new_secretflow_spu/
new_configs/new_secretflow_spu/
new_log/new_secretflow_spu/
new_output/new_secretflow_spu/
new_expTab/new_secretflow_spu/
new_expFig/new_secretflow_spu/
new_checkpoints/new_secretflow_spu/
```

Before implementation work, record:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

---

# 4. Official SecretFlow/SPU Documentation

Use the official documentation as the primary installation/API reference:

```text
SPU Quick Start:
https://secret-flow.antgroup.com/docs/spu/en/getting_started/quick_start.html
```

Also inspect the current official installation and SecretFlow-SPU device documentation when needed.

Do not assume an old API from memory.

The installed version is authoritative.

Record the exact versions used.

The formal paper backend must be identified as:

```text
SecretFlow-SPU
```

Prefer the SecretFlow device/SPU API for the formal implementation rather than treating a minimal quickstart helper as the paper backend.

---

# 5. Phase A — Server and CPU Hardware Audit

Before installing SecretFlow, inspect the server:

```bash
uname -a
cat /etc/os-release
lscpu
nproc
free -h
df -h
numactl --hardware || true
conda info --envs
```

Determine:

- CPU model;
- sockets;
- physical cores;
- logical CPUs;
- SMT;
- NUMA nodes;
- memory;
- available disk;
- current CPU affinity restrictions;
- whether BLAS/OpenMP libraries are available.

Generate:

```text
new_spu_hardware_audit.json
new_spu_hardware_audit.md
```

The SPU implementation must be optimized for the actual server, not for a generic laptop.

---

# 6. Phase B — New Isolated Conda Environment

Do not install SecretFlow into the TFE environment.

Create a new environment.

Start with a Python version compatible with the current official SecretFlow/SPU release.

A possible initial attempt is:

```bash
conda create -n spu_e2e python=3.10 -y
conda activate spu_e2e
python -m pip install --upgrade pip setuptools wheel
pip install -U secretflow
```

But do not insist on Python 3.10 if the current official package requires another compatible version.

If installation fails:

1. read the current official compatibility requirements;
2. diagnose the actual package conflict;
3. create a compatible isolated environment;
4. do not downgrade unrelated system libraries;
5. do not break the TFE environment.

Record:

```text
python
secretflow
spu
jax
jaxlib
numpy
protobuf
ray
```

and all relevant transitive versions.

Generate:

```text
new_spu_environment_manifest.txt
new_spu_pip_freeze.txt
new_spu_install_report.md
```

---

# 7. Phase C — Official Quickstart Smoke Test

Before porting the paper code, prove that SPU itself works.

Run a minimal smoke sequence following the installed-version official tutorial.

## S1. Import smoke

Verify imports for:

```text
secretflow
spu
jax
jax.numpy
```

## S2. Three-party SPU device smoke

Create a local three-party SPU configuration.

Prefer ABY3 when supported by the installed SecretFlow/SPU version.

Record:

```text
protocol
field/ring
fixed-point settings
party count
```

## S3. Secure arithmetic smoke

Run a tiny secure:

```text
addition
comparison
matrix multiplication
```

and compare final revealed values with plaintext.

## S4. Secure linear-learning smoke

Run a tiny secure linear-head computation:

```text
X, y
→ logits
→ loss/update
→ reveal final W,b
```

This smoke must complete before porting the full pipeline.

Generate:

```text
new_spu_quickstart_smoke.json
new_spu_quickstart_smoke.md
```

---

# 8. Phase D — Reuse Existing Feature Assets Aggressively

Do NOT rerun Stage A or the feature encoder if reusable features exist.

The first priority is to reuse the feature assets already produced for the current paper.

Inspect the existing Full and compact/Ours feature caches for:

```text
CIFAR-10
CIFAR-100
Tiny-ImageNet
```

For every reused asset record:

```text
path
dataset
method
seed
shape
dtype
feature dimension
sample count
label format
SHA256
development split
anchor split if needed
```

Check whether the existing 2048-D feature representations can be consumed directly by JAX/SPU.

If conversion from `.pt` or `.npz` to another host-side representation is needed:

- preserve values;
- preserve row order;
- preserve labels;
- preserve split membership;
- compare hashes where possible;
- numerically verify the conversion.

Target:

```text
new feature extraction = 0
new Stage-A distillation = 0
```

Generate:

```text
new_spu_feature_reuse_audit.json
new_spu_feature_conversion_audit.md
```

---

# 9. SPU Ours Must Use the Best Validated Stage-B Version

For SPU, Ours must use the version supported by the strongest completed Stage-B ablation:

```text
secret-shared compact client features
→ secure lightweight-head training
→ Secure Average
→ class-balanced labeled compact-anchor CE refinement
→ refined global initializer
```

This is intentionally independent of whether the existing old TFE Figure 2 used an earlier Average-only Stage B.

The completed Stage-B evidence showed that:

```text
Secure Average + Anchor CE
```

has independent value over both:

```text
Secure Average only
Random + Anchor CE
```

in the validated heterogeneous setting.

Therefore SPU should use the **Anchor-refined Stage-B version**.

Do not re-run those ablations on SPU.

---

# 10. Full Baseline Semantics on SPU

Implement a matched Full baseline using the paper's Full-data path.

Full and Ours must share, wherever semantically appropriate:

```text
model architecture
Stage-C Adaptive DP optimizer
privacy target
development evaluation
logging cadence
target definitions
CPU configuration
SPU protocol
numeric representation
```

They should differ in the intended paper factors:

```text
protected data volume
Stage-B initialization path
```

Do not artificially weaken Full.

---

# 11. Stage C on SPU

Port the current paper's Adaptive Stage-C semantics:

```text
Poisson sampling
→ per-example linear-head gradients
→ adaptive clipping
→ clipped-gradient aggregation
→ Gaussian perturbation
→ secure parameter update
```

Preserve the same conceptual privacy mechanism.

Use the existing paper configuration as the starting point.

Do not rewrite Stage C into a different optimizer merely because it is easier in SPU.

Maintain the current privacy-claim boundary:

```text
conditional Stage-C DP only
not end-to-end record-level DP
```

---

# 12. Autonomous Diagnosis and Optimization

Codex is explicitly allowed and expected to diagnose and optimize the SPU implementation autonomously.

The priority order is:

```text
1. correctness
2. semantic fidelity
3. numerical stability
4. CPU throughput
5. memory usage
6. final development utility
```

Do not immediately abort when the first SPU implementation is slow or numerically weak.

Diagnose the bottleneck and iterate.

However, tuning must remain scientifically controlled.

## 12.1 Freely optimize backend/system parameters

Codex may actively optimize:

```text
CPU thread counts
process placement
CPU affinity
NUMA binding
OpenMP threads
MKL threads
OpenBLAS threads
JAX/XLA CPU settings supported by the installed version
SPU compiler/runtime settings
field/ring selection if semantically valid
fixed-point fractional precision
batch graph capacity
secure computation chunking
data-transfer batching
JAX compilation reuse
jit/static-shape reuse
checkpoint frequency
logging frequency when it does not alter measurement semantics
```

These are systems/numerical engineering parameters.

## 12.2 Limited algorithmic tuning is allowed

Start from the best existing paper configuration.

If SPU accuracy or convergence is materially degraded after correctness is verified, Codex may perform **small sequential diagnostic tuning**, not a large grid.

Allowed one-factor-at-a-time adjustments include:

```text
learning rate
local-head epochs
anchor-refinement steps
anchor-refinement LR
anchor regularization
Stage-C LR
initial clipping bound
adaptive clipping target/rate
numeric precision
```

Rules:

1. change only one factor at a time;
2. use development validation only;
3. keep a tuning registry;
4. explain why each adjustment is needed;
5. stop when a stable useful configuration is reached;
6. freeze the final configuration before formal runs;
7. never tune on test/locked holdout;
8. never perform an unbounded search merely to obtain a positive result.

If SPU requires algorithmic settings different from TFE, record the differences clearly.

Do not falsely call the resulting comparison perfectly configuration-matched.

Generate:

```text
new_spu_tuning_registry.csv
new_spu_tuning_report.md
new_spu_final_config_freeze.json
```

---

# 13. CPU Performance — Use the Server Aggressively but Correctly

The formal SPU experiments should use as much available CPU performance as possible.

Do not simply set every process to all logical CPUs.

SecretFlow/SPU may run multiple logical parties/processes, so oversubscription can make performance worse.

## 13.1 CPU objective

Target approximately:

```text
total active compute threads across SPU parties
≈ available physical cores
```

Then test SMT/logical CPUs only if measured throughput improves.

Leave enough CPU capacity for orchestration.

## 13.2 Tune CPU systematically

Inspect and optimize:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
JAX/XLA supported CPU flags
taskset / sched affinity
numactl
SPU process affinity
party-level core partitioning
```

## 13.3 Short microbenchmarks

Use short representative SPU workloads.

Codex may run several short system configurations and choose the fastest stable one.

Do not use model accuracy for CPU tuning.

Measure:

```text
secure step/s
wall time
CPU utilization
peak RSS
load average
context switching if available
```

Freeze the best CPU configuration before the formal six runs.

Generate:

```text
new_spu_cpu_tuning.csv
new_spu_cpu_freeze.json
new_spu_cpu_tuning_report.md
```

The aim is to **maximize practical CPU utilization and secure-training throughput**.

---

# 14. Pipeline Smoke Tests Before Full Runs

Do not launch a long formal run immediately.

Use CIFAR-10 first.

## P1. Feature smoke

Verify reused feature shape, label mapping, split, and dtype.

## P2. Stage-B smoke

Run a very short version of:

```text
secure local-head training
→ secure average
→ secure anchor refinement
```

Compare with a plaintext reference.

## P3. Stage-C smoke

Run only a few Adaptive DP steps.

Verify:

```text
finite weights
finite accuracy
finite clipping bound
noise path works
privacy accountant works
no plaintext fallback
```

## P4. End-to-end trace smoke

Verify that:

```text
Stage-B runtime
Stage-B→C transition
Stage-C cumulative secure runtime
development accuracy checkpoints
epsilon
```

are all logged correctly.

Only after all smoke gates pass should formal runs begin.

Generate:

```text
new_spu_pipeline_smoke.json
new_spu_pipeline_smoke.md
```

---

# 15. Timing Semantics

The existing paper's main figure uses:

```text
Development accuracy versus cumulative online secure runtime
```

For SPU, preserve the same conceptual measurement.

Record separately:

```text
environment setup
data loading
JAX/SPU compilation
secret sharing/infeed
Stage-B online secure runtime
Stage-C online secure runtime
evaluation/reveal
raw wall-clock runtime
```

For the plotted x-axis, use:

```text
cumulative online secure runtime
```

with a clearly documented boundary.

Do not hide compilation.

If compilation is excluded from the figure, report it separately.

If the existing TFE figure excludes setup/graph compilation, do the analogous thing for SPU.

---

# 16. Target Accuracy Definitions

Before formal SPU runs, inspect and reuse the current frozen target definitions used by the paper's Figure 2.

Do not redefine the target from SPU's own final accuracy after seeing the result.

For each dataset record:

```text
T@80 target
T@90 target
T@95 target
source
```

Use the same targets for SPU Full and SPU Ours.

If the current Figure-2 targets are unsuitable for the new best Stage-B SPU method because the method semantics differ, report that explicitly in chat before altering any target definition.

Do not silently move the goalposts.

---

# 17. Formal SPU Run Order

Run seed42 only.

Recommended order:

```text
1. CIFAR-10 Ours
2. CIFAR-10 Full

3. CIFAR-100 Ours
4. CIFAR-100 Full

5. Tiny-ImageNet Ours
6. Tiny-ImageNet Full
```

Before Tiny-ImageNet, estimate runtime and memory from the first four runs.

Tiny is still mandatory for this task unless the server clearly cannot complete it within reasonable resources.

If Tiny is infeasible, do not substitute plaintext.

Report the infeasibility with measured evidence.

---

# 18. Formal Metrics

For every run record:

```text
backend
protocol
dataset
seed
method
sample count
feature dimension
Stage-B secure time
Stage-B→C transition time
Stage-C checkpoint step
development accuracy
cumulative online secure runtime
final development accuracy
total online secure runtime
T@80
T@90
T@95
conditional epsilon
delta
compile time
raw wall-clock time
peak RSS
CPU utilization
```

Communication is optional.

Only report communication if SecretFlow/SPU exposes a reliable measurement.

Otherwise:

```text
communication = N/A
```

Do not fabricate communication values.

---

# 19. SPU Figure 2 — Three Panels, Same Dataset Layout as Existing Figure 2

Generate a new SPU Figure-2-style result with exactly three dataset panels:

```text
CIFAR-10
CIFAR-100
Tiny-ImageNet
```

Each panel contains:

```text
Full
Ours
```

The SPU plot should follow the same visual language as the current Figure 2:

```text
Title:
Development accuracy versus cumulative online secure runtime

x-axis:
Cumulative online secure runtime (seconds, log scale)

y-axis:
Development validation accuracy

Full:
blue

Ours:
red
```

Add where valid:

```text
T@80
T@90
T@95
Stage-B→C transition
final development accuracy
conditional epsilon
```

All SPU datasets are:

```text
n = 1
seed42
```

Therefore:

```text
NO standard-deviation shading
NO fake error band
```

Tiny is also n=1.

---

# 20. Never Overwrite the Existing Figure 2

The existing TFE Figure 2 must remain untouched.

Do NOT write to its current filenames.

Generate SPU-specific filenames such as:

```text
new_figure2_spu_seed42_secure_time.png
new_figure2_spu_seed42_secure_time.pdf
new_figure2_spu_seed42_plot_data.csv
```

Also generate, if useful for the paper, a **new combined dual-backend candidate** without overwriting anything:

```text
new_figure2_dual_backend_2x3_seed42.png
new_figure2_dual_backend_2x3_seed42.pdf
```

Preferred combined layout:

```text
                     CIFAR-10       CIFAR-100       Tiny-ImageNet

TFE / ABY3           existing        existing         existing

SecretFlow-SPU       new seed42      new seed42       new seed42
```

Important:

- do not regenerate or alter the existing TFE row data;
- load existing TFE plot data as read-only;
- SPU row is seed42 only;
- label sample counts clearly.

The original Figure 2 remains preserved.

---

# 21. No SPU Ablations

Once the six SPU Full/Ours formal runs and figures are complete, STOP.

Do not continue with:

```text
KD
Consensus
disagreement
heterogeneity trigger
Random+Anchor
BestLocal+Anchor
Fixed vs Adaptive
precision16 vs20
extra seeds
```

The paper's ablation evidence remains TFE-based.

SPU exists to validate end-to-end portability across a second MPL backend.

---

# 22. Correctness and Result Gates

For each SPU run:

```text
no NaN/Inf
no plaintext fallback
same development split
same label mapping
same feature lineage
privacy accountant completes
conditional epsilon is recorded
```

For Ours, confirm the actual path is:

```text
compact features
→ secure local heads
→ secure average
→ secure anchor CE refinement
→ Adaptive Stage C
```

For the backend-level paper claim, the desired outcome is:

```text
Ours reaches the frozen target faster than Full
```

on SPU.

Do not force this outcome through unlimited tuning.

If a dataset fails, diagnose:

```text
numeric precision
SPU compilation
optimizer behavior
fixed-point error
learning rate
anchor refinement
Adaptive clipping dynamics
noise path
```

and perform bounded one-factor tuning.

If the conclusion still does not hold after the registered diagnostics, report the negative result honestly.

---

# 23. Checkpointing and Recovery

Long runs must be recoverable.

Save, where technically safe:

```text
Stage-B initializer
Stage-C model state
adaptive clipping state
privacy accountant state
runtime counters
secure-random provider state/counters needed for safe resume
```

Never resume in a way that repeats DP noise.

Use unique run IDs.

---

# 24. Required Output Files

Generate at minimum:

```text
new_spu_hardware_audit.json
new_spu_hardware_audit.md

new_spu_environment_manifest.txt
new_spu_pip_freeze.txt
new_spu_install_report.md

new_spu_quickstart_smoke.json
new_spu_quickstart_smoke.md

new_spu_feature_reuse_audit.json
new_spu_feature_conversion_audit.md

new_spu_pipeline_smoke.json
new_spu_pipeline_smoke.md

new_spu_tuning_registry.csv
new_spu_tuning_report.md
new_spu_final_config_freeze.json

new_spu_cpu_tuning.csv
new_spu_cpu_freeze.json
new_spu_cpu_tuning_report.md

new_spu_cifar10_full_seed42_trace.csv
new_spu_cifar10_ours_seed42_trace.csv
new_spu_cifar100_full_seed42_trace.csv
new_spu_cifar100_ours_seed42_trace.csv
new_spu_tiny_full_seed42_trace.csv
new_spu_tiny_ours_seed42_trace.csv

new_spu_end2end_seed42_main_table.csv
new_spu_end2end_seed42_runtime_breakdown.csv
new_spu_end2end_seed42_correctness_audit.md
new_spu_end2end_seed42_final_report.md

new_figure2_spu_seed42_plot_data.csv
new_figure2_spu_seed42_secure_time.png
new_figure2_spu_seed42_secure_time.pdf

new_figure2_dual_backend_2x3_seed42.png
new_figure2_dual_backend_2x3_seed42.pdf
```

Never overwrite the old Figure 2.

---

# 25. Progress Reporting

Every meaningful progress message should report:

```text
Phase
Conda environment
SecretFlow/SPU versions
Dataset
Method
Seed
Backend/protocol
CPU config
Current stage
Completed secure steps
Current dev accuracy
Cumulative online secure seconds
Current conditional epsilon
Peak RSS
CPU utilization
Fallback? yes/no
Test/locked holdout read? yes/no
Current diagnosis
Next single action
```

---

# 26. Stop Conditions

## Positive stop

Stop when:

```text
SecretFlow-SPU environment installed and frozen
official quickstart smoke passes
existing feature caches successfully reused
paper-pipeline smoke passes
CPU configuration optimized and frozen
all six seed42 formal runs complete
three-panel SPU Figure 2 generated
optional 2×3 dual-backend candidate generated
correctness/runtime report complete
```

Do not continue to ablations.

## Negative stop

Stop with a diagnostic report if:

```text
SecretFlow-SPU cannot be installed cleanly
required secure operators cannot be implemented without plaintext substitution
Stage C cannot preserve the intended DP mechanism
Tiny-ImageNet is demonstrably infeasible in available resources
SPU runtime/memory is unstable after bounded backend optimization
```

Do not fake completion.

---

# 27. Scientific Integrity

Never:

- modify previous TFE code/results;
- overwrite the existing Figure 2;
- label a plaintext fallback as SPU;
- use test/locked holdout for tuning;
- claim multi-seed SPU statistics;
- draw a variance band for n=1;
- claim end-to-end record-level DP;
- claim SPU is faster than TFE solely from incomparable runtime boundaries;
- report unmeasured communication;
- rerun all ablations merely because SPU was added.

The intended SPU paper conclusion, if supported, is:

> The end-to-end Full-vs-Ours time-to-target trend is reproduced on SecretFlow-SPU across CIFAR-10, CIFAR-100, and Tiny-ImageNet using a single seed per dataset, while all detailed ablations remain on the primary TFE backend.
