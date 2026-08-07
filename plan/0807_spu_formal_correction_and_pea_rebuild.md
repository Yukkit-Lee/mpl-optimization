# SecretFlow-SPU formal correction and PEA Figure-2 rebuild

Status: `ACTIVE`  
Parent task: `plan/0807_new_secretflow_spu_end2end_figure2_codex_prompt_v2.md`  
Branch: `new-secretflow-spu-end2end`

## 1. Objective

Repair the SecretFlow-SPU implementation so that the six seed42 Full/Ours
runs satisfy the original security, conditional Stage-C DP, timing,
configuration-freeze, recovery, and reporting requirements.  Rebuild the SPU
and dual-backend figures using the visual grammar of
`new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png`.

The engineering target is to match or exceed the TFE development utility and
online secure throughput where possible.  This is an optimization target, not
a license to move frozen accuracy targets, select favorable incomplete runs,
or claim an unsupported win.  Backend modernization alone does not guarantee
higher accuracy.

## 2. Quarantined predecessor artifacts

Commits `b63bec9`, `8d5ff54`, `2ff21b9`, and `9cf8992` produced five nominal
SPU runs and figures.  They are preserved as development diagnostics but are
not paper-formal evidence because:

1. Stage-B reveals all client heads after every epoch;
2. formal Gaussian shares are generated together on the driver from public,
   deterministic `RandomState(seed42)` state;
3. the formal runner never invokes or checkpoints the privacy accountant;
4. effective batch overrides differ from the pre-formal config freeze;
5. compilation/infeed/reveal are not separated from online secure time;
6. Tiny Full has only a generic Ray actor crash, not a measured OOM proof;
7. the required formal metadata and exact trace names are incomplete.

These artifacts must not be deleted or silently rewritten.  Result builders
must reject them as current formal input.  Corrected runs use schema/run IDs
ending in `_corrected_v2` and a separate output namespace.

## 3. Phase R0 — correction preflight and claim quarantine

Deliver:

- `new_log/new_secretflow_spu/corrected_v2/new_spu_correction_preflight.md`
- `new_log/new_secretflow_spu/corrected_v2/new_spu_artifact_status.json`

Hard gates:

- old results marked `QUARANTINED_DIAGNOSTIC`;
- old TFE code/results remain read-only;
- no old SPU evidence is deleted;
- exact parent-plan requirements are carried forward;
- corrected output/checkpoint namespaces are isolated.

## 4. Phase R1 — secure-state and DP correction

Status: `COMPLETE` (security smoke, real-data gate and planned interruption/resume gate PASS).

### R1.1 Stage B

- Keep all five client weight/bias states as `SPUObject` values across epochs.
- Never reveal a client-local head or per-client update.
- Perform periodic FedAvg and rebroadcast inside SPU.
- Reveal only the explicitly permitted global initializer/evaluation boundary.
- Record every reveal site in a machine-readable reveal audit.
- Preserve exact mini-batch semantics or register and freeze any algorithmic
  change before a formal run.

### R1.2 Stage C noise

- Each PYU independently creates its own Gaussian share using private CSPRNG
  entropy local to that party.
- The driver must never receive party seeds, RNG states, individual shares, or
  the reconstructed total noise.
- Shares are summed only inside SPU.
- A formal run must not use public experiment seed42 as DP-noise entropy.
- Recovery state is private per party and prevents share reuse after resume.

### R1.3 Privacy accountant

- Use the audited Opacus RDP accountant for every formal checkpoint.
- Account for sampled-gradient steps and adaptive-count releases.
- Record sample rate, steps, count steps, noise multipliers, delta, epsilon,
  accountant version, and accountant-state hash.
- Hard fail when conditional epsilon exceeds the frozen target.

### R1.4 Recovery

- Save step, optimizer/model state, clip bound, sampling progress, accountant
  progress, and private party RNG counters without exposing private entropy.
- Resume must retain earlier trace rows and cumulative time.
- A recovery smoke must prove that no sample/noise step is repeated.

## 5. Phase R2 — timing and CPU correction

Status: `IN_PROGRESS` (cgroup sampler, disjoint phase timers, and cross-process accumulation PASS; final freeze benchmark pending).

Measure separately:

```text
environment/Ray setup
data loading and host packing
SPU/JAX compilation
secret sharing and infeed
Stage-B online secure execution
Stage-B to Stage-C transition
Stage-C online secure execution
development evaluation and reveal
checkpoint I/O
raw wall-clock time
```

The Figure-2 x axis uses only the documented cumulative online secure boundary.
Do not label raw wall time as online secure time.

Fix CPU utilization to subtract idle/iowait correctly and measure only the
container-allowed CPU set where possible.  Re-run representative microbenchmarks
after the secure-state implementation is stable.  Freeze thread/affinity,
unroll, graph capacity, chunk size, and compilation-cache policy before formal
runs.

## 6. Phase R3 — correctness and bounded development tuning

Status: `COMPLETE` (security, recovery, timing, full/tail chunk gates and bounded batch/LR tuning PASS).

Run CIFAR-10 gates first:

1. secure Stage-B state persistence and no-local-reveal test;
2. independent-party CSPRNG share test;
3. full P1--P4 corrected pipeline smoke;
4. interrupted/resumed trace and accountant equivalence smoke;
5. online-time boundary audit;
6. plaintext numerical alignment within registered tolerance.

Only after correctness passes, use development-only one-factor-at-a-time
tuning.  Optimize in this order:

1. numerical precision and semantic fidelity;
2. batch-preserving graph chunking/unroll/compilation reuse;
3. learning-rate adaptation if an effective batch change is necessary;
4. local epochs and Anchor CE parameters only if utility remains degraded;
5. Stage-C LR/clip parameters only with an explicit causal diagnosis.

For every trial record the hypothesis, one changed factor, effective command,
utility, timing, epsilon, acceptance/rejection, and all config hashes.  Stop at
a stable useful configuration; do not perform an unbounded positive-result
search.

## 7. Phase R4 — final freeze and formal runs

Status: `IN_PROGRESS` (six-cell corrected-v2 config frozen; formal runs 4/6).

Before any corrected formal run, freeze:

- effective per-dataset/method algorithm configuration;
- CPU/backend configuration;
- target accuracies from the existing TFE target freeze;
- timing/reveal/accounting boundaries;
- data/split/cache hashes;
- source commit and environment manifest hashes.

Run order remains:

```text
CIFAR-10 Ours, CIFAR-10 Full,
CIFAR-100 Ours, CIFAR-100 Full,
Tiny-ImageNet Ours, Tiny-ImageNet Full
```

All runs are seed42 only.  No SPU ablations or extra seeds.  Tiny Full may be
declared infeasible only after collecting direct cgroup memory peak/events,
worker RSS, graph/input sizes, exit signal, and bounded chunking alternatives.
A generic Ray connection error is not enough.

## 8. Phase R5 — PEA-style Figure 2

Match the existing PEA visual grammar:

- three aligned dataset panels;
- CIFAR-10/CIFAR-100/Tiny-ImageNet display names;
- percentage y-axis and log-seconds x-axis;
- blue Full and red True Ours;
- top summary boxes with accuracy, feature count, T@90, epsilon;
- horizontal T@80/T@90/T@95 lines;
- Stage B→C vertical dotted line for each method;
- terminal point markers;
- legend including `n=1` for SPU;
- no variance shading for SPU seed42;
- footer stating timing and conditional-DP boundaries.

The dual-backend 2×3 figure must use the same axes, palette, typography, labels,
and claim boundaries in both rows.  It must not make incomparable TFE/SPU
runtime claims without an explicit timing-boundary audit.

## 9. Acceptance gates

The corrected task is complete only if all are true:

- six corrected formal cells complete, or a directly measured and audited
  negative stop satisfies the parent plan;
- no client-local Stage-B state is revealed;
- formal DP noise is unpredictable to the driver and independently shared;
- formal epsilon/delta are present and within the frozen boundary;
- effective runtime config equals the final freeze;
- compilation/infeed/evaluation/raw wall are separately reported;
- exact required artifacts and trace names exist;
- Figure-2 plot data hashes match the corrected traces;
- figures pass visual inspection;
- old TFE and quarantined SPU artifacts remain preserved;
- all modifications update progress, commit, and push before experiments.
