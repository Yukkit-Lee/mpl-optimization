# Corrected-v2 SPU formal freeze

Status: `FROZEN_BEFORE_FORMAL_RUNS`

All six cells use seed 42, SecretFlow-SPU ABY3/FM128/fxp16, taskset 96-127,
OMP 8 per party, unroll 1, and Stage-C blocks of 10.  Batch/LR choices come
only from the pre-registered development screen.  No official test or locked
holdout record was read.

| Dataset | Method | Batch | LR | Batch chunk | Stage-B dev selection |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | True Ours | 8 | 0.25 | 10 | 68.72% |
| CIFAR-10 | Full | 64 | 0.8 | 10 | 72.58% |
| CIFAR-100 | True Ours | 32 | 1.0 | 25 | 47.78% |
| CIFAR-100 | Full | 64 | 0.8 | 16 (+11 tail) | 43.06% |
| Tiny-ImageNet | True Ours | 64 | 0.8 | 25 | 31.00% |
| Tiny-ImageNet | Full | 64 | 0.4 | 16 (+5 tail) | 27.63% |

The three Full selections retain their prior batch-16 development baselines
within 0.08/0.02/0.02 percentage points while reducing batch iterations 4x.
Chunking changes only how consecutive fixed batches are transported/executed;
it does not add, remove, or reorder optimizer updates.

Reported epsilon remains conditional on Stage C.  Stage B is confidential MPC
but data-dependent and non-DP; no end-to-end record-level DP claim is allowed.
