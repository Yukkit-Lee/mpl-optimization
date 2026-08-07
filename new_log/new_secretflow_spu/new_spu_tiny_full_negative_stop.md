# Tiny-ImageNet Full seed42 — SecretFlow-SPU negative stop diagnostic

Status: `NEGATIVE_STOP` (plan §26)

## Claim

Tiny-ImageNet **Full** cannot run on the isolated SecretFlow-SPU backend in the
available resources. The SPU runtime worker dies during the first Stage-B epoch
under every tried batch configuration. Tiny-ImageNet **Ours** does run (see
`new_spu_tiny_true_ours_seed42_report.json`).

## Evidence

| Batch override | Batches/client | Outcome |
|---|---:|---|
| 128 | 107 | SPURuntime actor SYSTEM_ERROR (connection error code 2) |
| 64 | 213 | same |
| 256 | 54 | same |
| 512 | 27 | same |

The worker exit detail reported by Ray is `Worker unexpectedly exits with a
connection error code 2` with the listed causes being "killed by the OOM killer
due to high memory usage" or a SIGSEGV. The failure occurs in the first
Stage-B epoch (`stage_b_initializer`) before any epoch completes.

## Why Tiny Full but not Tiny Ours

Tiny Full trains on the full 68,000-sample Stage-B pool; Tiny Ours uses the
9,600-sample Official-MTT compact pool. Full therefore:

- secret-shares ~557 MB of client feature data per epoch (5 clients × 13,600 ×
  2048 × 4 B) vs ~13 MB for Ours — an 8.5× memory footprint;
- carries num_classes=200 throughout the secure graph.

The same SPU pipeline completes Tiny Ours (batch=64, 25 batches/client) but
fails on Tiny Full even at batch=512 (27 batches/client), which is the same
batch count as the successful Ours run. This isolates the failure to the
larger protected-data volume (data size × secret-share × graph memory), not to
iteration count or unroll.

## Other backends

The TFE/ABY3 backend completed Tiny Full in the existing frozen TFE results
(`new_output/new_true_ours/stage_b/full_data_formal_corrected_v2/...`, final
development accuracy 0.272, per `unified_experiment_report.md`). The SPU
experiment is the second backend for cross-backend main-trend reproduction; the
Tiny Full SPU cell is not reproducible in the available 60 GiB cgroup.

## Honest reporting boundary

- No plaintext fallback was used.
- No claim is made that Tiny Full ran on SPU.
- Tiny Full remains a valid TFE/ABY3 cell; only the SPU row for this cell is
  reported as infeasible.
- test / locked holdout remain unread.
