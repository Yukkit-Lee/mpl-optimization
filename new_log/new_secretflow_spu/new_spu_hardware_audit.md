# SecretFlow-SPU hardware audit

Status: `COMPLETE`

Audit time: `2026-08-06T16:40:11Z` (`2026-08-07` Asia/Shanghai)

## Effective compute resources

The host exposes two Intel Xeon Platinum 8352V sockets, 64 physical cores,
128 logical CPUs, two NUMA nodes, and AVX-512. Those host totals are not the
experiment allocation. The container is restricted to CPUs `96-127`, has a
32-CPU quota, and `nproc` reports 32. These CPUs are one SMT sibling for each
of cores 32-63 on socket/NUMA node 1.

The cgroup memory limit is 60 GiB with no swap. This limit is authoritative;
the 251 GiB reported by `free` is host-visible capacity and must not be used as
the experiment memory budget. At audit time the cgroup used about 2.24 GiB and
its observed peak was about 2.66 GiB.

`/root/autodl-tmp` has 120 GiB total and 33.7 GB available (74% used). The
repository occupies about 68 GiB. Feature caches will therefore be reused in
place, and `new_checkpoints/new_secretflow_spu/` remains local and ignored.

## Numeric and placement capabilities

- Base NumPy is backed by scipy-openblas 0.3.30 with dynamic architecture.
- `libgomp` and system BLAS are available.
- AVX2 and AVX-512 are available.
- `numactl` is not installed; affinity can still be controlled with `taskset`
  and per-process/thread settings.
- Existing conda environments are base, distillation, and legacy TFE. The new
  SPU backend will use a separate `spu_e2e` environment.

## CPU decision

The initial system-tuning budget is the 32 allowed CPU units, not all 128 host
logical CPUs. Three local SPU parties must share that allocation while leaving
capacity for orchestration. Party placement and thread counts will be selected
by short secure microbenchmarks; no formal run will begin before the chosen
configuration is frozen.

## Pre-implementation Git state

Before implementation, `git status --short --branch`, `git rev-parse HEAD`, and
`git diff --stat` recorded a clean `unified-experiments` worktree at parent
commit `52eb89ec27fd88940df295df2ebc686dd7d522e8`. Work continues on the isolated
branch `new-secretflow-spu-end2end`.

The machine-readable audit and exact command list are in
`new_spu_hardware_audit.json` beside this file.
