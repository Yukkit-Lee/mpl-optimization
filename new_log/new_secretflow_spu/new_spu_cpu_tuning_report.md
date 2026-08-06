# SecretFlow-SPU CPU tuning report

Status: `COMPLETE` — freeze `new_spu_cpu_freeze.json`

## Objective

Select the thread configuration that maximizes secure-training throughput on
the container's 32-CPU quota (cpuset `96-127`). Three local SPU parties share
that quota while leaving headroom for orchestration. **No model accuracy was
used**; the selection is based on a representative fixed-shape Stage-C secure
workload (capacity 64, feature 2048, 10 classes, ABY3/FM128/fxp16).

## Method

For each candidate `OMP_NUM_THREADS = N`, a fresh process started
SecretFlow-SPU with `sf.init(num_cpus=30, omp_num_threads=N)` under
`taskset -c 96-127`. The representative Stage-C secure step (three-party
Gaussian noise shares summed inside SPU, adaptive clip, secure update) was run
3 repeats of 12 steps after a compile warm-up. Metrics: secure steps/s, compute
wall time, CPU utilization (/proc/stat sampling), driver peak RSS.

## Results

| Config | Threads/party | Party total | Secure steps/s | Compute wall (s) | CPU util | Peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|
| cfg_omp4 | 4 | 12 | 5.422 | 6.64 | 1.00 | 0.496 |
| cfg_omp6 | 6 | 18 | 5.235 | 6.88 | 1.00 | 0.496 |
| **cfg_omp8** | **8** | **24** | **5.585** | **6.45** | **1.00** | **0.496** |
| cfg_omp10 | 10 | 30 | 5.121 | 7.03 | 1.00 | 0.496 |
| cfg_omp12 | 12 | 36 | 2.284 | 15.76 | 1.00 | 0.495 |
| cfg_omp16 | 16 | 48 | 1.406 | 25.61 | 1.00 | 0.497 |

## Interpretation

- `cfg_omp8` (8 threads per party, 24 total) is the throughput optimum at
  **5.585 secure steps/s**. It fits inside the 32-CPU quota with 8 CPUs of
  headroom for Ray orchestration.
- Threads 4–10 are statistically flat (5.12–5.59 steps/s); 8 is the safe
  optimum.
- `cfg_omp12`/`cfg_omp16` collapse to 2.28/1.41 steps/s because 36/48 threads
  far exceed the 32-CPU quota and the three parties thrash for cores. This
  confirms the handoff rule: three parties must not each take large thread
  counts.
- CPU utilization is saturated (1.00) in every config at the container level;
  throughput, not utilization, is the discriminator.

## Freeze

All six formal seed42 runs will use:

```text
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
JAX_PLATFORMS=cpu RAY_USAGE_STATS_ENABLED=0
taskset -c 96-127 <spu_e2e python>
sf.init(parties=[alice,bob,carol], address="local", num_cpus=30, omp_num_threads=8)
```

Workload shape used for tuning: capacity 64 / feature 2048 / classes 10,
ABY3, FM128, fxp_fraction_bits 16. test/locked holdout remain unread.
