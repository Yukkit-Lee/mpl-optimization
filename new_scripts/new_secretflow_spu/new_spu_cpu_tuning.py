#!/usr/bin/env python3
"""CPU short microbenchmark for the isolated SecretFlow-SPU pipeline.

Runs a representative fixed-shape secure Stage-C workload on the real
secretflow.SPU device under a candidate thread configuration and records
secure steps/s, wall time, CPU utilization (via /proc/stat sampling), and
driver peak RSS. No model accuracy is used for CPU selection.

Usage (fresh process per candidate, threads passed via environment):
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OMP_NUM_THREADS_SPU=8 python new_spu_cpu_tuning.py --tag cfg_omp8 \
      --steps 12 --rows_csv path.csv

The SPU party OMP thread count is set by OMP_NUM_THREADS; sf.init passes
omp_num_threads from the same value.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

from new_secretflow_spu_backend import register_official_sfl_strategies
from new_spu_pipeline_lib import one_hot
from new_spu_stage_c_core import make_secure_stage_c_scan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_log/new_secretflow_spu"


def sample_cpu_seconds() -> float:
    """Aggregate CPU busy time from /proc/stat (all cores, seconds)."""
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("cpu "):
                parts = [float(v) for v in line.split()[1:]]
                return sum(parts) / 100.0
    return 0.0


def read_cpu_utilization(pre_busy: float, pre_total: float, post_busy: float, post_total: float) -> float:
    return (post_busy - pre_busy) / max(post_total - pre_total, 1e-9)


def read_total_cpu() -> float:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("cpu "):
                return sum(float(v) for v in line.split()[1:]) / 100.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--rows_csv", default=str(OUT / "new_spu_cpu_tuning.csv"))
    args = parser.parse_args()

    omp = int(os.environ.get("OMP_NUM_THREADS", "8"))
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    started = time.monotonic()
    pre_busy = sample_cpu_seconds()
    pre_total = read_total_cpu()

    adapter = register_official_sfl_strategies()
    parties = ["alice", "bob", "carol"]
    sf.init(
        parties=parties,
        address="local",
        num_cpus=30,
        omp_num_threads=omp,
        log_to_driver=False,
        include_dashboard=False,
    )
    try:
        alice = sf.PYU("alice")
        runtime_config = {
            "protocol": "ABY3",
            "field": "FM128",
            "fxp_fraction_bits": 16,
        }
        cdef = cluster_def(parties, runtime_config=runtime_config)
        spu_device = sf.SPU(cdef)

        # Representative fixed-shape Stage-C secure workload.
        capacity = 64
        feature_dim = 2048
        num_classes = 10
        steps = int(args.steps)
        rng = np.random.RandomState(42)
        x_seq = rng.rand(steps, capacity, feature_dim).astype(np.float32)
        y_labels = rng.randint(0, num_classes, size=(steps, capacity))
        y_seq = np.zeros((steps, capacity, num_classes), dtype=np.float32)
        y_seq[np.arange(steps)[:, None], np.arange(capacity)[None, :], y_labels] = 1.0
        mask_seq = np.ones((steps, capacity), dtype=np.float32)
        adaptive_mask = np.ones((steps,), dtype=np.float32)
        init_w = rng.normal(0, 0.01, (feature_dim, num_classes)).astype(np.float32)
        init_b = np.zeros((num_classes,), dtype=np.float32)
        init_c = np.float32(1.0)
        # Three-party noise shares summed inside SPU.
        nw_shares = [
            rng.normal(0.0, 1.0 / np.sqrt(3.0), (steps, feature_dim, num_classes)).astype(np.float32)
            for _ in range(3)
        ]
        nb_shares = [
            rng.normal(0.0, 1.0 / np.sqrt(3.0), (steps, num_classes)).astype(np.float32)
            for _ in range(3)
        ]
        nn_shares = [
            rng.normal(0.0, 1.0 / np.sqrt(3.0), (steps,)).astype(np.float32)
            for _ in range(3)
        ]
        x_s = alice(lambda: x_seq)().to(spu_device)
        y_s = alice(lambda: y_seq)().to(spu_device)
        m_s = alice(lambda: mask_seq)().to(spu_device)
        am_s = alice(lambda: adaptive_mask)().to(spu_device)
        nw_s = [alice(lambda v=v: v)().to(spu_device) for v in nw_shares]
        nb_s = [alice(lambda v=v: v)().to(spu_device) for v in nb_shares]
        nn_s = [alice(lambda v=v: v)().to(spu_device) for v in nn_shares]

        stage_c_fn = make_secure_stage_c_scan(
            0.002, 0.94, 8.0, 16, 0.5, 0.02, 0.05, 4.0, 0, 1, steps
        )

        def make_secure_run():
            def _run(
                x_v, y_v, m_v,
                nw0, nw1, nw2, nb0, nb1, nb2, nn0, nn1, nn2,
                am_v, w, b, c,
            ):
                return stage_c_fn(
                    x_v, y_v, m_v,
                    nw0 + nw1 + nw2,
                    nb0 + nb1 + nb2,
                    nn0 + nn1 + nn2,
                    am_v, w, b, c,
                )

            _run.__name__ = "secure_run_cpu_tuning"
            return _run

        run_fn = make_secure_run()
        run_fn_full = spu_device(run_fn)

        # Warm-up: compile once.
        _ = sf.reveal(
            run_fn_full(
                x_s, y_s, m_s,
                nw_s[0], nw_s[1], nw_s[2],
                nb_s[0], nb_s[1], nb_s[2],
                nn_s[0], nn_s[1], nn_s[2],
                am_s, init_w, init_b, init_c,
            )
        )
        compute_started = time.monotonic()
        repeats = 3
        for _ in range(repeats):
            _ = sf.reveal(
                run_fn_full(
                    x_s, y_s, m_s,
                    nw_s[0], nw_s[1], nw_s[2],
                    nb_s[0], nb_s[1], nb_s[2],
                    nn_s[0], nn_s[1], nn_s[2],
                    am_s, init_w, init_b, init_c,
                )
            )
        compute_elapsed = time.monotonic() - compute_started
        secure_steps = steps * repeats
        steps_per_second = secure_steps / compute_elapsed
        post_busy = sample_cpu_seconds()
        post_total = read_total_cpu()
        cpu_util = read_cpu_utilization(pre_busy, pre_total, post_busy, post_total)
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0

        row = {
            "tag": args.tag,
            "omp_threads": omp,
            "protocol": "ABY3",
            "field": "FM128",
            "fxp_fraction_bits": 16,
            "capacity": capacity,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "secure_steps": secure_steps,
            "compute_wall_seconds": round(compute_elapsed, 4),
            "secure_steps_per_second": round(steps_per_second, 4),
            "cpu_utilization_fraction": round(cpu_util, 4),
            "peak_driver_rss_gib": round(peak_rss_gib, 4),
            "total_wall_seconds": round(time.monotonic() - started, 4),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        csv_path = Path(args.rows_csv)
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
