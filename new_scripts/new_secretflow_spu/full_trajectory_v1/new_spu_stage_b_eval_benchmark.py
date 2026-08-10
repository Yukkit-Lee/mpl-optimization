#!/usr/bin/env python3
"""Evaluation-only exact-shape benchmark for Stage-B checkpoint logits.

Used by Phase G1 (CIFAR-10) and Phase G2 (Tiny D=2048/K=200) to pick the
evaluation chunk size and confirm OMP8 vs OMP10. Performance selection uses
wall / phase time / cgroup CPU delta / run-local memory only, never accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "new_scripts/new_secretflow_spu"
HERE = Path(__file__).resolve().parent
for location in (str(HERE), str(BASE), str(ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from new_secretflow_spu_backend import register_official_sfl_strategies  # noqa: E402
from new_spu_stage_b_eval_core import (  # noqa: E402
    assert_chunk_cover,
    make_secure_stage_b_eval_forward,
    split_development_rows,
)

OUT = ROOT / "new_log/new_secretflow_spu/full_trajectory_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--feature_dim", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--dev_rows", type=int, required=True)
    parser.add_argument("--eval_chunk_rows", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    adapter = register_official_sfl_strategies()
    names = ["alice", "bob", "carol"]
    sf.init(
        parties=names, address="local", num_cpus=30, omp_num_threads=8,
        log_to_driver=False, include_dashboard=False,
    )
    try:
        pyus = [sf.PYU(name) for name in names]
        alice = pyus[0]
        protocol = {"protocol": "ABY3", "field": "FM128", "fxp_fraction_bits": 16}
        spu = sf.SPU(cluster_def(names, runtime_config=protocol))
        rng = np.random.RandomState(0)

        chunks = split_development_rows(args.dev_rows, args.eval_chunk_rows)
        assert_chunk_cover(chunks, args.dev_rows)

        avg_w = rng.normal(0, 0.01, (args.feature_dim, args.num_classes)).astype(np.float32)
        avg_b = np.zeros((args.num_classes,), dtype=np.float32)
        avg_w_s = alice(lambda v=avg_w: v)().to(spu)
        avg_b_s = alice(lambda v=avg_b: v)().to(spu)

        forward_exec = spu_device_multi(spu, make_secure_stage_b_eval_forward())

        # Compile warmup with shape-identical public data.
        z_dev = np.zeros((args.eval_chunk_rows, args.feature_dim), np.float32)
        z_w = np.zeros((args.feature_dim, args.num_classes), np.float32)
        z_b = np.zeros((args.num_classes,), np.float32)
        with timer_ctx("compile_warmup"):
            _ = sf.reveal(forward_exec(z_dev, z_w, z_b))

        phase_times = []
        for rep in range(args.repeats):
            t0 = time.monotonic()
            logits_parts = []
            for start, end in chunks:
                rows_now = end - start
                dev_chunk = rng.rand(rows_now, args.feature_dim).astype(np.float32)
                dev_s = alice(lambda v=dev_chunk: v)().to(spu)
                logits_s = forward_exec(dev_s, avg_w_s, avg_b_s)
                logits_parts.append(np.asarray(sf.reveal(logits_s), dtype=np.float32))
            logits = np.concatenate(logits_parts, axis=0)
            if logits.shape != (args.dev_rows, args.num_classes):
                raise AssertionError(
                    "logits shape {} != ({},{})".format(
                        logits.shape, args.dev_rows, args.num_classes
                    )
                )
            if not np.all(np.isfinite(logits)):
                raise AssertionError("non-finite logits in benchmark")
            phase_times.append(time.monotonic() - t0)

        row = {
            "tag": args.tag,
            "feature_dim": args.feature_dim,
            "num_classes": args.num_classes,
            "dev_rows": args.dev_rows,
            "eval_chunk_rows": args.eval_chunk_rows,
            "chunk_count": len(chunks),
            "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
            "repeat_wall_seconds": [round(t, 4) for t in phase_times],
            "mean_eval_wall_seconds": round(float(np.mean(phase_times)), 4),
            "peak_driver_rss_gib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 4
            ),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "new_spu_stage_b_eval_benchmark.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, sort_keys=True))
    finally:
        sf.shutdown()


def spu_device_multi(spu_device, function):
    from new_spu_corrected_runtime import spu_multi_return
    return spu_multi_return(spu_device, function)


import contextlib


@contextlib.contextmanager
def timer_ctx(name):
    start = time.monotonic()
    yield
    print("[benchmark] {}: {:.2f}s".format(name, time.monotonic() - start), flush=True)


if __name__ == "__main__":
    main()
