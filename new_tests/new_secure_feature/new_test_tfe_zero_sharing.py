#!/usr/bin/env python3
# 用途：验证 TFE 安全随机算子的确定性，以及 ABY3 零共享是否严格抵消。

from __future__ import print_function

import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase3_tfe_zero_sharing_test.json",
)

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.operations import secure_random


SEED = [
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
]


def _signed_mod_sum(values):
    """使用 uint64 环运算检查三份算术共享之和，避免 Python 溢出误判。"""
    total = np.asarray(values[0]).view(np.uint64).copy()
    for value in values[1:]:
        total = total + np.asarray(value).view(np.uint64)
    return total


def main():
    tf.compat.v1.reset_default_graph()
    deterministic_a = secure_random.seeded_random_uniform(
        [64, 4500], seed=SEED, maxval=1129, dtype=tf.int32
    )
    deterministic_b = secure_random.seeded_random_uniform(
        [64, 4500], seed=SEED, maxval=1129, dtype=tf.int32
    )
    secure_seed_value = secure_random.secure_seed()

    protocol = tfe.protocol.ABY3()
    with protocol:
        zero_groups = []
        for _ in range(16):
            shares = protocol._gen_zero_sharing([2048, 3])
            zero_groups.append([share.value for share in shares])

    with tfe.Session() as session:
        output_a, output_b, sampled_seed, evaluated_groups = session.run(
            [
                deterministic_a,
                deterministic_b,
                secure_seed_value,
                zero_groups,
            ]
        )

    deterministic_equal = bool(np.array_equal(output_a, output_b))
    zero_nonzero_counts = []
    zero_max_values = []
    for group in evaluated_groups:
        modular_sum = _signed_mod_sum(group)
        zero_nonzero_counts.append(int(np.count_nonzero(modular_sum)))
        zero_max_values.append(int(np.max(modular_sum)))

    payload = {
        "status": "PASS"
        if deterministic_equal and max(zero_nonzero_counts) == 0
        else "FAIL",
        "seeded_random_deterministic": deterministic_equal,
        "secure_seed_shape": list(np.asarray(sampled_seed).shape),
        "secure_seed_dtype": str(np.asarray(sampled_seed).dtype),
        "zero_groups": len(evaluated_groups),
        "zero_nonzero_counts": zero_nonzero_counts,
        "zero_max_uint64": zero_max_values,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
