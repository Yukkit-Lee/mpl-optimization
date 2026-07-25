#!/usr/bin/env python3
# 用途：扫描 TFE/ABY3 2048 维私有内积的安全分块大小，定位固定点溢出阈值。

from __future__ import print_function

import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from new_stage_b_secure_init import secure_chunked_matmul

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase3_tfe_chunked_matmul_test.json",
)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    rng = np.random.RandomState(42)
    x = rng.normal(0.0, 0.01, size=(1, 2048)).astype(np.float32)
    x = x / max(1.0, float(np.linalg.norm(x)))
    weight = rng.normal(0.0, 0.001, size=(2048, 1)).astype(np.float32)
    expected = np.matmul(x, weight)
    chunk_sizes = [2048, 256, 64, 16, 4]

    with tfe.protocol.ABY3():
        private_x = tfe.define_private_variable(x)
        private_weight = tfe.define_private_variable(weight)
        outputs = [
            secure_chunked_matmul(tfe, private_x, private_weight, chunk)
            for chunk in chunk_sizes
        ]
        with tfe.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            values = session.run([value.reveal() for value in outputs])

    rows = []
    for chunk, value in zip(chunk_sizes, values):
        error = float(np.max(np.abs(np.asarray(value) - expected)))
        rows.append(
            {
                "chunk_size": int(chunk),
                "value": float(np.asarray(value).reshape(-1)[0]),
                "expected": float(expected.reshape(-1)[0]),
                "max_abs_error": error,
                "pass": bool(error <= 0.01),
            }
        )
    payload = {
        "status": "PASS" if any(row["pass"] for row in rows) else "FAIL",
        "feature_dim": 2048,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
