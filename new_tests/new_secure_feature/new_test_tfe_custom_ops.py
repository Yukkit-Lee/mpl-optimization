#!/usr/bin/env python3
# 用途：验证 TF 1.15.5 自定义算子可支持 ABY3 私有比较、指数、逆平方根与安全随机数。

import hashlib
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

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE
from tf_encrypted.operations import aux
from tf_encrypted.operations import secure_random
from tf_encrypted.operations.aux import aux as aux_impl
from tf_encrypted.operations.secure_random import secure_random as secure_random_impl


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase3_tfe_custom_ops_test.json",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    left = np.asarray([0.25, 2.0, 4.0], dtype=np.float32)
    right = np.asarray([0.5, 1.0, 4.0], dtype=np.float32)

    with tfe.protocol.ABY3():
        private_left = tfe.define_private_variable(left)
        private_right = tfe.define_private_variable(right)
        compared = tfe.less_equal(private_left, private_right)
        exponent = tfe.exp(private_left * -0.25)
        inverse_sqrt = tfe.inv_sqrt(private_left + 1.0)

        random_tensor = secure_random.random_uniform(
            [32], minval=-100000, maxval=100000, dtype=tf.int64
        )
        session = KE.get_session()
        session.run(tf.compat.v1.global_variables_initializer())
        compared_value, exponent_value, inverse_sqrt_value, random_value = session.run(
            [
                compared.reveal(),
                exponent.reveal(),
                inverse_sqrt.reveal(),
                random_tensor,
            ]
        )
    KE.clear_session()

    expected_compare = np.less_equal(left, right).astype(np.int64)
    expected_exp = np.exp(left * -0.25)
    expected_inverse_sqrt = 1.0 / np.sqrt(left + 1.0)
    aux_path = aux_impl.SO_PATH.format(
        dn=os.path.dirname(tfe.__file__), tfv=tf.__version__
    )
    random_path = secure_random_impl.SO_PATH.format(
        dn=os.path.dirname(tfe.__file__), tfv=tf.__version__
    )
    result = {
        "status": "PASS",
        "python": sys.version,
        "tensorflow": tf.__version__,
        "tf_encrypted": getattr(tfe, "__version__", "unknown"),
        "aux_loaded": aux_impl.aux_module is not None,
        "secure_random_loaded": secure_random.supports_secure_randomness(),
        "comparison": np.asarray(compared_value).astype(np.int64).tolist(),
        "comparison_expected": expected_compare.tolist(),
        "exp_max_abs_error": float(
            np.max(np.abs(np.asarray(exponent_value) - expected_exp))
        ),
        "inv_sqrt_max_abs_error": float(
            np.max(np.abs(np.asarray(inverse_sqrt_value) - expected_inverse_sqrt))
        ),
        "secure_random_unique_count": int(len(np.unique(random_value))),
        "aux_so": aux_path,
        "aux_sha256": _sha256(aux_path),
        "secure_random_so": random_path,
        "secure_random_sha256": _sha256(random_path),
    }
    if not result["aux_loaded"] or not result["secure_random_loaded"]:
        result["status"] = "FAIL"
    if result["comparison"] != result["comparison_expected"]:
        result["status"] = "FAIL"
    if result["secure_random_unique_count"] <= 1:
        result["status"] = "FAIL"
    if result["exp_max_abs_error"] > 0.05:
        result["status"] = "FAIL"
    if result["inv_sqrt_max_abs_error"] > 0.05:
        result["status"] = "FAIL"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
