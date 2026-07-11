#!/usr/bin/env python3
"""
Minimal reproducible validation for TF Encrypted baseline readiness.

This script checks:
1) package import and version visibility
2) local MPC matmul correctness (core secure computation path)
3) simple logistic model forward pass shape sanity
"""

import json
import platform
import sys
import time

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


def env_report():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tf_encrypted": getattr(tfe, "__version__", "unknown"),
    }


def check_secure_matmul():
    np.random.seed(0)
    a = np.ones((8, 8), dtype=np.float32)
    x = tfe.define_private_variable(a)
    y = x.matmul(x).matmul(x)
    expected = a.dot(a).dot(a)

    t0 = time.time()
    with tfe.Session() as sess:
        sess.run(tfe.global_variables_initializer(), tag="init")
        actual = sess.run(y.reveal(), tag="reveal")
    elapsed = time.time() - t0

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    return {"ok": True, "elapsed_sec": round(elapsed, 4), "shape": list(actual.shape)}


def check_logistic_forward():
    np.random.seed(1)
    features = tfe.define_private_variable(np.random.randn(16, 4).astype(np.float32))
    weights = tfe.define_private_variable(np.random.randn(4, 1).astype(np.float32))
    logits = features.matmul(weights)

    with tfe.Session() as sess:
        sess.run(tfe.global_variables_initializer(), tag="init")
        out = sess.run(logits.reveal(), tag="reveal")

    assert out.shape == (16, 1), "unexpected output shape"
    return {"ok": True, "shape": list(out.shape), "mean": float(np.mean(out))}


def main():
    result = {
        "env": env_report(),
        "checks": {},
    }

    try:
        result["checks"]["secure_matmul"] = check_secure_matmul()
    except Exception as exc:  # pragma: no cover - diagnostics path
        result["checks"]["secure_matmul"] = {"ok": False, "error": repr(exc)}

    try:
        result["checks"]["logistic_forward"] = check_logistic_forward()
    except Exception as exc:  # pragma: no cover - diagnostics path
        result["checks"]["logistic_forward"] = {"ok": False, "error": repr(exc)}

    ok = all(v.get("ok", False) for v in result["checks"].values())
    result["baseline_ready"] = ok
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
