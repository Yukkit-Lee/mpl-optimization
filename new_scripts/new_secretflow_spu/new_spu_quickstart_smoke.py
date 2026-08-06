#!/usr/bin/env python3
"""Official SPU-distributed quickstart smoke for the pinned SPU environment.

This validates the bare SPU tutorial runtime only.  The paper pipeline uses the
SecretFlow SPU device API and is validated separately before formal runs.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import secretflow
import spu
import spu.utils.distributed as ppd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "new_log/new_secretflow_spu"
TXT_DIR = OUT_DIR / "txt"
JSON_PATH = OUT_DIR / "new_spu_quickstart_smoke.json"
MD_PATH = OUT_DIR / "new_spu_quickstart_smoke.md"
CLUSTER_LOG = TXT_DIR / "new_spu_quickstart_cluster.log"
PORTS = (61327, 61328, 61329, 61437, 61438, 61439)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_cluster(process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SPU cluster exited early with code {process.returncode}")
        if all(port_is_open(port) for port in PORTS[:3]):
            return
        time.sleep(0.2)
    raise TimeoutError("SPU quickstart cluster did not become ready")


def make_x() -> np.ndarray:
    return np.asarray([1.25, -2.0, 3.5], dtype=np.float32)


def make_y() -> np.ndarray:
    return np.asarray([0.75, 4.0, -1.5], dtype=np.float32)


def secure_add(x, y):
    return x + y


def secure_compare(x, y):
    return x > y


def secure_matmul(a, b):
    return a @ b


def secure_linear_step(x, y, weight, bias, learning_rate):
    logits = x @ weight + bias
    residual = logits - y
    loss = jnp.mean(residual * residual)
    grad_weight = (2.0 / x.shape[0]) * (x.T @ residual)
    grad_bias = (2.0 / x.shape[0]) * jnp.sum(residual, axis=0)
    new_weight = weight - learning_rate * grad_weight
    new_bias = bias - learning_rate * grad_bias
    return logits, loss, new_weight, new_bias


def run_smoke() -> dict:
    started = time.monotonic()
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    conflicts = [port for port in PORTS if port_is_open(port)]
    if conflicts:
        raise RuntimeError(f"quickstart ports already in use: {conflicts}")

    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    with CLUSTER_LOG.open("w", encoding="utf-8") as cluster_log:
        cluster = subprocess.Popen(
            [sys.executable, "-m", "spu.utils.distributed", "up"],
            cwd=str(ROOT),
            env=env,
            stdout=cluster_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_cluster(cluster)
            ppd.init(ppd.SAMPLE_NODES_DEF, ppd.SAMPLE_DEVICES_DEF)
            spu_device = ppd.device("SPU")
            p1 = ppd.device("P1")
            p2 = ppd.device("P2")

            x_plain = make_x()
            y_plain = make_y()
            x = p1(make_x)()
            y = p2(make_y)()
            add_obj = spu_device(secure_add)(x, y)
            compare_obj = spu_device(secure_compare)(x, y)
            add_value = np.asarray(ppd.get(add_obj))
            compare_value = np.asarray(ppd.get(compare_obj))

            a_plain = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32)
            b_plain = np.asarray([[2.0, 0.5], [-1.0, 4.0]], dtype=np.float32)
            a = p1(lambda: a_plain)()
            b = p2(lambda: b_plain)()
            matmul_obj = spu_device(secure_matmul)(a, b)
            matmul_value = np.asarray(ppd.get(matmul_obj))

            train_x_plain = np.asarray(
                [[1.0, 0.5], [-0.5, 2.0], [1.5, -1.0], [0.0, 1.0]],
                dtype=np.float32,
            )
            train_y_plain = np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )
            weight_plain = np.asarray([[0.1, -0.2], [0.05, 0.15]], dtype=np.float32)
            bias_plain = np.asarray([0.0, 0.0], dtype=np.float32)
            learning_rate = np.float32(0.1)
            train_x = p1(lambda: train_x_plain)()
            train_y = p2(lambda: train_y_plain)()
            linear_fn = spu_device(secure_linear_step)
            logits_obj, loss_obj, weight_obj, bias_obj = linear_fn(
                train_x,
                train_y,
                weight_plain,
                bias_plain,
                learning_rate,
            )
            logits_value, loss_value, weight_value, bias_value = ppd.get(
                (logits_obj, loss_obj, weight_obj, bias_obj)
            )

            logits_ref = train_x_plain @ weight_plain + bias_plain
            residual_ref = logits_ref - train_y_plain
            loss_ref = np.mean(residual_ref * residual_ref)
            grad_weight_ref = (2.0 / train_x_plain.shape[0]) * (
                train_x_plain.T @ residual_ref
            )
            grad_bias_ref = (2.0 / train_x_plain.shape[0]) * np.sum(
                residual_ref, axis=0
            )
            weight_ref = weight_plain - learning_rate * grad_weight_ref
            bias_ref = bias_plain - learning_rate * grad_bias_ref

            checks = {
                "addition": bool(np.allclose(add_value, x_plain + y_plain, atol=1e-5)),
                "comparison": bool(np.array_equal(compare_value, x_plain > y_plain)),
                "matrix_multiplication": bool(
                    np.allclose(matmul_value, a_plain @ b_plain, atol=1e-4)
                ),
                "linear_logits": bool(
                    np.allclose(logits_value, logits_ref, atol=2e-4)
                ),
                "linear_loss": bool(np.allclose(loss_value, loss_ref, atol=2e-4)),
                "linear_weight_update": bool(
                    np.allclose(weight_value, weight_ref, atol=2e-4)
                ),
                "linear_bias_update": bool(
                    np.allclose(bias_value, bias_ref, atol=2e-4)
                ),
            }
            if not all(checks.values()):
                raise AssertionError(f"SPU/plaintext mismatch: {checks}")

            runtime_config = dict(
                ppd.SAMPLE_DEVICES_DEF["SPU"]["config"]["runtime_config"]
            )
            report = {
                "schema": "new_spu_quickstart_smoke_v1",
                "status": "COMPLETE",
                "backend": "SPU official distributed quickstart utility",
                "formal_backend": "SecretFlow-SPU device API (validated separately)",
                "formal_fallback": False,
                "versions": {
                    "python": sys.version.split()[0],
                    "secretflow": secretflow.__version__,
                    "spu": package_version("spu"),
                    "jax": jax.__version__,
                    "jaxlib": package_version("jaxlib"),
                    "numpy": np.__version__,
                },
                "device": {
                    "party_count": 3,
                    "protocol": runtime_config["protocol"],
                    "field": runtime_config["field"],
                    "fxp_fraction_bits_configured": "runtime_default",
                    "enable_pphlo_profile": runtime_config["enable_pphlo_profile"],
                    "nodes": list(ppd.SAMPLE_NODES_DEF),
                },
                "checks": checks,
                "revealed": {
                    "addition": add_value.tolist(),
                    "comparison": compare_value.tolist(),
                    "matrix_multiplication": matmul_value.tolist(),
                    "linear_logits": np.asarray(logits_value).tolist(),
                    "linear_loss": float(np.asarray(loss_value)),
                    "linear_weight": np.asarray(weight_value).tolist(),
                    "linear_bias": np.asarray(bias_value).tolist(),
                },
                "maximum_absolute_errors": {
                    "addition": float(np.max(np.abs(add_value - (x_plain + y_plain)))),
                    "matrix_multiplication": float(
                        np.max(np.abs(matmul_value - (a_plain @ b_plain)))
                    ),
                    "linear_logits": float(
                        np.max(np.abs(np.asarray(logits_value) - logits_ref))
                    ),
                    "linear_loss": float(abs(float(loss_value) - float(loss_ref))),
                    "linear_weight": float(
                        np.max(np.abs(np.asarray(weight_value) - weight_ref))
                    ),
                    "linear_bias": float(
                        np.max(np.abs(np.asarray(bias_value) - bias_ref))
                    ),
                },
                "elapsed_seconds": time.monotonic() - started,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024.0 * 1024.0),
                "test_or_locked_holdout_read": False,
                "notes": [
                    "Inputs originated on distinct P1/P2 PYU demo nodes and were secret-shared into a three-party ABY3 SPU.",
                    "Only final smoke values were revealed for correctness comparison.",
                    "The tutorial helper is not used as the formal paper backend.",
                ],
            }
            return report
        finally:
            # The quickstart launcher creates one process per party.  Stop the
            # entire isolated process group so a failed smoke cannot leave
            # servers listening on the fixed tutorial ports.
            os.killpg(cluster.pid, signal.SIGTERM)
            try:
                cluster.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(cluster.pid, signal.SIGKILL)
                cluster.wait(timeout=5)


def write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    errors = report["maximum_absolute_errors"]
    checks = report["checks"]
    lines = [
        "# SecretFlow-SPU quickstart smoke",
        "",
        f"Status: `{report['status']}`",
        "",
        "This gate follows the installed SPU official distributed quickstart: three",
        "local parties, ABY3, FM128, and JAX-authored secure functions. It is a",
        "backend smoke only; formal paper code uses the SecretFlow SPU device API.",
        "",
        "## Checks",
        "",
        "| Check | Pass | Maximum absolute error |",
        "|---|---|---:|",
        f"| addition | {checks['addition']} | {errors['addition']:.3g} |",
        f"| comparison | {checks['comparison']} | exact |",
        f"| matrix multiplication | {checks['matrix_multiplication']} | {errors['matrix_multiplication']:.3g} |",
        f"| linear logits | {checks['linear_logits']} | {errors['linear_logits']:.3g} |",
        f"| linear loss | {checks['linear_loss']} | {errors['linear_loss']:.3g} |",
        f"| linear weight update | {checks['linear_weight_update']} | {errors['linear_weight']:.3g} |",
        f"| linear bias update | {checks['linear_bias_update']} | {errors['linear_bias']:.3g} |",
        "",
        "Only final values needed for correctness were revealed. No dataset, test",
        "split, locked holdout, Stage A artifact, or TFE artifact was read.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = run_smoke()
    write_report(report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
