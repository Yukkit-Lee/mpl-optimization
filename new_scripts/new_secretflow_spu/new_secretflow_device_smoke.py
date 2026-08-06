#!/usr/bin/env python3
"""Gate the real SecretFlow SPU device API (not the bare SPU demo helper)."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

from new_secretflow_spu_backend import register_official_sfl_strategies


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_log/new_secretflow_spu"
JSON_PATH = OUT / "new_spu_secretflow_device_smoke.json"
MD_PATH = OUT / "new_spu_secretflow_device_smoke.md"


def private_x() -> np.ndarray:
    return np.asarray([[1.0, 0.5], [-0.5, 2.0], [1.5, -1.0], [0.0, 1.0]], dtype=np.float32)


def private_y() -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def secure_step(x, y, weight, bias):
    logits = x @ weight + bias
    residual = logits - y
    loss = jnp.mean(residual * residual)
    grad_weight = (2.0 / x.shape[0]) * (x.T @ residual)
    grad_bias = (2.0 / x.shape[0]) * jnp.sum(residual, axis=0)
    return logits, loss, weight - 0.1 * grad_weight, bias - 0.1 * grad_bias


def main() -> None:
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    started = time.monotonic()
    adapter = register_official_sfl_strategies()
    parties = ["alice", "bob", "carol"]
    sf.init(
        parties=parties,
        address="local",
        num_cpus=30,
        omp_num_threads=8,
        log_to_driver=True,
        include_dashboard=False,
    )
    try:
        alice, bob = sf.PYU("alice"), sf.PYU("bob")
        runtime_config = {
            "protocol": "ABY3",
            "field": "FM128",
            "fxp_fraction_bits": 16,
        }
        cdef = cluster_def(parties, runtime_config=runtime_config)
        spu_device = sf.SPU(cdef)
        x_plain, y_plain = private_x(), private_y()
        weight_plain = np.asarray([[0.1, -0.2], [0.05, 0.15]], dtype=np.float32)
        bias_plain = np.zeros(2, dtype=np.float32)
        x_secret = alice(private_x)().to(spu_device)
        y_secret = bob(private_y)().to(spu_device)
        result = spu_device(secure_step)(x_secret, y_secret, weight_plain, bias_plain)
        logits, loss, weight, bias = [np.asarray(value) for value in sf.reveal(result)]

        logits_ref = x_plain @ weight_plain + bias_plain
        residual_ref = logits_ref - y_plain
        loss_ref = np.mean(residual_ref * residual_ref)
        weight_ref = weight_plain - 0.1 * (2.0 / len(x_plain)) * (x_plain.T @ residual_ref)
        bias_ref = bias_plain - 0.1 * (2.0 / len(x_plain)) * np.sum(residual_ref, axis=0)
        errors = {
            "logits": float(np.max(np.abs(logits - logits_ref))),
            "loss": float(abs(float(loss) - float(loss_ref))),
            "weight": float(np.max(np.abs(weight - weight_ref))),
            "bias": float(np.max(np.abs(bias - bias_ref))),
        }
        checks = {
            "logits": bool(np.allclose(logits, logits_ref, atol=3e-4)),
            "loss": bool(np.allclose(loss, loss_ref, atol=3e-4)),
            "weight": bool(np.allclose(weight, weight_ref, atol=3e-4)),
            "bias": bool(np.allclose(bias, bias_ref, atol=3e-4)),
        }
        if not all(checks.values()):
            raise AssertionError(f"SecretFlow SPU/plaintext mismatch: {checks}")
        report = {
            "schema": "new_spu_secretflow_device_smoke_v1",
            "status": "COMPLETE",
            "backend": "SecretFlow-SPU device API",
            "scheduler": "official source-SFL SimulationStrategy on Ray",
            "source_sfl_commit": "c383e40f665063d7f7d87e437a73e015f87c435c",
            "versions": {
                "python": sys.version.split()[0],
                "secretflow": sf.__version__,
                "sfl": version("sfl"),
                "spu": version("spu"),
                "ray": version("ray"),
                "secretflow_rayfed": version("secretflow-rayfed"),
            },
            "device": {"parties": parties, **runtime_config},
            "adapter": adapter,
            "checks": checks,
            "maximum_absolute_errors": errors,
            "elapsed_seconds": time.monotonic() - started,
            "peak_driver_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0,
            "plaintext_debug_mode": False,
            "bare_spu_distributed_helper": False,
            "formal_fallback": False,
            "test_read": False,
            "locked_holdout_read": False,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# SecretFlow-SPU device smoke",
            "",
            "Status: `COMPLETE`",
            "",
            "This gate used the real `secretflow.SPU` device API with the official",
            "source-SFL SimulationStrategy, three Ray-hosted parties, ABY3/FM128,",
            "and 16 fractional bits. Debug/PYU emulation and the bare SPU helper",
            "were both disabled.",
            "",
            "| Check | Pass | Maximum absolute error |",
            "|---|---|---:|",
        ]
        for name in ("logits", "loss", "weight", "bias"):
            lines.append(f"| {name} | {checks[name]} | {errors[name]:.3g} |")
        lines.extend(["", "Private X and y originated on distinct PYUs. Only the final smoke outputs were revealed."])
        MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
