#!/usr/bin/env python3
"""Gate the JAX operators required by the paper pipeline on real SecretFlow-SPU.

Verifies the SPU JAX frontend supports: matmul/transpose, sqrt/exp/log,
softmax, jax.random.normal (noise path), and jax.lax.scan (step loop).
If any of these is unsupported the Stage-C noise/loop design must change
before implementation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import secretflow as sf
from secretflow.utils.testing import cluster_def

from new_secretflow_spu_backend import register_official_sfl_strategies


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_log/new_secretflow_spu"
JSON_PATH = OUT / "new_spu_jax_op_gate.json"


def secure_ops(x, w, b, c, key):
    # 1. matmul + bias
    logits = x @ w + b
    # 2. sqrt / exp / log / min / max (adaptive clip)
    norm = jnp.sum(x * x, axis=1, keepdims=True)
    factor = jnp.minimum(c / jnp.sqrt(norm + 1e-5), 1.0)
    # 3. softmax (anchor CE / per-example gradient)
    shifted = logits - jnp.max(logits, axis=1, keepdims=True)
    probs = jnp.exp(shifted) / jnp.sum(jnp.exp(shifted), axis=1, keepdims=True)
    # 4. jax.random.normal (Gaussian noise path)
    noise = jax.random.normal(key, (4, 3), dtype=jnp.float32)
    # 5. clamp / select (adaptive clip bound update)
    exp_c = c * jnp.exp(-0.05 * (jnp.mean(factor) - 0.5))
    clamped_c = jnp.clip(exp_c, 0.05, 4.0)
    below = jnp.where(norm <= c * c, 0.0, 1.0)
    return logits, probs, factor, noise, clamped_c, below


def secure_scan(x_init, w, steps):
    # fori/scan loop: cumulative sum over fixed steps.
    # `steps` is captured as a Python int (closure) so lax.scan's `length`
    # argument is static at trace time, not a SPU tracer.
    def body(acc, _):
        return acc + (x_init @ w), None

    (result, _) = lax.scan(
        body, jnp.zeros_like(x_init @ w), None, length=int(steps)
    )
    return result


def main() -> None:
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    adapter = register_official_sfl_strategies()
    parties = ["alice", "bob", "carol"]
    sf.init(
        parties=parties,
        address="local",
        num_cpus=30,
        omp_num_threads=8,
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

        x_plain = np.random.RandomState(0).rand(4, 8).astype(np.float32)
        w_plain = np.random.RandomState(1).rand(8, 3).astype(np.float32)
        b_plain = np.zeros(3, dtype=np.float32)
        c_plain = np.float32(1.0)
        key_plain = jax.random.PRNGKey(123)

        x_secret = alice(lambda: x_plain)().to(spu_device)

        logits, probs, factor, noise, clamped_c, below = sf.reveal(
            spu_device(secure_ops)(
                x_secret, w_plain, b_plain, c_plain, key_plain
            )
        )
        logits = np.asarray(logits)
        probs = np.asarray(probs)
        factor = np.asarray(factor)
        noise = np.asarray(noise)
        clamped_c = float(np.asarray(clamped_c).reshape(-1)[0])
        below = np.asarray(below)

        logits_ref = x_plain @ w_plain + b_plain
        shifted = logits_ref - np.max(logits_ref, axis=1, keepdims=True)
        probs_ref = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        norm_ref = np.sum(x_plain * x_plain, axis=1, keepdims=True)
        factor_ref = np.minimum(c_plain / np.sqrt(norm_ref + 1e-5), 1.0)
        exp_c_ref = c_plain * np.exp(-0.05 * (np.mean(factor_ref) - 0.5))
        clamped_c_ref = float(np.clip(exp_c_ref, 0.05, 4.0))
        below_ref = np.where(norm_ref <= c_plain * c_plain, 0.0, 1.0)

        results = {
            "matmul": bool(np.allclose(logits, logits_ref, atol=2e-3)),
            "softmax": bool(np.allclose(probs, probs_ref, atol=2e-3)),
            "adaptive_clip_factor": bool(np.allclose(factor, factor_ref, atol=2e-3)),
            "adaptive_clip_bound_update": bool(
                abs(clamped_c - clamped_c_ref) < 2e-3
            ),
            "clip_select": bool(np.array_equal(below, below_ref)),
            "jax_random_normal": bool(
                np.all(np.isfinite(noise)) and np.std(noise) > 0.5
            ),
        }

        # scan loop gate: x_init @ w accumulated `steps` times.
        # Bind steps as a static Python int through a closure factory (not
        # functools.partial, which lacks __name__ required by SecretFlow).
        def make_secure_scan(steps_value):
            def _secure_scan(x_init, w):
                return secure_scan(x_init, w, steps_value)

            _secure_scan.__name__ = "secure_scan_steps_{}".format(steps_value)
            return _secure_scan

        steps = 3
        scan_result = sf.reveal(
            spu_device(make_secure_scan(steps))(x_secret, w_plain)
        )
        scan_result = np.asarray(scan_result)
        scan_ref = np.zeros_like(x_plain @ w_plain)
        for _ in range(steps):
            scan_ref = scan_ref + x_plain @ w_plain
        results["jax_lax_scan_loop"] = bool(
            np.allclose(scan_result, scan_ref, atol=2e-3)
        )

        report = {
            "schema": "new_spu_jax_op_gate_v1",
            "status": "COMPLETE",
            "backend": "SecretFlow-SPU device API",
            "protocol": runtime_config,
            "adapter": adapter,
            "checks": results,
            "notes": {
                "jax_random_normal": (
                    "noise drawn inside the SPU function from a host-provided PRNG key; "
                    "formal noise path must still be audited vs TFE three-party CSPRNG shares"
                ),
                "jax_lax_scan_loop": (
                    "lax.scan supported; stage-C step loop can run inside SPU"
                ),
            },
            "all_required_ops_supported": all(results.values()),
            "test_read": False,
            "locked_holdout_read": False,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    finally:
        sf.shutdown()


if __name__ == "__main__":
    main()
