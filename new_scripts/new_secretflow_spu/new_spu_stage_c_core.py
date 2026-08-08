#!/usr/bin/env python3
"""Secure Stage-C Adaptive DP for SecretFlow-SPU.

Mirrors the audited TFE semantics (new_stage_c_secure_dpsgd.py):

    host CSPRNG Poisson sampling IDs
    -> per-example linear-head gradients
    -> adaptive global clipping
    -> clipped-gradient aggregation
    -> Gaussian perturbation via three-party noise shares summed inside SPU
    -> secure parameter update + adaptive clip-bound update

Privacy claims remain exactly the TFE boundary:
``conditional Stage-C DP only``; Stage-A has no DP; Stage-B is MPC
confidentiality only; no end-to-end record-level DP.
"""

from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from new_spu_pipeline_lib import one_hot, softmax_plain, stable_sha256


def make_secure_stage_c_scan(
    learning_rate,
    noise_multiplier,
    count_noise_multiplier,
    normalization_batch_size,
    gamma,
    eta_c,
    c_min,
    c_max,
    warmup_steps,
    update_interval,
    steps,
):
    """Closure factory for the whole Stage-C Adaptive-DP loop inside SPU.

    All scalar hyperparameters are bound as Python constants (not SPU tracers)
    so lax.scan ``length`` and arithmetic scale factors are static at trace
    time. Array data (x/y/mask/noise shares/adaptive mask) remain SPU args.

    x_seq: (S, capacity, D); y_seq: (S, capacity, K); mask_seq: (S, capacity).
    noise_w/noise_b/noise_count are the summed three-party N(0,1) shares
    (S, D, K) / (S, K) / (S,), pre-combined inside SPU by the caller.
    ``adaptive_mask`` (S,) is 1 where the clip bound is adaptively updated
    (step >= warmup and (step+1) % update_interval == 0), matching the TFE
    warmup/interval gating.
    """
    lr = float(learning_rate)
    sigma = float(noise_multiplier)
    count_sigma = float(count_noise_multiplier)
    bnorm = float(normalization_batch_size)
    eta = float(eta_c)
    gam = float(gamma)
    c_lo = float(c_min)
    c_hi = float(c_max)
    steps = int(steps)

    def _fn(
        x_seq,
        y_seq,
        mask_seq,
        noise_w,
        noise_b,
        noise_count,
        adaptive_mask,
        init_w,
        init_b,
        init_c,
    ):
        def body(carry, xs):
            w, b, c = carry
            x, y, mask, nw, nb, nc, adapt = xs

            logits = x @ w + b
            error = (logits - y) * jnp.expand_dims(mask, axis=1)
            error_sq = jnp.sum(error * error, axis=1)
            feature_sq = jnp.sum(x * x, axis=1)
            weight_norm_sq = error_sq * feature_sq
            bias_norm_sq = error_sq
            norm_sq = weight_norm_sq + bias_norm_sq

            # global adaptive clipping factor: min(1, C / sqrt(norm_sq))
            raw_factor = c * lax.rsqrt(norm_sq + 1e-5)
            factor = 1.0 - lax.max(1.0 - raw_factor, 0.0)

            below = lax.select(
                lax.le(norm_sq, c * c), jnp.zeros_like(mask), jnp.ones_like(mask)
            ) * mask

            clipped_error = error * jnp.expand_dims(factor, axis=1)
            clipped_grad_w = (x.T @ clipped_error) / bnorm
            clipped_grad_b = jnp.sum(clipped_error, axis=0) / bnorm

            noise_scale = sigma * c / bnorm
            noisy_grad_w = clipped_grad_w + nw * noise_scale
            noisy_grad_b = clipped_grad_b + nb * noise_scale
            new_w = w - lr * noisy_grad_w
            new_b = b - lr * noisy_grad_b

            fraction_below = jnp.sum(below) / bnorm
            noisy_fraction = jnp.clip(
                fraction_below + nc * (count_sigma / bnorm), 0.0, 1.0
            )
            exponent = -eta * (noisy_fraction - gam)
            candidate_c = jnp.clip(c * jnp.exp(exponent), c_lo, c_hi)
            # Only apply the clip-bound update on scheduled adaptive steps.
            new_c = jnp.where(adapt > 0.0, candidate_c, c)
            return (new_w, new_b, new_c), None

        xs = (x_seq, y_seq, mask_seq, noise_w, noise_b, noise_count, adaptive_mask)
        init = (init_w, init_b, init_c)
        (final_w, final_b, final_c), _ = lax.scan(body, init, xs, length=steps)
        return final_w, final_b, final_c

    _fn.__name__ = "secure_stage_c_scan_{}".format(steps)
    return _fn


def pack_step_sequences(
    x_batches,
    y_batches,
    masks,
    capacity: int,
):
    """Assemble per-step padded batches into (S, capacity, ...) arrays."""
    x_batches = [np.asarray(xb, dtype=np.float32) for xb in x_batches]
    y_batches = [np.asarray(yb, dtype=np.float32) for yb in y_batches]
    masks = [np.asarray(m, dtype=np.float32) for m in masks]
    dim = x_batches[0].shape[1]
    num_classes = y_batches[0].shape[1]
    steps = len(x_batches)
    x_seq = np.zeros((steps, capacity, dim), dtype=np.float32)
    y_seq = np.zeros((steps, capacity, num_classes), dtype=np.float32)
    mask_seq = np.zeros((steps, capacity), dtype=np.float32)
    for s in range(steps):
        n = x_batches[s].shape[0]
        x_seq[s, :n] = x_batches[s]
        y_seq[s, :n] = y_batches[s]
        mask_seq[s, :n] = masks[s]
    return x_seq, y_seq, mask_seq


def gaussian_share(shape, seed, stddev=1.0 / np.sqrt(3.0)):
    """Generate a single party's Gaussian noise share (host-side helper used
    inside a PYU worker). Each party draws its own share with its own RNG."""
    rng = np.random.RandomState(seed)
    return (rng.normal(0.0, float(stddev), size=shape)).astype(np.float32)


def poisson_expected_batches(data_size, expected_batch_size, steps, random_bytes=None):
    """Host-side Poisson sampling over the whole run (sampling IDs only)."""
    from new_spu_pipeline_lib import poisson_batch_indices

    batches = []
    for _ in range(steps):
        batches.append(
            poisson_batch_indices(data_size, expected_batch_size, random_bytes)
        )
    return batches


def plaintext_stage_c_reference(
    x_encoder_train,
    y_encoder_train,
    x_reference,
    y_reference,
    init_w,
    init_b,
    init_c,
    batch_ids_list,
    num_classes,
    capacity,
    normalization_batch_size,
    learning_rate,
    noise_multiplier,
    count_noise_multiplier,
    gamma,
    eta_c,
    c_min,
    c_max,
    warmup_steps,
    update_interval,
    clip_bound=None,
    # noise shares already summed in host for reference comparison
    noise_w_host=None,
    noise_b_host=None,
    noise_count_host=None,
):
    """Host-side NumPy twin of ``secure_stage_c_scan`` using the same noise
    values (passed in summed form) so the secure/noise/update math can be
    compared exactly. Only for correctness gate, never for formal output."""
    w = np.asarray(init_w, dtype=np.float32).copy()
    b = np.asarray(init_b, dtype=np.float32).copy()
    c = float(init_c)
    y_ref_hot = one_hot(np.asarray(y_reference, dtype=np.int64), num_classes)
    history = []
    for step, ids in enumerate(batch_ids_list):
        xb = np.zeros((capacity, x_encoder_train.shape[1]), dtype=np.float32)
        yb = np.zeros((capacity, num_classes), dtype=np.float32)
        mask = np.zeros(capacity, dtype=np.float32)
        if len(ids):
            xb[: len(ids)] = np.asarray(x_encoder_train[ids], dtype=np.float32)
            yb[: len(ids)] = one_hot(np.asarray(y_encoder_train[ids], dtype=np.int64), num_classes)
            mask[: len(ids)] = 1.0
        logits = xb @ w + b
        error = (logits - yb) * mask[:, None]
        error_sq = np.sum(error * error, axis=1)
        feature_sq = np.sum(xb * xb, axis=1)
        norm_sq = error_sq * feature_sq + error_sq
        factor = np.minimum(c / np.sqrt(norm_sq + 1e-5), 1.0)
        below = np.where(norm_sq <= c * c, 0.0, 1.0) * mask
        clipped_error = error * factor[:, None]
        clipped_grad_w = (xb.T @ clipped_error) / normalization_batch_size
        clipped_grad_b = np.sum(clipped_error, axis=0) / normalization_batch_size
        noise_scale = noise_multiplier * c / normalization_batch_size
        grad_w = clipped_grad_w + noise_w_host[step] * noise_scale
        grad_b = clipped_grad_b + noise_b_host[step] * noise_scale
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        fraction_below = np.sum(below) / normalization_batch_size
        noisy_fraction = np.clip(
            fraction_below + noise_count_host[step] * (count_noise_multiplier / normalization_batch_size),
            0.0,
            1.0,
        )
        use_adaptive = step >= warmup_steps and (step + 1) % update_interval == 0
        if use_adaptive:
            c = float(np.clip(c * np.exp(-eta_c * (noisy_fraction - gamma)), c_min, c_max))
        dev_logits = x_reference @ w + b
        dev_acc = float(np.mean(np.argmax(dev_logits, axis=1) == np.asarray(y_reference, dtype=np.int64)))
        history.append(
            {
                "step": step + 1,
                "c": c,
                "sampled_batch_size": len(ids),
                "adaptive": bool(use_adaptive),
                "development_accuracy": dev_acc,
            }
        )
    return w, b, c, history


def stage_c_claims() -> dict:
    return {
        "backend": "SecretFlow-SPU device API",
        "sampling_mode": "poisson",
        "sampling_rng": "os_csprng",
        "noise_path": "three-party Gaussian shares summed inside SPU",
        "adaptive_clip": True,
        "clipping_scope": "global",
        "plaintext_debug_mode": False,
        "fallback": False,
        "accountant": "Opacus RDP (steps + count-noise steps)",
    }
