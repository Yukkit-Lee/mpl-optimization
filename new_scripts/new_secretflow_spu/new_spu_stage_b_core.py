#!/usr/bin/env python3
"""Secure Stage-B for SecretFlow-SPU: local lightweight heads + secure average
+ class-balanced labeled compact-anchor CE refinement.

Mirrors the audited TFE semantics (new_stage_b_secure_init.py secure-average
path plus the post-aggregation Anchor CE from the anchor-postagg worker) but
runs the whole training graph inside one SPU JAX function.

Design notes
------------
- ``x_clients`` shape (C, L, D), ``y_clients_onehot`` shape (C, L, K), where
  C = 5 clients, L = samples per client (balanced split keeps them equal for
  Ours), D = 2048, K = num_classes.
- Ours cifar10 uses shared_initialization=True, so all client heads start from
  the same ``init_w/init_b``.
- Local updates are full-batch MSE steps per epoch (batch semantics identical
  in the limit; smoke uses short budgets). Aggregation coefficients are public.
- Anchor CE refinement is softmax CE + proximal (L2 about the average) terms,
  matching ``train_anchor_ce_trajectory`` in the audited core.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from new_spu_pipeline_lib import (
    one_hot,
    softmax_plain,
    stable_sha256,
)


def secure_stage_b(
    x_clients,
    y_clients_onehot,
    anchor_x,
    anchor_y_onehot,
    init_w,
    init_b,
    local_epochs,
    local_lr,
    weight_decay,
    agg_coeffs,
    anchor_steps,
    anchor_lr,
    anchor_mu,
):
    """Full Stage-B secure graph executed inside SPU.

    Args are JAX arrays; all shapes static. Python loops are trace-expanded
    (epochs/clients/steps are plain Python ints bound as defaults), so no lax
    control flow is required. Hyperparameters must be passed as plain Python
    scalars (or bound through a closure factory), never as SPU tracers.
    """
    num_clients = x_clients.shape[0]
    samples_per_client = x_clients.shape[1]
    num_anchor = anchor_x.shape[0]
    local_epochs = int(local_epochs)
    anchor_steps = int(anchor_steps)

    # Per-client local heads, shared initialization.
    weights = [init_w]
    biases = [init_b]
    for client_id in range(1, num_clients):
        weights.append(init_w)
        biases.append(init_b)

    for epoch in range(local_epochs):
        for client_id in range(num_clients):
            x = x_clients[client_id]
            y = y_clients_onehot[client_id]
            logits = x @ weights[client_id] + biases[client_id]
            error = logits - y
            grad_w = (x.T @ error) / float(samples_per_client)
            grad_b = jnp.sum(error, axis=0) / float(samples_per_client)
            grad_w = grad_w + float(weight_decay) * weights[client_id]
            weights[client_id] = weights[client_id] - float(local_lr) * grad_w
            biases[client_id] = biases[client_id] - float(local_lr) * grad_b

    # Secure average with public coefficients.
    avg_w = weights[0] * float(agg_coeffs[0])
    avg_b = biases[0] * float(agg_coeffs[0])
    for client_id in range(1, num_clients):
        avg_w = avg_w + weights[client_id] * float(agg_coeffs[client_id])
        avg_b = avg_b + biases[client_id] * float(agg_coeffs[client_id])

    # Class-balanced labeled compact-anchor CE refinement (proximal).
    reference_w = avg_w
    reference_b = avg_b
    for step in range(anchor_steps):
        logits = anchor_x @ avg_w + avg_b
        shifted = logits - jnp.max(logits, axis=1, keepdims=True)
        probs = jnp.exp(shifted) / jnp.sum(jnp.exp(shifted), axis=1, keepdims=True)
        error = probs - anchor_y_onehot
        grad_w = (anchor_x.T @ error) / float(num_anchor)
        grad_b = jnp.sum(error, axis=0) / float(num_anchor)
        if float(anchor_mu) > 0.0:
            grad_w = grad_w + 2.0 * float(anchor_mu) * (avg_w - reference_w)
            grad_b = grad_b + 2.0 * float(anchor_mu) * (avg_b - reference_b)
        avg_w = avg_w - float(anchor_lr) * grad_w
        avg_b = avg_b - float(anchor_lr) * grad_b

    return avg_w, avg_b


def make_secure_stage_b(
    local_epochs,
    local_lr,
    weight_decay,
    agg_coeffs,
    anchor_steps,
    anchor_lr,
    anchor_mu,
):
    """Closure factory binding Stage-B hyperparameters as Python constants so
    they are not treated as SPU tracers. Array inputs stay as SPU arguments."""

    def _fn(x_clients, y_clients_onehot, anchor_x, anchor_y_onehot, init_w, init_b):
        return secure_stage_b(
            x_clients,
            y_clients_onehot,
            anchor_x,
            anchor_y_onehot,
            init_w,
            init_b,
            int(local_epochs),
            float(local_lr),
            float(weight_decay),
            tuple(float(v) for v in agg_coeffs),
            int(anchor_steps),
            float(anchor_lr),
            float(anchor_mu),
        )

    _fn.__name__ = "secure_stage_b_anchor_ce"
    return _fn


def make_secure_stage_b_formal(
    local_lr,
    weight_decay,
    seed,
    unroll=1,
):
    """Formal Stage-B single-epoch secure graph with mini-batch local training.

    Matches the audited TFE semantics. The epoch loop lives in the Python
    driver (each epoch re-shuffles batches via deterministic_batches, exactly
    like TFE session.run), so each SPU call trains every client over its
    deterministic mini-batches once and returns the updated local heads.

    Inputs (all SPU arguments; scalar hyperparameters bound as Python ints):
      x_batches:  (C, B, M, D)  per-client fixed mini-batch feature tensors
      y_batches:  (C, B, M, K)  one-hot labels
      init_w:     (D, K)        current local head weights
      init_b:     (K,)          current local head biases
    Returns updated (C, D, K) local weights and (C, K) local biases.
    """
    import jax.lax as lax

    def _fn(x_batches, y_batches, init_w, init_b):
        num_clients = x_batches.shape[0]
        num_batches = x_batches.shape[1]
        batch_size = x_batches.shape[2]

        def client_scan(carry, xs):
            w, b = carry
            xb, yb = xs
            logits = xb @ w + b
            error = logits - yb
            grad_w = (xb.T @ error) / float(batch_size)
            grad_b = jnp.sum(error, axis=0) / float(batch_size)
            grad_w = grad_w + float(weight_decay) * w
            new_w = w - float(local_lr) * grad_w
            new_b = b - float(local_lr) * grad_b
            return (new_w, new_b), None

        new_weights = []
        new_biases = []
        for client_id in range(num_clients):
            (w, b), _ = lax.scan(
                client_scan,
                (init_w[client_id], init_b[client_id]),
                (x_batches[client_id], y_batches[client_id]),
                length=int(num_batches),
                unroll=int(unroll),
            )
            new_weights.append(w)
            new_biases.append(b)
        return jnp.stack(new_weights, axis=0), jnp.stack(new_biases, axis=0)

    _fn.__name__ = "secure_stage_b_one_epoch_{}".format(seed)
    return _fn


def make_secure_stage_b_average(
    agg_coeffs,
    seed,
):
    """Secure weighted average of client local heads."""

    def _fn(client_w, client_b):
        num_clients = client_w.shape[0]
        avg_w = client_w[0] * float(agg_coeffs[0])
        avg_b = client_b[0] * float(agg_coeffs[0])
        for client_id in range(1, num_clients):
            avg_w = avg_w + client_w[client_id] * float(agg_coeffs[client_id])
            avg_b = avg_b + client_b[client_id] * float(agg_coeffs[client_id])
        return avg_w, avg_b

    _fn.__name__ = "secure_stage_b_average_{}".format(seed)
    return _fn


def make_secure_anchor_ce(
    anchor_steps,
    anchor_lr,
    anchor_mu,
    seed,
):
    """Secure class-balanced labeled compact-anchor CE refinement (proximal).

    Inputs: anchor_x (A, D), anchor_y_onehot (A, K), init_w (D, K), init_b (K,).
    """

    def _fn(anchor_x, anchor_y_onehot, init_w, init_b):
        avg_w = init_w
        avg_b = init_b
        reference_w = init_w
        reference_b = init_b
        num_anchor = anchor_x.shape[0]
        for _ in range(int(anchor_steps)):
            logits = anchor_x @ avg_w + avg_b
            shifted = logits - jnp.max(logits, axis=1, keepdims=True)
            probs = jnp.exp(shifted) / jnp.sum(
                jnp.exp(shifted), axis=1, keepdims=True
            )
            error = probs - anchor_y_onehot
            grad_w = (anchor_x.T @ error) / float(num_anchor)
            grad_b = jnp.sum(error, axis=0) / float(num_anchor)
            if float(anchor_mu) > 0.0:
                grad_w = grad_w + 2.0 * float(anchor_mu) * (avg_w - reference_w)
                grad_b = grad_b + 2.0 * float(anchor_mu) * (avg_b - reference_b)
            avg_w = avg_w - float(anchor_lr) * grad_w
            avg_b = avg_b - float(anchor_lr) * grad_b
        return avg_w, avg_b

    _fn.__name__ = "secure_anchor_ce_{}".format(seed)
    return _fn


def plaintext_stage_b_reference(
    x_clients,
    y_clients,
    anchor_x,
    anchor_y,
    init_w,
    init_b,
    num_classes,
    local_epochs,
    local_lr,
    weight_decay,
    agg_coeffs,
    anchor_steps,
    anchor_lr,
    anchor_mu,
):
    """Host-side NumPy twin of ``secure_stage_b`` for correctness comparison."""
    x_clients = np.asarray(x_clients, dtype=np.float32)
    y_clients = np.asarray(y_clients, dtype=np.int64)
    y_hot = np.stack(
        [one_hot(y_clients[i], num_classes) for i in range(y_clients.shape[0])]
    )
    anchor_x = np.asarray(anchor_x, dtype=np.float32)
    anchor_y_hot = one_hot(np.asarray(anchor_y, dtype=np.int64), num_classes)
    weights = [np.asarray(init_w, dtype=np.float32).copy() for _ in range(len(x_clients))]
    biases = [np.asarray(init_b, dtype=np.float32).copy() for _ in range(len(x_clients))]
    samples_per_client = x_clients.shape[1]
    for _ in range(int(local_epochs)):
        for client_id in range(len(x_clients)):
            logits = x_clients[client_id] @ weights[client_id] + biases[client_id]
            error = logits - y_hot[client_id]
            grad_w = (x_clients[client_id].T @ error) / float(samples_per_client)
            grad_b = np.sum(error, axis=0) / float(samples_per_client)
            grad_w = grad_w + float(weight_decay) * weights[client_id]
            weights[client_id] = weights[client_id] - float(local_lr) * grad_w
            biases[client_id] = biases[client_id] - float(local_lr) * grad_b
    agg = np.asarray(agg_coeffs, dtype=np.float64)
    avg_w = np.sum(np.stack(weights, axis=0) * agg[:, None, None], axis=0)
    avg_b = np.sum(np.stack(biases, axis=0) * agg[:, None], axis=0)
    ref_w = avg_w.copy()
    ref_b = avg_b.copy()
    for _ in range(int(anchor_steps)):
        logits = anchor_x @ avg_w + avg_b
        probs = softmax_plain(logits)
        error = probs - anchor_y_hot
        grad_w = (anchor_x.T @ error) / float(len(anchor_x))
        grad_b = np.sum(error, axis=0) / float(len(anchor_x))
        if float(anchor_mu) > 0.0:
            grad_w = grad_w + 2.0 * float(anchor_mu) * (avg_w - ref_w)
            grad_b = grad_b + 2.0 * float(anchor_mu) * (avg_b - ref_b)
        avg_w = avg_w - float(anchor_lr) * grad_w
        avg_b = avg_b - float(anchor_lr) * grad_b
    return avg_w, avg_b


def stage_b_claims() -> dict:
    return {
        "backend": "SecretFlow-SPU device API",
        "secure_average_only": False,
        "secure_anchor_ce": True,
        "anchor_source": "official_mtt_labeled_compact_anchors",
        "initializer_privacy_status": "data_dependent_non_dp",
        "plaintext_debug_mode": False,
        "fallback": False,
    }
