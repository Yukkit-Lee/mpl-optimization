#!/usr/bin/env python3
"""Stage-B checkpoint evaluation core for the full-trajectory SPU runner.

Replicates the audited TFE checkpoint-evaluation semantics without exposing
any client head:

    secret client heads
      -> SPU-weighted temporary average (make_secure_stage_b_average)
      -> SPU development features @ averaged head (chunked for memory)
      -> reveal ONLY aggregate development logits
      -> driver computes aggregate accuracy from development labels

Invariants
----------
- evaluation is side-effect free: client_w_s / client_b_s are never mutated;
- no averaged weight/bias, no client head, no per-client logits are revealed;
- only aggregate development logits pass through the reveal ledger with
  ``used_for_training=false``;
- chunking preserves row order/labels; the concatenated chunk shapes must be
  audited for zero missing/duplicate/reordered rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp


def make_secure_stage_b_eval_forward():
    """SPU function: logits = dev_x @ avg_w + avg_b for one development chunk.

    The averaged head is passed in as a secret SPU value (result of
    make_secure_stage_b_average).  Only the returned logits may be revealed.
    """
    def _forward(dev_x_chunk, avg_w, avg_b):
        return dev_x_chunk @ avg_w + avg_b

    _forward.__name__ = "stage_b_eval_forward"
    return _forward


def make_secure_stage_b_eval_forward_placeholder():
    """Public-shape placeholder used only to trigger XLA compilation warmup."""
    def _forward(dev_x_chunk, avg_w, avg_b):
        return dev_x_chunk @ avg_w + avg_b

    _forward.__name__ = "stage_b_eval_forward_warmup"
    return _forward


def split_development_rows(num_rows: int, chunk_rows: int):
    """Deterministic chunk boundaries that preserve row order and cover all rows.

    Returns a list of (start, end) slices covering [0, num_rows) exactly once,
    in order.  The last chunk may be shorter.
    """
    if num_rows <= 0:
        raise ValueError("development rows must be positive")
    if chunk_rows <= 0:
        raise ValueError("eval_chunk_rows must be positive")
    chunks = []
    start = 0
    while start < num_rows:
        end = min(start + chunk_rows, num_rows)
        chunks.append((start, end))
        start = end
    if not chunks:
        raise ValueError("no development chunks produced")
    return chunks


def assert_chunk_cover(chunks, num_rows: int) -> None:
    """Hard gate: chunks exactly tile [0, num_rows) in order, no gaps/overlaps."""
    expected = 0
    for start, end in chunks:
        if start != expected:
            raise AssertionError("development chunk gap at row {}".format(expected))
        if end <= start:
            raise AssertionError("degenerate chunk {}:{}".format(start, end))
        expected = end
    if expected != num_rows:
        raise AssertionError(
            "development chunks cover {} rows, expected {}".format(expected, num_rows)
        )
