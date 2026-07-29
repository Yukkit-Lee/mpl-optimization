#!/usr/bin/env python3
"""Small deterministic tests for disagreement and weighted refinement."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "new_scripts/new_stage_b_anchor_refinement"
sys.path.insert(0, str(SCRIPT))

from new_stage_b_anchor_refinement_core import (
    build_anchor_weights,
    disagreement_scores,
    rank_auroc,
    train_weighted_anchor_ce,
)


def test_disagreement_order():
    clients = np.asarray(
        [
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.1, 0.9]],
        ],
        dtype=np.float32,
    )
    scores = disagreement_scores(clients)
    assert scores["d1_variance"][1] > scores["d1_variance"][0]
    assert scores["d2_mean_js"][1] > scores["d2_mean_js"][0]
    assert scores["d3_vote_entropy"][1] > scores["d3_vote_entropy"][0]


def test_auroc():
    assert abs(rank_auroc(np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1])) - 1.0) < 1e-12


def test_top_weights():
    scores = np.arange(10, dtype=np.float32)
    labels = np.arange(10) % 2
    weights = build_anchor_weights(scores, labels, "threshold", top_ratio=0.2)
    assert int(np.sum(weights > 0)) == 2


def test_uniform_refinement_finite():
    x = np.eye(2, dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    w = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    out_w, out_b, metrics = train_weighted_anchor_ce(
        x, y, np.ones(2), w, b, 5, 0.1, 0.0
    )
    assert np.isfinite(out_w).all() and np.isfinite(out_b).all()
    assert metrics["active_anchor_ratio"] == 1.0

