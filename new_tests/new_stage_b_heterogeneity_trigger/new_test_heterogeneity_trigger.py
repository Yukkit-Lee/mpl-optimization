#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(ROOT / "new_scripts/new_stage_b_heterogeneity_trigger")
)

from new_stage_b_heterogeneity_trigger_core import (
    class_count_matrix,
    heterogeneity_metrics,
    route_accuracy,
)


def test_iid_zero_heterogeneity():
    counts = np.asarray([[2, 2], [2, 2]], dtype=np.int64)
    values = heterogeneity_metrics(counts)
    assert abs(values["H1_mean_tv"]) < 1e-12
    assert abs(values["H2_entropy_deficit"]) < 1e-12
    assert values["H3_missing_class_ratio"] == 0.0


def test_skew_larger_than_iid():
    iid = heterogeneity_metrics(np.asarray([[2, 2], [2, 2]]))
    skew = heterogeneity_metrics(np.asarray([[4, 0], [0, 4]]))
    for key in ("H1_mean_tv", "H2_entropy_deficit", "H3_missing_class_ratio"):
        assert skew[key] > iid[key]


def test_count_matrix():
    labels = np.asarray([0, 0, 1, 1])
    parts = [np.asarray([0, 2]), np.asarray([1, 3])]
    assert np.array_equal(
        class_count_matrix(labels, parts, 2),
        np.asarray([[1, 1], [1, 1]]),
    )


def test_route_and_regret():
    low = route_accuracy(0.1, 0.5, 0.8, 0.6)
    high = route_accuracy(0.9, 0.5, 0.2, 0.7)
    assert low["trigger_bit"] == 0 and low["oracle_regret"] == 0.0
    assert high["trigger_bit"] == 1 and high["oracle_regret"] == 0.0

