#!/usr/bin/env python3
"""Unit tests for the MTT-anchor plaintext implementation."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "new_scripts/new_stage_b_kd"
sys.path.insert(0, str(SCRIPT_DIR))

from new_stage_b_kd_core import (  # noqa: E402
    class_aware_teacher,
    client_partitions,
    mtt_anchor_split,
    normalize_reliability,
    train_student,
)


class StageBKDCoreTest(unittest.TestCase):
    def test_mtt_anchor_split_is_balanced_disjoint_and_complete(self):
        labels = np.repeat(np.arange(10), 48)
        local_ids, anchor_ids = mtt_anchor_split(labels, 10, 8, 42)
        self.assertEqual(len(local_ids), 400)
        self.assertEqual(len(anchor_ids), 80)
        self.assertEqual(len(np.intersect1d(local_ids, anchor_ids)), 0)
        self.assertEqual(len(np.unique(np.concatenate([local_ids, anchor_ids]))), 480)
        self.assertTrue(np.all(np.bincount(labels[local_ids], minlength=10) == 40))
        self.assertTrue(np.all(np.bincount(labels[anchor_ids], minlength=10) == 8))

    def test_client_partitions_are_exact_for_all_distributions(self):
        labels = np.repeat(np.arange(10), 40)
        for distribution in ("iid", "mild", "severe"):
            partitions = client_partitions(labels, distribution, 42)
            combined = np.concatenate(partitions)
            self.assertEqual(len(combined), len(labels))
            self.assertEqual(len(np.unique(combined)), len(labels))
            self.assertTrue(all(len(value) > 0 for value in partitions))

    def test_normalized_class_weights_obey_cap(self):
        reliability = np.asarray(
            [[5.0, 0.0], [1.0, 2.0], [0.5, 1.0], [0.0, 0.5], [-1.0, 0.0]]
        )
        weights = normalize_reliability(reliability, 0.5, 0.4, 0.25)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)
        self.assertLessEqual(float(weights.max()), 0.400001)
        self.assertTrue(np.isfinite(weights).all())

    def test_teacher_uses_output_class_weights_without_labels(self):
        probabilities = np.asarray(
            [
                [[0.9, 0.1], [0.8, 0.2]],
                [[0.2, 0.8], [0.1, 0.9]],
            ],
            dtype=np.float32,
        )
        weights = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
        teacher = class_aware_teacher(probabilities, weights)
        self.assertEqual(teacher.shape, (2, 2))
        np.testing.assert_allclose(teacher.sum(axis=1), 1.0, atol=1e-6)
        self.assertGreater(float(teacher[0, 0]), float(teacher[0, 1]))
        self.assertGreater(float(teacher[1, 1]), float(teacher[1, 0]))

    def test_student_training_is_finite(self):
        rng = np.random.RandomState(42)
        x_anchor = rng.normal(size=(20, 4)).astype(np.float32)
        y_anchor = np.asarray([0, 1] * 10, dtype=np.int64)
        teacher = np.zeros((20, 2), dtype=np.float32)
        teacher[np.arange(20), y_anchor] = 0.9
        teacher[np.arange(20), 1 - y_anchor] = 0.1
        average_weight = np.zeros((4, 2), dtype=np.float32)
        average_bias = np.zeros(2, dtype=np.float32)
        weight, bias, diagnostics = train_student(
            x_anchor,
            y_anchor,
            teacher,
            average_weight,
            average_bias,
            temperature=2.0,
            kd_lambda=0.5,
            average_regularization=1e-4,
            student_steps=20,
            student_lr=0.1,
        )
        self.assertTrue(np.isfinite(weight).all())
        self.assertTrue(np.isfinite(bias).all())
        self.assertGreaterEqual(diagnostics["distance_from_average"], 0.0)


if __name__ == "__main__":
    unittest.main()
