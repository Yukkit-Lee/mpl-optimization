#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：验证共享 encoder、synthetic split、scaler 和 L2 上界。"""

import unittest

import torch

from new_secure_feature_models import build_mtt_convnet, model_schema_hash
from new_stage_a_shared_encoder import fit_and_apply_scaler, stratified_synthetic_splits


class SharedEncoderAlignmentTests(unittest.TestCase):
    def test_all_dataset_encoders_have_2048_features(self):
        for dataset, shape in (
            ("cifar10", (2, 3, 32, 32)),
            ("cifar100", (2, 3, 32, 32)),
            ("tiny", (2, 3, 64, 64)),
        ):
            model = build_mtt_convnet(dataset, 42, torch.device("cpu"))
            self.assertEqual(tuple(model.encode(torch.randn(*shape)).shape), (2, 2048))

    def test_same_seed_has_same_encoder_signature_and_weights(self):
        first = build_mtt_convnet("cifar10", 42, torch.device("cpu"))
        second = build_mtt_convnet("cifar10", 42, torch.device("cpu"))
        self.assertEqual(model_schema_hash(first), model_schema_hash(second))
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_synthetic_splits_are_disjoint(self):
        labels = torch.arange(10).repeat_interleave(50)
        splits = stratified_synthetic_splits(labels, 42)
        values = [set(tensor.tolist()) for tensor in splits.values()]
        self.assertFalse(values[0] & values[1])
        self.assertFalse(values[0] & values[2])
        self.assertFalse(values[1] & values[2])
        self.assertEqual(sum(map(len, values)), len(labels))

    def test_fixed_holdout_per_class_preserves_more_encoder_samples(self):
        labels = torch.arange(10).repeat_interleave(50)
        splits = stratified_synthetic_splits(labels, 42, holdout_per_class=1)
        self.assertEqual(len(splits["encoder_train"]), 480)
        self.assertEqual(len(splits["reference"]), 10)
        self.assertEqual(len(splits["selection"]), 10)

    def test_scaler_uses_train_statistics_and_enforces_norm(self):
        train = torch.randn(30, 2048)
        reference = torch.randn(10, 2048)
        transformed, scaler = fit_and_apply_scaler(
            train, {"encoder_train": train, "reference": reference}, 1.0
        )
        self.assertEqual(tuple(scaler["mean"].shape), (2048,))
        self.assertLessEqual(float(transformed["encoder_train"].norm(dim=1).max()), 1.00001)
        self.assertLessEqual(float(transformed["reference"].norm(dim=1).max()), 1.00001)


if __name__ == "__main__":
    unittest.main()
