#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：验证 Stage-A DP 会计单调性、组合规则和蒸馏 raw-data 隔离边界。"""

import ast
import json
import unittest
from pathlib import Path

from opacus.accountants import RDPAccountant

from new_stage_a_dp_experts import compose_rdp_histories


ROOT = Path(__file__).resolve().parents[2]


def epsilon(noise: float, steps: int, sample_rate: float = 0.01, delta: float = 1e-5) -> float:
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)


class StageADPTrajectoryTests(unittest.TestCase):
    def test_epsilon_increases_with_steps(self):
        self.assertGreater(epsilon(1.2, 100), epsilon(1.2, 20))

    def test_epsilon_decreases_with_noise(self):
        self.assertLess(epsilon(2.0, 100), epsilon(1.0, 100))

    def test_sequential_composition_not_smaller_than_one_trajectory(self):
        history = [(1.2, 0.01, 100)]
        one = compose_rdp_histories([history], 1e-5)
        two = compose_rdp_histories([history, history], 1e-5)
        self.assertGreaterEqual(two, one)

    def test_distillation_source_has_no_raw_dataset_loader(self):
        source_path = ROOT / "new_stage_a_dp_distill.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.add(node.module or "")
        self.assertNotIn("torchvision.datasets", imported_names)
        self.assertNotIn("get_dataset", source)
        self.assertNotIn("download=True", source)

    def test_dp_and_public_output_names_are_separate(self):
        dp_source = (ROOT / "new_stage_a_dp_distill.py").read_text(encoding="utf-8")
        self.assertIn("new_dp_mtt_images_", dp_source)
        self.assertNotIn("official_mtt", dp_source)

    def test_formal_configuration_requires_secure_rng(self):
        config_path = ROOT / "new_configs" / "new_secure_feature" / "new_formal_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIs(config["stage_a_dp_mtt"]["secure_rng_required"], True)

    def test_secure_rng_has_no_silent_fallback(self):
        source = (ROOT / "new_stage_a_dp_experts.py").read_text(encoding="utf-8")
        backend = (ROOT / "new_cryptographic_gaussian.py").read_text(encoding="utf-8")
        self.assertIn("ChaCha20GaussianSampler", source)
        self.assertIn("install_opacus_chacha20_noise", source)
        self.assertIn("external_mode_0600_sidecar", source)
        self.assertNotIn('"secure_sampler_state":', source)
        self.assertIn("algorithms.ChaCha20", backend)
        self.assertIn("secure_mode=True", backend)


if __name__ == "__main__":
    unittest.main()
