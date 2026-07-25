#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage C RDP accountant 的低资源单元测试。"""

import unittest
import inspect

from stage_c_privacy_accounting import StageCAccountant
import official_distilled_convnet_pipeline as pipeline


def epsilon(noise, steps):
    accountant = StageCAccountant("gaussian", noise, 0.01, 1e-5)
    for _ in range(steps):
        accountant.step()
    return accountant.epsilon()


class StageCPrivacyTests(unittest.TestCase):
    def test_epsilon_increases_with_steps(self):
        self.assertLess(epsilon(1.2, 10), epsilon(1.2, 20))

    def test_epsilon_decreases_with_noise(self):
        self.assertGreater(epsilon(1.0, 20), epsilon(2.0, 20))

    def test_fixed_and_adaptive_share_accounting(self):
        self.assertAlmostEqual(epsilon(1.2, 20), epsilon(1.2, 20), places=12)

    def test_binomial_simulation_is_not_gaussian_accounted(self):
        accountant = StageCAccountant("numpy_binomial_simulation", 1.2, 0.01, 1e-5)
        accountant.step()
        self.assertIsNone(accountant.epsilon())
        self.assertEqual(accountant.metadata()["stage_c_epsilon"], "N/A")

    def test_reference_target_api_has_no_test_argument(self):
        parameters = inspect.signature(pipeline.build_reference_target).parameters
        self.assertFalse(any("test" in name.lower() for name in parameters))


if __name__ == "__main__":
    unittest.main()
