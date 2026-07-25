#!/usr/bin/env python3
# 用途：验证 Stage A 原始记录隐私与 Stage C 蒸馏特征记录隐私没有错误相加。

import inspect
import unittest

import new_privacy_accountant
import new_stage_c_secure_dpsgd


class PrivacyScopeTest(unittest.TestCase):
    def test_rdp_is_monotonic_in_steps(self):
        first = new_privacy_accountant.compute_epsilon(
            10, 0.1, 1.2, 1e-5, count_noise=8.0
        )
        second = new_privacy_accountant.compute_epsilon(
            20, 0.1, 1.2, 1e-5, count_noise=8.0
        )
        self.assertGreater(second, first)

    def test_no_epsilon_a_plus_c_claim(self):
        source = inspect.getsource(new_stage_c_secure_dpsgd)
        self.assertIn('"epsilon_a_plus_epsilon_c_claimed": False', source)
        self.assertNotIn("epsilon_total =", source)


if __name__ == "__main__":
    unittest.main()
