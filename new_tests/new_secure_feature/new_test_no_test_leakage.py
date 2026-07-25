#!/usr/bin/env python3
# 用途：验证 Stage B/C 的训练与调参缓存不要求 test features 或 test labels。

import inspect
import unittest

import new_stage_b_secure_init
import new_stage_c_secure_dpsgd


class NoTestLeakageTest(unittest.TestCase):
    def test_feature_loader_does_not_require_test(self):
        source = inspect.getsource(new_stage_b_secure_init.load_feature_cache)
        self.assertNotIn('"x_test"', source)
        self.assertNotIn('"y_test"', source)

    def test_stage_c_tuning_uses_reference(self):
        source = inspect.getsource(new_stage_c_secure_dpsgd.build_graph)
        self.assertIn('data["x_reference"]', source)
        self.assertNotIn('data["x_test"]', source)


if __name__ == "__main__":
    unittest.main()
