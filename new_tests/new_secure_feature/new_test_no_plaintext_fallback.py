#!/usr/bin/env python3
# 用途：验证安全 Stage B/C 缺失 TFE custom ops 时硬失败，不回退明文。

import inspect
import unittest

import new_stage_b_secure_init
import new_stage_c_secure_dpsgd


class NoPlaintextFallbackTest(unittest.TestCase):
    def test_tfe_import_requires_custom_ops(self):
        source = inspect.getsource(new_stage_b_secure_init.import_tfe_or_fail)
        self.assertIn("raise RuntimeError", source)
        self.assertIn("secure_random custom op", source)

    def test_stage_c_uses_tfe_builder(self):
        source = inspect.getsource(new_stage_c_secure_dpsgd.build_graph)
        self.assertIn("import_tfe_or_fail()", source)
        self.assertNotIn("fallback_to_plaintext", source)


if __name__ == "__main__":
    unittest.main()
