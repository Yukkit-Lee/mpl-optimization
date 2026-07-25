#!/usr/bin/env bash
# 用途：单核串行运行 Phase 1-3 的现代环境与 TFE/ABY3 回归测试。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"
cd "${PROJECT_ROOT}"

MODERN_PYTHON=/home/lixujie/.conda/envs/distillation/bin/python
TFE_PYTHON=/home/lixujie/.conda/envs/tfe/bin/python

taskset -c 0 "${MODERN_PYTHON}" -m unittest \
  new_tests.new_secure_feature.new_test_stage_a_dp_trajectory \
  new_tests.new_secure_feature.new_test_shared_encoder_alignment

taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_custom_ops.py
taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_zero_sharing.py
taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_chunked_matmul.py
taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_linear_update_2048.py
taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_repeated_local.py \
  --client_id 1 \
  --feature_cache new_cache/new_secure_feature/new_phase3_correctness/new_stage_b_synthetic_2048.npz \
  --output new_log/new_secure_feature/json/new_phase3_tfe_repeated_regression.json
NEW_DIAGNOSTIC_ATTEMPT=regression taskset -c 0 "${TFE_PYTHON}" \
  new_tests/new_secure_feature/new_test_tfe_multihead_updates.py
