#!/usr/bin/env bash
set -euo pipefail

NEW_STAGE_B_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEW_STAGE_B_THREADS="$(nproc)"
NEW_STAGE_B_LOG_DIR="${NEW_STAGE_B_ROOT}/new_log/new_stage_b_kd/txt"
mkdir -p "${NEW_STAGE_B_LOG_DIR}"

export OMP_NUM_THREADS="${NEW_STAGE_B_THREADS}"
export MKL_NUM_THREADS="${NEW_STAGE_B_THREADS}"
export OPENBLAS_NUM_THREADS="${NEW_STAGE_B_THREADS}"
export NUMEXPR_NUM_THREADS="${NEW_STAGE_B_THREADS}"
export VECLIB_MAXIMUM_THREADS="${NEW_STAGE_B_THREADS}"

cd "${NEW_STAGE_B_ROOT}"
python new_scripts/new_stage_b_kd/new_preflight_stage_b_kd.py \
  2>&1 | tee "${NEW_STAGE_B_LOG_DIR}/new_stage_b_kd_preflight.log"
python new_scripts/new_stage_b_kd/new_run_stage_b_kd_plaintext.py \
  --datasets cifar10 cifar100 \
  --seed 42 \
  2>&1 | tee "${NEW_STAGE_B_LOG_DIR}/new_stage_b_kd_plaintext_seed42.log"
