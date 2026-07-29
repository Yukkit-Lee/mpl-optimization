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

cd "${NEW_STAGE_B_ROOT}"
python new_scripts/new_stage_b_kd/new_preflight_stage_b_kd.py
python new_scripts/new_stage_b_kd/new_diagnose_anchor_supervision.py \
  2>&1 | tee "${NEW_STAGE_B_LOG_DIR}/new_stage_b_kd_anchor_supervision.log"
