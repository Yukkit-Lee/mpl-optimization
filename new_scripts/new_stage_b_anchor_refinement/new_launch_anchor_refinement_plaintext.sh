#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="stage_b_anchor_refinement_plaintext"
LOG_DIR="${ROOT_DIR}/new_log/new_stage_b_anchor_refinement/txt"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 1
fi

CPU_COUNT="$(nproc)"
COMMAND="cd '${ROOT_DIR}' && export OMP_NUM_THREADS='${CPU_COUNT}' MKL_NUM_THREADS='${CPU_COUNT}' OPENBLAS_NUM_THREADS='${CPU_COUNT}' NUMEXPR_NUM_THREADS='${CPU_COUNT}' PYTHONUNBUFFERED=1 && python new_scripts/new_stage_b_anchor_refinement/new_run_anchor_refinement_plaintext.py 2>&1 | tee '${LOG_DIR}/new_stage_b_anchor_refinement_plaintext.log'"
tmux new-session -d -s "${SESSION}" "${COMMAND}"
echo "started ${SESSION} with ${CPU_COUNT} threads"
