#!/usr/bin/env bash
# 用途：串行消费指定 GPU/TFE 的 new_ 任务队列。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"

RESOURCE="${1:?resource required}"
QUEUE="${2:?queue file required}"
while IFS=$'\t' read -r task_id checkpoint command; do
  [[ -z "${task_id}" || "${task_id}" == \#* ]] && continue
  status="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_${task_id}.json"
  if [[ -f "${status}" ]] && grep -q '\"status\":\"COMPLETE\"' "${status}"; then
    echo "skip complete: ${task_id}"
    continue
  fi
  bash "${SCRIPT_DIR}/new_run_task_with_recovery.sh" \
    "${task_id}" "${RESOURCE}" "${checkpoint}" bash -lc "${command}"
done < "${QUEUE}"
