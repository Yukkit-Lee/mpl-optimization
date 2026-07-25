#!/usr/bin/env bash
# 用途：系统重启或 tmux 丢失后，根据磁盘上的活动队列自动恢复单 GPU 正式实验。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"

ACTIVE_FILE="${NEW_PROJECT_ROOT}/new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
SUPERVISOR_STATUS="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_single_gpu_supervisor.json"
SESSION=new_secure_feature_single_gpu_formal
mkdir -p "$(dirname "${SUPERVISOR_STATUS}")"

exec 8>/tmp/new_secure_feature_single_gpu_supervisor.lock
flock -n 8 || exit 0

atomic_status() {
  local payload="$1"
  local tmp="${SUPERVISOR_STATUS}.new_tmp_$$"
  printf '%s\n' "${payload}" > "${tmp}"
  mv -f "${tmp}" "${SUPERVISOR_STATUS}"
}

queue_has_failed_task() {
  local task_id checkpoint command status
  while IFS=$'\t' read -r task_id checkpoint command; do
    [[ -z "${task_id}" || "${task_id}" == \#* ]] && continue
    status="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_${task_id}.json"
    if [[ -f "${status}" ]] && grep -q '"status":"FAILED"' "${status}"; then
      return 0
    fi
  done < "${QUEUE}"
  return 1
}

queue_is_complete() {
  local task_id checkpoint command status seen=0
  while IFS=$'\t' read -r task_id checkpoint command; do
    [[ -z "${task_id}" || "${task_id}" == \#* ]] && continue
    seen=1
    status="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_${task_id}.json"
    [[ -f "${status}" ]] && grep -q '"status":"COMPLETE"' "${status}" || return 1
  done < "${QUEUE}"
  (( seen == 1 ))
}

while true; do
  if [[ ! -s "${ACTIVE_FILE}" ]]; then
    atomic_status "$(printf '{"status":"IDLE","checked_at":"%s"}' "$(date --iso-8601=seconds)")"
    sleep 60
    continue
  fi
  QUEUE="$(head -n 1 "${ACTIVE_FILE}")"
  if [[ ! -f "${QUEUE}" ]]; then
    atomic_status "$(printf '{"status":"BLOCKED_MISSING_QUEUE","queue":"%s","checked_at":"%s"}' \
      "${QUEUE}" "$(date --iso-8601=seconds)")"
    sleep 60
    continue
  fi
  if queue_has_failed_task; then
    atomic_status "$(printf '{"status":"NEEDS_ATTENTION","queue":"%s","checked_at":"%s"}' \
      "${QUEUE}" "$(date --iso-8601=seconds)")"
    sleep 60
    continue
  fi
  if queue_is_complete; then
    atomic_status "$(printf '{"status":"QUEUE_COMPLETE","queue":"%s","checked_at":"%s"}' \
      "${QUEUE}" "$(date --iso-8601=seconds)")"
    sleep 60
    continue
  fi
  bash "${SCRIPT_DIR}/new_create_single_gpu_formal_session.sh" >/dev/null
  if ! pgrep -af "new_run_task_queue.sh gpu0 ${QUEUE}" >/dev/null; then
    CMD="cd ${NEW_PROJECT_ROOT} && exec bash ${SCRIPT_DIR}/new_run_task_queue.sh gpu0 ${QUEUE}"
    "${NEW_TMUX}" send-keys -t "${SESSION}:gpu0" "${CMD}" C-m
  fi
  atomic_status "$(printf '{"status":"RUNNING_OR_RESUMABLE","queue":"%s","checked_at":"%s","boot_id":"%s"}' \
    "${QUEUE}" "$(date --iso-8601=seconds)" "$(cat /proc/sys/kernel/random/boot_id)")"
  sleep 60
done
