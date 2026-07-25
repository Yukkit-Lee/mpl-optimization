#!/usr/bin/env bash
# 用途：带 GPU/TFE 文件锁、heartbeat、PID、状态和最多两次恢复的任务包装器。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"

if [[ $# -lt 4 ]]; then
  echo "用法: $0 <task_id> <resource:gpu0|gpu1|tfe|report> <checkpoint_path> <command...>" >&2
  exit 2
fi

TASK_ID="$1"; RESOURCE="$2"; CHECKPOINT_PATH="$3"; shift 3
case "${RESOURCE}" in
  gpu0) LOCK=/tmp/new_secure_feature_gpu0.lock; export CUDA_VISIBLE_DEVICES=0 ;;
  gpu1) LOCK=/tmp/new_secure_feature_gpu1.lock; export CUDA_VISIBLE_DEVICES=1 ;;
  tfe) LOCK=/tmp/new_secure_feature_tfe.lock; export CUDA_VISIBLE_DEVICES="" ;;
  report) LOCK=/tmp/new_secure_feature_report.lock; export CUDA_VISIBLE_DEVICES="" ;;
  *) echo "未知资源: ${RESOURCE}" >&2; exit 2 ;;
esac

BASE="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature"
LOG="${BASE}/txt/new_${TASK_ID}.log"
PID_FILE="${BASE}/pids/new_${TASK_ID}.pid"
HEARTBEAT="${BASE}/heartbeat/new_${TASK_ID}.heartbeat"
STATUS="${BASE}/status/new_${TASK_ID}.json"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${PID_FILE}")" "$(dirname "${HEARTBEAT}")" "$(dirname "${STATUS}")"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "资源锁已占用: ${LOCK}" >&2
  exit 75
fi

available_gib="$(free -g | awk '/^Mem:/ {print $7}')"
project_free_gib="$(df -BG "${NEW_PROJECT_ROOT}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
home_free_gib="$(df -BG "${NEW_DATA_ROOT}" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
if (( available_gib < 16 || project_free_gib < 25 || home_free_gib < 25 )); then
  printf '{"task":"%s","status":"BLOCKED_RESOURCE","ram_gib":%s,"project_free_gib":%s,"home_free_gib":%s}\n' \
    "${TASK_ID}" "${available_gib}" "${project_free_gib}" "${home_free_gib}" > "${STATUS}"
  exit 76
fi

echo "$$" > "${PID_FILE}"
heartbeat_loop() {
  while true; do
    date --iso-8601=seconds > "${HEARTBEAT}"
    sleep 60
  done
}
heartbeat_loop &
HEARTBEAT_PID=$!
cleanup() {
  kill "${HEARTBEAT_PID}" 2>/dev/null || true
  rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

attempt=0
max_retries=2
while (( attempt <= max_retries )); do
  printf '{"task":"%s","resource":"%s","status":"RUNNING","attempt":%d,"checkpoint":"%s","started_at":"%s"}\n' \
    "${TASK_ID}" "${RESOURCE}" "${attempt}" "${CHECKPOINT_PATH}" "$(date --iso-8601=seconds)" > "${STATUS}"
  echo "[$(date --iso-8601=seconds)] task=${TASK_ID} resource=${RESOURCE} attempt=${attempt}" | tee -a "${LOG}"
  set +e
  NEW_RESUME_FROM="${CHECKPOINT_PATH}" "$@" >> "${LOG}" 2>&1
  code=$?
  set -e
  if (( code == 0 )); then
    printf '{"task":"%s","resource":"%s","status":"COMPLETE","attempt":%d,"checkpoint":"%s","finished_at":"%s"}\n' \
      "${TASK_ID}" "${RESOURCE}" "${attempt}" "${CHECKPOINT_PATH}" "$(date --iso-8601=seconds)" > "${STATUS}"
    exit 0
  fi
  printf '{"task":"%s","resource":"%s","status":"RETRYING","attempt":%d,"exit_code":%d,"checkpoint":"%s"}\n' \
    "${TASK_ID}" "${RESOURCE}" "${attempt}" "${code}" "${CHECKPOINT_PATH}" > "${STATUS}"
  attempt=$((attempt + 1))
  sleep 15
done

printf '{"task":"%s","resource":"%s","status":"FAILED","attempts":%d,"checkpoint":"%s","finished_at":"%s"}\n' \
  "${TASK_ID}" "${RESOURCE}" "$((max_retries + 1))" "${CHECKPOINT_PATH}" "$(date --iso-8601=seconds)" > "${STATUS}"
exit 1
