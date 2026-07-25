#!/usr/bin/env bash
# 用途：每 60 秒记录 GPU/CPU/RAM/swap/磁盘/PID/heartbeat 年龄。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"
OUT="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/txt/new_resource_monitor.log"
mkdir -p "$(dirname "${OUT}")"
while true; do
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader 2>&1 | sed 's/^/gpu=/'
    uptime | sed 's/^/cpu_load=/'
    free -h | sed 's/^/memory=/'
    df -h "${NEW_PROJECT_ROOT}" "${NEW_DATA_ROOT}" | sed 's/^/disk=/'
    pgrep -af 'new_.*\\.py|new_run_task' | sed 's/^/pid=/' || true
    now="$(date +%s)"
    for heartbeat in "${NEW_OUTPUT_ROOT}"/new_log/new_secure_feature/heartbeat/new_*.heartbeat; do
      [[ -e "${heartbeat}" ]] || continue
      modified="$(stat -c %Y "${heartbeat}")"
      echo "heartbeat=$(basename "${heartbeat}") age_seconds=$((now - modified))"
    done
    echo
  } >> "${OUT}"
  sleep 60
done
