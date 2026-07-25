#!/usr/bin/env bash
# 用途：在 tmux 的 gpu0/gpu1/tfe 窗口启动互斥任务队列。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"
SESSION=new_secure_feature_formal
bash "${SCRIPT_DIR}/new_create_tmux_formal_session.sh"

declare -A QUEUES=(
  [gpu0]="${NEW_PROJECT_ROOT}/new_configs/new_secure_feature/new_gpu0_tasks.tsv"
  [gpu1]="${NEW_PROJECT_ROOT}/new_configs/new_secure_feature/new_gpu1_tasks.tsv"
  [tfe]="${NEW_PROJECT_ROOT}/new_configs/new_secure_feature/new_tfe_tasks.tsv"
)
for resource in gpu0 gpu1 tfe; do
  queue="${QUEUES[$resource]}"
  [[ -f "${queue}" ]] || { echo "missing queue: ${queue}"; continue; }
  pane_pid="$("${NEW_TMUX}" display-message -p -t "${SESSION}:${resource}" '#{pane_pid}')"
  if pgrep -P "${pane_pid}" -f new_run_task_queue.sh >/dev/null 2>&1; then
    echo "queue already active: ${resource}"
    continue
  fi
  "${NEW_TMUX}" send-keys -t "${SESSION}:${resource}" \
    "bash ${SCRIPT_DIR}/new_run_task_queue.sh ${resource} ${queue}" C-m
done
if [[ "${NEW_TMUX_ATTACH:-1}" == "1" ]] && [[ -t 0 ]] && [[ -t 1 ]]; then
  "${NEW_TMUX}" attach-session -t "${SESSION}"
else
  echo "queues submitted to tmux session: ${SESSION}"
fi
