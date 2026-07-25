#!/usr/bin/env bash
# 用途：幂等创建单 GPU 正式实验 tmux，会话内仅允许一个训练队列。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"

SESSION=new_secure_feature_single_gpu_formal
if "${NEW_TMUX}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 0
fi
"${NEW_TMUX}" new-session -d -s "${SESSION}" -n gpu0 -c "${NEW_PROJECT_ROOT}"
for window in tfe monitor report; do
  "${NEW_TMUX}" new-window -t "${SESSION}" -n "${window}" -c "${NEW_PROJECT_ROOT}"
done
"${NEW_TMUX}" send-keys -t "${SESSION}:monitor" \
  "exec bash ${SCRIPT_DIR}/new_monitor_resources.sh" C-m
echo "created tmux session: ${SESSION}"
"${NEW_TMUX}" list-windows -t "${SESSION}"
