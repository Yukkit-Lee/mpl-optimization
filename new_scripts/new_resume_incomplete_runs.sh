#!/usr/bin/env bash
# 用途：重新进入 tmux 并恢复所有未标记 COMPLETE 的任务。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"
NEW_TMUX_ATTACH=0 bash "${SCRIPT_DIR}/new_run_dual_gpu_formal_experiments.sh"
