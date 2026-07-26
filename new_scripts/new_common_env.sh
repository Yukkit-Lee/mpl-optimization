#!/usr/bin/env bash
# 用途：所有 new_ 正式任务统一加载的 AutoDL 高吞吐环境。
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=UTF-8
export NEW_SINGLE_GPU="${NEW_SINGLE_GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export NEW_PROJECT_ROOT="${NEW_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
DEFAULT_WORKSPACE_ROOT="$(cd "${NEW_PROJECT_ROOT}/.." && pwd)"
export NEW_MTT_ROOT="${NEW_MTT_ROOT:-${DEFAULT_WORKSPACE_ROOT}/mtt-distillation}"
export NEW_DATA_ROOT="${NEW_DATA_ROOT:-${DEFAULT_WORKSPACE_ROOT}/new_data}"
export NEW_OFFICIAL_MTT_ROOT="${NEW_OFFICIAL_MTT_ROOT:-${NEW_DATA_ROOT}/mtt_data/georgecazenavette}"
export NEW_OUTPUT_ROOT="${NEW_OUTPUT_ROOT:-${NEW_PROJECT_ROOT}}"
export NEW_SECRET_ROOT="${NEW_SECRET_ROOT:-${NEW_DATA_ROOT}/new_secure_feature_secrets}"
export NEW_ENV_ROOT="${NEW_ENV_ROOT:-${DEFAULT_WORKSPACE_ROOT}/new_envs}"
export NEW_MODERN_PYTHON="${NEW_MODERN_PYTHON:-${NEW_ENV_ROOT}/distillation/bin/python}"
export NEW_TFE_PYTHON="${NEW_TFE_PYTHON:-${NEW_ENV_ROOT}/tfe/bin/python}"
export NEW_TMUX="${NEW_TMUX:-$(command -v tmux || printf 'tmux')}"
export CUDA_VISIBLE_DEVICES="${NEW_SINGLE_GPU}"
