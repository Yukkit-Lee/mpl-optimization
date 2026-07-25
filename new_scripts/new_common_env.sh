#!/usr/bin/env bash
# 用途：所有 new_ 正式任务统一加载的单线程安全环境。
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export MALLOC_ARENA_MAX=1

export NEW_PROJECT_ROOT="${NEW_PROJECT_ROOT:-/mnt/code/lixujie/tfe_experiments}"
export NEW_MTT_ROOT="${NEW_MTT_ROOT:-/mnt/code/lixujie/mtt-distillation}"
export NEW_DATA_ROOT="${NEW_DATA_ROOT:-/home/lixujie/tmpWork}"
export NEW_OFFICIAL_MTT_ROOT="${NEW_OFFICIAL_MTT_ROOT:-/home/lixujie/tmpWork/mtt_data/georgecazenavette}"
export NEW_OUTPUT_ROOT="${NEW_OUTPUT_ROOT:-${NEW_PROJECT_ROOT}}"
export NEW_MODERN_PYTHON="${NEW_MODERN_PYTHON:-/home/lixujie/.conda/envs/distillation/bin/python}"
export NEW_TFE_PYTHON="${NEW_TFE_PYTHON:-/home/lixujie/.conda/envs/tfe/bin/python}"
export NEW_TMUX="${NEW_TMUX:-/home/lixujie/tmpWork/new_tmux_runtime/bin/tmux}"
