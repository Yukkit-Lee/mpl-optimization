#!/usr/bin/env bash
# 用途：以单任务锁、单 GPU 和单 CPU/BLAS 线程安全运行实验。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${ROOT}/.cross_stage_experiment.lock"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "用法: $0 <command> [args...]" >&2
  exit 2
fi

cd "${ROOT}"
flock -n "${LOCK_FILE}" "$@"
