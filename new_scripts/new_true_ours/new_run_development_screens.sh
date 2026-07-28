#!/usr/bin/env bash
set -euo pipefail

project_root="${NEW_PROJECT_ROOT:-/root/autodl-tmp/tfe_experiments}"
python_bin="${NEW_MODERN_PYTHON:-/root/miniconda3/bin/python}"
log_dir="${project_root}/new_log/new_true_ours/txt"

export NEW_PROJECT_ROOT="${project_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"

mkdir -p "${log_dir}"
cd "${project_root}"

"${python_bin}" \
  new_scripts/new_true_ours/new_screen_true_ours_fedavg.py \
  --datasets cifar10 cifar100 --seed 42 \
  >"${log_dir}/new_stage_b_fedavg_screen_seed42_corrected.log" 2>&1

"${python_bin}" \
  new_scripts/new_true_ours/new_screen_true_ours_stage_b.py \
  --datasets tiny --seed 42 \
  >"${log_dir}/new_stage_b_screen_tiny_seed42.log" 2>&1

"${python_bin}" \
  new_scripts/new_true_ours/new_screen_true_ours_fedavg.py \
  --datasets tiny --seed 42 \
  >"${log_dir}/new_stage_b_fedavg_screen_tiny_seed42.log" 2>&1
