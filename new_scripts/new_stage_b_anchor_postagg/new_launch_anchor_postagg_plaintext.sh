#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/tfe_experiments

task_threads="$(nproc)"
export OMP_NUM_THREADS="${task_threads}"
export OPENBLAS_NUM_THREADS="${task_threads}"
export MKL_NUM_THREADS="${task_threads}"
export NUMEXPR_NUM_THREADS="${task_threads}"
export VECLIB_MAXIMUM_THREADS="${task_threads}"

mkdir -p new_log/new_stage_b_anchor_postagg/txt
exec /usr/bin/time -v \
  python new_scripts/new_stage_b_anchor_postagg/new_run_anchor_postagg_plaintext.py \
  2>&1 | tee new_log/new_stage_b_anchor_postagg/txt/new_stage_b_anchor_postagg_plaintext.log
