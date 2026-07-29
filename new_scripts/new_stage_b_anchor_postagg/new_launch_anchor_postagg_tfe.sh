#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/tfe_experiments
task_threads="$(nproc)"
export OMP_NUM_THREADS="${task_threads}"
export OPENBLAS_NUM_THREADS="${task_threads}"
export MKL_NUM_THREADS="${task_threads}"
export NUMEXPR_NUM_THREADS="${task_threads}"

mkdir -p new_log/new_stage_b_anchor_postagg/txt
python new_scripts/new_stage_b_anchor_postagg/new_run_anchor_postagg_tfe.py \
  2>&1 | tee new_log/new_stage_b_anchor_postagg/txt/new_stage_b_anchor_postagg_tfe_runner.log
