#!/usr/bin/env bash
# 严格串行的SPU n=3剩余队列监督器。由tmux运行；每条cell需audit、提交、推送后才启动下一条。
# 当前已运行cifar100/true_ours/seed43；后续顺序为ours44、full43、full44。失败即退出，绝不关机。
set -euo pipefail
repo=/root/autodl-tmp/tfe_experiments
cd "$repo"
export RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu TASKSET_CPUSET=96-127
export PYTHONPATH="$repo/new_scripts/new_secretflow_spu/full_trajectory_v1:$repo/new_scripts/new_secretflow_spu/corrected_v2:$repo/new_scripts/new_secretflow_spu:$repo"
py=/root/autodl-tmp/new_envs/spu_e2e/bin/python
runner=new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_formal.py
auditor=new_scripts/new_secretflow_spu/full_trajectory_v1/new_audit_spu_full_trajectory.py
logdir=new_log/new_secretflow_spu/full_trajectory_v1

wait_complete() {
  local method="$1" seed="$2" id="full_trajectory_v1_formal_cifar100_${method}_seed${seed}"
  local report="$logdir/${id}_report.json"
  while [[ ! -f "$report" ]]; do sleep 60; done
  "$py" - "$report" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get('status') == 'COMPLETE' else 1)
PY
}

audit_commit_push() {
  local method="$1" seed="$2" id="full_trajectory_v1_formal_cifar100_${method}_seed${seed}"
  "$py" "$auditor" --dataset cifar100 --method "$method" --seed "$seed"
  printf '%s\n' "- $(date -u +%F) serial supervisor: CIFAR-100 ${method} seed${seed} COMPLETE and strengthened audit PASS; exact artifacts committed/pushed before next cell." >> progressLog/project_progress.md
  git add "$logdir/new_spu_full_trajectory_cifar100_${method}_seed${seed}.log" "$logdir/${id}_audit.json" "new_output/new_secretflow_spu/full_trajectory_v1/${id}"
  git add -f "$logdir/${id}_report.json"
  git add progressLog/project_progress.md
  git commit -m "Add audited serial SPU C100 ${method} seed ${seed}"
  GIT_ASKPASS=/tmp/codex_git_askpass.sh GIT_TERMINAL_PROMPT=0 git push origin unified-experiments
}

run_cell() {
  local method="$1" seed="$2" batch="$3" chunk="$4" lr="$5"
  taskset -c 96-127 "$py" "$runner" --dataset cifar100 --method "$method" --seed "$seed" --batch_size "$batch" --batch_chunk_batches "$chunk" --local_lr "$lr" --unroll 1 --block_steps 10 --stage_b_eval_interval 5 --stage_b_eval_chunk_rows 250 2>&1 | tee "$logdir/new_spu_full_trajectory_cifar100_${method}_seed${seed}.log"
}

wait_complete true_ours 43
audit_commit_push true_ours 43
run_cell true_ours 44 32 25 1.0
wait_complete true_ours 44
audit_commit_push true_ours 44
run_cell full_data 43 64 16 0.8
wait_complete full_data 43
audit_commit_push full_data 43
run_cell full_data 44 64 16 0.8
wait_complete full_data 44
audit_commit_push full_data 44
echo "SERIAL_C100_QUEUE_COMPLETE: final aggregation/verification/shutdown still required."
