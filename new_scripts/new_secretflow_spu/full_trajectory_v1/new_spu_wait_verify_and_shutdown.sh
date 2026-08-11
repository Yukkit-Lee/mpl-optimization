#!/usr/bin/env bash
# 在严格串行 C100 队列之后执行的收尾守护器。
#
# 此脚本不运行任何训练、不会重跑 cell，也不会替代逐条 audit。它只在四条
# C100 seed43/44 新增正式 cell 都已经由 supervisor 写出 COMPLETE report 和
# passed audit 后，调用最终 12-cell 验收门；任何队列异常/缺失证据都会退出，
# 绝不关机。
#
# 建议在独立 tmux 会话运行：
#   tmux new-session -d -s finalization_guard \
#     'cd /root/autodl-tmp/tfe_experiments && exec bash \
#      new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_wait_verify_and_shutdown.sh \
#      2>&1 | tee -a new_log/new_secretflow_spu/full_trajectory_v1/new_spu_finalization_guard.log'

set -euo pipefail

repo="/root/autodl-tmp/tfe_experiments"
cd "$repo"
logdir="new_log/new_secretflow_spu/full_trajectory_v1"
queue_pattern='new_spu_serial_remaining_queue\.sh'
runner_pattern='new_spu_full_trajectory_formal\.py'
verifier="new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_final_verify_and_shutdown.sh"

required=(
  "full_trajectory_v1_formal_cifar100_true_ours_seed43"
  "full_trajectory_v1_formal_cifar100_true_ours_seed44"
  "full_trajectory_v1_formal_cifar100_full_data_seed43"
  "full_trajectory_v1_formal_cifar100_full_data_seed44"
)

completed_count() {
  local count=0 run report audit
  for run in "${required[@]}"; do
    report="$logdir/${run}_report.json"
    audit="$logdir/${run}_audit.json"
    if [[ -f "$report" && -f "$audit" ]] && \
       /root/autodl-tmp/new_envs/spu_e2e/bin/python - "$report" "$audit" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
ok = report.get("status") == "COMPLETE" and audit.get("passed") is True
raise SystemExit(0 if ok else 1)
PY
    then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

while :; do
  complete="$(completed_count)"
  echo "[$(date -u +%FT%TZ)] finalization guard: ${complete}/4 remaining C100 cells audited COMPLETE"
  if [[ "$complete" -eq 4 ]]; then
    break
  fi

  # supervisor 或正式 runner 消失而证据仍不完整，说明需要人工诊断；不能将
  # 静默失败当作完成，更不能关闭服务器。
  if ! pgrep -af "$queue_pattern" >/dev/null && ! pgrep -af "$runner_pattern" >/dev/null; then
    echo "FINALIZATION_GUARD_FAILED: queue/runner ended before all required audited cells existed" >&2
    exit 1
  fi
  sleep 60
done

# supervisor 的最后一次 push 可能刚完成；最终门包含 HEAD=origin 的检查。
if ! bash "$verifier" --check; then
  echo "FINALIZATION_GUARD_FAILED: final verification refused shutdown" >&2
  exit 1
fi

echo "FINALIZATION_GUARD_PASSED: scheduling protected shutdown."
exec bash "$verifier" --shutdown --confirm-shutdown
