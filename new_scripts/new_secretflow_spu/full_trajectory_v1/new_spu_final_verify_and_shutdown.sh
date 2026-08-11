#!/usr/bin/env bash
# SPU C10/C100 n=3 最终验收与受保护关机入口。
#
# 用法：
#   bash new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_final_verify_and_shutdown.sh --check
#   bash new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_final_verify_and_shutdown.sh --shutdown --confirm-shutdown
#
# 默认及 --check 只读检查，绝不停止任务或关机。--shutdown 同时要求显式
# --confirm-shutdown；并且只有全部验收门通过后才会执行 /sbin/shutdown -h now。
# 用户已有的其他未提交绘图/论文修改不作为失败条件；本脚本只要求本SPU正式
# 产物路径和 project_progress.md 没有未提交变更，并要求 HEAD 已推送到
# origin/unified-experiments。

set -euo pipefail

mode="check"
if [[ $# -eq 0 || "$1" == "--check" ]]; then
  :
elif [[ "$1" == "--shutdown" && "${2:-}" == "--confirm-shutdown" && $# -eq 2 ]]; then
  mode="shutdown"
else
  echo "Usage: $0 [--check | --shutdown --confirm-shutdown]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

log_dir="new_log/new_secretflow_spu/full_trajectory_v1"
out_dir="new_output/new_secretflow_spu/full_trajectory_v1"
required=()
for dataset in cifar10 cifar100; do
  for method in true_ours full_data; do
    for seed in 42 43 44; do
      required+=("full_trajectory_v1_formal_${dataset}_${method}_seed${seed}")
    done
  done
done

fail=0
note_fail() { echo "FAIL: $*" >&2; fail=1; }
note_pass() { echo "PASS: $*"; }

# 1) 训练不可仍在运行：不把尚未落盘的结果误判为可关机。
if tmux ls 2>/dev/null | grep -Eq 'spu_(serial|n3)_'; then
  note_fail "仍有SPU tmux会话"
else
  note_pass "无SPU tmux会话"
fi
if pgrep -af 'new_spu_full_trajectory_formal\.py' >/dev/null; then
  note_fail "仍有SPU正式训练进程"
else
  note_pass "无SPU正式训练进程"
fi

# 2) 每条正式cell必须有COMPLETE report、passed audit、三份轨迹/配置文件。
for run_id in "${required[@]}"; do
  report="$log_dir/${run_id}_report.json"
  audit="$log_dir/${run_id}_audit.json"
  trace="$out_dir/$run_id/new_spu_full_trajectory_trace.csv"
  config="$out_dir/$run_id/effective_config.json"
  if [[ ! -f "$report" || ! -f "$audit" || ! -f "$trace" || ! -f "$config" ]]; then
    note_fail "$run_id 缺少 report/audit/trace/effective_config"
    continue
  fi
  if ! /root/autodl-tmp/new_envs/spu_e2e/bin/python - "$report" "$audit" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
ok = (
    report.get("status") == "COMPLETE"
    and report.get("run_kind") == "formal"
    and report.get("official_test_read") is False
    and report.get("locked_holdout_read") is False
    and audit.get("passed") is True
)
raise SystemExit(0 if ok else 1)
PY
  then
    note_fail "$run_id 的COMPLETE/范围/audit门失败"
  else
    note_pass "$run_id 完整且audit通过"
  fi
done

# 3) 仅限制每条正式 cell 的四份必需产物和进度文档。不能把运行时 tee
# 日志、旧 gate 产物或用户自己的其它图表修改误判为本任务未提交；它们不属于
# 最终验收的必要证据，也不应被本脚本删除或强制提交。
status_paths=("progressLog/project_progress.md")
for run_id in "${required[@]}"; do
  status_paths+=(
    "$log_dir/${run_id}_report.json"
    "$log_dir/${run_id}_audit.json"
    "$out_dir/$run_id/new_spu_full_trajectory_trace.csv"
    "$out_dir/$run_id/effective_config.json"
  )
done
if git status --porcelain -- "${status_paths[@]}" | grep -q .; then
  note_fail "SPU正式产物或进度文档仍有未提交变更"
else
  note_pass "SPU正式产物与进度文档已提交"
fi

# 4) 关机前必须确认当前提交与已同步的远端跟踪引用一致，避免结果只留在本机。
# 这里刻意不做 git fetch：关机脚本必须在无交互认证时也能安全工作；每个正式
# cell完成后由运行器使用配置的GIT_ASKPASS执行明确的 push，随后该跟踪引用更新。
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/unified-experiments)" ]]; then
  note_fail "HEAD未与origin/unified-experiments同步"
else
  note_pass "HEAD已推送到origin/unified-experiments"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "FINAL_VERIFICATION=FAILED（未执行关机）" >&2
  exit 1
fi

echo "FINAL_VERIFICATION=PASSED"
if [[ "$mode" == "shutdown" ]]; then
  echo "所有验收门通过，60秒后关机；使用 shutdown -c 可在窗口内取消。"
  /sbin/shutdown -h +1 "SPU final verification passed; requested shutdown"
else
  echo "检查模式：未执行关机。"
fi
