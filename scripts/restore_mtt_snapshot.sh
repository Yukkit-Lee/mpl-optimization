#!/usr/bin/env bash
# 用途：从 migration/ 中保存的真实 MTT 快照恢复仓库、补丁和新增源码。
set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mtt_root="${MTT_ROOT:-$(dirname "${project_root}")/mtt-distillation}"
snapshot_root="${project_root}/migration/mtt"

if [[ ! -f "${snapshot_root}/mtt-history.bundle" ]]; then
  echo "缺少 MTT Git bundle: ${snapshot_root}/mtt-history.bundle" >&2
  exit 2
fi

if [[ -e "${mtt_root}" ]]; then
  echo "目标已存在，为避免覆盖已停止: ${mtt_root}" >&2
  exit 3
fi

git clone "${snapshot_root}/mtt-history.bundle" "${mtt_root}"
git -C "${mtt_root}" checkout main
git -C "${mtt_root}" apply --binary "${snapshot_root}/mtt-working-tree.patch"
tar -xzf "${snapshot_root}/mtt-untracked-source.tar.gz" -C "${mtt_root}"

echo "MTT 已恢复到: ${mtt_root}"
echo "基线 commit: $(cat "${snapshot_root}/commit.txt")"
git -C "${mtt_root}" status --short --branch

