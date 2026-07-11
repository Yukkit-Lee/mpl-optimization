#!/usr/bin/env bash
# 用途：自动检查并安装运行 module_c_adaptive_dpsgd.py / module_c_adaptive_bridge.py 所需依赖。
# 用法：
#   bash tools/env_auto_check_install_adaptive.sh           # 仅检查
#   bash tools/env_auto_check_install_adaptive.sh --install # 检查+自动安装缺失依赖
#   bash tools/env_auto_check_install_adaptive.sh --install --include-tfe

set -euo pipefail

DO_INSTALL=0
INCLUDE_TFE=0

for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    --include-tfe) INCLUDE_TFE=1 ;;
    *) echo "[错误] 未知参数: $arg"; exit 1 ;;
  esac
done

PY_BIN="${PYTHON_BIN:-python}"

echo "======================================================================"
echo "环境自动检查：Adaptive DPSGD"
echo "======================================================================"
echo "Python 可执行: $(command -v "$PY_BIN" || true)"
"$PY_BIN" - <<'PY'
import sys, platform
print("Python版本:", sys.version.replace("\n", " "))
print("平台:", platform.platform())
PY

REQ_COMMON=(
  numpy
  scipy
  sklearn
  matplotlib
  tqdm
  threadpoolctl
)

REQ_TORCH=(
  torch
)

REQ_TFE=(
  tensorflow
  tf_encrypted
)

check_pkg() {
  local pkg="$1"
  "$PY_BIN" - <<PY >/dev/null 2>&1
import importlib
import sys
name = "$pkg"
if name == "sklearn":
    import sklearn
else:
    importlib.import_module(name)
PY
}

missing=()
for p in "${REQ_COMMON[@]}" "${REQ_TORCH[@]}"; do
  if check_pkg "$p"; then
    echo "[OK] $p"
  else
    echo "[MISS] $p"
    missing+=("$p")
  fi
done

if [[ "$INCLUDE_TFE" == "1" ]]; then
  for p in "${REQ_TFE[@]}"; do
    if check_pkg "$p"; then
      echo "[OK] $p"
    else
      echo "[MISS] $p"
      missing+=("$p")
    fi
  done
fi

echo "----------------------------------------------------------------------"
if [[ ${#missing[@]} -eq 0 ]]; then
  echo "依赖检查通过：无缺失包。"
  exit 0
fi

echo "缺失依赖: ${missing[*]}"

if [[ "$DO_INSTALL" != "1" ]]; then
  echo "仅检查模式：未执行安装。"
  echo "如需自动安装，请运行："
  echo "  bash tools/env_auto_check_install_adaptive.sh --install"
  echo "若要同时检查/安装 TFE 依赖："
  echo "  bash tools/env_auto_check_install_adaptive.sh --install --include-tfe"
  exit 0
fi

# 安装策略：
# 1) 优先安装通用包
# 2) torch 按 Python 版本选择
#    - Python<=3.6：安装 torch==1.10.2（CPU）
#    - Python>=3.8：安装 torch（默认最新稳定）

echo "开始自动安装缺失依赖..."

install_common=()
for p in "${missing[@]}"; do
  if [[ "$p" != "torch" && "$p" != "tensorflow" && "$p" != "tf_encrypted" ]]; then
    install_common+=("$p")
  fi
done

if [[ ${#install_common[@]} -gt 0 ]]; then
  echo "安装通用依赖: ${install_common[*]}"
  "$PY_BIN" -m pip install --upgrade "${install_common[@]}"
fi

if printf '%s\n' "${missing[@]}" | grep -qx "torch"; then
  py_minor=$("$PY_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
  echo "检测到缺失 torch，Python=$py_minor"

  if [[ "$py_minor" == "3.6" || "$py_minor" == "3.7" ]]; then
    echo "安装 torch==1.10.2（兼容旧 Python，CPU 版）..."
    "$PY_BIN" -m pip install "torch==1.10.2"
  else
    echo "安装最新稳定 torch..."
    "$PY_BIN" -m pip install torch
  fi
fi

if printf '%s\n' "${missing[@]}" | grep -qx "tensorflow"; then
  echo "安装 tensorflow（注意：旧环境可能需要你手动锁版本）..."
  "$PY_BIN" -m pip install tensorflow
fi

if printf '%s\n' "${missing[@]}" | grep -qx "tf_encrypted"; then
  echo "安装 tf-encrypted..."
  "$PY_BIN" -m pip install tf-encrypted
fi

echo "----------------------------------------------------------------------"
echo "安装完成，正在复检..."

failed=0
for p in "${REQ_COMMON[@]}" "${REQ_TORCH[@]}"; do
  if check_pkg "$p"; then
    echo "[OK] $p"
  else
    echo "[FAIL] $p"
    failed=1
  fi
done

if [[ "$INCLUDE_TFE" == "1" ]]; then
  for p in "${REQ_TFE[@]}"; do
    if check_pkg "$p"; then
      echo "[OK] $p"
    else
      echo "[FAIL] $p"
      failed=1
    fi
  done
fi

if [[ "$failed" == "1" ]]; then
  echo "存在安装失败项，请根据日志手动处理。"
  exit 2
fi

echo "依赖复检通过。"
