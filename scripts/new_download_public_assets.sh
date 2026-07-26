#!/usr/bin/env bash
# Download only redistributable raw archives; official MTT tensors are obtained
# from the authors' tensor-download page and then verified with the local manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/new_scripts/new_common_env.sh"

download() {
  local url="$1" destination="$2"
  mkdir -p "$(dirname "${destination}")"
  [[ -f "${destination}" ]] || curl -fL --retry 5 --retry-delay 5 -o "${destination}.part" "${url}"
  [[ -f "${destination}" ]] || mv "${destination}.part" "${destination}"
}

download "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
  "${NEW_DATA_ROOT}/mtt_cifar10_work/data/cifar-10-python.tar.gz"
download "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
  "${NEW_DATA_ROOT}/mtt_cifar100_work/data/cifar-100-python.tar.gz"
download "http://cs231n.stanford.edu/tiny-imagenet-200.zip" \
  "${NEW_DATA_ROOT}/mtt_tiny_work/data/tiny-imagenet-200.zip"

cat <<EOF
Raw archives downloaded. Official non-ZCA MTT tensors are intentionally not
bundled in Git. Download CIFAR-10/CIFAR-100/Tiny tensor sets from:
  https://georgecazenavette.github.io/mtt-distillation/tensors/index.html
Place them under:
  ${NEW_OFFICIAL_MTT_ROOT}/{cifar10_50,cifar100_50,tiny_50}/
Then run:
  ${NEW_MODERN_PYTHON} ${PROJECT_ROOT}/scripts/new_verify_official_mtt_assets.py
EOF

