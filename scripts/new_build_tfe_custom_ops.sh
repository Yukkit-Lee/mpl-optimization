#!/usr/bin/env bash
# Reproducibly build TF-Encrypted 0.8.0 custom ops for TensorFlow 1.15.5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/new_scripts/new_common_env.sh"

TFE_REPO_URL="${NEW_TFE_REPO_URL:-https://github.com/tf-encrypted/tf-encrypted.git}"
TFE_COMMIT="${NEW_TFE_COMMIT:-51de98f9c2208d5d4bd716bf7316d3cd4e0bf434}"
BUILD_ROOT="${NEW_TFE_BUILD_ROOT:-${NEW_DATA_ROOT}/build/tf-encrypted-0.8.0}"
JOBS="${NEW_BUILD_JOBS:-1}"

for command in git make g++ curl tar autoconf automake libtoolize sha256sum; do
  command -v "${command}" >/dev/null || {
    echo "Missing build prerequisite: ${command}" >&2
    exit 2
  }
done
[[ -x "${NEW_TFE_PYTHON}" ]] || {
  echo "NEW_TFE_PYTHON is not executable: ${NEW_TFE_PYTHON}" >&2
  exit 2
}

export PATH="$(dirname "${NEW_TFE_PYTHON}"):${PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ ! -d "${BUILD_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${BUILD_ROOT}")"
  git clone "${TFE_REPO_URL}" "${BUILD_ROOT}"
fi
git -C "${BUILD_ROOT}" fetch --tags origin
git -C "${BUILD_ROOT}" checkout --detach "${TFE_COMMIT}"
actual_commit="$(git -C "${BUILD_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${TFE_COMMIT}" ]] || {
  echo "Unexpected TFE commit: ${actual_commit}" >&2
  exit 3
}

tf_version="$("${NEW_TFE_PYTHON}" -c 'import tensorflow as tf; print(tf.__version__)')"
[[ "${tf_version}" == "1.15.5" ]] || {
  echo "Expected TensorFlow 1.15.5, found ${tf_version}" >&2
  exit 3
}

make -C "${BUILD_ROOT}" -j"${JOBS}" build
site_packages="$("${NEW_TFE_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
package_root="${site_packages}/tf_encrypted/operations"
install -m 0755 \
  "${BUILD_ROOT}/tf_encrypted/operations/aux/aux_module_tf_1.15.5.so" \
  "${package_root}/aux/aux_module_tf_1.15.5.so"
install -m 0755 \
  "${BUILD_ROOT}/tf_encrypted/operations/secure_random/secure_random_module_tf_1.15.5.so" \
  "${package_root}/secure_random/secure_random_module_tf_1.15.5.so"

sha256sum \
  "${package_root}/aux/aux_module_tf_1.15.5.so" \
  "${package_root}/secure_random/secure_random_module_tf_1.15.5.so"
"${NEW_TFE_PYTHON}" - <<'PY'
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.operations.secure_random import secure_random
print("TensorFlow", tf.__version__)
print("TF-Encrypted", getattr(tfe, "__version__", "0.8.0"))
print("secure_random_loaded", secure_random is not None)
PY

