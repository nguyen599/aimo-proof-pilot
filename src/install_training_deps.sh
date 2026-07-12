#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export UV_BREAK_SYSTEM_PACKAGES=1
export GIT_LFS_SKIP_SMUDGE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_CONFIG_FILE="${CUDA_CONFIG_FILE:-${SCRIPT_DIR}/cuda_config.sh}"
# shellcheck source=src/cuda_config.sh
source "${CUDA_CONFIG_FILE}"

APP_DIR="${APP_DIR:-/app}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${APP_DIR}/requirements.txt}"
INSTALL_SINGULARITY_CE="${INSTALL_SINGULARITY_CE:-0}"
INSTALL_APPTAINER_IN_IMAGE="${INSTALL_APPTAINER_IN_IMAGE:-0}"
INSTALL_MODAL_SIMP_EXTRAS="${INSTALL_MODAL_SIMP_EXTRAS:-0}"
INSTALL_MAGI_ATTENTION="${INSTALL_MAGI_ATTENTION:-1}"

OPEN_INSTRUCT_REPO="${OPEN_INSTRUCT_REPO:-https://github.com/nguyen599/open-instruct.git}"
OPEN_INSTRUCT_REF="${OPEN_INSTRUCT_REF:-main}"
OPEN_INSTRUCT_DIR="${OPEN_INSTRUCT_DIR:-/opt/open-instruct}"

OLMO_CORE_REPO="${OLMO_CORE_REPO:-https://github.com/nguyen599/OLMo-core.git}"
OLMO_CORE_REF="${OLMO_CORE_REF:-main}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-/opt/OLMo-core}"

uv_pip() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        uv pip "$@"
    else
        uv pip --system "$@"
    fi
}

install_selected_nccl_wheel() {
    if python - "${NVIDIA_NCCL_DIST}" >/dev/null 2>&1 <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version(sys.argv[1])
except PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if installed == "2.29.3" else 1)
PY
    then
        echo "Skipping ${NVIDIA_NCCL_DIST} install; version 2.29.3 is already installed."
    else
        uv_pip install --no-cache-dir --no-deps "${NVIDIA_NCCL_DIST}==2.29.3"
    fi
}

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    git-lfs \
    libaio-dev \
    libnuma-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    wget
rm -rf /var/lib/apt/lists/*

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --no-cache-dir --upgrade uv
fi

set +x
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
fi
set -x

uv_pip install --upgrade --no-cache-dir pip setuptools wheel packaging ninja
uv_pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"
# Transformers 5.8 imports the optional `kernels` registry while Open-Instruct
# enumerates auto model classes. Current `kernels` releases raise at import
# time because several registry entries omit revision/version metadata.
uv_pip uninstall kernels || true
uv_pip install --no-cache-dir --no-deps ring-flash-attn==0.1.8
install_selected_nccl_wheel

if [ "${INSTALL_MODAL_SIMP_EXTRAS}" = "1" ]; then
    INSTALL_MAGI_ATTENTION="${INSTALL_MAGI_ATTENTION}" \
        bash "${APP_DIR}/install_modal_simp_extras.sh"
fi

if [ "${INSTALL_SINGULARITY_CE}" = "1" ]; then
    bash "${APP_DIR}/install_singularity_ce.sh"
fi
if [ "${INSTALL_APPTAINER_IN_IMAGE}" = "1" ]; then
    echo "INSTALL_APPTAINER_IN_IMAGE=1 requested, but this image build does not install Apptainer in-container."
    echo "Use the host build helper to install Apptainer on the builder node instead."
else
    echo "Skipping Apptainer install inside the SIF image."
fi

if [ ! -d "${OLMO_CORE_DIR}/.git" ]; then
    git clone "${OLMO_CORE_REPO}" "${OLMO_CORE_DIR}"
fi
git -C "${OLMO_CORE_DIR}" fetch --depth 1 origin "${OLMO_CORE_REF}" || true
git -C "${OLMO_CORE_DIR}" checkout "${OLMO_CORE_REF}"

if [ ! -d "${OPEN_INSTRUCT_DIR}/.git" ]; then
    git clone "${OPEN_INSTRUCT_REPO}" "${OPEN_INSTRUCT_DIR}"
fi
git -C "${OPEN_INSTRUCT_DIR}" fetch --depth 1 origin "${OPEN_INSTRUCT_REF}" || true
git -C "${OPEN_INSTRUCT_DIR}" checkout "${OPEN_INSTRUCT_REF}"
OPEN_INSTRUCT_DIR="${OPEN_INSTRUCT_DIR}" python "${APP_DIR}/patch_open_instruct_adamw8bit.py"

# Some base images include the unrelated `beaker` package. OLMo-core's
# Beaker integration expects `beaker-py`, whose import package is also
# named `beaker`.
uv_pip uninstall beaker || true
uv_pip install --no-cache-dir -e "${OLMO_CORE_DIR}[beaker,wandb]"
uv_pip install --no-cache-dir --no-deps -e "${OPEN_INSTRUCT_DIR}"
uv_pip install --upgrade --pre --no-cache-dir --no-deps \
    --index-url "${TORCHAO_INDEX_URL}" "torchao"
# Editable dependency installs may reintroduce optional packages; keep the final
# runtime aligned with the Open-Instruct and OLMo-core paths tested in Modal.
uv_pip uninstall kernels || true
uv_pip install --no-cache-dir --no-deps ring-flash-attn==0.1.8
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("ring_flash_attn")
path = (
    Path(next(iter(spec.submodule_search_locations))) / "adapters" / "hf_adapter.py"
    if spec is not None and spec.submodule_search_locations
    else Path("/__missing_ring_flash_attn__/hf_adapter.py")
)
if path.exists():
    source = path.read_text()
    old = """try:
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )
except ImportError:
    # transformers <= 4.53.x
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )
"""
    new = """try:
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )
except ImportError:
    # transformers >= 5 removed this helper; ring-flash-attn uses it as a bool.
    is_flash_attn_greater_or_equal_2_10 = True
"""
    if old in source:
        path.write_text(source.replace(old, new))
PY
# The shared CUDA base may include an mslk wheel built against a different
# torch ABI. torchao treats mslk as optional, but its presence can make
# `import torchao` fail, so remove it in the OLMo training layer.
uv_pip uninstall mslk || true
install_selected_nccl_wheel
