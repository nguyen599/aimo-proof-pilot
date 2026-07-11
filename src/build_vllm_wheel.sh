#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_CONFIG_FILE="${CUDA_CONFIG_FILE:-${SCRIPT_DIR}/cuda_config.sh}"
# shellcheck source=src/cuda_config.sh
source "${CUDA_CONFIG_FILE}"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-f5a8d73377d0f0a4e00cba172f9fbd0d50471b07}"
VLLM_VERSION="${VLLM_VERSION:-0.23.1rc1.dev699+gf5a8d7337}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/opt/vllm-source}"
VLLM_WHEEL_DIR="${VLLM_WHEEL_DIR:-/root/vllm-dist}"
VLLM_WHEEL_NAME="${VLLM_WHEEL_NAME:-vllm-${VLLM_VERSION}-cp38-abi3-linux_x86_64.whl}"
VLLM_WHEEL_PATH="${VLLM_WHEEL_DIR}/${VLLM_WHEEL_NAME}"
VLLM_BUILD_MAX_JOBS="${VLLM_BUILD_MAX_JOBS:-8}"
VLLM_BUILD_NVCC_THREADS="${VLLM_BUILD_NVCC_THREADS:-4}"
VLLM_BUILD_CUDA_ARCH_LIST="${VLLM_BUILD_CUDA_ARCH_LIST:-9.0;10.0;12.0}"
VLLM_INSTALL_WHEEL="${VLLM_INSTALL_WHEEL:-1}"
VLLM_UPLOAD_WHEEL="${VLLM_UPLOAD_WHEEL:-0}"
VLLM_REUSE_PREBUILT="${VLLM_REUSE_PREBUILT:-1}"
VLLM_WHEEL_REPO="${VLLM_WHEEL_REPO:-nguyen599/prebuild-wheels-util}"
VLLM_WHEEL_REPO_PATH="${VLLM_WHEEL_REPO_PATH:-${VLLM_PREBUILT_WHEELS_DIR}/${VLLM_WHEEL_NAME}}"

ensure_hf_cli() {
    if ! command -v hf >/dev/null 2>&1; then
        uv pip install --system --no-cache-dir "huggingface_hub[cli]>=1.0.0"
    fi
}

download_prebuilt_wheel() {
    local tmp_dir
    local token_args=()

    set +x
    ensure_hf_cli
    if [ -n "${HF_TOKEN:-}" ]; then
        token_args=(--token "${HF_TOKEN}")
    fi
    tmp_dir="$(mktemp -d)"
    if HF_XET_HIGH_PERFORMANCE=1 hf download \
        --repo-type dataset \
        --local-dir "${tmp_dir}" \
        "${token_args[@]}" \
        "${VLLM_WHEEL_REPO}" \
        "${VLLM_WHEEL_REPO_PATH}"; then
        set -x
        mkdir -p "${VLLM_WHEEL_DIR}"
        mv "${tmp_dir}/${VLLM_WHEEL_REPO_PATH}" "${VLLM_WHEEL_PATH}"
        rm -rf "${tmp_dir}"
        return 0
    fi
    set -x
    rm -rf "${tmp_dir}"
    return 1
}

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --no-cache-dir --upgrade uv
fi

if [ "${VLLM_REUSE_PREBUILT}" = "1" ] && download_prebuilt_wheel; then
    echo "Reusing ${VLLM_WHEEL_REPO}/${VLLM_WHEEL_REPO_PATH}"
    if [ "${VLLM_INSTALL_WHEEL}" = "1" ]; then
        uv pip install --system --no-cache-dir --no-deps "${VLLM_WHEEL_PATH}"
    fi
    echo "VLLM_WHEEL_PATH=${VLLM_WHEEL_PATH}"
    echo "VLLM_WHEEL_REPO_PATH=${VLLM_WHEEL_REPO_PATH}"
    exit 0
fi

if [ ! -d "${VLLM_SOURCE_DIR}/.git" ]; then
    rm -rf "${VLLM_SOURCE_DIR}"
    git clone --filter=blob:none --no-checkout "${VLLM_REPO}" "${VLLM_SOURCE_DIR}"
fi
git -C "${VLLM_SOURCE_DIR}" remote set-url origin "${VLLM_REPO}"
git -C "${VLLM_SOURCE_DIR}" fetch --depth 1 origin "${VLLM_REF}"
git -C "${VLLM_SOURCE_DIR}" checkout --force FETCH_HEAD
git -C "${VLLM_SOURCE_DIR}" clean -fdx

cd "${VLLM_SOURCE_DIR}"
python use_existing_torch.py
if [ "${CUDA_MAJOR}" = "12" ]; then
    sed -i \
        -e 's/nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' \
        -e '/^humming-kernels\[cu13\]/d' \
        requirements/cuda.txt
fi

uv pip install --system --no-cache-dir -r requirements/build/cuda.txt
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
uv pip install --system --no-cache-dir \
    --index-strategy unsafe-best-match \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    --extra-index-url "${FLASHINFER_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    -r requirements/cuda.txt
uv pip install --system --no-cache-dir \
    --index-strategy unsafe-best-match \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "cuda-python[all]==${CUDA_PYTHON_VERSION}"
uv pip install --system --no-cache-dir --no-deps \
    "${NVIDIA_CUBLASMP_DIST}==${CUDA_CUBLASMP_VERSION}" \
    "${NVIDIA_NCCL_DIST}==2.29.3"

rm -rf "${VLLM_WHEEL_DIR}"
mkdir -p "${VLLM_WHEEL_DIR}"
MAX_JOBS="${VLLM_BUILD_MAX_JOBS}" \
NVCC_THREADS="${VLLM_BUILD_NVCC_THREADS}" \
TORCH_CUDA_ARCH_LIST="${VLLM_BUILD_CUDA_ARCH_LIST}" \
CMAKE_BUILD_TYPE=Release \
VLLM_MAIN_CUDA_VERSION="${CUDA_MAJOR_MINOR}" \
VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${VLLM_VERSION}" \
    python -m build --wheel --no-isolation --outdir "${VLLM_WHEEL_DIR}"

built_wheel="$(find "${VLLM_WHEEL_DIR}" -maxdepth 1 -name 'vllm-*.whl' -size +0c -print -quit)"
if [ -z "${built_wheel}" ]; then
    echo "vLLM build completed without producing a wheel in ${VLLM_WHEEL_DIR}" >&2
    exit 1
fi
if [ "${built_wheel}" != "${VLLM_WHEEL_PATH}" ]; then
    mv "${built_wheel}" "${VLLM_WHEEL_PATH}"
fi

if [ "${VLLM_INSTALL_WHEEL}" = "1" ]; then
    uv pip install --system --no-cache-dir --no-deps "${VLLM_WHEEL_PATH}"
fi

if [ "${VLLM_UPLOAD_WHEEL}" = "1" ]; then
    token_args=()
    set +x
    ensure_hf_cli
    if [ -n "${HF_TOKEN:-}" ]; then
        token_args=(--token "${HF_TOKEN}")
    fi
    HF_XET_HIGH_PERFORMANCE=1 hf upload \
        --repo-type dataset \
        --private \
        "${token_args[@]}" \
        "${VLLM_WHEEL_REPO}" \
        "${VLLM_WHEEL_PATH}" \
        "${VLLM_WHEEL_REPO_PATH}"
    set -x
fi

echo "VLLM_WHEEL_PATH=${VLLM_WHEEL_PATH}"
echo "VLLM_WHEEL_REPO_PATH=${VLLM_WHEEL_REPO_PATH}"
