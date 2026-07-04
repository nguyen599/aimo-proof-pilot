#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export UV_BREAK_SYSTEM_PACKAGES=1

CUDA_VERSION="${CUDA_VERSION:-13.0.2}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-90}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
NV_GROUPED_GEMM_TORCH_CUDA_ARCH_LIST="${NV_GROUPED_GEMM_TORCH_CUDA_ARCH_LIST:-9.0}"
FLASHINFER_INDEX_URL="${FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/cu130}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl}"
FLASH_ATTN_3_WHEEL_URL="${FLASH_ATTN_3_WHEEL_URL:-https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.05.11-5e0e3b1/flash_attn_3-3.0.0+20260511.cu130torch2110cxx11abitrue.ab6632-cp39-abi3-linux_x86_64.whl}"
FLASH_ATTN_4_WHEEL_URL="${FLASH_ATTN_4_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta15/flash_attn_4-4.0.0b15-py3-none-any.whl}"
# Stable vLLM 0.24.0 release wheel (what prime-rl@main pins) instead of a wheels.vllm.ai nightly, so
# the baked build matches the runtime fetch rather than being a throwaway artifact overwritten at launch.
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl}"
TVM_FFI_SPEC="${TVM_FFI_SPEC:-apache-tvm-ffi<0.1.12}"
NVIDIA_CUTLASS_DSL_SPEC="${NVIDIA_CUTLASS_DSL_SPEC:-nvidia-cutlass-dsl[cu13]==4.5.2}"
PREBUILT_WHEELS_DIR="${PREBUILT_WHEELS_DIR:-torch2.11+cu130}"
TRANSFORMER_ENGINE_WHEEL_REPO="nguyen599/prebuild-wheels-util"
TRANSFORMER_ENGINE_WHEEL_FILE="${TRANSFORMER_ENGINE_WHEEL_FILE:-${PREBUILT_WHEELS_DIR}/transformer_engine-2.17.0.dev0-cp312-cp312-linux_x86_64.whl}"
APEX_PREBUILT_WHEEL_FILE="${APEX_PREBUILT_WHEEL_FILE:-${PREBUILT_WHEELS_DIR}/apex-0.1-cp312-cp312-linux_x86_64.whl}"
CAUSAL_CONV1D_PREBUILT_WHEEL_FILE="${CAUSAL_CONV1D_PREBUILT_WHEEL_FILE:-${PREBUILT_WHEELS_DIR}/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl}"
MAMBA_SSM_PREBUILT_WHEEL_FILE="${MAMBA_SSM_PREBUILT_WHEEL_FILE:-${PREBUILT_WHEELS_DIR}/mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl}"
INSTALL_COMPILED_MODAL_KERNELS="${INSTALL_COMPILED_MODAL_KERNELS:-1}"
INSTALL_APEX="${INSTALL_APEX:-1}"
INSTALL_FLASH_ATTN_4="${INSTALL_FLASH_ATTN_4:-1}"
INSTALL_TRANSFORMER_ENGINE="${INSTALL_TRANSFORMER_ENGINE:-1}"
INSTALL_MEGATRON_CORE="${INSTALL_MEGATRON_CORE:-1}"
INSTALL_LIGER_KERNEL="${INSTALL_LIGER_KERNEL:-1}"
INSTALL_VERL_PACKAGE="${INSTALL_VERL_PACKAGE:-1}"
INSTALL_PRIME_RL_DEPS="${INSTALL_PRIME_RL_DEPS:-1}"
MEGATRON_CORE_REPO="${MEGATRON_CORE_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_CORE_REF="${MEGATRON_CORE_REF:-main}"
MEGATRON_CORE_DIR="${MEGATRON_CORE_DIR:-/opt/Megatron-LM}"
LIGER_KERNEL_REPO="${LIGER_KERNEL_REPO:-https://github.com/linkedin/Liger-Kernel.git}"
LIGER_KERNEL_REF="${LIGER_KERNEL_REF:-main}"
LIGER_KERNEL_DIR="${LIGER_KERNEL_DIR:-/opt/Liger-Kernel}"
VERL_PACKAGE="${VERL_PACKAGE:-git+https://github.com/verl-project/verl.git}"
TORCHTITAN_REQUIREMENT="${TORCHTITAN_REQUIREMENT:-torchtitan @ git+https://github.com/pytorch/torchtitan.git@23e4dfc}"
DION_REQUIREMENT="${DION_REQUIREMENT:-dion @ git+https://github.com/samsja/dion.git@d891eeb}"
DEEP_EP_REQUIREMENT="${DEEP_EP_REQUIREMENT:-deep-ep @ https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.5.0/deep_ep-1.2.1+29d31c0-cp312-cp312-linux_x86_64.whl}"
VERIFY_MAMBA_SSM_IMPORT="${VERIFY_MAMBA_SSM_IMPORT:-0}"
VERIFY_TRANSFORMER_ENGINE_IMPORT="${VERIFY_TRANSFORMER_ENGINE_IMPORT:-0}"
INSTALL_FULL_SYSTEM_APT="${INSTALL_FULL_SYSTEM_APT:-1}"
INSTALL_CONTAINER_BUILD_APT="${INSTALL_CONTAINER_BUILD_APT:-1}"
INSTALL_ROOTLESS_PROOT_WORKAROUNDS="${INSTALL_ROOTLESS_PROOT_WORKAROUNDS:-0}"
MODAL_EXTRAS_CACHE="${MODAL_EXTRAS_CACHE:-/root}"
FLASH_ATTN_WHEEL_NAME="flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_3_WHEEL_NAME="flash_attn_3-3.0.0+20260511.cu130torch2110cxx11abitrue.ab6632-cp39-abi3-linux_x86_64.whl"
FLASH_ATTN_4_WHEEL_NAME="flash_attn_4-4.0.0b15-py3-none-any.whl"
FLASH_ATTN_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/${FLASH_ATTN_WHEEL_NAME}"
FLASH_ATTN_3_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/${FLASH_ATTN_3_WHEEL_NAME}"
FLASH_ATTN_4_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/${FLASH_ATTN_4_WHEEL_NAME}"
TRANSFORMER_ENGINE_WHEEL_NAME="$(basename "${TRANSFORMER_ENGINE_WHEEL_FILE}")"
TRANSFORMER_ENGINE_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/${TRANSFORMER_ENGINE_WHEEL_NAME}"
APEX_PREBUILT_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/$(basename "${APEX_PREBUILT_WHEEL_FILE}")"
CAUSAL_CONV1D_PREBUILT_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/$(basename "${CAUSAL_CONV1D_PREBUILT_WHEEL_FILE}")"
MAMBA_SSM_PREBUILT_WHEEL_PATH="${MODAL_EXTRAS_CACHE}/$(basename "${MAMBA_SSM_PREBUILT_WHEEL_FILE}")"

mkdir -p "${MODAL_EXTRAS_CACHE}"

python_module_ok() {
    local module_name="$1"
    python - "${module_name}" >/dev/null 2>&1 <<'PY'
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
}

python_dist_ok() {
    local dist_name="$1"
    local expected_version="${2:-}"
    python - "${dist_name}" "${expected_version}" >/dev/null 2>&1 <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

dist_name = sys.argv[1]
expected_version = sys.argv[2]
try:
    installed_version = version(dist_name)
except PackageNotFoundError:
    raise SystemExit(1)
if expected_version and installed_version != expected_version:
    raise SystemExit(1)
PY
}

download_if_missing() {
    local output_path="$1"
    local url="$2"
    if [ -s "${output_path}" ]; then
        echo "Reusing cached download: ${output_path}"
        return 0
    fi
    wget -q -O "${output_path}" "${url}"
}

ensure_hf_cli() {
    if ! command -v hf >/dev/null 2>&1; then
        uv pip install --system --no-cache-dir "huggingface_hub[cli]>=1.0.0"
    fi
}

try_download_hf_wheel() {
    local repo="$1"
    local repo_path="$2"
    local output_path="$3"
    local download_status
    local tmp_dir
    local hf_token_args=()

    if [ -s "${output_path}" ]; then
        echo "Reusing cached HF wheel: ${output_path}"
        return 0
    fi

    set +x
    ensure_hf_cli
    if [ -n "${HF_TOKEN:-}" ]; then
        hf_token_args=(--token "${HF_TOKEN}")
    fi
    tmp_dir="$(mktemp -d "${MODAL_EXTRAS_CACHE}/hf-wheel-download.XXXXXX")"
    if HF_XET_HIGH_PERFORMANCE=1 hf download \
        --repo-type dataset \
        --local-dir "${tmp_dir}" \
        "${hf_token_args[@]}" \
        "${repo}" \
        "${repo_path}"; then
        download_status=0
    else
        download_status="$?"
    fi
    set -x
    if [ "${download_status}" -eq 0 ]; then
        mv "${tmp_dir}/${repo_path}" "${output_path}"
        rm -rf "${tmp_dir}"
        echo "Downloaded HF wheel: ${repo}/${repo_path} -> ${output_path}"
        return 0
    fi

    rm -rf "${tmp_dir}"
    echo "HF wheel not available, will fall back if supported: ${repo}/${repo_path}" >&2
    return 1
}

find_cached_wheel() {
    local pattern="$1"
    find "${MODAL_EXTRAS_CACHE}" -maxdepth 1 -name "${pattern}" -size +0c -print -quit
}

require_cached_wheel() {
    local pattern="$1"
    local wheel_path
    wheel_path="$(find_cached_wheel "${pattern}")"
    if [ -z "${wheel_path}" ]; then
        echo "Expected cached wheel matching ${pattern} in ${MODAL_EXTRAS_CACHE}, but none was found." >&2
        return 1
    fi
    printf '%s\n' "${wheel_path}"
}

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --no-cache-dir --upgrade uv
fi

# PyTorch's CUDA 13 image uses dist-packages; several wheel build scripts look
# for site-packages explicitly.
ln -sfn /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/site-packages || true

apt_packages=(
    apt-utils \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++-11 \
    gcc-11 \
    git \
    ibverbs-providers \
    ibverbs-utils \
    libaio-dev \
    libibverbs-dev \
    libnuma-dev \
    librdmacm-dev \
    libtool \
    make \
    ninja-build \
    p7zip-full \
    patchelf \
    rdma-core \
    pkg-config \
    wget \
    zlib1g-dev
)

# These match the modal_simp system dependency layer. Rootless proot builds can
# override these to 0, but normal sudo/fakeroot builds should keep them enabled.
if [ "${INSTALL_FULL_SYSTEM_APT}" = "1" ]; then
    apt_packages+=(
        libopenmpi-dev
        software-properties-common
    )
fi

if [ "${INSTALL_CONTAINER_BUILD_APT}" = "1" ]; then
    apt_packages+=(
        cryptsetup
        fuse2fs
        fuse3
        libfuse3-dev
        libseccomp-dev
        libsubid-dev
        libsubid4
        libudev-dev
        runc
        squashfs-tools
        squashfs-tools-ng
        uidmap
    )
fi

# Rootless proot builds can install files as root but package postinst
# `groupadd` calls may fail to rewrite /etc/group. Keep that workaround
# disabled for normal privileged builds.
if [ "${INSTALL_ROOTLESS_PROOT_WORKAROUNDS}" = "1" ]; then
    if ! getent group rdma >/dev/null 2>&1; then
        echo "rdma:x:103:" >> /etc/group
        if [ -f /etc/gshadow ]; then
            echo "rdma:!::" >> /etc/gshadow
        fi
    fi
fi

apt-get update
apt-get install -y --no-install-recommends "${apt_packages[@]}"

if [ ! -f /etc/apt/preferences.d/cuda-repository-pin-600 ]; then
    cd /root
    wget -q -O cuda-keyring_1.1-1_all.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
fi

apt-get install -y --no-install-recommends --allow-change-held-packages \
    cuda-toolkit=13.0.2-1 \
    cublasmp-cuda-13=0.8.0.2023-1 \
    libcudnn9-cuda-13=9.19.0.56-1 \
    libcudnn9-dev-cuda-13=9.19.0.56-1 \
    libcudnn9-headers-cuda-13=9.19.0.56-1 \
    libnccl2=2.29.3-1+cuda13.1 \
    libnccl-dev=2.29.3-1+cuda13.1 \
    nvshmem-cuda-13

rm -rf /var/lib/apt/lists/*

uv pip install --system --upgrade --no-cache-dir \
    packaging \
    setuptools==80.0 \
    wheel \
    wheel_stub

if python_module_ok ring_flash_attn; then
    echo "Skipping ring-flash-attention install; ring_flash_attn is already importable."
else
    uv pip install --system --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/nguyen599/ring-flash-attention.git"
fi

if python_dist_ok vllm; then
    echo "Skipping vLLM wheel install; vllm is already installed."
else
    uv pip install --system --no-cache-dir --no-build-isolation \
        "${VLLM_WHEEL_URL}"
fi

uv pip install --system --no-cache-dir \
    "cuda-python[all]==${CUDA_VERSION}" \
    "cuda-toolkit[all]==${CUDA_VERSION}" \
    nvidia-cublasmp-cu13==0.8.0.2023 \
    nvidia-cuda-cccl \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-frontend==1.24.0 \
    "${NVIDIA_CUTLASS_DSL_SPEC}" \
    nvidia-ml-py \
    nvshmem4py-cu13

# TileLang currently pulls an apache-tvm-ffi build that can collide with
# mamba-ssm/ModelOpt imports at runtime. Keep the older FFI ABI pinned until
# TileLang and TVM FFI converge again.
uv pip install --system --no-cache-dir --force-reinstall "${TVM_FFI_SPEC}"

if python_module_ok flashinfer; then
    echo "Skipping FlashInfer install; flashinfer is already importable."
else
    uv pip install --system --no-cache-dir \
        --extra-index-url "${FLASHINFER_INDEX_URL}" \
        flashinfer-python==0.6.12 \
        flashinfer-cubin==0.6.12 \
        lmcache \
        mooncake-transfer-engine
fi

cd /root
download_if_missing "${FLASH_ATTN_WHEEL_PATH}" "${FLASH_ATTN_WHEEL_URL}"
download_if_missing "${FLASH_ATTN_3_WHEEL_PATH}" "${FLASH_ATTN_3_WHEEL_URL}"
if [ "${INSTALL_FLASH_ATTN_4}" = "1" ]; then
    download_if_missing "${FLASH_ATTN_4_WHEEL_PATH}" "${FLASH_ATTN_4_WHEEL_URL}"
fi
if python_module_ok flash_attn; then
    echo "Skipping FlashAttention 2 install; flash_attn is already importable."
else
    uv pip install --system --no-cache-dir --no-deps \
        "${FLASH_ATTN_WHEEL_PATH}"
fi
if python_dist_ok flash-attn-3; then
    echo "Skipping FlashAttention 3 install; flash-attn-3 is already installed."
else
    uv pip install --system --no-cache-dir --no-deps \
        "${FLASH_ATTN_3_WHEEL_PATH}"
fi
if [ "${INSTALL_FLASH_ATTN_4}" = "1" ]; then
    if python_module_ok flash_attn.cute; then
        echo "Skipping FlashAttention 4 install; flash_attn.cute is already importable."
    else
        uv pip install --system --no-cache-dir --no-deps \
            "${FLASH_ATTN_4_WHEEL_PATH}"
    fi
else
    echo "Skipping FlashAttention 4 install because INSTALL_FLASH_ATTN_4=${INSTALL_FLASH_ATTN_4}."
fi

if [ "${INSTALL_TRANSFORMER_ENGINE}" = "1" ]; then
    if python_module_ok transformer_engine.pytorch; then
        echo "Skipping Transformer Engine install; transformer_engine.pytorch is already importable."
    else
        try_download_hf_wheel \
            "${TRANSFORMER_ENGINE_WHEEL_REPO}" \
            "${TRANSFORMER_ENGINE_WHEEL_FILE}" \
            "${TRANSFORMER_ENGINE_WHEEL_PATH}"
        uv pip install --system --compile-bytecode --no-cache-dir \
            "${TRANSFORMER_ENGINE_WHEEL_PATH}"
    fi
else
    echo "Skipping Transformer Engine install because INSTALL_TRANSFORMER_ENGINE=${INSTALL_TRANSFORMER_ENGINE}."
fi

if [ "${INSTALL_COMPILED_MODAL_KERNELS}" = "1" ]; then
    if [ "${INSTALL_APEX}" = "1" ]; then
        cd /root
        try_download_hf_wheel \
            "nguyen599/prebuild-wheels-util" \
            "${APEX_PREBUILT_WHEEL_FILE}" \
            "${APEX_PREBUILT_WHEEL_PATH}" || true
        APEX_WHEEL="$(find_cached_wheel 'apex-*.whl')"
        if [ -z "${APEX_WHEEL}" ]; then
            if [ ! -d apex ]; then
                git clone https://github.com/NVIDIA/apex
            fi
            cd /root/apex
            CXX=g++ \
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
            NVCC_APPEND_FLAGS="--threads 2" \
            APEX_DISTRIBUTED_ADAM=1 \
            APEX_PARALLEL_BUILD="2" \
            APEX_CPP_EXT=1 \
            APEX_CUDA_EXT=1 \
            pip wheel --wheel-dir="${MODAL_EXTRAS_CACHE}" -v --no-build-isolation --no-cache-dir \
                --config-settings="--build-option=--cpp_ext" \
                --config-settings="--build-option=--cuda_ext" .
            APEX_WHEEL="$(require_cached_wheel 'apex-0.1-*.whl')"
        else
            echo "Reusing cached Apex wheel: ${APEX_WHEEL}"
        fi
        if python_module_ok apex; then
            echo "Skipping Apex install; apex is already importable."
        else
            uv pip install --system --no-cache-dir --no-build-isolation \
                "${APEX_WHEEL}"
        fi
        python - <<'PY'
import apex
from apex.optimizers import FusedAdam

print("Apex import OK", apex.__file__)
print("Apex FusedAdam import OK", FusedAdam)
PY
    else
        echo "Skipping Apex build/install because INSTALL_APEX=${INSTALL_APEX}."
    fi

    if python_dist_ok nv-grouped-gemm; then
        echo "Skipping nv-grouped-gemm install; distribution is already installed."
    else
        TORCH_CUDA_ARCH_LIST="${NV_GROUPED_GEMM_TORCH_CUDA_ARCH_LIST}" \
        uv pip install --system --compile-bytecode --no-cache-dir \
            --no-build-isolation --no-deps nv-grouped-gemm~=1.1
    fi

    cd /root
    try_download_hf_wheel \
        "nguyen599/prebuild-wheels-util" \
        "${CAUSAL_CONV1D_PREBUILT_WHEEL_FILE}" \
        "${CAUSAL_CONV1D_PREBUILT_WHEEL_PATH}" || true
    CAUSAL_CONV1D_WHEEL="$(find_cached_wheel 'causal_conv1d-*.whl')"
    if [ -z "${CAUSAL_CONV1D_WHEEL}" ]; then
        if [ ! -d causal-conv1d ]; then
            git clone https://github.com/nguyen599/causal-conv1d.git
        fi
        cd /root/causal-conv1d
        BUILD_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
            pip wheel --wheel-dir="${MODAL_EXTRAS_CACHE}" --no-build-isolation --no-deps .
        CAUSAL_CONV1D_WHEEL="$(require_cached_wheel 'causal_conv1d-*.whl')"
    else
        echo "Reusing cached causal-conv1d wheel: ${CAUSAL_CONV1D_WHEEL}"
    fi
    try_download_hf_wheel \
        "nguyen599/prebuild-wheels-util" \
        "${MAMBA_SSM_PREBUILT_WHEEL_FILE}" \
        "${MAMBA_SSM_PREBUILT_WHEEL_PATH}" || true
    MAMBA_SSM_WHEEL="$(find_cached_wheel 'mamba_ssm-*.whl')"
    if [ -z "${MAMBA_SSM_WHEEL}" ]; then
        if [ ! -d mamba ]; then
            git clone https://github.com/nguyen599/mamba.git
        fi
        cd /root/mamba
        BUILD_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
            pip wheel --wheel-dir="${MODAL_EXTRAS_CACHE}" --no-build-isolation --no-deps .
        MAMBA_SSM_WHEEL="$(require_cached_wheel 'mamba_ssm-*.whl')"
    else
        echo "Reusing cached mamba-ssm wheel: ${MAMBA_SSM_WHEEL}"
    fi
    if python_module_ok causal_conv1d; then
        echo "Skipping causal-conv1d install; causal_conv1d is already importable."
    else
        uv pip install --system --compile-bytecode --no-cache-dir --no-deps "${CAUSAL_CONV1D_WHEEL}"
    fi
    if python_module_ok mamba_ssm; then
        echo "Skipping mamba-ssm install; mamba_ssm is already importable."
    else
        uv pip install --system --compile-bytecode --no-cache-dir --no-deps "${MAMBA_SSM_WHEEL}"
    fi
fi

uv pip install --system --no-cache-dir \
    hatchling \
    editables \
    poetry_dynamic_versioning \
    poetry \
    grpcio-tools

if [ "${INSTALL_VERL_PACKAGE}" = "1" ]; then
    uv pip install --system --no-cache-dir "${VERL_PACKAGE}"
else
    echo "Skipping VERL package install because INSTALL_VERL_PACKAGE=${INSTALL_VERL_PACKAGE}."
fi

if [ "${INSTALL_PRIME_RL_DEPS}" = "1" ]; then
    uv pip install --system --no-cache-dir \
        "${TORCHTITAN_REQUIREMENT}" \
        "${DION_REQUIREMENT}" \
        "${DEEP_EP_REQUIREMENT}"
else
    echo "Skipping Prime-RL dependency install because INSTALL_PRIME_RL_DEPS=${INSTALL_PRIME_RL_DEPS}."
fi

if python_dist_ok nvidia-resiliency-ext; then
    echo "Skipping nvidia-resiliency-ext install; distribution is already installed."
else
    uv pip install --system --no-cache-dir --no-build-isolation \
        "git+https://github.com/NVIDIA/nvidia-resiliency-ext.git"
fi

if [ "${INSTALL_MEGATRON_CORE}" = "1" ]; then
    if [ ! -d "${MEGATRON_CORE_DIR}/.git" ]; then
        rm -rf "${MEGATRON_CORE_DIR}"
        git clone --filter=blob:none --no-checkout "${MEGATRON_CORE_REPO}" "${MEGATRON_CORE_DIR}"
    fi
    git -C "${MEGATRON_CORE_DIR}" remote set-url origin "${MEGATRON_CORE_REPO}"
    git -C "${MEGATRON_CORE_DIR}" fetch --depth 1 origin "${MEGATRON_CORE_REF}"
    git -C "${MEGATRON_CORE_DIR}" checkout --force FETCH_HEAD
    uv pip install --system --no-cache-dir --no-deps "${MEGATRON_CORE_DIR}"
    PYTHONPATH="${MEGATRON_CORE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
print("Skip Megatron check")
PY
else
    echo "Skipping Megatron Core install because INSTALL_MEGATRON_CORE=${INSTALL_MEGATRON_CORE}."
fi

if [ "${INSTALL_LIGER_KERNEL}" = "1" ]; then
    if [ ! -d "${LIGER_KERNEL_DIR}/.git" ]; then
        rm -rf "${LIGER_KERNEL_DIR}"
        git clone --filter=blob:none --no-checkout "${LIGER_KERNEL_REPO}" "${LIGER_KERNEL_DIR}"
    fi
    git -C "${LIGER_KERNEL_DIR}" remote set-url origin "${LIGER_KERNEL_REPO}"
    git -C "${LIGER_KERNEL_DIR}" fetch --depth 1 origin "${LIGER_KERNEL_REF}"
    git -C "${LIGER_KERNEL_DIR}" checkout --force FETCH_HEAD
    uv pip install --system --no-cache-dir --no-deps "${LIGER_KERNEL_DIR}"

else
    echo "Skipping Liger Kernel install because INSTALL_LIGER_KERNEL=${INSTALL_LIGER_KERNEL}."
fi

# Keep the NCCL Python wheel aligned with the apt NCCL used by the tested
# modal_simp image and by the OLMo-core FP8/AdamW8bit probes.
if python_dist_ok nvidia-nccl-cu13 2.29.3; then
    echo "Skipping nvidia-nccl-cu13 install; version 2.29.3 is already installed."
else
    uv pip install --system --no-cache-dir --no-deps nvidia-nccl-cu13==2.29.3
fi

# Re-pin nvidia-cutlass-dsl to vLLM's own ==4.5.2 LAST. flashinfer / quack / flash-attn-4 pull it
# forward to 4.6.0 via `>=`, but 4.6.0 dropped `cute.core.ThrMma`, which breaks flash_attn.cute,
# QuACK, AND Transformer Engine at import. Re-pinning here (after every consumer is installed, and
# after the nccl re-pin) restores 4.5.2 and still satisfies all their `>=` bounds.
if python_dist_ok nvidia-cutlass-dsl 4.5.2; then
    echo "Skipping nvidia-cutlass-dsl re-pin; version 4.5.2 is already installed."
else
    uv pip install --system --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2"
fi

verify_modules=(flash_attn flashinfer vllm)
if [ "${INSTALL_FLASH_ATTN_4}" = "1" ]; then
    verify_modules+=(flash_attn.cute)
fi
if [ "${INSTALL_COMPILED_MODAL_KERNELS}" = "1" ]; then
    verify_modules+=(causal_conv1d)
    if [ "${VERIFY_MAMBA_SSM_IMPORT}" = "1" ]; then
        verify_modules+=(mamba_ssm)
    else
        echo "Skipping mamba_ssm import check because VERIFY_MAMBA_SSM_IMPORT=${VERIFY_MAMBA_SSM_IMPORT}."
    fi
    if [ "${INSTALL_APEX}" = "1" ]; then
        verify_modules+=(apex)
    fi
fi
if [ "${INSTALL_TRANSFORMER_ENGINE}" = "1" ]; then
    if [ "${VERIFY_TRANSFORMER_ENGINE_IMPORT}" = "1" ]; then
        verify_modules+=(transformer_engine.pytorch)
    else
        echo "Skipping Transformer Engine import check because it requires libcuda.so.1."
    fi
fi

python - <<'PY'
import torch

print("modal_simp extras OK")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY
