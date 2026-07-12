#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

ulimit -n "$(ulimit -Hn)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/opt/aimo-proof-pilot-venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
CACHE_ROOT="${CACHE_ROOT:-/alloc/aimo-proof-pilot-cache}"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3-pip

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --no-cache-dir --upgrade uv
fi

uv python install "${PYTHON_VERSION}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    mkdir -p "$(dirname "${VENV_DIR}")"
    uv venv --python "${PYTHON_VERSION}" --seed "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

mkdir -p \
    "${CACHE_ROOT}/hf" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/torchinductor" \
    "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/flashinfer" \
    "${CACHE_ROOT}/vllm" \
    "${CACHE_ROOT}/wandb" \
    "${CACHE_ROOT}/wheels"

export CUDA_HOME="/usr/local/cuda-12.8"
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export FLASHINFER_CACHE_DIR="${CACHE_ROOT}/flashinfer"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export HF_XET_HIGH_PERFORMANCE=1
export UV_BREAK_SYSTEM_PACKAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
export UV_EXTRA_INDEX_URL="${PYTORCH_INDEX_URL}"
export UV_INDEX_STRATEGY=unsafe-best-match
uv pip install --upgrade pip setuptools wheel packaging ninja
uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}"

cat >/etc/profile.d/aimo-proof-pilot.sh <<EOF
export VIRTUAL_ENV="${VENV_DIR}"
export CUDA_HOME="${CUDA_HOME}"
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDACXX}"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export PYTHONPATH="/opt/aimo-proof-pilot/src:/opt/open-instruct:/opt/OLMo-core/src:/opt/Megatron-LM\${PYTHONPATH:+:\${PYTHONPATH}}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"
export WANDB_DIR="${WANDB_DIR}"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
chmod 0644 /etc/profile.d/aimo-proof-pilot.sh
touch /root/.bashrc
if ! grep -qxF 'source /etc/profile.d/aimo-proof-pilot.sh' /root/.bashrc; then
    printf '%s\n' 'source /etc/profile.d/aimo-proof-pilot.sh' >> /root/.bashrc
fi

CUDA_VERSION="${CUDA_VERSION}" \
APP_DIR="${SCRIPT_DIR}" \
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt" \
INSTALL_MODAL_SIMP_EXTRAS=1 \
INSTALL_FULL_SYSTEM_APT=1 \
INSTALL_CONTAINER_BUILD_APT=0 \
INSTALL_SINGULARITY_CE=0 \
INSTALL_APPTAINER_IN_IMAGE=0 \
INSTALL_COMPILED_MODAL_KERNELS=1 \
INSTALL_APEX=1 \
INSTALL_FLASH_ATTN_4=1 \
INSTALL_TRANSFORMER_ENGINE=1 \
INSTALL_MEGATRON_CORE=1 \
INSTALL_LIGER_KERNEL=1 \
INSTALL_VERL_PACKAGE=1 \
INSTALL_PRIME_RL_DEPS=1 \
MODAL_EXTRAS_CACHE="${CACHE_ROOT}/wheels" \
VLLM_BUILD_FROM_SOURCE=0 \
bash "${SCRIPT_DIR}/install_training_deps.sh"

# Noninteractive SSH commands do not source /etc/profile.d on all Prime images.
# Expose only the training entry points through /usr/local/bin; Ubuntu services
# that use /usr/bin/python3 explicitly continue to use the system interpreter.
for command_name in \
    python python3 python3.12 pip pip3 pip3.12 \
    torchrun vllm hf wandb accelerate deepspeed ray; do
    if [ -x "${VENV_DIR}/bin/${command_name}" ]; then
        ln -sfn "${VENV_DIR}/bin/${command_name}" "/usr/local/bin/${command_name}"
    fi
done
ln -sfn "${CUDA_HOME}" /usr/local/cuda
ln -sfn "${CUDA_HOME}/bin/nvcc" /usr/local/bin/nvcc
printf '%s\n' \
    "${CUDA_HOME}/lib64" \
    /usr/local/nvidia/lib \
    /usr/local/nvidia/lib64 \
    >/etc/ld.so.conf.d/aimo-proof-pilot-cuda.conf
ldconfig

python - <<'PY'
import torch
import vllm
from apex.optimizers import FusedAdam
from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam

print("Prime node environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("apex.FusedAdam", FusedAdam)
print("transformer_engine.FusedAdam", TEFusedAdam)
PY
