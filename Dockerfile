# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OPEN_INSTRUCT_REF=main
ARG OLMO_CORE_REF=main
ARG SUBMISSION_REPO=https://github.com/nguyen599/aimo-proof-pilot.git
ARG SUBMISSION_REF=main
ARG SUBMISSION_DIR=/opt/aimo-proof-pilot
ARG INSTALL_MODAL_SIMP_EXTRAS=1
ARG INSTALL_FULL_SYSTEM_APT=1
ARG INSTALL_CONTAINER_BUILD_APT=0
ARG INSTALL_SINGULARITY_CE=0
ARG INSTALL_APPTAINER_IN_IMAGE=0
ARG INSTALL_COMPILED_MODAL_KERNELS=1
ARG INSTALL_APEX=1
ARG INSTALL_FLASH_ATTN_4=1
ARG INSTALL_TRANSFORMER_ENGINE=1
ARG INSTALL_MEGATRON_CORE=1
ARG INSTALL_LIGER_KERNEL=1
ARG GITHUB_TOKEN=""
ARG HF_TOKEN=""

ENV DEBIAN_FRONTEND=noninteractive \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    SUBMISSIONS_REPO=${SUBMISSION_REPO} \
    SUBMISSIONS_REF=${SUBMISSION_REF} \
    SUBMISSIONS_RUNTIME_DIR=/tmp/aimo-proof-pilot-runtime \
    OPEN_INSTRUCT_DIR=/opt/open-instruct \
    OLMO_CORE_DIR=/opt/OLMo-core \
    MEGATRON_CORE_DIR=/opt/Megatron-LM \
    PYTHONPATH=/app:${SUBMISSION_DIR}/src:/opt/open-instruct:/opt/OLMo-core/src:/opt/Megatron-LM \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    XDG_CACHE_HOME=/cache/xdg \
    TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    FLASHINFER_CACHE_DIR=/cache/flashinfer \
    VLLM_CACHE_ROOT=/cache/vllm \
    WANDB_DIR=/cache/wandb \
    WANDB_CACHE_DIR=/cache/wandb_cache \
    WANDB_CONFIG_DIR=/cache/wandb_config \
    USE_HUB_KERNELS=NO \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_DIST_INIT_BARRIER=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    NCCL_DEBUG=WARN \
    NCCL_NVLS_ENABLE=1 \
    HTTPX_LOG_LEVEL=WARNING \
    OMP_NUM_THREADS=8 \
    HF_XET_HIGH_PERFORMANCE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu

WORKDIR /opt

RUN --mount=type=secret,id=github_token,required=false \
    set -euo pipefail; \
    export GITHUB_TOKEN="${GITHUB_TOKEN:-$(cat /run/secrets/github_token 2>/dev/null || true)}"; \
    if [ -n "${GITHUB_TOKEN}" ]; then \
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi; \
    git clone --filter=blob:none "${SUBMISSION_REPO}" "${SUBMISSION_DIR}"; \
    cd "${SUBMISSION_DIR}"; \
    git fetch --depth 1 origin "${SUBMISSION_REF}" || true; \
    git checkout --force "${SUBMISSION_REF}" || git checkout --force FETCH_HEAD; \
    mkdir -p /app; \
    cp -a "${SUBMISSION_DIR}/src/." /app/; \
    git config --global --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf || true

WORKDIR /app

RUN chmod +x /app/*.sh && \
    mkdir -p /cache/hf /cache/xdg /cache/torch /cache/torchinductor /cache/triton \
        /cache/flashinfer /cache/vllm /cache/wandb /cache/wandb_cache /cache/wandb_config && \
    chmod -R 777 /cache

RUN --mount=type=secret,id=github_token,required=false \
    --mount=type=secret,id=hf_token,required=false \
    set -euo pipefail; \
    export GITHUB_TOKEN="${GITHUB_TOKEN:-$(cat /run/secrets/github_token 2>/dev/null || true)}"; \
    export HF_TOKEN="${HF_TOKEN:-$(cat /run/secrets/hf_token 2>/dev/null || true)}"; \
    if [ -n "${GITHUB_TOKEN}" ]; then \
        git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi; \
    OPEN_INSTRUCT_REF="${OPEN_INSTRUCT_REF}" \
    OLMO_CORE_REF="${OLMO_CORE_REF}" \
    INSTALL_MODAL_SIMP_EXTRAS="${INSTALL_MODAL_SIMP_EXTRAS}" \
    INSTALL_FULL_SYSTEM_APT="${INSTALL_FULL_SYSTEM_APT}" \
    INSTALL_CONTAINER_BUILD_APT="${INSTALL_CONTAINER_BUILD_APT}" \
    INSTALL_SINGULARITY_CE="${INSTALL_SINGULARITY_CE}" \
    INSTALL_APPTAINER_IN_IMAGE="${INSTALL_APPTAINER_IN_IMAGE}" \
    INSTALL_COMPILED_MODAL_KERNELS="${INSTALL_COMPILED_MODAL_KERNELS}" \
    INSTALL_APEX="${INSTALL_APEX}" \
    INSTALL_FLASH_ATTN_4="${INSTALL_FLASH_ATTN_4}" \
    INSTALL_TRANSFORMER_ENGINE="${INSTALL_TRANSFORMER_ENGINE}" \
    INSTALL_MEGATRON_CORE="${INSTALL_MEGATRON_CORE}" \
    INSTALL_LIGER_KERNEL="${INSTALL_LIGER_KERNEL}" \
    bash /app/install_training_deps.sh; \
    git config --global --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf || true

RUN python -m py_compile /app/train.py /app/train_engine.py /app/train_engine_rl.py && \
    python - <<'PY'
import torch

print("docker training image OK")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

ENTRYPOINT ["python", "/app/train.py"]
CMD ["--help"]
