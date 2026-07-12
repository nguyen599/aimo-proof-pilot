# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG CUDA_BASE_IMAGE_VERSION=12.8
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda${CUDA_BASE_IMAGE_VERSION}-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CUDA_VERSION
ARG CUDA_BASE_IMAGE_VERSION
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
ARG VLLM_BUILD_FROM_SOURCE=0
ARG GITHUB_TOKEN=""
ARG HF_TOKEN=""

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_BASE_IMAGE_VERSION=${CUDA_BASE_IMAGE_VERSION} \
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

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

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
    CUDA_VERSION="${CUDA_VERSION}" \
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
    VLLM_BUILD_FROM_SOURCE="${VLLM_BUILD_FROM_SOURCE}" \
    bash /app/install_training_deps.sh; \
    git config --global --unset-all url."https://${GITHUB_TOKEN}@github.com/".insteadOf || true

RUN python -m py_compile /app/train.py /app/train_engine.py /app/train_engine_rl.py && \
    python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import torch

print("docker training image OK")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
expected_te_core = "transformer-engine-cu12" if torch.version.cuda.startswith("12.") else "transformer-engine-cu13"
unexpected_te_core = "transformer-engine-cu13" if expected_te_core.endswith("cu12") else "transformer-engine-cu12"
print(expected_te_core, version(expected_te_core))
try:
    version(unexpected_te_core)
except PackageNotFoundError:
    pass
else:
    raise RuntimeError(f"unexpected Transformer Engine core installed: {unexpected_te_core}")
PY

# Generic Docker GPU providers expose container port 22 but do not inject an SSH daemon into
# arbitrary images. Keep SSH key-only; the runtime entrypoint accepts provider key environment
# variables and falls back to the repository's docker_authorized_keys key.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-server && \
    mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    printf '%s\n' \
        'PermitRootLogin prohibit-password' \
        'PasswordAuthentication no' \
        'KbdInteractiveAuthentication no' \
        'PubkeyAuthentication yes' \
        > /etc/ssh/sshd_config.d/99-aimo-proof-pilot.conf && \
    rm -rf /var/lib/apt/lists/*

COPY --chmod=600 docker_authorized_keys /root/.ssh/authorized_keys
COPY --chmod=755 docker_ssh_entrypoint.sh /usr/local/bin/prime-template-entrypoint.sh
RUN ln -s /usr/local/bin/prime-template-entrypoint.sh /usr/local/bin/aimo-proof-pilot-entrypoint.sh
RUN ssh-keygen -A && sshd -t

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/prime-template-entrypoint.sh"]
