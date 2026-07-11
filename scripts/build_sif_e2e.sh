#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_DIR}/.." && pwd)"

DEF_NAME="${DEF_NAME:-sft-phase1_train_20260608.def}"
SIF_NAME="${SIF_NAME:-sft-phase1_train_20260608.sudo.sif}"
OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/artifacts/singularity/out}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/artifacts/singularity/logs}"
CACHE_DIR="${CACHE_DIR:-${WORKSPACE_ROOT}/artifacts/singularity/cache-sudo}"
TMP_DIR="${TMP_DIR:-${WORKSPACE_ROOT}/artifacts/singularity/tmp-sudo}"
SIF_PATH="${SIF_PATH:-${OUT_DIR}/${SIF_NAME}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${SIF_NAME%.sif}.build.log}"
NOTEST="${NOTEST:-0}"
VERIFY="${VERIFY:-1}"
CUDA_VERSION="${CUDA_VERSION:-}"
SIF_BASE_IMAGE="${SIF_BASE_IMAGE:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}" "${TMP_DIR}"

build_args=(build --force)
if [ -n "${CUDA_VERSION}" ]; then
    build_args+=(--build-arg "CUDA_VERSION=${CUDA_VERSION}")
fi
if [ -n "${SIF_BASE_IMAGE}" ]; then
    build_args+=(--build-arg "BASE_IMAGE=${SIF_BASE_IMAGE}")
fi
if [ "${NOTEST}" = "1" ]; then
    build_args+=(--notest)
fi
build_args+=("${SIF_PATH}" "${DEF_NAME}")

echo "Building ${SIF_PATH}"
echo "Source: ${REPO_DIR}"
echo "Log: ${LOG_PATH}"

(
    cd "${REPO_DIR}"
    sudo -v
    sudo -E env \
        "GITHUB_TOKEN=${GITHUB_TOKEN:-}" \
        "SINGULARITY_CACHEDIR=${CACHE_DIR}" \
        "SINGULARITY_TMPDIR=${TMP_DIR}" \
        singularity "${build_args[@]}"
) 2>&1 | tee "${LOG_PATH}"

sha256sum "${SIF_PATH}" | tee "${SIF_PATH}.sha256"

if [ "${VERIFY}" = "1" ]; then
    VERIFY_ROOT="${WORKSPACE_ROOT}/artifacts/singularity/verify-cache"
    mkdir -p "${VERIFY_ROOT}/home" "${VERIFY_ROOT}/xdg" "${VERIFY_ROOT}/flashinfer"

    HOME="${VERIFY_ROOT}/home" \
    XDG_CACHE_HOME="${VERIFY_ROOT}/xdg" \
    FLASHINFER_CACHE_DIR="${VERIFY_ROOT}/flashinfer" \
    PYTHONDONTWRITEBYTECODE=1 \
    singularity exec "${SIF_PATH}" python -B - <<'PY'
import flash_attn
import flashinfer
import open_instruct
import olmo_core
import torch
import torchao
from torchao.optim import AdamW8bit

print("imports OK")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("torchao", torchao.__version__)
print("AdamW8bit", AdamW8bit.__name__)
PY

    GLOBAL_RANK=2 \
    WORLD_SIZE=3 \
    MASTER_ADDR=10.0.0.1 \
    MASTER_PORT=29500 \
    singularity run "${SIF_PATH}" launcher-dryrun
fi
