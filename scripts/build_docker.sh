#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=src/cuda_config.sh
source "${REPO_DIR}/src/cuda_config.sh"

DOCKERFILE="${DOCKERFILE:-${REPO_DIR}/Dockerfile}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.11.0-cuda${CUDA_BASE_IMAGE_VERSION}-cudnn9-devel}"
IMAGE_TAG="${IMAGE_TAG:-aimo-proof-pilot:${CUDA_WHEEL_TAG}}"
VLLM_BUILD_FROM_SOURCE="${VLLM_BUILD_FROM_SOURCE:-0}"

echo "Building ${IMAGE_TAG} with CUDA ${CUDA_VERSION} from ${BASE_IMAGE}"
build_args=(
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "CUDA_BASE_IMAGE_VERSION=${CUDA_BASE_IMAGE_VERSION}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "VLLM_BUILD_FROM_SOURCE=${VLLM_BUILD_FROM_SOURCE}"
    -f "${DOCKERFILE}"
    -t "${IMAGE_TAG}"
)
if [ -n "${HF_TOKEN:-}" ]; then
    build_args+=(--secret id=hf_token,env=HF_TOKEN)
fi
if [ -n "${GITHUB_TOKEN:-}" ]; then
    build_args+=(--secret id=github_token,env=GITHUB_TOKEN)
fi

DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "$@" "${REPO_DIR}"
