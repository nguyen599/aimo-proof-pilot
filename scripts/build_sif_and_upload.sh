#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/opt/sif-build}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/out}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
CACHE_DIR="${CACHE_DIR:-${WORK_DIR}/cache}"
TMP_DIR="${TMP_DIR:-${WORK_DIR}/tmp}"

DEF_NAME="${DEF_NAME:-sft-phase1_train_20260608.def}"
SIF_NAME="${SIF_NAME:-sft-phase1_train_20260608.sif}"
SIF_PATH="${SIF_PATH:-${OUT_DIR}/${SIF_NAME}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${SIF_NAME%.sif}.build.log}"
BUILDER="${BUILDER:-singularity}"
VERIFY="${VERIFY:-1}"
UPLOAD="${UPLOAD:-1}"
HF_REPO_ID="${HF_REPO_ID:-nguyen599/sif-image}"
HF_REMOTE_PATH="${HF_REMOTE_PATH:-${SIF_NAME}}"
INSTALL_SINGULARITY="${INSTALL_SINGULARITY:-1}"
INSTALL_APPTAINER="${INSTALL_APPTAINER:-1}"
APPTAINER_VERSION="${APPTAINER_VERSION:-v1.5.0}"
GO_VERSION="${GO_VERSION:-1.26.3}"
APPTAINER_MAKE_JOBS="${APPTAINER_MAKE_JOBS:-2}"
BUILD_VERBOSE="${BUILD_VERBOSE:-1}"
BUILD_MONITOR_INTERVAL_SECONDS="${BUILD_MONITOR_INTERVAL_SECONDS:-60}"
CUDA_VERSION="${CUDA_VERSION:-}"
SIF_BASE_IMAGE="${SIF_BASE_IMAGE:-}"

APT_PACKAGES=(
    ca-certificates
    curl
    git
    python3-pip
    wget
)

APPTAINER_APT_PACKAGES=(
    autoconf
    automake
    build-essential
    cryptsetup
    fakeroot
    fuse2fs
    fuse3
    libfuse3-dev
    libseccomp-dev
    libtool
    pkg-config
    runc
    squashfs-tools
    squashfs-tools-ng
    uidmap
    zlib1g-dev
)

SINGULARITY_APT_PACKAGES=(
    autoconf
    automake
    build-essential
    cryptsetup
    dh-apparmor
    fakeroot
    fuse2fs
    fuse3
    libattr1-dev
    libfuse3-dev
    libprotobuf-c-dev
    libseccomp-dev
    libtalloc-dev
    libtool
    pkg-config
    runc
    squashfs-tools
    squashfs-tools-ng
    uidmap
    zlib1g-dev
)

OPTIONAL_APT_PACKAGES=(
    libsubid-dev
)

as_root() {
    if [ "$(id -u)" = "0" ]; then
        "$@"
    else
        sudo "$@"
    fi
}

install_host_deps() {
    export DEBIAN_FRONTEND=noninteractive
    local packages=("${APT_PACKAGES[@]}")
    local needs_apptainer_deps=0
    local needs_singularity_deps=0
    if [ "${INSTALL_APPTAINER}" = "1" ] && ! command -v apptainer >/dev/null 2>&1; then
        needs_apptainer_deps=1
        packages+=("${APPTAINER_APT_PACKAGES[@]}")
    fi
    if [ "${INSTALL_SINGULARITY}" = "1" ] && ! command -v singularity >/dev/null 2>&1; then
        needs_singularity_deps=1
        packages+=("${SINGULARITY_APT_PACKAGES[@]}")
    fi
    if [ "${#packages[@]}" -gt 0 ]; then
        mapfile -t packages < <(printf '%s\n' "${packages[@]}" | sort -u)
    fi
    as_root apt-get update
    as_root apt-get install -y --no-install-recommends "${packages[@]}"
    if [ "${needs_apptainer_deps}" = "1" ] || [ "${needs_singularity_deps}" = "1" ]; then
        for pkg in "${OPTIONAL_APT_PACKAGES[@]}"; do
            if apt-cache show "$pkg" >/dev/null 2>&1; then
                as_root apt-get install -y --no-install-recommends "$pkg"
            else
                echo "Optional apt package not available, skipping: $pkg"
            fi
        done
    fi
}

install_apptainer() {
    if command -v apptainer >/dev/null 2>&1; then
        apptainer version
        return
    fi
    mkdir -p "${WORK_DIR}/src"
    cd "${WORK_DIR}/src"
    if ! command -v go >/dev/null 2>&1; then
        wget -q -O "go${GO_VERSION}.linux-amd64.tar.gz" "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        as_root rm -rf /usr/local/go
        as_root tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
        as_root ln -sfn /usr/local/go/bin/go /usr/local/bin/go
    fi
    if [ ! -d apptainer/.git ]; then
        git clone https://github.com/apptainer/apptainer.git
    fi
    git -C apptainer fetch --depth 1 origin "${APPTAINER_VERSION}" || true
    git -C apptainer checkout "${APPTAINER_VERSION}"
    cd apptainer
    ./mconfig
    make -C builddir -j"${APPTAINER_MAKE_JOBS}"
    as_root make -C builddir install
    apptainer version
}

install_singularity() {
    if command -v singularity >/dev/null 2>&1; then
        singularity --version
        return
    fi
    SINGULARITY_BUILD_ROOT="${WORK_DIR}/src/singularity-ce" bash "${REPO_DIR}/src/install_singularity_ce.sh"
}

install_upload_deps() {
    python3 -m pip install --upgrade --no-cache-dir "huggingface_hub[hf_transfer]"
}

build_command_prefix() {
    local builder_cmd="$1"
    if [ "${BUILD_VERBOSE}" = "1" ] && "${builder_cmd}" --help 2>&1 | grep -q -- "--verbose"; then
        printf '%s\n' "${builder_cmd}" "--verbose"
    else
        printf '%s\n' "${builder_cmd}"
    fi
}

monitor_build_progress() {
    local build_pid="$1"
    local interval="${BUILD_MONITOR_INTERVAL_SECONDS}"
    [ "${interval}" -gt 0 ] || return 0
    while kill -0 "${build_pid}" 2>/dev/null; do
        sleep "${interval}" || true
        if ! kill -0 "${build_pid}" 2>/dev/null; then
            break
        fi
        echo "--- build still running: $(date -Is) ---"
        if [ -e "${SIF_PATH}" ]; then
            ls -lh "${SIF_PATH}" || true
            du -h "${SIF_PATH}" || true
        else
            echo "SIF file has not been created yet: ${SIF_PATH}"
        fi
        df -h "${OUT_DIR}" "${TMP_DIR}" || true
        pgrep -fa "mksquashfs|unsquashfs|${BUILDER}|apptainer|singularity" || true
    done
}

build_sif() {
    mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${CACHE_DIR}" "${TMP_DIR}"
    local builder_cmd="${BUILDER}"
    if [ "${BUILDER}" = "apptainer" ]; then
        builder_cmd="apptainer"
        export APPTAINER_CACHEDIR="${CACHE_DIR}/apptainer"
        export APPTAINER_TMPDIR="${TMP_DIR}/apptainer"
    else
        builder_cmd="singularity"
        export SINGULARITY_CACHEDIR="${CACHE_DIR}/singularity"
        export SINGULARITY_TMPDIR="${TMP_DIR}/singularity"
    fi
    mkdir -p "${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}" "${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"

    echo "Building ${SIF_PATH} with ${builder_cmd}"
    echo "Log: ${LOG_PATH}"
    mapfile -t builder_prefix < <(build_command_prefix "${builder_cmd}")
    (
        cd "${REPO_DIR}"
        build_args=(build --force)
        if [ -n "${CUDA_VERSION}" ]; then
            build_args+=(--build-arg "CUDA_VERSION=${CUDA_VERSION}")
        fi
        if [ -n "${SIF_BASE_IMAGE}" ]; then
            build_args+=(--build-arg "BASE_IMAGE=${SIF_BASE_IMAGE}")
        fi
        build_args+=("${SIF_PATH}" "${DEF_NAME}")
        "${builder_prefix[@]}" "${build_args[@]}"
    ) > >(tee "${LOG_PATH}") 2>&1 &
    local build_pid="$!"
    monitor_build_progress "${build_pid}" &
    local monitor_pid="$!"
    local build_status=0
    wait "${build_pid}" || build_status="$?"
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    if [ "${build_status}" -ne 0 ]; then
        return "${build_status}"
    fi
    sha256sum "${SIF_PATH}" | tee "${SIF_PATH}.sha256"
}

verify_sif() {
    [ "${VERIFY}" = "1" ] || return 0
    local runner="${BUILDER}"
    local module
    if [ "${BUILDER}" != "apptainer" ]; then
        runner="singularity"
    fi

    # Transformer Engine is installed but requires libcuda.so.1, which is not available on CPU builders.
    for module in flash_attn flash_attn.cute flashinfer open_instruct olmo_core ring_flash_attn torchao; do
        HOME="${TMP_DIR}/home" \
        XDG_CACHE_HOME="${TMP_DIR}/xdg" \
        FLASHINFER_CACHE_DIR="${TMP_DIR}/flashinfer" \
        PYTHONDONTWRITEBYTECODE=1 \
        "${runner}" exec "${SIF_PATH}" python -B - "${module}" <<'PY'
import importlib
import sys

module = sys.argv[1]
importlib.import_module(module)
print(f"{module} import OK")
PY
    done

    HOME="${TMP_DIR}/home" \
    XDG_CACHE_HOME="${TMP_DIR}/xdg" \
    FLASHINFER_CACHE_DIR="${TMP_DIR}/flashinfer" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${runner}" exec "${SIF_PATH}" python -B - <<'PY'
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
    "${runner}" run "${SIF_PATH}" launcher-dryrun --no-fetch-update
}

upload_sif() {
    [ "${UPLOAD}" = "1" ] || return 0
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "HF_TOKEN is required when UPLOAD=1" >&2
        return 1
    fi
    install_upload_deps
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python3 - "${SIF_PATH}" "${HF_REPO_ID}" "${HF_REMOTE_PATH}" <<'PY'
import os
import sys
from huggingface_hub import HfApi, create_repo

path, repo_id, remote_path = sys.argv[1:4]
token = os.environ["HF_TOKEN"]
create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=path,
    path_in_repo=remote_path,
    repo_id=repo_id,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj=f"{path}.sha256",
    path_in_repo=f"{remote_path}.sha256",
    repo_id=repo_id,
    repo_type="dataset",
)
print(f"uploaded {path} to dataset {repo_id}:{remote_path}")
PY
}

main() {
    install_host_deps
    if [ "${INSTALL_SINGULARITY}" = "1" ]; then
        install_singularity
    fi
    if [ "${INSTALL_APPTAINER}" = "1" ]; then
        install_apptainer
    fi
    build_sif
    verify_sif
    upload_sif
}

main "$@"
