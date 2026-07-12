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

# --- Remote-shell daemon as the default entrypoint (crash-resilient supervisor). Training runs
# THROUGH the relay shell sessions the daemon exposes (open a shell, then run /app/train.py ...),
# instead of train.py being PID 1. The daemon is outbound-only HTTPS to a private HF Space relay
# (NII-approved). Config — HF_TOKEN, RELAY_SPACE, CLIENT_ID — is provided at RUNTIME; no secrets baked.
COPY remote-shell/daemon /app/remote-shell/daemon
# Install daemon dependencies into the same Python used by the entrypoint.
# Avoid a separate /opt/venv-daemon path because some container runtimes mount it without execute
# permission, which prevents the relay daemon from starting.
RUN uv pip install --system --no-cache-dir -r /app/remote-shell/daemon/requirements.txt
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
# PID 1. The container runs INDEFINITELY — only an external SIGKILL / teardown stops it. The
# remote-shell daemon is restarted on any crash/exit. A LOUD banner is printed to the container's
# STDOUT (what the launcher sees via `docker logs` / the terminal), and the daemon's own output is
# tee'd to STDOUT too, so it's obvious the daemon is up.
set +e
trap '' TERM INT HUP PIPE QUIT      # ignore graceful signals; only SIGKILL ends this loop
LOGS=/tmp/imochallenge/logs; LOG="$LOGS/relay-daemon.log"
mkdir -p "$LOGS" 2>/dev/null
announce() {
    printf '\n============================================================\n'
    printf '  REMOTE-SHELL DAEMON %s\n' "$1"
    printf '  client_id = %s\n' "${CLIENT_ID:-$(hostname 2>/dev/null)}"
    printf '  container runs until killed  |  logs: %s\n' "$LOG"
    printf '============================================================\n\n'
}
announce "STARTING"
# One-shot preflight smoke test (env/system checks: distributed env vars, CUDA driver, /tmp+$HOME
# writable, creds, disk). Its PASS/WARN/FAIL report goes to STDOUT + the daemon log, so it's visible
# via `docker logs` / the control panel at boot. It does NOT gate the daemon — the daemon must stay up
# for remote control even if a check FAILs (so you can fix + relaunch remotely).
if [ -f /app/smoke_test_opd.py ]; then
    /usr/bin/python /app/smoke_test_opd.py 2>&1 | tee -a "$LOG" || true
fi
while :; do
    announce "RUNNING (daemon (re)starting)"
    python /app/remote-shell/daemon/client.py 2>&1 | tee -a "$LOG"
    printf '[supervisor %s] daemon exited — restart in 5s\n' "$(date -u '+%F %T' 2>/dev/null)" | tee -a "$LOG"
    sleep 5 2>/dev/null || true
done
EOF
RUN chmod +x /app/entrypoint.sh

# opd-run / opd-status: launch training FULLY DETACHED so it survives daemon crashes/restarts and
# the shell closing, and stays identifiable + monitorable.
COPY <<'EOF' /usr/local/bin/opd-run
#!/bin/bash
# opd-run <name> <cmd...> — setsid+nohup a command -> /tmp/imochallenge/logs/<name>.log + <name>.pid
set -euo pipefail
[ "$#" -ge 2 ] || { echo "usage: opd-run <name> <command...>" >&2; exit 2; }
name=$1; shift
run=/tmp/imochallenge/run; logs=/tmp/imochallenge/logs; mkdir -p "$run" "$logs"
log="$logs/$name.log"; pidf="$run/$name.pid"
if [ -f "$pidf" ] && kill -0 "$(cat "$pidf" 2>/dev/null || echo -1)" 2>/dev/null; then
    echo "opd-run: '$name' already running (pid $(cat "$pidf"))" >&2; exit 1
fi
setsid bash -c 'echo "[opd-run start $(date -u)] $*"; "$@"; echo "[opd-run exit $? $(date -u)]"' _ "$@" >>"$log" 2>&1 &
echo $! >"$pidf"
echo "opd-run: '$name' started (pid $(cat "$pidf"))  log=$log"
EOF
COPY <<'EOF' /usr/local/bin/opd-status
#!/bin/bash
# opd-status [name] — list detached opd-run jobs (alive/dead + last log lines)
run=/tmp/imochallenge/run; logs=/tmp/imochallenge/logs; shopt -s nullglob
if [ "$#" -ge 1 ]; then set -- "$1"; else set --; for f in "$run"/*.pid; do set -- "$@" "$(basename "$f" .pid)"; done; fi
[ "$#" -gt 0 ] || { echo "no runs under $run"; exit 0; }
for n in "$@"; do
    pidf="$run/$n.pid"; log="$logs/$n.log"; pid=$(cat "$pidf" 2>/dev/null || echo '?')
    if [ "$pid" != '?' ] && kill -0 "$pid" 2>/dev/null; then st=RUNNING; else st=stopped; fi
    printf '== %-20s [%s] pid=%s\n' "$n" "$st" "$pid"
    [ -f "$log" ] && tail -n 3 "$log" | sed 's/^/   /'
done
EOF
RUN chmod +x /usr/local/bin/opd-run /usr/local/bin/opd-status

# Was: ENTRYPOINT ["python", "/app/train.py"]. train.py still runs — just from inside a relay shell,
# e.g.:  python /app/train.py --backend prime_rl --prime_algorithm opd ...
ENTRYPOINT ["/app/entrypoint.sh"]
