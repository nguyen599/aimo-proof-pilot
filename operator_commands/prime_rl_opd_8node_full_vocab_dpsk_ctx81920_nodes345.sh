#!/usr/bin/env bash
set -euo pipefail

# Standalone Prime-RL OPD run for one-node testing or the 8-node operator cluster.
# One-node layout (`PRIME_NODE_LAYOUT=1node`):
#   GPUs 0,1,2,3: student/policy vLLM rollout (TP=1, DP=4)
#   GPUs 4,5: trainer (FSDP world size 2)
#   GPUs 6,7: DeepSeek-V4-Flash teacher vLLM scorer (TP=2)
# Default full-cluster layout:
#   Nodes 0,1: distributed trainer (node 0 also runs the orchestrator), 16 GPUs
#   Nodes 2,3,4,5: student/policy vLLM rollout, 8 GPUs each
#   Node 6: idle/spare
#   Node 7: DeepSeek-V4-Flash teacher vLLM hidden-state scorer, 8 GPUs
# Requires Prime-RL's independent per-rank filesystem-reference padding and
# scalar full-vocab trainer-metric normalization fixes.
#
# To select another layout, set PRIME_NODE_LAYOUT=1node, 6node, or 3node:
#   1node: node 0 runs trainer, policy, teacher, and orchestrator locally.
#   6node: node 0 trainer, nodes 1,2,3,4 policy, node 5 teacher.
#   3node: node 0 trainer, node 1 policy, node 2 teacher.
# The explicit PRIME_TRAIN_NODES, PRIME_POLICY_NODES, and PRIME_TEACHER_NODE
# variables override both layouts. PRIME_TRAIN_NODE remains a backward-
# compatible alias for a single trainer node.

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
NODE_LAYOUT="${PRIME_NODE_LAYOUT:-8node}"
SINGLE_NODE_MODE=0

case "${NODE_LAYOUT}" in
  1node|single|local)
    SINGLE_NODE_MODE=1
    DEFAULT_TRAIN_NODE="${PRIME_SINGLE_NODE_LABEL:-0}"
    DEFAULT_POLICY_NODES="${DEFAULT_TRAIN_NODE}"
    DEFAULT_TEACHER_NODE="${DEFAULT_TRAIN_NODE}"
    DEFAULT_TRAIN_GPU_COUNT=2
    DEFAULT_POLICY_TP=1
    DEFAULT_POLICY_DP=4
    DEFAULT_TEACHER_TP=2
    DEFAULT_TEACHER_DP=1
    DEFAULT_TEACHER_GPU_IDS="6,7"
    ;;
  3node|345)
    DEFAULT_TRAIN_NODE="0"
    DEFAULT_POLICY_NODES="1"
    DEFAULT_TEACHER_NODE="2"
    DEFAULT_TRAIN_GPU_COUNT=8
    DEFAULT_POLICY_TP=1
    DEFAULT_POLICY_DP=8
    DEFAULT_TEACHER_TP=8
    DEFAULT_TEACHER_DP=1
    DEFAULT_TEACHER_GPU_IDS="0,1,2,3,4,5,6,7"
    ;;
  6node|full)
    DEFAULT_TRAIN_NODE="0"
    DEFAULT_POLICY_NODES="1,2,3,4"
    DEFAULT_TEACHER_NODE="5"
    DEFAULT_TRAIN_GPU_COUNT=8
    DEFAULT_POLICY_TP=1
    DEFAULT_POLICY_DP=8
    DEFAULT_TEACHER_TP=8
    DEFAULT_TEACHER_DP=1
    DEFAULT_TEACHER_GPU_IDS="0,1,2,3,4,5,6,7"
    ;;
  8node|full8)
    DEFAULT_TRAIN_NODE="0,1"
    DEFAULT_POLICY_NODES="2,3,4,5"
    DEFAULT_TEACHER_NODE="7"
    DEFAULT_TRAIN_GPU_COUNT=8
    DEFAULT_POLICY_TP=1
    DEFAULT_POLICY_DP=8
    DEFAULT_TEACHER_TP=8
    DEFAULT_TEACHER_DP=1
    DEFAULT_TEACHER_GPU_IDS="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "[prime-opd] invalid PRIME_NODE_LAYOUT=${NODE_LAYOUT}; expected 1node, 8node, 6node, or 3node" >&2
    exit 1
    ;;
esac

if (( SINGLE_NODE_MODE == 1 )) && [[ "${NODE_LABEL}" == "none" ]]; then
  NODE_LABEL="${DEFAULT_TRAIN_NODE}"
fi

TRAIN_NODES="${PRIME_TRAIN_NODES:-${PRIME_TRAIN_NODE:-${DEFAULT_TRAIN_NODE}}}"
POLICY_NODES="${PRIME_POLICY_NODES:-${DEFAULT_POLICY_NODES}}"
TEACHER_NODE="${PRIME_TEACHER_NODE:-${DEFAULT_TEACHER_NODE}}"

csv_count() {
  local csv=$1
  local part
  local count=0
  IFS=',' read -ra parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "${part}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

csv_contains() {
  local csv=$1
  local needle=$2
  local part
  IFS=',' read -ra parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ "${part}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

csv_first() {
  local csv=$1
  local part
  IFS=',' read -ra parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "${part}" ]]; then
      printf '%s\n' "${part}"
      return 0
    fi
  done
  return 1
}

csv_index() {
  local csv=$1
  local needle=$2
  local part
  local index=0
  IFS=',' read -ra parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    [[ -n "${part}" ]] || continue
    if [[ "${part}" == "${needle}" ]]; then
      printf '%s\n' "${index}"
      return 0
    fi
    index=$((index + 1))
  done
  return 1
}

TRAIN_NODE_COUNT="$(csv_count "${TRAIN_NODES}")"
POLICY_NODE_COUNT="$(csv_count "${POLICY_NODES}")"
if (( TRAIN_NODE_COUNT < 1 )); then
  echo "[prime-opd] PRIME_TRAIN_NODES must contain at least one node" >&2
  exit 1
fi
if (( POLICY_NODE_COUNT < 1 )); then
  echo "[prime-opd] PRIME_POLICY_NODES must contain at least one node" >&2
  exit 1
fi
TRAIN_NODE="$(csv_first "${TRAIN_NODES}")"

if (( SINGLE_NODE_MODE == 0 )); then
  if csv_contains "${TRAIN_NODES}" "${TEACHER_NODE}" || csv_contains "${POLICY_NODES}" "${TEACHER_NODE}"; then
    echo "[prime-opd] teacher node ${TEACHER_NODE} overlaps train=${TRAIN_NODES} or policy=${POLICY_NODES}" >&2
    exit 1
  fi
  IFS=',' read -ra TRAIN_NODE_PARTS <<< "${TRAIN_NODES}"
  for train_node_part in "${TRAIN_NODE_PARTS[@]}"; do
    train_node_part="${train_node_part//[[:space:]]/}"
    [[ -n "${train_node_part}" ]] || continue
    if csv_contains "${POLICY_NODES}" "${train_node_part}"; then
      echo "[prime-opd] trainer node ${train_node_part} also appears in policy nodes ${POLICY_NODES}" >&2
      exit 1
    fi
  done
fi

if (( SINGLE_NODE_MODE == 1 )); then
  if (( TRAIN_NODE_COUNT != 1 )) || [[ "${TRAIN_NODE}" != "${TEACHER_NODE}" || "${POLICY_NODES}" != "${TRAIN_NODE}" ]]; then
    echo "[prime-opd] 1node layout requires train, policy, and teacher to use the same node; got train=${TRAIN_NODES} policy=${POLICY_NODES} teacher=${TEACHER_NODE}" >&2
    exit 1
  fi
  if [[ "${NODE_LABEL}" != "${TRAIN_NODE}" ]]; then
    echo "[prime-opd] node=${NODE_LABEL} host=$(hostname) is not single-node target ${TRAIN_NODE}; skipping."
    exit 0
  fi
  PRIME_COMPONENT_ROLE="full"
elif [[ "${NODE_LABEL}" == "${TRAIN_NODE}" ]]; then
  PRIME_COMPONENT_ROLE="trainer_orchestrator"
elif csv_contains "${TRAIN_NODES}" "${NODE_LABEL}"; then
  PRIME_COMPONENT_ROLE="trainer_worker"
elif [[ "${NODE_LABEL}" == "${TEACHER_NODE}" ]]; then
  PRIME_COMPONENT_ROLE="teacher_inference"
elif csv_contains "${POLICY_NODES}" "${NODE_LABEL}"; then
  PRIME_COMPONENT_ROLE="policy_inference"
else
  echo "[prime-opd] node=${NODE_LABEL} host=$(hostname) not in train=${TRAIN_NODES} policy=${POLICY_NODES} teacher=${TEACHER_NODE}; skipping."
  exit 0
fi

RUN_NAME="${OLMO_RUN_DIR_NAME:-${PRIME_3NODE_RUN_NAME:-prime_rl_opd_full_vocab_dpsk_ctx8192_8node}}"
LOCK_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"

# If an earlier command was stopped while waiting for remote endpoints, its
# bash wrapper can keep holding the old role lock even though no Prime-RL
# process is active. Clean only old role command shells for this same node.
# Disabled by default: this runs inside an operator-managed command.sh parent,
# and broad command-shell cleanup can kill the current launch before training
# starts. Leave it opt-in for manual recovery of truly stale wrappers.
if [[ "${PRIME_3NODE_KILL_STALE_ROLE_SHELLS:-0}" == "1" ]]; then
  CURRENT_COMMAND_SCRIPT="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"
  CURRENT_PARENT_PID="${PPID:-}"
  mapfile -t STALE_ROLE_SHELL_PIDS < <(
    ps -eo pid=,args= \
      | awk -v self="$$" -v parent="${CURRENT_PARENT_PID}" -v node="${NODE_LABEL}" -v current_script="${CURRENT_COMMAND_SCRIPT}" '
          $1 != self && $1 != parent && index($0, current_script) == 0 && $0 ~ "/olmo_operator/node" node "/commands/.*/command.sh" { print $1 }
        '
  )
  if (( ${#STALE_ROLE_SHELL_PIDS[@]} > 0 )); then
    echo "[prime-opd-3node] terminating stale role command shell(s): ${STALE_ROLE_SHELL_PIDS[*]}"
    kill "${STALE_ROLE_SHELL_PIDS[@]}" 2>/dev/null || true
    sleep 3
    for stale_pid in "${STALE_ROLE_SHELL_PIDS[@]}"; do
      if kill -0 "${stale_pid}" 2>/dev/null; then
        kill -9 "${stale_pid}" 2>/dev/null || true
      fi
    done
  fi
fi

ROLE_LOCK="${PRIME_3NODE_ROLE_LOCK:-/tmp/prime_rl_opd_3node_${LOCK_RUN_NAME}_${NODE_LABEL}_${PRIME_COMPONENT_ROLE}.lock}"
exec 29>"${ROLE_LOCK}"
if ! flock -n 29; then
  echo "[prime-opd-3node] node=${NODE_LABEL} role=${PRIME_COMPONENT_ROLE} already running; lock=${ROLE_LOCK}; skipping duplicate operator."
  exit 0
fi

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
  export NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

detect_trainer_attention_backend() {
  local gpu_names="${PRIME_GPU_NAMES_OVERRIDE:-}"
  local compute_caps="${PRIME_GPU_COMPUTE_CAPS_OVERRIDE:-}"
  local requested="${PRIME_TRAINER_ATTN:-}"

  if [[ -z "${gpu_names}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  fi
  if [[ -z "${compute_caps}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  fi

  # OLMo3 mixes full and sliding attention. Magi FA4 has no sliding-window
  # interface, while FA3 is Hopper-only, so Blackwell must use the FA2 sink
  # backend even when a stale environment requests FA3/FA4 explicitly.
  if grep -Eiq '(^|[^[:alnum:]])(GB)?B(200|300)([^[:alnum:]]|$)|Blackwell' <<< "${gpu_names}" \
      || grep -Eq '^(10|11|12)\.' <<< "${compute_caps}"; then
    printf '%s\n' "olmo3_sink_fa2"
    return
  fi

  if [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return
  fi

  # Use FA3 only when all visible devices report Hopper's SM90 capability.
  # Mixed/unknown fleets take the portable FA2 path.
  if [[ -n "${compute_caps}" ]] \
      && ! grep -Evq '^[[:space:]]*9\.[0-9]+[[:space:]]*$' <<< "${compute_caps}"; then
    printf '%s\n' "olmo3_sink_fa3"
    return
  fi

  printf '%s\n' "olmo3_sink_fa2"
}

TRAINER_ATTN="$(detect_trainer_attention_backend)"
GPU_NAMES_ONE_LINE="$(tr '\n' ';' <<< "${PRIME_GPU_NAMES_OVERRIDE:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)}" | sed 's/;*$//')"
echo "[prime-opd] trainer attention backend=${TRAINER_ATTN} gpu_names=${GPU_NAMES_ONE_LINE:-unavailable}"

# The pinned vLLM wheel is built against CUDA 13 while some raw hosts expose a
# CUDA 12.x toolkit. NVIDIA's pip runtime can coexist with that toolkit, but
# its library directory is not added to the dynamic loader path automatically.
CUDA13_RUNTIME_DIR="${PRIME_CUDA13_RUNTIME_LIB_DIR:-}"
if [[ -z "${CUDA13_RUNTIME_DIR}" && "${PRIME_RL_RUNTIME_INSTALL_VLLM:-0}" == "1" ]]; then
  CUDA13_RUNTIME_DIR="$(find /opt /root/.local -path '*/site-packages/nvidia/cu13/lib' -type d -print -quit 2>/dev/null || true)"
fi
if [[ -n "${CUDA13_RUNTIME_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDA13_RUNTIME_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[prime-opd] using CUDA 13 pip runtime libraries from ${CUDA13_RUNTIME_DIR}"
fi

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
# DeepSeek-V4-Flash + MARLIN MoE has hit allocator asserts during vLLM's
# breakable CUDA graph profiling in Prime-RL worker mode. Explicitly opting out
# keeps the normal non-eager vLLM compile/cudagraph path, matching the standalone
# vllm serve baseline for this checkpoint.
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
# The DeepSeek-V4 sparse MLA startup warmup can fail with an invalid resource
# handle on the cluster even after normal vLLM startup gets past model load.
# This does not change the runtime attention backend; it only skips optional
# warmup dummy runs.
export VLLM_SKIP_DEEPSEEK_V4_SPARSE_MLA_WARMUP="${VLLM_SKIP_DEEPSEEK_V4_SPARSE_MLA_WARMUP:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-rl-full-vocab}"
# Teacher inference owns a full eight-GPU node. Match the scorer client to the
# engine's 16-request capacity; CPU parameter offload remains disabled unless
# PRIME_OPD_TEACHER_CPU_OFFLOAD_GB is set explicitly.
export PRIME_RL_PREFILL_HIDDEN_CONCURRENCY="${PRIME_RL_PREFILL_HIDDEN_CONCURRENCY:-16}"
export PRIME_RL_PREFILL_HIDDEN_RETRY_TIMEOUT_SECONDS="${PRIME_RL_PREFILL_HIDDEN_RETRY_TIMEOUT_SECONDS:-1200}"
export PRIME_RL_TEACHER_INFERENCE_MAX_RESTARTS="${PRIME_RL_TEACHER_INFERENCE_MAX_RESTARTS:-3}"
export PRIME_RL_TEACHER_INFERENCE_RESTART_DELAY_SECONDS="${PRIME_RL_TEACHER_INFERENCE_RESTART_DELAY_SECONDS:-10}"
export PRIME_RL_DETERMINISTIC_DP_WORKER_PORTS="${PRIME_RL_DETERMINISTIC_DP_WORKER_PORTS:-1}"
# GitHub DNS can be flaky on the NII nodes. Runtime repo fetches and Prime-RL
# submodule setup are retryable, so keep the window long enough for transient
# resolver outages without requiring a manual resubmit.
export RUNTIME_GIT_RETRY_ATTEMPTS="${RUNTIME_GIT_RETRY_ATTEMPTS:-12}"
export RUNTIME_GIT_RETRY_BASE_SECONDS="${RUNTIME_GIT_RETRY_BASE_SECONDS:-10}"
export RUNTIME_GIT_RETRY_MAX_SECONDS="${RUNTIME_GIT_RETRY_MAX_SECONDS:-90}"
export RUNTIME_DEPENDENCY_RETRY_ATTEMPTS="${RUNTIME_DEPENDENCY_RETRY_ATTEMPTS:-12}"
export RUNTIME_DEPENDENCY_RETRY_BASE_SECONDS="${RUNTIME_DEPENDENCY_RETRY_BASE_SECONDS:-10}"
export RUNTIME_DEPENDENCY_RETRY_MAX_SECONDS="${RUNTIME_DEPENDENCY_RETRY_MAX_SECONDS:-90}"
# Pin Prime-RL runtime vLLM to the wrapper's known-good wheel by default.
# This is required for the old NII SIF, which predates the rebuilt image. Set
# PRIME_RL_RUNTIME_INSTALL_VLLM=0 only when an image has been validated with
# its bundled vLLM wheel.
export PRIME_RL_RUNTIME_INSTALL_VLLM="${PRIME_RL_RUNTIME_INSTALL_VLLM:-1}"
export PRIME_RL_RUNTIME_VLLM_EXPECTED_VERSION="${PRIME_RL_RUNTIME_VLLM_EXPECTED_VERSION:-0.23.1rc1.dev699+gf5a8d7337}"
export PRIME_RL_RUNTIME_VLLM_WHEEL_URL="${PRIME_RL_RUNTIME_VLLM_WHEEL_URL:-https://wheels.vllm.ai/f5a8d73377d0f0a4e00cba172f9fbd0d50471b07/vllm-0.23.1rc1.dev699%2Bgf5a8d7337-cp38-abi3-manylinux_2_28_x86_64.whl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER_TP="${PRIME_OPD_TEACHER_TP:-${DEFAULT_TEACHER_TP}}"
TEACHER_DP="${PRIME_OPD_TEACHER_DP:-${DEFAULT_TEACHER_DP}}"
TEACHER_GPU_IDS="${PRIME_OPD_TEACHER_GPU_IDS:-${DEFAULT_TEACHER_GPU_IDS}}"
TEACHER_IS_BLACKWELL=0
if grep -Eiq '(^|[^[:alnum:]])(GB)?B(200|300)([^[:alnum:]]|$)|Blackwell' <<< "${GPU_NAMES_ONE_LINE}"; then
  TEACHER_IS_BLACKWELL=1
fi
TEACHER_VLLM_EXTRA_DEFAULT="$(
  TEACHER_TP="${TEACHER_TP}" \
  TEACHER_IS_BLACKWELL="${TEACHER_IS_BLACKWELL}" \
  TEACHER_KV_CACHE_MEMORY_BYTES="${PRIME_OPD_TEACHER_KV_CACHE_MEMORY_BYTES:-0}" \
  TEACHER_CPU_OFFLOAD_GB="${PRIME_OPD_TEACHER_CPU_OFFLOAD_GB:-0}" \
  python - <<'PY'
import json
import os

tp = int(os.environ["TEACHER_TP"])
is_blackwell = os.environ.get("TEACHER_IS_BLACKWELL") == "1"
kv_cache_memory_bytes = int(os.environ.get("TEACHER_KV_CACHE_MEMORY_BYTES", "0") or "0")
cpu_offload_gb = float(os.environ.get("TEACHER_CPU_OFFLOAD_GB", "0") or "0")
extra = {
    "kv_cache_dtype": "fp8",
    "block_size": 256,
    "enable_expert_parallel": True,
    "compilation_config": {
        "pass_config": {
            "fuse_allreduce_rms": False,
            "fi_allreduce_fusion_max_size_mb": 0,
        },
    },
    "additional_config": {},
}
if is_blackwell:
    extra["attention_config"] = {"use_fp4_indexer_cache": True}
    extra["moe_backend"] = "deep_gemm_mega_moe"
else:
    extra["linear_backend"] = "deep_gemm"
if kv_cache_memory_bytes > 0:
    extra["kv_cache_memory_bytes"] = kv_cache_memory_bytes
if cpu_offload_gb > 0:
    extra["cpu_offload_gb"] = cpu_offload_gb
if tp > 2:
    extra["disable_custom_all_reduce"] = True
print(json.dumps(extra, separators=(",", ":")))
PY
)"

RENDEZVOUS_DIR="${PRIME_3NODE_RENDEZVOUS_DIR:-/tmp/prime_rl_opd_3node/${RUN_NAME}}"
mkdir -p "${RENDEZVOUS_DIR}"
HIDDEN_STATE_DIR="${PRIME_OPD_FULL_VOCAB_HIDDEN_PATH:-${RENDEZVOUS_DIR}/teacher_hidden_states}"
mkdir -p "${HIDDEN_STATE_DIR}"
export PRIME_RL_HIDDEN_STATE_TTL_SECONDS="${PRIME_RL_HIDDEN_STATE_TTL_SECONDS:-21600}"
export PRIME_RL_HIDDEN_STATE_SWEEP_INTERVAL_SECONDS="${PRIME_RL_HIDDEN_STATE_SWEEP_INTERVAL_SECONDS:-600}"
export PRIME_RL_HIDDEN_STATE_MAX_PENDING_BYTES="${PRIME_RL_HIDDEN_STATE_MAX_PENDING_BYTES:-137438953472}"
export PRIME_RL_HIDDEN_STATE_BUDGET_POLL_SECONDS="${PRIME_RL_HIDDEN_STATE_BUDGET_POLL_SECONDS:-2}"

# Keep transient install/build/cache files in /tmp by default. The shared
# cluster's /dev/shm is fast but can make Git checkouts fail with index.lock
# write errors under pip's VCS install path.
TMP_ROOT="${PRIME_3NODE_TMP_ROOT:-/tmp/pp3/${RUN_NAME}/${NODE_LABEL}_${PRIME_COMPONENT_ROLE}}"
mkdir -p "${TMP_ROOT}"/{tmp,xdg,pip,triton,torchinductor,ray,vllm,rpc,flashinfer,deep_gemm}
export TMPDIR="${TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${TMP_ROOT}/xdg"
export PIP_CACHE_DIR="${TMP_ROOT}/pip"
export TRITON_CACHE_DIR="${TMP_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${TMP_ROOT}/torchinductor"
export RAY_TMPDIR="${TMP_ROOT}/ray"
export VLLM_CACHE_ROOT="${TMP_ROOT}/vllm"
export UV_CACHE_DIR="${TMP_ROOT}/uv"
# vLLM's data-parallel coordinator creates Unix sockets directly below this
# directory.  ``TMP_ROOT`` includes the descriptive run name and can exceed
# Linux's 107-byte sockaddr_un limit once vLLM appends a UUID.  Keep the IPC
# root short, but derive it from the run name and local role so concurrent runs
# on a shared node do not collide.
VLLM_RPC_RUN_ID="$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-12)"
VLLM_RPC_NODE_ID="${NODE_LABEL}_${PRIME_COMPONENT_ROLE}"
# Keep this outside the conventional /tmp/vllm-* namespace. Prime-RL and
# cluster cleanup jobs remove that namespace while retiring stale engines,
# which can race a newly starting role and delete its ZMQ parent directory.
export VLLM_RPC_BASE_PATH="${PRIME_VLLM_RPC_BASE_PATH:-/tmp/vrpc/${VLLM_RPC_RUN_ID}-${VLLM_RPC_NODE_ID}}"
export FLASHINFER_WORKSPACE_BASE="${TMP_ROOT}/flashinfer"
export FLASHINFER_CUBIN_DIR="${TMP_ROOT}/flashinfer/.cache/flashinfer/cubins"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-${TMP_ROOT}/deep_gemm}"
mkdir -p "${DG_JIT_CACHE_DIR}"
mkdir -p "${VLLM_RPC_BASE_PATH}"

TEACHER_HIDDEN_BACKEND="${PRIME_OPD_TEACHER_HIDDEN_BACKEND:-hook}"
case "${TEACHER_HIDDEN_BACKEND}" in
  extractor|vllm_extractor|official_extractor)
    export PRIME_RL_HIDDEN_STATE_BACKEND="vllm_extractor"
    TEACHER_HIDDEN_STORAGE="${PRIME_OPD_TEACHER_HIDDEN_STORAGE:-${TMP_ROOT}/hidden_states}"
    TEACHER_EXTRACTOR_LAYER_ID="${PRIME_OPD_TEACHER_EXTRACTOR_LAYER_ID:-42}"
    mkdir -p "${TEACHER_HIDDEN_STORAGE}"
    TEACHER_VLLM_EXTRA_DEFAULT="$(
      TEACHER_HIDDEN_STORAGE="${TEACHER_HIDDEN_STORAGE}" TEACHER_EXTRACTOR_LAYER_ID="${TEACHER_EXTRACTOR_LAYER_ID}" TEACHER_IS_BLACKWELL="${TEACHER_IS_BLACKWELL}" python - <<'PY'
import json
import os

layer_id = int(os.environ["TEACHER_EXTRACTOR_LAYER_ID"])
extra = {
    "kv_cache_dtype": "fp8",
    "block_size": 256,
    "enable_expert_parallel": True,
    "disable_custom_all_reduce": True,
    "compilation_config": {
        "pass_config": {
            "fuse_allreduce_rms": False,
            "fi_allreduce_fusion_max_size_mb": 0,
        },
    },
    "additional_config": {},
    "enable_chunked_prefill": False,
    "speculative_config": {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [layer_id],
            },
        },
    },
    "kv_transfer_config": {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": os.environ["TEACHER_HIDDEN_STORAGE"],
        },
    },
}
if os.environ.get("TEACHER_IS_BLACKWELL") == "1":
    extra["attention_config"] = {"use_fp4_indexer_cache": True}
    extra["moe_backend"] = "deep_gemm_mega_moe"
else:
    extra["linear_backend"] = "deep_gemm"
print(json.dumps(extra))
PY
    )"
    echo "[prime-opd-3node] using vLLM extractor hidden-state layer id ${TEACHER_EXTRACTOR_LAYER_ID}"
    ;;
  hook|"")
    export PRIME_RL_HIDDEN_STATE_BACKEND="hook"
    ;;
  *)
    echo "[prime-opd-3node] invalid PRIME_OPD_TEACHER_HIDDEN_BACKEND=${TEACHER_HIDDEN_BACKEND}; expected hook or extractor" >&2
    exit 1
    ;;
esac

HOST_NAME="$(hostname 2>/dev/null || echo unknown-host)"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="${HOST_NAME}"
fi
printf '%s\n' "${HOST_NAME}" > "${RENDEZVOUS_DIR}/node${NODE_LABEL}.host"
printf '%s\n' "${HOST_IP}" > "${RENDEZVOUS_DIR}/node${NODE_LABEL}.ip"

echo "[prime-opd-3node] run=${RUN_NAME}"
echo "[prime-opd-3node] node=${NODE_LABEL} role=${PRIME_COMPONENT_ROLE} host=${HOST_NAME} ip=${HOST_IP}"
echo "[prime-opd-3node] rendezvous=${RENDEZVOUS_DIR}"

wait_for_file() {
  local path=$1
  local timeout_s=${2:-900}
  local start
  start=$(date +%s)
  while [[ ! -s "${path}" ]]; do
    if (( $(date +%s) - start > timeout_s )); then
      echo "[prime-opd-3node] timeout waiting for ${path}" >&2
      ls -la "${RENDEZVOUS_DIR}" >&2 || true
      exit 1
    fi
    sleep 2
  done
}

wait_for_http() {
  local url=$1
  local timeout_s=${2:-7200}
  local start
  start=$(date +%s)
  until curl -fsS --max-time 5 "${url}" >/dev/null 2>&1; do
    if (( $(date +%s) - start > timeout_s )); then
      echo "[prime-opd-3node] timeout waiting for ${url}" >&2
      exit 1
    fi
    sleep 10
  done
}

POLICY_PORT="${PRIME_POLICY_PORT:-8000}"
TEACHER_PORT="${PRIME_OPD_TEACHER_PORT:-8001}"
REQUIRED_NODES_CSV="${TRAIN_NODES},${POLICY_NODES},${TEACHER_NODE}"
IFS=',' read -ra REQUIRED_NODE_PARTS <<< "${REQUIRED_NODES_CSV}"
for rank in "${REQUIRED_NODE_PARTS[@]}"; do
  rank="${rank//[[:space:]]/}"
  [[ -z "${rank}" ]] && continue
  wait_for_file "${RENDEZVOUS_DIR}/node${rank}.ip" 900
done

TRAIN_IP="$(cat "${RENDEZVOUS_DIR}/node${TRAIN_NODE}.ip")"
TEACHER_IP="$(cat "${RENDEZVOUS_DIR}/node${TEACHER_NODE}.ip")"
POLICY_BASE_URL=""
IFS=',' read -ra POLICY_NODE_PARTS <<< "${POLICY_NODES}"
for policy_node in "${POLICY_NODE_PARTS[@]}"; do
  policy_node="${policy_node//[[:space:]]/}"
  [[ -z "${policy_node}" ]] && continue
  POLICY_IP="$(cat "${RENDEZVOUS_DIR}/node${policy_node}.ip")"
  if [[ -n "${POLICY_BASE_URL}" ]]; then
    POLICY_BASE_URL+=","
  fi
  POLICY_BASE_URL+="http://${POLICY_IP}:${POLICY_PORT}/v1"
done
TEACHER_BASE_URL="http://${TEACHER_IP}:${TEACHER_PORT}/v1"

TRAINER_NODE_RANK="$(csv_index "${TRAIN_NODES}" "${NODE_LABEL}" 2>/dev/null || printf '%s' '-1')"
TRAINER_MASTER_PORT="${PRIME_TRAINER_MASTER_PORT:-29400}"
echo "[prime-opd-3node] layout train=${TRAIN_NODES} policy=${POLICY_NODES} teacher=${TEACHER_NODE} train_node_count=${TRAIN_NODE_COUNT} policy_node_count=${POLICY_NODE_COUNT}"
echo "[prime-opd-3node] train_ip=${TRAIN_IP}"
echo "[prime-opd-3node] policy_base_url=${POLICY_BASE_URL}"
echo "[prime-opd-3node] teacher_base_url=${TEACHER_BASE_URL}"

if [[ "${PRIME_COMMAND_PREVIEW:-0}" != "1" && "${PRIME_3NODE_CLEAN_ROLE_PROCS:-0}" == "1" ]]; then
  echo "[prime-opd-3node] cleaning stale Prime-RL/vLLM processes on role node ${NODE_LABEL}"
  pkill -9 -f "[p]ython.*prime_rl" 2>/dev/null || true
  pkill -9 -f "[t]orchrun.*prime_rl" 2>/dev/null || true
  # Prime-RL rewrites trainer worker process titles, so their command lines no
  # longer match the python/torchrun patterns after a parent exits.
  ps -eo pid=,args= \
    | awk '$2 == "PRIME-RL::Trainer" {print $1}' \
    | xargs -r kill -9 2>/dev/null || true
  pkill -9 -f "[v]llm" 2>/dev/null || true
  pkill -9 -f "[V]LLM::" 2>/dev/null || true
  pkill -9 -f "[E]ngineCore" 2>/dev/null || true
  pkill -9 -f "[A]PIServer" 2>/dev/null || true
  pkill -9 "vllm" 2>/dev/null || true
  nvidia-smi pmon -c 1 2>/dev/null \
    | awk 'NR > 2 && $2 ~ /^[0-9]+$/ && $9 ~ /VLLM::|EngineCore|APIServer/ {print $2}' \
    | xargs -r kill -9 2>/dev/null || true
  rm -rf /dev/shm/vllm-* /dev/shm/vllm_* /tmp/vllm-* /tmp/vllm_* /tmp/torch-* /tmp/torchelastic_* 2>/dev/null || true
fi

# Ensure this launch's private IPC root exists after local stale-process cleanup.
mkdir -p "${VLLM_RPC_BASE_PATH}"

MODEL_PATH="${PRIME_OPD_MODEL_PATH:-/tmp/models/opd-32b-deploy/opd-32b-deploy}"
TEACHER_MODEL_PATH="${PRIME_OPD_TEACHER_MODEL_PATH:-/tmp/models/dpsk-v4-flash}"
TRAIN_PYTHON="${PRIME_TRAIN_PYTHON:-/usr/bin/python}"
TRAIN_ENTRYPOINT="${PRIME_TRAIN_ENTRYPOINT:-/app/train.py}"
RUNTIME_BASE="${PRIME_3NODE_RUNTIME_BASE:-${TMP_ROOT}/runtime}"
mkdir -p "${RUNTIME_BASE}"
RUNTIME_ROOT="${PRIME_3NODE_RUNTIME_ROOT:-${RUNTIME_BASE}/aimo-proof-pilot}"
OPEN_INSTRUCT_RUNTIME_ROOT="${PRIME_3NODE_OPEN_INSTRUCT_RUNTIME_ROOT:-${RUNTIME_BASE}/open-instruct}"
OLMO_CORE_RUNTIME_ROOT="${PRIME_3NODE_OLMO_CORE_RUNTIME_ROOT:-${RUNTIME_BASE}/OLMo-core}"
VERL_RUNTIME_ROOT="${PRIME_3NODE_VERL_RUNTIME_ROOT:-${RUNTIME_BASE}/VERL}"
PRIME_RL_RUNTIME_ROOT="${PRIME_3NODE_PRIME_RL_RUNTIME_ROOT:-${RUNTIME_BASE}/prime-rl}"
# Hybrid training uses pre-rendered one-turn proof, verifier, selector, and
# refine prompts together with generated multi-turn IMO trajectories.
PER_TURN_DATASET_PATH="${PRIME_OPD_PER_TURN_DATASET_PATH:-/tmp/data/dsflash-proof-distill-v2-test/data/per_turn.parquet}"
DATASET_PATH="${PRIME_OPD_DATASET_PATH:-${PER_TURN_DATASET_PATH}}"
MULTI_TURN_DATASET_PATH="${PRIME_OPD_MULTI_TURN_DATASET_PATH:-${RUNTIME_ROOT}/data/imo_data_1959_2024.csv}"
VERIFIABLE_DATASET_PATH="${PRIME_OPD_VERIFIABLE_DATASET_PATH:-${RUNTIME_ROOT}/data/astral-bench.csv}"
EVAL_VERIFIABLE_DATASET_PATH="${PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH:-${RUNTIME_ROOT}/data/hmmt_feb_2026.csv}"
LOG_ROOT="${PRIME_OPD_LOG_ROOT:-${TMP_ROOT}/logs}"
if [[ "${PRIME_COMPONENT_ROLE}" == "trainer_orchestrator" || "${PRIME_COMPONENT_ROLE}" == "trainer_worker" ]]; then
  # Distributed trainer ranks must consume the same rollout shards and weight
  # broadcasts. Keep caches/logs role-local, but place state shared by both
  # trainer nodes under one path on the cluster filesystem.
  SHARED_TRAIN_ROOT="${PRIME_3NODE_SHARED_TRAIN_ROOT:-/tmp/pp3/${RUN_NAME}/shared_trainer}"
  OUTPUT_ROOT="${PRIME_OPD_OUTPUT_ROOT:-${SHARED_TRAIN_ROOT}/output}"
  CHECKPOINT_ROOT="${PRIME_OPD_CHECKPOINT_ROOT:-${SHARED_TRAIN_ROOT}/checkpoints}"
else
  OUTPUT_ROOT="${PRIME_OPD_OUTPUT_ROOT:-${TMP_ROOT}/output}"
  CHECKPOINT_ROOT="${PRIME_OPD_CHECKPOINT_ROOT:-${TMP_ROOT}/checkpoints/${RUN_NAME}_${PRIME_COMPONENT_ROLE}}"
fi

# Match the Beaker production shape. Completion remains below the context limit
# so prompts and generated traces retain headroom in both policy and teacher.
CTX_LEN="${PRIME_OPD_CTX_LEN:-131072}"
VLLM_CTX_LEN="${PRIME_OPD_VLLM_MAX_MODEL_LEN:-133120}"
TEACHER_VLLM_CTX_LEN="${PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN:-${VLLM_CTX_LEN}}"
OPD_DISTILL_MODE="${PRIME_OPD_DISTILL_MODE:-token_logprobs}"
COMPLETION_TOKENS="${PRIME_OPD_COMPLETION_TOKENS:-65000}"
EVAL_COMPLETION_TOKENS="${PRIME_OPD_EVAL_COMPLETION_TOKENS:-${COMPLETION_TOKENS}}"
BATCHED_TOKENS="${PRIME_OPD_BATCHED_TOKENS:-65536}"
# DeepSeek-V4-Flash is close to the H200 memory limit even in FP8/MXFP4.
# The teacher endpoint is used for serialized hidden-state scoring, so keep its
# startup profiling shape much smaller than the policy rollout endpoint.
# Keep teacher prefill chunks independent of max_model_len. Larger values make
# DeepSeek-V4 compile thousands of DeepGEMM warmup shapes before binding its API
# port; 4096 is also the last verified full-vocab hidden-state configuration.
TEACHER_BATCHED_TOKENS="${PRIME_OPD_TEACHER_BATCHED_TOKENS:-4096}"
MAX_STEPS="${MAX_TRAIN_STEPS:-1000}"
BATCH_SIZE="${PRIME_BATCH_SIZE:-2}"
PROOF_DATASET_MODE="${PRIME_PROOF_DATASET_MODE:-hybrid}"
case "${PROOF_DATASET_MODE}" in
  single|single_turn|per_turn|hybrid|single_and_multi|mixed_turns)
    GROUP_SIZE="${PRIME_GROUP_SIZE:-1}"
    CANDIDATE_GATE="${PRIME_PROOF_CANDIDATE_GATE:-false}"
    ;;
  *)
    GROUP_SIZE="${PRIME_GROUP_SIZE:-2}"
    CANDIDATE_GATE="${PRIME_PROOF_CANDIDATE_GATE:-true}"
    ;;
esac
CANDIDATE_CONTINUE_COUNT="${PRIME_PROOF_CANDIDATE_CONTINUE_COUNT:-2}"
PACKED_SEQUENCES_PER_STEP="${PRIME_PACKED_SEQUENCES_PER_STEP:-64}"
TOKEN_BATCH_SIZE=$((CTX_LEN * PACKED_SEQUENCES_PER_STEP))
INFLIGHT_PER_POLICY_NODE="${PRIME_OPD_INFLIGHT_ROLLOUTS_PER_POLICY_NODE:-48}"
DEFAULT_MAX_INFLIGHT=$((INFLIGHT_PER_POLICY_NODE * POLICY_NODE_COUNT))
MAX_INFLIGHT="${PRIME_OPD_MAX_INFLIGHT_ROLLOUTS:-${DEFAULT_MAX_INFLIGHT}}"
MAX_INFLIGHT_QUESTIONS="${PRIME_OPD_MAX_INFLIGHT_QUESTIONS:-0}"
if [[ -n "${PRIME_OPD_MAX_INFLIGHT_ROLLOUTS:-}" ]]; then
  echo "[prime-opd-3node] max_inflight_rollouts=${MAX_INFLIGHT} (explicit override)"
else
  echo "[prime-opd-3node] max_inflight_rollouts=${MAX_INFLIGHT} (${INFLIGHT_PER_POLICY_NODE}/policy_node)"
fi
echo "[prime-opd-3node] proof_dataset_mode=${PROOF_DATASET_MODE} candidate_gate=${CANDIDATE_GATE} group_size=${GROUP_SIZE} continue_after_proof=${CANDIDATE_CONTINUE_COUNT} randomized_multi_turn_continue_fraction=${PRIME_PROOF_MULTI_TURN_CONTINUE_FRACTION:-0.25}"
if (( MAX_INFLIGHT_QUESTIONS > 0 )); then
  echo "[prime-opd-3node] max_inflight_questions=${MAX_INFLIGHT_QUESTIONS} (fresh questions pause at cap; continuation turns remain eligible)"
fi
echo "[prime-opd-3node] full-environment batching token_batch_size=${TOKEN_BATCH_SIZE} (${PACKED_SEQUENCES_PER_STEP} packed sequences x seq_len ${CTX_LEN})"
MAX_OFF_POLICY="${PRIME_MAX_OFF_POLICY_STEPS:-24}"
TRAIN_GPU_COUNT="${PRIME_TRAIN_GPUS:-${DEFAULT_TRAIN_GPU_COUNT}}"
POLICY_TP="${PRIME_VLLM_TP:-${DEFAULT_POLICY_TP}}"
POLICY_DP="${PRIME_VLLM_DP:-${DEFAULT_POLICY_DP}}"
POLICY_GPU_COUNT=$((POLICY_TP * POLICY_DP))
if (( POLICY_GPU_COUNT < 1 || POLICY_GPU_COUNT > 8 )); then
  echo "[prime-opd-3node] invalid policy topology: PRIME_VLLM_TP=${POLICY_TP} PRIME_VLLM_DP=${POLICY_DP} requires ${POLICY_GPU_COUNT} GPUs on one 8-GPU policy node" >&2
  exit 1
fi
POLICY_API_SERVER_COUNT="${PRIME_VLLM_API_SERVER_COUNT:-${POLICY_DP}}"
# max_num_seqs is per DP worker, not per policy node. Six workers per rank
# yield the intended 48 requests per eight-GPU policy node.
POLICY_REQS_PER_DP="${PRIME_OPD_POLICY_REQS_PER_DP:-6}"
POLICY_MAX_NUM_SEQS_DEFAULT="${POLICY_REQS_PER_DP}"
POLICY_MAX_NUM_SEQS="${PRIME_OPD_POLICY_MAX_NUM_SEQS:-${POLICY_MAX_NUM_SEQS_DEFAULT}}"
NODE_PORT_OFFSET=0
if [[ "${NODE_LABEL}" =~ ^[0-9]+$ ]]; then
  NODE_PORT_OFFSET="${NODE_LABEL}"
fi
POLICY_DP_RPC_PORT="${PRIME_VLLM_DATA_PARALLEL_RPC_PORT:-$((37000 + NODE_PORT_OFFSET))}"
TEACHER_DP_RPC_PORT="${PRIME_OPD_TEACHER_VLLM_DATA_PARALLEL_RPC_PORT:-38005}"
TEACHER_GPU_COUNT="$(csv_count "${TEACHER_GPU_IDS}")"
TEACHER_PARALLEL_GPU_COUNT=$((TEACHER_TP * TEACHER_DP))
if (( TEACHER_GPU_COUNT != TEACHER_PARALLEL_GPU_COUNT )); then
  echo "[prime-opd] teacher topology mismatch: gpu_ids=${TEACHER_GPU_IDS} has ${TEACHER_GPU_COUNT} GPUs, but TP=${TEACHER_TP} DP=${TEACHER_DP} requires ${TEACHER_PARALLEL_GPU_COUNT}" >&2
  exit 1
fi
if (( SINGLE_NODE_MODE == 1 )); then
  ONE_NODE_GPU_COUNT=$((TRAIN_GPU_COUNT + POLICY_GPU_COUNT + TEACHER_GPU_COUNT))
  if (( ONE_NODE_GPU_COUNT > 8 )); then
    echo "[prime-opd] 1node topology requests ${ONE_NODE_GPU_COUNT} GPUs (train=${TRAIN_GPU_COUNT}, policy=${POLICY_GPU_COUNT}, teacher=${TEACHER_GPU_COUNT}); only 8 are available" >&2
    exit 1
  fi
  ONE_NODE_TRAINER_START=$((POLICY_GPU_COUNT))
  ONE_NODE_TEACHER_START=$((POLICY_GPU_COUNT + TRAIN_GPU_COUNT))
  IFS=',' read -ra TEACHER_GPU_PARTS <<< "${TEACHER_GPU_IDS}"
  for teacher_gpu in "${TEACHER_GPU_PARTS[@]}"; do
    teacher_gpu="${teacher_gpu//[[:space:]]/}"
    if [[ ! "${teacher_gpu}" =~ ^[0-7]$ ]] || (( teacher_gpu < ONE_NODE_TEACHER_START )); then
      echo "[prime-opd] 1node teacher GPU id ${teacher_gpu} overlaps policy/trainer GPUs 0-$((ONE_NODE_TEACHER_START - 1)) or is outside 0-7" >&2
      exit 1
    fi
  done
fi
echo "[prime-opd-3node] policy_topology tp=${POLICY_TP} dp=${POLICY_DP} api_servers=${POLICY_API_SERVER_COUNT} max_num_seqs_per_dp_rank=${POLICY_MAX_NUM_SEQS} total_sequence_capacity=$((POLICY_DP * POLICY_MAX_NUM_SEQS)) dp_rpc_port=${POLICY_DP_RPC_PORT}"
echo "[prime-opd-3node] trainer_gpus=${TRAIN_GPU_COUNT} teacher_topology tp=${TEACHER_TP} dp=${TEACHER_DP} gpu_ids=${TEACHER_GPU_IDS} teacher_dp_rpc_port=${TEACHER_DP_RPC_PORT}"
DFLASH_ENABLE="${PRIME_DFLASH_ENABLE:-0}"
DFLASH_DRAFT_MODEL=""
if [[ "${DFLASH_ENABLE}" == "1" ]]; then
  DFLASH_DRAFT_MODEL="${PRIME_DFLASH_DRAFT_MODEL:-}"
fi
if [[ "${DFLASH_ENABLE}" == "1" && -z "${DFLASH_DRAFT_MODEL}" ]]; then
  for dflash_candidate in \
    "/tmp/models/dflash-32b-draft-v2test-phaseL" \
    "/tmp/model/dflash-32b-draft-v2test-phaseL"; do
    if [[ -f "${dflash_candidate}/config.json" ]]; then
      DFLASH_DRAFT_MODEL="${dflash_candidate}"
      break
    fi
  done
fi
DFLASH_NUM_SPECULATIVE_TOKENS="${PRIME_DFLASH_NUM_SPECULATIVE_TOKENS:-10}"
POLICY_VLLM_EXTRA_DEFAULT="$(
  POLICY_TP="${POLICY_TP}" \
  DFLASH_DRAFT_MODEL="${DFLASH_DRAFT_MODEL}" \
  DFLASH_NUM_SPECULATIVE_TOKENS="${DFLASH_NUM_SPECULATIVE_TOKENS}" \
  python - <<'PY'
import json
import os

tp = int(os.environ["POLICY_TP"])
draft_model = os.environ.get("DFLASH_DRAFT_MODEL", "").strip()
extra = {
    "kv_cache_dtype": "fp8",
    "block_size": 256,
    "compilation_config": {
        "pass_config": {
            "fuse_allreduce_rms": False,
        },
    },
}
if tp > 2:
    extra["disable_custom_all_reduce"] = True
if draft_model:
    extra["speculative_config"] = {
        "method": "dflash",
        "model": draft_model,
        "num_speculative_tokens": int(os.environ["DFLASH_NUM_SPECULATIVE_TOKENS"]),
        "draft_tensor_parallel_size": 1,
    }
print(json.dumps(extra, separators=(",", ":")))
PY
)"
if [[ -n "${DFLASH_DRAFT_MODEL}" ]]; then
  echo "[prime-opd-3node] policy DFlash enabled draft_model=${DFLASH_DRAFT_MODEL} num_speculative_tokens=${DFLASH_NUM_SPECULATIVE_TOKENS}"
else
  echo "[prime-opd-3node] policy DFlash disabled; set PRIME_DFLASH_ENABLE=1 to enable"
fi
POLICY_GPU_IDS_DEFAULT=""
for ((gpu_idx = 0; gpu_idx < POLICY_GPU_COUNT; gpu_idx++)); do
  if [[ -n "${POLICY_GPU_IDS_DEFAULT}" ]]; then
    POLICY_GPU_IDS_DEFAULT+=","
  fi
  POLICY_GPU_IDS_DEFAULT+="${gpu_idx}"
done
POLICY_GPU_IDS="${PRIME_POLICY_GPU_IDS:-${POLICY_GPU_IDS_DEFAULT}}"

# The host/operator process exports GLOBAL_RANK/NODE_RANK for node selection.
# Each Prime-RL component below is intentionally a single-node process, so do
# not let train.py's runtime-fetch coordination reinterpret nodes 3/4/5 as
# multinode training ranks and wait forever for a node-0 marker.
TRAIN_PY_ENV=(
  env
  -u GLOBAL_RANK
  -u NODE_RANK
  -u SLURM_NODEID
  -u RANK
  -u LOCAL_RANK
  -u WORLD_SIZE
)

launch_train() {
  if [[ "${PRIME_COMMAND_PREVIEW:-0}" == "1" ]]; then
    printf '[prime-opd] command preview:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  local monitor_interval="${PRIME_RESOURCE_MONITOR_INTERVAL_SECONDS:-15}"
  if [[ "${monitor_interval}" =~ ^[0-9]+$ ]] && (( monitor_interval > 0 )); then
    local monitored_pid=$$
    local resource_log="${LOG_ROOT}/resource_usage.log"
    mkdir -p "${LOG_ROOT}"
    (
      while kill -0 "${monitored_pid}" 2>/dev/null; do
        {
          printf '\n[%s] cgroup_memory_current=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          cat /sys/fs/cgroup/memory.current 2>/dev/null || printf 'unavailable\n'
          printf 'cgroup_memory_peak='
          cat /sys/fs/cgroup/memory.peak 2>/dev/null || printf 'unavailable\n'
          printf 'cgroup_memory_events='
          tr '\n' ' ' < /sys/fs/cgroup/memory.events 2>/dev/null || true
          printf '\n'
          ps -eo pid,ppid,rss,comm,args --sort=-rss | head -n 17
        } >> "${resource_log}" 2>&1
        sleep "${monitor_interval}"
      done
    ) &
  fi
  exec "$@"
}

if [[ -n "${PRIME_TOKEN_BATCH_SIZE:-}" ]]; then
  BATCH_TARGET_ARGS=(--prime_token_batch_size "${PRIME_TOKEN_BATCH_SIZE}")
else
  BATCH_TARGET_ARGS=(--prime_packed_sequences_per_step "${PACKED_SEQUENCES_PER_STEP}")
fi

COMMON_ARGS=(
  --fetch-update
  --submissions-repo "${SUBMISSIONS_REPO:-https://github.com/nguyen599/aimo-proof-pilot.git}"
  --submissions-ref "${SUBMISSIONS_REF:-main}"
  --prime-rl-ref "${PRIME_RL_REF:-main}"
  --submissions-runtime-dir "${RUNTIME_ROOT}"
  --open-instruct-runtime-dir "${OPEN_INSTRUCT_RUNTIME_ROOT}"
  --olmo-core-runtime-dir "${OLMO_CORE_RUNTIME_ROOT}"
  --verl-runtime-dir "${VERL_RUNTIME_ROOT}"
  --prime-rl-runtime-dir "${PRIME_RL_RUNTIME_ROOT}"
  --runtime-fetch-state-dir "${TMP_ROOT}/train-runtime-fetch"
  --runtime-training-deps-dir "${TMP_ROOT}/olmo-train-runtime-deps"
  --node_rank 0
  --num_nodes 1
  --backend prime_rl
  --model_path "${MODEL_PATH}"
  --tokenizer_path "${MODEL_PATH}"
  --dataset_path "${DATASET_PATH}"
  --output_path "${OUTPUT_ROOT}"
  --logdir "${LOG_ROOT}"
  --max_train_steps "${MAX_STEPS}"
  --max_seq_length "${CTX_LEN}"
  --rollout_max_completion_tokens "${COMPLETION_TOKENS}"
  --optimizer "${PRIME_OPTIMIZER:-te_fused_adamw}"
  --learning_rate "${PRIME_LEARNING_RATE:-2e-7}"
  --prime_lr_scheduler "${PRIME_LR_SCHEDULER:-cosine}"
  --prime_lr_warmup_steps "${PRIME_LR_WARMUP_STEPS:-10}"
  --prime_lr_decay_steps "${PRIME_LR_DECAY_STEPS:-10}"
  --prime_lr_min "${PRIME_LR_MIN:-3e-8}"
  --weight_decay "${PRIME_WEIGHT_DECAY:-0.0}"
  --max_grad_norm "${PRIME_MAX_GRAD_NORM:-1.0}"
  --prime_algorithm opd
  --prime_opd_distill_mode "${OPD_DISTILL_MODE}"
  --prime_opd_full_vocab_hidden_transport "${PRIME_OPD_FULL_VOCAB_HIDDEN_TRANSPORT:-filesystem}"
  --prime_opd_full_vocab_hidden_path "${HIDDEN_STATE_DIR}"
  --prime_opd_full_vocab_hidden_codec "${PRIME_OPD_FULL_VOCAB_HIDDEN_CODEC:-had_int6_blk32}"
  --prime_opd_full_vocab_teacher_lm_head_path "${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH:-${TEACHER_MODEL_PATH}}"
  --prime_opd_full_vocab_teacher_lm_head_key "${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_KEY:-head.weight}"
  --prime_opd_full_vocab_teacher_hidden_dtype "${PRIME_OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE:-bfloat16}"
  --prime_opd_full_vocab_token_chunk_size "${PRIME_OPD_FULL_VOCAB_TOKEN_CHUNK_SIZE:-512}"
  --prime_opd_full_vocab_vocab_chunk_size "${PRIME_OPD_FULL_VOCAB_VOCAB_CHUNK_SIZE:-8192}"
  --prime_env_id proof-opd-env
  --prime_env_name proof_math
  --prime_proof_dataset_path "${DATASET_PATH}"
  --prime_proof_dataset_mode "${PROOF_DATASET_MODE}"
  --prime_proof_multi_turn_dataset_path "${MULTI_TURN_DATASET_PATH}"
  --prime_proof_multi_turn_fraction "${PRIME_OPD_MULTI_TURN_FRACTION:-0.20}"
  --prime_proof_multi_turn_continue_fraction "${PRIME_PROOF_MULTI_TURN_CONTINUE_FRACTION:-0.25}"
  --prime_proof_verifiable_dataset_path "${VERIFIABLE_DATASET_PATH}"
  --prime_proof_verifiable_fraction "${PRIME_OPD_VERIFIABLE_FRACTION:-0.20}"
  --prime_proof_verifiable_answer_column auto
  --prime_proof_mix_seed "${PRIME_OPD_VERIFIABLE_MIX_SEED:-34521}"
  --prime_proof_problem_column auto
  --prime_proof_solution_column auto
  --prime_proof_judge_backend none
  --prime_proof_max_examples "${PRIME_PROOF_MAX_EXAMPLES:-0}"
  --prime_proof_enable_meta_verification "${PRIME_PROOF_ENABLE_META_VERIFICATION:-true}"
  --prime_proof_num_verifiers "${PRIME_PROOF_NUM_VERIFIERS:-4}"
  --prime_proof_refine_rounds "${PRIME_PROOF_REFINE_ROUNDS:-1}"
  --prime_proof_refine_review_n "${PRIME_PROOF_REFINE_REVIEW_N:-2}"
  --prime_proof_selector_top_k "${PRIME_PROOF_SELECTOR_TOP_K:-3}"
  --prime_proof_enable_selector "${PRIME_PROOF_ENABLE_SELECTOR:-false}"
  --prime_proof_candidate_gate "${CANDIDATE_GATE}"
  --prime_proof_candidate_continue_count "${CANDIDATE_CONTINUE_COUNT}"
  --prime_eval_verifiable_dataset_path "${EVAL_VERIFIABLE_DATASET_PATH}"
  --prime_eval_interval "${PRIME_OPD_EVAL_INTERVAL:-50}"
  --prime_eval_skip_first_step "${PRIME_OPD_EVAL_SKIP_FIRST_STEP:-true}"
  --prime_eval_num_examples "${PRIME_OPD_EVAL_NUM_EXAMPLES:-33}"
  --prime_eval_group_size "${PRIME_OPD_EVAL_GROUP_SIZE:-1}"
  --prime_eval_max_completion_tokens "${EVAL_COMPLETION_TOKENS}"
  --prime_eval_refine_rounds "${PRIME_OPD_EVAL_REFINE_ROUNDS:-0}"
  --prime_eval_num_verifiers "${PRIME_OPD_EVAL_NUM_VERIFIERS:-1}"
  --prime_eval_refine_review_n "${PRIME_OPD_EVAL_REFINE_REVIEW_N:-1}"
  --prime_eval_answer_column auto
  --prime_batch_size "${BATCH_SIZE}"
  --prime_group_size "${GROUP_SIZE}"
  "${BATCH_TARGET_ARGS[@]}"
  --prime_max_inflight_rollouts "${MAX_INFLIGHT}"
  --prime_max_off_policy_steps "${MAX_OFF_POLICY}"
  --prime_gpus_per_node 8
  --prime_trainer_model_impl custom
  --prime_trainer_attn "${TRAINER_ATTN}"
  --prime_trainer_dp_replicate "${PRIME_TRAINER_DP_REPLICATE:-${TRAIN_NODE_COUNT}}"
  --prime_trainer_context_parallel_size "${PRIME_TRAINER_CP:-1}"
  --prime_trainer_cp_style ulysses
  --prime_trainer_fsdp_cpu_offload false
  --prime_trainer_optim_cpu_offload "${PRIME_TRAINER_OPTIM_CPU_OFFLOAD:-false}"
  --prime_trainer_fp8 "${PRIME_TRAINER_FP8:-true}"
  # Compiling every decoder layer during the first full-vocab backward can
  # exceed the one-node container's host-memory cgroup. Keep compilation
  # available as an explicit override after the eager path is established.
  --prime_trainer_compile "${PRIME_TRAINER_COMPILE:-true}"
  --prime_weight_broadcast_type "${PRIME_WEIGHT_BROADCAST_TYPE:-filesystem}"
  --prime_weight_broadcast_port "${PRIME_WEIGHT_BROADCAST_PORT:-29501}"
  --prime_weight_broadcast_timeout "${PRIME_WEIGHT_BROADCAST_TIMEOUT:-7200}"
  --prime_weight_broadcast_quantize_in_weight_transfer "${PRIME_WEIGHT_BROADCAST_QUANTIZE:-false}"
  --prime_checkpoint_interval "${PRIME_CHECKPOINT_INTERVAL:-100}"
  --prime_checkpoint_keep_last "${PRIME_CHECKPOINT_KEEP_LAST:-20}"
  --prime_checkpoint_keep_interval "${PRIME_CHECKPOINT_KEEP_INTERVAL:-0}"
  --prime_checkpoint_output_dir "${CHECKPOINT_ROOT}"
  --prime_checkpoint_weights_only "${PRIME_CHECKPOINT_WEIGHTS_ONLY:-true}"
  --prime_checkpoint_wait_for_weights_timeout "${PRIME_CHECKPOINT_WAIT_FOR_WEIGHTS_TIMEOUT:-7200}"
  --prime_skip_model_check true
  --prime_temperature "${PRIME_TEMPERATURE:-1.0}"
  --prime_top_p 0.95
  --with_tracking
  --wandb_mode online
  --wandb_project "${WANDB_PROJECT}"
)
if (( MAX_INFLIGHT_QUESTIONS > 0 )); then
  COMMON_ARGS+=(--prime_max_inflight_questions "${MAX_INFLIGHT_QUESTIONS}")
fi

case "${PRIME_COMPONENT_ROLE}" in
  policy_inference)
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_policy_node${NODE_LABEL}"
    # Run TP=1,DP=8 by default: eight single-GPU policy instances avoid OLMo3
    # TP Q/K RMSNorm all-gathers during decode while still using all 8 GPUs.
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${PRIME_VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
    export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${PRIME_VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-2147483648}"
    launch_train "${TRAIN_PY_ENV[@]}" "${TRAIN_PYTHON}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
      --prime_component policy_inference \
      --prime_policy_port "${POLICY_PORT}" \
      --prime_policy_gpu_ids "${POLICY_GPU_IDS}" \
      --prime_train_gpus 0 \
      --prime_infer_gpus "${POLICY_GPU_COUNT}" \
      --prime_vllm_tensor_parallel_size "${POLICY_TP}" \
      --prime_vllm_data_parallel_size "${POLICY_DP}" \
      --prime_vllm_max_model_len "${VLLM_CTX_LEN}" \
      --prime_vllm_dtype bfloat16 \
      --prime_vllm_enforce_eager "${PRIME_VLLM_ENFORCE_EAGER:-false}" \
      --prime_vllm_quantization "${PRIME_VLLM_QUANTIZATION:-fp8}" \
      --prime_vllm_gpu_memory_utilization "${PRIME_VLLM_GPU_MEMORY_UTILIZATION:-0.95}" \
      --prime_vllm_api_server_count "${POLICY_API_SERVER_COUNT}" \
      --prime_vllm_data_parallel_rpc_port "${POLICY_DP_RPC_PORT}" \
      --prime_vllm_use_deep_gemm "${PRIME_VLLM_USE_DEEP_GEMM:-false}" \
      --prime_vllm_max_num_seqs "${POLICY_MAX_NUM_SEQS}" \
      --prime_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
      --prime_vllm_reasoning_parser deepseek_v4 \
      --prime_vllm_extra "${PRIME_VLLM_EXTRA:-${POLICY_VLLM_EXTRA_DEFAULT}}"
    ;;

  teacher_inference)
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_teacher_node${NODE_LABEL}"
    # The pinned vLLM wheel can route TP allreduce+RMS fusion through
    # flashinfer.comm, which imports tilelang's libcudart stub on this cluster
    # and fails with missing cudaDeviceReset. Keep the DeepGEMM linear backend
    # for DeepSeek-V4-Flash, but avoid FlashInfer allreduce fusion here.
    # Prompt-logprob scoring also materializes an FP32 log-softmax workspace, so
    # leave transient VRAM free instead of assigning it all to the KV cache.
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${PRIME_OPD_TEACHER_VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
    export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${PRIME_OPD_TEACHER_VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-2147483648}"
    launch_train "${TRAIN_PY_ENV[@]}" "${TRAIN_PYTHON}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
      --prime_component teacher_inference \
      --prime_train_gpus 0 \
      --prime_infer_gpus 0 \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_teacher_tokenizer_path "${MODEL_PATH}" \
      --prime_opd_start_teacher true \
      --prime_opd_teacher_gpu_ids "${TEACHER_GPU_IDS}" \
      --prime_opd_teacher_port "${TEACHER_PORT}" \
      --prime_opd_teacher_ready_timeout "${PRIME_OPD_TEACHER_READY_TIMEOUT:-7200}" \
      --prime_opd_teacher_vllm_tensor_parallel_size "${TEACHER_TP}" \
      --prime_opd_teacher_vllm_data_parallel_size "${TEACHER_DP}" \
      --prime_opd_teacher_vllm_max_model_len "${TEACHER_VLLM_CTX_LEN}" \
      --prime_opd_teacher_vllm_dtype bfloat16 \
      --prime_opd_teacher_vllm_enforce_eager "${PRIME_OPD_TEACHER_VLLM_ENFORCE_EAGER:-false}" \
      --prime_opd_teacher_vllm_quantization "${PRIME_OPD_TEACHER_VLLM_QUANTIZATION:-none}" \
      --prime_opd_teacher_vllm_gpu_memory_utilization "${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.90}" \
      --prime_opd_teacher_vllm_api_server_count "${PRIME_OPD_TEACHER_VLLM_API_SERVER_COUNT:-1}" \
      --prime_opd_teacher_vllm_data_parallel_rpc_port "${TEACHER_DP_RPC_PORT}" \
      --prime_opd_teacher_vllm_use_deep_gemm "${PRIME_OPD_TEACHER_USE_DEEP_GEMM:-false}" \
      --prime_opd_teacher_vllm_max_num_seqs "${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-16}" \
      --prime_opd_teacher_vllm_max_num_batched_tokens "${TEACHER_BATCHED_TOKENS}" \
      --prime_opd_teacher_vllm_reasoning_parser deepseek_v4 \
      --prime_opd_teacher_vllm_extra "${PRIME_OPD_TEACHER_VLLM_EXTRA:-${TEACHER_VLLM_EXTRA_DEFAULT}}"
    ;;

  trainer_orchestrator)
    echo "[prime-opd-3node] starting trainer; train_engine_rl will wait for policy and teacher endpoints"
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_trainer"
    export OLMO_OUTPUT_RUN_DIR_NAME="${RUN_NAME}_trainer"
    export OLMO_LOG_RUN_DIR_NAME="${RUN_NAME}_trainer_node${NODE_LABEL}"
    launch_train "${TRAIN_PY_ENV[@]}" "${TRAIN_PYTHON}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
      --prime_component trainer_orchestrator \
      --prime_train_gpus "${TRAIN_GPU_COUNT}" \
      --prime_infer_gpus 0 \
      --prime_policy_base_url "${POLICY_BASE_URL}" \
      --prime_policy_admin_base_url "${POLICY_BASE_URL}" \
      --prime_policy_dp_rank_count "${PRIME_POLICY_DP_RANK_COUNT:-${POLICY_DP}}" \
      --prime_vllm_tensor_parallel_size "${POLICY_TP}" \
      --prime_vllm_data_parallel_size "${POLICY_DP}" \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_teacher_tokenizer_path "${MODEL_PATH}" \
      --prime_opd_start_teacher false \
      --prime_opd_teacher_base_url "${TEACHER_BASE_URL}" \
      --prime_opd_teacher_vllm_tensor_parallel_size "${TEACHER_TP}" \
      --prime_opd_teacher_vllm_data_parallel_size "${TEACHER_DP}" \
      --prime_trainer_num_nodes "${TRAIN_NODE_COUNT}" \
      --prime_trainer_node_rank "${TRAINER_NODE_RANK}" \
      --prime_trainer_master_addr "${TRAIN_IP}" \
      --prime_trainer_master_port "${TRAINER_MASTER_PORT}" \
      --prime_trainer_rdzv_id "${RUN_NAME}" \
      --prime_trainer_rdzv_timeout "${PRIME_TRAINER_RDZV_TIMEOUT:-7200}"
    ;;

  trainer_worker)
    echo "[prime-opd] starting distributed trainer worker rank ${TRAINER_NODE_RANK}/${TRAIN_NODE_COUNT}"
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_trainer"
    export OLMO_OUTPUT_RUN_DIR_NAME="${RUN_NAME}_trainer"
    export OLMO_LOG_RUN_DIR_NAME="${RUN_NAME}_trainer_node${NODE_LABEL}"
    launch_train "${TRAIN_PY_ENV[@]}" "${TRAIN_PYTHON}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
      --prime_component trainer_worker \
      --prime_train_gpus "${TRAIN_GPU_COUNT}" \
      --prime_infer_gpus 0 \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_teacher_tokenizer_path "${MODEL_PATH}" \
      --prime_opd_start_teacher false \
      --prime_trainer_num_nodes "${TRAIN_NODE_COUNT}" \
      --prime_trainer_node_rank "${TRAINER_NODE_RANK}" \
      --prime_trainer_master_addr "${TRAIN_IP}" \
      --prime_trainer_master_port "${TRAINER_MASTER_PORT}" \
      --prime_trainer_rdzv_id "${RUN_NAME}" \
      --prime_trainer_rdzv_timeout "${PRIME_TRAINER_RDZV_TIMEOUT:-7200}"
    ;;

  full)
    echo "[prime-opd] starting one-node full stack: policy GPUs 0-$((POLICY_GPU_COUNT - 1)), trainer GPUs ${POLICY_GPU_COUNT}-$((POLICY_GPU_COUNT + TRAIN_GPU_COUNT - 1)), teacher GPUs ${TEACHER_GPU_IDS}"
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_full_node${NODE_LABEL}"
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${PRIME_VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
    export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${PRIME_VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-2147483648}"
    launch_train "${TRAIN_PY_ENV[@]}" "${TRAIN_PYTHON}" "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" \
      --prime_component full \
      --prime_train_gpus "${TRAIN_GPU_COUNT}" \
      --prime_infer_gpus "${POLICY_GPU_COUNT}" \
      --prime_policy_port "${POLICY_PORT}" \
      --prime_vllm_tensor_parallel_size "${POLICY_TP}" \
      --prime_vllm_data_parallel_size "${POLICY_DP}" \
      --prime_vllm_max_model_len "${VLLM_CTX_LEN}" \
      --prime_vllm_dtype bfloat16 \
      --prime_vllm_enforce_eager "${PRIME_VLLM_ENFORCE_EAGER:-false}" \
      --prime_vllm_quantization "${PRIME_VLLM_QUANTIZATION:-fp8}" \
      --prime_vllm_gpu_memory_utilization "${PRIME_VLLM_GPU_MEMORY_UTILIZATION:-0.95}" \
      --prime_vllm_api_server_count "${POLICY_API_SERVER_COUNT}" \
      --prime_vllm_data_parallel_rpc_port "${POLICY_DP_RPC_PORT}" \
      --prime_vllm_use_deep_gemm "${PRIME_VLLM_USE_DEEP_GEMM:-false}" \
      --prime_vllm_max_num_seqs "${POLICY_MAX_NUM_SEQS}" \
      --prime_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
      --prime_vllm_reasoning_parser deepseek_v4 \
      --prime_vllm_extra "${PRIME_VLLM_EXTRA:-${POLICY_VLLM_EXTRA_DEFAULT}}" \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_teacher_tokenizer_path "${MODEL_PATH}" \
      --prime_opd_start_teacher true \
      --prime_opd_teacher_gpu_ids "${TEACHER_GPU_IDS}" \
      --prime_opd_teacher_port "${TEACHER_PORT}" \
      --prime_opd_teacher_ready_timeout "${PRIME_OPD_TEACHER_READY_TIMEOUT:-7200}" \
      --prime_opd_teacher_vllm_tensor_parallel_size "${TEACHER_TP}" \
      --prime_opd_teacher_vllm_data_parallel_size "${TEACHER_DP}" \
      --prime_opd_teacher_vllm_max_model_len "${TEACHER_VLLM_CTX_LEN}" \
      --prime_opd_teacher_vllm_dtype bfloat16 \
      --prime_opd_teacher_vllm_enforce_eager "${PRIME_OPD_TEACHER_VLLM_ENFORCE_EAGER:-false}" \
      --prime_opd_teacher_vllm_quantization "${PRIME_OPD_TEACHER_VLLM_QUANTIZATION:-none}" \
      --prime_opd_teacher_vllm_gpu_memory_utilization "${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.90}" \
      --prime_opd_teacher_vllm_api_server_count "${PRIME_OPD_TEACHER_VLLM_API_SERVER_COUNT:-1}" \
      --prime_opd_teacher_vllm_data_parallel_rpc_port "${TEACHER_DP_RPC_PORT}" \
      --prime_opd_teacher_vllm_use_deep_gemm "${PRIME_OPD_TEACHER_USE_DEEP_GEMM:-false}" \
      --prime_opd_teacher_vllm_max_num_seqs "${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-16}" \
      --prime_opd_teacher_vllm_max_num_batched_tokens "${TEACHER_BATCHED_TOKENS}" \
      --prime_opd_teacher_vllm_reasoning_parser deepseek_v4 \
      --prime_opd_teacher_vllm_extra "${PRIME_OPD_TEACHER_VLLM_EXTRA:-${TEACHER_VLLM_EXTRA_DEFAULT}}"
    ;;
esac
