#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPLICAS="${OPD_EXPECTED_REPLICAS:-8}"
REPLICA_COUNT="${BEAKER_REPLICA_COUNT:?BEAKER_REPLICA_COUNT is required; submit this as a replicated Beaker task}"
REPLICA_RANK="${BEAKER_REPLICA_RANK:?BEAKER_REPLICA_RANK is required; submit this as a replicated Beaker task}"

if [[ "${REPLICA_COUNT}" != "${EXPECTED_REPLICAS}" ]]; then
  echo "[beaker-opd] expected ${EXPECTED_REPLICAS} replicas, got ${REPLICA_COUNT}" >&2
  exit 2
fi
if [[ ! "${REPLICA_RANK}" =~ ^[0-9]+$ ]] || (( REPLICA_RANK < 0 || REPLICA_RANK >= REPLICA_COUNT )); then
  echo "[beaker-opd] invalid replica rank ${REPLICA_RANK}/${REPLICA_COUNT}" >&2
  exit 2
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd ';' -)"
GPU_CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "[beaker-opd] each replica requires 8 GPUs, found ${GPU_COUNT}" >&2
  exit 2
fi
if [[ "${OPD_REQUIRE_BLACKWELL:-1}" == "1" ]] \
    && ! grep -Eiq '(B200|B300|Blackwell)' <<< "${GPU_NAMES}"; then
  echo "[beaker-opd] expected B200/B300 GPUs, found ${GPU_NAMES}" >&2
  exit 2
fi

RUN_NAME="${OPD_RUN_NAME:?OPD_RUN_NAME must be identical and unique across all replicas}"
SHARED_ROOT="${OPD_SHARED_ROOT:-/weka/aimo-proof-pilot}"
RUN_ROOT="${OPD_RUN_ROOT:-${SHARED_ROOT}/runs/${RUN_NAME}}"
MODEL_ROOT="${OPD_MODEL_ROOT:-${SHARED_ROOT}/models}"
STUDENT_MODEL="${OPD_STUDENT_MODEL_PATH:-${MODEL_ROOT}/opd-32b-deploy/opd-32b-deploy}"
TEACHER_MODEL="${OPD_TEACHER_MODEL_PATH:-${MODEL_ROOT}/dpsk-v4-flash}"
DATASET_PATH="${OPD_DATASET_PATH:-${SHARED_ROOT}/data/per_turn.parquet}"
LOCAL_ROOT="${OPD_LOCAL_ROOT:-/tmp/aimo-proof-pilot/${RUN_NAME}/node${REPLICA_RANK}}"
CONTROL_DIR="${RUN_ROOT}/beaker-control"
READY_FILE="${CONTROL_DIR}/launch.ready"

validate_assets() {
  local required
  for required in "${STUDENT_MODEL}/config.json" "${TEACHER_MODEL}/config.json" "${DATASET_PATH}"; do
    if [[ ! -f "${required}" ]]; then
      echo "[beaker-opd] required asset is missing: ${required}" >&2
      return 1
    fi
  done
}

mkdir -p "${CONTROL_DIR}" "${LOCAL_ROOT}"
if (( REPLICA_RANK == 0 )); then
  if [[ -e "${READY_FILE}" && "${OPD_ALLOW_EXISTING_RUN:-0}" != "1" ]]; then
    echo "[beaker-opd] ${RUN_NAME} already has a launch marker; choose a new OPD_RUN_NAME" >&2
    exit 4
  fi
  if [[ ! -w "${SHARED_ROOT}" ]]; then
    echo "[beaker-opd] shared WEKA root is not writable: ${SHARED_ROOT}" >&2
    exit 3
  fi
  if [[ "${OPD_AUTO_DOWNLOAD_ASSETS:-1}" == "1" ]]; then
    OPD_SHARED_ROOT="${SHARED_ROOT}" \
    OPD_MODEL_ROOT="${MODEL_ROOT}" \
    OPD_STUDENT_MODEL_PATH="${STUDENT_MODEL}" \
    OPD_TEACHER_MODEL_PATH="${TEACHER_MODEL}" \
    OPD_DATASET_PATH="${DATASET_PATH}" \
      /usr/local/bin/beaker-opd-prepare-assets
  fi
  validate_assets || exit 3
  {
    echo "run_name=${RUN_NAME}"
    echo "replicas=${REPLICA_COUNT}"
    echo "leader=${BEAKER_LEADER_REPLICA_HOSTNAME:-unknown}"
    echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${READY_FILE}.tmp"
  mv "${READY_FILE}.tmp" "${READY_FILE}"
else
  asset_ready_timeout="${OPD_ASSET_READY_TIMEOUT:-14400}"
  for _ in $(seq 1 $((asset_ready_timeout / 2))); do
    [[ -f "${READY_FILE}" ]] && break
    sleep 2
  done
  [[ -f "${READY_FILE}" ]] || { echo "[beaker-opd] timeout waiting for ${READY_FILE}" >&2; exit 4; }
  validate_assets || exit 3
fi

export GLOBAL_RANK="${REPLICA_RANK}"
export OLMO_RUN_DIR_NAME="${RUN_NAME}"
export PRIME_NODE_LAYOUT=8node

ROLE_LAYOUT="${OPD_ROLE_LAYOUT:-1|6|1}"
IFS='|' read -r TRAIN_NODE_COUNT POLICY_NODE_COUNT TEACHER_NODE_COUNT ROLE_LAYOUT_EXTRA <<< "${ROLE_LAYOUT}"
if [[ -n "${ROLE_LAYOUT_EXTRA:-}" ]] \
    || ! "${TRAIN_NODE_COUNT:-}" =~ ^[1-9][0-9]*$ \
    || ! "${POLICY_NODE_COUNT:-}" =~ ^[1-9][0-9]*$ \
    || ! "${TEACHER_NODE_COUNT:-}" =~ ^[1-9][0-9]*$; then
  echo "[beaker-opd] OPD_ROLE_LAYOUT must be TRAIN|POLICY|TEACHER positive counts; got ${ROLE_LAYOUT}" >&2
  exit 2
fi
if (( TRAIN_NODE_COUNT + POLICY_NODE_COUNT + TEACHER_NODE_COUNT != REPLICA_COUNT )); then
  echo "[beaker-opd] role counts ${ROLE_LAYOUT} do not sum to ${REPLICA_COUNT} replicas" >&2
  exit 2
fi
if (( TEACHER_NODE_COUNT != 1 )); then
  echo "[beaker-opd] this launcher currently requires exactly one teacher node; got ${TEACHER_NODE_COUNT}" >&2
  exit 2
fi

csv_range() {
  local start=$1
  local count=$2
  local result=""
  local index
  for ((index = start; index < start + count; index++)); do
    result="${result:+${result},}${index}"
  done
  printf '%s\n' "${result}"
}

export PRIME_TRAIN_NODES="$(csv_range 0 "${TRAIN_NODE_COUNT}")"
export PRIME_POLICY_NODES="$(csv_range "${TRAIN_NODE_COUNT}" "${POLICY_NODE_COUNT}")"
export PRIME_TEACHER_NODE="$((TRAIN_NODE_COUNT + POLICY_NODE_COUNT))"

export PRIME_TRAIN_PYTHON="${PRIME_TRAIN_PYTHON:-$(command -v python)}"
export PRIME_TRAIN_ENTRYPOINT="${PRIME_TRAIN_ENTRYPOINT:-/app/train.py}"
export PRIME_3NODE_TMP_ROOT="${LOCAL_ROOT}"
export PRIME_3NODE_RUNTIME_BASE="${LOCAL_ROOT}/runtime"
export PRIME_3NODE_RENDEZVOUS_DIR="${RUN_ROOT}/rdzv"
export PRIME_OPD_OUTPUT_ROOT="${RUN_ROOT}/output"
export PRIME_OPD_LOG_ROOT="${RUN_ROOT}/logs"
export PRIME_OPD_CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
export PRIME_OPD_FULL_VOCAB_HIDDEN_PATH="${RUN_ROOT}/hidden_states"

export PRIME_OPD_MODEL_PATH="${STUDENT_MODEL}"
export PRIME_OPD_TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export PRIME_OPD_DATASET_PATH="${DATASET_PATH}"
export PRIME_PROOF_DATASET_MODE="${PRIME_PROOF_DATASET_MODE:-single}"
export PRIME_OPD_DISTILL_MODE="${PRIME_OPD_DISTILL_MODE:-full_vocab_hidden}"
export PRIME_OPD_FULL_VOCAB_HIDDEN_TRANSPORT="${PRIME_OPD_FULL_VOCAB_HIDDEN_TRANSPORT:-filesystem}"
export PRIME_OPD_FULL_VOCAB_HIDDEN_CODEC="${PRIME_OPD_FULL_VOCAB_HIDDEN_CODEC:-had_int6_blk32}"
export PRIME_OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE="${PRIME_OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE:-bfloat16}"

export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
export PRIME_OPD_CTX_LEN="${PRIME_OPD_CTX_LEN:-81920}"
export PRIME_OPD_VLLM_MAX_MODEL_LEN="${PRIME_OPD_VLLM_MAX_MODEL_LEN:-90112}"
export PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN="${PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN:-90112}"
export PRIME_OPD_COMPLETION_TOKENS="${PRIME_OPD_COMPLETION_TOKENS:-65000}"
export PRIME_OPD_EVAL_COMPLETION_TOKENS="${PRIME_OPD_EVAL_COMPLETION_TOKENS:-65000}"
export PRIME_PACKED_SEQUENCES_PER_STEP="${PRIME_PACKED_SEQUENCES_PER_STEP:-64}"

export PRIME_BATCH_SIZE="${PRIME_BATCH_SIZE:-2}"
export PRIME_GROUP_SIZE="${PRIME_GROUP_SIZE:-1}"
export PRIME_PROOF_CANDIDATE_GATE=false
export PRIME_PROOF_ENABLE_META_VERIFICATION=false
export PRIME_PROOF_NUM_VERIFIERS=1
export PRIME_PROOF_REFINE_ROUNDS=0
export PRIME_OPD_MAX_INFLIGHT_ROLLOUTS="${PRIME_OPD_MAX_INFLIGHT_ROLLOUTS:-$((48 * POLICY_NODE_COUNT))}"
export PRIME_OPD_MAX_INFLIGHT_QUESTIONS=0
export PRIME_MAX_OFF_POLICY_STEPS="${PRIME_MAX_OFF_POLICY_STEPS:-24}"

export PRIME_TRAIN_GPUS=8
export PRIME_TRAINER_CP="${PRIME_TRAINER_CP:-1}"
export PRIME_TRAINER_FP8="${PRIME_TRAINER_FP8:-true}"
export PRIME_TRAINER_COMPILE="${PRIME_TRAINER_COMPILE:-false}"
export PRIME_TRAINER_OPTIM_CPU_OFFLOAD="${PRIME_TRAINER_OPTIM_CPU_OFFLOAD:-false}"
unset PRIME_TRAINER_ATTN

export PRIME_VLLM_TP=1
export PRIME_VLLM_DP=8
export PRIME_VLLM_API_SERVER_COUNT=8
export PRIME_OPD_POLICY_MAX_NUM_SEQS="${PRIME_OPD_POLICY_MAX_NUM_SEQS:-6}"
export PRIME_OPD_BATCHED_TOKENS="${PRIME_OPD_BATCHED_TOKENS:-65536}"
export PRIME_VLLM_ENFORCE_EAGER=false

export PRIME_OPD_TEACHER_TP=8
export PRIME_OPD_TEACHER_DP=1
export PRIME_OPD_TEACHER_GPU_IDS=0,1,2,3,4,5,6,7
export PRIME_OPD_TEACHER_MAX_NUM_SEQS="${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-1}"
export PRIME_OPD_TEACHER_BATCHED_TOKENS="${PRIME_OPD_TEACHER_BATCHED_TOKENS:-4096}"
export PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION="${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.96}"
export PRIME_OPD_TEACHER_VLLM_ENFORCE_EAGER=false

export PRIME_CHECKPOINT_INTERVAL="${PRIME_CHECKPOINT_INTERVAL:-100}"
export PRIME_CHECKPOINT_KEEP_LAST="${PRIME_CHECKPOINT_KEEP_LAST:-2}"
export PRIME_CHECKPOINT_WEIGHTS_ONLY="${PRIME_CHECKPOINT_WEIGHTS_ONLY:-true}"
export PRIME_OPD_EVAL_INTERVAL="${PRIME_OPD_EVAL_INTERVAL:-50}"
export PRIME_OPD_EVAL_SKIP_FIRST_STEP=true

export PRIME_RL_RUNTIME_INSTALL_VLLM=0
export PRIME_RL_HIDDEN_STATE_BACKEND=hook
export PRIME_OPD_TEACHER_HIDDEN_BACKEND=hook
export PRIME_RESOURCE_MONITOR_INTERVAL_SECONDS="${PRIME_RESOURCE_MONITOR_INTERVAL_SECONDS:-60}"
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-rl-full-vocab}"
export WANDB_SHARED_RUN_ID="${WANDB_SHARED_RUN_ID:-$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-32)}"
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-^=mlx5_bond_0}"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

echo "[beaker-opd] run=${RUN_NAME} replica=${REPLICA_RANK}/${REPLICA_COUNT} host=$(hostname)"
echo "[beaker-opd] gpu_names=${GPU_NAMES} compute_caps=${GPU_CAPS} shared_root=${SHARED_ROOT}"
echo "[beaker-opd] topology layout=${ROLE_LAYOUT} trainer=${PRIME_TRAIN_NODES} policy=${PRIME_POLICY_NODES} teacher=${PRIME_TEACHER_NODE}; B200/B300 trainer attention must resolve to olmo3_sink_fa2"

exec bash /opt/aimo-proof-pilot-beaker/prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh
