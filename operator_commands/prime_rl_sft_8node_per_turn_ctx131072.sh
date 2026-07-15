#!/usr/bin/env bash
set -euo pipefail

# Native Prime-RL SFT over per_turn.parquet.
#
# Default NII layout:
#   nodes 0..7: one external torchrun world, eight H200s per node
#   HSDP:       one eight-GPU FSDP island per node, replicated across nodes
#   microbatch: 2 packed sequences per GPU
#   accumulation: 1 microstep (128 global sequences per optimizer step)
#
# Useful overrides:
#   PRIME_SFT_TRAIN_NODES=0                 one-node smoke test
#   PRIME_TRAINER_ATTN=olmo3_sink_fa3_native
#                                             use the native FA3 sink kernel
#   PRIME_COMMAND_PREVIEW=1                 print the resolved command

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
TRAIN_NODES="${PRIME_SFT_TRAIN_NODES:-0,1,2,3,4,5,6,7}"

csv_count() {
  local csv=$1
  local part
  local count=0
  IFS=',' read -ra parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    [[ -n "${part}" ]] && count=$((count + 1))
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
    [[ "${part}" == "${needle}" ]] && return 0
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
if (( TRAIN_NODE_COUNT < 1 )); then
  echo "[prime-sft] PRIME_SFT_TRAIN_NODES must contain at least one node" >&2
  exit 1
fi
if [[ "${NODE_LABEL}" == "none" && "${TRAIN_NODE_COUNT}" == "1" ]]; then
  NODE_LABEL="$(csv_first "${TRAIN_NODES}")"
fi
if ! csv_contains "${TRAIN_NODES}" "${NODE_LABEL}"; then
  echo "[prime-sft] node=${NODE_LABEL} host=$(hostname) is not in train nodes ${TRAIN_NODES}; skipping."
  exit 0
fi
TRAINER_NODE_RANK="$(csv_index "${TRAIN_NODES}" "${NODE_LABEL}")"
TRAIN_NODE="$(csv_first "${TRAIN_NODES}")"

RUN_NAME="${PRIME_SFT_RUN_NAME:-${OLMO_RUN_DIR_NAME:-prime_rl_sft_per_turn_ctx131072_8node}}"
SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"
ROLE_LOCK="${PRIME_SFT_ROLE_LOCK:-/tmp/prime_rl_sft_${SAFE_RUN_NAME}_node${NODE_LABEL}.lock}"
exec 29>"${ROLE_LOCK}"
if ! flock -n 29; then
  echo "[prime-sft] node=${NODE_LABEL} already has this run active; lock=${ROLE_LOCK}"
  exit 0
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

  # Magi FA3 is Hopper-only. OLMo3Sink mixes full and sliding attention, so
  # Blackwell uses Magi FA2 until its FA4 sink API exposes sliding windows.
  if grep -Eiq '(^|[^[:alnum:]])(GB)?B(200|300)([^[:alnum:]]|$)|Blackwell' <<< "${gpu_names}" \
      || grep -Eq '^[[:space:]]*(10|11|12)\.' <<< "${compute_caps}"; then
    printf '%s\n' "olmo3_sink_fa2"
    return
  fi
  if [[ -n "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return
  fi
  if [[ -n "${compute_caps}" ]] \
      && ! grep -Evq '^[[:space:]]*9\.[0-9]+[[:space:]]*$' <<< "${compute_caps}"; then
    printf '%s\n' "olmo3_sink_fa3"
    return
  fi
  printf '%s\n' "olmo3_sink_fa2"
}

TRAINER_ATTN="$(detect_trainer_attention_backend)"
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"
echo "[prime-sft] attention=${TRAINER_ATTN} gpu_names=${GPU_NAMES:-unavailable}"

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="online"
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-sft}"
export MAGI_ATTENTION_SKIP_CUDA_BUILD="${MAGI_ATTENTION_SKIP_CUDA_BUILD:-1}"

# SFT does not import vLLM. Keep the old NII SIF compatible without downloading
# the 280 MiB runtime vLLM wheel and its large optional dependency tree.
export PRIME_RL_RUNTIME_INSTALL_VLLM="${PRIME_RL_RUNTIME_INSTALL_VLLM:-0}"
export PRIME_RL_RUNTIME_INSTALL_TORCH="${PRIME_RL_RUNTIME_INSTALL_TORCH:-0}"
export RUNTIME_GIT_RETRY_ATTEMPTS="${RUNTIME_GIT_RETRY_ATTEMPTS:-12}"
export RUNTIME_GIT_RETRY_BASE_SECONDS="${RUNTIME_GIT_RETRY_BASE_SECONDS:-10}"
export RUNTIME_GIT_RETRY_MAX_SECONDS="${RUNTIME_GIT_RETRY_MAX_SECONDS:-90}"
export RUNTIME_DEPENDENCY_RETRY_ATTEMPTS="${RUNTIME_DEPENDENCY_RETRY_ATTEMPTS:-12}"
export RUNTIME_DEPENDENCY_RETRY_BASE_SECONDS="${RUNTIME_DEPENDENCY_RETRY_BASE_SECONDS:-10}"
export RUNTIME_DEPENDENCY_RETRY_MAX_SECONDS="${RUNTIME_DEPENDENCY_RETRY_MAX_SECONDS:-90}"

SHARED_ROOT="${PRIME_SFT_SHARED_ROOT:-/tmp/prime-sft-runs/${RUN_NAME}}"
RENDEZVOUS_DIR="${PRIME_SFT_RENDEZVOUS_DIR:-${SHARED_ROOT}/rendezvous}"
RUNTIME_BASE="${PRIME_SFT_RUNTIME_BASE:-${SHARED_ROOT}/runtime}"
RUNTIME_DEPS_DIR="${PRIME_SFT_RUNTIME_DEPS_DIR:-${SHARED_ROOT}/runtime-deps}"
OUTPUT_ROOT="${PRIME_SFT_OUTPUT_ROOT:-${SHARED_ROOT}/output}"
LOG_ROOT="${PRIME_SFT_LOG_ROOT:-${SHARED_ROOT}/logs}"
CACHE_ROOT="${PRIME_SFT_CACHE_DIR:-${SHARED_ROOT}/normalized-data}"
CHECKPOINT_ROOT="${PRIME_SFT_CHECKPOINT_DIR:-${SHARED_ROOT}/checkpoints}"
LOCAL_TMP_ROOT="${PRIME_SFT_LOCAL_TMP_ROOT:-/tmp/prime-sft-local/${RUN_NAME}/node${NODE_LABEL}}"
mkdir -p "${RENDEZVOUS_DIR}" "${RUNTIME_BASE}" "${RUNTIME_DEPS_DIR}" \
  "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}" "${CHECKPOINT_ROOT}" "${LOCAL_TMP_ROOT}"

export TMPDIR="${LOCAL_TMP_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${LOCAL_TMP_ROOT}/xdg"
export PIP_CACHE_DIR="${LOCAL_TMP_ROOT}/pip"
export TRITON_CACHE_DIR="${LOCAL_TMP_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_TMP_ROOT}/torchinductor"
export UV_CACHE_DIR="${LOCAL_TMP_ROOT}/uv"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${UV_CACHE_DIR}"

HOST_NAME="$(hostname 2>/dev/null || echo unknown-host)"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -n "${HOST_IP}" ]] || HOST_IP="${HOST_NAME}"
printf '%s\n' "${HOST_NAME}" > "${RENDEZVOUS_DIR}/node${NODE_LABEL}.host"
printf '%s\n' "${HOST_IP}" > "${RENDEZVOUS_DIR}/node${NODE_LABEL}.ip"

wait_for_file() {
  local path=$1
  local timeout_s=${2:-1800}
  local start
  start=$(date +%s)
  while [[ ! -s "${path}" ]]; do
    if (( $(date +%s) - start > timeout_s )); then
      echo "[prime-sft] timeout waiting for ${path}" >&2
      ls -la "${RENDEZVOUS_DIR}" >&2 || true
      exit 1
    fi
    sleep 2
  done
}

IFS=',' read -ra TRAIN_NODE_PARTS <<< "${TRAIN_NODES}"
for rank in "${TRAIN_NODE_PARTS[@]}"; do
  rank="${rank//[[:space:]]/}"
  [[ -n "${rank}" ]] && wait_for_file "${RENDEZVOUS_DIR}/node${rank}.ip"
done
MASTER_ADDR="$(cat "${RENDEZVOUS_DIR}/node${TRAIN_NODE}.ip")"
MASTER_PORT="${PRIME_SFT_MASTER_PORT:-29400}"
export MASTER_ADDR MASTER_PORT
export RUNTIME_FETCH_COORDINATION_ID="${RUN_NAME}"

MODEL_PATH="${PRIME_SFT_MODEL_PATH:-/tmp/models/opd-32b-deploy/opd-32b-deploy}"
DATASET_PATH="${PRIME_SFT_DATASET_PATH:-/tmp/data/opd-v2-test/data/per_turn.parquet}"
DATASET_URL="${PRIME_SFT_DATASET_URL:-https://huggingface.co/datasets/ycchen/dsflash-proof-distill-v2-test/resolve/main/data/per_turn.parquet?download=true}"

# Resolve the public parquet once before all eight train wrappers enter their
# asset phase. This avoids eight simultaneous 1.6 GiB downloads to one path.
DATASET_READY="${RENDEZVOUS_DIR}/dataset.ready"
DATASET_ERROR="${RENDEZVOUS_DIR}/dataset.error"
if [[ "${PRIME_COMMAND_PREVIEW:-0}" != "1" ]]; then
  if [[ "${NODE_LABEL}" == "${TRAIN_NODE}" ]]; then
    rm -f "${DATASET_READY}" "${DATASET_ERROR}"
    if [[ ! -s "${DATASET_PATH}" ]]; then
      mkdir -p "$(dirname "${DATASET_PATH}")"
      DATASET_TMP="${DATASET_PATH}.tmp-$$"
      if ! curl -fL --retry 12 --retry-all-errors --connect-timeout 30 \
          -o "${DATASET_TMP}" "${DATASET_URL}"; then
        rm -f "${DATASET_TMP}"
        printf 'dataset download failed: %s\n' "${DATASET_URL}" > "${DATASET_ERROR}"
        exit 1
      fi
      mv "${DATASET_TMP}" "${DATASET_PATH}"
    fi
    printf '%s\n' "${DATASET_PATH}" > "${DATASET_READY}"
  else
    while [[ ! -s "${DATASET_READY}" || ! -s "${DATASET_PATH}" ]]; do
      if [[ -s "${DATASET_ERROR}" ]]; then
        cat "${DATASET_ERROR}" >&2
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[prime-sft] model is missing config.json: ${MODEL_PATH}" >&2
    exit 1
  fi
fi

MAX_STEPS="${PRIME_SFT_MAX_STEPS:-1000}"
SEQ_LEN="${PRIME_SFT_SEQ_LEN:-131072}"
TRAIN_GPUS_PER_NODE=8
TRAIN_WORLD_SIZE=$((TRAIN_NODE_COUNT * TRAIN_GPUS_PER_NODE))
MICRO_BATCH_SIZE="${PRIME_SFT_MICRO_BATCH_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${PRIME_SFT_CONTEXT_PARALLEL_SIZE:-1}"
REQUESTED_GRAD_ACCUM_STEPS="${PRIME_SFT_GRAD_ACCUM_STEPS:-1}"

for value_name in MICRO_BATCH_SIZE CONTEXT_PARALLEL_SIZE REQUESTED_GRAD_ACCUM_STEPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[prime-sft] ${value_name} must be a positive integer; got ${value}" >&2
    exit 1
  fi
done

if [[ -n "${PRIME_SFT_GLOBAL_BATCH_SIZE:-}" ]]; then
  GLOBAL_BATCH_SIZE="${PRIME_SFT_GLOBAL_BATCH_SIZE}"
else
  derived_batch_numerator=$((TRAIN_WORLD_SIZE * MICRO_BATCH_SIZE * REQUESTED_GRAD_ACCUM_STEPS))
  if (( derived_batch_numerator % CONTEXT_PARALLEL_SIZE != 0 )); then
    echo "[prime-sft] world_size * micro_batch_size * grad_accum_steps must be divisible by CP" >&2
    exit 1
  fi
  GLOBAL_BATCH_SIZE=$((derived_batch_numerator / CONTEXT_PARALLEL_SIZE))
fi

if ! [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[prime-sft] GLOBAL_BATCH_SIZE must be a positive integer; got ${GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
  echo "[prime-sft] global batch ${GLOBAL_BATCH_SIZE} must be divisible by microbatch ${MICRO_BATCH_SIZE}" >&2
  exit 1
fi
accum_numerator=$((GLOBAL_BATCH_SIZE * CONTEXT_PARALLEL_SIZE))
accum_denominator=$((TRAIN_WORLD_SIZE * MICRO_BATCH_SIZE))
if (( accum_numerator % accum_denominator != 0 )); then
  echo "[prime-sft] batch_size * CP must be divisible by world_size * micro_batch_size" >&2
  exit 1
fi
GRAD_ACCUM_STEPS=$((accum_numerator / accum_denominator))
if [[ -n "${PRIME_SFT_GRAD_ACCUM_STEPS:-}" ]] \
    && (( GRAD_ACCUM_STEPS != REQUESTED_GRAD_ACCUM_STEPS )); then
  echo "[prime-sft] explicit global batch resolves to grad_accum=${GRAD_ACCUM_STEPS}, requested ${REQUESTED_GRAD_ACCUM_STEPS}" >&2
  exit 1
fi

WANDB_SHARED_RUN_ID="${PRIME_SFT_WANDB_RUN_ID:-$(printf '%s' "${RUN_NAME}" | sha256sum | cut -c1-32)}"
export WANDB_SHARED_MODE=1 WANDB_SHARED_RUN_ID
export OLMO_RUN_DIR_NAME="${RUN_NAME}_node${NODE_LABEL}"
export OLMO_OUTPUT_RUN_DIR_NAME="${RUN_NAME}"
export OLMO_LOG_RUN_DIR_NAME="${RUN_NAME}/node${NODE_LABEL}"

TRAIN_PYTHON="${PRIME_TRAIN_PYTHON:-/usr/bin/python}"
TRAIN_ENTRYPOINT="${PRIME_TRAIN_ENTRYPOINT:-/app/train.py}"
TRAIN_PY_ENV=(
  env
  -u GLOBAL_RANK
  -u NODE_RANK
  -u SLURM_NODEID
  -u RANK
  -u LOCAL_RANK
  -u WORLD_SIZE
)

COMMAND=(
  "${TRAIN_PY_ENV[@]}"
  "${TRAIN_PYTHON}"
  "${TRAIN_ENTRYPOINT}"
  --fetch-update
  --submissions-repo "${SUBMISSIONS_REPO:-https://github.com/nguyen599/aimo-proof-pilot.git}"
  --submissions-ref "${SUBMISSIONS_REF:-main}"
  --prime-rl-ref "${PRIME_RL_REF:-main}"
  --submissions-runtime-dir "${RUNTIME_BASE}/aimo-proof-pilot"
  --open-instruct-runtime-dir "${RUNTIME_BASE}/open-instruct"
  --olmo-core-runtime-dir "${RUNTIME_BASE}/OLMo-core"
  --verl-runtime-dir "${RUNTIME_BASE}/VERL"
  --prime-rl-runtime-dir "${RUNTIME_BASE}/prime-rl"
  --runtime-fetch-state-dir "${SHARED_ROOT}/runtime-fetch"
  --runtime-training-deps-dir "${RUNTIME_DEPS_DIR}"
  --node_rank "${TRAINER_NODE_RANK}"
  --num_nodes "${TRAIN_NODE_COUNT}"
  --backend prime_rl
  --prime_algorithm sft
  --prime_component sft_trainer
  --model_path "${MODEL_PATH}"
  --tokenizer_path "${MODEL_PATH}"
  --dataset_path "${DATASET_PATH}"
  --dataset_hf_repo "${PRIME_SFT_DATASET_HF_REPO:-ycchen/dsflash-proof-distill-v2-test}"
  --dataset_hf_filename "${PRIME_SFT_DATASET_HF_FILENAME:-data/per_turn.parquet}"
  --output_path "${OUTPUT_ROOT}"
  --logdir "${LOG_ROOT}"
  --max_train_steps "${MAX_STEPS}"
  --max_seq_length "${SEQ_LEN}"
  --optimizer "${PRIME_SFT_OPTIMIZER:-te_fused_adamw}"
  --learning_rate "${PRIME_SFT_LEARNING_RATE:-2e-7}"
  --prime_lr_scheduler cosine
  --prime_lr_warmup_steps "${PRIME_SFT_LR_WARMUP_STEPS:-10}"
  --prime_lr_min "${PRIME_SFT_MIN_LR:-3e-8}"
  --weight_decay "${PRIME_SFT_WEIGHT_DECAY:-0.1}"
  --max_grad_norm "${PRIME_SFT_MAX_GRAD_NORM:-1.0}"
  --prime_te_adamw_exp_avg_dtype bfloat16
  --prime_te_adamw_exp_avg_sq_dtype bfloat16
  --prime_te_adamw_master_weight_dtype bfloat16
  --prime_te_adamw_master_weights false
  --prime_te_adamw_store_param_remainders false
  --prime_sft_cache_dir "${CACHE_ROOT}"
  --prime_sft_stages "${PRIME_SFT_STAGES:-prove,verify,select,refine}"
  --prime_sft_validation_problem_count "${PRIME_SFT_VALIDATION_PROBLEMS:-33}"
  --prime_sft_seed "${PRIME_SFT_SEED:-34521}"
  --prime_sft_prepare_timeout "${PRIME_SFT_PREPARE_TIMEOUT:-7200}"
  --prime_sft_global_batch_size "${GLOBAL_BATCH_SIZE}"
  --prime_sft_micro_batch_size "${MICRO_BATCH_SIZE}"
  --prime_sft_nodes_per_fsdp_group "${PRIME_SFT_NODES_PER_FSDP_GROUP:-1}"
  --prime_sft_pack_function cat
  --prime_sft_overflow_policy skip
  --prime_sft_loss_impl liger_fused
  --prime_sft_eval_interval "${PRIME_SFT_EVAL_INTERVAL:-50}"
  --prime_sft_activation_offloading "${PRIME_SFT_ACTIVATION_OFFLOADING:-true}"
  --prime_sft_activation_offloading_max_inflight "${PRIME_SFT_ACTIVATION_OFFLOADING_MAX_INFLIGHT:-1}"
  --prime_gpus_per_node "${TRAIN_GPUS_PER_NODE}"
  --prime_train_gpus "${TRAIN_GPUS_PER_NODE}"
  --prime_trainer_num_nodes "${TRAIN_NODE_COUNT}"
  --prime_trainer_node_rank "${TRAINER_NODE_RANK}"
  --prime_trainer_master_addr "${MASTER_ADDR}"
  --prime_trainer_master_port "${MASTER_PORT}"
  --prime_trainer_rdzv_id "${RUN_NAME}"
  --prime_trainer_rdzv_timeout "${PRIME_SFT_RDZV_TIMEOUT:-7200}"
  --prime_trainer_model_impl custom
  --prime_trainer_attn "${TRAINER_ATTN}"
  --prime_trainer_dp_replicate "${PRIME_SFT_DP_REPLICATE:-${TRAIN_NODE_COUNT}}"
  --prime_trainer_context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --prime_trainer_cp_style ulysses
  --prime_trainer_fsdp_cpu_offload false
  --prime_trainer_optim_cpu_offload "${PRIME_SFT_OPTIM_CPU_OFFLOAD:-true}"
  --prime_trainer_optimization_dtype bfloat16
  --prime_trainer_reduce_dtype bfloat16
  --prime_trainer_fp8 "${PRIME_SFT_FP8:-true}"
  --prime_trainer_compile "${PRIME_SFT_COMPILE:-true}"
  --prime_checkpoint_output_dir "${CHECKPOINT_ROOT}"
  --prime_checkpoint_interval "${PRIME_SFT_CHECKPOINT_INTERVAL:-100}"
  --prime_checkpoint_keep_last "${PRIME_SFT_CHECKPOINT_KEEP_LAST:-20}"
  --prime_checkpoint_keep_interval "${PRIME_SFT_CHECKPOINT_KEEP_INTERVAL:-0}"
  --prime_checkpoint_weights_only true
  --with_tracking
  --wandb_mode online
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${RUN_NAME}"
)

echo "[prime-sft] run=${RUN_NAME} node=${NODE_LABEL} trainer_rank=${TRAINER_NODE_RANK}/${TRAIN_NODE_COUNT} host=${HOST_NAME} ip=${HOST_IP}"
echo "[prime-sft] master=${MASTER_ADDR}:${MASTER_PORT} model=${MODEL_PATH} dataset=${DATASET_PATH}"
echo "[prime-sft] seq_len=${SEQ_LEN} world_size=${TRAIN_WORLD_SIZE} micro_batch=${MICRO_BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} global_batch=${GLOBAL_BATCH_SIZE} tokens_per_step=$((SEQ_LEN * GLOBAL_BATCH_SIZE))"

if [[ "${PRIME_COMMAND_PREVIEW:-0}" == "1" ]]; then
  printf '[prime-sft] command preview:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
