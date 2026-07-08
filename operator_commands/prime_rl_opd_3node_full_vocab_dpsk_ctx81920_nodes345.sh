#!/usr/bin/env bash
set -euo pipefail

# Standalone 3-node Prime-RL OPD run for the 6-node operator cluster.
# Nodes 0,1,2 are reserved for teammates and exit immediately.
# Node 3: trainer + orchestrator, 8 GPUs
# Node 4: student/policy vLLM rollout, 8 GPUs
# Node 5: DeepSeek-V4-Flash teacher vLLM hidden-state scorer, 8 GPUs

NODE_LABEL="${GLOBAL_RANK:-${NODE_RANK:-${SLURM_NODEID:-${RANK:-none}}}}"
case "${NODE_LABEL}" in
  3) PRIME_COMPONENT_ROLE="trainer_orchestrator" ;;
  4) PRIME_COMPONENT_ROLE="policy_inference" ;;
  5) PRIME_COMPONENT_ROLE="teacher_inference" ;;
  *)
    echo "[prime-opd-3node] node=${NODE_LABEL} host=$(hostname) reserved for another user; skipping."
    exit 0
    ;;
esac

RUN_NAME="${OLMO_RUN_DIR_NAME:-${PRIME_3NODE_RUN_NAME:-prime_rl_opd_3node_full_vocab_dpsk_ctx81920_nodes345}}"
LOCK_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_.-' '_')"

# If an earlier command was stopped while waiting for remote endpoints, its
# bash wrapper can keep holding the old role lock even though no Prime-RL
# process is active. Clean only old role command shells for this same node.
if [[ "${PRIME_3NODE_KILL_STALE_ROLE_SHELLS:-1}" == "1" ]]; then
  mapfile -t STALE_ROLE_SHELL_PIDS < <(
    ps -eo pid=,args= \
      | awk -v self="$$" -v node="${NODE_LABEL}" '
          $1 != self && $0 ~ "/olmo_operator/node" node "/commands/.*/command.sh" { print $1 }
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

ROLE_LOCK="/dev/shm/prime_rl_opd_3node_${LOCK_RUN_NAME}_${NODE_LABEL}_${PRIME_COMPONENT_ROLE}.lock"
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

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-rl-full-vocab-3node}"
export PRIME_RL_PREFILL_HIDDEN_CONCURRENCY="${PRIME_RL_PREFILL_HIDDEN_CONCURRENCY:-1}"
# Keep the container's upgraded vLLM by default. Setting this to 1 is still
# supported for explicit wheel override tests.
export PRIME_RL_RUNTIME_INSTALL_VLLM="${PRIME_RL_RUNTIME_INSTALL_VLLM:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

POLICY_VLLM_EXTRA_DEFAULT='{"kv_cache_dtype":"fp8","block_size":256,"disable_custom_all_reduce":true}'
TEACHER_VLLM_EXTRA_DEFAULT='{"kv_cache_dtype":"fp8","block_size":256,"enable_expert_parallel":true,"linear_backend":"deep_gemm"}'

RENDEZVOUS_DIR="${PRIME_3NODE_RENDEZVOUS_DIR:-/tmp/prime_rl_opd_3node/${RUN_NAME}}"
mkdir -p "${RENDEZVOUS_DIR}"

# Keep transient install/build/cache files in RAM-backed /dev/shm by default on
# the shared operator cluster. The host maps /tmp to the shared /groups
# filesystem, which is inode/quota constrained and can fail pip installs.
TMP_ROOT="${PRIME_3NODE_TMP_ROOT:-/dev/shm/pp3/${RUN_NAME}/${NODE_LABEL}_${PRIME_COMPONENT_ROLE}}"
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
export VLLM_RPC_BASE_PATH="${TMP_ROOT}/rpc"
export FLASHINFER_WORKSPACE_BASE="${TMP_ROOT}/flashinfer"
export FLASHINFER_CUBIN_DIR="${TMP_ROOT}/flashinfer/.cache/flashinfer/cubins"
export DG_JIT_CACHE_DIR="${TMP_ROOT}/deep_gemm"
mkdir -p "${VLLM_RPC_BASE_PATH}"

TEACHER_HIDDEN_BACKEND="${PRIME_OPD_TEACHER_HIDDEN_BACKEND:-hook}"
case "${TEACHER_HIDDEN_BACKEND}" in
  extractor|vllm_extractor|official_extractor)
    export PRIME_RL_HIDDEN_STATE_BACKEND="vllm_extractor"
    TEACHER_HIDDEN_STORAGE="${PRIME_OPD_TEACHER_HIDDEN_STORAGE:-${TMP_ROOT}/hidden_states}"
    mkdir -p "${TEACHER_HIDDEN_STORAGE}"
    TEACHER_VLLM_EXTRA_DEFAULT="$(
      TEACHER_HIDDEN_STORAGE="${TEACHER_HIDDEN_STORAGE}" python - <<'PY'
import json
import os

print(json.dumps({
    "kv_cache_dtype": "fp8",
    "block_size": 256,
    "enable_expert_parallel": True,
    "linear_backend": "deep_gemm",
    "enable_chunked_prefill": False,
    "speculative_config": {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [43],
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
}))
PY
    )"
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

for rank in 3 4 5; do
  wait_for_file "${RENDEZVOUS_DIR}/node${rank}.ip" 900
done

TRAIN_IP="$(cat "${RENDEZVOUS_DIR}/node3.ip")"
POLICY_IP="$(cat "${RENDEZVOUS_DIR}/node4.ip")"
TEACHER_IP="$(cat "${RENDEZVOUS_DIR}/node5.ip")"
POLICY_PORT="${PRIME_POLICY_PORT:-8000}"
TEACHER_PORT="${PRIME_OPD_TEACHER_PORT:-8001}"
POLICY_BASE_URL="http://${POLICY_IP}:${POLICY_PORT}/v1"
TEACHER_BASE_URL="http://${TEACHER_IP}:${TEACHER_PORT}/v1"

echo "[prime-opd-3node] train_ip=${TRAIN_IP}"
echo "[prime-opd-3node] policy_base_url=${POLICY_BASE_URL}"
echo "[prime-opd-3node] teacher_base_url=${TEACHER_BASE_URL}"

if [[ "${PRIME_3NODE_CLEAN_ROLE_PROCS:-1}" == "1" ]]; then
  echo "[prime-opd-3node] cleaning stale Prime-RL/vLLM processes on role node ${NODE_LABEL}"
  pkill -9 -f "[p]ython.*prime_rl" 2>/dev/null || true
  pkill -9 -f "[t]orchrun.*prime_rl" 2>/dev/null || true
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

MODEL_PATH="${PRIME_OPD_MODEL_PATH:-/tmp/models/opd-32b-deploy/opd-32b-deploy}"
TEACHER_MODEL_PATH="${PRIME_OPD_TEACHER_MODEL_PATH:-/tmp/models/dpsk-v4-flash}"
RUNTIME_BASE="${PRIME_3NODE_RUNTIME_BASE:-${TMP_ROOT}/runtime}"
mkdir -p "${RUNTIME_BASE}"
RUNTIME_ROOT="${PRIME_3NODE_RUNTIME_ROOT:-${RUNTIME_BASE}/aimo-proof-pilot}"
OPEN_INSTRUCT_RUNTIME_ROOT="${PRIME_3NODE_OPEN_INSTRUCT_RUNTIME_ROOT:-${RUNTIME_BASE}/open-instruct}"
OLMO_CORE_RUNTIME_ROOT="${PRIME_3NODE_OLMO_CORE_RUNTIME_ROOT:-${RUNTIME_BASE}/OLMo-core}"
RLCSD_RUNTIME_ROOT="${PRIME_3NODE_RLCSD_RUNTIME_ROOT:-${RUNTIME_BASE}/RLCSD}"
VERL_RUNTIME_ROOT="${PRIME_3NODE_VERL_RUNTIME_ROOT:-${RUNTIME_BASE}/VERL}"
PRIME_RL_RUNTIME_ROOT="${PRIME_3NODE_PRIME_RL_RUNTIME_ROOT:-${RUNTIME_BASE}/prime-rl}"
DATASET_PATH="${PRIME_OPD_DATASET_PATH:-${RUNTIME_ROOT}/imo_data_1959_2024.csv}"
VERIFIABLE_DATASET_PATH="${PRIME_OPD_VERIFIABLE_DATASET_PATH:-${RUNTIME_ROOT}/astralbench.csv}"
EVAL_VERIFIABLE_DATASET_PATH="${PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH:-${RUNTIME_ROOT}/aime_2026.csv}"
OUTPUT_ROOT="${PRIME_OPD_OUTPUT_ROOT:-${TMP_ROOT}/output}"
LOG_ROOT="${PRIME_OPD_LOG_ROOT:-${TMP_ROOT}/logs}"
CHECKPOINT_ROOT="${PRIME_OPD_CHECKPOINT_ROOT:-${TMP_ROOT}/checkpoints/${RUN_NAME}_${PRIME_COMPONENT_ROLE}}"

CTX_LEN="${PRIME_OPD_CTX_LEN:-81920}"
VLLM_CTX_LEN="${PRIME_OPD_VLLM_MAX_MODEL_LEN:-98304}"
COMPLETION_TOKENS="${PRIME_OPD_COMPLETION_TOKENS:-81920}"
EVAL_COMPLETION_TOKENS="${PRIME_OPD_EVAL_COMPLETION_TOKENS:-81920}"
BATCHED_TOKENS="${PRIME_OPD_BATCHED_TOKENS:-65536}"
TEACHER_BATCHED_TOKENS="${PRIME_OPD_TEACHER_BATCHED_TOKENS:-8192}"
MAX_STEPS="${MAX_TRAIN_STEPS:-1000}"
BATCH_SIZE="${PRIME_BATCH_SIZE:-2}"
GROUP_SIZE="${PRIME_GROUP_SIZE:-2}"
MAX_INFLIGHT="${PRIME_OPD_MAX_INFLIGHT_ROLLOUTS:-48}"
MAX_OFF_POLICY="${PRIME_MAX_OFF_POLICY_STEPS:-24}"
POLICY_TP="${PRIME_VLLM_TP:-2}"
POLICY_DP="${PRIME_VLLM_DP:-4}"
POLICY_GPU_COUNT=$((POLICY_TP * POLICY_DP))
if (( POLICY_GPU_COUNT < 1 || POLICY_GPU_COUNT > 8 )); then
  echo "[prime-opd-3node] invalid policy topology: PRIME_VLLM_TP=${POLICY_TP} PRIME_VLLM_DP=${POLICY_DP} requires ${POLICY_GPU_COUNT} GPUs on one 8-GPU policy node" >&2
  exit 1
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

COMMON_ARGS=(
  --fetch-update
  --submissions-repo "${SUBMISSIONS_REPO:-https://github.com/nguyen599/aimo-proof-pilot.git}"
  --submissions-ref "${SUBMISSIONS_REF:-main}"
  --prime-rl-ref "${PRIME_RL_REF:-main}"
  --submissions-runtime-dir "${RUNTIME_ROOT}"
  --open-instruct-runtime-dir "${OPEN_INSTRUCT_RUNTIME_ROOT}"
  --olmo-core-runtime-dir "${OLMO_CORE_RUNTIME_ROOT}"
  --rlcsd-runtime-dir "${RLCSD_RUNTIME_ROOT}"
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
  --learning_rate "${PRIME_LEARNING_RATE:-1e-7}"
  --weight_decay "${PRIME_WEIGHT_DECAY:-0.0}"
  --max_grad_norm "${PRIME_MAX_GRAD_NORM:-1.0}"
  --prime_algorithm opd
  --prime_opd_distill_mode full_vocab_hidden
  --prime_opd_full_vocab_teacher_lm_head_path "${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH:-${TEACHER_MODEL_PATH}}"
  --prime_opd_full_vocab_teacher_lm_head_key "${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_KEY:-head.weight}"
  --prime_opd_full_vocab_teacher_hidden_dtype "${PRIME_OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE:-bfloat16}"
  --prime_opd_full_vocab_token_chunk_size "${PRIME_OPD_FULL_VOCAB_TOKEN_CHUNK_SIZE:-16}"
  --prime_opd_full_vocab_vocab_chunk_size "${PRIME_OPD_FULL_VOCAB_VOCAB_CHUNK_SIZE:-8192}"
  --prime_env_id proof-opd-env
  --prime_env_name proof_math
  --prime_proof_dataset_path "${DATASET_PATH}"
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
  --prime_proof_refine_rounds "${PRIME_PROOF_REFINE_ROUNDS:-0}"
  --prime_proof_refine_review_n "${PRIME_PROOF_REFINE_REVIEW_N:-2}"
  --prime_eval_verifiable_dataset_path "${EVAL_VERIFIABLE_DATASET_PATH}"
  --prime_eval_interval "${PRIME_OPD_EVAL_INTERVAL:-50}"
  --prime_eval_num_examples "${PRIME_OPD_EVAL_NUM_EXAMPLES:-8}"
  --prime_eval_group_size "${PRIME_OPD_EVAL_GROUP_SIZE:-1}"
  --prime_eval_max_completion_tokens "${EVAL_COMPLETION_TOKENS}"
  --prime_eval_refine_rounds "${PRIME_OPD_EVAL_REFINE_ROUNDS:-0}"
  --prime_eval_num_verifiers "${PRIME_OPD_EVAL_NUM_VERIFIERS:-1}"
  --prime_eval_refine_review_n "${PRIME_OPD_EVAL_REFINE_REVIEW_N:-1}"
  --prime_eval_answer_column auto
  --prime_batch_size "${BATCH_SIZE}"
  --prime_group_size "${GROUP_SIZE}"
  --prime_max_inflight_rollouts "${MAX_INFLIGHT}"
  --prime_max_off_policy_steps "${MAX_OFF_POLICY}"
  --prime_gpus_per_node 8
  --prime_trainer_model_impl custom
  --prime_trainer_attn olmo3_sink_fa3
  --prime_trainer_context_parallel_size "${PRIME_TRAINER_CP:-1}"
  --prime_trainer_cp_style ulysses
  --prime_trainer_fsdp_cpu_offload false
  --prime_trainer_optim_cpu_offload "${PRIME_TRAINER_OPTIM_CPU_OFFLOAD:-false}"
  --prime_trainer_fp8 "${PRIME_TRAINER_FP8:-true}"
  --prime_weight_broadcast_type "${PRIME_WEIGHT_BROADCAST_TYPE:-filesystem}"
  --prime_weight_broadcast_port "${PRIME_WEIGHT_BROADCAST_PORT:-29501}"
  --prime_weight_broadcast_timeout "${PRIME_WEIGHT_BROADCAST_TIMEOUT:-7200}"
  --prime_weight_broadcast_quantize_in_weight_transfer "${PRIME_WEIGHT_BROADCAST_QUANTIZE:-false}"
  --prime_checkpoint_interval "${PRIME_CHECKPOINT_INTERVAL:-100}"
  --prime_checkpoint_keep_last "${PRIME_CHECKPOINT_KEEP_LAST:-2}"
  --prime_checkpoint_keep_interval "${PRIME_CHECKPOINT_KEEP_INTERVAL:-0}"
  --prime_checkpoint_output_dir "${CHECKPOINT_ROOT}"
  --prime_checkpoint_weights_only "${PRIME_CHECKPOINT_WEIGHTS_ONLY:-true}"
  --prime_checkpoint_wait_for_weights_timeout "${PRIME_CHECKPOINT_WAIT_FOR_WEIGHTS_TIMEOUT:-7200}"
  --prime_skip_model_check true
  --prime_temperature 1.0
  --prime_top_p 0.95
  --with_tracking
  --wandb_mode online
  --wandb_project "${WANDB_PROJECT}"
)

case "${PRIME_COMPONENT_ROLE}" in
  policy_inference)
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_policy_node${NODE_LABEL}"
    # Run TP=2,DP=4 by default: four 2-GPU policy instances on node4 reduce
    # small-model communication cost while still using all 8 GPUs.
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${PRIME_VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
    export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${PRIME_VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-2147483648}"
    exec "${TRAIN_PY_ENV[@]}" /usr/bin/python /app/train.py "${COMMON_ARGS[@]}" \
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
      --prime_vllm_use_deep_gemm "${PRIME_VLLM_USE_DEEP_GEMM:-false}" \
      --prime_vllm_max_num_seqs "${PRIME_OPD_POLICY_MAX_NUM_SEQS:-16}" \
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
    export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${PRIME_OPD_TEACHER_VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
    export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${PRIME_OPD_TEACHER_VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-2147483648}"
    exec "${TRAIN_PY_ENV[@]}" /usr/bin/python /app/train.py "${COMMON_ARGS[@]}" \
      --prime_component teacher_inference \
      --prime_train_gpus 0 \
      --prime_infer_gpus 0 \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_start_teacher true \
      --prime_opd_teacher_gpu_ids "0,1,2,3,4,5,6,7" \
      --prime_opd_teacher_port "${TEACHER_PORT}" \
      --prime_opd_teacher_ready_timeout "${PRIME_OPD_TEACHER_READY_TIMEOUT:-7200}" \
      --prime_opd_teacher_vllm_tensor_parallel_size "${PRIME_OPD_TEACHER_TP:-8}" \
      --prime_opd_teacher_vllm_data_parallel_size "${PRIME_OPD_TEACHER_DP:-1}" \
      --prime_opd_teacher_vllm_max_model_len "${VLLM_CTX_LEN}" \
      --prime_opd_teacher_vllm_dtype bfloat16 \
      --prime_opd_teacher_vllm_enforce_eager "${PRIME_OPD_TEACHER_VLLM_ENFORCE_EAGER:-false}" \
      --prime_opd_teacher_vllm_quantization "${PRIME_OPD_TEACHER_VLLM_QUANTIZATION:-fp8}" \
      --prime_opd_teacher_vllm_gpu_memory_utilization "${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.82}" \
      --prime_opd_teacher_vllm_use_deep_gemm "${PRIME_OPD_TEACHER_USE_DEEP_GEMM:-true}" \
      --prime_opd_teacher_vllm_max_num_seqs "${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-4}" \
      --prime_opd_teacher_vllm_max_num_batched_tokens "${TEACHER_BATCHED_TOKENS}" \
      --prime_opd_teacher_vllm_reasoning_parser deepseek_v4 \
      --prime_opd_teacher_vllm_extra "${PRIME_OPD_TEACHER_VLLM_EXTRA:-${TEACHER_VLLM_EXTRA_DEFAULT}}"
    ;;

  trainer_orchestrator)
    echo "[prime-opd-3node] starting trainer; train_engine_rl will wait for policy and teacher endpoints"
    export OLMO_RUN_DIR_NAME="${RUN_NAME}_trainer_node${NODE_LABEL}"
    exec "${TRAIN_PY_ENV[@]}" /usr/bin/python /app/train.py "${COMMON_ARGS[@]}" \
      --prime_component trainer_orchestrator \
      --prime_train_gpus 8 \
      --prime_infer_gpus 0 \
      --prime_policy_base_url "${POLICY_BASE_URL}" \
      --prime_policy_admin_base_url "${POLICY_BASE_URL}" \
      --prime_policy_dp_rank_count "${PRIME_POLICY_DP_RANK_COUNT:-${POLICY_DP}}" \
      --prime_vllm_tensor_parallel_size "${POLICY_TP}" \
      --prime_vllm_data_parallel_size "${POLICY_DP}" \
      --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
      --prime_opd_start_teacher false \
      --prime_opd_teacher_base_url "${TEACHER_BASE_URL}" \
      --prime_opd_teacher_vllm_tensor_parallel_size "${PRIME_OPD_TEACHER_TP:-8}" \
      --prime_opd_teacher_vllm_data_parallel_size "${PRIME_OPD_TEACHER_DP:-1}"
    ;;
esac
