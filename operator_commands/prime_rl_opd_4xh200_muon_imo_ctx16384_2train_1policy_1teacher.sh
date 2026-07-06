#!/usr/bin/env bash
set -euo pipefail

# Load repo-root .env (copy .env.example and fill in) so one file configures the
# whole run. Note: sourcing overrides already-exported variables of the same name.
ENV_FILE="${AIMO_ENV_FILE:-$(cd "$(dirname "$0")/.." && pwd)/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  . "${ENV_FILE}"
  set +a
fi

RUN_NAME="${OLMO_RUN_DIR_NAME:-prime_rl_opd_muon_imo_mixed_ctx20480_4gpu_2train_1policy_1teacher_$(date -u +%Y%m%d_%H%M%S)}"
export OLMO_RUN_DIR_NAME="${RUN_NAME}"

MODEL_PATH="${PRIME_OPD_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-v33-s150/opd-32b-v33-s150}"
TEACHER_MODEL_PATH="${PRIME_OPD_TEACHER_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy}"
OPTIMIZER="${PRIME_OPTIMIZER:-te_fused_adamw}"
DATASET_PATH="${PRIME_OPD_DATASET_PATH:-/tmp/aimo-proof-pilot-runtime/imo_data_1959_2024.csv}"
VERIFIABLE_DATASET_PATH="${PRIME_OPD_VERIFIABLE_DATASET_PATH:-/tmp/aimo-proof-pilot-runtime/astralbench.csv}"
VERIFIABLE_FRACTION="${PRIME_OPD_VERIFIABLE_FRACTION:-0.20}"
VERIFIABLE_MIX_SEED="${PRIME_OPD_VERIFIABLE_MIX_SEED:-34521}"
EVAL_VERIFIABLE_DATASET_PATH="${PRIME_OPD_EVAL_VERIFIABLE_DATASET_PATH:-/tmp/aimo-proof-pilot-runtime/aime_2026.csv}"
EVAL_INTERVAL="${PRIME_OPD_EVAL_INTERVAL:-10}"
EVAL_NUM_EXAMPLES="${PRIME_OPD_EVAL_NUM_EXAMPLES:-10}"
EVAL_GROUP_SIZE="${PRIME_OPD_EVAL_GROUP_SIZE:-1}"
EVAL_REFINE_ROUNDS="${PRIME_OPD_EVAL_REFINE_ROUNDS:-0}"
EVAL_NUM_VERIFIERS="${PRIME_OPD_EVAL_NUM_VERIFIERS:-1}"
EVAL_REFINE_REVIEW_N="${PRIME_OPD_EVAL_REFINE_REVIEW_N:-1}"
PROOF_MAX_EXAMPLES="${PRIME_PROOF_MAX_EXAMPLES:-0}"
CTX_LEN="${PRIME_OPD_CTX_LEN:-20480}"
VLLM_CTX_LEN="${PRIME_OPD_VLLM_MAX_MODEL_LEN:-${PRIME_VLLM_MAX_MODEL_LEN:-40960}}"
COMPLETION_TOKENS="${PRIME_OPD_COMPLETION_TOKENS:-20480}"
EVAL_COMPLETION_TOKENS="${PRIME_OPD_EVAL_COMPLETION_TOKENS:-${COMPLETION_TOKENS}}"
BATCHED_TOKENS="${PRIME_OPD_BATCHED_TOKENS:-16384}"
TEACHER_GPU_MEMORY_UTILIZATION="${PRIME_OPD_TEACHER_GPU_MEMORY_UTILIZATION:-0.90}"
TEACHER_MAX_NUM_SEQS="${PRIME_OPD_TEACHER_MAX_NUM_SEQS:-8}"
TEACHER_VLLM_EXTRA="${PRIME_OPD_TEACHER_VLLM_EXTRA:-}"
TEACHER_USE_DEEP_GEMM="${PRIME_OPD_TEACHER_USE_DEEP_GEMM:-}"
POLICY_GPU_MEMORY_UTILIZATION="${PRIME_OPD_POLICY_GPU_MEMORY_UTILIZATION:-0.90}"
POLICY_MAX_NUM_SEQS="${PRIME_OPD_POLICY_MAX_NUM_SEQS:-16}"
POLICY_USE_DEEP_GEMM="${PRIME_OPD_POLICY_USE_DEEP_GEMM:-false}"
POLICY_ENFORCE_EAGER="${PRIME_VLLM_ENFORCE_EAGER:-false}"
MAX_INFLIGHT_ROLLOUTS="${PRIME_OPD_MAX_INFLIGHT_ROLLOUTS:-24}"
PROOF_NUM_VERIFIERS="${PRIME_PROOF_NUM_VERIFIERS:-4}"
PROOF_ENABLE_META_VERIFICATION="${PRIME_PROOF_ENABLE_META_VERIFICATION:-true}"
PROOF_REFINE_ROUNDS="${PRIME_PROOF_REFINE_ROUNDS:-0}"
PROOF_REFINE_REVIEW_N="${PRIME_PROOF_REFINE_REVIEW_N:-2}"
PROOF_REFINE_EARLY_STOP_REWARD="${PRIME_PROOF_REFINE_EARLY_STOP_REWARD:-0.95}"
CHECKPOINT_INTERVAL="${PRIME_CHECKPOINT_INTERVAL:-10}"
CHECKPOINT_KEEP_LAST="${PRIME_CHECKPOINT_KEEP_LAST:-2}"
CHECKPOINT_KEEP_INTERVAL="${PRIME_CHECKPOINT_KEEP_INTERVAL:-0}"
CHECKPOINT_WEIGHTS_ONLY="${PRIME_CHECKPOINT_WEIGHTS_ONLY:-true}"

TRAIN_GPUS="${PRIME_TRAIN_GPUS:-2}"
INFER_GPUS="${PRIME_INFER_GPUS:-1}"
GPUS_PER_NODE="${PRIME_GPUS_PER_NODE:-4}"
TEACHER_GPU_IDS="${PRIME_OPD_TEACHER_GPU_IDS:-3}"
TEACHER_TP="${PRIME_OPD_TEACHER_TP:-1}"
TEACHER_DP="${PRIME_OPD_TEACHER_DP:-1}"
TEACHER_ENFORCE_EAGER="${PRIME_OPD_TEACHER_VLLM_ENFORCE_EAGER:-false}"
POLICY_TP="${PRIME_VLLM_TP:-1}"
POLICY_DP="${PRIME_VLLM_DP:-1}"
TRAINER_CP="${PRIME_TRAINER_CP:-1}"
BATCH_SIZE="${PRIME_BATCH_SIZE:-2}"
GROUP_SIZE="${PRIME_GROUP_SIZE:-2}"
WEIGHT_BROADCAST_TYPE="${PRIME_WEIGHT_BROADCAST_TYPE:-filesystem}"
WEIGHT_BROADCAST_PORT="${PRIME_WEIGHT_BROADCAST_PORT:-29501}"
WEIGHT_BROADCAST_TIMEOUT="${PRIME_WEIGHT_BROADCAST_TIMEOUT:-3600}"
WEIGHT_BROADCAST_QUANTIZE="${PRIME_WEIGHT_BROADCAST_QUANTIZE:-false}"
OPD_DISTILL_MODE="${PRIME_OPD_DISTILL_MODE:-token_logprobs}"
OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH="${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH:-${TEACHER_MODEL_PATH}}"
OPD_FULL_VOCAB_TEACHER_LM_HEAD_KEY="${PRIME_OPD_FULL_VOCAB_TEACHER_LM_HEAD_KEY:-}"
OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE="${PRIME_OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE:-bfloat16}"
OPD_FULL_VOCAB_TOKEN_CHUNK_SIZE="${PRIME_OPD_FULL_VOCAB_TOKEN_CHUNK_SIZE:-64}"
OPD_FULL_VOCAB_VOCAB_CHUNK_SIZE="${PRIME_OPD_FULL_VOCAB_VOCAB_CHUNK_SIZE:-8192}"

if [[ -z "${TEACHER_VLLM_EXTRA}" && "${TEACHER_MODEL_PATH}" == *"/dpsk-v4-flash"* ]]; then
  TEACHER_VLLM_EXTRA='{"kv_cache_dtype":"fp8"}'
fi
if [[ -z "${TEACHER_USE_DEEP_GEMM}" ]]; then
  TEACHER_USE_DEEP_GEMM="false"
fi

echo "[prime-opd] run_name=${RUN_NAME}"
echo "[prime-opd] proof_dataset=${DATASET_PATH}"
echo "[prime-opd] verifiable_dataset=${VERIFIABLE_DATASET_PATH}"
echo "[prime-opd] verifiable_fraction=${VERIFIABLE_FRACTION} mix_seed=${VERIFIABLE_MIX_SEED}"
echo "[prime-opd] eval_verifiable_dataset=${EVAL_VERIFIABLE_DATASET_PATH}"
echo "[prime-opd] eval_interval=${EVAL_INTERVAL} eval_examples=${EVAL_NUM_EXAMPLES} eval_group_size=${EVAL_GROUP_SIZE} eval_refine_rounds=${EVAL_REFINE_ROUNDS}"
echo "[prime-opd] max_examples=${PROOF_MAX_EXAMPLES} ctx=${CTX_LEN} rollout_max_completion=${COMPLETION_TOKENS}"
echo "[prime-opd] gpu_layout train=${TRAIN_GPUS} infer=${INFER_GPUS} teacher=${TEACHER_GPU_IDS} gpus_per_node=${GPUS_PER_NODE}"
echo "[prime-opd] vllm policy_tp=${POLICY_TP} policy_dp=${POLICY_DP} teacher_tp=${TEACHER_TP} teacher_dp=${TEACHER_DP}"
echo "[prime-opd] weight_broadcast type=${WEIGHT_BROADCAST_TYPE} port=${WEIGHT_BROADCAST_PORT} timeout=${WEIGHT_BROADCAST_TIMEOUT} quantize=${WEIGHT_BROADCAST_QUANTIZE}"
echo "[prime-opd] distill_mode=${OPD_DISTILL_MODE} full_vocab_teacher_lm_head=${OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH}"
echo "[prime-opd] optimizer=${OPTIMIZER}"
echo "[prime-opd] vllm_enforce_eager policy=${POLICY_ENFORCE_EAGER} teacher=${TEACHER_ENFORCE_EAGER}"
echo "[prime-opd] teacher_vllm_extra=${TEACHER_VLLM_EXTRA:-<none>}"
echo "[prime-opd] vllm_deep_gemm policy=${POLICY_USE_DEEP_GEMM} teacher=${TEACHER_USE_DEEP_GEMM}"
echo "[prime-opd] proof_num_verifiers=${PROOF_NUM_VERIFIERS} meta=${PROOF_ENABLE_META_VERIFICATION} refine_rounds=${PROOF_REFINE_ROUNDS} refine_review_n=${PROOF_REFINE_REVIEW_N} refine_early_stop_reward=${PROOF_REFINE_EARLY_STOP_REWARD}"
echo "[prime-opd] checkpoint_interval=${CHECKPOINT_INTERVAL} checkpoint_keep_last=${CHECKPOINT_KEEP_LAST} checkpoint_keep_interval=${CHECKPOINT_KEEP_INTERVAL} checkpoint_weights_only=${CHECKPOINT_WEIGHTS_ONLY}"

/usr/bin/python /app/train.py \
  --fetch-update \
  --submissions-ref "${SUBMISSIONS_REF:-main}" \
  --prime-rl-ref "${PRIME_RL_REF:-main}" \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir "/tmp/olmo-train-runtime-deps-prime-rl-opd-${RUN_NAME}" \
  --backend prime_rl \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path /vol/olmo_train_assets/output/prime_rl_opd_4x_real \
  --logdir /vol/olmo_train_assets/logs/prime_rl_opd_4x_real \
  --max_train_steps "${MAX_TRAIN_STEPS:-30}" \
  --max_seq_length "${CTX_LEN}" \
  --rollout_max_completion_tokens "${COMPLETION_TOKENS}" \
  --optimizer "${OPTIMIZER}" \
  --learning_rate 1e-7 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --prime_checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --prime_checkpoint_keep_last "${CHECKPOINT_KEEP_LAST}" \
  --prime_checkpoint_keep_interval "${CHECKPOINT_KEEP_INTERVAL}" \
  --prime_checkpoint_weights_only "${CHECKPOINT_WEIGHTS_ONLY}" \
  --prime_algorithm opd \
  --prime_opd_distill_mode "${OPD_DISTILL_MODE}" \
  --prime_opd_full_vocab_teacher_lm_head_path "${OPD_FULL_VOCAB_TEACHER_LM_HEAD_PATH}" \
  --prime_opd_full_vocab_teacher_lm_head_key "${OPD_FULL_VOCAB_TEACHER_LM_HEAD_KEY}" \
  --prime_opd_full_vocab_teacher_hidden_dtype "${OPD_FULL_VOCAB_TEACHER_HIDDEN_DTYPE}" \
  --prime_opd_full_vocab_token_chunk_size "${OPD_FULL_VOCAB_TOKEN_CHUNK_SIZE}" \
  --prime_opd_full_vocab_vocab_chunk_size "${OPD_FULL_VOCAB_VOCAB_CHUNK_SIZE}" \
  --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
  --prime_opd_start_teacher true \
  --prime_opd_teacher_gpu_ids "${TEACHER_GPU_IDS}" \
  --prime_opd_teacher_port 8001 \
  --prime_opd_teacher_vllm_tensor_parallel_size "${TEACHER_TP}" \
  --prime_opd_teacher_vllm_data_parallel_size "${TEACHER_DP}" \
  --prime_opd_teacher_vllm_max_model_len "${VLLM_CTX_LEN}" \
  --prime_opd_teacher_vllm_dtype bfloat16 \
  --prime_opd_teacher_vllm_enforce_eager "${TEACHER_ENFORCE_EAGER}" \
  --prime_opd_teacher_vllm_gpu_memory_utilization "${TEACHER_GPU_MEMORY_UTILIZATION}" \
  --prime_opd_teacher_vllm_use_deep_gemm "${TEACHER_USE_DEEP_GEMM}" \
  --prime_opd_teacher_vllm_max_num_seqs "${TEACHER_MAX_NUM_SEQS}" \
  --prime_opd_teacher_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
  --prime_opd_teacher_vllm_extra "${TEACHER_VLLM_EXTRA}" \
  --prime_env_id proof-opd-env \
  --prime_env_name proof_math \
  --prime_proof_dataset_path "${DATASET_PATH}" \
  --prime_proof_verifiable_dataset_path "${VERIFIABLE_DATASET_PATH}" \
  --prime_proof_verifiable_fraction "${VERIFIABLE_FRACTION}" \
  --prime_proof_verifiable_answer_column auto \
  --prime_proof_mix_seed "${VERIFIABLE_MIX_SEED}" \
  --prime_proof_problem_column auto \
  --prime_proof_solution_column auto \
  --prime_proof_judge_backend none \
  --prime_proof_max_examples "${PROOF_MAX_EXAMPLES}" \
  --prime_proof_enable_meta_verification "${PROOF_ENABLE_META_VERIFICATION}" \
  --prime_proof_num_verifiers "${PROOF_NUM_VERIFIERS}" \
  --prime_proof_refine_rounds "${PROOF_REFINE_ROUNDS}" \
  --prime_proof_refine_review_n "${PROOF_REFINE_REVIEW_N}" \
  --prime_proof_refine_early_stop_reward "${PROOF_REFINE_EARLY_STOP_REWARD}" \
  --prime_eval_verifiable_dataset_path "${EVAL_VERIFIABLE_DATASET_PATH}" \
  --prime_eval_interval "${EVAL_INTERVAL}" \
  --prime_eval_num_examples "${EVAL_NUM_EXAMPLES}" \
  --prime_eval_group_size "${EVAL_GROUP_SIZE}" \
  --prime_eval_max_completion_tokens "${EVAL_COMPLETION_TOKENS}" \
  --prime_eval_refine_rounds "${EVAL_REFINE_ROUNDS}" \
  --prime_eval_num_verifiers "${EVAL_NUM_VERIFIERS}" \
  --prime_eval_refine_review_n "${EVAL_REFINE_REVIEW_N}" \
  --prime_eval_answer_column auto \
  --prime_batch_size "${BATCH_SIZE}" \
  --prime_group_size "${GROUP_SIZE}" \
  --prime_max_inflight_rollouts "${MAX_INFLIGHT_ROLLOUTS}" \
  --prime_train_gpus "${TRAIN_GPUS}" \
  --prime_infer_gpus "${INFER_GPUS}" \
  --prime_gpus_per_node "${GPUS_PER_NODE}" \
  --prime_trainer_model_impl custom \
  --prime_trainer_attn olmo3_sink_fa3 \
  --prime_trainer_context_parallel_size "${TRAINER_CP}" \
  --prime_trainer_cp_style ulysses \
  --prime_trainer_fsdp_cpu_offload false \
  --prime_trainer_optim_cpu_offload "${PRIME_TRAINER_OPTIM_CPU_OFFLOAD:-false}" \
  --prime_trainer_fp8 "${PRIME_TRAINER_FP8:-true}" \
  --prime_weight_broadcast_type "${WEIGHT_BROADCAST_TYPE}" \
  --prime_weight_broadcast_port "${WEIGHT_BROADCAST_PORT}" \
  --prime_weight_broadcast_timeout "${WEIGHT_BROADCAST_TIMEOUT}" \
  --prime_weight_broadcast_quantize_in_weight_transfer "${WEIGHT_BROADCAST_QUANTIZE}" \
  --prime_vllm_tensor_parallel_size "${POLICY_TP}" \
  --prime_vllm_data_parallel_size "${POLICY_DP}" \
  --prime_vllm_max_model_len "${VLLM_CTX_LEN}" \
  --prime_vllm_dtype bfloat16 \
  --prime_vllm_enforce_eager "${POLICY_ENFORCE_EAGER}" \
  --prime_vllm_quantization fp8 \
  --prime_vllm_gpu_memory_utilization "${POLICY_GPU_MEMORY_UTILIZATION}" \
  --prime_vllm_use_deep_gemm "${POLICY_USE_DEEP_GEMM}" \
  --prime_vllm_max_num_seqs "${POLICY_MAX_NUM_SEQS}" \
  --prime_vllm_max_num_batched_tokens "${BATCHED_TOKENS}" \
  --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true \
  --prime_temperature 0.7 \
  --prime_top_p 0.95 \
  --with_tracking \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-olmo3-prime-rl}"
