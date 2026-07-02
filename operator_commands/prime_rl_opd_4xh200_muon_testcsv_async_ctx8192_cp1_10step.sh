#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${OLMO_RUN_DIR_NAME:-prime_rl_opd_teadamw_bf16_async_ctx8192_cp1_testcsv_$(date -u +%Y%m%d_%H%M%S)}"
export OLMO_RUN_DIR_NAME="${RUN_NAME}"

MODEL_PATH="${PRIME_OPD_SMOKE_MODEL_PATH:-/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy}"
TEACHER_MODEL_PATH="${PRIME_OPD_SMOKE_TEACHER_MODEL_PATH:-${MODEL_PATH}}"

export PRIME_RL_DION_MUON_EAGER="${PRIME_RL_DION_MUON_EAGER:-true}"
export PRIME_RL_DION_MUON_MAX_CONCURRENT_TASKS="${PRIME_RL_DION_MUON_MAX_CONCURRENT_TASKS:-1}"

/usr/bin/python /app/train.py \
  --fetch-update \
  --submissions-ref main \
  --prime-rl-ref main \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --runtime-training-deps-dir /tmp/olmo-train-runtime-deps-prime-rl-opd-async \
  --backend prime_rl \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --dataset_path /workspace/submissions-instructions/test.csv \
  --output_path /vol/olmo_train_assets/output/prime_rl_opd_4x_manual \
  --logdir /vol/olmo_train_assets/logs/prime_rl_opd_4x_manual \
  --max_train_steps "${MAX_TRAIN_STEPS:-10}" \
  --max_seq_length 8192 \
  --rollout_max_completion_tokens "${ROLLOUT_MAX_COMPLETION_TOKENS:-8192}" \
  --optimizer "${PRIME_OPTIMIZER:-te_fused_adamw}" \
  --prime_te_adamw_exp_avg_dtype bfloat16 \
  --prime_te_adamw_exp_avg_sq_dtype bfloat16 \
  --prime_te_adamw_master_weight_dtype bfloat16 \
  --prime_te_adamw_master_weights false \
  --learning_rate 1e-6 \
  --weight_decay 0.0 \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --prime_algorithm opd \
  --prime_opd_teacher_model "${TEACHER_MODEL_PATH}" \
  --prime_opd_start_teacher true \
  --prime_opd_teacher_gpu_ids 3 \
  --prime_opd_teacher_port 8001 \
  --prime_opd_teacher_vllm_tensor_parallel_size 1 \
  --prime_opd_teacher_vllm_data_parallel_size 1 \
  --prime_opd_teacher_vllm_max_model_len "${PRIME_OPD_TEACHER_VLLM_MAX_MODEL_LEN:-40960}" \
  --prime_opd_teacher_vllm_dtype bfloat16 \
  --prime_opd_teacher_vllm_enforce_eager true \
  --prime_opd_teacher_vllm_quantization fp8 \
  --prime_opd_teacher_vllm_gpu_memory_utilization 0.95 \
  --prime_opd_teacher_vllm_max_num_seqs 16 \
  --prime_opd_teacher_vllm_max_num_batched_tokens "${PRIME_OPD_TEACHER_VLLM_MAX_NUM_BATCHED_TOKENS:-65536}" \
  --prime_env_id deepseek-math-v2-env \
  --prime_env_name proof_math \
  --prime_proof_dataset_path /workspace/submissions-instructions/test.csv \
  --prime_proof_judge_backend none \
  --prime_proof_max_examples 3 \
  --prime_batch_size 2 \
  --prime_group_size 2 \
  --prime_max_inflight_rollouts 8 \
  --prime_max_off_policy_steps "${PRIME_MAX_OFF_POLICY_STEPS:-8}" \
  --prime_train_gpus 2 \
  --prime_infer_gpus 1 \
  --prime_gpus_per_node 4 \
  --prime_trainer_model_impl custom \
  --prime_trainer_attn olmo3_sink_fa3 \
  --prime_trainer_context_parallel_size 1 \
  --prime_trainer_cp_style ulysses \
  --prime_trainer_fsdp_cpu_offload false \
  --prime_trainer_optim_cpu_offload false \
  --prime_trainer_fp8 false \
  --prime_vllm_tensor_parallel_size 1 \
  --prime_vllm_data_parallel_size 1 \
  --prime_vllm_max_model_len "${PRIME_VLLM_MAX_MODEL_LEN:-40960}" \
  --prime_vllm_dtype bfloat16 \
  --prime_vllm_enforce_eager true \
  --prime_vllm_quantization fp8 \
  --prime_vllm_gpu_memory_utilization 0.95 \
  --prime_vllm_max_num_seqs 16 \
  --prime_vllm_max_num_batched_tokens "${PRIME_VLLM_MAX_NUM_BATCHED_TOKENS:-65536}" \
  --prime_vllm_reasoning_parser deepseek_v4 \
  --prime_skip_model_check true \
  --prime_temperature 0.7 \
  --prime_top_p 0.95 \
  --with_tracking \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-olmo3-prime-rl}"
