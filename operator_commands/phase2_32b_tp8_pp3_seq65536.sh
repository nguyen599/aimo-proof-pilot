#!/usr/bin/env bash
set -euo pipefail

export RES_OPTIONS="attempts:5 timeout:2"
export RUNTIME_GIT_RETRY_ATTEMPTS=6
export RUNTIME_GIT_RETRY_BASE_SECONDS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_NVLS_ENABLE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_ibn1,mlx5_ibn2,mlx5_ibn3,mlx5_ibn4,mlx5_ibn5,mlx5_ibn6,mlx5_ibn7,mlx5_ibn8
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OLMO_SELECTIVE_AC_RECOMPUTE_MM_EVERY=0

MASTER_PORT=29560
RUN_NAME="phase2_32b_tp8_pp3_seq65536"
export MASTER_PORT
export OLMO_RUN_DIR_NAME="${RUN_NAME}"

printf '%s host=%s GLOBAL_RANK=%s WORLD_SIZE=%s MASTER_ADDR=%s MASTER_PORT=%s\n' \
  "${RUN_NAME}" \
  "$(hostname)" \
  "${GLOBAL_RANK:-unset}" \
  "${WORLD_SIZE:-unset}" \
  "${MASTER_ADDR:-unset}" \
  "${MASTER_PORT}"

exec python /tmp/submissions-instructions-runtime/src/train.py \
  --fetch-update \
  --runtime-fetch-state-dir "/tmp/train-runtime-fetch-${RUN_NAME}" \
  --submissions-ref main \
  --backend olmo_core_sft \
  --model_path /tmp/olmo3_phase2/model/allenai-Olmo-3.1-32B-Think-ckpt-2k \
  --chat_template_model /tmp/olmo3_phase2/model/allenai-Olmo-3.1-32B-Think-ckpt-2k \
  --dataset_path /groups/gcg51557/experiments/0371_aimo/containers/team1/train_phase2.parquet \
  --output_path "/tmp/olmo3_phase2/outputs/${RUN_NAME}" \
  --logdir "/tmp/olmo3_phase2/logs/${RUN_NAME}" \
  --olmo_core_checkpoint_cache /tmp/olmo3_phase2/cache/olmo32b_2k_core_checkpoint \
  --olmo_core_dataset_cache /tmp/olmo3_phase2/cache/olmo32b_train_phase2_seq65536 \
  --num_gpus 8 \
  --num_nodes "${WORLD_SIZE}" \
  --model_arch olmo3_32b \
  --tensor_parallel_degree 8 \
  --tensor_parallel_async true \
  --pipeline_parallel_degree 3 \
  --pipeline_schedule Interleaved1F1B \
  --pipeline_split_points 11,22,32,43,54 \
  --log_tokenized_sample true \
  --tokenized_sample_max_tokens 100000 \
  --tokenized_sample_max_chars 1000000 \
  --max_seq_length 65536 \
  --per_device_batch_size 3 \
  --rank_microbatch_size_sequences 3 \
  --gradient_accumulation_steps 12 \
  --global_batch_size_tokens 2359296 \
  --max_train_steps 1500 \
  --learning_rate 1e-6 \
  --attn_implementation flash_3 \
  --optimizer te_fused_adamw \
  --optimizer_state_dtype auto \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --activation_memory_budget 0.0 \
  --activation_checkpointing_mode selected_ops \
  --compile_model true \
  --force_compile_model true \
  --float8 false \
  --checkpointing_steps 100 \
  --ephemeral_save_interval 0 \
  --checkpoint_keep_last 5 \
  --hf_checkpoint_upload true \
  --hf_checkpoint_repo nguyen599/olmo3-ckpt-phase2 \
  --hf_checkpoint_upload_workers 20 \
  --dataset_messages_mode auto \
  --dataset_transform_profile olmo \
  --dataset_num_proc 64 \
  --data_loader_num_workers 16 \
  --data_loader_prefetch_factor 3 \
  --offline false \
  --with_tracking \
  --wandb_mode online \
  --wandb_project olmo3-32b-sft \
  --hf_log_upload true \
  --world_size_mode nodes \
  --node_rank "${GLOBAL_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}"
