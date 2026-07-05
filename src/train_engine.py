import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import netrc
import os
import re
import shutil
import site
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from train_logging import (
    PeriodicHFLogUploader,
    collect_run_environment_info,
    configure_logging,
    log_dependency_versions,
    package_versions,
    hf_log_token,
    hf_transfer_heartbeat,
    primary_wandb_log_process,
    quiet_hf_transfer,
    retry_hf_operation,
    upload_logdir_to_hf,
    wandb_rank_metadata,
    wandb_rank_suffix,
)
from train_memory import CudaMemoryHistoryRecorder, add_cuda_memory_args
from train_operator import add_operator_args, operator_run_name, run_operator_mode
from train_utils import path_name_token, sanitize_slug_part, truncate_slug


SUPPORTED_DATA_SUFFIXES = {".json", ".jsonl", ".parquet"}
DEFAULT_QWEN_CHAT_TEMPLATE_MODEL = "Qwen/Qwen3.6-27B"
OLMO_CORE_DATASET_CACHE_VERSION = "qwen_no_tools_system_v1"
OLMO_CORE_DATASET_CACHE_VERSIONS = {
    "qwen": OLMO_CORE_DATASET_CACHE_VERSION,
    "olmo": "olmo_chat_template_v1",
}
DEFAULT_MODEL_HF_REPOS = {
    "olmo3_7b": "allenai/Olmo-3.1-7B-RL-Zero-Math",
    "olmo3_32b": "allenai/Olmo-3.1-32B-Think",
}
GRPO_BACKENDS = {"grpo_fast", "grpo_olmo_core"}
VERL_BACKENDS = {"verl_rlcsd"}
RAY_RL_BACKENDS = GRPO_BACKENDS | VERL_BACKENDS
RLCSD_PROOF_PROMPT_VERSION = "run_py_deepseek_generation_v1"
RLCSD_TEACHER_METHODS = {"rlcsd", "rlsd_ectr", "opsd_ectr", "rlsd", "opsd", "sdpo", "srpo"}
RLCSD_SIMPLE_ASYNC_METHODS = {"grpo", "cispo"}
DEFAULT_MODEL_N_LAYERS = {
    "olmo3_7b": 32,
    "olmo3_32b": 64,
}
TORCHRUN_CHILD_ENV_KEYS = (
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)
INTERNAL_TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)
HF_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
HF_SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
SWEEP_VALUE_FLAGS = {
    "--learning_rates",
    "--optimizers",
    "--sweep_name",
}
SWEEP_BOOL_FLAGS = {"--sweep_dry_run", "--sweep_continue_on_failure"}
CHILD_OVERRIDE_VALUE_FLAGS = SWEEP_VALUE_FLAGS | {
    "--learning_rate",
    "--optimizer",
    "--output_path",
    "--logdir",
    "--cache_dir",
    "--olmo_core_checkpoint_cache",
    "--olmo_core_dataset_cache",
}
CHILD_OVERRIDE_BOOL_FLAGS = SWEEP_BOOL_FLAGS
GRPO_WRAPPER_VALUE_FLAGS = {
    "--backend",
    "--internal_backend",
    "--model_path",
    "--dataset_path",
    "--output_path",
    "--logdir",
    "--num_gpus",
    "--world_size_mode",
    "--node_rank",
    "--master_addr",
    "--master_port",
    "--cache_dir",
    "--olmo_core_checkpoint_cache",
    "--olmo_core_dataset_cache",
    "--global_batch_size_tokens",
    "--global_batch_size_sequences",
    "--convert_validation",
    "--dataset_messages_mode",
    "--dataset_num_proc",
    "--dataset_map_batch_size",
    "--dataset_batched_tokenization",
    "--dataset_transform_profile",
    "--dataset_backend",
    "--chat_template_model",
    "--dataset_weight",
    "--dataset_split",
    "--log_tokenized_sample",
    "--tokenized_sample_max_tokens",
    "--tokenized_sample_max_chars",
    "--open_instruct_dir",
    "--wandb_mode",
    "--hf_log_upload",
    "--hf_log_repo",
    "--hf_log_path_prefix",
    "--hf_log_upload_interval_seconds",
    "--hf_dependency_upgrade",
    "--hf_dependency_upgrade_packages",
    "--checkpointing_steps",
    "--ephemeral_save_interval",
    "--checkpoint_keep_last",
    "--hf_checkpoint_upload",
    "--hf_checkpoint_repo",
    "--hf_checkpoint_path_prefix",
    "--hf_checkpoint_keep_last",
    "--hf_checkpoint_upload_workers",
    "--hf_checkpoint_upload_report_interval_seconds",
    "--hf_checkpoint_convert",
    "--hf_checkpoint_convert_tokenizer",
    "--hf_checkpoint_convert_device",
    "--hf_checkpoint_converted_suffix",
    "--hf_checkpoint_convert_keep_local",
    "--offline",
    "--collect_env_info",
    "--env_info_command_timeout",
    "--learning_rates",
    "--optimizers",
    "--sweep_name",
    "--rl_preset",
    "--grpo_ray_port",
    "--grpo_ray_dashboard_port",
    "--grpo_ray_start_timeout",
    "--grpo_ray_num_cpus",
    "--grpo_ray_worker_start_retries",
    "--grpo_ray_temp_dir",
    "--grpo_worker_poll_interval",
    "--grpo_stop_ray_on_exit",
    "--grpo_judge_preflight",
    "--grpo_judge_preflight_prompt",
    "--grpo_deepspeed_stage",
    "--grpo_deepspeed_zpg",
    "--grpo_deepspeed_offload_param",
    "--grpo_deepspeed_offload_optimizer",
    "--grpo_sequence_parallel_size",
    "--max_seq_length",
    "--max_train_steps",
    "--num_train_epochs",
    "--per_device_batch_size",
    "--rank_microbatch_size_sequences",
    "--gradient_accumulation_steps",
    "--activation_memory_budget",
    "--compile_model",
    "--optimizer_state_dtype",
    "--optimizer",
    "--lm_loss_implementation",
    "--data_loader_num_workers",
    "--data_loader_prefetch_factor",
    "--data_loader_num_threads",
    "--adamw_8bit_block_size",
    "--float8",
    "--float8_recipe",
    "--te_feedforward",
    "--te_feedforward_glu",
    "--te_layernorm",
    "--liger_layernorm",
    "--liger_megatron_layernorm",
    "--quack_layernorm",
    "--feed_forward_chunk_size_tokens",
    "--feed_forward_memory_profile",
    "--feed_forward_memory_profile_ranks",
    "--feed_forward_memory_profile_max_calls",
    "--feed_forward_memory_profile_sync",
    "--feed_forward_memory_profile_allow_compile",
    "--cuda_memory_history",
    "--cuda_memory_history_max_entries",
    "--cuda_memory_history_top_allocations",
    "--cuda_memory_history_dump_pickle",
    "--tokenizer_path",
    "--model_hf_repo",
    "--config_name",
    "--rope_scaling_factor",
    "--rope_scaling_old_context_len",
    "--rope_scaling_beta_fast",
    "--rope_scaling_beta_slow",
    "--model_arch",
    "--tensor_parallel_degree",
    "--tensor_parallel_async",
    "--context_parallel_degree",
    "--context_parallel_style",
    "--context_parallel_head_stride",
    "--pipeline_parallel_degree",
    "--pipeline_schedule",
    "--pipeline_split_points",
    "--attn_implementation",
    "--activation_checkpointing_mode",
    "--activation_checkpoint_modules",
    "--activation_checkpoint_block_interval",
    "--force_compile_model",
}
GRPO_WRAPPER_BOOL_FLAGS = {
    "--dry_run_launch",
    "--skip_cache_prepare",
    "--sweep_dry_run",
    "--sweep_continue_on_failure",
    "--allow_unsafe_float8_tp",
    "--adamw_8bit_bf16_stochastic_round",
    "--disable_checkpoints",
}
VALID_OPTIMIZERS = {
    "adamw",
    "skip_step_adamw",
    "adamw_8bit",
    "adamw_8bits",
    "torchao_adamw_8bit",
    "te_fused_adamw",
}
DEFAULT_RUNTIME_CACHE_ROOT = "/tmp/olmo_train_runtime_cache"
DEFAULT_OUTPUT_ROOT = "/tmp/olmo_train_outputs"
DEFAULT_LOG_ROOT = "/tmp/olmo_train_logs"
TRAIN_ENGINE_STARTED_AT = time.time()
DEFAULT_HF_DEPENDENCY_UPGRADE_PACKAGES = "huggingface_hub>=1.18.0,typer>=0.21.1,click>=8.4.1"
MIN_HF_DOWNLOAD_DEPENDENCIES = {
    "huggingface_hub": "1.18.0",
    "typer": "0.21.1",
    "click": "8.4.1",
}
LOG_LINE_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}\s+)?\d{2}:\d{2}:\d{2}(?:,\d{3})?\s+"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+(?P<message>.*)$"
)
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
@dataclass(frozen=True)
class SweepTrial:
    index: int
    learning_rate: float
    learning_rate_token: str
    optimizer: str
    slug: str


def optional_int_arg(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "auto", ""}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer or 'none', got {value!r}") from exc


def pipeline_split_points_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "auto"}:
        return ""
    try:
        split_points = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integer split points, got {value!r}"
        ) from exc
    if not split_points:
        return ""
    if any(point < 0 for point in split_points):
        raise argparse.ArgumentTypeError("--pipeline_split_points values must be >= 0.")
    if split_points != sorted(set(split_points)):
        raise argparse.ArgumentTypeError("--pipeline_split_points must be unique and increasing.")
    return ",".join(str(point) for point in split_points)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline OLMo SFT/RL training entrypoint.",
        allow_abbrev=False,
    )
    parser.add_argument("--model_path", default=None, help="Local base model path.")
    parser.add_argument("--dataset_path", default=None, help="Local raw SFT/RL dataset file or directory.")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Directory for trained checkpoints. Defaults to /tmp/olmo_train_outputs/<auto-run-name>.",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Directory for training logs. Defaults to /tmp/olmo_train_logs/<auto-run-name>.",
    )
    parser.add_argument(
        "--run_dir_mode",
        default="auto",
        choices=("auto", "none"),
        help=(
            "auto creates a fresh run subdirectory under output_path/logdir for each top-level run. "
            "Use none to write exactly to the provided paths."
        ),
    )
    parser.add_argument(
        "--run_dir_name",
        default=None,
        help=(
            "Optional run subdirectory name used when --run_dir_mode auto. "
            "Defaults to OLMO_RUN_DIR_NAME or a coordinated UTC timestamp."
        ),
    )
    parser.add_argument(
        "--backend",
        default="open_instruct_wrapper",
        choices=("open_instruct_wrapper", "olmo_core_sft", "grpo_fast", "grpo_olmo_core", "verl_rlcsd"),
        help="Training implementation to run.",
    )
    parser.add_argument(
        "--internal_backend",
        default=None,
        choices=("olmo_core_sft_worker",),
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--num_gpus", type=int, default=0, help="GPUs per node. 0 auto-detects.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=0,
        help="Number of training nodes. 0 auto-detects from PBS WORLD_SIZE as a node count or defaults to 1.",
    )
    parser.add_argument(
        "--world_size_mode",
        default="nodes",
        choices=("auto", "nodes", "processes"),
        help=(
            "How to interpret PBS WORLD_SIZE when --num_nodes is 0. "
            "nodes means one rank per Singularity container; processes means total torch ranks; "
            "auto keeps compatibility with process-style Modal simulations."
        ),
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=None,
        help="Rank of this node. Defaults to PBS GLOBAL_RANK, then NODE_RANK.",
    )
    parser.add_argument("--master_addr", default=None, help="Master node hostname/IP for multinode training.")
    parser.add_argument("--master_port", type=int, default=None, help="Master port for torchrun.")

    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Base learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--per_device_batch_size",
        "--per_device_train_batch_size",
        dest="per_device_batch_size",
        type=int,
        default=1,
        help=(
            "Legacy per-device batch argument. For OLMo-core direct training this is the requested "
            "rank-local model microbatch in packed sequences; TP/CP/PP ranks cooperate on those "
            "same sequences rather than each receiving independent examples."
        ),
    )
    parser.add_argument(
        "--rank_microbatch_size_sequences",
        type=int,
        default=0,
        help=(
            "OLMo-core direct backend only: force the rank-local forward microbatch size in packed sequences. "
            "0 keeps the SFT script's automatic planner. Set this to --per_device_batch_size when you want "
            "VRAM to scale with that value."
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10,
        help="Short smoke-test limit by default. Set <=0 to train for full epochs.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", default="cosine", choices=("linear", "cosine", "constant"))
    parser.add_argument("--activation_memory_budget", type=float, default=0.3)
    parser.add_argument("--compile_model", default="true", choices=("true", "false"))
    parser.add_argument(
        "--force_compile_model",
        default="false",
        choices=("true", "false"),
        help=(
            "Keep torch.compile enabled even for topologies that train.py normally guards off, "
            "such as combined TP+PP. Use only for explicit experiments."
        ),
    )
    parser.add_argument(
        "--activation_checkpointing_mode",
        default="auto",
        choices=("auto", "none", "budget", "full", "selected_blocks", "selected_modules", "selected_ops"),
        help=(
            "OLMo-core activation checkpointing strategy. auto preserves the existing behavior: "
            "budget AC when compile is enabled, otherwise feed-forward selected_modules AC. "
            "Use full to checkpoint every transformer block without requiring torch.compile."
        ),
    )
    parser.add_argument(
        "--activation_checkpoint_modules",
        default="blocks.*.feed_forward",
        help="Comma-separated module globs for --activation_checkpointing_mode selected_modules.",
    )
    parser.add_argument(
        "--activation_checkpoint_block_interval",
        type=int,
        default=1,
        help="Block interval for --activation_checkpointing_mode selected_blocks.",
    )
    parser.add_argument(
        "--optimizer_state_dtype",
        default="auto",
        choices=("auto", "float32", "bfloat16"),
        help=(
            "Optimizer moment dtype for OLMo-core. auto uses bfloat16 for single-GPU long-context "
            "smoke runs and leaves the framework default for multi-GPU runs."
        ),
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=(
            "adamw",
            "skip_step_adamw",
            "adamw_8bit",
            "adamw_8bits",
            "torchao_adamw_8bit",
            "te_fused_adamw",
        ),
        help=(
            "Optimizer for OLMo-core SFT. adamw_8bit uses torchao.optim.AdamW8bit; "
            "te_fused_adamw uses Transformer Engine FusedAdam with AdamW mode."
        ),
    )
    parser.add_argument(
        "--lm_loss_implementation",
        default="fused_linear",
        choices=("default", "fused_linear"),
        help=(
            "OLMo-core LM-head loss implementation. fused_linear uses Liger's low-memory "
            "linear + cross-entropy kernel and avoids materializing full logits."
        ),
    )
    parser.add_argument(
        "--data_loader_num_workers",
        type=int,
        default=16,
        help="OLMo-core NumPy data loader worker processes per training rank.",
    )
    parser.add_argument(
        "--data_loader_prefetch_factor",
        type=optional_int_arg,
        default=3,
        help="Batches to prefetch per OLMo-core data loader worker; use 'none' to leave unset.",
    )
    parser.add_argument(
        "--data_loader_num_threads",
        type=optional_int_arg,
        default=None,
        help=(
            "Threads used inside each OLMo-core data loader path; use 'none' for framework default. "
            "Usually keep this unset when --data_loader_num_workers is high."
        ),
    )
    parser.add_argument(
        "--adamw_8bit_block_size",
        type=int,
        default=256,
        help="Block size passed to torchao.optim.AdamW8bit when --optimizer adamw_8bit.",
    )
    parser.add_argument(
        "--adamw_8bit_bf16_stochastic_round",
        action="store_true",
        help="Enable torchao AdamW8bit bfloat16 stochastic rounding.",
    )
    parser.add_argument(
        "--float8",
        default="false",
        choices=("true", "false"),
        help="Enable OLMo-core torchao Float8Linear training for direct backend experiments.",
    )
    parser.add_argument(
        "--float8_recipe",
        default="recommended",
        choices=(
            "recommended",
            "tensorwise",
            "rowwise",
            "rowwise_with_gw_hp",
            "mxfp8_cublas_rceil",
            "blockwise",
            "blockwise_triton",
        ),
        help=(
            "Float8 recipe for --float8 true. recommended uses AOFloat8LinearConfig.recommended(); "
            "mxfp8_cublas_rceil uses torchao MXFP8 when the installed torchao/GPU support it; "
            "blockwise uses torchao's SM90 Float8BlockwiseLinear training prototype."
        ),
    )
    parser.add_argument(
        "--te_feedforward",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable OLMo-core Transformer Engine Linear modules for dense feed-forward layers. "
            "Current safe mode supports TP=1 only; TE attention can still be used with TP>1."
        ),
    )
    parser.add_argument(
        "--te_feedforward_glu",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable Transformer Engine fused GLU activation for OLMo-core dense feed-forward layers. "
            "This keeps OLMo linears/checkpoints and is intended for TP>1 experiments."
        ),
    )
    parser.add_argument(
        "--te_layernorm",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable Transformer Engine RMSNorm kernels inside OLMo-core RMSNorm/QwenRMSNorm modules. "
            "This preserves OLMo checkpoint parameter names."
        ),
    )
    parser.add_argument(
        "--liger_layernorm",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable Liger Triton RMSNorm kernels inside OLMo-core RMSNorm/QwenRMSNorm modules. "
            "This preserves OLMo checkpoint parameter names and is intended for speed probes."
        ),
    )
    parser.add_argument(
        "--liger_megatron_layernorm",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable Liger's Megatron-Core RMSNorm wrapper inside OLMo-core RMSNorm/QwenRMSNorm "
            "modules. This preserves OLMo checkpoint parameter names and tests the upstream "
            "Megatron integration path."
        ),
    )
    parser.add_argument(
        "--quack_layernorm",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable Quack CuTe RMSNorm kernels inside OLMo-core RMSNorm/QwenRMSNorm modules. "
            "This preserves OLMo checkpoint parameter names and is intended for speed probes."
        ),
    )
    parser.add_argument(
        "--feed_forward_chunk_size_tokens",
        type=int,
        default=0,
        help=(
            "Chunk OLMo-core feed-forward projections over this many local tokens. "
            "Use 0 to disable. This reduces w1/w3 activation peak memory without changing checkpoints."
        ),
    )
    parser.add_argument(
        "--feed_forward_memory_profile",
        default="false",
        choices=("true", "false"),
        help="Log CUDA memory around each OLMo-core feed-forward projection.",
    )
    parser.add_argument(
        "--feed_forward_memory_profile_ranks",
        default="all",
        help="Ranks to log for --feed_forward_memory_profile, e.g. all, 0, rank:0, local:0.",
    )
    parser.add_argument(
        "--feed_forward_memory_profile_max_calls",
        type=int,
        default=1,
        help="Maximum feed-forward calls to profile per module; use 0 for unlimited.",
    )
    parser.add_argument(
        "--feed_forward_memory_profile_sync",
        default="false",
        choices=("true", "false"),
        help="Synchronize CUDA before each feed-forward memory profile log point.",
    )
    parser.add_argument(
        "--feed_forward_memory_profile_allow_compile",
        default="false",
        choices=("true", "false"),
        help=(
            "Allow feed-forward memory profiling calls inside torch.compile graphs. "
            "Default false skips those calls while Dynamo is compiling to avoid graph breaks and OOMs."
        ),
    )
    parser.add_argument(
        "--torch_profiler",
        default="false",
        choices=("true", "false"),
        help="Enable OLMo-core torch.profiler traces for short speed/debug runs.",
    )
    parser.add_argument(
        "--torch_profiler_skip_first",
        type=int,
        default=1,
        help="Profiler steps to skip before the wait/warmup/active schedule starts.",
    )
    parser.add_argument(
        "--torch_profiler_wait",
        type=int,
        default=1,
        help="Profiler idle steps before warmup.",
    )
    parser.add_argument(
        "--torch_profiler_warmup",
        type=int,
        default=1,
        help="Profiler warmup steps whose traces are discarded.",
    )
    parser.add_argument(
        "--torch_profiler_active",
        type=int,
        default=2,
        help="Profiler active steps to record per cycle.",
    )
    parser.add_argument(
        "--torch_profiler_repeat",
        type=int,
        default=1,
        help="Profiler schedule repeat count.",
    )
    parser.add_argument(
        "--torch_profiler_ranks",
        default="pp",
        help=(
            "Ranks to profile: pp, tp, cp, dp, ep, all, none/rank0, or empty. "
            "pp captures one local rank per pipeline stage."
        ),
    )
    parser.add_argument(
        "--torch_profiler_with_stack",
        default="false",
        choices=("true", "false"),
        help="Record Python stack traces in torch.profiler output. This can be expensive.",
    )
    parser.add_argument(
        "--torch_profiler_profile_memory",
        default="false",
        choices=("true", "false"),
        help="Record tensor allocation/deallocation events in torch.profiler output.",
    )
    parser.add_argument(
        "--torch_profiler_cuda_sync_events",
        default="false",
        choices=("true", "false"),
        help="Enable CUDA sync event recording for critical-path analysis.",
    )
    add_cuda_memory_args(parser)
    parser.add_argument(
        "--allow_unsafe_float8_tp",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Float8Linear with tensor parallelism is allowed; "
            "keep this only for older command lines."
        ),
    )

    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path. Defaults to model_path.")
    parser.add_argument(
        "--model_hf_repo",
        default=None,
        help=(
            "Optional HF repo ID used to auto-download missing model weights. "
            "Defaults are selected from DEFAULT_MODEL_HF_REPOS by --model_arch."
        ),
    )
    parser.add_argument("--config_name", default=None, help="OLMo-core config name, inferred when possible.")
    parser.add_argument(
        "--rope_scaling_factor",
        default="auto",
        help=(
            "YaRN RoPE scaling factor. auto keeps the checkpoint factor and raises it when "
            "--max_seq_length exceeds old_context_len*factor; none disables train.py's override."
        ),
    )
    parser.add_argument("--rope_scaling_old_context_len", type=int, default=8192)
    parser.add_argument("--rope_scaling_beta_fast", type=float, default=32.0)
    parser.add_argument("--rope_scaling_beta_slow", type=float, default=1.0)
    parser.add_argument(
        "--model_arch",
        default="auto",
        choices=("auto", "olmo3_7b", "olmo3_32b"),
        help="Model architecture for OLMo-core direct SFT.",
    )
    parser.add_argument(
        "--tensor_parallel_degree",
        type=int,
        default=0,
        help="Tensor parallel degree for OLMo-core direct SFT. 0 chooses 8 for 32B when available, otherwise 1.",
    )
    parser.add_argument(
        "--tensor_parallel_async",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable OLMo-core/PyTorch experimental async tensor parallelism. This enables "
            "Inductor micro-pipeline TP and symmetric-memory collectives and requires TP>1 "
            "with effective torch.compile enabled."
        ),
    )
    parser.add_argument(
        "--context_parallel_degree",
        type=int,
        default=1,
        help="Context parallel degree for OLMo-core direct SFT. Use 1 unless long context OOM requires CP.",
    )
    parser.add_argument(
        "--context_parallel_style",
        default="ring",
        choices=("ring", "llama3", "zig_zag", "zigzag", "ulysses"),
        help=(
            "Context parallel attention style when --context_parallel_degree > 1. "
            "ring is an alias for the current llama3 ring-attention style; ulysses uses all-to-all CP."
        ),
    )
    parser.add_argument(
        "--context_parallel_head_stride",
        type=int,
        default=4,
        help="Head stride for ring/llama3/zig_zag context parallelism. Ignored for --context_parallel_style ulysses.",
    )
    parser.add_argument(
        "--pipeline_parallel_degree",
        type=int,
        default=1,
        help="Pipeline parallel degree for OLMo-core direct SFT. Use 3 for the target 3-node 32B run.",
    )
    parser.add_argument(
        "--pipeline_schedule",
        default="Interleaved1F1B",
        choices=(
            "1F1B",
            "Interleaved1F1B",
            "GPipe",
            "LoopedBFS",
            "InterleavedZeroBubble",
            "ZBVZeroBubble",
        ),
        help="Pipeline schedule for OLMo-core direct SFT when --pipeline_parallel_degree > 1.",
    )
    parser.add_argument(
        "--pipeline_split_points",
        type=pipeline_split_points_arg,
        default="",
        help=(
            "Comma-separated OLMo-core transformer block split points for pipeline stages. "
            "Empty auto-calculates from --model_arch, --pipeline_schedule, and --pipeline_parallel_degree. "
            "Example: 20,42 for PP=3 standard stages, or 10,21,32,42,53 for six interleaved stages."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help=(
            "Attention backend override. OLMo-core values: torch, flash_2, flash_3, flash_4, te. "
            "Aliases accepted: sdpa/eager -> torch, flash_attention_2/3/4 -> flash_2/3/4, "
            "transformer_engine/te_attn -> te. "
            "Unset or auto keeps framework auto/default behavior."
        ),
    )
    parser.add_argument(
        "--attention_sink",
        default="false",
        choices=("true", "false"),
        help=(
            "Enable OLMo-core trainable per-head attention sinks. "
            "Supported initially with torch and flash_3 attention backends."
        ),
    )
    parser.add_argument(
        "--attention_sink_init_value",
        type=float,
        default=-10.0,
        help="Initial logit value for trainable attention sinks when --attention_sink true.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Runtime cache root. Defaults to /tmp/olmo_train_runtime_cache.",
    )
    parser.add_argument(
        "--olmo_core_checkpoint_cache",
        default=None,
        help="Cache for HF-to-OLMo-core checkpoint conversion. Defaults under /tmp/olmo_train_runtime_cache.",
    )
    parser.add_argument(
        "--olmo_core_dataset_cache",
        default=None,
        help="Cache for raw-to-OLMo-core dataset conversion. Defaults under /tmp/olmo_train_runtime_cache.",
    )
    parser.add_argument(
        "--global_batch_size_tokens",
        type=int,
        default=0,
        help=(
            "Global optimizer batch in tokens for OLMo-core direct SFT. TP, CP, and PP shard "
            "one model replica and do not multiply this value. Mutually exclusive with "
            "--global_batch_size_sequences; 0 derives it from rank microbatch, DP, and accumulation."
        ),
    )
    parser.add_argument(
        "--global_batch_size_sequences",
        type=int,
        default=0,
        help=(
            "Global optimizer batch in packed sequences for OLMo-core direct SFT. It is converted "
            "to tokens as this value times --max_seq_length. Mutually exclusive with "
            "--global_batch_size_tokens."
        ),
    )
    parser.add_argument(
        "--convert_validation",
        default="false",
        choices=("true", "false"),
        help="Run expensive HF-to-OLMo-core conversion validation.",
    )
    parser.add_argument(
        "--chat_template_name",
        default=None,
        help="Optional Open-Instruct chat template name for OLMo-core dataset conversion.",
    )
    parser.add_argument(
        "--chat_template_model",
        default=None,
        help=(
            "Tokenizer/model ID to copy only the chat_template from. Defaults to the selected tokenizer/model "
            "for OLMo-profile conversion and to Qwen/Qwen3.6-27B for the Qwen tool-profile conversion. "
            "Use 'none' or 'tokenizer-default' to force the selected tokenizer/model template."
        ),
    )
    parser.add_argument(
        "--dataset_messages_mode",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset_num_proc",
        type=int,
        default=0,
        help="Worker count for Open-Instruct dataset conversion. 0 uses the framework default; 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--dataset_map_batch_size",
        type=int,
        default=1000,
        help=(
            "Batch size for Open-Instruct dataset.map tokenization/statistics passes. "
            "Keep small for long contexts because each row can hold three long token arrays."
        ),
    )
    parser.add_argument(
        "--dataset_batched_tokenization",
        default="true",
        choices=("true", "false"),
        help="Use the batched Qwen message tokenization transform during Open-Instruct dataset conversion.",
    )
    parser.add_argument(
        "--dataset_transform_profile",
        default="qwen",
        choices=("qwen", "olmo"),
        help=(
            "SFT message transform profile. qwen keeps the Qwen tool/no-tools template behavior; "
            "olmo renders messages with the selected tokenizer chat template without Qwen system/tool injection."
        ),
    )
    parser.add_argument(
        "--dataset_backend",
        default="auto",
        choices=("auto", "hf", "polars"),
        help="Dataset conversion backend. auto uses polars for supported local parquet Qwen SFT data, otherwise HF datasets.",
    )
    parser.add_argument("--dataset_weight", default="1.0", help="Mixer weight for the local dataset.")
    parser.add_argument("--dataset_split", default="train", help="Dataset split name.")
    parser.add_argument("--skip_cache_prepare", action="store_true", help="Skip cache_dataset_only phase.")
    parser.add_argument(
        "--log_tokenized_sample",
        default="false",
        choices=("true", "false"),
        help="Log one tokenizer/template preview before launching training. Disabled by default to avoid dataset examples in logs.",
    )
    parser.add_argument(
        "--tokenized_sample_max_tokens",
        type=int,
        default=256,
        help="Maximum token IDs to print for the tokenizer preview.",
    )
    parser.add_argument(
        "--tokenized_sample_max_chars",
        type=int,
        default=4000,
        help="Maximum decoded/rendered characters to print for the tokenizer preview.",
    )
    parser.add_argument("--open_instruct_dir", default=None, help="Path to Open-Instruct checkout in the image.")
    parser.add_argument("--with_tracking", action="store_true", help="Enable W&B tracking in Open-Instruct.")
    parser.add_argument(
        "--wandb_mode",
        default="online",
        choices=("auto", "online", "offline", "disabled"),
        help="W&B mode. auto uses online with --with_tracking and offline otherwise.",
    )
    parser.add_argument("--wandb_project", default="olmo-sft")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--rl_preset",
        default=None,
        choices=(
            "olmo3_7b_4xh200_smoke",
            "olmo3_32b_3node_smoke",
            "olmo3_32b_2gen_1train_smoke",
            "olmo3_32b_2gen_1train_long",
        ),
        help=(
            "GRPO-only preset that fills missing Open-Instruct arguments for a smoke run. "
            "Explicit CLI flags always override preset defaults."
        ),
    )
    parser.add_argument(
        "--grpo_ray_port",
        type=int,
        default=None,
        help="GRPO-only Ray head port. Defaults to MASTER_PORT/--master_port, then 29400.",
    )
    parser.add_argument(
        "--grpo_ray_dashboard_port",
        type=int,
        default=8265,
        help="GRPO-only Ray dashboard port.",
    )
    parser.add_argument(
        "--grpo_ray_start_timeout",
        type=int,
        default=900,
        help="Seconds to wait for Ray cluster startup/status checks.",
    )
    parser.add_argument(
        "--grpo_ray_num_cpus",
        type=int,
        default=96,
        help="CPUs to advertise per node to Ray for GRPO. Keep this bounded to avoid slow worker prestart storms.",
    )
    parser.add_argument(
        "--grpo_ray_worker_start_retries",
        type=int,
        default=3,
        help="Number of attempts for a non-driver GRPO node to join the Ray cluster.",
    )
    parser.add_argument(
        "--grpo_ray_temp_dir",
        default=None,
        help=(
            "GRPO-only Ray temp directory template. Supports {host}, {node_rank}, and {port}. "
            "Defaults to a per-node /tmp/ray-grpo path to avoid socket collisions on shared /tmp mounts."
        ),
    )
    parser.add_argument(
        "--grpo_worker_poll_interval",
        type=int,
        default=10,
        help="Seconds between Ray-head liveness checks on non-driver GRPO nodes.",
    )
    parser.add_argument(
        "--grpo_stop_ray_on_exit",
        default="true",
        choices=("true", "false"),
        help="Stop Ray on this node after the GRPO driver/worker monitor exits.",
    )
    parser.add_argument(
        "--grpo_judge_preflight",
        default="false",
        choices=("true", "false"),
        help="Run a small OpenAI-compatible judge request on the driver before starting Ray.",
    )
    parser.add_argument(
        "--grpo_judge_preflight_prompt",
        default="Return OK.",
        help="Prompt used by --grpo_judge_preflight.",
    )
    parser.add_argument(
        "--grpo_deepspeed_stage",
        type=int,
        default=None,
        help=(
            "Forwarded to GRPO as --deepspeed_stage. Use 3 for ZeRO-3 learner sharding/offload. "
            "If unset, the GRPO preset or Open-Instruct default applies."
        ),
    )
    parser.add_argument(
        "--grpo_deepspeed_zpg",
        type=int,
        default=None,
        help="Forwarded to GRPO as --deepspeed_zpg. Defaults to the GRPO preset/Open-Instruct value when unset.",
    )
    parser.add_argument(
        "--grpo_deepspeed_offload_param",
        default=None,
        choices=("true", "false"),
        help="Forwarded to GRPO as --deepspeed_offload_param to offload ZeRO-3 parameters to CPU.",
    )
    parser.add_argument(
        "--grpo_deepspeed_offload_optimizer",
        default=None,
        choices=("true", "false"),
        help="Forwarded to GRPO as --deepspeed_offload_optimizer to offload optimizer state to CPU.",
    )
    parser.add_argument(
        "--grpo_sequence_parallel_size",
        type=int,
        default=None,
        help="Forwarded to GRPO as --sequence_parallel_size for learner sequence parallelism.",
    )
    parser.add_argument(
        "--rlcsd_dir",
        default=None,
        help="Path to an RLCSD checkout. Defaults to RLCSD_DIR, /tmp/RLCSD-runtime, /opt/RLCSD, or ./RLCSD.",
    )
    parser.add_argument(
        "--rlcsd_method",
        default="rlcsd",
        choices=("rlcsd", "grpo", "cispo", "rlsd", "rlsd_ectr", "opsd", "opsd_ectr", "sdpo", "srpo"),
        help="RLCSD/verl policy loss mode for --backend verl_rlcsd.",
    )
    parser.add_argument(
        "--rlcsd_train_file",
        default=None,
        help="Prepared verl train parquet. If unset, --dataset_path is converted to verl proof parquet.",
    )
    parser.add_argument(
        "--rlcsd_val_file",
        default=None,
        help="Prepared verl validation parquet. Defaults to the train file for smoke tests.",
    )
    parser.add_argument(
        "--rlcsd_data_cache",
        default=None,
        help="Directory for prepared verl/RLCSD datasets. Defaults under --cache_dir.",
    )
    parser.add_argument(
        "--rlcsd_max_rows",
        type=int,
        default=0,
        help="Limit prepared proof rows for smoke tests. 0 uses all rows.",
    )
    parser.add_argument(
        "--rlcsd_group_size",
        type=int,
        default=8,
        help="Number of sampled rollouts per prompt for RLCSD.",
    )
    parser.add_argument(
        "--rlcsd_async_rollout",
        default="false",
        choices=("true", "false"),
        help=(
            "Use verl's fully-async split trainer/rollouter path for --backend verl_rlcsd. "
            "This enables separate trainer and rollout resource pools."
        ),
    )
    parser.add_argument(
        "--rlcsd_train_nodes",
        type=int,
        default=0,
        help="Trainer nodes for --rlcsd_async_rollout true. 0 defaults to 1.",
    )
    parser.add_argument(
        "--rlcsd_train_gpus_per_node",
        type=int,
        default=0,
        help="Trainer GPUs per node for --rlcsd_async_rollout true. 0 defaults to --num_gpus.",
    )
    parser.add_argument(
        "--rlcsd_rollout_nodes",
        type=int,
        default=0,
        help="Rollout nodes for --rlcsd_async_rollout true. 0 defaults to remaining nodes after trainer nodes.",
    )
    parser.add_argument(
        "--rlcsd_rollout_gpus_per_node",
        type=int,
        default=0,
        help="Rollout GPUs per node for --rlcsd_async_rollout true. 0 defaults to --num_gpus.",
    )
    parser.add_argument(
        "--rlcsd_rollout_data_parallel_size",
        type=int,
        default=1,
        help=(
            "vLLM rollout data-parallel size. Keep 1 for multiple standalone TP replicas; "
            "increase only when using vLLM DP inside one rollout replica."
        ),
    )
    parser.add_argument(
        "--rlcsd_async_staleness_threshold",
        type=float,
        default=0.1,
        help="Fully-async rollout sample staleness threshold.",
    )
    parser.add_argument(
        "--rlcsd_async_trigger_parameter_sync_step",
        type=int,
        default=1,
        help="Fully-async trainer steps between actor-to-rollout weight syncs.",
    )
    parser.add_argument(
        "--rlcsd_async_require_batches",
        type=int,
        default=1,
        help="Number of PPO mini-batches the async trainer consumes per queue read.",
    )
    parser.add_argument(
        "--rlcsd_async_partial_rollout",
        default="true",
        choices=("true", "false"),
        help="Allow in-flight rollout requests to resume after async weight sync.",
    )
    parser.add_argument(
        "--rlcsd_async_total_rollout_steps",
        type=int,
        default=0,
        help="Override rollout.total_rollout_steps for fully-async mode. 0 derives it from max_train_steps.",
    )
    parser.add_argument(
        "--rlcsd_rollout_tensor_parallel_size",
        type=int,
        default=8,
        help="vLLM tensor parallel size for the RLCSD rollout engine.",
    )
    parser.add_argument(
        "--rlcsd_vllm_gpu_memory_utilization",
        type=float,
        default=0.6,
        help="vLLM GPU memory utilization for RLCSD rollouts.",
    )
    parser.add_argument(
        "--rlcsd_vllm_quantization",
        default="",
        help="vLLM rollout quantization mode for RLCSD/verl, for example 'fp8'. Empty leaves verl default.",
    )
    parser.add_argument(
        "--olmo3_sink",
        default="false",
        choices=("true", "false"),
        help=(
            "Convert OLMo3 HF weights to the local olmo3_sink architecture and register "
            "the matching Transformers/vLLM classes for --backend verl_rlcsd."
        ),
    )
    parser.add_argument(
        "--olmo3_sink_init_value",
        type=float,
        default=-10.0,
        help="Initial value for newly-added per-head attention sinks when converting a stock OLMo3 checkpoint.",
    )
    parser.add_argument(
        "--olmo3_sink_cache_dir",
        default="",
        help="Directory for converted olmo3_sink HF checkpoints. Defaults under --cache_dir.",
    )
    parser.add_argument(
        "--olmo3_sink_attn_implementation",
        default="flash_attention_3",
        help="Transformers attention implementation used by the verl actor/ref model when --olmo3_sink true.",
    )
    parser.add_argument(
        "--rlcsd_max_prompt_length",
        type=int,
        default=5536,
        help="Maximum prompt tokens for verl/RLCSD dataset filtering.",
    )
    parser.add_argument(
        "--rlcsd_max_response_length",
        type=int,
        default=60000,
        help="Maximum rollout response tokens for proof generation.",
    )
    parser.add_argument(
        "--rlcsd_actor_max_token_len_per_gpu",
        type=int,
        default=0,
        help="verl actor ppo_max_token_len_per_gpu. 0 uses max_prompt+max_response.",
    )
    parser.add_argument(
        "--rlcsd_ppo_mini_batch_size",
        type=int,
        default=0,
        help="verl actor ppo_mini_batch_size. 0 uses rollout batch size.",
    )
    parser.add_argument(
        "--rlcsd_ppo_micro_batch_size_per_gpu",
        type=int,
        default=1,
        help="verl actor ppo_micro_batch_size_per_gpu.",
    )
    parser.add_argument(
        "--rlcsd_rollout_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for OLMo proof rollouts.",
    )
    parser.add_argument(
        "--rlcsd_rollout_top_p",
        type=float,
        default=0.95,
        help="Sampling top-p for OLMo proof rollouts.",
    )
    parser.add_argument(
        "--rlcsd_rollout_top_k",
        type=int,
        default=-1,
        help="Sampling top-k for OLMo proof rollouts. -1 disables top-k.",
    )
    parser.add_argument(
        "--rlcsd_positive_threshold",
        type=float,
        default=0.75,
        help="Reward threshold for positive proof teacher hints.",
    )
    parser.add_argument(
        "--rlcsd_negative_threshold",
        type=float,
        default=0.25,
        help="Reward threshold for negative proof teacher hints.",
    )
    parser.add_argument(
        "--rlcsd_proof_reward_weight",
        type=float,
        default=0.76,
        help="Paper-style proof reward weight alpha for DeepSeekMath-V2 proof generation reward.",
    )
    parser.add_argument(
        "--rlcsd_self_eval_reward_weight",
        type=float,
        default=0.24,
        help="Paper-style self-evaluation reward weight beta for DeepSeekMath-V2 proof generation reward.",
    )
    parser.add_argument(
        "--rlcsd_partial_format_score",
        type=float,
        default=0.7,
        help=(
            "Format multiplier when ## Solution is present but the self-evaluation section "
            "or boxed self-score is missing, often because generation hit max tokens."
        ),
    )
    parser.add_argument(
        "--rlcsd_clip_ratio_low",
        type=float,
        default=-1.0,
        help="Override actor clip_ratio_low for simple verl losses. -1 uses method defaults.",
    )
    parser.add_argument(
        "--rlcsd_clip_ratio_high",
        type=float,
        default=-1.0,
        help="Override actor clip_ratio_high for simple verl losses. -1 uses method defaults.",
    )
    parser.add_argument(
        "--rlcsd_extra_overrides",
        default="",
        help="Extra comma-separated Hydra overrides forwarded to verl/RLCSD.",
    )
    parser.add_argument(
        "--hf_log_upload",
        default="true",
        choices=("true", "false"),
        help="Upload the final logdir to a Hugging Face dataset repo from the primary parent process.",
    )
    parser.add_argument(
        "--hf_log_repo",
        default="nguyen599/aimo-proof-pilot-log",
        help="Private Hugging Face dataset repo used for final logdir uploads.",
    )
    parser.add_argument(
        "--hf_log_path_prefix",
        default="training-logs",
        help="Path prefix inside --hf_log_repo for final logdir uploads.",
    )
    parser.add_argument(
        "--hf_log_upload_interval_seconds",
        type=int,
        default=1200,
        help=(
            "Upload the current logdir to --hf_log_repo every N seconds while training. "
            "Set <=0 to disable periodic uploads; final upload still follows --hf_log_upload."
        ),
    )
    parser.add_argument(
        "--hf_dependency_upgrade",
        default="true",
        choices=("true", "false"),
        help=(
            "Before HF auto-download, install newer Hugging Face CLI dependencies into a writable /tmp target "
            "when the image has older versions. This does not modify the SIF filesystem."
        ),
    )
    parser.add_argument(
        "--hf_dependency_upgrade_packages",
        default=DEFAULT_HF_DEPENDENCY_UPGRADE_PACKAGES,
        help="Comma-separated pip specs used by --hf_dependency_upgrade.",
    )
    add_operator_args(parser)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--ephemeral_save_interval", type=int, default=100)
    parser.add_argument(
        "--checkpoint_keep_last",
        type=int,
        default=2,
        help=(
            "Keep only the newest N local OLMo-core checkpoints in output_path. "
            "Set <=0 to disable permanent-checkpoint pruning."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_upload",
        default="false",
        choices=("true", "false"),
        help="Upload saved OLMo-core checkpoints asynchronously to a Hugging Face dataset repo from rank 0.",
    )
    parser.add_argument(
        "--hf_checkpoint_repo",
        default="nguyen599/olmo3-ckpt",
        help="Private Hugging Face dataset repo used for asynchronous checkpoint uploads.",
    )
    parser.add_argument(
        "--hf_checkpoint_path_prefix",
        default="checkpoints",
        help="Path prefix inside --hf_checkpoint_repo for checkpoint uploads.",
    )
    parser.add_argument(
        "--hf_checkpoint_keep_last",
        type=int,
        default=10,
        help="Keep only the newest N uploaded checkpoint folders in the HF dataset. Set <=0 to disable remote pruning.",
    )
    parser.add_argument(
        "--hf_checkpoint_upload_workers",
        type=int,
        default=8,
        help=(
            "Workers used by huggingface_hub.upload_large_folder for checkpoint uploads. "
            "Set 0 to let huggingface_hub choose half of the available CPU cores."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_upload_report_interval_seconds",
        type=int,
        default=60,
        help=(
            "Seconds between upload_large_folder progress reports when "
            "HF_CHECKPOINT_UPLOAD_PRINT_REPORT=1. Reports are quiet by default to keep logs small."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_disable_file",
        default="",
        help=(
            "Runtime marker file that disables future HF checkpoint uploads when it exists. "
            "Defaults to <run output_path>/.disable_hf_checkpoint_upload when unset."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_convert",
        default="true",
        choices=("true", "false"),
        help=(
            "After uploading an OLMo-core checkpoint, convert it to Hugging Face format and upload "
            "the converted folder as a sibling path such as step100-hf. Only runs on rank 0."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_convert_tokenizer",
        default="auto",
        help=(
            "Tokenizer passed to OLMo-core convert_checkpoint_to_hf.py. "
            "'auto' uses the default HF repo for --model_arch, e.g. allenai/Olmo-3.1-32B-Think."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_convert_device",
        default="cpu",
        help="Device passed to OLMo-core convert_checkpoint_to_hf.py, usually cpu on the host.",
    )
    parser.add_argument(
        "--hf_checkpoint_converted_suffix",
        default="hf",
        help=(
            "Suffix for converted HF checkpoint folders in the Hub. "
            "Use 'hf' for folders such as step100-hf."
        ),
    )
    parser.add_argument(
        "--hf_checkpoint_convert_keep_local",
        default="false",
        choices=("true", "false"),
        help="Keep local converted HF checkpoint folders after successful upload. Default removes them to save /tmp.",
    )
    parser.add_argument(
        "--disable_checkpoints",
        action="store_true",
        help="Disable OLMo-core checkpoint callbacks for smoke tests that only need logs.",
    )
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--offline", default="false", choices=("true", "false"), help="Force HF offline mode.")
    parser.add_argument(
        "--dry_run_launch",
        action="store_true",
        help="Resolve the PBS/container launcher shape and print the internal torchrun command, then exit.",
    )
    parser.add_argument(
        "--collect_env_info",
        default="true",
        choices=("true", "false"),
        help="Write a redacted environment/rank/GPU/storage/network JSON report into logdir at startup.",
    )
    parser.add_argument(
        "--env_info_command_timeout",
        type=int,
        default=15,
        help="Per-command timeout in seconds for environment probe commands.",
    )
    parser.add_argument(
        "--learning_rates",
        default=None,
        help=(
            "Comma-separated learning rates to run sequentially, e.g. "
            "8e-7,1e-6,2e-6,3e-6,4e-6,5e-6. This is the preferred sweep flag."
        ),
    )
    parser.add_argument(
        "--optimizers",
        default=None,
        help=(
            "Comma-separated optimizers to run sequentially, e.g. skip_step_adamw,adamw_8bit. "
            "This is the preferred optimizer-sweep flag."
        ),
    )
    parser.add_argument(
        "--sweep_name",
        default=None,
        help="Subdirectory name under output_path/logdir for sweep trials. Defaults to lr_optimizer_sweep.",
    )
    parser.add_argument(
        "--sweep_dry_run",
        action="store_true",
        help="Print and record sweep trial commands without launching training.",
    )
    parser.add_argument(
        "--sweep_continue_on_failure",
        action="store_true",
        help="Continue launching later sweep trials after a failed trial. Default stops on first failure.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, remaining = parser.parse_known_args(raw_argv)
    args._raw_argv = raw_argv
    args.grpo_extra_args = remaining

    if args.backend not in RAY_RL_BACKENDS and remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if args.backend not in RAY_RL_BACKENDS and args.operator_mode != "true":
        if not args.model_path:
            parser.error("--model_path is required unless an RL backend is used.")
        if not args.dataset_path:
            parser.error("--dataset_path is required unless an RL backend is used.")
    if args.backend in RAY_RL_BACKENDS and args.internal_backend is not None:
        parser.error("--internal_backend is only supported by SFT worker launches.")
    if args.backend in RAY_RL_BACKENDS and not cli_has_flag(raw_argv, "--offline"):
        args.offline = "false"
    return args


def hf_checkpoint_upload_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "hf_checkpoint_upload", "false") == "true" and bool(
        getattr(args, "hf_checkpoint_repo", "").strip()
    )


def hf_checkpoint_path_prefix(args: argparse.Namespace) -> str:
    raw_prefix = getattr(args, "hf_checkpoint_path_prefix", "checkpoints") or ""
    parts = [sanitize_slug_part(part) for part in raw_prefix.strip("/").split("/") if part.strip()]
    return "/".join(parts)


def hf_checkpoint_run_path(args: argparse.Namespace, output_path: Path) -> str:
    prefix = hf_checkpoint_path_prefix(args)
    run_name = path_name_token(str(output_path), "run")
    return f"{prefix}/{run_name}" if prefix else run_name


def hf_checkpoint_disable_file(args: argparse.Namespace, output_path: Path) -> Path:
    raw_path = str(getattr(args, "hf_checkpoint_disable_file", "") or "").strip()
    if raw_path.lower() not in {"", "none", "null", "auto"}:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else output_path / path
    return output_path / ".disable_hf_checkpoint_upload"


def hf_checkpoint_convert_enabled(args: argparse.Namespace) -> bool:
    return (
        hf_checkpoint_upload_enabled(args)
        and getattr(args, "hf_checkpoint_convert", "false") == "true"
    )


def hf_checkpoint_convert_tokenizer(args: argparse.Namespace) -> str:
    raw_tokenizer = str(getattr(args, "hf_checkpoint_convert_tokenizer", "auto") or "auto").strip()
    if raw_tokenizer.lower() not in {"", "auto", "none", "null"}:
        return raw_tokenizer
    try:
        model_arch = normalize_model_arch(
            str(getattr(args, "model_arch", "auto") or "auto"),
            Path(getattr(args, "model_path", ".")),
        )
    except Exception:
        model_arch = ""
    default_repo = DEFAULT_MODEL_HF_REPOS.get(model_arch)
    if default_repo:
        return default_repo
    tokenizer_path = getattr(args, "tokenizer_path", None)
    if tokenizer_path:
        return str(tokenizer_path)
    return str(getattr(args, "model_path", ""))


def hf_checkpoint_normalized_converted_suffix(suffix: str) -> str:
    safe_suffix = sanitize_slug_part(suffix or "hf")
    if not safe_suffix.startswith("-"):
        safe_suffix = f"-{safe_suffix}"
    return safe_suffix


def hf_checkpoint_converted_name(folder: Path, suffix: str) -> str:
    return f"{folder.name}{hf_checkpoint_normalized_converted_suffix(suffix)}"


def hf_checkpoint_converted_folder(folder: Path, suffix: str) -> Path:
    return folder.parent / ".hf_converted_checkpoints" / hf_checkpoint_converted_name(folder, suffix)


def repo_checkpoint_step_from_component(component: str) -> int | None:
    match = re.fullmatch(r"(?:step|global_step_)(\d+)(?:[-_].*)?", component)
    return int(match.group(1)) if match else None


def checkpoint_step_from_path(path: str | Path) -> int | None:
    match = re.fullmatch(r"(?:step|global_step_)(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def checkpoint_is_ephemeral(path: str | Path) -> bool:
    metadata_path = Path(path) / ".metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except Exception:
        logging.warning("Could not read checkpoint metadata for %s; treating it as permanent.", path)
        return False
    return bool(metadata.get("ephemeral"))


def prepare_checkpoint_large_upload_staging(folder: Path, path_in_repo: str) -> Path:
    """Create a resumable Hub upload tree using symlinks instead of copying checkpoint data."""
    source = folder.expanduser().resolve()
    stage_key = hashlib.sha256(f"{source}:{path_in_repo}".encode("utf-8")).hexdigest()[:16]
    stage_root = source.parent / ".hf_large_upload_staging" / stage_key
    staged_checkpoint = stage_root / Path(path_in_repo)
    staged_checkpoint.mkdir(parents=True, exist_ok=True)

    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source)
        staged_path = staged_checkpoint / relative_path
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_path.is_symlink():
            try:
                if staged_path.resolve() == source_path.resolve():
                    continue
            except FileNotFoundError:
                pass
            staged_path.unlink()
        elif staged_path.exists():
            staged_path.unlink()
        try:
            staged_path.symlink_to(source_path)
        except OSError:
            os.link(source_path, staged_path)

    return stage_root


def upload_large_checkpoint_folder_to_hf(
    *,
    folder: Path,
    repo_id: str,
    path_in_repo: str,
    token: str | None,
    upload_workers: int,
    report_interval_seconds: int,
    disable_file: Path | None,
    label: str,
) -> bool:
    from huggingface_hub import HfApi

    if disable_file is not None and disable_file.exists():
        logging.warning("Skipping %s upload because disable marker exists: %s", label, disable_file)
        return False

    staging_root = prepare_checkpoint_large_upload_staging(folder, path_in_repo)

    def upload_once():
        if disable_file is not None and disable_file.exists():
            raise RuntimeError(f"HF {label} upload disabled by marker file: {disable_file}")
        api = HfApi(token=token)
        logging.info(
            "Uploading %s with upload_large_folder to HF dataset %s/%s: %s "
            "(staging=%s workers=%s quiet_progress=%s)",
            label,
            repo_id,
            path_in_repo,
            folder,
            staging_root,
            upload_workers or "auto",
            os.environ.get("HF_CHECKPOINT_UPLOAD_PRINT_REPORT", "0") != "1",
        )
        with quiet_hf_transfer(), hf_transfer_heartbeat(
            f"HF {label} upload to {repo_id}/{path_in_repo}",
            report_interval_seconds,
        ):
            return api.upload_large_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(staging_root),
                private=True,
                ignore_patterns=["**/__pycache__/**", "**/.nfs*"],
                num_workers=upload_workers or None,
                print_report=os.environ.get("HF_CHECKPOINT_UPLOAD_PRINT_REPORT", "0") == "1",
                print_report_every=max(1, report_interval_seconds),
            )

    try:
        retry_hf_operation(
            f"HF {label} upload to {repo_id}/{path_in_repo}",
            upload_once,
            abort_if=(lambda: disable_file is not None and disable_file.exists()),
        )
        logging.info(
            "Uploaded %s to HF dataset: https://huggingface.co/datasets/%s/tree/main/%s",
            label,
            repo_id,
            path_in_repo,
        )
        shutil.rmtree(staging_root, ignore_errors=True)
        return True
    except Exception:
        logging.exception(
            "HF %s upload failed for %s. Resumable metadata remains in %s.",
            label,
            folder,
            staging_root,
        )
        return False


def read_checkpoint_upload_state(state_path: Path) -> set[str]:
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except Exception:
        logging.warning("Could not read HF checkpoint upload state file: %s", state_path)
        return set()
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, dict):
        uploaded = raw.get("uploaded")
        if isinstance(uploaded, list):
            return {str(item) for item in uploaded}
    return set()


def write_checkpoint_upload_state(state_path: Path, uploaded: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps({"uploaded": sorted(uploaded)}, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def verl_latest_checkpoint_step(output_path: Path) -> int | None:
    tracker = output_path / "latest_checkpointed_iteration.txt"
    try:
        raw = tracker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        logging.warning("Could not read verl checkpoint tracker: %s", tracker)
        return None
    try:
        return int(raw)
    except ValueError:
        logging.warning("Invalid verl checkpoint tracker value in %s: %r", tracker, raw)
        return None


def completed_verl_checkpoint_paths(output_path: Path) -> list[tuple[int, Path]]:
    latest_step = verl_latest_checkpoint_step(output_path)
    if latest_step is None:
        return []
    checkpoints: list[tuple[int, Path]] = []
    for child in output_path.glob("global_step_*"):
        if not child.is_dir():
            continue
        step = checkpoint_step_from_path(child)
        if step is None or step > latest_step:
            continue
        checkpoints.append((step, child))
    return sorted(checkpoints, key=lambda item: item[0])


def verl_checkpoint_repo_path(run_path: str, folder: Path) -> str:
    folder_name = sanitize_slug_part(folder.name)
    return f"{run_path.rstrip('/')}/{folder_name}" if run_path else folder_name


def watch_verl_checkpoints_for_hf_upload(
    *,
    output_path: Path,
    args: argparse.Namespace,
    stop_event: threading.Event,
) -> None:
    if not hf_checkpoint_upload_enabled(args):
        return

    repo_id = args.hf_checkpoint_repo.strip()
    run_path = hf_checkpoint_run_path(args, output_path)
    token = hf_log_token()
    state_path = output_path / ".hf_checkpoint_upload_state.json"
    disable_file = hf_checkpoint_disable_file(args, output_path)
    upload_workers = max(0, int(args.hf_checkpoint_upload_workers))
    report_interval_seconds = max(1, int(args.hf_checkpoint_upload_report_interval_seconds))
    scan_interval_seconds = max(10, min(60, report_interval_seconds))
    uploaded = read_checkpoint_upload_state(state_path)

    if token is None:
        logging.warning(
            "HF checkpoint upload enabled for %s but no HF token was found. Upload may fail.",
            repo_id,
        )

    logging.info(
        "Started verl HF checkpoint watcher: output_path=%s repo=%s run_path=%s scan_interval=%ss",
        output_path,
        repo_id,
        run_path,
        scan_interval_seconds,
    )

    while True:
        try:
            if disable_file.exists():
                logging.warning("Verl HF checkpoint watcher disabled by marker file: %s", disable_file)
            else:
                for step, folder in completed_verl_checkpoint_paths(output_path):
                    key = folder.name
                    if key in uploaded:
                        continue
                    path_in_repo = verl_checkpoint_repo_path(run_path, folder)
                    uploaded_ok = upload_large_checkpoint_folder_to_hf(
                        folder=folder,
                        repo_id=repo_id,
                        path_in_repo=path_in_repo,
                        token=token,
                        upload_workers=upload_workers,
                        report_interval_seconds=report_interval_seconds,
                        disable_file=disable_file,
                        label=f"verl checkpoint step {step}",
                    )
                    if uploaded_ok:
                        uploaded.add(key)
                        write_checkpoint_upload_state(state_path, uploaded)
        except Exception:
            logging.exception("Verl HF checkpoint watcher iteration failed.")

        if stop_event.is_set():
            break
        stop_event.wait(scan_interval_seconds)

    # Final scan after training exits, so a checkpoint saved near process shutdown is not missed.
    if not disable_file.exists():
        for step, folder in completed_verl_checkpoint_paths(output_path):
            key = folder.name
            if key in uploaded:
                continue
            path_in_repo = verl_checkpoint_repo_path(run_path, folder)
            uploaded_ok = upload_large_checkpoint_folder_to_hf(
                folder=folder,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                token=token,
                upload_workers=upload_workers,
                report_interval_seconds=report_interval_seconds,
                disable_file=disable_file,
                label=f"verl checkpoint step {step}",
            )
            if uploaded_ok:
                uploaded.add(key)
                write_checkpoint_upload_state(state_path, uploaded)
    logging.info("Stopped verl HF checkpoint watcher: uploaded=%s", sorted(uploaded))


def install_checkpoint_upload_and_retention_callback(
    config: object,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    hf_enabled = hf_checkpoint_upload_enabled(args)
    local_keep_last = int(getattr(args, "checkpoint_keep_last", 2))
    remote_keep_last = int(getattr(args, "hf_checkpoint_keep_last", 2))
    if not hf_enabled and local_keep_last <= 0:
        return

    try:
        from olmo_core.distributed.utils import get_rank
        from olmo_core.io import is_url
        from olmo_core.train.callbacks.callback import Callback
    except ImportError:
        logging.warning("Checkpoint HF upload/retention callback skipped; OLMo-core callback API is unavailable.")
        return

    @dataclass
    class CheckpointUploadAndRetentionCallback(Callback):
        priority: ClassVar[int] = 2

        repo_id: str = ""
        run_path: str = ""
        hf_enabled: bool = False
        local_keep_last: int = 2
        remote_keep_last: int = 2
        upload_workers: int = 8
        upload_report_interval_seconds: int = 60
        disable_file: str = ""
        convert_enabled: bool = False
        convert_tokenizer: str = ""
        convert_device: str = "cpu"
        convert_validation: bool = False
        converted_suffix: str = "-hf"
        convert_keep_local: bool = False

        _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
        _futures: dict[str, Future] = field(default_factory=dict, init=False, repr=False)
        _known_paths: dict[str, int] = field(default_factory=dict, init=False, repr=False)
        _pending_upload_paths: list[str] = field(default_factory=list, init=False, repr=False)
        _closed: bool = field(default=False, init=False, repr=False)
        _disable_logged: bool = field(default=False, init=False, repr=False)

        def _is_primary_rank(self) -> bool:
            try:
                return get_rank() == 0
            except Exception:
                rank = os.environ.get("RANK") or os.environ.get("GLOBAL_RANK")
                return rank in (None, "", "0")

        def _ensure_executor(self) -> ThreadPoolExecutor:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-ckpt-upload")
            return self._executor

        def _upload_disabled(self) -> bool:
            if not self.hf_enabled:
                return True
            if not self.disable_file:
                return False
            disabled = Path(self.disable_file).exists()
            if disabled and not self._disable_logged:
                logging.warning("HF checkpoint upload disabled by marker file: %s", self.disable_file)
                self._disable_logged = True
            return disabled

        def _remember_checkpoint(self, path: str | Path) -> None:
            step = checkpoint_step_from_path(path)
            if step is not None:
                self._known_paths[str(path)] = step

        def _local_checkpoint_paths(self) -> list[tuple[int, str]]:
            paths = dict(self._known_paths)
            save_folder = getattr(self.trainer, "save_folder", "")
            if save_folder and not is_url(save_folder):
                save_dir = Path(save_folder)
                if save_dir.is_dir():
                    for child in save_dir.iterdir():
                        if not child.is_dir():
                            continue
                        step = checkpoint_step_from_path(child)
                        if step is not None:
                            paths[str(child)] = step
            existing = [(step, path) for path, step in paths.items() if Path(path).exists()]
            return sorted(existing, key=lambda item: item[0])

        def _repo_checkpoint_path(self, path: str | Path) -> str:
            folder_name = sanitize_slug_part(Path(path).name)
            return f"{self.run_path.rstrip('/')}/{folder_name}" if self.run_path else folder_name

        def _repo_converted_checkpoint_path(self, path: str | Path) -> str:
            folder_name = sanitize_slug_part(hf_checkpoint_converted_name(Path(path), self.converted_suffix))
            return f"{self.run_path.rstrip('/')}/{folder_name}" if self.run_path else folder_name

        def _checkpointer_callback(self) -> object | None:
            for callback in self.trainer.callbacks.values():
                if callback.__class__.__name__ == "CheckpointerCallback":
                    return callback
            return None

        def _checkpoint_save_due_this_step(self) -> bool:
            checkpointer = self._checkpointer_callback()
            if checkpointer is None or not getattr(checkpointer, "enabled", True):
                return False
            step = self.step
            fixed_steps = getattr(checkpointer, "fixed_steps", None)
            if fixed_steps is not None and step in fixed_steps:
                return True
            save_interval = getattr(checkpointer, "save_interval", None)
            if save_interval is not None and save_interval > 0 and step % save_interval == 0:
                return True
            ephemeral_interval = getattr(checkpointer, "ephemeral_save_interval", None)
            if ephemeral_interval is None or ephemeral_interval <= 0:
                return False
            if step % ephemeral_interval != 0 or getattr(self.trainer, "block_ephemeral_checkpoints", False):
                return False
            cooldown = getattr(checkpointer, "ephemeral_cooldown", None)
            latest_step = getattr(checkpointer, "_latest_checkpoint_step", -1)
            return cooldown is None or (step - latest_step) >= cooldown

        def _final_checkpoint_save_due(self) -> bool:
            checkpointer = self._checkpointer_callback()
            if checkpointer is None or not getattr(checkpointer, "enabled", True):
                return False
            return self.step > getattr(checkpointer, "_latest_checkpoint_step", -1)

        def _conversion_env(self, folder: Path) -> dict[str, str]:
            env = os.environ.copy()
            olmo_core_dir = find_olmo_core_dir(env.get("OLMO_CORE_DIR"))
            app_dir = Path(__file__).resolve().parent
            pythonpath_parts = [str(olmo_core_dir / "src"), str(app_dir), str(Path("/app"))]
            if env.get("PYTHONPATH"):
                pythonpath_parts.extend(env["PYTHONPATH"].split(os.pathsep))
            env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in pythonpath_parts if part))
            configure_runtime_cache_environment(env, folder.parent / ".hf_conversion_cache")
            return env

        def _upload_large_folder_to_hf(self, folder: Path, path_in_repo: str, token: str | None, label: str) -> bool:
            from huggingface_hub import HfApi

            if self._upload_disabled():
                logging.info("Skipping %s upload because HF checkpoint upload is disabled: %s", label, folder)
                return False
            staging_root = prepare_checkpoint_large_upload_staging(folder, path_in_repo)
            try:
                def upload_once():
                    if self._upload_disabled():
                        raise RuntimeError(f"HF {label} upload disabled by marker file: {self.disable_file}")
                    api = HfApi(token=token)
                    logging.info(
                        "Uploading %s with upload_large_folder to HF dataset %s/%s: %s "
                        "(staging=%s workers=%s quiet_progress=%s)",
                        label,
                        self.repo_id,
                        path_in_repo,
                        folder,
                        staging_root,
                        self.upload_workers or "auto",
                        os.environ.get("HF_CHECKPOINT_UPLOAD_PRINT_REPORT", "0") != "1",
                    )
                    with quiet_hf_transfer(), hf_transfer_heartbeat(
                        f"HF {label} upload to {self.repo_id}/{path_in_repo}",
                        self.upload_report_interval_seconds,
                    ):
                        return api.upload_large_folder(
                            repo_id=self.repo_id,
                            repo_type="dataset",
                            folder_path=str(staging_root),
                            private=True,
                            ignore_patterns=["**/__pycache__/**", "**/.nfs*"],
                            num_workers=self.upload_workers or None,
                            print_report=os.environ.get("HF_CHECKPOINT_UPLOAD_PRINT_REPORT", "0") == "1",
                            print_report_every=max(1, self.upload_report_interval_seconds),
                        )

                retry_hf_operation(
                    f"HF {label} upload to {self.repo_id}/{path_in_repo}",
                    upload_once,
                    abort_if=self._upload_disabled,
                )
                logging.info(
                    "Uploaded %s to HF dataset: https://huggingface.co/datasets/%s/tree/main/%s",
                    label,
                    self.repo_id,
                    path_in_repo,
                )
                shutil.rmtree(staging_root, ignore_errors=True)
                return True
            except Exception:
                logging.exception(
                    "HF %s upload failed for %s. Resumable metadata remains in %s.",
                    label,
                    folder,
                    staging_root,
                )
                return False

        def _convert_checkpoint_to_hf(self, folder: Path) -> Path | None:
            if not self.convert_enabled:
                return None
            if self._upload_disabled():
                logging.info("Skipping HF checkpoint conversion because upload is disabled: %s", folder)
                return None
            converted_folder = hf_checkpoint_converted_folder(folder, self.converted_suffix)
            marker_path = converted_folder / ".conversion_complete.json"
            if marker_path.is_file():
                logging.info("Using existing converted HF checkpoint folder: %s", converted_folder)
                return converted_folder

            if converted_folder.exists():
                logging.warning("Removing incomplete converted HF checkpoint folder before retry: %s", converted_folder)
                shutil.rmtree(converted_folder)
            converted_folder.parent.mkdir(parents=True, exist_ok=True)

            olmo_core_dir = find_olmo_core_dir(os.environ.get("OLMO_CORE_DIR"))
            conversion_script = olmo_core_dir / "src" / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"
            if not conversion_script.is_file():
                logging.warning("HF checkpoint conversion skipped; script not found: %s", conversion_script)
                return None

            command = [
                sys.executable,
                str(conversion_script),
                "-i",
                str(folder),
                "-o",
                str(converted_folder),
                "--tokenizer",
                self.convert_tokenizer,
                "--device",
                self.convert_device,
            ]
            if not self.convert_validation:
                command.append("--skip-validation")
            logging.info(
                "Converting OLMo-core checkpoint to HF format: input=%s output=%s tokenizer=%s device=%s "
                "validation=%s",
                folder,
                converted_folder,
                self.convert_tokenizer,
                self.convert_device,
                self.convert_validation,
            )
            run_command(command, self._conversion_env(folder))
            marker_path.write_text(
                json.dumps(
                    {
                        "source": str(folder),
                        "tokenizer": self.convert_tokenizer,
                        "device": self.convert_device,
                        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return converted_folder

        def _convert_and_upload_hf_checkpoint(self, folder: Path, token: str | None) -> None:
            try:
                converted_folder = self._convert_checkpoint_to_hf(folder)
                if converted_folder is None:
                    return
                converted_repo_path = self._repo_converted_checkpoint_path(folder)
                uploaded = self._upload_large_folder_to_hf(
                    converted_folder,
                    converted_repo_path,
                    token,
                    "converted HF checkpoint",
                )
                if uploaded and not self.convert_keep_local:
                    logging.info("Removing local converted HF checkpoint after upload: %s", converted_folder)
                    shutil.rmtree(converted_folder, ignore_errors=True)
            except Exception:
                logging.exception("HF checkpoint conversion/upload failed for %s.", folder)

        def _upload_checkpoint_and_prune_remote(self, path: str) -> None:
            if self._upload_disabled():
                logging.info("HF checkpoint upload skipped by marker before processing: %s", path)
                return
            folder = Path(path)
            if not folder.is_dir():
                logging.warning("HF checkpoint upload skipped; checkpoint folder does not exist: %s", folder)
                return
            if checkpoint_is_ephemeral(folder):
                logging.info("Skipping HF upload for ephemeral checkpoint: %s", folder)
                return

            try:
                from huggingface_hub import HfApi
            except ImportError:
                logging.warning("HF checkpoint upload skipped; huggingface_hub is not installed.")
                return

            token = hf_log_token()
            if not token:
                logging.warning(
                    "HF checkpoint upload may fail because no HF token was found. "
                    "Set HF_TOKEN or login with huggingface-cli for private repo access."
                )

            path_in_repo = self._repo_checkpoint_path(folder)
            uploaded = self._upload_large_folder_to_hf(folder, path_in_repo, token, "checkpoint")
            if uploaded:
                self._convert_and_upload_hf_checkpoint(folder, token)
                self._prune_remote_checkpoints(token)

        def _prune_remote_checkpoints(self, token: str | None) -> None:
            if self._upload_disabled():
                return
            if self.remote_keep_last <= 0:
                return
            run_prefix = self.run_path.rstrip("/")
            if not run_prefix:
                return
            from huggingface_hub import HfApi

            try:
                files = retry_hf_operation(
                    f"HF checkpoint repo listing for {self.repo_id}",
                    lambda: HfApi(token=token).list_repo_files(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=token,
                    ),
                )
            except Exception:
                logging.exception("Could not list HF checkpoint repo for retention pruning.")
                return

            step_dirs: dict[int, set[str]] = {}
            path_prefix = f"{run_prefix}/"
            for file_path in files:
                if not file_path.startswith(path_prefix):
                    continue
                rest = file_path[len(path_prefix) :]
                step_dir = rest.split("/", 1)[0]
                step = repo_checkpoint_step_from_component(step_dir)
                if step is not None:
                    step_dirs.setdefault(step, set()).add(step_dir)

            for step in sorted(step_dirs)[: -self.remote_keep_last]:
                for step_dir in sorted(step_dirs[step]):
                    old_path = f"{run_prefix}/{step_dir}"
                    try:
                        logging.info(
                            "Removing old HF checkpoint folder due to keep_last=%s: %s",
                            self.remote_keep_last,
                            old_path,
                        )
                        retry_hf_operation(
                            f"HF checkpoint remote prune for {self.repo_id}/{old_path}",
                            lambda old_path=old_path: HfApi(token=token).delete_folder(
                                repo_id=self.repo_id,
                                repo_type="dataset",
                                path_in_repo=old_path,
                                commit_message=f"Remove old OLMo checkpoint: {old_path}",
                                token=token,
                            ),
                        )
                    except Exception:
                        logging.exception("Could not remove old HF checkpoint folder: %s", old_path)

        def _wait_for_future(self, path: str, future: Future | None) -> None:
            if future is None:
                return
            try:
                future.result()
            except Exception:
                logging.exception("Background checkpoint upload failed before local retention pruning: %s", path)

        def _schedule_pending_uploads(self) -> None:
            if not self.hf_enabled:
                self._pending_upload_paths.clear()
                return
            if self._upload_disabled():
                self._pending_upload_paths.clear()
                for future in self._futures.values():
                    future.cancel()
                return
            while self._pending_upload_paths:
                path_str = self._pending_upload_paths.pop(0)
                if path_str in self._futures:
                    continue
                future = self._ensure_executor().submit(self._upload_checkpoint_and_prune_remote, path_str)
                self._futures[path_str] = future

        def _prune_local_checkpoints(self, target_keep: int | None = None) -> None:
            keep_last = self.local_keep_last if target_keep is None else target_keep
            if keep_last <= 0 and target_keep is None:
                return
            checkpoints = self._local_checkpoint_paths()
            while len(checkpoints) > keep_last:
                _, old_path = checkpoints.pop(0)
                future = self._futures.get(old_path)
                if future is not None and not future.done():
                    logging.info("Waiting for pending HF upload before removing old local checkpoint: %s", old_path)
                    self._wait_for_future(old_path, future)
                try:
                    logging.info("Removing old local checkpoint due to keep_last=%s: %s", keep_last, old_path)
                    shutil.rmtree(old_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    logging.exception("Could not remove old local checkpoint: %s", old_path)
                    break

        def pre_train(self) -> None:
            if not self._is_primary_rank():
                return
            self._prune_local_checkpoints()

        def post_train_batch(self) -> None:
            if not self._is_primary_rank() or self.local_keep_last <= 0:
                return
            if self._checkpoint_save_due_this_step():
                self._schedule_pending_uploads()
                self._prune_local_checkpoints(target_keep=max(0, self.local_keep_last - 1))

        def post_checkpoint_saved(self, path: str | Path) -> None:
            if not self._is_primary_rank():
                return
            self._remember_checkpoint(path)
            if self.hf_enabled:
                self._pending_upload_paths.append(str(path))
            self._prune_local_checkpoints()

        def post_step(self) -> None:
            if self._is_primary_rank():
                self._schedule_pending_uploads()

        def _drain_uploads(self) -> None:
            self._schedule_pending_uploads()
            if self._upload_disabled():
                return
            for path, future in list(self._futures.items()):
                self._wait_for_future(path, future)
            self._prune_local_checkpoints()

        def post_train(self) -> None:
            if not self._is_primary_rank():
                return
            if self.local_keep_last > 0 and self._final_checkpoint_save_due():
                self._schedule_pending_uploads()
                self._prune_local_checkpoints(target_keep=max(0, self.local_keep_last - 1))
            else:
                self._drain_uploads()

        def on_error(self, exc: BaseException) -> None:
            del exc
            if self._is_primary_rank():
                self._drain_uploads()

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            if self._is_primary_rank():
                self._drain_uploads()
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None

    callback = CheckpointUploadAndRetentionCallback(
        repo_id=args.hf_checkpoint_repo.strip(),
        run_path=hf_checkpoint_run_path(args, output_path),
        hf_enabled=hf_enabled,
        local_keep_last=local_keep_last,
        remote_keep_last=remote_keep_last,
        upload_workers=max(0, int(args.hf_checkpoint_upload_workers)),
        upload_report_interval_seconds=max(1, int(args.hf_checkpoint_upload_report_interval_seconds)),
        disable_file=str(hf_checkpoint_disable_file(args, output_path)),
        convert_enabled=hf_checkpoint_convert_enabled(args),
        convert_tokenizer=hf_checkpoint_convert_tokenizer(args),
        convert_device=str(args.hf_checkpoint_convert_device),
        convert_validation=str(getattr(args, "convert_validation", "false")) == "true",
        converted_suffix=str(args.hf_checkpoint_converted_suffix),
        convert_keep_local=args.hf_checkpoint_convert_keep_local == "true",
    )
    config.trainer.callbacks["hf_checkpoint_upload"] = callback
    logging.warning(
        "Installed OLMo checkpoint upload/retention callback: hf_enabled=%s repo=%s run_path=%s "
        "local_keep_last=%s remote_keep_last=%s upload_workers=%s report_interval=%ss "
        "disable_file=%s convert_enabled=%s convert_tokenizer=%s convert_device=%s converted_suffix=%s "
        "convert_keep_local=%s.",
        callback.hf_enabled,
        callback.repo_id,
        callback.run_path,
        callback.local_keep_last,
        callback.remote_keep_last,
        callback.upload_workers or "auto",
        callback.upload_report_interval_seconds,
        callback.disable_file,
        callback.convert_enabled,
        callback.convert_tokenizer,
        callback.convert_device,
        callback.converted_suffix,
        callback.convert_keep_local,
    )


def parse_csv_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"Expected at least one comma-separated value, got {value!r}.")
    return tokens


def parse_pipeline_split_points(value: str | None) -> list[int] | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "auto"}:
        return None
    split_points = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if split_points != sorted(set(split_points)):
        raise ValueError("--pipeline_split_points must be unique and increasing.")
    return split_points


def pipeline_schedule_stages_per_rank(schedule: str) -> int:
    multi_stage_schedules = {
        "Interleaved1F1B",
        "LoopedBFS",
        "InterleavedZeroBubble",
        "ZBVZeroBubble",
    }
    return 2 if schedule in multi_stage_schedules else 1


def balanced_counts(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def auto_pipeline_rank_layer_counts(n_layers: int, pp_degree: int, model_arch: str | None) -> list[int]:
    if pp_degree <= 0:
        raise ValueError("pipeline_parallel_degree must be positive")
    if pp_degree == 1:
        return [n_layers]
    if model_arch == "olmo3_32b" and pp_degree == 3:
        return [20, 22, 22]
    if model_arch == "olmo3_7b" and pp_degree == 3:
        return [10, 11, 11]

    base = n_layers // pp_degree
    remainder = n_layers % pp_degree
    counts = [base] * pp_degree
    for idx in range(remainder):
        counts[pp_degree - remainder + idx] += 1
    if remainder == 0 and counts[0] > 1:
        counts[0] -= 1
        counts[-1] += 1
    return counts


def auto_pipeline_split_points(
    n_layers: int,
    pp_degree: int,
    pipeline_schedule: str,
    model_arch: str | None,
) -> list[int] | None:
    if pp_degree <= 1:
        return None
    stages_per_rank = pipeline_schedule_stages_per_rank(pipeline_schedule)
    total_stages = pp_degree * stages_per_rank
    if total_stages > n_layers:
        raise ValueError(
            f"Cannot auto-calculate pipeline splits: total stages {total_stages} exceeds n_layers={n_layers}."
        )

    rank_layer_counts = auto_pipeline_rank_layer_counts(n_layers, pp_degree, model_arch)
    stage_sizes = [0] * total_stages
    for rank, rank_layers in enumerate(rank_layer_counts):
        rank_stage_sizes = balanced_counts(rank_layers, stages_per_rank)
        for virtual_idx, stage_size in enumerate(rank_stage_sizes):
            stage_sizes[rank + virtual_idx * pp_degree] = stage_size

    split_points: list[int] = []
    cursor = 0
    for stage_size in stage_sizes[:-1]:
        cursor += stage_size
        split_points.append(cursor)
    return split_points


def effective_pipeline_split_points(
    args: argparse.Namespace,
    model_arch: str | None = None,
    n_layers: int | None = None,
) -> list[int] | None:
    explicit = parse_pipeline_split_points(args.pipeline_split_points)
    if explicit:
        return explicit
    pp_degree = effective_pipeline_parallel_degree(args)
    if pp_degree <= 1:
        return None
    resolved_arch = model_arch if model_arch and model_arch != "auto" else getattr(args, "model_arch", None)
    if resolved_arch == "auto":
        resolved_arch = None
    resolved_layers = n_layers or DEFAULT_MODEL_N_LAYERS.get(resolved_arch or "")
    if resolved_layers is None:
        return None
    return auto_pipeline_split_points(
        resolved_layers,
        pp_degree,
        args.pipeline_schedule,
        resolved_arch,
    )


def effective_learning_rates_arg(args: argparse.Namespace) -> str | None:
    return args.learning_rates


def effective_optimizers_arg(args: argparse.Namespace) -> str | None:
    return args.optimizers


def parse_learning_rate_sweep(value: str | None, default: float) -> list[tuple[str, float]]:
    tokens = parse_csv_tokens(value)
    if not tokens:
        return [(format_float_token(default), default)]
    parsed: list[tuple[str, float]] = []
    for token in tokens:
        try:
            learning_rate = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid learning rate in --learning_rates: {token!r}") from exc
        if learning_rate <= 0:
            raise ValueError(f"Learning rates must be positive; got {token!r}.")
        parsed.append((token, learning_rate))
    return parsed


def normalize_optimizer_value(value: str) -> str:
    optimizer = value.strip()
    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Invalid optimizer in --optimizers: {value!r}. "
            f"Expected one of: {', '.join(sorted(VALID_OPTIMIZERS))}."
        )
    if optimizer in {"adamw_8bits", "torchao_adamw_8bit"}:
        return "adamw_8bit"
    return optimizer


def parse_optimizer_sweep(value: str | None, default: str) -> list[str]:
    tokens = parse_csv_tokens(value)
    if not tokens:
        return [normalize_optimizer_value(default)]
    return [normalize_optimizer_value(token) for token in tokens]


def format_float_token(value: float) -> str:
    return f"{value:.8g}"


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(parent.expanduser().resolve())
        return True
    except ValueError:
        return False


def default_sweep_shared_root(parent_output: Path, parent_logdir: Path, sweep_name: str) -> Path:
    env_root = os.environ.get("OLMO_SWEEP_SHARED_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve() / sweep_name
    candidate = parent_output / f"_shared_{sweep_name}"
    if path_is_within(candidate, parent_logdir):
        return parent_logdir.parent / f"{parent_logdir.name}_{sweep_name}_shared"
    return candidate


def run_arch_token(args: argparse.Namespace) -> str:
    if args.model_arch != "auto":
        return sanitize_slug_part(args.model_arch)
    try:
        config_name = infer_config_name(Path(args.model_path).expanduser().resolve())
    except Exception:
        config_name = None
    if config_name == "olmo3_7B":
        return "olmo3_7b"
    if config_name == "olmo3_32B":
        return "olmo3_32b"
    return "arch-auto"


def learning_rate_path_token(args: argparse.Namespace) -> str:
    value = effective_learning_rates_arg(args)
    if value:
        return "lrs-" + sanitize_slug_part(value)
    return "lr-" + sanitize_slug_part(format_float_token(args.learning_rate))


def optimizer_path_token(args: argparse.Namespace) -> str:
    value = effective_optimizers_arg(args)
    if value:
        return "opts-" + sanitize_slug_part(value)
    return "opt-" + sanitize_slug_part(normalized_optimizer_name(args))


def effective_chat_template_model(args: argparse.Namespace) -> str | None:
    raw_value = getattr(args, "chat_template_model", None)
    if raw_value:
        normalized = str(raw_value).strip().lower()
        if normalized in {"none", "null", "false", "0", "tokenizer-default", "tokenizer_default", "default"}:
            return None
        return str(raw_value)
    if getattr(args, "dataset_transform_profile", "qwen") == "qwen":
        return DEFAULT_QWEN_CHAT_TEMPLATE_MODEL
    return None


def dataset_transform_names(args: argparse.Namespace) -> list[str]:
    profile = getattr(args, "dataset_transform_profile", "qwen")
    if profile == "olmo":
        return ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    if profile != "qwen":
        raise ValueError(f"Unsupported --dataset_transform_profile: {profile}")
    tokenize_fn = (
        "sft_qwen_messages_tokenize_and_truncate_batched_v1"
        if getattr(args, "dataset_batched_tokenization", "true") == "true"
        else "sft_qwen_messages_tokenize_and_truncate_v1"
    )
    return [tokenize_fn, "sft_tulu_filter_v1"]


def olmo_core_dataset_cache_version(args: argparse.Namespace) -> str:
    profile = getattr(args, "dataset_transform_profile", "qwen")
    try:
        return OLMO_CORE_DATASET_CACHE_VERSIONS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported --dataset_transform_profile: {profile}") from exc


def build_config_run_name(
    args: argparse.Namespace,
    learning_rate_token: str | None = None,
    optimizer: str | None = None,
) -> str:
    lr_token = learning_rate_token or format_float_token(args.learning_rate)
    optimizer_token = optimizer or normalized_optimizer_name(args)
    parts = [
        run_arch_token(args),
        f"tp{args.tensor_parallel_degree or 'auto'}",
        f"pp{args.pipeline_parallel_degree}",
        f"cp{args.context_parallel_degree}",
        f"seq{args.max_seq_length}",
        f"pdbs{args.per_device_batch_size}",
        f"ga{args.gradient_accumulation_steps}",
        f"lr{sanitize_slug_part(lr_token)}",
        sanitize_slug_part(optimizer_token),
        f"steps{args.max_train_steps}" if args.max_train_steps > 0 else "fullepoch",
    ]
    if args.context_parallel_degree > 1:
        parts.append("cpstyle-" + sanitize_slug_part(effective_context_parallel_style(args)))
    if args.pipeline_parallel_degree > 1:
        parts.append("sched-" + sanitize_slug_part(args.pipeline_schedule))
    split_points = effective_pipeline_split_points(
        args,
        args.model_arch if args.model_arch != "auto" else None,
    )
    if split_points:
        parts.append("split" + sanitize_slug_part("-".join(str(point) for point in split_points)))
    if args.rank_microbatch_size_sequences > 0:
        parts.append(f"mbs{args.rank_microbatch_size_sequences}")
    if args.lm_loss_implementation != "default":
        parts.append(sanitize_slug_part(args.lm_loss_implementation))
    if args.compile_model == "true":
        parts.append("compile")
    if args.force_compile_model == "true":
        parts.append("forcecompile")
    if args.activation_checkpointing_mode != "auto":
        parts.append("ac-" + sanitize_slug_part(args.activation_checkpointing_mode))
    if args.float8 == "true":
        parts.append("fp8")
        if args.float8_recipe != "recommended":
            parts.append(sanitize_slug_part(args.float8_recipe))
    if args.te_feedforward == "true":
        parts.append("teffn")
    if args.te_feedforward_glu == "true":
        parts.append("teglu")
    if args.te_layernorm == "true":
        parts.append("tenorm")
    if args.liger_layernorm == "true":
        parts.append("ligernorm")
    if args.liger_megatron_layernorm == "true":
        parts.append("ligermegatronnorm")
    if args.quack_layernorm == "true":
        parts.append("quacknorm")
    if args.feed_forward_chunk_size_tokens > 0:
        parts.append(f"ffchunk{args.feed_forward_chunk_size_tokens}")
    if args.torch_profiler == "true":
        parts.append("profiler")
    return truncate_slug("_".join(parts))


def build_auto_run_name(args: argparse.Namespace) -> str:
    env_name = os.environ.get("OLMO_RUN_NAME")
    if env_name:
        return truncate_slug(sanitize_slug_part(env_name))
    if getattr(args, "operator_mode", "false") == "true":
        return operator_run_name(args)
    if args.backend in VERL_BACKENDS:
        model_token = path_name_token(args.model_path or DEFAULT_MODEL_HF_REPOS.get(args.model_arch, ""), "model")
        data_token = path_name_token(args.dataset_path or args.rlcsd_train_file or "dataset", "data")
        return truncate_slug(
            "_".join(
                [
                    "verl-rlcsd",
                    args.rlcsd_method,
                    model_token,
                    data_token,
                    f"tp{args.rlcsd_rollout_tensor_parallel_size}",
                    f"n{args.rlcsd_group_size}",
                    f"resp{args.rlcsd_max_response_length}",
                    "lr-" + sanitize_slug_part(format_float_token(args.learning_rate)),
                ]
            ),
            max_length=180,
        )
    if args.backend in GRPO_BACKENDS:
        raw_argv = list(getattr(args, "_raw_argv", sys.argv[1:]))
        exp_name = cli_value(raw_argv, "--exp_name")
        preset = args.rl_preset or "custom"
        model_name = (
            cli_value(raw_argv, "--model_name_or_path")
            or args.model_path
            or DEFAULT_MODEL_HF_REPOS.get(args.model_arch, "")
        )
        model_token = path_name_token(model_name, "model")
        lr_token = cli_value(raw_argv, "--learning_rate") or format_float_token(args.learning_rate)
        learner_values = cli_values(raw_argv, "--num_learners_per_node")
        learners = "x".join(learner_values) if learner_values else "auto"
        vllm_engines = cli_value(raw_argv, "--vllm_num_engines") or "auto"
        vllm_tp = cli_value(raw_argv, "--vllm_tensor_parallel_size") or "auto"
        parts = [
            "grpo",
            sanitize_slug_part(exp_name or preset),
            model_token,
            f"learners{sanitize_slug_part(learners)}",
            f"vllm{sanitize_slug_part(vllm_engines)}xtp{sanitize_slug_part(vllm_tp)}",
            f"lr{sanitize_slug_part(lr_token)}",
        ]
        return truncate_slug("_".join(part for part in parts if part))
    return build_config_run_name(args)


def top_level_run_dir_enabled(args: argparse.Namespace) -> bool:
    if getattr(args, "run_dir_mode", "auto") == "none":
        return False
    if getattr(args, "internal_backend", None) is not None:
        return False
    if os.environ.get("OLMO_SWEEP_CHILD") == "1":
        return False
    if getattr(args, "operator_mode", "false") == "true":
        return False
    return True


def explicit_run_dir_name(args: argparse.Namespace) -> str | None:
    explicit = getattr(args, "run_dir_name", None) or os.environ.get("OLMO_RUN_DIR_NAME")
    if explicit:
        return truncate_slug(sanitize_slug_part(explicit), max_length=120)
    return None


def timestamp_run_dir_name() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_pid{os.getpid()}"


def run_dir_marker_path(args: argparse.Namespace, base_output: Path, base_logdir: Path) -> Path:
    payload = {
        "raw_argv": list(getattr(args, "_raw_argv", sys.argv[1:])),
        "base_output": str(base_output),
        "base_logdir": str(base_logdir),
        "master_addr": getattr(args, "master_addr", None) or os.environ.get("MASTER_ADDR"),
        "master_port": getattr(args, "master_port", None) or os.environ.get("MASTER_PORT"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    marker_root = Path(os.environ.get("OLMO_RUN_DIR_MARKER_ROOT", str(base_logdir / "_run_dir_markers"))).expanduser()
    return marker_root / f"{fingerprint}.json"


def read_run_dir_marker(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def write_run_dir_marker(path: Path, run_dir_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "run_dir_name": run_dir_name,
        "created_time": time.time(),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "pid": os.getpid(),
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def coordinated_timestamp_run_dir_name(args: argparse.Namespace, base_output: Path, base_logdir: Path) -> str:
    node_rank = effective_node_rank(args)
    num_nodes = effective_num_nodes(args)
    if node_rank is None or num_nodes <= 1:
        return timestamp_run_dir_name()
    marker_path = run_dir_marker_path(args, base_output, base_logdir)
    min_created_time = TRAIN_ENGINE_STARTED_AT - 15.0
    if node_rank == 0:
        run_dir_name = timestamp_run_dir_name()
        write_run_dir_marker(marker_path, run_dir_name)
        return run_dir_name

    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        marker = read_run_dir_marker(marker_path)
        if marker is not None:
            run_dir_name = marker.get("run_dir_name")
            created_time = marker.get("created_time", 0)
            if isinstance(run_dir_name, str) and isinstance(created_time, (int, float)) and created_time >= min_created_time:
                return run_dir_name
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for node_rank=0 run-dir marker: {marker_path}")


def resolve_top_level_run_dir_name(args: argparse.Namespace, base_output: Path, base_logdir: Path) -> str:
    return explicit_run_dir_name(args) or coordinated_timestamp_run_dir_name(args, base_output, base_logdir)


def apply_top_level_run_subdir(args: argparse.Namespace) -> None:
    if not top_level_run_dir_enabled(args):
        return
    base_output = Path(args.output_path).expanduser()
    base_logdir = Path(args.logdir).expanduser()
    run_dir_name = resolve_top_level_run_dir_name(args, base_output, base_logdir)
    args._base_output_path = str(base_output)
    args._base_logdir = str(base_logdir)
    args._run_dir_name = run_dir_name
    args.output_path = str(base_output / run_dir_name)
    args.logdir = str(base_logdir / run_dir_name)


def ensure_output_and_logdir(args: argparse.Namespace) -> None:
    if sweep_requested(args):
        if not args.output_path:
            args.output_path = str(Path(os.environ.get("OLMO_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser())
        if not args.logdir:
            args.logdir = str(Path(os.environ.get("OLMO_LOG_ROOT", DEFAULT_LOG_ROOT)).expanduser())
        apply_top_level_run_subdir(args)
        return

    run_name = build_auto_run_name(args)
    if not args.output_path:
        output_root = Path(os.environ.get("OLMO_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).expanduser()
        args.output_path = str(output_root / run_name)
    if not args.logdir:
        log_root = Path(os.environ.get("OLMO_LOG_ROOT", DEFAULT_LOG_ROOT)).expanduser()
        args.logdir = str(log_root / run_name)
    apply_top_level_run_subdir(args)


def build_sweep_trials(args: argparse.Namespace) -> list[SweepTrial]:
    learning_rates = parse_learning_rate_sweep(effective_learning_rates_arg(args), args.learning_rate)
    optimizers = parse_optimizer_sweep(effective_optimizers_arg(args), normalized_optimizer_name(args))
    trials: list[SweepTrial] = []
    for learning_rate_token, learning_rate in learning_rates:
        for optimizer in optimizers:
            index = len(trials)
            slug = (
                f"{index:03d}_"
                f"lr{sanitize_slug_part(learning_rate_token)}_"
                f"opt-{sanitize_slug_part(optimizer)}"
            )
            trials.append(
                SweepTrial(
                    index=index,
                    learning_rate=learning_rate,
                    learning_rate_token=learning_rate_token,
                    optimizer=optimizer,
                    slug=slug,
                )
            )
    return trials


def sweep_requested(args: argparse.Namespace) -> bool:
    return bool(effective_learning_rates_arg(args) or effective_optimizers_arg(args))


def should_run_sweep(args: argparse.Namespace) -> bool:
    return args.internal_backend is None and os.environ.get("OLMO_SWEEP_CHILD") != "1" and sweep_requested(args)


def strip_cli_args(argv: list[str], value_flags: set[str], bool_flags: set[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag_name = token.split("=", 1)[0]
        if flag_name in bool_flags:
            index += 1
            continue
        if flag_name in value_flags:
            index += 1
            if "=" not in token and index < len(argv):
                index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def strip_child_override_args(argv: list[str]) -> list[str]:
    return strip_cli_args(argv, CHILD_OVERRIDE_VALUE_FLAGS, CHILD_OVERRIDE_BOOL_FLAGS)


def append_cli_override(command: list[str], flag: str, value: str | int | float | Path) -> None:
    command.extend([flag, str(value)])


def sweep_status_writer(args: argparse.Namespace) -> bool:
    node_rank = effective_node_rank(args)
    return node_rank in {None, 0}


def append_sweep_status(status_path: Path, record: dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, sort_keys=True) + "\n")
        file_obj.flush()


def candidate_repo_dirs(repo_name: str) -> list[Path]:
    current_file = Path(__file__).resolve()
    candidates = [Path(f"/opt/{repo_name}"), Path(f"/app/{repo_name}"), Path.cwd() / repo_name]
    candidates.extend(parent / repo_name for parent in current_file.parents)
    return candidates


def find_open_instruct_dir(user_path: str | None) -> Path:
    candidates = [Path(user_path)] if user_path else []
    candidates.extend(candidate_repo_dirs("open-instruct"))
    for candidate in candidates:
        script = candidate / "open_instruct" / "olmo_core_finetune.py"
        if script.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find open_instruct/olmo_core_finetune.py. "
        "Set --open_instruct_dir or OPEN_INSTRUCT_DIR."
    )


def find_olmo_core_dir(user_path: str | None = None) -> Path:
    candidates = [Path(user_path)] if user_path else []
    candidates.extend(candidate_repo_dirs("OLMo-core"))
    candidates.extend(candidate_repo_dirs("olmo-core"))
    for candidate in candidates:
        if (candidate / "src" / "olmo_core").is_dir():
            return candidate
    raise FileNotFoundError("Could not find OLMo-core. Set OLMO_CORE_DIR or install it into /opt/OLMo-core.")


def import_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_config_name(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open() as file_obj:
        config = json.load(file_obj)
    if config.get("model_type") != "olmo3":
        return None
    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")
    if hidden_size == 4096 and num_layers == 32:
        return "olmo3_7B"
    if hidden_size == 5120 and num_layers == 64:
        return "olmo3_32B"
    return None


def normalize_model_arch(model_arch: str, model_path: Path) -> str:
    if model_arch != "auto":
        return model_arch
    config_name = infer_config_name(model_path)
    if config_name == "olmo3_7B":
        return "olmo3_7b"
    if config_name == "olmo3_32B":
        return "olmo3_32b"
    raise ValueError(f"Could not infer OLMo model architecture from {model_path}; pass --model_arch.")


def indexed_weight_shards_missing(model_path: Path, index_name: str) -> list[str] | None:
    index_path = model_path / index_name
    if not index_path.is_file():
        return None
    with index_path.open() as file_obj:
        index = json.load(file_obj)
    shards = sorted({str(name) for name in index.get("weight_map", {}).values()})
    if not shards:
        return []
    return sorted(name for name in shards if not (model_path / name).is_file())


def model_path_has_complete_hf_weights(model_path: Path) -> bool:
    if checkpoint_is_olmo_core(model_path):
        return True
    if not model_path.is_dir():
        return False
    for index_name in HF_WEIGHT_INDEX_FILES:
        missing = indexed_weight_shards_missing(model_path, index_name)
        if missing is not None:
            return len(missing) == 0
    return any((model_path / filename).is_file() for filename in HF_SINGLE_WEIGHT_FILES)


def read_hf_config_json(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def model_path_is_olmo3_sink(model_path: Path) -> bool:
    config = read_hf_config_json(model_path)
    architectures = config.get("architectures") or []
    return config.get("model_type") == "olmo3_sink" or "Olmo3SinkForCausalLM" in architectures


def olmo3_sink_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "olmo3_sink", "false")).strip().lower() == "true"


def olmo3_sink_cache_dir(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.olmo3_sink_cache_dir:
        return Path(args.olmo3_sink_cache_dir).expanduser().resolve()
    return cache_dir / "olmo3_sink_models"


def olmo3_sink_marker_name(args: argparse.Namespace, source: str, source_path: Path | None) -> str:
    config_path = source_path / "config.json" if source_path is not None else None
    payload = {
        "source": source,
        "source_config_mtime": config_path.stat().st_mtime if config_path is not None and config_path.exists() else None,
        "source_config_size": config_path.stat().st_size if config_path is not None and config_path.exists() else None,
        "sink_init_value": args.olmo3_sink_init_value,
        "version": "olmo3_sink_hf_v1",
    }
    return "olmo3_sink_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def prepare_olmo3_sink_model(args: argparse.Namespace, cache_dir: Path) -> Path:
    source_arg = str(args.model_path)
    source_path = Path(source_arg).expanduser()
    resolved_source_path = source_path.resolve() if source_path.exists() else None
    source = str(resolved_source_path) if resolved_source_path is not None else source_arg
    if resolved_source_path is not None and model_path_is_olmo3_sink(resolved_source_path):
        logging.info("Using existing olmo3_sink model: %s", resolved_source_path)
        args.model_path = str(resolved_source_path)
        return resolved_source_path

    target_root = olmo3_sink_cache_dir(args, cache_dir)
    source_token = sanitize_slug_part(resolved_source_path.name if resolved_source_path is not None else source)
    init_token = format_float_token(float(args.olmo3_sink_init_value))
    target = target_root / f"{source_token}_olmo3_sink_init{init_token}"
    marker_dir = cache_dir / "multinode_prepare_markers"

    def convert_once() -> str:
        if model_path_is_olmo3_sink(target) and model_path_has_complete_hf_weights(target):
            logging.info("Using cached olmo3_sink checkpoint: %s", target)
            return str(target)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Converting OLMo3 checkpoint to olmo3_sink: source=%s target=%s init=%s",
            source,
            target,
            args.olmo3_sink_init_value,
        )
        from olmo3_sink.convert import convert

        convert(source, str(target), float(args.olmo3_sink_init_value), "bfloat16")
        if not model_path_is_olmo3_sink(target) or not model_path_has_complete_hf_weights(target):
            raise RuntimeError(f"olmo3_sink conversion did not produce a complete HF checkpoint: {target}")
        return str(target)

    marker_name = olmo3_sink_marker_name(args, source, resolved_source_path)
    prepared = Path(run_once_on_node0(args, marker_dir, marker_name, convert_once)).expanduser().resolve()
    if not model_path_is_olmo3_sink(prepared) or not model_path_has_complete_hf_weights(prepared):
        raise RuntimeError(f"Prepared olmo3_sink model is incomplete: {prepared}")
    args.model_path = str(prepared)
    logging.info("Using olmo3_sink model path for verl/RLCSD: %s", args.model_path)
    return prepared


def missing_model_weights_message(model_path: Path) -> str:
    if not model_path.exists():
        return f"model_path does not exist: {model_path}"
    if not model_path.is_dir():
        return f"model_path is not a directory: {model_path}"
    for index_name in HF_WEIGHT_INDEX_FILES:
        missing = indexed_weight_shards_missing(model_path, index_name)
        if missing:
            return f"model_path is missing {len(missing)} shard(s) from {index_name}, first missing: {missing[0]}"
        if missing == []:
            return f"model_path has {index_name}, but it does not list any weight shards"
    if (model_path / "config.json").is_file():
        return f"model_path has config/tokenizer metadata but no HF weight files: {model_path}"
    return f"model_path has no recognized HF weight files: {model_path}"


def infer_model_arch_from_path_name(model_path: Path) -> str | None:
    text = str(model_path).lower()
    if re.search(r"(^|[^0-9])32[-_ ]?b([^0-9]|$)", text):
        return "olmo3_32b"
    if re.search(r"(^|[^0-9])7[-_ ]?b([^0-9]|$)", text):
        return "olmo3_7b"
    return None


def model_arch_for_download(args: argparse.Namespace, model_path: Path) -> str:
    if args.model_arch != "auto":
        return args.model_arch
    try:
        return normalize_model_arch(args.model_arch, model_path)
    except Exception:
        inferred = infer_model_arch_from_path_name(model_path)
        if inferred:
            return inferred
    raise ValueError(
        "Cannot choose a default HF model repo because --model_arch=auto and the local path has no readable OLMo "
        f"config. Pass --model_arch, --model_hf_repo, or edit DEFAULT_MODEL_HF_REPOS. model_path={model_path}"
    )


def hf_repo_for_model_download(args: argparse.Namespace, model_path: Path) -> str:
    if args.model_hf_repo:
        return args.model_hf_repo
    model_arch = model_arch_for_download(args, model_path)
    repo_id = DEFAULT_MODEL_HF_REPOS.get(model_arch)
    if not repo_id:
        raise ValueError(
            f"No default HF repo configured for model_arch={model_arch!r}. "
            "Set --model_hf_repo or edit DEFAULT_MODEL_HF_REPOS."
        )
    return repo_id


def path_is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_dir = path / f".write_probe_{os.getpid()}_{time.time_ns()}"
        probe_dir.mkdir(parents=True, exist_ok=False)
        probe = probe_dir / "probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        probe_dir.rmdir()
        return True
    except Exception:
        return False


def runtime_cache_root(args: argparse.Namespace | None = None) -> Path:
    explicit = getattr(args, "cache_dir", None) if args is not None else None
    root = explicit or os.environ.get("OLMO_RUNTIME_CACHE_ROOT") or DEFAULT_RUNTIME_CACHE_ROOT
    return Path(root).expanduser().resolve()


def default_cache_dir(args: argparse.Namespace, name: str = "cache") -> Path:
    return runtime_cache_root(args) / name


def default_olmo_checkpoint_cache(args: argparse.Namespace) -> Path:
    return runtime_cache_root(args) / "olmo_core_checkpoint"


def default_olmo_dataset_cache(args: argparse.Namespace) -> Path:
    return runtime_cache_root(args) / "olmo_core_dataset"


def configure_runtime_cache_environment(env: dict[str, str], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    preserve_existing = env.get("OLMO_PRESERVE_EXISTING_CACHE_ENV", "").strip().lower() in {"1", "true", "yes", "on"}
    ensure_writable_cache_env(env, "HF_HOME", cache_dir / "hf", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "HF_HUB_CACHE", cache_dir / "hf", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "HUGGINGFACE_HUB_CACHE", cache_dir / "hf", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "TRANSFORMERS_CACHE", cache_dir / "hf", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "HF_ASSETS_CACHE", cache_dir / "hf_assets", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "HF_XET_CACHE", cache_dir / "hf_xet", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "TORCHINDUCTOR_CACHE_DIR", cache_dir / "torchinductor", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "TRITON_CACHE_DIR", cache_dir / "triton", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "TORCH_HOME", cache_dir / "torch", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "XDG_CACHE_HOME", cache_dir / "xdg", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "FLASHINFER_CACHE_DIR", cache_dir / "flashinfer", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "VLLM_CACHE_ROOT", cache_dir / "vllm", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "WANDB_DIR", cache_dir / "wandb", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "WANDB_CACHE_DIR", cache_dir / "wandb_cache", preserve_existing=preserve_existing)
    ensure_writable_cache_env(env, "WANDB_CONFIG_DIR", cache_dir / "wandb_config", preserve_existing=preserve_existing)
    env.setdefault("TMPDIR", "/tmp")
    env.setdefault("TMP", "/tmp")
    env.setdefault("TEMP", "/tmp")


def ensure_writable_cache_env(
    env: dict[str, str],
    key: str,
    fallback: Path,
    preserve_existing: bool = False,
) -> None:
    current = env.get(key)
    if preserve_existing and current and path_is_writable_dir(Path(current).expanduser()):
        return
    fallback.mkdir(parents=True, exist_ok=True)
    env[key] = str(fallback)


def add_python_dependency_target(target_dir: Path) -> None:
    target = str(target_dir)
    if target not in sys.path:
        sys.path.insert(0, target)
    site.addsitedir(target)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts = [target, *[part for part in existing.split(os.pathsep) if part and part != target]]
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    else:
        os.environ["PYTHONPATH"] = target


def version_at_least(current: str | None, minimum: str) -> bool:
    if not current or current == "not-installed" or current.startswith("error:"):
        return False
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(current) >= Version(minimum)
        except InvalidVersion:
            return current >= minimum
    except Exception:
        return current >= minimum


def hf_download_dependency_upgrade_needed() -> bool:
    versions = package_versions()
    for package_name, minimum in MIN_HF_DOWNLOAD_DEPENDENCIES.items():
        if not version_at_least(versions.get(package_name), minimum):
            return True
    return False


def ensure_hf_download_dependencies(
    cache_dir: Path | None,
    upgrade: str = "true",
    package_specs: str = DEFAULT_HF_DEPENDENCY_UPGRADE_PACKAGES,
) -> None:
    if upgrade != "true":
        logging.info("HF dependency runtime upgrade disabled by --hf_dependency_upgrade=%s.", upgrade)
        return
    if os.environ.get("OLMO_HF_DEPENDENCY_UPGRADE_DONE") == "1":
        return

    cache_root = cache_dir or Path(DEFAULT_RUNTIME_CACHE_ROOT)
    target_dir = cache_root / "python_deps"
    pip_cache_dir = cache_root / "pip"
    if target_dir.is_dir():
        add_python_dependency_target(target_dir)

    versions = package_versions()
    logging.info(
        "HF download dependency check before upgrade: huggingface_hub=%s typer=%s click=%s target=%s",
        versions.get("huggingface_hub"),
        versions.get("typer"),
        versions.get("click"),
        target_dir,
    )
    if not hf_download_dependency_upgrade_needed():
        os.environ["OLMO_HF_DEPENDENCY_UPGRADE_DONE"] = "1"
        return

    specs = parse_csv_tokens(package_specs)
    if not specs:
        logging.warning("HF dependency runtime upgrade skipped because no package specs were provided.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    configure_runtime_cache_environment(env, cache_root)
    env["PIP_CACHE_DIR"] = str(pip_cache_dir)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target_dir),
        "--cache-dir",
        str(pip_cache_dir),
        "--disable-pip-version-check",
        *specs,
    ]
    try:
        run_command(command, env)
    except Exception as exc:
        logging.warning("HF dependency runtime upgrade failed; continuing with existing packages: %s", exc)
        return
    add_python_dependency_target(target_dir)
    os.environ["OLMO_HF_DEPENDENCY_UPGRADE_DONE"] = "1"
    logging.info("HF dependency runtime upgrade complete.")
    log_dependency_versions()


def download_hf_model_snapshot(
    repo_id: str,
    model_path: Path,
    cache_dir: Path | None,
    offline: bool = False,
    dependency_upgrade: str = "true",
    dependency_upgrade_packages: str = DEFAULT_HF_DEPENDENCY_UPGRADE_PACKAGES,
) -> None:
    if offline:
        raise FileNotFoundError(
            f"{missing_model_weights_message(model_path)}. Auto-download is disabled because --offline true."
        )
    model_path.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        configure_runtime_cache_environment(os.environ, cache_dir)
    ensure_hf_download_dependencies(cache_dir, dependency_upgrade, dependency_upgrade_packages)
    logging.warning("Auto-downloading missing model weights from %s into %s", repo_id, model_path)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            cache_dir=str(cache_dir / "hf") if cache_dir is not None else None,
            token=hf_log_token(),
        )
    except Exception as exc:
        logging.warning("huggingface_hub.snapshot_download failed for %s: %s", repo_id, exc)
        logging.warning("Retrying HF model download in a fresh Python subprocess.")
        env = os.environ.copy()
        if cache_dir is not None:
            configure_runtime_cache_environment(env, cache_dir)
        token = hf_log_token()
        if token:
            env.setdefault("HF_TOKEN", token)
        cache_path = str(cache_dir / "hf") if cache_dir is not None else ""
        code = (
            "import os, sys\n"
            "from huggingface_hub import snapshot_download\n"
            "repo_id, local_dir, cache_dir = sys.argv[1:4]\n"
            "token = os.environ.get('HF_TOKEN') or None\n"
            "snapshot_download(repo_id=repo_id, local_dir=local_dir, cache_dir=cache_dir or None, token=token)\n"
        )
        command = [sys.executable, "-c", code, repo_id, str(model_path), cache_path]
        run_command(command, env)


def ensure_model_weights(args: argparse.Namespace, model_path: Path, cache_dir: Path | None = None) -> Path:
    if model_path_has_complete_hf_weights(model_path):
        logging.info("Using complete local model weights at %s", model_path)
        return model_path
    repo_id = hf_repo_for_model_download(args, model_path)
    logging.warning("%s; attempting HF auto-download from %s.", missing_model_weights_message(model_path), repo_id)
    download_hf_model_snapshot(
        repo_id,
        model_path,
        cache_dir,
        offline=False,
        dependency_upgrade=getattr(args, "hf_dependency_upgrade", "true"),
        dependency_upgrade_packages=getattr(
            args,
            "hf_dependency_upgrade_packages",
            DEFAULT_HF_DEPENDENCY_UPGRADE_PACKAGES,
        ),
    )
    if not model_path_has_complete_hf_weights(model_path):
        raise FileNotFoundError(
            f"HF auto-download from {repo_id} finished, but weights are still incomplete: "
            f"{missing_model_weights_message(model_path)}"
        )
    return model_path


def ensure_model_weights_once(args: argparse.Namespace, model_path: Path, cache_dir: Path) -> Path:
    marker_basis = f"{model_path}|{args.model_arch}|{args.model_hf_repo or ''}"
    marker_name = "model_weights_" + hashlib.sha1(marker_basis.encode("utf-8")).hexdigest()[:12]
    return Path(
        run_once_on_node0(
            args,
            cache_dir / "multinode_prepare_markers",
            marker_name,
            lambda: ensure_model_weights(args, model_path, cache_dir),
        )
    )


def validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    config_path = model_path / "config.json"
    if config_path.is_file():
        if not model_path_has_complete_hf_weights(model_path):
            raise FileNotFoundError(missing_model_weights_message(model_path))


def iter_json_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open() as file_obj:
            for line_no, line in enumerate(file_obj, start=1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
        return

    with path.open() as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict) and isinstance(payload.get("train"), list):
        yield from payload["train"]
    else:
        raise ValueError(f"Unsupported JSON dataset structure in {path}; expected a list or a dict with train list.")


def prepare_json_messages_dataset(dataset_file: Path) -> Path:
    if dataset_file.suffix not in {".json", ".jsonl"}:
        return dataset_file
    records_seen = 0
    for record in iter_json_records(dataset_file):
        if not isinstance(record, dict):
            raise ValueError(f"Each dataset row must be a JSON object in {dataset_file}")
        records_seen += 1
    logging.info("Using raw JSON messages dataset at %s (%d records, no message rewriting)", dataset_file, records_seen)
    return dataset_file


def prepare_raw_dataset(dataset_path: Path, cache_dir: Path) -> str:
    if dataset_path.is_file():
        if dataset_path.suffix not in SUPPORTED_DATA_SUFFIXES:
            raise ValueError(f"Unsupported dataset file type: {dataset_path.suffix}")
        return str(prepare_json_messages_dataset(dataset_path))

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")

    if (dataset_path / "train").is_dir():
        return str(dataset_path)

    raw_files = sorted(path for path in dataset_path.iterdir() if path.suffix in SUPPORTED_DATA_SUFFIXES)
    if not raw_files:
        raise FileNotFoundError(f"No JSON/JSONL/parquet files found in {dataset_path}")
    if len(raw_files) == 1:
        return str(prepare_json_messages_dataset(raw_files[0]))

    suffixes = {path.suffix for path in raw_files}
    if suffixes <= {".json", ".jsonl"}:
        dataset_format = "json"
    elif suffixes == {".parquet"}:
        dataset_format = "parquet"
    else:
        raise ValueError(f"Cannot mix dataset formats in one directory: {sorted(suffixes)}")

    prepared_root = cache_dir / "prepared_raw_dataset"
    prepared_train = prepared_root / "train"
    if prepared_train.is_dir():
        logging.info("Using existing prepared dataset at %s", prepared_root)
        return str(prepared_root)

    prepared_files = [prepare_json_messages_dataset(path) for path in raw_files]
    logging.info("Preparing %d raw %s files for Open-Instruct", len(prepared_files), dataset_format)
    from datasets import load_dataset

    dataset = load_dataset(dataset_format, data_files=[str(path) for path in prepared_files], split="train")
    prepared_train.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(prepared_train))
    return str(prepared_root)


def raw_dataset_marker_name(args: argparse.Namespace, dataset_path: Path) -> str:
    payload = {
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "dataset_split": getattr(args, "dataset_split", "train"),
        "dataset_weight": getattr(args, "dataset_weight", "1.0"),
    }
    try:
        stat = dataset_path.stat()
    except OSError:
        stat = None
    if stat is not None:
        payload.update({"mtime_ns": stat.st_mtime_ns, "size": stat.st_size})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"raw_dataset_{digest}"


def olmo_core_checkpoint_marker_name(
    args: argparse.Namespace,
    model_path: Path,
    checkpoint_cache: Path,
    olmo_core_dir: Path,
    model_arch: str,
) -> str:
    conversion_script = olmo_core_dir / "src" / "examples" / "huggingface" / "convert_checkpoint_from_hf.py"
    payload = {
        "model_path": str(model_path.expanduser().resolve()),
        "checkpoint_cache": str(checkpoint_cache.expanduser().resolve()),
        "model_arch": model_arch,
        "convert_validation": getattr(args, "convert_validation", "true"),
        "conversion_script": str(conversion_script.expanduser().resolve()),
    }
    for key, path in (("model_path", model_path), ("conversion_script", conversion_script)):
        try:
            stat = path.stat()
        except OSError:
            continue
        payload[f"{key}_mtime_ns"] = stat.st_mtime_ns
        payload[f"{key}_size"] = stat.st_size
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"olmo_core_checkpoint_{digest}"


def olmo_core_dataset_marker_name(
    args: argparse.Namespace,
    dataset_ref: str,
    dataset_cache: Path,
    open_instruct_dir: Path,
) -> str:
    conversion_script = open_instruct_dir / "scripts" / "data" / "convert_sft_data_for_olmocore.py"
    tokenizer_path = Path(args.tokenizer_path or args.model_path).expanduser()
    chat_template_model = effective_chat_template_model(args)
    payload = {
        "dataset_ref": str(Path(dataset_ref).expanduser().resolve()),
        "dataset_weight": getattr(args, "dataset_weight", "1.0"),
        "dataset_split": getattr(args, "dataset_split", "train"),
        "dataset_cache": str(dataset_cache.expanduser().resolve()),
        "cache_version": olmo_core_dataset_cache_version(args),
        "max_seq_length": olmo_core_sequence_length(args.max_seq_length),
        "dataset_backend": getattr(args, "dataset_backend", "auto"),
        "tokenizer_name_or_path": str(tokenizer_path.resolve() if tokenizer_path.exists() else tokenizer_path),
        "chat_template_model": chat_template_model or "",
        "chat_template_name": getattr(args, "chat_template_name", "") or "",
        "dataset_transform_fn": dataset_transform_names(args),
        "conversion_script": str(conversion_script.expanduser().resolve()),
    }
    paths_to_stat: list[tuple[str, Path]] = [
        ("dataset_ref", Path(dataset_ref)),
        ("tokenizer", tokenizer_path),
        ("conversion_script", conversion_script),
    ]
    if chat_template_model:
        chat_template_path = Path(chat_template_model).expanduser()
        if chat_template_path.exists():
            paths_to_stat.append(("chat_template_model", chat_template_path))
    for key, path in paths_to_stat:
        try:
            stat = path.stat()
        except OSError:
            continue
        payload[f"{key}_mtime_ns"] = stat.st_mtime_ns
        payload[f"{key}_size"] = stat.st_size
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"olmo_core_dataset_{digest}"


def truncate_for_log(value: object, max_chars: int) -> str:
    text = str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def tensor_or_array_to_list(value: object) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "flatten"):
        value = value.flatten()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def first_jsonlike_record(path: Path) -> dict[str, object] | None:
    if path.suffix in {".json", ".jsonl"}:
        for record in iter_json_records(path):
            if isinstance(record, dict):
                return record
        return None

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            columns = [
                name
                for name in ("messages", "tool_schema", "question", "answer", "id", "question_uuid")
                if name in parquet_file.schema_arrow.names
            ]
            batch_iter = parquet_file.iter_batches(batch_size=1, columns=columns or None)
            for batch in batch_iter:
                rows = batch.to_pylist()
                return rows[0] if rows else None
        except Exception:
            from datasets import load_dataset

            dataset = load_dataset("parquet", data_files=str(path), split="train[:1]")
            return dict(dataset[0]) if len(dataset) else None

    return None


def first_dataset_record(dataset_path: Path) -> dict[str, object] | None:
    if dataset_path.is_file():
        return first_jsonlike_record(dataset_path)

    if not dataset_path.is_dir() or dataset_is_olmo_core_numpy(dataset_path):
        return None

    if (dataset_path / "train").is_dir():
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(str(dataset_path / "train"))
            return dict(dataset[0]) if len(dataset) else None
        except Exception:
            return None

    for path in sorted(dataset_path.iterdir()):
        if path.suffix in SUPPORTED_DATA_SUFFIXES:
            record = first_jsonlike_record(path)
            if record is not None:
                return record
    return None


def load_preview_tokenizer(
    tokenizer_path: Path,
    chat_template_name: str | None,
    chat_template_model: str | None,
    open_instruct_dir: Path,
):
    if str(open_instruct_dir) not in sys.path:
        sys.path.insert(0, str(open_instruct_dir))
    from open_instruct import dataset_transformation

    tokenizer_config = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path=str(tokenizer_path),
        chat_template_name=chat_template_name,
        chat_template_model=chat_template_model,
    )
    return tokenizer_config.tokenizer, dataset_transformation


def log_tokenizer_summary(
    tokenizer: object,
    tokenizer_path: Path,
    chat_template_name: str | None,
    chat_template_model: str | None,
    max_chars: int,
) -> None:
    chat_template = getattr(tokenizer, "chat_template", None)
    logging.info(
        "Tokenizer preview: path=%s class=%s vocab_size=%s len=%s bos=%r/%s eos=%r/%s pad=%r/%s chat_template=%s chat_template_model=%s",
        tokenizer_path,
        tokenizer.__class__.__name__,
        getattr(tokenizer, "vocab_size", "unknown"),
        len(tokenizer) if hasattr(tokenizer, "__len__") else "unknown",
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        chat_template_name or "tokenizer-default",
        chat_template_model or "none",
    )
    if chat_template:
        logging.info("Tokenizer chat template head: %s", truncate_for_log(chat_template.replace("\n", "\\n"), max_chars))


def log_raw_tokenized_sample(
    args: argparse.Namespace,
    dataset_path: Path,
    tokenizer_path: Path,
    open_instruct_dir: Path,
) -> None:
    record = first_dataset_record(dataset_path)
    if record is None:
        logging.info("Tokenizer preview: no raw JSON/JSONL/parquet sample found at %s", dataset_path)
        return

    chat_template_model = effective_chat_template_model(args)
    tokenizer, dataset_transformation = load_preview_tokenizer(
        tokenizer_path,
        args.chat_template_name,
        chat_template_model,
        open_instruct_dir,
    )
    log_tokenizer_summary(
        tokenizer,
        tokenizer_path,
        args.chat_template_name,
        chat_template_model,
        args.tokenized_sample_max_chars,
    )

    tools = dataset_transformation.normalize_optional_tool_schema(record.get("tool_schema"))
    transform_profile = getattr(args, "dataset_transform_profile", "qwen")
    if transform_profile == "olmo":
        messages = dataset_transformation.normalize_reasoning_messages(record.get("messages"))
        rendered_kwargs = {
            "conversation": messages,
            "tokenize": False,
            "add_generation_prompt": False,
        }
        tokenized_row = {"messages": messages}
        tokenized = dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            tokenized_row,
            tokenizer,
            olmo_core_sequence_length(args.max_seq_length),
        )
        if tools is not None:
            logging.warning(
                "Tokenizer preview found tool_schema but --dataset_transform_profile=olmo ignores Qwen tools."
            )
    else:
        messages = dataset_transformation.prepare_qwen_messages_for_template(
            record.get("messages"),
            tools,
        )
        rendered_kwargs = {
            "conversation": messages,
            "tokenize": False,
            "add_generation_prompt": False,
        }
        tokenized_row = {"messages": messages, "tool_schema": tools}
        if tools is not None:
            rendered_kwargs["tools"] = tools
        tokenized = dataset_transformation.sft_qwen_messages_tokenize_and_truncate_v1(
            tokenized_row,
            tokenizer,
            olmo_core_sequence_length(args.max_seq_length),
        )
    rendered = tokenizer.apply_chat_template(**rendered_kwargs)
    if tools is not None:
        logging.info(
            "Tokenizer preview tool schema: %s",
            truncate_for_log(json.dumps(tools, ensure_ascii=False), args.tokenized_sample_max_chars),
        )
    input_ids = tensor_or_array_to_list(tokenized[dataset_transformation.INPUT_IDS_KEY])
    labels = tensor_or_array_to_list(tokenized[dataset_transformation.LABELS_KEY])
    trainable = sum(1 for label in labels if label != dataset_transformation.MASKED_TOKEN_VALUE)
    preview_ids = input_ids[: max(0, args.tokenized_sample_max_tokens)]
    logging.info(
        "Tokenizer preview raw sample: source=%s messages=%s",
        dataset_path,
        truncate_for_log(json.dumps(messages, ensure_ascii=False), args.tokenized_sample_max_chars),
    )
    logging.info(
        "Tokenizer preview rendered text (%d chars):\n%s",
        len(rendered),
        truncate_for_log(rendered, args.tokenized_sample_max_chars),
    )
    logging.info(
        "Tokenizer preview token counts: total=%d trainable_labels=%d masked_labels=%d",
        len(input_ids),
        trainable,
        len(labels) - trainable,
    )
    logging.info("Tokenizer preview first %d token ids: %s", len(preview_ids), preview_ids)
    logging.info(
        "Tokenizer preview decoded first %d tokens:\n%s",
        len(preview_ids),
        truncate_for_log(tokenizer.decode(preview_ids), args.tokenized_sample_max_chars),
    )


def log_numpy_tokenized_sample(
    args: argparse.Namespace,
    dataset_path: Path,
    tokenizer_path: Path,
    open_instruct_dir: Path,
) -> None:
    import numpy as np

    token_files = sorted(dataset_path.glob("token_ids_part_*.npy"))
    label_files = sorted(dataset_path.glob("labels_mask_part_*.npy"))
    if not token_files:
        logging.info("Tokenizer preview: no token_ids_part_*.npy files found in %s", dataset_path)
        return

    cache_tokenizer_path = dataset_path / "tokenizer"
    effective_tokenizer_path = cache_tokenizer_path if cache_tokenizer_path.is_dir() else tokenizer_path
    tokenizer, _dataset_transformation = load_preview_tokenizer(
        effective_tokenizer_path,
        None if cache_tokenizer_path.is_dir() else args.chat_template_name,
        None if cache_tokenizer_path.is_dir() else effective_chat_template_model(args),
        open_instruct_dir,
    )
    log_tokenizer_summary(
        tokenizer,
        effective_tokenizer_path,
        "cached-tokenizer" if cache_tokenizer_path.is_dir() else args.chat_template_name,
        None if cache_tokenizer_path.is_dir() else effective_chat_template_model(args),
        args.tokenized_sample_max_chars,
    )

    def _select_token_dtype(vocab_size: int):
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            if (vocab_size - 1) <= np.iinfo(dtype).max:
                return dtype
        return np.uint64

    def _load_array(path: Path, dtype):
        try:
            return np.load(path, mmap_mode="r")
        except ValueError:
            return np.memmap(path, mode="r", dtype=dtype)

    max_tokens = max(0, args.tokenized_sample_max_tokens)
    token_dtype = _select_token_dtype(len(tokenizer))
    token_memmap = _load_array(token_files[0], token_dtype)
    preview_ids = [int(token_id) for token_id in token_memmap[:max_tokens].tolist()]
    trainable = "unknown"
    if label_files:
        label_memmap = _load_array(label_files[0], np.bool_)
        trainable = int(np.asarray(label_memmap[:max_tokens]).sum())
    logging.info("Tokenizer preview packed dataset: source=%s token_file=%s", dataset_path, token_files[0])
    logging.info("Tokenizer preview first %d packed token ids: %s", len(preview_ids), preview_ids)
    logging.info("Tokenizer preview trainable labels in preview window: %s", trainable)
    logging.info(
        "Tokenizer preview decoded first %d packed tokens:\n%s",
        len(preview_ids),
        truncate_for_log(tokenizer.decode(preview_ids), args.tokenized_sample_max_chars),
    )


def maybe_log_tokenized_sample(
    args: argparse.Namespace,
    raw_dataset_path: Path,
    tokenizer_path: Path,
    open_instruct_dir: Path,
    converted_dataset_path: Path | None = None,
) -> None:
    if args.log_tokenized_sample != "true":
        return
    if args.tokenized_sample_max_tokens < 0:
        raise ValueError("--tokenized_sample_max_tokens must be >= 0")

    raw_is_packed = dataset_is_olmo_core_numpy(raw_dataset_path)
    try:
        if raw_is_packed:
            log_numpy_tokenized_sample(args, raw_dataset_path, tokenizer_path, open_instruct_dir)
        else:
            log_raw_tokenized_sample(args, raw_dataset_path, tokenizer_path, open_instruct_dir)
    except Exception:
        logging.exception("Tokenizer raw preview failed")

    if converted_dataset_path is None or raw_is_packed or not dataset_is_olmo_core_numpy(converted_dataset_path):
        return
    try:
        log_numpy_tokenized_sample(args, converted_dataset_path, tokenizer_path, open_instruct_dir)
    except Exception:
        logging.exception("Tokenizer packed preview failed")


def cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def int_from_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            logging.warning("Ignoring non-integer %s=%r.", name, value)
    return None


def int_from_env_with_name(*names: str) -> tuple[int | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value), name
        except ValueError:
            logging.warning("Ignoring non-integer %s=%r.", name, value)
    return None, None


def strip_inherited_torchrun_env(env: dict[str, str]) -> None:
    """Remove scheduler rank aliases before spawning our internal torchrun."""
    for name in INTERNAL_TORCHRUN_ENV_KEYS:
        env.pop(name, None)


def effective_num_gpus(args: argparse.Namespace) -> int:
    if hasattr(args, "_resolved_num_gpus"):
        return args._resolved_num_gpus
    args._resolved_num_gpus = args.num_gpus or cuda_device_count() or 1
    return args._resolved_num_gpus


def effective_num_nodes(args: argparse.Namespace) -> int:
    if hasattr(args, "_resolved_num_nodes"):
        return args._resolved_num_nodes
    if args.num_nodes and args.num_nodes > 0:
        args._resolved_num_nodes = args.num_nodes
        return args._resolved_num_nodes

    explicit_nodes = int_from_env("PBS_NUM_NODES", "PBS_NNODES", "NUM_NODES", "SLURM_NNODES")
    if explicit_nodes:
        args._resolved_num_nodes = explicit_nodes
        return args._resolved_num_nodes

    world_size = int_from_env("WORLD_SIZE")
    if not world_size:
        args._resolved_num_nodes = 1
        return args._resolved_num_nodes
    if args.world_size_mode == "nodes":
        args._resolved_num_nodes = world_size
        return args._resolved_num_nodes

    num_gpus = effective_num_gpus(args)
    if args.world_size_mode == "processes":
        if world_size % num_gpus != 0:
            raise ValueError(
                f"WORLD_SIZE={world_size} is not divisible by GPUs per node={num_gpus}; "
                "cannot interpret it as total torch process count."
            )
        args._resolved_num_nodes = max(1, world_size // num_gpus)
        return args._resolved_num_nodes

    if world_size > num_gpus and world_size % num_gpus == 0:
        inferred_nodes = world_size // num_gpus
        logging.warning(
            "Interpreting WORLD_SIZE=%d as total torch processes, so num_nodes=%d with %d GPUs per node. "
            "Pass --world_size_mode nodes if WORLD_SIZE is already the node count.",
            world_size,
            inferred_nodes,
            num_gpus,
        )
        args._resolved_num_nodes = max(1, inferred_nodes)
        return args._resolved_num_nodes
    args._resolved_num_nodes = world_size
    return args._resolved_num_nodes


def effective_node_rank(args: argparse.Namespace) -> int | None:
    if hasattr(args, "_resolved_node_rank"):
        return args._resolved_node_rank
    if args.node_rank is not None:
        args._resolved_node_rank = args.node_rank
        return args._resolved_node_rank
    raw_rank, source_name = int_from_env_with_name("GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID")
    if raw_rank is None and os.environ.get("LOCAL_RANK") in {None, ""}:
        raw_rank, source_name = int_from_env_with_name("RANK")
    if raw_rank is None:
        args._resolved_node_rank = None
        return args._resolved_node_rank

    if source_name in {"GLOBAL_RANK", "RANK"} and args.world_size_mode != "nodes":
        num_nodes = effective_num_nodes(args)
        num_gpus = effective_num_gpus(args)
        if raw_rank >= num_nodes and num_gpus > 0 and raw_rank % num_gpus == 0:
            inferred_node_rank = raw_rank // num_gpus
            logging.warning(
                "Interpreting %s=%d as a global torch process rank, so node_rank=%d. "
                "Pass --node_rank explicitly if this value is already the node rank.",
                source_name,
                raw_rank,
                inferred_node_rank,
            )
            args._resolved_node_rank = inferred_node_rank
            return args._resolved_node_rank
    args._resolved_node_rank = raw_rank
    return args._resolved_node_rank


def effective_master_port(args: argparse.Namespace) -> int:
    if args.master_port is not None:
        return args.master_port
    return int_from_env("MASTER_PORT") or 29400


def effective_master_addr(args: argparse.Namespace) -> str | None:
    if args.master_addr:
        return args.master_addr
    for name in ("MASTER_ADDR", "PBS_MASTER_ADDR", "SLURM_LAUNCH_NODE_IPADDR"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def train_engine_git_commit() -> str:
    source_path = Path(__file__).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_path.parent.parent), "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception as exc:
        return f"unknown:{type(exc).__name__}"
    commit = completed.stdout.strip()
    return commit or f"unknown:returncode-{completed.returncode}"


def log_train_engine_runtime_info(args: argparse.Namespace) -> None:
    source_path = Path(__file__).resolve()
    logging.info(
        "Training source: train_engine=%s git_commit=%s submissions_ref=%s no_fetch_update=%s",
        source_path,
        train_engine_git_commit(),
        getattr(args, "submissions_ref", None),
        getattr(args, "no_fetch_update", None),
    )
    log_dependency_versions()


def launcher_environment_summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "effective_num_nodes": effective_num_nodes(args),
        "effective_num_gpus": effective_num_gpus(args),
        "effective_node_rank": effective_node_rank(args),
        "world_size_mode": args.world_size_mode,
        "master_addr": args.master_addr or os.environ.get("MASTER_ADDR"),
        "master_port": str(effective_master_port(args)),
        "train_engine_path": str(Path(__file__).resolve()),
        "train_engine_git_commit": train_engine_git_commit(),
    }


def effective_tensor_parallel_degree(args: argparse.Namespace, model_arch: str | None = None) -> int:
    if args.tensor_parallel_degree > 0:
        return args.tensor_parallel_degree
    arch = model_arch or getattr(args, "model_arch", "auto")
    if arch == "olmo3_32b" and effective_num_gpus(args) >= 8:
        return 8
    return 1


def effective_context_parallel_degree(args: argparse.Namespace) -> int:
    return max(1, args.context_parallel_degree)


def effective_context_parallel_style(args: argparse.Namespace) -> str:
    style = getattr(args, "context_parallel_style", "ring") or "ring"
    style = style.strip().lower().replace("-", "_")
    if style == "zigzag":
        return "zig_zag"
    if style == "ring":
        return "llama3"
    return style


def effective_context_parallel_head_stride(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "context_parallel_head_stride", 4) or 4))


def build_transformer_context_parallel_config(config_cls: object, args: argparse.Namespace, degree: int) -> object | None:
    if degree <= 1:
        return None
    style = effective_context_parallel_style(args)
    if style == "ulysses":
        return config_cls.ulysses(degree=degree)

    head_stride = effective_context_parallel_head_stride(args)
    if style == "llama3":
        return config_cls.llama3(degree=degree, head_stride=head_stride)
    if style == "zig_zag":
        return config_cls.zig_zag(degree=degree, head_stride=head_stride)
    raise ValueError(f"Unsupported context parallel style: {getattr(args, 'context_parallel_style', style)!r}")


def effective_pipeline_parallel_degree(args: argparse.Namespace) -> int:
    return max(1, args.pipeline_parallel_degree)


def effective_data_parallel_degree(args: argparse.Namespace, model_arch: str | None = None) -> int:
    world_size = effective_num_nodes(args) * effective_num_gpus(args)
    model_parallel = (
        effective_tensor_parallel_degree(args, model_arch)
        * effective_context_parallel_degree(args)
        * effective_pipeline_parallel_degree(args)
    )
    if world_size % model_parallel != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by TP*CP*PP={model_parallel} "
            f"(TP={effective_tensor_parallel_degree(args, model_arch)}, "
            f"CP={effective_context_parallel_degree(args)}, PP={effective_pipeline_parallel_degree(args)})"
        )
    return world_size // model_parallel


def validate_parallel_topology(args: argparse.Namespace, model_arch: str | None = None) -> None:
    world_size = effective_num_nodes(args) * effective_num_gpus(args)
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)
    model_parallel = tp_degree * cp_degree * pp_degree
    if world_size % model_parallel == 0:
        return
    hint = ""
    if effective_num_nodes(args) == 1 and effective_num_gpus(args) == 8 and tp_degree == 8 and pp_degree == 3:
        hint = " TP=8,PP=3 needs 24 GPUs; use the 3-node host or a one-node probe such as TP=4,PP=2."
    raise ValueError(
        f"world_size={world_size} must be divisible by TP*CP*PP={model_parallel} "
        f"(TP={tp_degree}, CP={cp_degree}, PP={pp_degree}).{hint}"
    )


def validate_pipeline_split_points(
    args: argparse.Namespace,
    pp_degree: int,
    n_layers: int | None = None,
    model_arch: str | None = None,
) -> None:
    split_points = effective_pipeline_split_points(args, model_arch or getattr(args, "model_arch", None), n_layers)
    if not split_points:
        return
    if pp_degree <= 1:
        raise ValueError("--pipeline_split_points requires --pipeline_parallel_degree > 1.")
    num_stages = len(split_points) + 1
    if num_stages % pp_degree != 0:
        raise ValueError(
            f"--pipeline_split_points creates {num_stages} stages, which is not divisible by "
            f"--pipeline_parallel_degree={pp_degree}."
        )
    if n_layers is not None and split_points[-1] > n_layers:
        raise ValueError(
            f"--pipeline_split_points last value {split_points[-1]} exceeds model n_layers={n_layers}."
        )


def validate_training_features(args: argparse.Namespace, model_arch: str | None = None) -> None:
    if args.rank_microbatch_size_sequences < 0:
        raise ValueError("--rank_microbatch_size_sequences must be >= 0.")
    if args.global_batch_size_tokens < 0:
        raise ValueError("--global_batch_size_tokens must be >= 0.")
    if args.global_batch_size_sequences < 0:
        raise ValueError("--global_batch_size_sequences must be >= 0.")
    if args.global_batch_size_tokens > 0 and args.global_batch_size_sequences > 0:
        raise ValueError(
            "--global_batch_size_tokens and --global_batch_size_sequences are mutually exclusive."
        )
    if args.context_parallel_head_stride <= 0:
        raise ValueError("--context_parallel_head_stride must be positive.")
    if args.activation_checkpoint_block_interval <= 0:
        raise ValueError("--activation_checkpoint_block_interval must be positive.")
    if args.torch_profiler == "true":
        profiler_values = {
            "--torch_profiler_skip_first": args.torch_profiler_skip_first,
            "--torch_profiler_wait": args.torch_profiler_wait,
            "--torch_profiler_warmup": args.torch_profiler_warmup,
            "--torch_profiler_active": args.torch_profiler_active,
            "--torch_profiler_repeat": args.torch_profiler_repeat,
        }
        for name, value in profiler_values.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
        if args.torch_profiler_active <= 0:
            raise ValueError("--torch_profiler_active must be > 0.")
        if args.torch_profiler_repeat <= 0:
            raise ValueError("--torch_profiler_repeat must be > 0.")
    if args.activation_checkpointing_mode == "selected_modules":
        parse_csv_tokens(args.activation_checkpoint_modules)
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)
    validate_pipeline_split_points(
        args,
        pp_degree,
        DEFAULT_MODEL_N_LAYERS.get(model_arch or ""),
        model_arch,
    )
    if args.attention_sink == "true":
        if cp_degree > 1:
            raise ValueError("--attention_sink true does not currently support --context_parallel_degree > 1.")
        attn_implementation = normalize_attn_implementation(args.attn_implementation)
        if attn_implementation is not None and attn_implementation not in {"torch", "flash_3"}:
            raise ValueError(
                "--attention_sink true currently supports only torch/eager and flash_3 backends; "
                f"got {args.attn_implementation!r}."
            )
    if args.tensor_parallel_async == "true":
        if tp_degree <= 1:
            raise ValueError("--tensor_parallel_async true requires --tensor_parallel_degree > 1.")
        if args.compile_model != "true":
            raise ValueError("--tensor_parallel_async true requires --compile_model true.")
    if args.float8 == "true" and tp_degree > 1:
        logging.warning(
            "Float8Linear with tensor parallelism is enabled (TP=%d). "
            "Use torchao>=0.17.0; older torchao builds failed in DTensor all-to-all. "
            "Long-context 32B runs may still OOM before the first step.",
            tp_degree,
        )
    if args.te_feedforward == "true":
        if args.float8 == "true":
            raise ValueError("--te_feedforward true cannot be combined with --float8 true.")
        if tp_degree > 1:
            raise ValueError(
                "--te_feedforward true currently supports only TP=1 because TE feed-forward "
                "internal TP is not yet compatible with the current OLMo-core checkpoint load path."
            )
    if args.te_feedforward_glu == "true":
        if args.float8 == "true":
            raise ValueError("--te_feedforward_glu true cannot be combined with --float8 true.")
        if args.te_feedforward == "true":
            raise ValueError("--te_feedforward_glu true cannot be combined with --te_feedforward true.")
        logging.warning(
            "Using experimental Transformer Engine fused feed-forward GLU activation. "
            "This keeps OLMo-core linears/checkpoints and is intended for TP>1 tests."
        )
    enabled_norm_modes = [
        name
        for name, enabled in (
            ("--te_layernorm", args.te_layernorm == "true"),
            ("--liger_layernorm", args.liger_layernorm == "true"),
            ("--liger_megatron_layernorm", args.liger_megatron_layernorm == "true"),
            ("--quack_layernorm", args.quack_layernorm == "true"),
        )
        if enabled
    ]
    if len(enabled_norm_modes) > 1:
        raise ValueError(
            "RMSNorm backend modes are mutually exclusive: "
            + ", ".join(enabled_norm_modes)
        )
    if args.te_layernorm == "true":
        if args.liger_layernorm == "true" or args.liger_megatron_layernorm == "true":
            raise ValueError(
                "--te_layernorm true cannot be combined with Liger layernorm modes."
            )
        logging.warning(
            "Using experimental Transformer Engine RMSNorm kernels. "
            "This preserves OLMo-core norm modules/checkpoint keys and is intended for TP>1 tests."
        )
    if args.liger_layernorm == "true":
        if args.liger_megatron_layernorm == "true":
            raise ValueError(
                "--liger_layernorm true cannot be combined with --liger_megatron_layernorm true."
            )
        logging.warning(
            "Using experimental Liger RMSNorm kernels. "
            "This preserves OLMo-core norm modules/checkpoint keys and is intended for TP>1 tests."
        )
    if args.liger_megatron_layernorm == "true":
        logging.warning(
            "Using experimental Liger Megatron RMSNorm kernels. "
            "This preserves OLMo-core norm modules/checkpoint keys and tests Liger's Megatron path."
        )
    if args.quack_layernorm == "true":
        logging.warning(
            "Using experimental Quack CuTe RMSNorm kernels. "
            "This preserves OLMo-core norm modules/checkpoint keys and is intended for TP>1 tests."
        )
    if args.feed_forward_chunk_size_tokens < 0:
        raise ValueError("--feed_forward_chunk_size_tokens must be >= 0.")
    if args.feed_forward_chunk_size_tokens > 0:
        if tp_degree > 1:
            raise ValueError(
                "--feed_forward_chunk_size_tokens is not currently safe with TP>1. "
                "Use --feed_forward_memory_profile true to diagnose the TP OOM first."
            )
        logging.warning(
            "Using experimental OLMo-core feed-forward token chunking with chunk_size_tokens=%d. "
            "This lowers w1/w3 activation peaks at the cost of extra GEMM launches.",
            args.feed_forward_chunk_size_tokens,
        )
    if args.feed_forward_memory_profile_max_calls < 0:
        raise ValueError("--feed_forward_memory_profile_max_calls must be >= 0.")
    if args.feed_forward_memory_profile == "true" and args.compile_model == "true":
        if args.feed_forward_memory_profile_allow_compile != "true":
            logging.warning(
                "Feed-forward memory profiling is enabled, but torch.compile is also enabled. "
                "OLMo-core will skip profiler calls inside compiled FFN graphs by default to avoid "
                "debug instrumentation changing memory lifetime. Use --compile_model false for the "
                "most accurate FFN point-by-point profile, or set "
                "--feed_forward_memory_profile_allow_compile true only for an explicit risky probe."
            )
    if args.cuda_memory_history != "false":
        if args.cuda_memory_history_max_entries <= 0:
            raise ValueError("--cuda_memory_history_max_entries must be > 0 when CUDA memory history is enabled.")
        logging.warning(
            "Using compile-safe CUDA allocator memory history mode=%s max_entries=%d. "
            "This records allocator events outside the model graph and is preferred over inline FFN "
            "memory logging when --compile_model true is required.",
            args.cuda_memory_history,
            args.cuda_memory_history_max_entries,
        )
    if normalized_optimizer_name(args) == "adamw_8bit":
        logging.warning(
            "Using experimental torchao AdamW8bit optimizer. Test checkpoint resume and "
            "loss behavior before using it for a long production run."
        )
    if normalized_optimizer_name(args) == "te_fused_adamw":
        logging.warning(
            "Using experimental Transformer Engine FusedAdamW optimizer. "
            "This is GPU-only; validate checkpoint save/resume before a long production run."
        )


def normalize_attn_implementation(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if not normalized or normalized == "auto":
        return None
    aliases = {
        "flash": "flash_2",
        "flash2": "flash_2",
        "flash_attn": "flash_2",
        "flash_attention": "flash_2",
        "flash_attention_2": "flash_2",
        "fa2": "flash_2",
        "flash3": "flash_3",
        "flash_attention_3": "flash_3",
        "fa3": "flash_3",
        "flash4": "flash_4",
        "flash_attention_4": "flash_4",
        "fa4": "flash_4",
        "sdpa": "torch",
        "eager": "torch",
        "pytorch": "torch",
        "transformer_engine": "te",
        "transformer_engine_attention": "te",
        "te_attn": "te",
        "te_attention": "te",
    }
    return aliases.get(normalized, normalized)


def read_hf_rope_scaling(model_path: Path) -> dict[str, object]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open() as file_obj:
            config = json.load(file_obj)
    except Exception:
        return {}
    rope_scaling = config.get("rope_scaling")
    return rope_scaling if isinstance(rope_scaling, dict) else {}


def disabled_rope_scaling_factor(value: str | float | int | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"none", "false", "off", "disable", "disabled", "0", "0.0"}


def effective_rope_scaling_factor(
    args: argparse.Namespace,
    model_path: Path | None = None,
    model_arch: str | None = None,
) -> float | None:
    raw_value = str(args.rope_scaling_factor).strip().lower()
    if disabled_rope_scaling_factor(raw_value):
        return None
    if args.rope_scaling_old_context_len <= 0:
        raise ValueError("--rope_scaling_old_context_len must be positive.")

    if raw_value not in {"", "auto"}:
        try:
            factor = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --rope_scaling_factor={args.rope_scaling_factor!r}; use auto, none, or a positive float."
            ) from exc
        if factor <= 0:
            raise ValueError("--rope_scaling_factor must be positive, or use none to disable the override.")
        return factor

    rope_scaling = read_hf_rope_scaling(model_path) if model_path is not None else {}
    pretrained_factor = 1.0
    if rope_scaling.get("factor") is not None:
        pretrained_factor = max(pretrained_factor, float(rope_scaling["factor"]))

    old_context_len = args.rope_scaling_old_context_len
    if rope_scaling.get("original_max_position_embeddings") is not None:
        old_context_len = int(rope_scaling["original_max_position_embeddings"])
    if old_context_len <= 0:
        raise ValueError("Resolved RoPE old context length must be positive.")

    resolved_arch = model_arch or getattr(args, "model_arch", "auto")
    is_olmo3 = resolved_arch in {"olmo3_7b", "olmo3_32b", "olmo3_7B", "olmo3_32B"}
    if not is_olmo3 and not rope_scaling:
        return None

    requested_factor = olmo_core_sequence_length(args.max_seq_length) / old_context_len
    factor = max(pretrained_factor, requested_factor)
    return factor if factor > 1.0 else None


def log_launch_summary(args: argparse.Namespace, model_arch: str | None = None) -> None:
    resolved_arch = model_arch if model_arch is not None and model_arch != "auto" else None
    try:
        if resolved_arch is None:
            resolved_arch = normalize_model_arch(args.model_arch, Path(args.model_path).expanduser().resolve())
    except Exception:
        resolved_arch = model_arch or args.model_arch

    num_nodes = effective_num_nodes(args)
    num_gpus = effective_num_gpus(args)
    node_rank = effective_node_rank(args)
    world_size = num_nodes * num_gpus
    tp_degree = effective_tensor_parallel_degree(args, resolved_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)
    dp_degree = world_size // (tp_degree * cp_degree * pp_degree) if world_size % (tp_degree * cp_degree * pp_degree) == 0 else -1
    logging.info(
        "Launch summary: backend=%s arch=%s nodes=%d gpus_per_node=%d world_size=%d node_rank=%s "
        "TP=%d CP=%d CP_style=%s PP=%d DP=%s master=%s:%s world_size_mode=%s",
        args.backend,
        resolved_arch,
        num_nodes,
        num_gpus,
        world_size,
        node_rank if node_rank is not None else "unset",
        tp_degree,
        cp_degree,
        effective_context_parallel_style(args) if cp_degree > 1 else "none",
        pp_degree,
        dp_degree if dp_degree >= 0 else "invalid",
        args.master_addr or os.environ.get("MASTER_ADDR", "unset"),
        args.master_port or os.environ.get("MASTER_PORT", "unset"),
        args.world_size_mode,
    )
    attn_implementation = normalize_attn_implementation(args.attn_implementation)
    if attn_implementation is not None:
        logging.info("Attention backend override requested: %s.", attn_implementation)
    if pp_degree > 1:
        split_points = effective_pipeline_split_points(args, resolved_arch)
        split_source = "explicit" if parse_pipeline_split_points(args.pipeline_split_points) else "auto"
        logging.info(
            "Pipeline split points resolved: %s source=%s schedule=%s.",
            split_points if split_points else "framework-default",
            split_source if split_points else "none",
            args.pipeline_schedule,
        )
    logging.info(
        "Dataset transform: profile=%s chat_template_name=%s chat_template_model=%s transform_names=%s",
        getattr(args, "dataset_transform_profile", "qwen"),
        args.chat_template_name or "tokenizer-default",
        effective_chat_template_model(args) or "none",
        dataset_transform_names(args),
    )
    rope_factor = effective_rope_scaling_factor(args, Path(args.model_path).expanduser().resolve(), resolved_arch)
    if rope_factor is not None:
        logging.info(
            "RoPE scaling factor resolved: %.6g (old_context_len=%d, requested_seq_len=%d).",
            rope_factor,
            args.rope_scaling_old_context_len,
            olmo_core_sequence_length(args.max_seq_length),
        )


def build_env(open_instruct_dir: Path, cache_dir: Path, offline: bool = False, args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    strip_inherited_torchrun_env(env)
    olmo_core_dir = Path(env.get("OLMO_CORE_DIR", "/opt/OLMo-core"))
    app_dir = Path(__file__).resolve().parent
    pythonpath_parts = [str(open_instruct_dir), str(olmo_core_dir / "src"), str(app_dir), str(Path("/app"))]
    pythonpath_parts.extend(str(path / "src") for path in candidate_repo_dirs("OLMo-core") if path.is_dir())
    if env.get("PYTHONPATH"):
        pythonpath_parts.extend(env["PYTHONPATH"].split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in pythonpath_parts if part))
    configure_runtime_cache_environment(env, cache_dir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args is not None and getattr(args, "dataset_num_proc", 0) > 0:
        env["BEAKER_ASSIGNED_CPU_COUNT"] = str(args.dataset_num_proc)
    if args is not None and getattr(args, "dataset_map_batch_size", 0) > 0:
        env["OPEN_INSTRUCT_DATASET_MAP_BATCH_SIZE"] = str(args.dataset_map_batch_size)
    if args is not None and args.wandb_mode != "auto":
        env["WANDB_MODE"] = args.wandb_mode
    elif args is not None and args.with_tracking:
        env["WANDB_MODE"] = "online"
    else:
        env.setdefault("WANDB_MODE", "offline")
    if args is not None:
        node_rank = effective_node_rank(args)
        if node_rank is not None:
            env["OLMO_LOG_NODE_RANK"] = str(node_rank)
    apply_wandb_api_key_from_netrc(env, args)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    return env


def apply_wandb_api_key_from_netrc(env: dict[str, str], args: argparse.Namespace | None = None) -> None:
    if args is None or not args.with_tracking:
        return
    if env.get("WANDB_API_KEY"):
        return
    if env.get("WANDB_MODE") in {"offline", "disabled"}:
        return
    try:
        authenticators = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return
    if authenticators is None or not authenticators[2]:
        return
    env["WANDB_API_KEY"] = authenticators[2]
    logging.info("Loaded W&B API key from netrc for online tracking.")


def build_training_env(
    open_instruct_dir: Path,
    olmo_core_dir: Path,
    cache_dir: Path,
    offline: bool = False,
    args: argparse.Namespace | None = None,
) -> dict[str, str]:
    env = build_env(open_instruct_dir, cache_dir, offline=offline, args=args)
    env["OPEN_INSTRUCT_DIR"] = str(open_instruct_dir)
    env["OLMO_CORE_DIR"] = str(olmo_core_dir)
    parts = [str(open_instruct_dir), str(olmo_core_dir / "src"), str(Path("/app"))]
    if env.get("PYTHONPATH"):
        parts.extend(env["PYTHONPATH"].split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))
    return env


def validate_open_instruct_optimizer(args: argparse.Namespace) -> None:
    optimizer = normalized_optimizer_name(args)
    if optimizer == "adamw_8bit":
        logging.warning(
            "Using --optimizer adamw_8bit with the Open-Instruct wrapper. This requires "
            "torchao and olmo_torchao_optim.py to be importable in PYTHONPATH."
        )


def run_command(command: list[str], env: dict[str, str]) -> None:
    logging.info("Running command: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        relay_child_log_line(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def run_verl_command_with_checkpoint_watcher(
    command: list[str],
    env: dict[str, str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    if not hf_checkpoint_upload_enabled(args):
        run_command(command, env)
        return

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=watch_verl_checkpoints_for_hf_upload,
        kwargs={"output_path": output_path, "args": args, "stop_event": stop_event},
        name="verl-hf-checkpoint-watcher",
        daemon=True,
    )
    watcher.start()
    try:
        run_command(command, env)
    finally:
        stop_event.set()
        watcher.join()


def should_suppress_child_log_line(line: str) -> bool:
    if os.environ.get("OLMO_SUPPRESS_RAY_FILE_SYSTEM_MONITOR_LOGS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        if "file_system_monitor.cc" in line and "Object creation will fail if spilling is required" in line:
            return True

    extra_patterns = os.environ.get("OLMO_CHILD_LOG_SUPPRESS_PATTERNS", "")
    for pattern in (part.strip() for part in extra_patterns.split("||")):
        if pattern and pattern in line:
            return True
    return False


def command_to_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def cli_has_flag(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def cli_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def cli_values(argv: list[str], flag: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                values.append(argv[index])
                index += 1
            continue
        if token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def parse_cli_bool(argv: list[str], flag: str, default: bool = False) -> bool:
    value = cli_value(argv, flag)
    if value is None:
        return default if not cli_has_flag(argv, flag) else True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_cli_int(argv: list[str], flag: str, default: int) -> int:
    value = cli_value(argv, flag)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{flag} must be an integer, got {value!r}") from exc


def parse_cli_int_list(argv: list[str], flag: str, default: list[int]) -> list[int]:
    values = cli_values(argv, flag)
    if not values:
        return list(default)
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{flag} must contain only integers, got {values!r}") from exc


def append_grpo_default(grpo_args: list[str], flag: str, *values: object) -> None:
    if cli_has_flag(grpo_args, flag):
        return
    grpo_args.append(flag)
    grpo_args.extend(str(value) for value in values)


def append_grpo_default_bool(grpo_args: list[str], flag: str) -> None:
    if not cli_has_flag(grpo_args, flag):
        grpo_args.append(flag)


def grpo_model_default(args: argparse.Namespace, model_arch: str) -> str:
    if args.model_path:
        return args.model_path
    return DEFAULT_MODEL_HF_REPOS[model_arch]


def apply_grpo_preset_defaults(args: argparse.Namespace, grpo_args: list[str]) -> None:
    if args.rl_preset is None:
        return

    if args.rl_preset == "olmo3_7b_4xh200_smoke":
        append_grpo_default(grpo_args, "--exp_name", "olmo3_7b_rl_4xh200_smoke")
        append_grpo_default(grpo_args, "--model_name_or_path", grpo_model_default(args, "olmo3_7b"))
        append_grpo_default(grpo_args, "--dataset_mixer_list", args.dataset_path or "allenai/Dolci-Think-RL-7B", "1.0")
        append_grpo_default(grpo_args, "--dataset_mixer_list_splits", args.dataset_split)
        append_grpo_default(grpo_args, "--num_learners_per_node", 2)
        append_grpo_default(grpo_args, "--vllm_num_engines", 2)
        append_grpo_default(grpo_args, "--vllm_tensor_parallel_size", 1)
        append_grpo_default(grpo_args, "--response_length", 4096)
        append_grpo_default(grpo_args, "--pack_length", 6144)
        append_grpo_default(grpo_args, "--num_unique_prompts_rollout", 16)
        append_grpo_default(grpo_args, "--num_samples_per_prompt_rollout", 4)
    elif args.rl_preset in {"olmo3_32b_3node_smoke", "olmo3_32b_2gen_1train_smoke", "olmo3_32b_2gen_1train_long"}:
        exp_name = "olmo3_32b_rl_2gen_1train_smoke"
        if args.rl_preset == "olmo3_32b_3node_smoke":
            exp_name = "olmo3_32b_rl_3node_smoke"
        elif args.rl_preset == "olmo3_32b_2gen_1train_long":
            exp_name = "olmo3_32b_rl_2gen_1train_long"
        append_grpo_default(grpo_args, "--exp_name", exp_name)
        append_grpo_default(grpo_args, "--model_name_or_path", grpo_model_default(args, "olmo3_32b"))
        append_grpo_default(grpo_args, "--dataset_mixer_list", args.dataset_path or "allenai/Dolci-Think-RL-7B", "1.0")
        append_grpo_default(grpo_args, "--dataset_mixer_list_splits", args.dataset_split)
        # 24-GPU target: one 8xH200 learner node and two 8xH200 rollout/vLLM nodes.
        append_grpo_default(grpo_args, "--num_learners_per_node", 8)
        append_grpo_default(grpo_args, "--vllm_num_engines", 2)
        append_grpo_default(grpo_args, "--vllm_tensor_parallel_size", 8)
        append_grpo_default(grpo_args, "--vllm_disable_custom_all_reduce", "true")
        append_grpo_default(grpo_args, "--gather_whole_model", "False")
        if args.rl_preset == "olmo3_32b_2gen_1train_long":
            append_grpo_default(grpo_args, "--max_prompt_token_length", 5536)
            append_grpo_default(grpo_args, "--response_length", 60000)
            append_grpo_default(grpo_args, "--pack_length", 65536)
            append_grpo_default(grpo_args, "--num_unique_prompts_rollout", 4)
            append_grpo_default(grpo_args, "--num_samples_per_prompt_rollout", 2)
            append_grpo_default(grpo_args, "--total_episodes", 8)
            append_grpo_default(grpo_args, "--llm_judge_model", "local_vllm")
            append_grpo_default(grpo_args, "--deepseekmath_v2_judge_backend", "local_vllm")
            append_grpo_default(grpo_args, "--llm_judge_max_tokens", 40000)
            append_grpo_default(grpo_args, "--deepseekmath_v2_max_tokens", 40000)
            append_grpo_default(grpo_args, "--deepseekmath_v2_max_context_length", 40000)
            append_grpo_default(grpo_args, "--deepseekmath_v2_context_margin_tokens", 256)
            append_grpo_default(grpo_args, "--deepseekmath_v2_min_completion_tokens", 2048)
            append_grpo_default(grpo_args, "--deepseekmath_v2_temperature", 0.7)
            append_grpo_default(grpo_args, "--deepseekmath_v2_top_p", 0.95)
            append_grpo_default(grpo_args, "--deepseekmath_v2_timeout", 1800)
            append_grpo_default(grpo_args, "--deepseekmath_v2_proof_weight", 0.76)
            append_grpo_default(grpo_args, "--deepseekmath_v2_self_eval_weight", 0.24)
            append_grpo_default(grpo_args, "--deepseekmath_v2_partial_format_score", 0.7)
            append_grpo_default(grpo_args, "--deepseekmath_v2_enable_meta_verification", "true")
            append_grpo_default(grpo_args, "--deepseekmath_v2_require_format", "true")
            append_grpo_default(grpo_args, "--remap_verifier", "proof_math=deepseekmath_v2")
        else:
            append_grpo_default(grpo_args, "--response_length", 4096)
            append_grpo_default(grpo_args, "--pack_length", 6144)
            append_grpo_default(grpo_args, "--num_unique_prompts_rollout", 16)
            append_grpo_default(grpo_args, "--num_samples_per_prompt_rollout", 4)

    append_grpo_default(grpo_args, "--beta", 0.0)
    append_grpo_default(grpo_args, "--num_mini_batches", 1)
    append_grpo_default(grpo_args, "--num_epochs", 1)
    append_grpo_default(grpo_args, "--learning_rate", args.learning_rate)
    append_grpo_default(grpo_args, "--per_device_train_batch_size", 1)
    append_grpo_default(grpo_args, "--kl_estimator", 2)
    append_grpo_default(grpo_args, "--max_prompt_token_length", 2048)
    append_grpo_default(grpo_args, "--chat_template_name", "olmo_thinker")
    append_grpo_default(grpo_args, "--non_stop_penalty", "False")
    append_grpo_default(grpo_args, "--mask_truncated_completions", "False")
    append_grpo_default(grpo_args, "--temperature", 0.7)
    append_grpo_default(grpo_args, "--vllm_top_p", 0.95)
    append_grpo_default(grpo_args, "--ground_truths_key", "ground_truth")
    append_grpo_default(grpo_args, "--sft_messages_key", "messages")
    append_grpo_default(grpo_args, "--total_episodes", 1024)
    append_grpo_default(grpo_args, "--deepspeed_stage", 3)
    append_grpo_default(grpo_args, "--lr_scheduler_type", "constant")
    append_grpo_default(grpo_args, "--apply_verifiable_reward", "true")
    append_grpo_default(grpo_args, "--verification_reward", 1.0)
    append_grpo_default(grpo_args, "--llm_judge_max_tokens", 2048)
    append_grpo_default(grpo_args, "--deepseekmath_v2_max_tokens", 2048)
    append_grpo_default(grpo_args, "--deepseekmath_v2_max_context_length", 102400)
    append_grpo_default(grpo_args, "--deepseekmath_v2_timeout", 120)
    append_grpo_default(grpo_args, "--checkpoint_state_freq", 20)
    append_grpo_default(grpo_args, "--save_freq", 20)
    append_grpo_default(grpo_args, "--backend_timeout", 1200)
    append_grpo_default(grpo_args, "--vllm_enforce_eager", "false")
    append_grpo_default_bool(grpo_args, "--gradient_checkpointing")


def normalize_grpo_sampling_args(grpo_args: list[str]) -> None:
    num_samples = parse_cli_int(grpo_args, "--num_samples_per_prompt_rollout", 1)
    if num_samples == 1 and not cli_has_flag(grpo_args, "--filter_zero_std_samples"):
        grpo_args.extend(["--filter_zero_std_samples", "False"])


def maybe_prepare_grpo_dataset(args: argparse.Namespace, cache_dir: Path) -> str | None:
    if not args.dataset_path:
        return None
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if dataset_path.suffix.lower() != ".csv":
        return str(dataset_path)

    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
    except StopIteration as exc:
        raise ValueError(f"CSV dataset is empty: {dataset_path}") from exc
    normalized_columns = {column.strip().lower(): column for column in columns}
    if "messages" in normalized_columns:
        logging.info("Using local GRPO CSV dataset with existing messages column: %s", dataset_path)
        return str(dataset_path)

    problem_column = normalized_columns.get("problem") or normalized_columns.get("question")
    solution_column = normalized_columns.get("solution")
    if not problem_column or not solution_column:
        raise ValueError(
            "Local GRPO CSV dataset must contain either messages, or problem/question plus solution columns: "
            f"{dataset_path} columns={columns}"
        )

    prepared_path = cache_dir / "prepared_grpo_datasets" / f"{dataset_path.stem}_grpo.parquet"
    node_rank = effective_node_rank(args)
    if effective_num_nodes(args) <= 1 and node_rank is None:
        node_rank = 0
    if node_rank in (None, 0):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_proofbench_rl_data.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"Could not find ProofBench RL prep script: {script_path}")
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Preparing local CSV GRPO dataset from %s to %s using DeepSeekMath-V2 generation prompt "
            "(problem_column=%s solution_column=%s).",
            dataset_path,
            prepared_path,
            problem_column,
            solution_column,
        )
        run_command(
            [
                sys.executable,
                str(script_path),
                "--input",
                str(dataset_path),
                "--output",
                str(prepared_path),
                "--problem_column",
                problem_column,
                "--solution_column",
                solution_column,
            ],
            os.environ.copy(),
        )
    else:
        logging.info(
            "Using expected ProofBench V2 GRPO dataset path on non-driver node_rank=%s: %s",
            node_rank,
            prepared_path,
        )
    return str(prepared_path)


def build_grpo_forward_args(args: argparse.Namespace, output_path: Path, cache_dir: Path) -> list[str]:
    raw_argv = list(getattr(args, "_raw_argv", sys.argv[1:]))
    grpo_args = strip_cli_args(raw_argv, GRPO_WRAPPER_VALUE_FLAGS, GRPO_WRAPPER_BOOL_FLAGS)

    if args.model_path:
        append_grpo_default(grpo_args, "--model_name_or_path", args.model_path)
    if args.tokenizer_path:
        append_grpo_default(grpo_args, "--tokenizer_name_or_path", args.tokenizer_path)
    grpo_dataset_path = maybe_prepare_grpo_dataset(args, cache_dir)
    if grpo_dataset_path:
        append_grpo_default(grpo_args, "--dataset_mixer_list", grpo_dataset_path, args.dataset_weight)
        append_grpo_default(grpo_args, "--dataset_mixer_list_splits", args.dataset_split)

    if args.grpo_deepspeed_stage is not None:
        append_grpo_default(grpo_args, "--deepspeed_stage", args.grpo_deepspeed_stage)
    if args.grpo_deepspeed_zpg is not None:
        append_grpo_default(grpo_args, "--deepspeed_zpg", args.grpo_deepspeed_zpg)
    if args.grpo_deepspeed_offload_param is not None:
        append_grpo_default(grpo_args, "--deepspeed_offload_param", args.grpo_deepspeed_offload_param)
    if args.grpo_deepspeed_offload_optimizer is not None:
        append_grpo_default(grpo_args, "--deepspeed_offload_optimizer", args.grpo_deepspeed_offload_optimizer)
    if args.grpo_sequence_parallel_size is not None:
        append_grpo_default(grpo_args, "--sequence_parallel_size", args.grpo_sequence_parallel_size)

    apply_grpo_preset_defaults(args, grpo_args)
    normalize_grpo_sampling_args(grpo_args)

    append_grpo_default(grpo_args, "--output_dir", output_path)
    append_grpo_default(grpo_args, "--dataset_local_cache_dir", cache_dir / "grpo_dataset_cache")
    append_grpo_default(grpo_args, "--num_nodes", effective_num_nodes(args))
    append_grpo_default(grpo_args, "--push_to_hub", "False")
    append_grpo_default(grpo_args, "--try_auto_save_to_beaker", "False")
    append_grpo_default(grpo_args, "--try_launch_beaker_eval_jobs_on_weka", "False")
    if not cli_has_flag(grpo_args, "--wandb_project"):
        grpo_args.extend(["--wandb_project", args.wandb_project if args.wandb_project != "olmo-sft" else "olmo-rl"])
    if args.wandb_entity and not cli_has_flag(grpo_args, "--wandb_entity"):
        grpo_args.extend(["--wandb_entity", args.wandb_entity])
    if not cli_has_flag(grpo_args, "--per_device_train_batch_size") and args.per_device_batch_size:
        grpo_args.extend(["--per_device_train_batch_size", str(args.per_device_batch_size)])

    missing = []
    if not cli_has_flag(grpo_args, "--model_name_or_path"):
        missing.append("--model_name_or_path or --model_path")
    if not cli_has_flag(grpo_args, "--dataset_mixer_list"):
        missing.append("--dataset_mixer_list or --dataset_path")
    if missing:
        raise ValueError(
            "GRPO backend is missing required Open-Instruct input(s): "
            + ", ".join(missing)
            + ". Use --rl_preset for a smoke-test template."
        )
    return grpo_args


def grpo_resource_summary(args: argparse.Namespace, grpo_args: list[str]) -> dict[str, object]:
    learners = parse_cli_int_list(grpo_args, "--num_learners_per_node", [1])
    vllm_num_engines = parse_cli_int(grpo_args, "--vllm_num_engines", 1)
    vllm_tensor_parallel_size = parse_cli_int(grpo_args, "--vllm_tensor_parallel_size", 1)
    single_gpu_mode = parse_cli_bool(grpo_args, "--single_gpu_mode", False)
    learner_gpus = sum(learners)
    separate_vllm_gpus = 0 if single_gpu_mode else vllm_num_engines * vllm_tensor_parallel_size
    required_gpus = learner_gpus + separate_vllm_gpus
    available_gpus = effective_num_nodes(args) * effective_num_gpus(args)
    return {
        "num_learners_per_node": learners,
        "learner_gpus": learner_gpus,
        "vllm_num_engines": vllm_num_engines,
        "vllm_tensor_parallel_size": vllm_tensor_parallel_size,
        "single_gpu_mode": single_gpu_mode,
        "separate_vllm_gpus": separate_vllm_gpus,
        "required_gpus": required_gpus,
        "available_gpus": available_gpus,
    }


def validate_grpo_resources(args: argparse.Namespace, grpo_args: list[str]) -> None:
    summary = grpo_resource_summary(args, grpo_args)
    logging.info("GRPO resource summary: %s", json.dumps(summary, sort_keys=True))
    if summary["single_gpu_mode"] and int(summary["vllm_tensor_parallel_size"]) > 1:
        raise ValueError("--single_gpu_mode cannot be used with --vllm_tensor_parallel_size > 1.")
    if int(summary["required_gpus"]) > int(summary["available_gpus"]):
        raise ValueError(
            "GRPO topology requests more GPUs than the wrapper can see: "
            f"required={summary['required_gpus']} available={summary['available_gpus']}. "
            "Required GPUs are sum(num_learners_per_node) plus "
            "vllm_num_engines*vllm_tensor_parallel_size unless --single_gpu_mode is used."
        )


def grpo_ray_port(args: argparse.Namespace) -> int:
    return args.grpo_ray_port or effective_master_port(args)


def grpo_head_addr(args: argparse.Namespace) -> str:
    if args.master_addr:
        return args.master_addr
    if os.environ.get("MASTER_ADDR"):
        return os.environ["MASTER_ADDR"]
    if effective_num_nodes(args) <= 1:
        return "127.0.0.1"
    raise ValueError("GRPO multinode launch requires --master_addr or MASTER_ADDR.")


def grpo_ray_address(args: argparse.Namespace) -> str:
    return f"{grpo_head_addr(args)}:{grpo_ray_port(args)}"


def grpo_ray_temp_dir(args: argparse.Namespace) -> Path:
    host = socket.gethostname()
    node_rank = effective_node_rank(args)
    node_token = "none" if node_rank is None else str(node_rank)
    port = str(grpo_ray_port(args))
    template = (
        args.grpo_ray_temp_dir
        or os.environ.get("GRPO_RAY_TEMP_DIR")
        or "/tmp/ray-grpo/{host}-node{node_rank}-port{port}"
    )
    if "{" in template:
        value = template.format(host=host, node_rank=node_token, port=port)
    else:
        value = template
        if effective_num_nodes(args) > 1:
            value = str(Path(value) / f"{host}-node{node_token}-port{port}")
    return Path(value).expanduser().resolve()


def run_logged_subprocess(command: list[str], env: dict[str, str], check: bool = True) -> int:
    logging.info("Running command: %s", command_to_text(command))
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    suppressed_ray_idle_lines = 0
    for line in output.splitlines():
        if len(line) > 1000 and "ray::IDLE" in line:
            suppressed_ray_idle_lines += 1
            continue
        relay_child_log_line(line)
    if suppressed_ray_idle_lines:
        logging.info("Suppressed %d long Ray idle-worker shutdown log lines.", suppressed_ray_idle_lines)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command_to_text(command)}")
    return completed.returncode


def ray_status_ok(address: str, env: dict[str, str]) -> bool:
    completed = subprocess.run(
        ["ray", "status", "--address", address],
        env=env,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def wait_for_ray_status(address: str, env: dict[str, str], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ray_status_ok(address, env):
            logging.info("Ray cluster is reachable at %s.", address)
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for Ray cluster at {address}")


def stop_ray(env: dict[str, str]) -> None:
    if shutil.which("ray") is None:
        logging.warning("ray executable not found; cannot stop Ray.")
        return
    run_logged_subprocess(["ray", "stop", "--force"], env, check=False)


def start_ray_head(args: argparse.Namespace, env: dict[str, str]) -> None:
    if shutil.which("ray") is None:
        raise FileNotFoundError("ray executable not found. Install ray in the Singularity image.")
    stop_ray(env)
    temp_dir = grpo_ray_temp_dir(args)
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ray",
        "start",
        "--head",
        f"--port={grpo_ray_port(args)}",
        "--include-dashboard=false",
        f"--temp-dir={temp_dir}",
    ]
    if args.grpo_ray_num_cpus and args.grpo_ray_num_cpus > 0:
        command.append(f"--num-cpus={args.grpo_ray_num_cpus}")
    run_logged_subprocess(command, env)
    wait_for_ray_status(grpo_ray_address(args), env, args.grpo_ray_start_timeout)


def start_ray_worker(args: argparse.Namespace, env: dict[str, str]) -> None:
    if shutil.which("ray") is None:
        raise FileNotFoundError("ray executable not found. Install ray in the Singularity image.")
    stop_ray(env)
    address = grpo_ray_address(args)
    wait_for_ray_status(address, env, args.grpo_ray_start_timeout)
    temp_dir = grpo_ray_temp_dir(args)
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ray",
        "start",
        f"--address={address}",
        f"--temp-dir={temp_dir}",
    ]
    if args.grpo_ray_num_cpus and args.grpo_ray_num_cpus > 0:
        command.append(f"--num-cpus={args.grpo_ray_num_cpus}")
    retries = max(1, int(args.grpo_ray_worker_start_retries))
    for attempt in range(1, retries + 1):
        try:
            logging.info("Starting Ray worker attempt %s/%s at %s.", attempt, retries, address)
            run_logged_subprocess(command, env)
            break
        except Exception:
            if attempt >= retries:
                raise
            logging.exception("Ray worker start attempt %s/%s failed; retrying after cleanup.", attempt, retries)
            stop_ray(env)
            time.sleep(10 * attempt)
            wait_for_ray_status(address, env, args.grpo_ray_start_timeout)
    wait_for_ray_status(address, env, args.grpo_ray_start_timeout)


def monitor_grpo_worker(args: argparse.Namespace, env: dict[str, str]) -> None:
    address = grpo_ray_address(args)
    logging.info("GRPO worker node is monitoring Ray head at %s.", address)
    while True:
        if not ray_status_ok(address, env):
            logging.info("Ray head is no longer reachable; worker monitor exiting.")
            return
        time.sleep(max(1, args.grpo_worker_poll_interval))


def grpo_judge_model(grpo_args: list[str]) -> str | None:
    return cli_value(grpo_args, "--llm_judge_model")


def grpo_judge_base_url(grpo_args: list[str]) -> str | None:
    return (
        cli_value(grpo_args, "--llm_judge_base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )


def grpo_judge_api_key_env(grpo_args: list[str]) -> str:
    return cli_value(grpo_args, "--llm_judge_api_key_env") or "OPENAI_API_KEY"


def run_grpo_judge_preflight(args: argparse.Namespace, grpo_args: list[str], env: dict[str, str]) -> None:
    if args.grpo_judge_preflight != "true":
        return
    model = grpo_judge_model(grpo_args)
    if not model:
        logging.info("GRPO judge preflight skipped because --llm_judge_model was not provided.")
        return
    base_url = grpo_judge_base_url(grpo_args)
    api_key = cli_value(grpo_args, "--llm_judge_api_key") or env.get(grpo_judge_api_key_env(grpo_args), "")
    if not api_key:
        raise ValueError(
            "GRPO judge preflight requires an API key. Set OPENAI_API_KEY, pass "
            "--llm_judge_api_key_env, or pass --llm_judge_api_key."
        )
    code = """
import asyncio
import os
import sys

from openai import AsyncOpenAI

async def main() -> None:
    model = sys.argv[1]
    base_url = sys.argv[2] or None
    prompt = sys.argv[3]
    api_key = os.environ["GRPO_JUDGE_PREFLIGHT_API_KEY"]
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60)
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 8,
        "timeout": 60,
    }
    if base_url and "openrouter.ai" in base_url and model.startswith("deepseek/"):
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
            "provider": {"only": ["deepseek"], "allow_fallbacks": False},
        }
    response = await client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    print((content or "")[:200])
    await client.close()

asyncio.run(main())
""".strip()
    logging.info("Running GRPO OpenAI-compatible judge preflight for model=%s base_url=%s.", model, base_url)
    preflight_env = dict(env)
    preflight_env["GRPO_JUDGE_PREFLIGHT_API_KEY"] = api_key
    run_logged_subprocess([sys.executable, "-c", code, model, base_url or "", args.grpo_judge_preflight_prompt], preflight_env)


def build_grpo_env(open_instruct_dir: Path, olmo_core_dir: Path, cache_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    env = build_training_env(open_instruct_dir, olmo_core_dir, cache_dir, offline=False, args=args)
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    # vLLM V1 can emit torch.dtype values through its Ray IPC payloads for
    # OLMo3 configs. Allow the documented pickle fallback for those objects.
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("OPEN_INSTRUCT_ALLOW_ASYNCIO_RAY_ACTOR", "1")
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env["RAY_ADDRESS"] = grpo_ray_address(args)
    env["RAY_TMPDIR"] = str(grpo_ray_temp_dir(args))
    return env


def run_grpo_fast(args: argparse.Namespace) -> None:
    if sweep_requested(args):
        raise ValueError(f"Learning-rate/optimizer sweeps are not implemented for --backend {args.backend}.")

    output_path = Path(args.output_path).expanduser().resolve()
    logdir = Path(args.logdir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir(args, "grpo_cache")
    configure_logging(logdir, args)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache_environment(os.environ, cache_dir)
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, logdir, "grpo_parent", launcher_environment_summary(args))

    open_instruct_dir = find_open_instruct_dir(args.open_instruct_dir or os.environ.get("OPEN_INSTRUCT_DIR"))
    olmo_core_dir = find_olmo_core_dir(os.environ.get("OLMO_CORE_DIR"))
    env = build_grpo_env(open_instruct_dir, olmo_core_dir, cache_dir, args)
    env.setdefault("OPEN_INSTRUCT_PROMPT_LOG_DIR", str(logdir / "prompt_previews"))
    env.setdefault("OPEN_INSTRUCT_PROMPT_LOG_MAX_CHARS", "0")
    grpo_args = build_grpo_forward_args(args, output_path, cache_dir)
    validate_grpo_resources(args, grpo_args)

    entry_module = "open_instruct.grpo_fast" if args.backend == "grpo_fast" else "open_instruct.grpo"
    entry_file = "grpo_fast.py" if args.backend == "grpo_fast" else "grpo.py"
    grpo_script = open_instruct_dir / "open_instruct" / entry_file
    if not grpo_script.is_file():
        raise FileNotFoundError(f"Could not find Open-Instruct GRPO entrypoint: {grpo_script}")
    command = [
        sys.executable,
        "-c",
        f"from {entry_module} import cli_main; cli_main()",
        *grpo_args,
    ]
    (logdir / "GRPO_COMMAND.txt").write_text(command_to_text(command) + "\n", encoding="utf-8")

    node_rank = effective_node_rank(args)
    if effective_num_nodes(args) <= 1 and node_rank is None:
        node_rank = 0
    if node_rank is None:
        raise ValueError("GRPO launch requires --node_rank or PBS GLOBAL_RANK.")
    if node_rank < 0 or node_rank >= effective_num_nodes(args):
        raise ValueError(f"Resolved GRPO node_rank={node_rank} outside num_nodes={effective_num_nodes(args)}.")

    logging.info(
        "GRPO launch summary: node_rank=%s num_nodes=%s gpus_per_node=%s ray=%s command=%s",
        node_rank,
        effective_num_nodes(args),
        effective_num_gpus(args),
        grpo_ray_address(args),
        command_to_text(command),
    )
    logging.info("GRPO Ray temp dir: %s", grpo_ray_temp_dir(args))

    if args.dry_run_launch:
        logging.info("GRPO dry-run requested; not starting Ray or launching Open-Instruct.")
        return

    try:
        if node_rank == 0:
            run_grpo_judge_preflight(args, grpo_args, env)
            start_ray_head(args, env)
            run_command(command, env)
        else:
            start_ray_worker(args, env)
            monitor_grpo_worker(args, env)
    finally:
        if args.grpo_stop_ray_on_exit == "true":
            stop_ray(env)


def find_rlcsd_dir(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env_name in ("RLCSD_DIR",):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path("/tmp/RLCSD-runtime"),
            Path("/opt/RLCSD"),
            Path.cwd() / "RLCSD",
            Path(__file__).resolve().parents[2] / "RLCSD",
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "self_distill_main.py").is_file() and (candidate / "third_party" / "verl").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find RLCSD checkout. Set --rlcsd_dir or RLCSD_DIR; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def build_verl_rlcsd_env(rlcsd_dir: Path, cache_dir: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    configure_runtime_cache_environment(env, cache_dir)
    env["RLCSD_DIR"] = str(rlcsd_dir)
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "20000")
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("RAY_ADDRESS", grpo_ray_address(args))
    env.setdefault("RAY_TMPDIR", str(grpo_ray_temp_dir(args)))
    env.setdefault("VERL_RLCSD_JUDGE_MODEL", "deepseek/deepseek-v4-pro")
    env.setdefault("VERL_RLCSD_JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    env.setdefault("VERL_RLCSD_JUDGE_TEMPERATURE", "1.0")
    env.setdefault("VERL_RLCSD_JUDGE_TOP_P", "0.95")
    env["VERL_RLCSD_PROOF_REWARD_WEIGHT"] = str(args.rlcsd_proof_reward_weight)
    env["VERL_RLCSD_SELF_EVAL_REWARD_WEIGHT"] = str(args.rlcsd_self_eval_reward_weight)
    env["VERL_RLCSD_PARTIAL_FORMAT_SCORE"] = str(args.rlcsd_partial_format_score)
    if olmo3_sink_enabled(args):
        env["VERL_RLCSD_OLMO3_SINK"] = "1"
        env["OLMO3_SINK_ATTN_IMPLEMENTATION"] = args.olmo3_sink_attn_implementation
    openrouter_key = env.get("OPENROUTER_API_KEY")
    if not openrouter_key and env.get("OPENAI_API_KEY"):
        env["OPENROUTER_API_KEY"] = env["OPENAI_API_KEY"]
    parts = [
        str(rlcsd_dir / "third_party" / "verl"),
        str(rlcsd_dir),
        str(Path(__file__).resolve().parent),
        str(Path("/app")),
    ]
    open_instruct_dir = os.environ.get("OPEN_INSTRUCT_DIR")
    if open_instruct_dir:
        parts.append(open_instruct_dir)
    if env.get("PYTHONPATH"):
        parts.extend(env["PYTHONPATH"].split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))
    apply_wandb_api_key_from_netrc(env, args)
    if args.wandb_mode != "auto":
        env["WANDB_MODE"] = args.wandb_mode
    elif args.with_tracking:
        env["WANDB_MODE"] = "online"
    return env


def rlcsd_dataset_marker_name(args: argparse.Namespace, dataset_path: Path) -> str:
    payload = {
        "dataset": str(dataset_path),
        "mtime": dataset_path.stat().st_mtime if dataset_path.exists() else None,
        "size": dataset_path.stat().st_size if dataset_path.exists() else None,
        "max_rows": args.rlcsd_max_rows,
        "prompt_version": RLCSD_PROOF_PROMPT_VERSION,
    }
    return "verl_rlcsd_dataset_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def maybe_prepare_verl_rlcsd_dataset(args: argparse.Namespace, cache_dir: Path) -> tuple[Path, Path]:
    if args.rlcsd_train_file:
        train_file = Path(args.rlcsd_train_file).expanduser().resolve()
        val_file = Path(args.rlcsd_val_file).expanduser().resolve() if args.rlcsd_val_file else train_file
        return train_file, val_file
    if not args.dataset_path:
        raise ValueError("--backend verl_rlcsd requires --dataset_path or --rlcsd_train_file.")

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    data_cache = (
        Path(args.rlcsd_data_cache).expanduser().resolve()
        if args.rlcsd_data_cache
        else cache_dir / "verl_rlcsd_datasets"
    )
    target = data_cache / f"{dataset_path.stem}_proof_verl_{RLCSD_PROOF_PROMPT_VERSION}.parquet"
    marker_dir = cache_dir / "multinode_prepare_markers"

    def prepare() -> str:
        from verl_rlcsd_adapter import prepare_verl_proof_dataset

        return str(
            prepare_verl_proof_dataset(
                dataset_path,
                target,
                max_rows=args.rlcsd_max_rows,
            )
        )

    train_file = Path(run_once_on_node0(args, marker_dir, rlcsd_dataset_marker_name(args, dataset_path), prepare))
    val_file = Path(args.rlcsd_val_file).expanduser().resolve() if args.rlcsd_val_file else train_file
    return train_file, val_file


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def append_hydra_arg(args_list: list[str], key: str, value: Any) -> None:
    args_list.append(f"{key}={_hydra_value(value)}")


def append_hydra_plus_arg(args_list: list[str], key: str, value: Any) -> None:
    args_list.append(f"+{key}={_hydra_value(value)}")


def parse_extra_hydra_overrides(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def rlcsd_async_rollout_enabled(args: argparse.Namespace) -> bool:
    return str(args.rlcsd_async_rollout).lower() == "true"


def resolve_rlcsd_async_topology(args: argparse.Namespace) -> tuple[int, int, int, int]:
    gpus_per_node = effective_num_gpus(args)
    num_nodes = effective_num_nodes(args)
    train_nodes = args.rlcsd_train_nodes if args.rlcsd_train_nodes > 0 else 1
    train_gpus = args.rlcsd_train_gpus_per_node if args.rlcsd_train_gpus_per_node > 0 else gpus_per_node
    default_rollout_nodes = max(1, num_nodes - train_nodes)
    rollout_nodes = args.rlcsd_rollout_nodes if args.rlcsd_rollout_nodes > 0 else default_rollout_nodes
    rollout_gpus = args.rlcsd_rollout_gpus_per_node if args.rlcsd_rollout_gpus_per_node > 0 else gpus_per_node
    return train_nodes, train_gpus, rollout_nodes, rollout_gpus


def build_verl_rlcsd_hydra_args(
    args: argparse.Namespace,
    train_file: Path,
    val_file: Path,
    output_path: Path,
) -> list[str]:
    gpus_per_node = effective_num_gpus(args)
    num_nodes = effective_num_nodes(args)
    async_rollout = rlcsd_async_rollout_enabled(args)
    if async_rollout:
        train_nodes, train_gpus, rollout_nodes, rollout_gpus = resolve_rlcsd_async_topology(args)
        total_gpus = max(1, rollout_nodes * rollout_gpus)
    else:
        train_nodes, train_gpus, rollout_nodes, rollout_gpus = num_nodes, gpus_per_node, 0, 0
        total_gpus = max(1, gpus_per_node * num_nodes)
    rollout_tp = max(1, args.rlcsd_rollout_tensor_parallel_size)
    rollout_dp = max(1, total_gpus // rollout_tp)
    rollout_batch = max(1, args.per_device_batch_size) * rollout_dp
    if async_rollout:
        rollout_batch = max(1, args.per_device_batch_size)
    ppo_mini_batch = args.rlcsd_ppo_mini_batch_size if args.rlcsd_ppo_mini_batch_size > 0 else rollout_batch
    total_seq_len = args.rlcsd_max_prompt_length + args.rlcsd_max_response_length
    actor_token_budget = (
        args.rlcsd_actor_max_token_len_per_gpu if args.rlcsd_actor_max_token_len_per_gpu > 0 else total_seq_len
    )
    experiment_name = build_auto_run_name(args)
    save_freq = max(1, args.checkpointing_steps)
    total_epochs = max(1, args.num_train_epochs)
    if args.max_train_steps > 0:
        # verl treats total_training_steps as the precise backward-step cap.
        total_epochs = max(total_epochs, 1)

    hydra_args: list[str] = []
    append_hydra_arg(hydra_args, "algorithm.adv_estimator", "grpo")
    append_hydra_arg(hydra_args, "algorithm.use_kl_in_reward", False)
    append_hydra_arg(hydra_args, "trainer.val_before_train", False)
    append_hydra_arg(hydra_args, "trainer.use_legacy_worker_impl", "disable" if async_rollout else "enable")
    append_hydra_arg(hydra_args, "trainer.critic_warmup", 0)
    trainer_loggers = "['console','tensorboard','file']"
    if args.with_tracking and args.wandb_mode != "disabled":
        trainer_loggers = "['console','tensorboard','file','wandb']"
    append_hydra_arg(hydra_args, "trainer.logger", trainer_loggers)
    append_hydra_arg(hydra_args, "trainer.project_name", args.wandb_project if args.wandb_project else "olmo3-rlcsd")
    append_hydra_arg(hydra_args, "trainer.experiment_name", experiment_name)
    append_hydra_arg(hydra_args, "trainer.default_local_dir", output_path)
    append_hydra_arg(hydra_args, "trainer.n_gpus_per_node", train_gpus)
    append_hydra_arg(hydra_args, "trainer.nnodes", train_nodes)
    append_hydra_arg(hydra_args, "trainer.save_freq", save_freq)
    append_hydra_arg(hydra_args, "trainer.test_freq", 0)
    append_hydra_arg(hydra_args, "trainer.total_epochs", total_epochs)
    if args.max_train_steps > 0:
        append_hydra_arg(hydra_args, "trainer.total_training_steps", args.max_train_steps)
    append_hydra_arg(hydra_args, "data.train_files", train_file)
    append_hydra_arg(hydra_args, "data.val_files", val_file)
    append_hydra_arg(hydra_args, "data.prompt_key", "prompt")
    append_hydra_arg(hydra_args, "data.train_batch_size", 0 if async_rollout else rollout_batch)
    if async_rollout:
        append_hydra_arg(hydra_args, "data.gen_batch_size", 1)
        append_hydra_arg(hydra_args, "data.return_raw_chat", True)
    append_hydra_arg(hydra_args, "data.val_batch_size", max(1, min(8, rollout_batch)))
    append_hydra_arg(hydra_args, "data.train_max_samples", -1)
    append_hydra_arg(hydra_args, "data.max_prompt_length", args.rlcsd_max_prompt_length)
    append_hydra_arg(hydra_args, "data.max_response_length", args.rlcsd_max_response_length)
    append_hydra_arg(hydra_args, "data.filter_overlong_prompts", True)
    append_hydra_arg(hydra_args, "data.truncation", "error")
    append_hydra_arg(hydra_args, "data.shuffle", False)
    append_hydra_plus_arg(hydra_args, "data.apply_chat_template_kwargs.enable_thinking", True)
    append_hydra_plus_arg(hydra_args, "data.val_apply_chat_template_kwargs.enable_thinking", True)
    append_hydra_plus_arg(hydra_args, "data.thinking_system_prompt", True)
    if olmo3_sink_enabled(args):
        append_hydra_arg(hydra_args, "data.trust_remote_code", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.nccl_timeout", 7200)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.path", args.model_path)
    if olmo3_sink_enabled(args):
        append_hydra_arg(hydra_args, "actor_rollout_ref.model.trust_remote_code", True)
        append_hydra_arg(hydra_args, "actor_rollout_ref.model.external_lib", "olmo3_sink.verl_bootstrap")
        append_hydra_plus_arg(
            hydra_args,
            "actor_rollout_ref.model.override_config.attn_implementation",
            args.olmo3_sink_attn_implementation,
        )
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.use_remove_padding", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.enable_gradient_checkpointing", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.lora.merge", False)
    append_hydra_arg(hydra_args, "actor_rollout_ref.model.lora_rank", 0)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.lr", args.learning_rate)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.weight_decay", args.weight_decay)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.optim.lr_warmup_steps", max(1, int(args.warmup_ratio * max(args.max_train_steps, 1))))
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.grad_clip", args.max_grad_norm)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_mini_batch_size", ppo_mini_batch)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", args.rlcsd_ppo_micro_batch_size_per_gpu)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_dynamic_bsz", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", actor_token_budget)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_kl_loss", args.rlcsd_method == "rlcsd")
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.kl_loss_coef", 0.001 if args.rlcsd_method == "rlcsd" else 0.0)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.kl_loss_type", "low_var_kl")
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.entropy_coeff", 0)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.strategy", "fsdp2")
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.model_dtype", "bf16")
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.param_offload", False)
    append_hydra_arg(hydra_args, "actor_rollout_ref.actor.fsdp_config.optimizer_offload", False)
    append_hydra_arg(
        hydra_args,
        "actor_rollout_ref.actor.policy_loss.loss_mode",
        "vanilla" if args.rlcsd_method == "grpo" else args.rlcsd_method,
    )
    clip_ratio_low = args.rlcsd_clip_ratio_low
    clip_ratio_high = args.rlcsd_clip_ratio_high
    if args.rlcsd_method == "cispo":
        if clip_ratio_low < 0:
            clip_ratio_low = 10.0
        if clip_ratio_high < 0:
            clip_ratio_high = 0.2
    if clip_ratio_low >= 0:
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.clip_ratio_low", clip_ratio_low)
    if clip_ratio_high >= 0:
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.clip_ratio_high", clip_ratio_high)
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.rlcsd_positive_threshold", args.rlcsd_positive_threshold)
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.rlcsd_negative_threshold", args.rlcsd_negative_threshold)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.temperature", args.rlcsd_rollout_temperature)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.top_p", args.rlcsd_rollout_top_p)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.top_k", args.rlcsd_rollout_top_k)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", total_seq_len)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.tensor_model_parallel_size", rollout_tp)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.name", "vllm")
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.gpu_memory_utilization", args.rlcsd_vllm_gpu_memory_utilization)
    if args.rlcsd_vllm_quantization:
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.quantization", args.rlcsd_vllm_quantization)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.n", args.rlcsd_group_size)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.load_format", "safetensors")
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.layered_summon", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.max_model_len", total_seq_len)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.max_num_batched_tokens", total_seq_len)
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.val_response_length", args.rlcsd_max_response_length)
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.privileged_text_mode", "solution_answer")
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.teacher_enable_thinking", True)
    append_hydra_plus_arg(hydra_args, "actor_rollout_ref.rollout.custom.thinking_system_prompt", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.val_kwargs.n", 1)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.val_kwargs.do_sample", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.val_kwargs.temperature", args.rlcsd_rollout_temperature)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.val_kwargs.top_p", args.rlcsd_rollout_top_p)
    append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.val_kwargs.top_k", args.rlcsd_rollout_top_k)
    append_hydra_arg(hydra_args, "actor_rollout_ref.ref.log_prob_use_dynamic_bsz", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", total_seq_len)
    append_hydra_arg(hydra_args, "actor_rollout_ref.ref.fsdp_config.param_offload", True)
    append_hydra_arg(hydra_args, "actor_rollout_ref.ref.strategy", "fsdp2")
    append_hydra_arg(hydra_args, "actor_rollout_ref.ref.fsdp_config.model_dtype", "bf16")
    if async_rollout:
        append_hydra_arg(hydra_args, "actor_rollout_ref.hybrid_engine", False)
        append_hydra_arg(hydra_args, "actor_rollout_ref.actor.use_rollout_log_probs", True)
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.mode", "async")
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.calculate_log_probs", True)
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.data_parallel_size", args.rlcsd_rollout_data_parallel_size)
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.nnodes", rollout_nodes)
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.n_gpus_per_node", rollout_gpus)
        append_hydra_arg(hydra_args, "actor_rollout_ref.rollout.checkpoint_engine.backend", "nccl")
        append_hydra_arg(hydra_args, "algorithm.rollout_correction.bypass_mode", True)
        append_hydra_arg(hydra_args, "rollout.nnodes", rollout_nodes)
        append_hydra_arg(hydra_args, "rollout.n_gpus_per_node", rollout_gpus)
        required_samples = max(1, ppo_mini_batch * max(1, args.rlcsd_async_require_batches))
        total_rollout_steps = args.rlcsd_async_total_rollout_steps
        if total_rollout_steps <= 0 and args.max_train_steps > 0:
            total_rollout_steps = max(1, args.max_train_steps) * required_samples
        if total_rollout_steps <= 0:
            total_rollout_steps = 100
        append_hydra_arg(hydra_args, "rollout.total_rollout_steps", total_rollout_steps)
        append_hydra_arg(hydra_args, "async_training.staleness_threshold", args.rlcsd_async_staleness_threshold)
        append_hydra_arg(
            hydra_args,
            "async_training.trigger_parameter_sync_step",
            max(1, args.rlcsd_async_trigger_parameter_sync_step),
        )
        append_hydra_arg(hydra_args, "async_training.require_batches", max(1, args.rlcsd_async_require_batches))
        append_hydra_arg(hydra_args, "async_training.partial_rollout", args.rlcsd_async_partial_rollout == "true")
        append_hydra_arg(hydra_args, "async_training.use_trainer_do_validate", False)
    append_hydra_arg(hydra_args, "reward.reward_manager.name", "naive")
    append_hydra_arg(hydra_args, "reward.custom_reward_function.path", Path(__file__).resolve().with_name("verl_rlcsd_adapter.py"))
    append_hydra_arg(hydra_args, "reward.custom_reward_function.name", "compute_score")
    hydra_args.extend(parse_extra_hydra_overrides(args.rlcsd_extra_overrides))
    return hydra_args


def run_verl_rlcsd(args: argparse.Namespace) -> None:
    if sweep_requested(args):
        raise ValueError("Learning-rate/optimizer sweeps are not implemented for --backend verl_rlcsd.")
    if rlcsd_async_rollout_enabled(args) and args.rlcsd_method not in RLCSD_SIMPLE_ASYNC_METHODS:
        raise ValueError(
            "--rlcsd_async_rollout true currently supports simple methods only: "
            f"{', '.join(sorted(RLCSD_SIMPLE_ASYNC_METHODS))}. "
            "RLCSD teacher-input construction is still tied to the legacy colocated trainer path."
        )

    output_path = Path(args.output_path).expanduser().resolve()
    logdir = Path(args.logdir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir(args, "verl_rlcsd_cache")
    configure_logging(logdir, args)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache_environment(os.environ, cache_dir)
    if olmo3_sink_enabled(args):
        prepare_olmo3_sink_model(args, cache_dir)
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, logdir, "verl_rlcsd_parent", launcher_environment_summary(args))

    rlcsd_dir = find_rlcsd_dir(args.rlcsd_dir)
    env = build_verl_rlcsd_env(rlcsd_dir, cache_dir, args)
    if rlcsd_async_rollout_enabled(args):
        env["VERL_RLCSD_FULLY_ASYNC"] = "1"
    train_file, val_file = maybe_prepare_verl_rlcsd_dataset(args, cache_dir)
    hydra_args = build_verl_rlcsd_hydra_args(args, train_file, val_file, output_path)
    launcher = Path(__file__).resolve().with_name("verl_rlcsd_main.py")
    command = [sys.executable, str(launcher), *hydra_args]
    (logdir / "VERL_RLCSD_COMMAND.txt").write_text(command_to_text(command) + "\n", encoding="utf-8")

    node_rank = effective_node_rank(args)
    if effective_num_nodes(args) <= 1 and node_rank is None:
        node_rank = 0
    if node_rank is None:
        raise ValueError("verl/RLCSD launch requires --node_rank or PBS GLOBAL_RANK.")
    if node_rank < 0 or node_rank >= effective_num_nodes(args):
        raise ValueError(f"Resolved verl/RLCSD node_rank={node_rank} outside num_nodes={effective_num_nodes(args)}.")

    logging.info(
        "verl/RLCSD launch summary: node_rank=%s num_nodes=%s gpus_per_node=%s async_rollout=%s ray=%s rlcsd_dir=%s command=%s",
        node_rank,
        effective_num_nodes(args),
        effective_num_gpus(args),
        rlcsd_async_rollout_enabled(args),
        grpo_ray_address(args),
        rlcsd_dir,
        command_to_text(command),
    )
    if args.dry_run_launch:
        logging.info("verl/RLCSD dry-run requested; not starting Ray or launching training.")
        return

    try:
        if node_rank == 0:
            start_ray_head(args, env)
            run_verl_command_with_checkpoint_watcher(command, env, args, output_path)
        else:
            start_ray_worker(args, env)
            monitor_grpo_worker(args, env)
    finally:
        if args.grpo_stop_ray_on_exit == "true":
            stop_ray(env)


def build_sweep_child_command(
    args: argparse.Namespace,
    trial: SweepTrial,
    sweep_name: str,
) -> tuple[list[str], dict[str, str]]:
    parent_output = Path(args.output_path).expanduser().resolve()
    parent_logdir = Path(args.logdir).expanduser().resolve()
    trial_run_name = build_config_run_name(args, trial.learning_rate_token, trial.optimizer)
    trial_output = parent_output / trial_run_name
    trial_logdir = parent_logdir / trial_run_name

    shared_cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir(args, "sweep_shared_cache")
    shared_checkpoint_cache = (
        Path(args.olmo_core_checkpoint_cache).expanduser().resolve()
        if args.olmo_core_checkpoint_cache
        else default_olmo_checkpoint_cache(args)
    )
    shared_dataset_cache = (
        Path(args.olmo_core_dataset_cache).expanduser().resolve()
        if args.olmo_core_dataset_cache
        else default_olmo_dataset_cache(args)
    )

    command = [sys.executable, str(Path(__file__).resolve()), *strip_child_override_args(sys.argv[1:])]
    append_cli_override(command, "--learning_rate", format_float_token(trial.learning_rate))
    append_cli_override(command, "--optimizer", trial.optimizer)
    append_cli_override(command, "--output_path", trial_output)
    append_cli_override(command, "--logdir", trial_logdir)
    append_cli_override(command, "--cache_dir", shared_cache_dir)
    append_cli_override(command, "--olmo_core_checkpoint_cache", shared_checkpoint_cache)
    append_cli_override(command, "--olmo_core_dataset_cache", shared_dataset_cache)
    append_cli_override(command, "--hf_log_upload", "false")

    paths = {
        "output_path": str(trial_output),
        "logdir": str(trial_logdir),
        "cache_dir": str(shared_cache_dir),
        "olmo_core_checkpoint_cache": str(shared_checkpoint_cache),
        "olmo_core_dataset_cache": str(shared_dataset_cache),
    }
    return command, paths


def run_sweep_child_command(command: list[str], env: dict[str, str], trial: SweepTrial) -> int:
    logging.info("Starting sweep trial %03d: %s", trial.index, " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        relay_sweep_child_log_line(line, trial)
    return process.wait()


def parse_child_log_line(line: str) -> tuple[int, str]:
    stripped = line.rstrip()
    match = LOG_LINE_RE.match(stripped)
    if match is None:
        return logging.INFO, stripped
    level_name = match.group("level")
    message = match.group("message")
    return LOG_LEVELS[level_name], message


def relay_child_log_line(line: str, prefix: str | None = None) -> None:
    if should_suppress_child_log_line(line):
        return
    level, message = parse_child_log_line(line)
    if prefix:
        message = f"{prefix} {message}"
    logging.getLogger().log(level, "%s", message, stacklevel=2)


def relay_sweep_child_log_line(line: str, trial: SweepTrial) -> None:
    if should_suppress_child_log_line(line):
        return
    level, message = parse_child_log_line(line)
    message = f"[Trial {trial.index:03d}] {message}"
    timestamp = datetime.now().strftime("%H:%M:%S,%f")[:-3]
    level_name = logging.getLevelName(level)
    formatted = f"{timestamp} {level_name} {message}\n"
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        sys.stdout.write(formatted)
        sys.stdout.flush()
        return
    for handler in root_logger.handlers:
        if level < handler.level:
            continue
        stream = getattr(handler, "stream", None)
        if stream is None:
            continue
        handler.acquire()
        try:
            stream.write(formatted)
            handler.flush()
        finally:
            handler.release()


def run_sweep(args: argparse.Namespace) -> int:
    parent_logdir = Path(args.logdir).expanduser().resolve()
    configure_logging(parent_logdir, args)
    guard_top_level_launcher(args)
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, parent_logdir, "sweep_parent", launcher_environment_summary(args))

    sweep_name = sanitize_slug_part(args.sweep_name or "lr_optimizer_sweep")
    trials = build_sweep_trials(args)
    writer = sweep_status_writer(args)
    status_path = parent_logdir / sweep_name / "sweep_status.jsonl"

    logging.info(
        "Starting sweep '%s' with %d trial(s): learning_rates=%s optimizers=%s dry_run=%s continue_on_failure=%s",
        sweep_name,
        len(trials),
        effective_learning_rates_arg(args) or format_float_token(args.learning_rate),
        effective_optimizers_arg(args) or normalized_optimizer_name(args),
        args.sweep_dry_run,
        args.sweep_continue_on_failure,
    )

    failures = 0
    for trial in trials:
        command, paths = build_sweep_child_command(args, trial, sweep_name)
        trial_logdir = Path(paths["logdir"])
        start_time = time.time()
        record: dict[str, object] = {
            "event": "trial_start",
            "sweep_name": sweep_name,
            "trial_index": trial.index,
            "trial_slug": trial.slug,
            "learning_rate": trial.learning_rate,
            "learning_rate_token": trial.learning_rate_token,
            "optimizer": trial.optimizer,
            "paths": paths,
            "command": command,
            "time": start_time,
        }
        if writer:
            trial_logdir.mkdir(parents=True, exist_ok=True)
            (trial_logdir / "TRIAL_COMMAND.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
            append_sweep_status(status_path, record)

        if args.sweep_dry_run:
            logging.info("Sweep dry-run trial %03d: %s", trial.index, " ".join(command))
            if writer:
                append_sweep_status(
                    status_path,
                    {
                        **record,
                        "event": "trial_complete",
                        "status": "dry_run",
                        "exit_code": 0,
                        "elapsed_seconds": 0.0,
                    },
                )
            continue

        env = os.environ.copy()
        env.update(
            {
                "OLMO_SWEEP_CHILD": "1",
                "OLMO_SWEEP_NAME": sweep_name,
                "OLMO_SWEEP_TRIAL_INDEX": str(trial.index),
                "OLMO_SWEEP_TRIAL_SLUG": trial.slug,
                "OLMO_SWEEP_LEARNING_RATE": format_float_token(trial.learning_rate),
                "OLMO_SWEEP_OPTIMIZER": trial.optimizer,
            }
        )
        status = run_sweep_child_command(command, env, trial)
        elapsed = time.time() - start_time
        if writer:
            append_sweep_status(
                status_path,
                {
                    **record,
                    "event": "trial_complete",
                    "exit_code": status,
                    "elapsed_seconds": elapsed,
                    "time": time.time(),
                },
            )
        if status != 0:
            failures += 1
            logging.error("Sweep trial %03d failed with exit code %d.", trial.index, status)
            if not args.sweep_continue_on_failure:
                return status
        else:
            logging.info("Sweep trial %03d completed in %.1f seconds.", trial.index, elapsed)

    if failures:
        logging.error("Sweep finished with %d failed trial(s).", failures)
        return 1
    logging.info("Sweep finished successfully with %d trial(s).", len(trials))
    return 0


def wait_for_prepare_marker(marker_file: Path, fail_file: Path, timeout_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = read_prepare_marker(marker_file)
        if result is not None:
            return result
        if fail_file.is_file():
            raise RuntimeError(f"Node 0 preparation failed: {fail_file.read_text().strip()}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for multinode preparation marker: {marker_file}")


def read_prepare_marker(marker_file: Path) -> str | None:
    if not marker_file.is_file():
        return None
    payload = json.loads(marker_file.read_text())
    return str(payload.get("result", ""))


def run_once_on_node0(
    args: argparse.Namespace,
    marker_dir: Path,
    marker_name: str,
    work_fn,
    timeout_seconds: int = 86400,
) -> str:
    num_nodes = effective_num_nodes(args)
    if num_nodes <= 1:
        return str(work_fn())

    node_rank = effective_node_rank(args)
    if node_rank is None:
        raise ValueError("Multinode preparation requires --node_rank or PBS GLOBAL_RANK.")

    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_dir / f"{marker_name}.done.json"
    fail_file = marker_dir / f"{marker_name}.failed.txt"

    if node_rank == 0:
        try:
            existing_result = read_prepare_marker(marker_file)
            if existing_result is not None:
                logging.info("Using existing node 0 preparation marker %s", marker_file)
                return existing_result
            if fail_file.exists():
                fail_file.unlink()
            result = str(work_fn())
            tmp_marker = marker_file.with_suffix(".tmp")
            tmp_marker.write_text(json.dumps({"result": result}) + "\n")
            tmp_marker.replace(marker_file)
            return result
        except Exception as exc:
            fail_file.write_text(f"{type(exc).__name__}: {exc}\n")
            raise

    logging.info("Waiting for node 0 preparation marker %s", marker_file)
    result = wait_for_prepare_marker(marker_file, fail_file, timeout_seconds)
    logging.info("Node 0 preparation complete: %s", result or marker_name)
    return result


def olmo_core_sequence_length(requested: int) -> int:
    if requested <= 0:
        raise ValueError("max_seq_length must be positive")
    return requested


def log_sequence_length_adjustment(requested: int) -> None:
    effective = olmo_core_sequence_length(requested)
    if effective != requested:
        logging.warning(
            "OLMo-core packed datasets require power-of-two sequence lengths; "
            "using %d internally for requested --max_seq_length %d.",
            effective,
            requested,
        )


def effective_checkpoint_intervals(args: argparse.Namespace) -> tuple[int, int | None]:
    checkpointing_steps = args.checkpointing_steps
    ephemeral_interval = args.ephemeral_save_interval
    if ephemeral_interval <= 0:
        return checkpointing_steps, None
    if ephemeral_interval < checkpointing_steps:
        return checkpointing_steps, ephemeral_interval

    checkpointing_steps = max(checkpointing_steps, ephemeral_interval + 1, 2)
    return checkpointing_steps, ephemeral_interval


def log_checkpoint_interval_adjustment(args: argparse.Namespace) -> None:
    checkpointing_steps, ephemeral_interval = effective_checkpoint_intervals(args)
    if checkpointing_steps != args.checkpointing_steps or ephemeral_interval != args.ephemeral_save_interval:
        logging.warning(
            "OLMo-core requires ephemeral_save_interval < checkpointing_steps; "
            "using checkpointing_steps=%s and ephemeral_save_interval=%s internally.",
            checkpointing_steps,
            ephemeral_interval,
        )


def effective_optimizer_state_dtype(args: argparse.Namespace) -> str | None:
    if args.optimizer_state_dtype == "float32":
        return None
    if args.optimizer_state_dtype == "bfloat16":
        return "bfloat16"

    num_gpus = effective_num_gpus(args)
    world_size = max(1, num_gpus) * max(1, effective_num_nodes(args))
    if world_size <= 1 and args.max_seq_length >= 16000:
        return "bfloat16"
    return None


def normalized_optimizer_name(args: argparse.Namespace) -> str:
    if args.optimizer in {"adamw_8bits", "torchao_adamw_8bit"}:
        return "adamw_8bit"
    return args.optimizer


def apply_optimizer_name_env(env: dict[str, str], args: argparse.Namespace) -> None:
    env["OLMO_OPTIMIZER"] = normalized_optimizer_name(args)
    env["OLMO_ADAMW_8BIT_BLOCK_SIZE"] = str(args.adamw_8bit_block_size)
    env["OLMO_ADAMW_8BIT_BF16_STOCHASTIC_ROUND"] = (
        "1" if args.adamw_8bit_bf16_stochastic_round else "0"
    )


def apply_optimizer_state_dtype_env(env: dict[str, str], args: argparse.Namespace) -> None:
    optim_dtype = effective_optimizer_state_dtype(args)
    if optim_dtype is None:
        env.pop("OLMO_OPTIM_DTYPE", None)
        return
    env["OLMO_OPTIM_DTYPE"] = optim_dtype
    logging.warning(
        "Using %s optimizer state for this run. Set --optimizer_state_dtype float32 "
        "to force the default OLMo-core optimizer state dtype.",
        optim_dtype,
    )


def apply_feed_forward_debug_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.feed_forward_memory_profile == "true":
        env["OLMO_FF_MEMORY_PROFILE"] = "1"
        env["OLMO_FF_MEMORY_PROFILE_RANKS"] = args.feed_forward_memory_profile_ranks
        env["OLMO_FF_MEMORY_PROFILE_MAX_CALLS"] = str(args.feed_forward_memory_profile_max_calls)
        env["OLMO_FF_MEMORY_PROFILE_SYNC"] = (
            "1" if args.feed_forward_memory_profile_sync == "true" else "0"
        )
        env["OLMO_FF_MEMORY_PROFILE_ALLOW_COMPILE"] = (
            "1" if args.feed_forward_memory_profile_allow_compile == "true" else "0"
        )
    else:
        env.pop("OLMO_FF_MEMORY_PROFILE", None)
        env.pop("OLMO_FF_MEMORY_PROFILE_RANKS", None)
        env.pop("OLMO_FF_MEMORY_PROFILE_MAX_CALLS", None)
        env.pop("OLMO_FF_MEMORY_PROFILE_SYNC", None)
        env.pop("OLMO_FF_MEMORY_PROFILE_ALLOW_COMPILE", None)


def apply_checkpoint_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.disable_checkpoints:
        env["OLMO_DISABLE_CHECKPOINTS"] = "1"
        logging.warning("Disabling OLMo-core checkpoint callbacks for this run.")
    else:
        env.pop("OLMO_DISABLE_CHECKPOINTS", None)


def build_finetune_args(args: argparse.Namespace, dataset_ref: str, cache_dir: Path, output_path: Path) -> list[str]:
    tokenizer_path = args.tokenizer_path or args.model_path
    config_name = args.config_name or infer_config_name(Path(args.model_path))
    sequence_length = olmo_core_sequence_length(args.max_seq_length)
    checkpointing_steps, ephemeral_interval = effective_checkpoint_intervals(args)
    rope_scaling_factor = effective_rope_scaling_factor(
        args,
        Path(args.model_path).expanduser().resolve(),
        config_name,
    )
    finetune_args = [
        "--model_name_or_path",
        args.model_path,
        "--tokenizer_name_or_path",
        tokenizer_path,
        "--max_seq_length",
        str(sequence_length),
        "--per_device_train_batch_size",
        str(args.per_device_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--weight_decay",
        str(args.weight_decay),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--num_epochs",
        str(args.num_train_epochs),
        "--checkpointing_steps",
        str(checkpointing_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--mixer_list",
        dataset_ref,
        args.dataset_weight,
        "--mixer_list_splits",
        args.dataset_split,
        "--local_cache_dir",
        str(cache_dir),
        "--output_dir",
        str(output_path),
        "--activation_memory_budget",
        str(args.activation_memory_budget),
        "--compile_model",
        args.compile_model,
    ]
    if ephemeral_interval is not None:
        finetune_args.extend(["--ephemeral_save_interval", str(ephemeral_interval)])
    if config_name:
        finetune_args.extend(["--config_name", config_name])
    if rope_scaling_factor is not None:
        finetune_args.extend(
            [
                "--rope_scaling_factor",
                str(rope_scaling_factor),
                "--rope_scaling_old_context_len",
                str(args.rope_scaling_old_context_len),
                "--rope_scaling_beta_fast",
                str(args.rope_scaling_beta_fast),
                "--rope_scaling_beta_slow",
                str(args.rope_scaling_beta_slow),
            ]
        )
    if args.max_grad_norm > 0:
        finetune_args.extend(["--max_grad_norm", str(args.max_grad_norm)])
    if args.max_train_steps > 0:
        finetune_args.extend(["--max_train_steps", str(args.max_train_steps)])
    attn_implementation = normalize_attn_implementation(args.attn_implementation)
    if attn_implementation:
        finetune_args.extend(["--attn_implementation", attn_implementation])
    if args.chat_template_name:
        finetune_args.extend(["--chat_template_name", args.chat_template_name])
    chat_template_model = effective_chat_template_model(args)
    if chat_template_model:
        finetune_args.extend(["--chat_template_model", chat_template_model])
    transform_names = dataset_transform_names(args)
    if transform_names:
        finetune_args.extend(["--transform_fn", *transform_names])
    if args.with_tracking:
        finetune_args.extend(["--with_tracking", "--wandb_project", args.wandb_project])
        if args.wandb_entity:
            finetune_args.extend(["--wandb_entity", args.wandb_entity])
    return finetune_args


def build_torchrun_prefix(args: argparse.Namespace) -> list[str]:
    num_gpus = effective_num_gpus(args)
    num_nodes = effective_num_nodes(args)
    prefix = ["torchrun", "--nproc_per_node", str(num_gpus)]
    if num_nodes > 1:
        node_rank = effective_node_rank(args)
        master_addr = effective_master_addr(args)
        if node_rank is None or not master_addr:
            logging.error(
                "Multinode launcher env snapshot before failure: args.node_rank=%s GLOBAL_RANK=%s NODE_RANK=%s "
                "RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s MASTER_ADDR=%s MASTER_PORT=%s",
                args.node_rank,
                os.environ.get("GLOBAL_RANK", "unset"),
                os.environ.get("NODE_RANK", "unset"),
                os.environ.get("RANK", "unset"),
                os.environ.get("LOCAL_RANK", "unset"),
                os.environ.get("WORLD_SIZE", "unset"),
                os.environ.get("MASTER_ADDR", "unset"),
                os.environ.get("MASTER_PORT", "unset"),
            )
            raise ValueError(
                "Multinode launch requires node rank and master address. "
                "Use --node_rank/--master_addr, or provide PBS GLOBAL_RANK, MASTER_ADDR, and WORLD_SIZE."
            )
        if node_rank < 0 or node_rank >= num_nodes:
            raise ValueError(
                f"Resolved node_rank={node_rank} is outside nnodes={num_nodes}. "
                "For the host PBS launcher, GLOBAL_RANK must be the node rank and WORLD_SIZE must be the node count. "
                "Use --world_size_mode auto or processes only for process-style local simulations."
            )
        prefix.extend(
            [
                "--nnodes",
                str(num_nodes),
                "--node_rank",
                str(node_rank),
                "--master_addr",
                master_addr,
                "--master_port",
                str(effective_master_port(args)),
            ]
        )
        logging.info(
            "Using multinode torchrun: nnodes=%d node_rank=%d nproc_per_node=%d master=%s:%d",
            num_nodes,
            node_rank,
            num_gpus,
            master_addr,
            effective_master_port(args),
        )
    else:
        logging.info("Using single-node torchrun: nproc_per_node=%d", num_gpus)
    return prefix


def run_dry_launch(args: argparse.Namespace) -> None:
    logdir = Path(args.logdir).expanduser().resolve()
    configure_logging(logdir, args)
    log_train_engine_runtime_info(args)
    try:
        model_arch = normalize_model_arch(args.model_arch, Path(args.model_path).expanduser().resolve())
    except Exception:
        model_arch = args.model_arch if args.model_arch != "auto" else None
    collect_run_environment_info(args, logdir, "dry_launch", launcher_environment_summary(args))
    log_launch_summary(args, model_arch)
    validate_parallel_topology(args, model_arch)
    validate_training_features(args, model_arch)
    log_requested_batch_topology(args, model_arch)
    torchrun_prefix = build_torchrun_prefix(args)
    logging.info("Dry-run launch requested; no dataset/model validation or training will run.")
    logging.info("Resolved internal torchrun prefix: %s", " ".join(torchrun_prefix))
    logging.info(
        "Launcher env snapshot: GLOBAL_RANK=%s WORLD_SIZE=%s MASTER_ADDR=%s MASTER_PORT=%s LOCAL_RANK=%s",
        os.environ.get("GLOBAL_RANK", "unset"),
        os.environ.get("WORLD_SIZE", "unset"),
        os.environ.get("MASTER_ADDR", "unset"),
        os.environ.get("MASTER_PORT", "unset"),
        os.environ.get("LOCAL_RANK", "unset"),
    )


def guard_top_level_launcher(args: argparse.Namespace) -> None:
    if args.internal_backend is not None:
        return
    torchrun_child_vars = [name for name in TORCHRUN_CHILD_ENV_KEYS if os.environ.get(name) not in {None, ""}]
    if torchrun_child_vars:
        raise RuntimeError(
            "Do not launch this submission with an external torchrun. "
            "Run one Singularity container per node and pass normal train.py arguments; "
            "src/train.py wraps torchrun internally after resolving GLOBAL_RANK, WORLD_SIZE, "
            f"MASTER_ADDR, and MASTER_PORT. Inherited torchrun env vars: {', '.join(torchrun_child_vars)}."
        )

    inherited_rank = os.environ.get("RANK")
    if inherited_rank in {None, ""}:
        return
    if os.environ.get("LOCAL_RANK") in {None, ""}:
        # Some host launchers export RANK as a node-rank alias. This is safe as
        # long as torchrun-specific child env vars are absent; we strip it from
        # subprocess environments before starting the internal torchrun.
        args._host_rank_alias = inherited_rank
        return
    raise RuntimeError(
        "Do not launch this submission with an external torchrun. "
        "Run one Singularity container per node and pass normal train.py arguments; "
        "src/train.py wraps torchrun internally after resolving GLOBAL_RANK, WORLD_SIZE, "
        "MASTER_ADDR, and MASTER_PORT. Inherited rank env vars: RANK, LOCAL_RANK."
    )


def dataset_is_olmo_core_numpy(dataset_path: Path) -> bool:
    return dataset_path.is_dir() and bool(list(dataset_path.glob("token_ids_part_*.npy"))) and bool(
        list(dataset_path.glob("labels_mask_part_*.npy"))
    )


def versioned_olmo_core_dataset_cache(args: argparse.Namespace, dataset_cache: Path) -> Path:
    return dataset_cache / olmo_core_dataset_cache_version(args)


def ensure_olmo_core_dataset(
    args: argparse.Namespace,
    dataset_ref: str,
    dataset_cache: Path,
    open_instruct_dir: Path,
    env: dict[str, str],
) -> Path:
    original_dataset_path = Path(args.dataset_path).expanduser().resolve()
    if dataset_is_olmo_core_numpy(original_dataset_path):
        logging.info("Using existing OLMo-core NumPy dataset at %s", original_dataset_path)
        return original_dataset_path

    cache_version = olmo_core_dataset_cache_version(args)
    dataset_cache = versioned_olmo_core_dataset_cache(args, dataset_cache)
    if dataset_is_olmo_core_numpy(dataset_cache):
        logging.info("Using cached OLMo-core NumPy dataset at %s", dataset_cache)
        return dataset_cache

    logging.info(
        "Preparing OLMo-core dataset cache version %s at %s",
        cache_version,
        dataset_cache,
    )
    dataset_cache.mkdir(parents=True, exist_ok=True)
    conversion_script = open_instruct_dir / "scripts" / "data" / "convert_sft_data_for_olmocore.py"
    command = [
        sys.executable,
        str(conversion_script),
        "--dataset_mixer_list",
        dataset_ref,
        args.dataset_weight,
        "--dataset_mixer_list_splits",
        args.dataset_split,
        "--tokenizer_name_or_path",
        args.tokenizer_path or args.model_path,
        "--output_dir",
        str(dataset_cache),
        "--max_seq_length",
        str(olmo_core_sequence_length(args.max_seq_length)),
        "--dataset_local_cache_dir",
        str(Path(args.cache_dir).expanduser().resolve() if args.cache_dir else dataset_cache / "hf_dataset_cache"),
        "--batch_size",
        str(args.dataset_map_batch_size),
        "--resume",
        "True",
        "--dataset_backend",
        args.dataset_backend,
    ]
    if args.chat_template_name:
        command.extend(["--chat_template_name", args.chat_template_name])
    chat_template_model = effective_chat_template_model(args)
    if chat_template_model:
        command.extend(["--chat_template_model", chat_template_model])
    transform_names = dataset_transform_names(args)
    if transform_names:
        command.extend(["--dataset_transform_fn", *transform_names])
    run_command(command, env)
    if not dataset_is_olmo_core_numpy(dataset_cache):
        raise RuntimeError(f"OLMo-core dataset conversion did not produce expected files in {dataset_cache}")
    return dataset_cache


def checkpoint_is_olmo_core(checkpoint_path: Path) -> bool:
    if (checkpoint_path / "model_and_optim" / ".metadata").is_file():
        return True
    if checkpoint_path.name == "model_and_optim" and (checkpoint_path / ".metadata").is_file():
        return True
    return False


def olmo_core_checkpoint_to_load(checkpoint_path: Path) -> Path:
    if checkpoint_path.name == "model_and_optim" and (checkpoint_path / ".metadata").is_file():
        return checkpoint_path
    if (checkpoint_path / "model_and_optim" / ".metadata").is_file():
        return checkpoint_path / "model_and_optim"
    raise FileNotFoundError(f"Could not find OLMo-core model_and_optim checkpoint under {checkpoint_path}")


def ensure_olmo_core_checkpoint(
    args: argparse.Namespace,
    model_path: Path,
    checkpoint_cache: Path,
    olmo_core_dir: Path,
    model_arch: str,
    env: dict[str, str],
) -> Path:
    if checkpoint_is_olmo_core(model_path):
        logging.info("Using existing OLMo-core checkpoint at %s", model_path)
        return olmo_core_checkpoint_to_load(model_path)

    if checkpoint_is_olmo_core(checkpoint_cache):
        logging.info("Using cached OLMo-core checkpoint at %s", checkpoint_cache)
        return olmo_core_checkpoint_to_load(checkpoint_cache)

    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    conversion_script = olmo_core_dir / "src" / "examples" / "huggingface" / "convert_checkpoint_from_hf.py"
    command = [
        sys.executable,
        str(conversion_script),
        "--checkpoint-input-path",
        str(model_path),
        "--output-dir",
        str(checkpoint_cache),
        "--model-arch",
        model_arch,
        "--tokenizer",
        "auto",
        "--device",
        "cuda" if cuda_device_count() else "cpu",
    ]
    if args.convert_validation != "true":
        command.append("--skip-validation")
    if args.attention_sink == "true":
        command.extend(
            [
                "--attention-sink",
                "--attention-sink-init-value",
                str(args.attention_sink_init_value),
            ]
        )
    run_command(command, env)
    return olmo_core_checkpoint_to_load(checkpoint_cache)


def olmo_core_checkpoint_experiment_config_path(checkpoint_to_load: Path) -> Path | None:
    candidates = []
    if checkpoint_to_load.name == "model_and_optim":
        candidates.append(checkpoint_to_load.parent / "config.json")
    candidates.append(checkpoint_to_load / "config.json")
    candidates.append(checkpoint_to_load.parent / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def apply_olmo_core_checkpoint_experiment_config(
    module: ModuleType,
    config: object,
    checkpoint_to_load: Path,
) -> None:
    config_path = olmo_core_checkpoint_experiment_config_path(checkpoint_to_load)
    if config_path is None:
        logging.info("No converted OLMo-core experiment config found for %s.", checkpoint_to_load)
        return

    with config_path.open(encoding="utf-8") as file_obj:
        experiment_config = json.load(file_obj)

    model_config = experiment_config.get("model")
    if model_config:
        config.model = module.TransformerConfig.from_dict(model_config)
        logging.warning(
            "Loaded OLMo-core model config from converted checkpoint config: %s "
            "(vocab_size=%s n_layers=%s).",
            config_path,
            getattr(config.model, "vocab_size", "unknown"),
            getattr(config.model, "n_layers", "unknown"),
        )

    tokenizer_config = (experiment_config.get("dataset") or {}).get("tokenizer")
    dataset_config = getattr(config, "dataset", None)
    if tokenizer_config and dataset_config is not None and hasattr(dataset_config, "tokenizer"):
        dataset_config.tokenizer = module.TokenizerConfig.from_dict(tokenizer_config)
        logging.warning(
            "Loaded OLMo-core dataset tokenizer config from converted checkpoint config: "
            "vocab_size=%s eos=%s pad=%s.",
            dataset_config.tokenizer.vocab_size,
            dataset_config.tokenizer.eos_token_id,
            dataset_config.tokenizer.pad_token_id,
        )


def default_global_batch_size_tokens(args: argparse.Namespace) -> int:
    model_arch = normalize_model_arch(args.model_arch, Path(args.model_path).expanduser().resolve())
    dp_degree = effective_data_parallel_degree(args, model_arch)
    pp_degree = effective_pipeline_parallel_degree(args)
    grad_accum_steps = max(args.gradient_accumulation_steps, pp_degree)
    rank_microbatch_sequences = (
        args.rank_microbatch_size_sequences
        if args.rank_microbatch_size_sequences > 0
        else args.per_device_batch_size
    )
    return (
        dp_degree
        * rank_microbatch_sequences
        * grad_accum_steps
        * olmo_core_sequence_length(args.max_seq_length)
    )


def resolved_global_batch_size_tokens(
    args: argparse.Namespace,
    model_arch: str | None = None,
) -> int:
    if args.global_batch_size_tokens > 0 and args.global_batch_size_sequences > 0:
        raise ValueError(
            "--global_batch_size_tokens and --global_batch_size_sequences are mutually exclusive."
        )
    if args.global_batch_size_tokens > 0:
        return args.global_batch_size_tokens
    if args.global_batch_size_sequences > 0:
        return args.global_batch_size_sequences * olmo_core_sequence_length(args.max_seq_length)
    return default_global_batch_size_tokens(args)


def log_requested_batch_topology(
    args: argparse.Namespace,
    model_arch: str | None = None,
) -> None:
    sequence_length = olmo_core_sequence_length(args.max_seq_length)
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)
    dp_degree = effective_data_parallel_degree(args, model_arch)
    world_size = effective_num_nodes(args) * effective_num_gpus(args)
    rank_microbatch_sequences = (
        args.rank_microbatch_size_sequences
        if args.rank_microbatch_size_sequences > 0
        else args.per_device_batch_size
    )
    global_batch_tokens = resolved_global_batch_size_tokens(args, model_arch)
    if global_batch_tokens % sequence_length != 0:
        raise ValueError(
            f"Resolved global batch {global_batch_tokens} tokens must be divisible by "
            f"sequence length {sequence_length}."
        )
    global_batch_sequences = global_batch_tokens // sequence_length
    denominator = rank_microbatch_sequences * dp_degree
    if denominator <= 0 or global_batch_sequences % denominator != 0:
        raise ValueError(
            f"Global batch {global_batch_sequences} sequence(s) must be divisible by "
            f"rank microbatch {rank_microbatch_sequences} * DP {dp_degree}."
        )
    effective_microbatches = global_batch_sequences // denominator
    logging.info(
        "Requested batch topology: world=%d TP=%d CP=%d PP=%d DP=%d; "
        "sequence_length=%d rank_microbatch=%d sequence(s) microbatches=%d "
        "global_batch=%d sequence(s)=%d token(s). "
        "Equation: global_tokens=sequence_length*rank_microbatch*microbatches*DP.",
        world_size,
        tp_degree,
        cp_degree,
        pp_degree,
        dp_degree,
        sequence_length,
        rank_microbatch_sequences,
        effective_microbatches,
        global_batch_sequences,
        global_batch_tokens,
    )


@dataclass
class LocalBatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int
    gpu_type: str
    requested_rank_microbatch_size_sequences: int = 0
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: int | None = None
    tp_degree: int = 1
    pp_degree: int = 1
    min_grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        if self.global_batch_size_tokens <= 0:
            raise ValueError("global_batch_size_tokens must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")

        max_tokens_per_rank = 16_384
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        if self.cp_degree is None and self.world_size > 1 and self.sequence_length > max_tokens_per_rank:
            cp_degree = 2
            while (self.sequence_length // cp_degree) > max_tokens_per_rank:
                cp_degree *= 2
            self.cp_degree = cp_degree

        cp_factor = self.cp_degree or 1
        model_parallel_factor = cp_factor * self.tp_degree * self.pp_degree
        if self.world_size % model_parallel_factor != 0:
            raise ValueError(
                f"world_size={self.world_size} must be divisible by "
                f"CP*TP*PP={model_parallel_factor}"
            )
        dp_world_size = self.world_size // model_parallel_factor
        if self.global_batch_size_tokens % dp_world_size != 0:
            raise ValueError(
                f"global_batch_size_tokens={self.global_batch_size_tokens} must be divisible by dp_world_size={dp_world_size}"
            )

        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size
        if rank_batch_size_tokens % self.sequence_length != 0:
            raise ValueError(
                "rank batch size must be divisible by sequence_length "
                f"(got {rank_batch_size_tokens} and {self.sequence_length})"
            )
        rank_batch_size_sequences = rank_batch_size_tokens // self.sequence_length
        limit = max(max_tokens_per_rank * cp_factor, self.sequence_length)
        min_grad_accum_steps = max(1, self.pp_degree, self.min_grad_accum_steps)
        if rank_batch_size_sequences < min_grad_accum_steps:
            raise ValueError(
                "global_batch_size_tokens is too small for pipeline parallelism: "
                f"rank batch size {rank_batch_size_tokens} tokens cannot provide "
                f"{min_grad_accum_steps} microbatches of sequence length {self.sequence_length}. "
                f"Use at least {self.sequence_length * min_grad_accum_steps * dp_world_size} global tokens."
            )
        if self.requested_rank_microbatch_size_sequences > 0:
            requested_microbatch_sequences = self.requested_rank_microbatch_size_sequences
            if rank_batch_size_sequences % requested_microbatch_sequences != 0:
                raise ValueError(
                    f"rank batch {rank_batch_size_sequences} sequence(s) must be divisible by "
                    f"requested rank microbatch {requested_microbatch_sequences} sequence(s)."
                )
            requested_grad_accum = rank_batch_size_sequences // requested_microbatch_sequences
            if requested_grad_accum < min_grad_accum_steps:
                raise ValueError(
                    f"requested rank microbatch {requested_microbatch_sequences} sequence(s) produces "
                    f"only {requested_grad_accum} microbatch(es), below the required minimum "
                    f"{min_grad_accum_steps} for PP={self.pp_degree}."
                )
            self.rank_microbatch_size_sequences = requested_microbatch_sequences
            self.rank_microbatch_size_tokens = requested_microbatch_sequences * self.sequence_length
            self.grad_accum_steps = requested_grad_accum
            return
        limit_sequences = max(1, limit // self.sequence_length)
        for grad_accum_steps in range(min_grad_accum_steps, rank_batch_size_sequences + 1):
            if rank_batch_size_sequences % grad_accum_steps != 0:
                continue
            microbatch_sequences = rank_batch_size_sequences // grad_accum_steps
            if microbatch_sequences <= limit_sequences:
                self.grad_accum_steps = grad_accum_steps
                self.rank_microbatch_size_sequences = microbatch_sequences
                self.rank_microbatch_size_tokens = microbatch_sequences * self.sequence_length
                break
        else:
            raise ValueError(
                "Could not choose a valid gradient accumulation schedule for "
                f"{rank_batch_size_sequences} rank-local sequence(s), "
                f"minimum grad_accum_steps={min_grad_accum_steps}, and "
                f"microbatch limit={limit_sequences} sequence(s)."
            )


def load_olmo_sft_module(olmo_core_dir: Path, model_arch: str) -> ModuleType:
    script_name = "Olmo-3-32B-SFT.py" if model_arch == "olmo3_32b" else "Olmo-3-7B-SFT.py"
    script_path = olmo_core_dir / "src" / "scripts" / "train" / "sft" / script_name
    return import_module_from_path(f"local_{script_name.replace('-', '_').replace('.', '_')}", script_path)


def patch_olmo_sft_module(module: ModuleType, output_path: Path, num_gpus: int, args: argparse.Namespace, model_arch: str) -> None:
    root_dir = output_path / "olmo_core_root"
    work_dir = output_path / "olmo_core_work"
    root_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    module.get_root_dir = lambda cluster: str(root_dir)
    module.get_work_dir = lambda root: str(work_dir)
    module.get_beaker_username = lambda: "local"

    def build_local_launch_config(**kwargs: object) -> object:
        from gantry.api import GitRepoState

        return module.BeakerLaunchConfig(
            name=str(kwargs.get("name") or output_path.name),
            cmd=list(kwargs.get("cmd") or []),
            budget=str(kwargs.get("budget") or "local"),
            workspace=str(kwargs.get("workspace") or "local"),
            clusters=[str(kwargs.get("cluster") or "local")],
            num_nodes=1,
            num_gpus=num_gpus,
            shared_filesystem=True,
            allow_dirty=True,
            env_secrets=[],
            google_credentials_secret=None,
            aws_config_secret=None,
            aws_credentials_secret=None,
            git=GitRepoState(
                repo="local-submission",
                repo_url="https://github.com/nguyen599/math-train",
                ref="local",
                branch=None,
                _is_remote=True,
            ),
        )

    module.build_launch_config = build_local_launch_config
    module.CLUSTER_TO_GPU_TYPE["local"] = "NVIDIA H200"
    module.GPUS_PER_NODE = num_gpus
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)

    def build_local_batch_size_config(**kwargs: object) -> LocalBatchSizeConfig:
        kwargs["tp_degree"] = tp_degree
        kwargs["pp_degree"] = pp_degree
        kwargs["cp_degree"] = cp_degree
        kwargs["requested_rank_microbatch_size_sequences"] = args.rank_microbatch_size_sequences
        if not args.global_batch_size_tokens and not args.global_batch_size_sequences:
            kwargs["min_grad_accum_steps"] = max(args.gradient_accumulation_steps, pp_degree)
        return LocalBatchSizeConfig(**kwargs)

    module.BatchSizeConfig = build_local_batch_size_config


def apply_olmo_attention_backend_override(config: object, args: argparse.Namespace) -> None:
    attn_implementation = normalize_attn_implementation(args.attn_implementation)
    if attn_implementation is None:
        return

    from olmo_core.nn.attention import AttentionBackendName, AttentionConfig

    try:
        backend = AttentionBackendName(attn_implementation)
    except ValueError as exc:
        valid = ", ".join(item.value for item in AttentionBackendName)
        raise ValueError(
            f"Unsupported --attn_implementation={args.attn_implementation!r}. "
            f"Use one of: {valid}. Aliases accepted include sdpa/eager->torch, "
            "flash_attention_2->flash_2, flash_attention_3->flash_3, "
            "flash_attention_4->flash_4, and transformer_engine/te_attn->te."
        ) from exc

    model_config = getattr(config, "model", None)
    if model_config is None:
        raise ValueError("Cannot apply --attn_implementation because the OLMo-core config has no model field.")

    changed = 0

    def update_attention(item: object) -> None:
        nonlocal changed
        if isinstance(item, AttentionConfig):
            item.backend = backend
            item.use_flash = None
            changed += 1

    if hasattr(model_config, "apply"):
        model_config.apply(update_attention)
    else:
        block = getattr(model_config, "block", None)
        blocks = block.values() if isinstance(block, dict) else [block]
        for block_config in blocks:
            update_attention(getattr(block_config, "sequence_mixer", None))
        for block_config in (getattr(model_config, "block_overrides", None) or {}).values():
            update_attention(getattr(block_config, "sequence_mixer", None))

    if changed == 0:
        raise ValueError("Cannot apply --attn_implementation because no OLMo-core AttentionConfig was found.")
    logging.warning("Using OLMo-core attention backend: %s (%d attention config object(s)).", backend.value, changed)


def apply_olmo_attention_sink_override(config: object, args: argparse.Namespace) -> None:
    if args.attention_sink != "true":
        return

    from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType

    attn_implementation = normalize_attn_implementation(args.attn_implementation)
    if attn_implementation is None:
        backend = AttentionBackendName.flash_3
    else:
        try:
            backend = AttentionBackendName(attn_implementation)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported --attn_implementation={args.attn_implementation!r} for "
                "--attention_sink true. Use torch/eager or flash_3."
            ) from exc

    if backend not in (AttentionBackendName.torch, AttentionBackendName.flash_3):
        raise ValueError(
            f"--attention_sink true currently supports only torch/eager and flash_3 backends; got {backend.value}."
        )

    model_config = getattr(config, "model", None)
    if model_config is None:
        raise ValueError("Cannot apply --attention_sink because the OLMo-core config has no model field.")

    changed = 0

    def update_attention(item: object) -> None:
        nonlocal changed
        if not isinstance(item, AttentionConfig):
            return
        if item.name != AttentionType.default:
            raise ValueError(
                f"--attention_sink true requires default OLMo-core attention; found {item.name}."
            )
        item.attention_sink = True
        item.attention_sink_init_value = args.attention_sink_init_value
        item.backend = backend
        item.use_flash = None
        changed += 1

    if hasattr(model_config, "apply"):
        model_config.apply(update_attention)
    else:
        block = getattr(model_config, "block", None)
        blocks = block.values() if isinstance(block, dict) else [block]
        for block_config in blocks:
            update_attention(getattr(block_config, "sequence_mixer", None))
        for block_config in (getattr(model_config, "block_overrides", None) or {}).values():
            update_attention(getattr(block_config, "sequence_mixer", None))

    if changed == 0:
        raise ValueError("Cannot apply --attention_sink because no OLMo-core AttentionConfig was found.")

    logging.warning(
        "Enabled OLMo-core attention sinks for %d attention config object(s): backend=%s init_value=%.6g.",
        changed,
        backend.value,
        args.attention_sink_init_value,
    )


def apply_olmo_lm_loss_override(config: object, args: argparse.Namespace) -> None:
    if args.lm_loss_implementation == "default":
        return

    from olmo_core.nn.lm_head import LMLossImplementation

    model_config = getattr(config, "model", None)
    lm_head_config = getattr(model_config, "lm_head", None)
    if lm_head_config is None:
        raise ValueError("Cannot apply --lm_loss_implementation because no OLMo-core LM head config was found.")

    lm_head_config.loss_implementation = LMLossImplementation(args.lm_loss_implementation)
    logging.warning("Using OLMo-core LM-head loss implementation: %s.", args.lm_loss_implementation)


def build_activation_checkpointing_config(
    module: ModuleType,
    args: argparse.Namespace,
    compile_model: bool,
) -> object | None:
    mode_arg = args.activation_checkpointing_mode
    mode_enum = module.TransformerActivationCheckpointingMode
    config_cls = module.TransformerActivationCheckpointingConfig

    if mode_arg == "none":
        return None

    if mode_arg == "auto":
        if compile_model and args.activation_memory_budget < 1.0:
            return config_cls(
                mode=mode_enum.budget,
                activation_memory_budget=args.activation_memory_budget,
            )
        if args.activation_memory_budget < 1.0:
            logging.warning(
                "Using selected_modules activation checkpointing for uncompiled training because "
                "--activation_memory_budget=%s but torch.compile is disabled. OLMo-core budget "
                "activation checkpointing requires compile_model=true. Use "
                "--activation_checkpointing_mode full to checkpoint every block.",
                args.activation_memory_budget,
            )
            return config_cls(
                mode=mode_enum.selected_modules,
                modules=["blocks.*.feed_forward"],
            )
        return None

    if mode_arg == "budget":
        if not compile_model:
            raise ValueError(
                "--activation_checkpointing_mode budget requires effective compile_model=true. "
                "Use --force_compile_model to bypass train.py compile guards, or use "
                "--activation_checkpointing_mode full for uncompiled TP+PP runs."
            )
        return config_cls(
            mode=mode_enum.budget,
            activation_memory_budget=args.activation_memory_budget,
        )

    if mode_arg == "full":
        return config_cls(mode=mode_enum.full)

    if mode_arg == "selected_blocks":
        return config_cls(
            mode=mode_enum.selected_blocks,
            block_interval=args.activation_checkpoint_block_interval,
        )

    if mode_arg == "selected_modules":
        return config_cls(
            mode=mode_enum.selected_modules,
            modules=parse_csv_tokens(args.activation_checkpoint_modules),
        )

    if mode_arg == "selected_ops":
        return config_cls(mode=mode_enum.selected_ops)

    raise ValueError(f"Unsupported --activation_checkpointing_mode={mode_arg!r}.")


def apply_olmo_rope_scaling_override(
    config: object,
    args: argparse.Namespace,
    model_path: Path,
    model_arch: str,
) -> None:
    factor = effective_rope_scaling_factor(args, model_path, model_arch)
    if factor is None:
        return

    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.rope import YaRNRoPEScalingConfig

    model_config = getattr(config, "model", None)
    if model_config is None:
        raise ValueError("Cannot apply RoPE scaling because the OLMo-core config has no model field.")

    rope_scaling = YaRNRoPEScalingConfig(
        factor=factor,
        beta_fast=args.rope_scaling_beta_fast,
        beta_slow=args.rope_scaling_beta_slow,
        old_context_len=args.rope_scaling_old_context_len,
    )
    changed = 0

    def update_attention(item: object) -> None:
        nonlocal changed
        if not isinstance(item, AttentionConfig) or item.rope is None:
            return
        if item.rope.scaling is None:
            return
        item.rope.scaling = rope_scaling
        changed += 1

    if hasattr(model_config, "apply"):
        model_config.apply(update_attention)
    else:
        block = getattr(model_config, "block", None)
        blocks = block.values() if isinstance(block, dict) else [block]
        for block_config in blocks:
            update_attention(getattr(block_config, "sequence_mixer", None))
        for block_config in (getattr(model_config, "block_overrides", None) or {}).values():
            update_attention(getattr(block_config, "sequence_mixer", None))

    if changed == 0:
        raise ValueError(
            "Cannot apply RoPE scaling because no existing scaled AttentionConfig was found. "
            "The direct OLMo3 SFT configs should call with_rope_scaling() before train.py patches them."
        )
    logging.warning(
        "Using YaRN RoPE scaling factor=%.6g old_context_len=%d beta_fast=%.6g beta_slow=%.6g "
        "for %d full-attention config object(s).",
        factor,
        args.rope_scaling_old_context_len,
        args.rope_scaling_beta_fast,
        args.rope_scaling_beta_slow,
        changed,
    )


def apply_olmo_data_loader_override(config: object, args: argparse.Namespace) -> None:
    if args.data_loader_num_workers < 0:
        raise ValueError("--data_loader_num_workers must be >= 0.")
    if args.data_loader_num_threads is not None and args.data_loader_num_threads < 0:
        raise ValueError("--data_loader_num_threads must be >= 0 or 'none'.")
    if args.data_loader_prefetch_factor is not None and args.data_loader_prefetch_factor <= 0:
        raise ValueError("--data_loader_prefetch_factor must be > 0 or 'none'.")

    data_loader = getattr(config, "data_loader", None)
    if data_loader is None:
        raise ValueError("Cannot configure data loader because the OLMo-core config has no data_loader field.")

    data_loader.num_workers = args.data_loader_num_workers
    data_loader.num_threads = args.data_loader_num_threads
    if args.data_loader_num_workers > 0:
        data_loader.prefetch_factor = args.data_loader_prefetch_factor
    else:
        data_loader.prefetch_factor = None
        if args.data_loader_prefetch_factor is not None:
            logging.warning("Ignoring --data_loader_prefetch_factor because --data_loader_num_workers=0.")

    logging.warning(
        "Using OLMo-core NumPy data loader settings: num_workers=%s prefetch_factor=%s num_threads=%s.",
        data_loader.num_workers,
        data_loader.prefetch_factor,
        data_loader.num_threads,
    )


def pipeline_microbatch_efficiency(num_microbatches: int, pp_degree: int) -> float | None:
    if num_microbatches <= 0 or pp_degree <= 1:
        return None
    return num_microbatches / float(num_microbatches + pp_degree - 1)


def recommended_rank_microbatch_sequences(rank_batch_sequences: int, pp_degree: int, target_efficiency: float = 0.85) -> int | None:
    if rank_batch_sequences <= 0 or pp_degree <= 1:
        return None
    divisors = [value for value in range(1, rank_batch_sequences + 1) if rank_batch_sequences % value == 0]
    for microbatch_sequences in reversed(divisors):
        num_microbatches = rank_batch_sequences // microbatch_sequences
        efficiency = pipeline_microbatch_efficiency(num_microbatches, pp_degree)
        if efficiency is not None and efficiency >= target_efficiency:
            return microbatch_sequences
    return 1


def apply_olmo_microbatch_override(config: object, args: argparse.Namespace, model_arch: str) -> None:
    train_module = getattr(config, "train_module", None)
    data_loader = getattr(config, "data_loader", None)
    if train_module is None or data_loader is None:
        raise ValueError("Cannot configure microbatch because the OLMo-core config is missing train_module/data_loader.")

    sequence_length = int(getattr(train_module, "max_sequence_length", olmo_core_sequence_length(args.max_seq_length)))
    dp_degree = effective_data_parallel_degree(args, model_arch)
    global_batch_size = int(getattr(data_loader, "global_batch_size"))
    rank_batch_tokens = global_batch_size // dp_degree

    if args.rank_microbatch_size_sequences > 0:
        requested_rank_microbatch_tokens = args.rank_microbatch_size_sequences * sequence_length
        if requested_rank_microbatch_tokens <= 0:
            raise ValueError("--rank_microbatch_size_sequences must be positive when set.")
        if rank_batch_tokens % requested_rank_microbatch_tokens != 0:
            raise ValueError(
                "Requested rank microbatch is incompatible with global batch: "
                f"rank_batch_tokens={rank_batch_tokens:,d}, "
                f"rank_microbatch_tokens={requested_rank_microbatch_tokens:,d}. "
                "Adjust --global_batch_size_tokens, --per_device_batch_size, "
                "--gradient_accumulation_steps, or --rank_microbatch_size_sequences."
            )
        train_module.rank_microbatch_size = requested_rank_microbatch_tokens
        logging.warning(
            "Forced OLMo-core rank microbatch size to %d sequence(s), %d tokens.",
            args.rank_microbatch_size_sequences,
            requested_rank_microbatch_tokens,
        )

    rank_microbatch_tokens = int(getattr(train_module, "rank_microbatch_size"))
    rank_microbatch_sequences = rank_microbatch_tokens // sequence_length
    rank_batch_sequences = rank_batch_tokens // sequence_length
    effective_grad_accum = rank_batch_tokens // rank_microbatch_tokens
    pp_degree = effective_pipeline_parallel_degree(args)
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    world_size = effective_num_nodes(args) * effective_num_gpus(args)
    model_parallel_degree = tp_degree * cp_degree * pp_degree
    global_batch_sequences = global_batch_size // sequence_length
    pp_efficiency = pipeline_microbatch_efficiency(effective_grad_accum, pp_degree)
    logging.warning(
        "Effective OLMo-core batch plan: global_batch_size=%d token(s), dp=%d, "
        "rank_batch=%d sequence(s), rank_microbatch=%d sequence(s), "
        "effective_grad_accum=%d, requested_per_device_batch_size=%d, requested_gradient_accum=%d.",
        global_batch_size,
        dp_degree,
        rank_batch_sequences,
        rank_microbatch_sequences,
        effective_grad_accum,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
    )
    logging.warning(
        "Batch topology: world_size=%d TP=%d CP=%d PP=%d DP=%d model_parallel_degree=%d; "
        "global_batch=%d packed sequence(s), rank_microbatch=%d sequence(s), microbatches=%d. "
        "TP/CP shard each microbatch and PP stages it across ranks; only DP multiplies independent data.",
        world_size,
        tp_degree,
        cp_degree,
        pp_degree,
        dp_degree,
        model_parallel_degree,
        global_batch_sequences,
        rank_microbatch_sequences,
        effective_grad_accum,
    )
    requested_microbatch_sequences = (
        args.rank_microbatch_size_sequences
        if args.rank_microbatch_size_sequences > 0
        else args.per_device_batch_size
    )
    requested_global_batch_tokens = (
        requested_microbatch_sequences
        * max(1, args.gradient_accumulation_steps)
        * dp_degree
        * sequence_length
    )
    if global_batch_size != requested_global_batch_tokens:
        logging.warning(
            "Explicit/derived global batch overrides requested MBS*GA*DP: configured=%d token(s) "
            "(%d sequence(s)); requested rank_microbatch=%d * gradient_accumulation=%d * DP=%d "
            "would be %d token(s) (%d sequence(s)). Effective accumulation is %d.",
            global_batch_size,
            global_batch_sequences,
            requested_microbatch_sequences,
            args.gradient_accumulation_steps,
            dp_degree,
            requested_global_batch_tokens,
            requested_global_batch_tokens // sequence_length,
            effective_grad_accum,
        )
    if pp_efficiency is not None:
        logging.warning(
            "Estimated pipeline schedule efficiency: PP=%d microbatches=%d schedule=%s "
            "ideal_efficiency=%.1f%% ideal_bubble=%.1f%%. "
            "This is a schedule lower-bound estimate, not measured GPU utilization.",
            pp_degree,
            effective_grad_accum,
            args.pipeline_schedule,
            pp_efficiency * 100.0,
            (1.0 - pp_efficiency) * 100.0,
        )
        recommended = recommended_rank_microbatch_sequences(rank_batch_sequences, pp_degree)
        if recommended is not None and recommended < rank_microbatch_sequences:
            recommended_microbatches = rank_batch_sequences // recommended
            recommended_efficiency = pipeline_microbatch_efficiency(recommended_microbatches, pp_degree) or 0.0
            logging.warning(
                "Low PP microbatch count is likely to cap throughput. For the same global token batch, try "
                "--rank_microbatch_size_sequences %d to use %d microbatches and raise ideal pipeline "
                "efficiency to %.1f%%. This may change per-step VRAM and kernel efficiency.",
                recommended,
                recommended_microbatches,
                recommended_efficiency * 100.0,
            )
    if rank_microbatch_sequences != args.per_device_batch_size and args.rank_microbatch_size_sequences <= 0:
        logging.warning(
            "--per_device_batch_size=%d is not the effective forward microbatch for this OLMo-core direct run; "
            "the SFT script's automatic planner selected rank_microbatch_size_sequences=%d. "
            "That is why VRAM can stay flat when changing --per_device_batch_size. "
            "Use --rank_microbatch_size_sequences %d to force VRAM to scale with that value.",
            args.per_device_batch_size,
            rank_microbatch_sequences,
            args.per_device_batch_size,
        )


def build_olmo_lr_scheduler(args: argparse.Namespace) -> object:
    from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, LinearWithWarmup

    if args.lr_scheduler_type == "linear":
        return LinearWithWarmup(warmup_fraction=args.warmup_ratio, alpha_f=0.0)
    if args.lr_scheduler_type == "cosine":
        return CosWithWarmup(warmup_fraction=args.warmup_ratio, alpha_f=0.0)
    if args.lr_scheduler_type == "constant":
        return ConstantWithWarmup(warmup_fraction=args.warmup_ratio)
    raise ValueError(f"Unsupported --lr_scheduler_type: {args.lr_scheduler_type!r}")


def effective_torch_profiler_ranks(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"", "none", "null", "rank0", "0"}:
        return None
    return normalized


def install_torch_profiler_callback(config: object, args: argparse.Namespace) -> None:
    if args.torch_profiler != "true":
        return
    try:
        from olmo_core.train.callbacks import ProfilerCallback
    except ImportError:
        logging.exception("Could not import OLMo-core ProfilerCallback.")
        raise

    callback = ProfilerCallback(
        skip_first=args.torch_profiler_skip_first,
        wait=args.torch_profiler_wait,
        warmup=args.torch_profiler_warmup,
        active=args.torch_profiler_active,
        repeat=args.torch_profiler_repeat,
        with_stack=args.torch_profiler_with_stack == "true",
        profile_memory=args.torch_profiler_profile_memory == "true",
        enable_cuda_sync_events=args.torch_profiler_cuda_sync_events == "true",
        ranks=effective_torch_profiler_ranks(args.torch_profiler_ranks),
        enabled=True,
    )
    config.trainer.callbacks["profiler"] = callback
    logging.warning(
        "Enabled OLMo-core torch.profiler callback: skip_first=%d wait=%d warmup=%d active=%d "
        "repeat=%d ranks=%s with_stack=%s profile_memory=%s cuda_sync_events=%s. "
        "Chrome traces will be written under the trainer work_dir/profiler and persisted with the run.",
        callback.skip_first,
        callback.wait,
        callback.warmup,
        callback.active,
        callback.repeat,
        callback.ranks if callback.ranks is not None else "rank0",
        callback.with_stack,
        callback.profile_memory,
        callback.enable_cuda_sync_events,
    )


def configure_olmo_sft_config(module: ModuleType, config: object, args: argparse.Namespace, output_path: Path) -> None:
    config.trainer.save_folder = str(output_path)
    if args.max_train_steps > 0:
        config.trainer.max_duration = module.Duration.steps(args.max_train_steps)
    else:
        config.trainer.max_duration = module.Duration.epochs(args.num_train_epochs)
    config.trainer.metrics_collect_interval = args.logging_steps
    install_torch_profiler_callback(config, args)
    apply_olmo_data_loader_override(config, args)
    model_arch = normalize_model_arch(args.model_arch, Path(args.model_path).expanduser().resolve())
    apply_olmo_microbatch_override(config, args, model_arch)
    apply_olmo_attention_backend_override(config, args)
    apply_olmo_attention_sink_override(config, args)
    apply_olmo_lm_loss_override(config, args)
    apply_olmo_rope_scaling_override(config, args, Path(args.model_path).expanduser().resolve(), model_arch)
    tp_degree = effective_tensor_parallel_degree(args, model_arch)
    cp_degree = effective_context_parallel_degree(args)
    pp_degree = effective_pipeline_parallel_degree(args)
    dp_degree = effective_data_parallel_degree(args, model_arch)
    compile_model = args.compile_model == "true"
    force_compile_model = args.force_compile_model == "true"
    if compile_model and args.float8 == "true" and tp_degree > 1:
        if force_compile_model:
            logging.warning(
                "Forcing torch.compile for Float8Linear with tensor parallelism (TP=%d). "
                "This path has failed previously in torchao/Inductor DTensor shard_dim_alltoall.",
                tp_degree,
            )
        else:
            logging.warning(
                "Disabling torch.compile for Float8Linear with tensor parallelism (TP=%d). "
                "The current torchao/Inductor path fails on DTensor shard_dim_alltoall; "
                "FP8 remains enabled for the uncompiled path. Use --force_compile_model to override.",
                tp_degree,
            )
            compile_model = False
    if compile_model and tp_degree > 1 and pp_degree > 1:
        if force_compile_model:
            logging.warning(
                "Forcing torch.compile for combined TP+PP (TP=%d, PP=%d). "
                "This path has failed previously in Dynamo/DTensor during pipeline dry-run.",
                tp_degree,
                pp_degree,
            )
        else:
            logging.warning(
                "Disabling torch.compile for combined TP+PP (TP=%d, PP=%d). "
                "This path currently fails in Dynamo/DTensor during pipeline dry-run; "
                "TP-only compile and plain PP compile remain available. Use --force_compile_model to override.",
                tp_degree,
                pp_degree,
            )
            compile_model = False
    if args.tensor_parallel_async == "true" and not compile_model:
        raise ValueError(
            "--tensor_parallel_async true requires effective torch.compile. "
            "For combined TP+PP, also set --force_compile_model true."
        )
    config.train_module.compile_model = compile_model
    config.train_module.ac_config = build_activation_checkpointing_config(module, args, compile_model)
    config.train_module.optim.lr = args.learning_rate
    config.train_module.optim.weight_decay = args.weight_decay
    if model_arch == "olmo3_32b" or tp_degree > 1 or cp_degree > 1 or pp_degree > 1:
        from olmo_core.train.train_module.transformer.config import (
            TransformerContextParallelConfig,
            TransformerPipelineParallelConfig,
            TransformerTensorParallelConfig,
        )
        from olmo_core.distributed.parallel.pipeline_parallel import PipelineScheduleType

        model_n_layers = getattr(getattr(config, "model", None), "n_layers", None)
        model_n_layers = model_n_layers or DEFAULT_MODEL_N_LAYERS.get(model_arch)
        pipeline_split_points = effective_pipeline_split_points(args, model_arch, model_n_layers)
        validate_pipeline_split_points(args, pp_degree, model_n_layers, model_arch)
        config.train_module.tp_config = (
            TransformerTensorParallelConfig(
                degree=tp_degree,
                enable_async=args.tensor_parallel_async == "true",
            )
            if tp_degree > 1
            else None
        )
        config.train_module.cp_config = build_transformer_context_parallel_config(
            TransformerContextParallelConfig,
            args,
            cp_degree,
        )
        config.train_module.pp_config = (
            TransformerPipelineParallelConfig(
                degree=pp_degree,
                schedule=PipelineScheduleType(args.pipeline_schedule),
                split_points=pipeline_split_points,
            )
            if pp_degree > 1
            else None
        )
        if getattr(config.train_module, "dp_config", None) is not None:
            current_shard_degree = getattr(config.train_module.dp_config, "shard_degree", None)
            config.train_module.dp_config.shard_degree = min(current_shard_degree or dp_degree, dp_degree)
            current_num_replicas = getattr(config.train_module.dp_config, "num_replicas", None)
            if current_num_replicas is not None and current_num_replicas > dp_degree:
                config.train_module.dp_config.num_replicas = dp_degree
        logging.warning(
            "Using OLMo-core parallel degrees: DP=%d TP=%d CP=%d PP=%d async_tp=%s.",
            dp_degree,
            tp_degree,
            cp_degree,
            pp_degree,
            args.tensor_parallel_async,
        )
        if cp_degree > 1:
            cp_style = effective_context_parallel_style(args)
            if cp_style == "ulysses":
                logging.warning("Using OLMo-core context parallel style: ulysses.")
            else:
                logging.warning(
                    "Using OLMo-core context parallel style: %s head_stride=%d.",
                    cp_style,
                    effective_context_parallel_head_stride(args),
                )
        if pp_degree > 1:
            logging.warning("Using OLMo-core pipeline schedule: %s.", args.pipeline_schedule)
            if pipeline_split_points:
                split_source = "explicit" if parse_pipeline_split_points(args.pipeline_split_points) else "auto"
                logging.warning(
                    "Using %s OLMo-core pipeline split points: %s.",
                    split_source,
                    pipeline_split_points,
                )
    optimizer = normalized_optimizer_name(args)
    optim_dtype = effective_optimizer_state_dtype(args)
    if optimizer == "adamw":
        from olmo_core.optim import AdamWConfig

        current_optim = config.train_module.optim
        config.train_module.optim = AdamWConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=getattr(current_optim, "betas", (0.9, 0.95)),
            eps=getattr(current_optim, "eps", 1e-8),
            foreach=False,
        )
        if optim_dtype == "bfloat16":
            logging.warning("Plain AdamW does not expose OLMo-core optimizer-state dtype control.")
        logging.warning("Using plain AdamW optimizer for this OLMo-core direct run.")
    elif optimizer == "skip_step_adamw":
        from olmo_core.optim import SkipStepAdamWConfig

        current_optim = config.train_module.optim
        config.train_module.optim = SkipStepAdamWConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=getattr(current_optim, "betas", (0.9, 0.95)),
            eps=getattr(current_optim, "eps", 1e-8),
            dtype=module.DType.bfloat16 if optim_dtype == "bfloat16" else getattr(current_optim, "dtype", None),
            foreach=getattr(current_optim, "foreach", True),
            step_increment_bugfix=getattr(current_optim, "step_increment_bugfix", True),
            rolling_interval_length=getattr(current_optim, "rolling_interval_length", 128),
            sigma_factor=getattr(current_optim, "sigma_factor", 6),
            compile=getattr(current_optim, "compile", False),
        )
        if optim_dtype == "bfloat16":
            logging.warning("Using bfloat16 SkipStepAdamW optimizer state for this OLMo-core direct run.")
        logging.warning("Using SkipStepAdamW optimizer for this OLMo-core direct run.")
    elif optimizer == "adamw_8bit":
        from olmo_torchao_optim import TorchAOAdamW8bitConfig

        current_optim = config.train_module.optim
        config.train_module.optim = TorchAOAdamW8bitConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=getattr(current_optim, "betas", (0.9, 0.95)),
            eps=getattr(current_optim, "eps", 1e-8),
            block_size=args.adamw_8bit_block_size,
            bf16_stochastic_round=args.adamw_8bit_bf16_stochastic_round,
        )
        if optim_dtype is not None:
            logging.warning("torchao AdamW8bit ignores --optimizer_state_dtype=%s.", args.optimizer_state_dtype)
        logging.warning(
            "Using torchao AdamW8bit optimizer with block_size=%d, bf16_stochastic_round=%s.",
            args.adamw_8bit_block_size,
            args.adamw_8bit_bf16_stochastic_round,
        )
    elif optimizer == "te_fused_adamw":
        from olmo_torchao_optim import TransformerEngineFusedAdamWConfig

        current_optim = config.train_module.optim
        te_optim_dtype = (
            "bfloat16"
            if optim_dtype == "bfloat16" or args.optimizer_state_dtype == "auto"
            else "float32"
        )
        config.train_module.optim = TransformerEngineFusedAdamWConfig(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=getattr(current_optim, "betas", (0.9, 0.95)),
            eps=getattr(current_optim, "eps", 1e-8),
            exp_avg_dtype=te_optim_dtype,
            exp_avg_sq_dtype=te_optim_dtype,
        )
        logging.warning(
            "Using Transformer Engine FusedAdamW optimizer with exp_avg_dtype=%s "
            "and exp_avg_sq_dtype=%s.",
            te_optim_dtype,
            te_optim_dtype,
        )
    elif optim_dtype == "bfloat16" and hasattr(config.train_module.optim, "dtype"):
        config.train_module.optim.dtype = module.DType.bfloat16
        logging.warning("Using bfloat16 optimizer state for this OLMo-core direct run.")
    if args.float8 == "true":
        from olmo_core.float8 import (
            AOFloat8LinearConfig,
            AOFloat8LinearRecipe,
            AOMXLinearConfig,
            Float8Config,
        )

        if args.float8_recipe == "recommended":
            config.train_module.float8_config = Float8Config(
                enabled=True,
                ao=AOFloat8LinearConfig.recommended(),
            )
        elif args.float8_recipe == "mxfp8_cublas_rceil":
            config.train_module.float8_config = Float8Config(
                enabled=True,
                ao_mx=AOMXLinearConfig.mxfp8_cublas_rceil(),
            )
        elif args.float8_recipe in {"blockwise", "blockwise_triton"}:
            config.train_module.float8_config = Float8Config(
                enabled=True,
                ao_blockwise=True,
                ao_blockwise_use_triton=args.float8_recipe == "blockwise_triton",
            )
        else:
            config.train_module.float8_config = Float8Config(
                enabled=True,
                ao_recipe=AOFloat8LinearRecipe(args.float8_recipe),
            )
        logging.warning("Enabled OLMo-core Float8Linear training with recipe=%s.", args.float8_recipe)
    config.train_module.te_feed_forward = args.te_feedforward == "true"
    if config.train_module.te_feed_forward:
        logging.warning(
            "Enabled OLMo-core Transformer Engine feed-forward Linear modules. "
            "This mode currently supports TP=1 only."
        )
    config.train_module.te_feed_forward_glu = args.te_feedforward_glu == "true"
    if config.train_module.te_feed_forward_glu:
        logging.warning(
            "Enabled OLMo-core Transformer Engine fused feed-forward GLU activation. "
            "This mode keeps OLMo linears/checkpoints and is intended for TP>1."
        )
    config.train_module.te_layer_norm = args.te_layernorm == "true"
    if config.train_module.te_layer_norm:
        logging.warning(
            "Enabled OLMo-core Transformer Engine RMSNorm kernels. "
            "This mode keeps OLMo norm modules/checkpoints."
        )
    config.train_module.liger_layer_norm = args.liger_layernorm == "true"
    if config.train_module.liger_layer_norm:
        logging.warning(
            "Enabled OLMo-core Liger RMSNorm kernels. "
            "This mode keeps OLMo norm modules/checkpoints."
        )
    config.train_module.liger_megatron_layer_norm = args.liger_megatron_layernorm == "true"
    if config.train_module.liger_megatron_layer_norm:
        logging.warning(
            "Enabled OLMo-core Liger Megatron RMSNorm kernels. "
            "This mode keeps OLMo norm modules/checkpoints."
        )
    config.train_module.quack_layer_norm = args.quack_layernorm == "true"
    if config.train_module.quack_layer_norm:
        logging.warning(
            "Enabled OLMo-core Quack CuTe RMSNorm kernels. "
            "This mode keeps OLMo norm modules/checkpoints."
        )
    config.train_module.feed_forward_chunk_size_tokens = args.feed_forward_chunk_size_tokens
    if config.train_module.feed_forward_chunk_size_tokens > 0:
        logging.warning(
            "Enabled OLMo-core feed-forward token chunking with chunk_size_tokens=%d.",
            config.train_module.feed_forward_chunk_size_tokens,
        )
    config.train_module.scheduler = build_olmo_lr_scheduler(args)
    config.train_module.max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None
    logging.warning(
        "Final OLMo-core direct config: compile_model=%s force_compile_model=%s "
        "activation_memory_budget=%s activation_checkpointing_mode=%s ac_config=%r "
        "optimizer=%s lr=%s weight_decay=%s scheduler=%s warmup_ratio=%s "
        "float8_config=%r te_feed_forward=%s te_feed_forward_glu=%s "
        "te_layer_norm=%s liger_layer_norm=%s liger_megatron_layer_norm=%s quack_layer_norm=%s "
        "feed_forward_chunk_size_tokens=%s.",
        config.train_module.compile_model,
        force_compile_model,
        args.activation_memory_budget,
        args.activation_checkpointing_mode,
        getattr(config.train_module, "ac_config", None),
        type(config.train_module.optim).__name__,
        getattr(config.train_module.optim, "lr", None),
        getattr(config.train_module.optim, "weight_decay", None),
        type(config.train_module.scheduler).__name__,
        args.warmup_ratio,
        getattr(config.train_module, "float8_config", None),
        getattr(config.train_module, "te_feed_forward", False),
        getattr(config.train_module, "te_feed_forward_glu", False),
        getattr(config.train_module, "te_layer_norm", False),
        getattr(config.train_module, "liger_layer_norm", False),
        getattr(config.train_module, "liger_megatron_layer_norm", False),
        getattr(config.train_module, "quack_layer_norm", False),
        getattr(config.train_module, "feed_forward_chunk_size_tokens", 0),
    )
    checkpointer = config.trainer.callbacks.get("checkpointer")
    if checkpointer is not None:
        if args.disable_checkpoints:
            checkpointer.enabled = False
            logging.warning("Disabled OLMo-core direct checkpointer callback.")
        else:
            checkpointing_steps, ephemeral_interval = effective_checkpoint_intervals(args)
            checkpointer.save_interval = checkpointing_steps
            checkpointer.ephemeral_save_interval = ephemeral_interval
            install_checkpoint_upload_and_retention_callback(config, args, output_path)
    wandb = config.trainer.callbacks.get("wandb")
    if wandb is not None:
        rank_metadata = wandb_rank_metadata(args)
        wandb.enabled = bool(args.with_tracking) and primary_wandb_log_process(args)
        wandb.project = args.wandb_project
        wandb.entity = args.wandb_entity or None
        wandb.group = output_path.name
        wandb.name = truncate_slug(f"{output_path.name}-rank{wandb_rank_suffix(args)}", max_length=120)
        tags = list(getattr(wandb, "tags", None) or [])
        tags.extend(
            [
                f"rank:{rank_metadata['rank']}",
                f"local_rank:{rank_metadata['local_rank']}",
                f"global_rank:{rank_metadata['global_rank']}",
                f"node_rank:{rank_metadata['node_rank']}",
            ]
        )
        wandb.tags = list(dict.fromkeys(tags))
        logging.warning(
            "W&B callback enabled=%s for rank metadata %s; only primary rank logs metrics by default.",
            wandb.enabled,
            rank_metadata,
        )


def run_olmo_core_sft_worker(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    logdir = Path(args.logdir).expanduser().resolve()
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, logdir, "olmo_core_worker", launcher_environment_summary(args))
    olmo_core_dir = find_olmo_core_dir(os.environ.get("OLMO_CORE_DIR"))
    model_arch = normalize_model_arch(args.model_arch, model_path)
    validate_parallel_topology(args, model_arch)
    validate_training_features(args, model_arch)
    apply_feed_forward_debug_env(os.environ, args)
    memory_history = CudaMemoryHistoryRecorder(args, logdir)
    module = load_olmo_sft_module(olmo_core_dir, model_arch)
    num_gpus = effective_num_gpus(args)
    patch_olmo_sft_module(module, output_path, num_gpus, args, model_arch)

    checkpoint_cache = Path(args.olmo_core_checkpoint_cache).expanduser().resolve() if args.olmo_core_checkpoint_cache else default_olmo_checkpoint_cache(args)
    checkpoint_to_load = olmo_core_checkpoint_to_load(checkpoint_cache if checkpoint_is_olmo_core(checkpoint_cache) else model_path)
    global_batch_size = resolved_global_batch_size_tokens(args, model_arch)

    try:
        memory_history.start()
        module.prepare_training_environment()
        config = module.SFTConfig.build(
            script=str(Path(__file__).resolve()),
            cmd="train",
            run_name=output_path.name,
            checkpoint=str(checkpoint_to_load),
            cluster="local",
            seq_len=olmo_core_sequence_length(args.max_seq_length),
            num_nodes=effective_num_nodes(args),
            global_batch_size=global_batch_size,
            overrides=[],
            budget="local",
            workspace="local",
            dataset_path=str(dataset_path),
        )
        apply_olmo_core_checkpoint_experiment_config(module, config, checkpoint_to_load)
        configure_olmo_sft_config(module, config, args, output_path)
        module.train(str(checkpoint_to_load), config, no_save_tokenizer=False)
    except Exception as exc:
        if memory_history.should_dump_for_exception(exc):
            memory_history.dump("exception")
        raise
    else:
        memory_history.dump_success_if_requested()
    finally:
        memory_history.stop()
        module.teardown_training_environment()


def run_olmo_core_sft(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    logdir = Path(args.logdir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir(args, "sft_cache")
    checkpoint_cache = Path(args.olmo_core_checkpoint_cache).expanduser().resolve() if args.olmo_core_checkpoint_cache else default_olmo_checkpoint_cache(args)
    dataset_cache = Path(args.olmo_core_dataset_cache).expanduser().resolve() if args.olmo_core_dataset_cache else default_olmo_dataset_cache(args)

    configure_logging(logdir, args)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache_environment(os.environ, cache_dir)
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, logdir, "olmo_core_parent", launcher_environment_summary(args))
    model_path = ensure_model_weights_once(args, model_path, cache_dir)
    args.model_path = str(model_path)
    log_sequence_length_adjustment(args.max_seq_length)
    log_checkpoint_interval_adjustment(args)
    log_launch_summary(args)
    validate_model_path(model_path)

    open_instruct_dir = find_open_instruct_dir(args.open_instruct_dir or os.environ.get("OPEN_INSTRUCT_DIR"))
    olmo_core_dir = find_olmo_core_dir(os.environ.get("OLMO_CORE_DIR"))
    env = build_training_env(open_instruct_dir, olmo_core_dir, cache_dir, offline=False, args=args)
    apply_optimizer_name_env(env, args)
    apply_optimizer_state_dtype_env(env, args)
    apply_feed_forward_debug_env(env, args)
    model_arch = normalize_model_arch(args.model_arch, model_path)
    validate_parallel_topology(args, model_arch)
    validate_training_features(args, model_arch)

    if dataset_is_olmo_core_numpy(dataset_path):
        converted_dataset = dataset_path
    else:
        if args.skip_cache_prepare:
            raise ValueError(
                "--skip_cache_prepare with --backend olmo_core_sft requires a pre-tokenized OLMo-core dataset."
            )
        marker_dir = cache_dir / "multinode_prepare_markers"
        dataset_ref = run_once_on_node0(
            args,
            marker_dir,
            raw_dataset_marker_name(args, dataset_path),
            lambda: prepare_raw_dataset(dataset_path, cache_dir),
        )
        converted_dataset = Path(
            run_once_on_node0(
                args,
                marker_dir,
                olmo_core_dataset_marker_name(args, dataset_ref, dataset_cache, open_instruct_dir),
                lambda: ensure_olmo_core_dataset(args, dataset_ref, dataset_cache, open_instruct_dir, env),
            )
        )
    tokenizer_preview_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else model_path
    preview_raw_path = dataset_path if dataset_is_olmo_core_numpy(dataset_path) else Path(dataset_ref)
    maybe_log_tokenized_sample(
        args,
        preview_raw_path,
        tokenizer_preview_path,
        open_instruct_dir,
        converted_dataset_path=converted_dataset,
    )
    converted_checkpoint = Path(
        run_once_on_node0(
            args,
            cache_dir / "multinode_prepare_markers",
            olmo_core_checkpoint_marker_name(args, model_path, checkpoint_cache, olmo_core_dir, model_arch),
            lambda: ensure_olmo_core_checkpoint(args, model_path, checkpoint_cache, olmo_core_dir, model_arch, env),
        )
    )

    worker_attn_implementation = normalize_attn_implementation(args.attn_implementation)
    worker_rope_scaling_factor = effective_rope_scaling_factor(args, model_path, model_arch)
    worker_pipeline_split_points = effective_pipeline_split_points(args, model_arch)
    worker_pipeline_split_points_arg = (
        ",".join(str(point) for point in worker_pipeline_split_points)
        if worker_pipeline_split_points
        else ""
    )
    worker_command = [
        *build_torchrun_prefix(args),
        str(Path(__file__).resolve()),
        "--internal_backend",
        "olmo_core_sft_worker",
        "--backend",
        "olmo_core_sft",
        "--model_path",
        str(converted_checkpoint),
        "--dataset_path",
        str(converted_dataset),
        "--output_path",
        str(output_path),
        "--logdir",
        str(logdir),
        "--num_gpus",
        str(effective_num_gpus(args)),
        "--num_nodes",
        str(effective_num_nodes(args)),
        "--world_size_mode",
        "nodes",
        "--master_port",
        str(effective_master_port(args)),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_batch_size",
        str(args.per_device_batch_size),
        "--rank_microbatch_size_sequences",
        str(args.rank_microbatch_size_sequences),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_seq_length",
        str(olmo_core_sequence_length(args.max_seq_length)),
        "--max_train_steps",
        str(args.max_train_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--weight_decay",
        str(args.weight_decay),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--activation_memory_budget",
        str(args.activation_memory_budget),
        "--compile_model",
        args.compile_model,
        "--force_compile_model",
        args.force_compile_model,
        "--activation_checkpointing_mode",
        args.activation_checkpointing_mode,
        "--activation_checkpoint_modules",
        args.activation_checkpoint_modules,
        "--activation_checkpoint_block_interval",
        str(args.activation_checkpoint_block_interval),
        "--optimizer_state_dtype",
        args.optimizer_state_dtype,
        "--optimizer",
        normalized_optimizer_name(args),
        "--lm_loss_implementation",
        args.lm_loss_implementation,
        "--attention_sink",
        args.attention_sink,
        "--attention_sink_init_value",
        str(args.attention_sink_init_value),
        "--data_loader_num_workers",
        str(args.data_loader_num_workers),
        "--data_loader_prefetch_factor",
        str(args.data_loader_prefetch_factor) if args.data_loader_prefetch_factor is not None else "none",
        "--adamw_8bit_block_size",
        str(args.adamw_8bit_block_size),
        "--float8",
        args.float8,
        "--float8_recipe",
        args.float8_recipe,
        "--te_feedforward",
        args.te_feedforward,
        "--te_feedforward_glu",
        args.te_feedforward_glu,
        "--te_layernorm",
        args.te_layernorm,
        "--liger_layernorm",
        args.liger_layernorm,
        "--liger_megatron_layernorm",
        args.liger_megatron_layernorm,
        "--quack_layernorm",
        args.quack_layernorm,
        "--feed_forward_chunk_size_tokens",
        str(args.feed_forward_chunk_size_tokens),
        "--feed_forward_memory_profile",
        args.feed_forward_memory_profile,
        "--feed_forward_memory_profile_ranks",
        args.feed_forward_memory_profile_ranks,
        "--feed_forward_memory_profile_max_calls",
        str(args.feed_forward_memory_profile_max_calls),
        "--feed_forward_memory_profile_sync",
        args.feed_forward_memory_profile_sync,
        "--feed_forward_memory_profile_allow_compile",
        args.feed_forward_memory_profile_allow_compile,
        "--torch_profiler",
        args.torch_profiler,
        "--torch_profiler_skip_first",
        str(args.torch_profiler_skip_first),
        "--torch_profiler_wait",
        str(args.torch_profiler_wait),
        "--torch_profiler_warmup",
        str(args.torch_profiler_warmup),
        "--torch_profiler_active",
        str(args.torch_profiler_active),
        "--torch_profiler_repeat",
        str(args.torch_profiler_repeat),
        "--torch_profiler_ranks",
        args.torch_profiler_ranks,
        "--torch_profiler_with_stack",
        args.torch_profiler_with_stack,
        "--torch_profiler_profile_memory",
        args.torch_profiler_profile_memory,
        "--torch_profiler_cuda_sync_events",
        args.torch_profiler_cuda_sync_events,
        "--cuda_memory_history",
        args.cuda_memory_history,
        "--cuda_memory_history_max_entries",
        str(args.cuda_memory_history_max_entries),
        "--cuda_memory_history_top_allocations",
        str(args.cuda_memory_history_top_allocations),
        "--cuda_memory_history_dump_pickle",
        args.cuda_memory_history_dump_pickle,
        *(["--adamw_8bit_bf16_stochastic_round"] if args.adamw_8bit_bf16_stochastic_round else []),
        *(["--allow_unsafe_float8_tp"] if args.allow_unsafe_float8_tp else []),
        *(["--disable_checkpoints"] if args.disable_checkpoints else []),
        "--checkpointing_steps",
        str(effective_checkpoint_intervals(args)[0]),
        "--ephemeral_save_interval",
        str(effective_checkpoint_intervals(args)[1] or 0),
        "--checkpoint_keep_last",
        str(args.checkpoint_keep_last),
        "--hf_checkpoint_upload",
        args.hf_checkpoint_upload,
        "--hf_checkpoint_repo",
        args.hf_checkpoint_repo,
        "--hf_checkpoint_path_prefix",
        args.hf_checkpoint_path_prefix,
        "--hf_checkpoint_keep_last",
        str(args.hf_checkpoint_keep_last),
        "--hf_checkpoint_upload_workers",
        str(args.hf_checkpoint_upload_workers),
        "--hf_checkpoint_upload_report_interval_seconds",
        str(args.hf_checkpoint_upload_report_interval_seconds),
        "--hf_checkpoint_disable_file",
        str(args.hf_checkpoint_disable_file),
        "--hf_checkpoint_convert",
        args.hf_checkpoint_convert,
        "--hf_checkpoint_convert_tokenizer",
        args.hf_checkpoint_convert_tokenizer,
        "--hf_checkpoint_convert_device",
        args.hf_checkpoint_convert_device,
        f"--hf_checkpoint_converted_suffix={args.hf_checkpoint_converted_suffix}",
        "--hf_checkpoint_convert_keep_local",
        args.hf_checkpoint_convert_keep_local,
        "--logging_steps",
        str(args.logging_steps),
        "--model_arch",
        model_arch,
        "--tensor_parallel_degree",
        str(effective_tensor_parallel_degree(args, model_arch)),
        "--tensor_parallel_async",
        args.tensor_parallel_async,
        "--context_parallel_degree",
        str(effective_context_parallel_degree(args)),
        "--context_parallel_style",
        args.context_parallel_style,
        "--context_parallel_head_stride",
        str(effective_context_parallel_head_stride(args)),
        "--pipeline_parallel_degree",
        str(effective_pipeline_parallel_degree(args)),
        "--pipeline_schedule",
        args.pipeline_schedule,
        *(["--pipeline_split_points", worker_pipeline_split_points_arg] if worker_pipeline_split_points_arg else []),
        *(["--attn_implementation", worker_attn_implementation] if worker_attn_implementation else []),
        "--rope_scaling_factor",
        str(worker_rope_scaling_factor) if worker_rope_scaling_factor is not None else "none",
        "--rope_scaling_old_context_len",
        str(args.rope_scaling_old_context_len),
        "--rope_scaling_beta_fast",
        str(args.rope_scaling_beta_fast),
        "--rope_scaling_beta_slow",
        str(args.rope_scaling_beta_slow),
        "--olmo_core_checkpoint_cache",
        str(checkpoint_cache),
        "--olmo_core_dataset_cache",
        str(converted_dataset),
        "--global_batch_size_tokens",
        str(resolved_global_batch_size_tokens(args, model_arch)),
        "--offline",
        args.offline,
        "--dataset_num_proc",
        str(args.dataset_num_proc),
        "--dataset_map_batch_size",
        str(args.dataset_map_batch_size),
        "--dataset_batched_tokenization",
        args.dataset_batched_tokenization,
        "--dataset_transform_profile",
        args.dataset_transform_profile,
        "--chat_template_model",
        args.chat_template_model,
        "--dataset_backend",
        args.dataset_backend,
        "--collect_env_info",
        args.collect_env_info,
        "--env_info_command_timeout",
        str(args.env_info_command_timeout),
    ]
    if args.chat_template_name:
        worker_command.extend(["--chat_template_name", args.chat_template_name])
    if args.node_rank is not None:
        worker_command.extend(["--node_rank", str(args.node_rank)])
    elif effective_node_rank(args) is not None:
        worker_command.extend(["--node_rank", str(effective_node_rank(args))])
    if args.data_loader_num_threads is not None:
        worker_command.extend(["--data_loader_num_threads", str(args.data_loader_num_threads)])
    worker_master_addr = effective_master_addr(args)
    if worker_master_addr:
        worker_command.extend(["--master_addr", worker_master_addr])
    if args.with_tracking:
        worker_command.extend(["--with_tracking", "--wandb_project", args.wandb_project])
        if args.wandb_entity:
            worker_command.extend(["--wandb_entity", args.wandb_entity])
    run_command(worker_command, env)


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    logdir = Path(args.logdir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else default_cache_dir(args, "open_instruct_cache")

    configure_logging(logdir, args)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_cache_environment(os.environ, cache_dir)
    log_train_engine_runtime_info(args)
    collect_run_environment_info(args, logdir, "open_instruct_parent", launcher_environment_summary(args))
    model_path = ensure_model_weights_once(args, model_path, cache_dir)
    args.model_path = str(model_path)
    log_sequence_length_adjustment(args.max_seq_length)
    log_checkpoint_interval_adjustment(args)
    log_launch_summary(args)
    validate_open_instruct_optimizer(args)
    validate_model_path(model_path)

    open_instruct_dir = find_open_instruct_dir(args.open_instruct_dir or os.environ.get("OPEN_INSTRUCT_DIR"))
    finetune_script = open_instruct_dir / "open_instruct" / "olmo_core_finetune.py"
    marker_dir = cache_dir / "multinode_prepare_markers"
    dataset_ref = run_once_on_node0(
        args,
        marker_dir,
        raw_dataset_marker_name(args, dataset_path),
        lambda: prepare_raw_dataset(dataset_path, cache_dir),
    )
    env = build_env(open_instruct_dir, cache_dir, offline=False, args=args)
    apply_optimizer_name_env(env, args)
    apply_optimizer_state_dtype_env(env, args)
    apply_feed_forward_debug_env(env, args)
    apply_checkpoint_env(env, args)
    tokenizer_preview_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else model_path
    maybe_log_tokenized_sample(args, Path(dataset_ref), tokenizer_preview_path, open_instruct_dir)
    finetune_args = build_finetune_args(args, dataset_ref, cache_dir, output_path)

    if not args.skip_cache_prepare:
        def cache_dataset() -> str:
            cache_command = [sys.executable, str(finetune_script), *finetune_args, "--cache_dataset_only"]
            run_command(cache_command, env)
            return "cache_dataset_only"

        run_once_on_node0(args, marker_dir, "open_instruct_cache_dataset", cache_dataset)

    train_command = [*build_torchrun_prefix(args), str(finetune_script), *finetune_args]
    run_command(train_command, env)


def main() -> int:
    args = parse_args()
    exit_code = 0
    hf_log_uploader: PeriodicHFLogUploader | None = None
    try:
        ensure_output_and_logdir(args)
        guard_top_level_launcher(args)
        if args.operator_mode == "true":
            run_operator_mode(args)
        else:
            hf_log_uploader = PeriodicHFLogUploader(args)
            hf_log_uploader.start()
            if should_run_sweep(args):
                exit_code = run_sweep(args)
            else:
                if args.backend in GRPO_BACKENDS:
                    run_grpo_fast(args)
                elif args.backend in VERL_BACKENDS:
                    run_verl_rlcsd(args)
                elif args.dry_run_launch:
                    run_dry_launch(args)
                elif args.internal_backend == "olmo_core_sft_worker":
                    configure_logging(Path(args.logdir).expanduser().resolve(), args)
                    log_launch_summary(args)
                    run_olmo_core_sft_worker(args)
                elif args.backend == "olmo_core_sft":
                    run_olmo_core_sft(args)
                else:
                    train(args)
    except Exception as exc:
        exit_code = 1
        logdir = Path(getattr(args, "logdir", None) or ".").expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        if not logging.getLogger().handlers:
            configure_logging(logdir, args)
        logging.exception("Training failed: %s", exc)
    finally:
        if hf_log_uploader is not None:
            hf_log_uploader.stop()
        if logging.getLogger().handlers:
            logging.info("Training exit status: %s", exit_code)
        try:
            status = "success" if exit_code == 0 else "failed"
            if hf_log_uploader is not None:
                hf_log_uploader.upload(status=status, exit_code=exit_code, upload_kind="final")
            else:
                upload_logdir_to_hf(args, status=status, exit_code=exit_code, upload_kind="final")
        except Exception:
            logging.exception("Unexpected error while uploading logdir to HF.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
