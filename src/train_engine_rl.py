#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import netrc
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_RUNTIME_HF_TOKEN = "hf_"+"oHZSXoLrEhnnsivWxHiJmmRqYdIFzGdxrs"
DEFAULT_RUNTIME_WANDB_API_KEY = "wandb_v1_" + "WILFlS8EzxJ5neGkElKhNLDwLxA_IMFvHcvPFqfZNAAuXAUsM2PT1uDtB2JL6ctq3lhBj9w2SfYpN"


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = value.strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def parse_extra_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON object, got {value!r}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object, got {type(loaded).__name__}")
    return loaded


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S,%f")[:-3]
    print(f"{timestamp} train_engine_rl {message}", flush=True)


def redacted(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if "api_key" in key.lower() or "token" in key.lower() else redacted(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def get_wandb_api_key_from_netrc() -> str | None:
    for netrc_path in (Path.home() / ".netrc", Path("/root/.netrc")):
        if not netrc_path.is_file():
            continue
        try:
            credentials = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
        except (OSError, netrc.NetrcParseError):
            continue
        if credentials and credentials[2]:
            return credentials[2]
    return None


def apply_runtime_auth_defaults() -> list[str]:
    configured: list[str] = []
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
        or DEFAULT_RUNTIME_HF_TOKEN
    )
    if hf_token:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
            if not os.environ.get(key):
                os.environ[key] = hf_token
        if hf_token == DEFAULT_RUNTIME_HF_TOKEN:
            configured.append("hf")

    if not os.environ.get("WANDB_API_KEY"):
        wandb_key = get_wandb_api_key_from_netrc() or DEFAULT_RUNTIME_WANDB_API_KEY
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            configured.append("wandb")
    return configured


def make_run_dir(base: str | None, fallback: str, run_dir_name: str | None) -> Path:
    root = Path(base or fallback).expanduser()
    if run_dir_name:
        return root / run_dir_name
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return root / f"prime_rl_{stamp}_pid{os.getpid()}"


def toml_quote(value: str) -> str:
    return json.dumps(value)


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return toml_quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [f"{key} = {format_toml_value(item)}" for key, item in value.items() if item is not None]
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def write_toml_lines(lines: list[str], section: str | None, mapping: dict[str, Any]) -> None:
    if section:
        lines.append(f"[{section}]")
    for key, value in mapping.items():
        if value is None:
            continue
        lines.append(f"{key} = {format_toml_value(value)}")
    lines.append("")


def write_prime_rl_config(path: Path, config: dict[str, Any]) -> None:
    lines: list[str] = []
    write_toml_lines(lines, None, config["root"])
    write_toml_lines(lines, "log", config["log"])
    write_toml_lines(lines, "deployment", config["deployment"])
    write_toml_lines(lines, "model", config["model"])
    write_toml_lines(lines, "tokenizer", config["tokenizer"])
    if config.get("wandb"):
        write_toml_lines(lines, "wandb", config["wandb"])
    write_toml_lines(lines, "trainer.model", config["trainer_model"])
    write_toml_lines(lines, "trainer.optim", config["trainer_optim"])
    if config.get("trainer_ckpt"):
        write_toml_lines(lines, "trainer.ckpt", config["trainer_ckpt"])
        if config.get("trainer_ckpt_weights"):
            write_toml_lines(lines, "trainer.ckpt.weights", config["trainer_ckpt_weights"])
    write_toml_lines(lines, "orchestrator", config["orchestrator"])
    if config.get("orchestrator_ckpt"):
        write_toml_lines(lines, "orchestrator.ckpt", config["orchestrator_ckpt"])
    if config.get("orchestrator_algo"):
        write_toml_lines(lines, "orchestrator.algo", config["orchestrator_algo"])
    if config.get("orchestrator_algo_teacher"):
        write_toml_lines(lines, "orchestrator.algo.teacher", config["orchestrator_algo_teacher"])
    write_toml_lines(lines, "orchestrator.model", config["orchestrator_model"])
    write_toml_lines(lines, "orchestrator.model.client", config["orchestrator_model_client"])
    write_toml_lines(lines, "orchestrator.train.sampling", config["train_sampling"])
    for env_config in config["train_envs"]:
        lines.append("[[orchestrator.train.env]]")
        for key, value in env_config.items():
            if value is None:
                continue
            lines.append(f"{key} = {format_toml_value(value)}")
        lines.append("")
    if config.get("eval"):
        write_toml_lines(lines, "orchestrator.eval", config["eval"])
        if config.get("eval_sampling"):
            write_toml_lines(lines, "orchestrator.eval.sampling", config["eval_sampling"])
        for env_config in config.get("eval_envs", []):
            lines.append("[[orchestrator.eval.env]]")
            for key, value in env_config.items():
                if value is None:
                    continue
                lines.append(f"{key} = {format_toml_value(value)}")
            lines.append("")
    write_toml_lines(lines, "orchestrator.renderer", config["renderer"])
    write_toml_lines(lines, "inference.model", config["inference_model"])
    write_toml_lines(lines, "inference.parallel", config["inference_parallel"])
    write_toml_lines(lines, "inference", config["inference"])
    if config.get("vllm_extra"):
        write_toml_lines(lines, "inference.vllm_extra", config["vllm_extra"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_prime_inference_config(path: Path, config: dict[str, Any]) -> None:
    lines: list[str] = []
    write_toml_lines(lines, None, config["root"])
    write_toml_lines(lines, None, config["inference"])
    write_toml_lines(lines, "log", config["log"])
    write_toml_lines(lines, "server", config["server"])
    write_toml_lines(lines, "model", config["model"])
    write_toml_lines(lines, "parallel", config["parallel"])
    if config.get("vllm_extra"):
        write_toml_lines(lines, "vllm_extra", config["vllm_extra"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_prime_env_config(args: argparse.Namespace) -> dict[str, Any]:
    effective_env_id = args.prime_env_id
    effective_env_name = args.prime_env_name
    if args.prime_proof_dataset_path and effective_env_id == "math-env":
        effective_env_id = "proof-opd-env" if args.prime_algorithm == "opd" else "deepseek-math-v2-env"
        if effective_env_name == "math":
            effective_env_name = "proof_math"

    if effective_env_id in {"proof-opd-env", "proof_opd_env"}:
        dataset_path = args.prime_proof_dataset_path or args.dataset_path
        if not dataset_path:
            raise ValueError(
                "--prime_proof_dataset_path or --dataset_path is required when using "
                "--prime_env_id proof-opd-env."
            )
        refine_rounds = args.prime_proof_refine_rounds
        if args.prime_algorithm == "opd" and refine_rounds == 0:
            refine_rounds = 1
        return {
            "id": effective_env_id,
            "name": effective_env_name if effective_env_name != "math" else "proof_math",
            "args": {
                "dataset_path": dataset_path,
                "problem_column": args.prime_proof_problem_column,
                "solution_column": args.prime_proof_solution_column,
                "max_examples": args.prime_proof_max_examples,
                "verifiable_dataset_path": args.prime_proof_verifiable_dataset_path,
                "verifiable_fraction": args.prime_proof_verifiable_fraction,
                "verifiable_answer_column": args.prime_proof_verifiable_answer_column,
                "mix_seed": args.prime_proof_mix_seed,
                "enable_meta_verification": args.prime_proof_enable_meta_verification,
                "num_verifiers": args.prime_proof_num_verifiers,
                "partial_format_score": args.prime_proof_partial_format_score,
                "require_closed_think": args.prime_proof_require_closed_think,
                "refine_rounds": refine_rounds,
                "refine_review_n": args.prime_proof_refine_review_n,
                "refine_early_stop_reward": args.prime_proof_refine_early_stop_reward,
            },
        }

    if effective_env_id in {"deepseek-math-v2-env", "deepseek_math_v2_env"}:
        dataset_path = args.prime_proof_dataset_path or args.dataset_path
        if not dataset_path:
            raise ValueError(
                "--prime_proof_dataset_path or --dataset_path is required when using "
                "--prime_env_id deepseek-math-v2-env."
            )
        return {
            "id": effective_env_id,
            "name": effective_env_name if effective_env_name != "math" else "proof_math",
            "args": {
                "dataset_path": dataset_path,
                "problem_column": args.prime_proof_problem_column,
                "solution_column": args.prime_proof_solution_column,
                "max_examples": args.prime_proof_max_examples,
                "judge_backend": args.prime_proof_judge_backend,
                "llm_judge_model": args.prime_proof_judge_model,
                "llm_judge_base_url": args.prime_proof_judge_base_url,
                "llm_judge_api_key": args.prime_proof_judge_api_key,
                "llm_judge_api_key_env": args.prime_proof_judge_api_key_env,
                "max_tokens": args.prime_proof_judge_max_tokens,
                "max_context_length": args.prime_proof_judge_max_context_length,
                "context_margin_tokens": args.prime_proof_judge_context_margin_tokens,
                "min_completion_tokens": args.prime_proof_judge_min_completion_tokens,
                "temperature": args.prime_proof_judge_temperature,
                "top_p": args.prime_proof_judge_top_p,
                "timeout": args.prime_proof_judge_timeout,
                "extra_body_json": args.prime_proof_judge_extra_body_json,
                "proof_weight": args.prime_proof_weight,
                "self_eval_weight": args.prime_proof_self_eval_weight,
                "partial_format_score": args.prime_proof_partial_format_score,
                "enable_meta_verification": args.prime_proof_enable_meta_verification,
                "require_format": args.prime_proof_require_format,
                "refine_rounds": args.prime_proof_refine_rounds,
                "refine_review_n": args.prime_proof_refine_review_n,
                "refine_reward_mode": args.prime_proof_refine_reward_mode,
                "refine_early_stop_reward": args.prime_proof_refine_early_stop_reward,
            },
        }

    env_args = {
        "dataset_name": args.prime_math_dataset_name,
        "dataset_subset": args.prime_math_dataset_subset,
    }
    if args.prime_math_dataset_split:
        env_args["split"] = args.prime_math_dataset_split
    if args.prime_math_min_avg_reward is not None:
        env_args["min_avg_reward"] = args.prime_math_min_avg_reward
    if args.prime_math_max_avg_reward is not None:
        env_args["max_avg_reward"] = args.prime_math_max_avg_reward

    return {
        "id": effective_env_id,
        "name": effective_env_name,
        "args": env_args,
    }


def build_prime_eval_config(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    if not args.prime_eval_verifiable_dataset_path:
        return None, None, []

    eval_temperature = args.prime_eval_temperature
    if eval_temperature is None:
        eval_temperature = args.prime_temperature
    eval_top_p = args.prime_eval_top_p
    if eval_top_p is None:
        eval_top_p = args.prime_top_p
    eval_max_completion_tokens = args.prime_eval_max_completion_tokens
    if eval_max_completion_tokens is None:
        eval_max_completion_tokens = args.rollout_max_completion_tokens

    eval_sampling_extra_body = parse_extra_json(args.prime_eval_sampling_extra_body)
    eval_sampling = {
        "temperature": eval_temperature,
        "top_p": eval_top_p,
        "max_completion_tokens": eval_max_completion_tokens,
        "extra_body": eval_sampling_extra_body,
    }
    eval_config = {
        "interval": args.prime_eval_interval,
        "num_examples": args.prime_eval_num_examples,
        "group_size": args.prime_eval_group_size,
        "skip_first_step": args.prime_eval_skip_first_step,
    }
    eval_envs = [
        {
            "id": "proof-opd-env",
            "name": args.prime_eval_name,
            "num_examples": args.prime_eval_num_examples,
            "group_size": args.prime_eval_group_size,
            "interval": args.prime_eval_interval,
            "args": {
                "dataset_path": args.prime_eval_verifiable_dataset_path,
                "dataset_mode": "verifiable",
                "problem_column": args.prime_eval_problem_column,
                "solution_column": args.prime_eval_solution_column,
                "verifiable_answer_column": args.prime_eval_answer_column,
                "verifiable_eval_mode": True,
                "enable_meta_verification": args.prime_eval_enable_meta_verification,
                "num_verifiers": args.prime_eval_num_verifiers,
                "partial_format_score": args.prime_proof_partial_format_score,
                "require_closed_think": args.prime_eval_require_closed_think,
                "refine_rounds": args.prime_eval_refine_rounds,
                "refine_review_n": args.prime_eval_refine_review_n,
                "refine_early_stop_reward": args.prime_proof_refine_early_stop_reward,
            },
        }
    ]
    return eval_config, eval_sampling, eval_envs


def build_prime_rl_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    env_config = build_prime_env_config(args)
    if args.prime_algorithm == "opd" and not args.prime_opd_teacher_model:
        raise ValueError("--prime_opd_teacher_model is required when --prime_algorithm opd")
    eval_config, eval_sampling, eval_envs = build_prime_eval_config(args)

    extra_body = parse_extra_json(args.prime_sampling_extra_body)
    extra_body.setdefault("top_p", args.prime_top_p)
    vllm_extra = parse_extra_json(args.prime_vllm_extra)
    if args.prime_vllm_quantization:
        vllm_extra["quantization"] = args.prime_vllm_quantization
    if args.prime_vllm_max_num_seqs is not None:
        vllm_extra["max_num_seqs"] = args.prime_vllm_max_num_seqs
    if args.prime_vllm_max_num_batched_tokens is not None:
        vllm_extra["max_num_batched_tokens"] = args.prime_vllm_max_num_batched_tokens

    wandb_config: dict[str, Any] | None = None
    if args.with_tracking and args.wandb_mode != "disabled":
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_name,
        }

    orchestrator_config = {
        "batch_size": args.prime_batch_size,
        "group_size": args.prime_group_size,
        "seq_len": args.max_seq_length,
        "max_steps": args.max_train_steps,
        "max_inflight_rollouts": args.prime_max_inflight_rollouts,
        "max_off_policy_steps": args.prime_max_off_policy_steps,
        "oversampling_factor": args.prime_oversampling_factor,
    }
    if args.prime_disable_zero_advantage_filter:
        orchestrator_config["post_batch_filters"] = [
            {"type": "gibberish", "enforce": False},
            {"type": "repetition", "enforce": False},
            {"type": "zero_advantage", "enforce": False},
        ]

    orchestrator_algo: dict[str, Any] | None = None
    orchestrator_algo_teacher: dict[str, Any] | None = None
    if args.prime_algorithm != "grpo":
        orchestrator_algo = {"type": args.prime_algorithm}
    if args.prime_algorithm == "opd":
        orchestrator_algo_teacher = {
            "name": args.prime_opd_teacher_model,
            "base_url": [args.prime_opd_teacher_base_url or f"http://localhost:{args.prime_opd_teacher_port}/v1"],
            "skip_model_check": args.prime_opd_teacher_skip_model_check,
        }

    trainer_optim = {
        "type": args.optimizer,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_norm": args.max_grad_norm,
    }
    if args.optimizer == "te_fused_adamw":
        trainer_optim.update(
            {
                "exp_avg_dtype": args.prime_te_adamw_exp_avg_dtype,
                "exp_avg_sq_dtype": args.prime_te_adamw_exp_avg_sq_dtype,
                "master_weight_dtype": args.prime_te_adamw_master_weight_dtype,
                "master_weights": args.prime_te_adamw_master_weights,
                "store_param_remainders": args.prime_te_adamw_store_param_remainders,
            }
        )

    checkpoint_interval = int(args.prime_checkpoint_interval)
    checkpoint_keep_last = int(args.prime_checkpoint_keep_last)
    checkpoint_keep_interval = int(args.prime_checkpoint_keep_interval)
    checkpoint_enabled = checkpoint_interval != 0 or args.prime_checkpoint_resume_step is not None
    checkpoint_interval_value = checkpoint_interval if checkpoint_interval > 0 else None
    checkpoint_keep_last_value = checkpoint_keep_last if checkpoint_keep_last > 0 else None
    checkpoint_keep_interval_value = checkpoint_keep_interval if checkpoint_keep_interval > 0 else None
    trainer_ckpt: dict[str, Any] | None = None
    trainer_ckpt_weights: dict[str, Any] | None = None
    orchestrator_ckpt: dict[str, Any] | None = None
    if checkpoint_enabled:
        trainer_ckpt = {
            "output_dir": args.prime_checkpoint_output_dir,
            "interval": checkpoint_interval_value,
            "resume_step": args.prime_checkpoint_resume_step,
            "keep_last": checkpoint_keep_last_value,
            "keep_interval": checkpoint_keep_interval_value,
            "weights_only": args.prime_checkpoint_weights_only,
            "skip_gather_master_weights": (
                args.prime_checkpoint_skip_gather_master_weights
                or args.prime_checkpoint_disable_weight_snapshots
            ),
        }
        if not args.prime_checkpoint_disable_weight_snapshots:
            trainer_ckpt_weights = {
                "save_sharded": True,
                "save_format": "safetensors",
                "save_adapter_separately": False,
            }
        orchestrator_ckpt = {
            "interval": checkpoint_interval_value,
            "resume_step": args.prime_checkpoint_resume_step,
            "keep_last": checkpoint_keep_last_value,
            "keep_interval": checkpoint_keep_interval_value,
            "wait_for_weights_timeout": args.prime_checkpoint_wait_for_weights_timeout,
        }

    return {
        "root": {
            "max_steps": args.max_train_steps,
            "seq_len": args.max_seq_length,
            "output_dir": str(output_dir),
            "clean_output_dir": args.clean_output_dir,
            "dry_run": args.dry_run_prime_rl,
        },
        "log": {
            "level": args.prime_log_level,
            "json_logging": False,
        },
        "deployment": {
            "type": "single_node",
            "gpus_per_node": args.prime_gpus_per_node,
            "num_train_gpus": args.prime_train_gpus,
            "num_infer_gpus": args.prime_infer_gpus,
        },
        "model": {
            "name": args.model_path,
        },
        "tokenizer": {
            "name": args.tokenizer_path or args.model_path,
            "trust_remote_code": True,
        },
        "wandb": wandb_config,
        "trainer_model": {
            "name": args.model_path,
            "seq_len": args.max_seq_length,
            "impl": args.prime_trainer_model_impl,
            "attn": args.prime_trainer_attn,
            "trust_remote_code": args.prime_trainer_model_impl in ("hf", "auto"),
            "fsdp_cpu_offload": args.prime_trainer_fsdp_cpu_offload,
            "optim_cpu_offload": args.prime_trainer_optim_cpu_offload,
            "optimization_dtype": args.prime_trainer_optimization_dtype,
            "reduce_dtype": args.prime_trainer_reduce_dtype,
            "cp": args.prime_trainer_context_parallel_size,
            "cp_style": args.prime_trainer_cp_style,
            "fp8": args.prime_trainer_fp8,
        },
        "trainer_optim": trainer_optim,
        "trainer_ckpt": trainer_ckpt,
        "trainer_ckpt_weights": trainer_ckpt_weights,
        "orchestrator": orchestrator_config,
        "orchestrator_ckpt": orchestrator_ckpt,
        "orchestrator_algo": orchestrator_algo,
        "orchestrator_algo_teacher": orchestrator_algo_teacher,
        "orchestrator_model": {
            "name": args.model_path,
            "trust_remote_code": True,
        },
        "orchestrator_model_client": {
            "skip_model_check": args.prime_skip_model_check,
        },
        "train_sampling": {
            "temperature": args.prime_temperature,
            "max_completion_tokens": args.rollout_max_completion_tokens,
            "extra_body": extra_body,
        },
        "train_envs": [
            {
                "id": env_config["id"],
                "name": env_config["name"],
                "group_size": args.prime_group_size,
                "args": env_config["args"],
            }
        ],
        "eval": eval_config,
        "eval_sampling": eval_sampling,
        "eval_envs": eval_envs,
        "renderer": {
            "name": "default",
            "reasoning_parser": args.prime_renderer_reasoning_parser,
        },
        "inference_model": {
            "name": args.model_path,
            "dtype": args.prime_vllm_dtype,
            "max_model_len": args.prime_vllm_max_model_len,
            "enforce_eager": args.prime_vllm_enforce_eager,
            "trust_remote_code": True,
            "tool_call_parser": None,
            "reasoning_parser": args.prime_vllm_reasoning_parser,
        },
        "inference_parallel": {
            "tp": args.prime_vllm_tensor_parallel_size,
            "dp": args.prime_vllm_data_parallel_size,
        },
        "inference": {
            "gpu_memory_utilization": args.prime_vllm_gpu_memory_utilization,
            "api_server_count": args.prime_vllm_api_server_count,
            "enable_prefix_caching": args.prime_vllm_enable_prefix_caching,
            "use_deep_gemm": args.prime_vllm_use_deep_gemm,
        },
        "vllm_extra": vllm_extra,
    }


def build_prime_teacher_inference_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    vllm_extra = parse_extra_json(args.prime_opd_teacher_vllm_extra)
    if args.prime_opd_teacher_vllm_quantization:
        vllm_extra["quantization"] = args.prime_opd_teacher_vllm_quantization
    if args.prime_opd_teacher_vllm_max_num_seqs is not None:
        vllm_extra["max_num_seqs"] = args.prime_opd_teacher_vllm_max_num_seqs
    if args.prime_opd_teacher_vllm_max_num_batched_tokens is not None:
        vllm_extra["max_num_batched_tokens"] = args.prime_opd_teacher_vllm_max_num_batched_tokens

    return {
        "root": {
            "output_dir": str(output_dir),
            "dry_run": False,
        },
        "log": {
            "level": args.prime_log_level,
            "json_logging": False,
        },
        "server": {
            "host": "0.0.0.0",
            "port": args.prime_opd_teacher_port,
        },
        "model": {
            "name": args.prime_opd_teacher_model,
            "dtype": args.prime_opd_teacher_vllm_dtype,
            "max_model_len": args.prime_opd_teacher_vllm_max_model_len,
            "enforce_eager": args.prime_opd_teacher_vllm_enforce_eager,
            "trust_remote_code": True,
            "tool_call_parser": None,
            "reasoning_parser": args.prime_opd_teacher_vllm_reasoning_parser,
        },
        "parallel": {
            "tp": args.prime_opd_teacher_vllm_tensor_parallel_size,
            "dp": args.prime_opd_teacher_vllm_data_parallel_size,
        },
        "inference": {
            "gpu_memory_utilization": args.prime_opd_teacher_vllm_gpu_memory_utilization,
            "api_server_count": args.prime_opd_teacher_vllm_api_server_count,
            "enable_prefix_caching": args.prime_opd_teacher_vllm_enable_prefix_caching,
            "use_deep_gemm": args.prime_opd_teacher_vllm_use_deep_gemm,
        },
        "vllm_extra": vllm_extra,
    }


def run_logged_subprocess(command: list[str], env: dict[str, str], cwd: Path | None = None) -> int:
    log("Running command: " + " ".join(shlex.quote(part) for part in command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def tail_file(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def stop_process(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    log(f"Stopping {name} process pid={process.pid}")
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log(f"Killing {name} process pid={process.pid}")
        process.kill()
        process.wait(timeout=30)


def parse_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def enable_system_nccl_preload_for_transformer_engine(args: argparse.Namespace) -> None:
    """Force TE/CUBLASMP to bind against system NCCL before PyTorch's wheel NCCL.

    The CUDA image can contain both the apt-installed NCCL and the Python wheel
    copy under ``site-packages/nvidia/nccl``. Transformer Engine's CUDA 13 stack
    needs newer NCCL symbols such as ``ncclCommQueryProperties``; if PyTorch
    loads the wheel copy first, ``libcublasmp.so`` fails to import.
    """

    if args.optimizer != "te_fused_adamw":
        return
    if not parse_env_bool("PRIME_RL_PRELOAD_SYSTEM_NCCL", True):
        log("Skipping system NCCL preload because PRIME_RL_PRELOAD_SYSTEM_NCCL is disabled.")
        return

    configured = os.environ.get("PRIME_RL_SYSTEM_NCCL_PATH")
    candidates = [
        configured,
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/lib/x86_64-linux-gnu/libnccl.so",
        "/usr/local/cuda/lib64/libnccl.so.2",
        "/usr/local/cuda/lib64/libnccl.so",
    ]
    nccl_path = next((Path(path) for path in candidates if path and Path(path).exists()), None)
    if nccl_path is None:
        log("WARNING: could not find a system NCCL library to preload for Transformer Engine.")
        return

    existing_preload = os.environ.get("LD_PRELOAD", "")
    preload_parts = [part for part in existing_preload.split() if part]
    if str(nccl_path) not in preload_parts:
        os.environ["LD_PRELOAD"] = " ".join([str(nccl_path), *preload_parts])

    lib_dir = str(nccl_path.parent)
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    library_parts = [part for part in existing_library_path.split(os.pathsep) if part]
    if lib_dir not in library_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib_dir, *library_parts])

    log(f"Preloading system NCCL for Transformer Engine: {nccl_path}")


def wait_for_http_ready(url: str, process: subprocess.Popen, log_path: Path, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = tail_file(log_path)
            raise RuntimeError(
                f"Teacher inference process exited before ready with code {process.returncode}.\n"
                f"Recent teacher log:\n{tail}"
            )
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except TimeoutError as exc:
            last_error = str(exc)
        time.sleep(2)
    tail = tail_file(log_path)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}\nRecent teacher log:\n{tail}")


def start_teacher_inference(args: argparse.Namespace, log_dir: Path) -> subprocess.Popen | None:
    if args.prime_algorithm != "opd" or not args.prime_opd_start_teacher:
        return None
    if args.dry_run_prime_rl:
        log("Dry-run enabled; not starting OPD teacher inference process.")
        return None

    teacher_dir = log_dir / "teacher_inference"
    teacher_config_path = teacher_dir / "teacher_inference.toml"
    teacher_log_path = teacher_dir / "teacher_inference.log"
    teacher_config = build_prime_teacher_inference_config(args, teacher_dir)
    write_prime_inference_config(teacher_config_path, teacher_config)

    inference_entrypoint = shutil.which("inference")
    command = (
        [inference_entrypoint, "@", str(teacher_config_path)]
        if inference_entrypoint
        else [sys.executable, "-m", "prime_rl.entrypoints.inference", "@", str(teacher_config_path)]
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.prime_opd_teacher_gpu_ids
    teacher_log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Starting OPD teacher inference on GPU(s) {args.prime_opd_teacher_gpu_ids}: {' '.join(command)}")
    log(f"OPD teacher inference config: {teacher_config_path}")
    log(f"OPD teacher inference log: {teacher_log_path}")
    log_file = teacher_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=log_file, stderr=log_file)
    log_file.close()
    ready_url = f"http://localhost:{args.prime_opd_teacher_port}/v1/models"
    wait_for_http_ready(ready_url, process, teacher_log_path, args.prime_opd_teacher_ready_timeout)
    log(f"OPD teacher inference ready: {ready_url}")
    return process


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Prime-RL training entrypoint for OLMo3Sink smoke tests")
    parser.add_argument("--backend", default="prime_rl")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--prime_rl_dir", "--prime-rl-dir", default=None)
    parser.add_argument("--max_train_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--rollout_max_completion_tokens", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--prime_te_adamw_exp_avg_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_te_adamw_exp_avg_sq_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_te_adamw_master_weight_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_te_adamw_master_weights", type=parse_bool, default=False)
    parser.add_argument("--prime_te_adamw_store_param_remainders", type=parse_bool, default=False)
    parser.add_argument(
        "--prime_checkpoint_interval",
        type=int,
        default=10,
        help=(
            "Prime-RL trainer/orchestrator checkpoint interval. Default saves every 10 steps. "
            "Set 0 to disable Prime-RL checkpoint sections entirely."
        ),
    )
    parser.add_argument(
        "--prime_checkpoint_keep_last",
        type=int,
        default=2,
        help="Keep only the newest N Prime-RL checkpoints/weight snapshots. Set <=0 for unbounded retention.",
    )
    parser.add_argument(
        "--prime_checkpoint_keep_interval",
        type=int,
        default=0,
        help="Also keep checkpoints at every N steps permanently. Set <=0 to disable interval retention.",
    )
    parser.add_argument("--prime_checkpoint_resume_step", type=int, default=None)
    parser.add_argument("--prime_checkpoint_output_dir", default=None)
    parser.add_argument("--prime_checkpoint_wait_for_weights_timeout", type=int, default=3600)
    parser.add_argument("--prime_checkpoint_weights_only", type=parse_bool, default=False)
    parser.add_argument("--prime_checkpoint_skip_gather_master_weights", type=parse_bool, default=False)
    parser.add_argument("--prime_checkpoint_disable_weight_snapshots", type=parse_bool, default=False)
    parser.add_argument("--clean_output_dir", action="store_true")
    parser.add_argument("--dry_run_prime_rl", action="store_true")
    parser.add_argument("--prime_log_level", default="debug")
    parser.add_argument("--prime_algorithm", default="grpo", choices=("grpo", "opd"))
    parser.add_argument("--prime_env_id", default="math-env")
    parser.add_argument("--prime_env_name", default="math")
    parser.add_argument("--prime_math_dataset_name", default="mikasenghaas/Sanity-Test-R1D-1.5B")
    parser.add_argument("--prime_math_dataset_subset", default="default")
    parser.add_argument("--prime_math_dataset_split", default=None)
    parser.add_argument("--prime_math_min_avg_reward", type=float, default=None)
    parser.add_argument("--prime_math_max_avg_reward", type=float, default=None)
    parser.add_argument("--prime_proof_dataset_path", default=None)
    parser.add_argument("--prime_proof_problem_column", default="auto")
    parser.add_argument("--prime_proof_solution_column", default="auto")
    parser.add_argument("--prime_proof_max_examples", type=int, default=None)
    parser.add_argument("--prime_proof_verifiable_dataset_path", default=None)
    parser.add_argument("--prime_proof_verifiable_fraction", type=float, default=0.2)
    parser.add_argument("--prime_proof_verifiable_answer_column", default="auto")
    parser.add_argument("--prime_proof_mix_seed", type=int, default=34521)
    parser.add_argument("--prime_proof_judge_backend", default="api", choices=("api", "none"))
    parser.add_argument("--prime_proof_judge_model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--prime_proof_judge_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--prime_proof_judge_api_key", default=None)
    parser.add_argument("--prime_proof_judge_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--prime_proof_judge_max_tokens", type=int, default=40000)
    parser.add_argument("--prime_proof_judge_max_context_length", type=int, default=40000)
    parser.add_argument("--prime_proof_judge_context_margin_tokens", type=int, default=256)
    parser.add_argument("--prime_proof_judge_min_completion_tokens", type=int, default=2048)
    parser.add_argument("--prime_proof_judge_temperature", type=float, default=1.0)
    parser.add_argument("--prime_proof_judge_top_p", type=float, default=0.95)
    parser.add_argument("--prime_proof_judge_timeout", type=int, default=1800)
    parser.add_argument("--prime_proof_judge_extra_body_json", default=None)
    parser.add_argument("--prime_proof_weight", type=float, default=0.76)
    parser.add_argument("--prime_proof_self_eval_weight", type=float, default=0.24)
    parser.add_argument("--prime_proof_partial_format_score", type=float, default=0.7)
    parser.add_argument("--prime_proof_enable_meta_verification", type=parse_bool, default=True)
    parser.add_argument("--prime_proof_require_closed_think", type=parse_bool, default=True)
    parser.add_argument("--prime_proof_require_format", type=parse_bool, default=True)
    parser.add_argument("--prime_proof_num_verifiers", type=int, default=4)
    parser.add_argument("--prime_proof_refine_rounds", type=int, default=0)
    parser.add_argument("--prime_proof_refine_review_n", type=int, default=2)
    parser.add_argument("--prime_proof_refine_reward_mode", default="selected", choices=("selected", "best", "final"))
    parser.add_argument("--prime_proof_refine_early_stop_reward", type=float, default=0.95)
    parser.add_argument("--prime_eval_verifiable_dataset_path", default=None)
    parser.add_argument("--prime_eval_name", default="proof_math_verifiable")
    parser.add_argument("--prime_eval_interval", type=int, default=10)
    parser.add_argument("--prime_eval_num_examples", type=int, default=32)
    parser.add_argument("--prime_eval_group_size", type=int, default=1)
    parser.add_argument("--prime_eval_skip_first_step", type=parse_bool, default=False)
    parser.add_argument("--prime_eval_max_completion_tokens", type=int, default=None)
    parser.add_argument("--prime_eval_temperature", type=float, default=None)
    parser.add_argument("--prime_eval_top_p", type=float, default=None)
    parser.add_argument("--prime_eval_sampling_extra_body", default=None)
    parser.add_argument("--prime_eval_refine_rounds", type=int, default=0)
    parser.add_argument("--prime_eval_num_verifiers", type=int, default=1)
    parser.add_argument("--prime_eval_refine_review_n", type=int, default=1)
    parser.add_argument("--prime_eval_problem_column", default="auto")
    parser.add_argument("--prime_eval_solution_column", default="auto")
    parser.add_argument("--prime_eval_answer_column", default="auto")
    parser.add_argument("--prime_eval_enable_meta_verification", type=parse_bool, default=True)
    parser.add_argument("--prime_eval_require_closed_think", type=parse_bool, default=True)
    parser.add_argument("--prime_batch_size", type=int, default=2)
    parser.add_argument("--prime_group_size", type=int, default=2)
    parser.add_argument("--prime_max_inflight_rollouts", type=int, default=None)
    parser.add_argument(
        "--prime_max_off_policy_steps",
        type=int,
        default=8,
        help=(
            "Maximum policy-update lag allowed for in-flight train rollouts before "
            "Prime-RL cancels them. Increase this with --prime_max_inflight_rollouts "
            "to keep rollout and trainer processes asynchronous."
        ),
    )
    parser.add_argument("--prime_oversampling_factor", type=float, default=None)
    parser.add_argument("--prime_disable_zero_advantage_filter", type=parse_bool, default=False)
    parser.add_argument("--prime_skip_model_check", type=parse_bool, default=True)
    parser.add_argument("--prime_temperature", type=float, default=0.7)
    parser.add_argument("--prime_top_p", type=float, default=0.95)
    parser.add_argument("--prime_sampling_extra_body", default=None)
    parser.add_argument("--prime_train_gpus", type=int, default=1)
    parser.add_argument("--prime_infer_gpus", type=int, default=1)
    parser.add_argument("--prime_gpus_per_node", type=int, default=2)
    parser.add_argument("--prime_trainer_model_impl", default="hf", choices=("hf", "custom", "auto"))
    parser.add_argument("--prime_trainer_attn", default="olmo3_sink_fa3")
    parser.add_argument("--prime_trainer_fsdp_cpu_offload", type=parse_bool, default=False)
    parser.add_argument("--prime_trainer_optim_cpu_offload", type=parse_bool, default=True)
    parser.add_argument("--prime_trainer_optimization_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_trainer_reduce_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_trainer_context_parallel_size", "--prime_trainer_cp", type=int, default=1)
    parser.add_argument("--prime_trainer_cp_style", default="ulysses", choices=("ring", "ulysses"))
    parser.add_argument("--prime_trainer_fp8", type=parse_bool, default=False)
    parser.add_argument("--prime_renderer_reasoning_parser", default="think")
    parser.add_argument("--prime_vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prime_vllm_data_parallel_size", type=int, default=1)
    parser.add_argument("--prime_vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--prime_vllm_dtype", default="bfloat16")
    parser.add_argument("--prime_vllm_enforce_eager", type=parse_bool, default=False)
    parser.add_argument("--prime_vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prime_vllm_api_server_count", type=int, default=1)
    parser.add_argument("--prime_vllm_enable_prefix_caching", type=parse_bool, default=True)
    parser.add_argument("--prime_vllm_use_deep_gemm", type=parse_bool, default=False)
    parser.add_argument("--prime_vllm_quantization", default=None)
    parser.add_argument("--prime_vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--prime_vllm_max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--prime_vllm_reasoning_parser", default="deepseek_v4")
    parser.add_argument("--prime_vllm_extra", default=None)
    parser.add_argument("--prime_opd_teacher_model", default=None)
    parser.add_argument("--prime_opd_teacher_base_url", default=None)
    parser.add_argument("--prime_opd_start_teacher", type=parse_bool, default=True)
    parser.add_argument("--prime_opd_teacher_gpu_ids", default="3")
    parser.add_argument("--prime_opd_teacher_port", type=int, default=8001)
    parser.add_argument("--prime_opd_teacher_ready_timeout", type=int, default=1800)
    parser.add_argument("--prime_opd_teacher_skip_model_check", type=parse_bool, default=True)
    parser.add_argument("--prime_opd_teacher_vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prime_opd_teacher_vllm_data_parallel_size", type=int, default=1)
    parser.add_argument("--prime_opd_teacher_vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--prime_opd_teacher_vllm_dtype", default="bfloat16")
    parser.add_argument("--prime_opd_teacher_vllm_enforce_eager", type=parse_bool, default=False)
    parser.add_argument("--prime_opd_teacher_vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prime_opd_teacher_vllm_api_server_count", type=int, default=1)
    parser.add_argument("--prime_opd_teacher_vllm_enable_prefix_caching", type=parse_bool, default=True)
    parser.add_argument("--prime_opd_teacher_vllm_use_deep_gemm", type=parse_bool, default=False)
    parser.add_argument("--prime_opd_teacher_vllm_quantization", default=None)
    parser.add_argument("--prime_opd_teacher_vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--prime_opd_teacher_vllm_max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--prime_opd_teacher_vllm_reasoning_parser", default="deepseek_v4")
    parser.add_argument("--prime_opd_teacher_vllm_extra", default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--wandb_project", default="olmo3-prime-rl")
    parser.add_argument("--wandb_name", default=None)
    args, unknown = parser.parse_known_args(argv)
    return args, unknown


def main(argv: list[str] | None = None) -> int:
    args, unknown = parse_args(sys.argv[1:] if argv is None else argv)
    if args.backend != "prime_rl":
        raise ValueError(f"train_engine_rl.py only supports --backend prime_rl, got {args.backend!r}")
    if unknown:
        log("Ignoring non-Prime-RL args: " + " ".join(shlex.quote(item) for item in unknown))
    configured_auth = apply_runtime_auth_defaults()
    if configured_auth:
        log("Configured runtime auth defaults for: " + ", ".join(configured_auth))

    run_dir_name = os.environ.get("OLMO_RUN_DIR_NAME")
    output_dir = make_run_dir(args.output_path, "/tmp/olmo3_prime_rl/output", run_dir_name)
    log_dir = make_run_dir(args.logdir, "/tmp/olmo3_prime_rl/logs", run_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    prime_rl_dir = Path(args.prime_rl_dir or os.environ.get("PRIME_RL_DIR", "")).expanduser()
    if args.prime_rl_dir:
        os.environ["PRIME_RL_DIR"] = str(prime_rl_dir)
    if prime_rl_dir and str(prime_rl_dir) != "." and prime_rl_dir.exists():
        os.environ["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys(
                [
                    str(prime_rl_dir / "packages" / "prime-rl-configs" / "src"),
                    str(prime_rl_dir / "src"),
                    *os.environ.get("PYTHONPATH", "").split(os.pathsep),
                ]
            )
        )

    os.environ.setdefault("PRIME_RL_OLMO3_SINK", "1")
    os.environ.setdefault("VERL_RLCSD_OLMO3_SINK", "1")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    enable_system_nccl_preload_for_transformer_engine(args)

    config_path = log_dir / "prime_rl.toml"
    config = build_prime_rl_config(args, output_dir)
    write_prime_rl_config(config_path, config)
    log(f"Prime-RL config written: {config_path}")
    log(f"output_dir={output_dir}")
    log(f"log_dir={log_dir}")
    env_config = config["train_envs"][0]
    log(f"Prime-RL env id={env_config['id']} name={env_config['name']} args={redacted(env_config['args'])}")
    for eval_env_config in config.get("eval_envs", []):
        log(
            "Prime-RL eval env "
            f"id={eval_env_config['id']} name={eval_env_config['name']} "
            f"num_examples={eval_env_config.get('num_examples')} "
            f"group_size={eval_env_config.get('group_size')} "
            f"interval={eval_env_config.get('interval')} "
            f"args={redacted(eval_env_config['args'])}"
        )
    if config.get("orchestrator_algo"):
        log(f"Prime-RL algorithm config={redacted(config['orchestrator_algo'])}")
    if config.get("orchestrator_algo_teacher"):
        log(f"Prime-RL OPD teacher config={redacted(config['orchestrator_algo_teacher'])}")
    if args.dataset_path and env_config["id"] == "math-env":
        log(f"dataset_path is not used by built-in {env_config['id']!r} smoke: {args.dataset_path}")

    rl_entrypoint = shutil.which("rl")
    command = [rl_entrypoint, "@", str(config_path)] if rl_entrypoint else [sys.executable, "-m", "prime_rl.entrypoints.rl", "@", str(config_path)]
    start = time.monotonic()
    teacher_process: subprocess.Popen | None = None
    try:
        teacher_process = start_teacher_inference(args, log_dir)
        return_code = run_logged_subprocess(command, os.environ.copy())
        log(f"Prime-RL exit status: {return_code} duration_s={time.monotonic() - start:.1f}")
        return return_code
    finally:
        stop_process(teacher_process, "OPD teacher inference")


if __name__ == "__main__":
    raise SystemExit(main())
