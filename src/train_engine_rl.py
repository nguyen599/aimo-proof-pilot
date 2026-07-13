#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import netrc
import os
import signal
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
DEFAULT_STUDENT_HF_REPO = "chankhavu/yccchen-olmo3-deploy"
DEFAULT_TEACHER_HF_REPO = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_DATASET_HF_REPO = "ycchen/dsflash-proof-distill-v2-test"
DEFAULT_DATASET_HF_FILENAME = "data/per_turn.parquet"


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


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def normalize_optional_choice(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "false", "off"}:
        return None
    return text


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S,%f")[:-3]
    print(f"{timestamp} train_engine_rl {message}", flush=True)


def resolve_wandb_shared_run_id(env: dict[str, str], log_dir: Path) -> str:
    explicit_run_id = env.get("WANDB_SHARED_RUN_ID")
    if explicit_run_id:
        return explicit_run_id
    run_identity = env.get("OLMO_RUN_DIR_NAME") or env.get("PRIME_3NODE_RUN_NAME") or str(log_dir.resolve())
    return hashlib.sha256(run_identity.encode("utf-8")).hexdigest()[:32]


def redacted(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if "api_key" in key.lower() or "token" in key.lower() else redacted(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def prime_rl_pythonpath_entries(prime_rl_dir: Path) -> list[str]:
    candidates = [
        prime_rl_dir / "packages" / "prime-rl-configs" / "src",
        prime_rl_dir / "src",
        prime_rl_dir / "deps" / "pydantic-config" / "src",
        prime_rl_dir / "deps" / "pydantic-config",
        prime_rl_dir / "deps" / "renderers" / "src",
        prime_rl_dir / "deps" / "renderers",
        prime_rl_dir / "deps" / "verifiers" / "src",
        prime_rl_dir / "deps" / "verifiers",
        prime_rl_dir / "deps" / "research-environments" / "src",
        prime_rl_dir / "deps" / "research-environments",
    ]
    return [str(path) for path in candidates if path.is_dir()]


def configure_cuda_toolchain_env() -> None:
    cuda_home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        return

    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", os.environ["CUDA_HOME"])
    os.environ.setdefault("CUDACXX", str(nvcc))
    os.environ.setdefault("NVCC", str(nvcc))

    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    cuda_bin = str(cuda_home / "bin")
    if cuda_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_parts])


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
    if not os.environ.get("WANDB_API_KEY"):
        wandb_key = get_wandb_api_key_from_netrc()
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            configured.append("wandb")
    return configured


def hf_repo_local_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")


def model_snapshot_complete(model_path: Path) -> bool:
    if not (model_path / "config.json").is_file():
        return False
    index_paths = sorted(model_path.glob("*.safetensors.index.json"))
    if index_paths:
        try:
            index = json.loads(index_paths[0].read_text(encoding="utf-8"))
            shards = set(index.get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError, AttributeError):
            return False
        return bool(shards) and all((model_path / shard).is_file() for shard in shards)
    return any(model_path.glob("*.safetensors")) or any(model_path.glob("pytorch_model*.bin"))


def download_model_snapshot(
    *,
    repo_id: str,
    target_root: Path,
    revision: str | None,
    subdir: str | None,
    cache_dir: Path | None,
) -> Path:
    normalized_subdir = (subdir or "").strip("/")
    model_path = target_root / normalized_subdir if normalized_subdir else target_root
    if model_snapshot_complete(model_path):
        log(f"Using complete local HF model snapshot: {model_path}")
        return model_path

    from huggingface_hub import snapshot_download

    target_root.mkdir(parents=True, exist_ok=True)
    log(f"Downloading HF model repo={repo_id} revision={revision or 'main'} target={target_root}")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_root),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=[f"{normalized_subdir}/**"] if normalized_subdir else None,
    )
    if not model_snapshot_complete(model_path):
        raise FileNotFoundError(f"HF model download completed but weights are incomplete: {model_path}")
    return model_path


def download_dataset_file(
    *,
    repo_id: str,
    filename: str,
    target_path: Path,
    revision: str | None,
    cache_dir: Path | None,
) -> Path:
    if target_path.is_file() and target_path.stat().st_size > 0:
        log(f"Using existing dataset file: {target_path}")
        return target_path

    from huggingface_hub import hf_hub_download

    target_path.parent.mkdir(parents=True, exist_ok=True)
    log(
        f"Downloading HF dataset repo={repo_id} file={filename} "
        f"revision={revision or 'main'} target={target_path}"
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    temporary = target_path.with_name(f".{target_path.name}.tmp-{os.getpid()}")
    try:
        os.link(downloaded, temporary)
    except OSError:
        shutil.copyfile(downloaded, temporary)
    temporary.replace(target_path)
    return target_path


def resolve_runtime_assets(args: argparse.Namespace) -> None:
    assets_dir = Path(args.hf_assets_dir).expanduser().resolve()
    cache_dir = Path(args.hf_cache_dir).expanduser().resolve() if args.hf_cache_dir else None
    component = args.prime_component

    student_repo = args.model_hf_repo
    student_needed = component in {"full", "policy_inference", "trainer_orchestrator", "trainer_worker"}
    if student_needed:
        if args.model_path and model_snapshot_complete(Path(args.model_path).expanduser()):
            args.model_path = str(Path(args.model_path).expanduser().resolve())
        else:
            if not student_repo:
                raise ValueError("--model_path or --model_hf_repo is required")
            target = (
                Path(args.model_path).expanduser()
                if args.model_path
                else assets_dir / "models" / hf_repo_local_name(student_repo)
            )
            args.model_path = str(
                download_model_snapshot(
                    repo_id=student_repo,
                    target_root=target,
                    revision=args.model_hf_revision,
                    subdir=args.model_hf_subdir,
                    cache_dir=cache_dir,
                )
            )
    elif not args.model_path:
        args.model_path = student_repo

    teacher_repo = args.prime_opd_teacher_hf_repo
    teacher_needed = args.prime_algorithm == "opd" and (
        component in {"full", "teacher_inference"}
        or (
            component in {"trainer_orchestrator", "trainer_worker"}
            and args.prime_opd_distill_mode == "full_vocab_hidden"
        )
    )
    if args.prime_algorithm == "opd" and teacher_needed:
        if args.prime_opd_teacher_model and model_snapshot_complete(
            Path(args.prime_opd_teacher_model).expanduser()
        ):
            args.prime_opd_teacher_model = str(Path(args.prime_opd_teacher_model).expanduser().resolve())
        else:
            if not teacher_repo:
                raise ValueError("--prime_opd_teacher_model or --prime_opd_teacher_hf_repo is required for OPD")
            target = (
                Path(args.prime_opd_teacher_model).expanduser()
                if args.prime_opd_teacher_model
                else assets_dir / "models" / hf_repo_local_name(teacher_repo)
            )
            args.prime_opd_teacher_model = str(
                download_model_snapshot(
                    repo_id=teacher_repo,
                    target_root=target,
                    revision=args.prime_opd_teacher_hf_revision,
                    subdir=args.prime_opd_teacher_hf_subdir,
                    cache_dir=cache_dir,
                )
            )
    elif args.prime_algorithm == "opd" and not args.prime_opd_teacher_model:
        args.prime_opd_teacher_model = teacher_repo

    dataset_needed = (
        component in {"full", "trainer_orchestrator"}
        and args.prime_env_id != "math-env"
    )
    if dataset_needed and args.dataset_hf_repo:
        if not args.dataset_hf_filename:
            raise ValueError("--dataset_hf_filename is required with --dataset_hf_repo")
        requested_path = args.prime_proof_dataset_path or args.dataset_path
        target = (
            Path(requested_path).expanduser()
            if requested_path
            else assets_dir / "data" / Path(args.dataset_hf_filename).name
        )
        resolved = download_dataset_file(
            repo_id=args.dataset_hf_repo,
            filename=args.dataset_hf_filename,
            target_path=target,
            revision=args.dataset_hf_revision,
            cache_dir=cache_dir,
        )
        args.dataset_path = str(resolved)
        if args.prime_proof_dataset_path:
            args.prime_proof_dataset_path = str(resolved)


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
    if config.get("weight_broadcast"):
        write_toml_lines(lines, "weight_broadcast", config["weight_broadcast"])
    if config.get("wandb"):
        write_toml_lines(lines, "wandb", config["wandb"])
    write_toml_lines(lines, "trainer.model", config["trainer_model"])
    write_toml_lines(lines, "trainer.optim", config["trainer_optim"])
    write_toml_lines(lines, "trainer.scheduler", config["trainer_scheduler"])
    if config.get("trainer_full_vocab_distill"):
        write_toml_lines(lines, "trainer.full_vocab_distill", config["trainer_full_vocab_distill"])
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
    if config.get("inference_model"):
        write_toml_lines(lines, "inference.model", config["inference_model"])
        write_toml_lines(lines, "inference.parallel", config["inference_parallel"])
        write_toml_lines(lines, "inference", config["inference"])
    if config.get("inference_model") and config.get("vllm_extra"):
        write_toml_lines(lines, "inference.vllm_extra", config["vllm_extra"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_prime_trainer_config(path: Path, config: dict[str, Any]) -> None:
    lines: list[str] = []
    write_toml_lines(
        lines,
        None,
        {
            "output_dir": config["root"]["output_dir"],
            "max_steps": config["root"]["max_steps"],
        },
    )
    write_toml_lines(lines, "log", config["log"])
    write_toml_lines(lines, "model", config["trainer_model"])
    write_toml_lines(lines, "tokenizer", config["tokenizer"])
    write_toml_lines(lines, "optim", config["trainer_optim"])
    write_toml_lines(lines, "scheduler", config["trainer_scheduler"])
    if config.get("trainer_full_vocab_distill"):
        write_toml_lines(lines, "full_vocab_distill", config["trainer_full_vocab_distill"])
    if config.get("trainer_ckpt"):
        write_toml_lines(lines, "ckpt", config["trainer_ckpt"])
        if config.get("trainer_ckpt_weights"):
            write_toml_lines(lines, "ckpt.weights", config["trainer_ckpt_weights"])
    if config.get("weight_broadcast"):
        write_toml_lines(lines, "weight_broadcast", config["weight_broadcast"])
    if config.get("wandb"):
        write_toml_lines(lines, "wandb", config["wandb"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_prime_orchestrator_config(path: Path, config: dict[str, Any]) -> None:
    lines: list[str] = []
    orchestrator_root = dict(config["orchestrator"])
    orchestrator_root["output_dir"] = str(Path(config["root"]["output_dir"]) / "run_default")
    write_toml_lines(lines, None, orchestrator_root)
    write_toml_lines(lines, "log", config["log"])
    write_toml_lines(lines, "tokenizer", config["tokenizer"])
    if config.get("weight_broadcast"):
        write_toml_lines(lines, "weight_broadcast", config["weight_broadcast"])
    if config.get("wandb"):
        write_toml_lines(lines, "wandb", config["wandb"])
    if config.get("orchestrator_ckpt"):
        write_toml_lines(lines, "ckpt", config["orchestrator_ckpt"])
    if config.get("orchestrator_algo"):
        write_toml_lines(lines, "algo", config["orchestrator_algo"])
    if config.get("orchestrator_algo_teacher"):
        write_toml_lines(lines, "algo.teacher", config["orchestrator_algo_teacher"])
    write_toml_lines(lines, "model", config["orchestrator_model"])
    write_toml_lines(lines, "model.client", config["orchestrator_model_client"])
    write_toml_lines(lines, "train.sampling", config["train_sampling"])
    for env_config in config["train_envs"]:
        lines.append("[[train.env]]")
        for key, value in env_config.items():
            if value is not None:
                lines.append(f"{key} = {format_toml_value(value)}")
        lines.append("")
    if config.get("eval"):
        write_toml_lines(lines, "eval", config["eval"])
        if config.get("eval_sampling"):
            write_toml_lines(lines, "eval.sampling", config["eval_sampling"])
        for env_config in config.get("eval_envs", []):
            lines.append("[[eval.env]]")
            for key, value in env_config.items():
                if value is not None:
                    lines.append(f"{key} = {format_toml_value(value)}")
            lines.append("")
    write_toml_lines(lines, "renderer", config["renderer"])
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
        dataset_mode = str(args.prime_proof_dataset_mode).strip().lower()
        single_turn_mode = dataset_mode in {"single", "single_turn", "per_turn"}
        hybrid_mode = dataset_mode in {"hybrid", "single_and_multi", "mixed_turns"}
        refine_rounds = args.prime_proof_refine_rounds
        if args.prime_algorithm == "opd" and refine_rounds == 0 and not single_turn_mode:
            refine_rounds = 1
        if single_turn_mode and args.prime_group_size != 1:
            raise ValueError(
                "--prime_proof_dataset_mode single requires --prime_group_size 1."
            )
        if single_turn_mode and args.prime_proof_candidate_gate:
            raise ValueError(
                "--prime_proof_candidate_gate is incompatible with "
                "--prime_proof_dataset_mode single."
            )
        if hybrid_mode and not args.prime_proof_multi_turn_dataset_path:
            raise ValueError(
                "--prime_proof_multi_turn_dataset_path is required with "
                "--prime_proof_dataset_mode hybrid."
            )
        if not 0.0 <= args.prime_proof_multi_turn_fraction <= 1.0:
            raise ValueError("--prime_proof_multi_turn_fraction must be in [0, 1].")
        if not 0.0 <= args.prime_proof_multi_turn_continue_fraction <= 1.0:
            raise ValueError(
                "--prime_proof_multi_turn_continue_fraction must be in [0, 1]."
            )
        if args.prime_proof_candidate_gate:
            if args.prime_group_size <= 1:
                raise ValueError(
                    "--prime_proof_candidate_gate requires --prime_group_size greater than 1."
                )
            if args.prime_proof_candidate_continue_count > args.prime_group_size:
                raise ValueError(
                    "--prime_proof_candidate_continue_count cannot exceed --prime_group_size."
                )
        return {
            "id": effective_env_id,
            "name": effective_env_name if effective_env_name != "math" else "proof_math",
            "args": {
                "dataset_path": dataset_path,
                "dataset_mode": dataset_mode,
                "multi_turn_dataset_path": args.prime_proof_multi_turn_dataset_path,
                "multi_turn_fraction": args.prime_proof_multi_turn_fraction,
                "multi_turn_continue_fraction": args.prime_proof_multi_turn_continue_fraction,
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
                "selector_top_k": args.prime_proof_selector_top_k,
                "selector_enabled": args.prime_proof_enable_selector,
                "candidate_gate_enabled": args.prime_proof_candidate_gate,
                "candidate_continue_count": args.prime_proof_candidate_continue_count,
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


def resolve_prime_token_batch_size(args: argparse.Namespace) -> int | None:
    explicit_tokens = args.prime_token_batch_size
    packed_sequences = int(args.prime_packed_sequences_per_step)
    if explicit_tokens is not None and packed_sequences > 0:
        raise ValueError(
            "Set only one of --prime_token_batch_size or "
            "--prime_packed_sequences_per_step."
        )
    if explicit_tokens is not None:
        return int(explicit_tokens)
    if packed_sequences <= 0:
        return None
    return int(args.max_seq_length) * packed_sequences


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
    prime_vllm_quantization = normalize_optional_choice(args.prime_vllm_quantization)
    if prime_vllm_quantization:
        vllm_extra["quantization"] = prime_vllm_quantization
    if args.prime_vllm_max_num_seqs is not None:
        vllm_extra["max_num_seqs"] = args.prime_vllm_max_num_seqs
    if args.prime_vllm_max_num_batched_tokens is not None:
        vllm_extra["max_num_batched_tokens"] = args.prime_vllm_max_num_batched_tokens

    weight_broadcast: dict[str, Any] = {
        "type": args.prime_weight_broadcast_type,
    }
    if args.prime_weight_broadcast_type == "nccl":
        weight_broadcast.update(
            {
                "port": args.prime_weight_broadcast_port,
                "timeout": args.prime_weight_broadcast_timeout,
                "quantize_in_weight_transfer": args.prime_weight_broadcast_quantize_in_weight_transfer,
            }
        )

    wandb_config: dict[str, Any] | None = None
    if args.with_tracking and args.wandb_mode != "disabled":
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_name,
        }

    token_batch_size = resolve_prime_token_batch_size(args)
    if token_batch_size is not None and args.prime_max_inflight_rollouts is None:
        raise ValueError(
            "--prime_max_inflight_rollouts is required when token-based batching is enabled."
        )
    if token_batch_size is not None and args.prime_oversampling_factor is not None:
        raise ValueError(
            "--prime_oversampling_factor is only valid with rollout-count batching."
        )

    orchestrator_config = {
        "batch_size": args.prime_batch_size if token_batch_size is None else None,
        "token_batch_size": token_batch_size,
        "group_size": args.prime_group_size,
        "seq_len": args.max_seq_length,
        "max_steps": args.max_train_steps,
        "max_inflight_rollouts": args.prime_max_inflight_rollouts,
        "max_inflight_questions": args.prime_max_inflight_questions,
        "max_off_policy_steps": args.prime_max_off_policy_steps,
        "oversampling_factor": (
            args.prime_oversampling_factor if token_batch_size is None else None
        ),
    }
    if args.prime_disable_zero_advantage_filter:
        orchestrator_config["post_batch_filters"] = [
            {"type": "gibberish", "enforce": False},
            {"type": "repetition", "enforce": False},
            {"type": "zero_advantage", "enforce": False},
        ]

    orchestrator_algo: dict[str, Any] | None = None
    orchestrator_algo_teacher: dict[str, Any] | None = None
    trainer_full_vocab_distill: dict[str, Any] | None = None
    if args.prime_algorithm != "grpo":
        orchestrator_algo = {"type": args.prime_algorithm}
    if args.prime_algorithm == "opd":
        orchestrator_algo = {"type": "opd"}
        orchestrator_algo_teacher = {
            "name": args.prime_opd_teacher_model,
            "base_url": [args.prime_opd_teacher_base_url or f"http://localhost:{args.prime_opd_teacher_port}/v1"],
            "skip_model_check": args.prime_opd_teacher_skip_model_check,
        }
        if args.prime_opd_distill_mode == "full_vocab_hidden":
            orchestrator_algo.update(
                {
                    "distill_mode": args.prime_opd_distill_mode,
                    "teacher_hidden_dtype": args.prime_opd_full_vocab_teacher_hidden_dtype,
                    "teacher_hidden_transport": args.prime_opd_full_vocab_hidden_transport,
                    "teacher_hidden_codec": args.prime_opd_full_vocab_hidden_codec,
                }
            )
            if args.prime_opd_full_vocab_hidden_transport == "filesystem":
                if not args.prime_opd_full_vocab_hidden_path:
                    raise ValueError(
                        "--prime_opd_full_vocab_hidden_path is required when filesystem hidden transport is enabled"
                    )
                orchestrator_algo["teacher_hidden_path"] = args.prime_opd_full_vocab_hidden_path
            trainer_full_vocab_distill = {
                "enabled": True,
                "teacher_lm_head_path": args.prime_opd_full_vocab_teacher_lm_head_path
                or args.prime_opd_teacher_model,
                "teacher_lm_head_key": args.prime_opd_full_vocab_teacher_lm_head_key,
                "token_chunk_size": args.prime_opd_full_vocab_token_chunk_size,
                "vocab_chunk_size": args.prime_opd_full_vocab_vocab_chunk_size,
                "teacher_hidden_dtype": args.prime_opd_full_vocab_teacher_hidden_dtype,
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

    trainer_scheduler: dict[str, Any] = {"type": args.prime_lr_scheduler}
    if args.prime_lr_scheduler in {"linear", "cosine"}:
        trainer_scheduler.update(
            {
                "warmup_steps": args.prime_lr_warmup_steps,
                "min_lr": args.prime_lr_min,
            }
        )
    if args.prime_lr_scheduler == "linear":
        trainer_scheduler["decay_steps"] = args.prime_lr_decay_steps

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

    policy_client_config: dict[str, Any] = {
        "skip_model_check": args.prime_skip_model_check,
    }
    policy_base_urls = parse_csv_list(args.prime_policy_base_url)
    if policy_base_urls:
        policy_client_config["base_url"] = policy_base_urls
        policy_client_config["dp_rank_count"] = args.prime_policy_dp_rank_count or args.prime_vllm_data_parallel_size
    policy_admin_base_urls = parse_csv_list(args.prime_policy_admin_base_url)
    if policy_admin_base_urls:
        policy_client_config["admin_base_url"] = policy_admin_base_urls

    include_local_inference = args.prime_component == "full" and args.prime_infer_gpus > 0

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
            "num_infer_gpus": args.prime_infer_gpus if include_local_inference else 0,
        },
        "model": {
            "name": args.model_path,
        },
        "tokenizer": {
            "name": args.tokenizer_path or args.model_path,
            "trust_remote_code": True,
        },
        "weight_broadcast": weight_broadcast,
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
            "dp_replicate": args.prime_trainer_dp_replicate,
            "cp": args.prime_trainer_context_parallel_size,
            "cp_style": args.prime_trainer_cp_style,
            "fp8": args.prime_trainer_fp8,
            "compile": None if args.prime_trainer_compile else "None",
        },
        "trainer_optim": trainer_optim,
        "trainer_scheduler": trainer_scheduler,
        "trainer_full_vocab_distill": trainer_full_vocab_distill,
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
        "orchestrator_model_client": policy_client_config,
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
        }
        if include_local_inference
        else None,
        "inference_parallel": {
            "tp": args.prime_vllm_tensor_parallel_size,
            "dp": args.prime_vllm_data_parallel_size,
        }
        if include_local_inference
        else None,
        "inference": {
            "gpu_memory_utilization": args.prime_vllm_gpu_memory_utilization,
            "api_server_count": args.prime_vllm_api_server_count,
            "data_parallel_rpc_port": args.prime_vllm_data_parallel_rpc_port,
            "enable_prefix_caching": args.prime_vllm_enable_prefix_caching,
            "use_deep_gemm": args.prime_vllm_use_deep_gemm,
            "enable_fp32_lm_head": args.prime_vllm_enable_fp32_lm_head,
        }
        if include_local_inference
        else None,
        "vllm_extra": vllm_extra,
    }


def build_prime_policy_inference_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    vllm_extra = parse_extra_json(args.prime_vllm_extra)
    prime_vllm_quantization = normalize_optional_choice(args.prime_vllm_quantization)
    if prime_vllm_quantization:
        vllm_extra["quantization"] = prime_vllm_quantization
    if args.prime_vllm_max_num_seqs is not None:
        vllm_extra["max_num_seqs"] = args.prime_vllm_max_num_seqs
    if args.prime_vllm_max_num_batched_tokens is not None:
        vllm_extra["max_num_batched_tokens"] = args.prime_vllm_max_num_batched_tokens

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
            "port": args.prime_policy_port,
        },
        "model": {
            "name": args.model_path,
            "dtype": args.prime_vllm_dtype,
            "max_model_len": args.prime_vllm_max_model_len,
            "enforce_eager": args.prime_vllm_enforce_eager,
            "trust_remote_code": True,
            "tool_call_parser": None,
            "reasoning_parser": args.prime_vllm_reasoning_parser,
        },
        "parallel": {
            "tp": args.prime_vllm_tensor_parallel_size,
            "dp": args.prime_vllm_data_parallel_size,
        },
        "inference": {
            "gpu_memory_utilization": args.prime_vllm_gpu_memory_utilization,
            "api_server_count": args.prime_vllm_api_server_count,
            "data_parallel_rpc_port": args.prime_vllm_data_parallel_rpc_port,
            "enable_prefix_caching": args.prime_vllm_enable_prefix_caching,
            "use_deep_gemm": args.prime_vllm_use_deep_gemm,
            "enable_fp32_lm_head": args.prime_vllm_enable_fp32_lm_head,
        },
        "vllm_extra": vllm_extra,
    }


def build_prime_teacher_inference_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    vllm_extra = parse_extra_json(args.prime_opd_teacher_vllm_extra)
    teacher_quantization = normalize_optional_choice(args.prime_opd_teacher_vllm_quantization)
    if teacher_quantization:
        vllm_extra["quantization"] = teacher_quantization
    if args.prime_opd_teacher_vllm_max_num_seqs is not None:
        vllm_extra["max_num_seqs"] = args.prime_opd_teacher_vllm_max_num_seqs
    if args.prime_opd_teacher_vllm_max_num_batched_tokens is not None:
        vllm_extra["max_num_batched_tokens"] = args.prime_opd_teacher_vllm_max_num_batched_tokens

    model_config: dict[str, Any] = {
        "name": args.prime_opd_teacher_model,
        "dtype": args.prime_opd_teacher_vllm_dtype,
        "max_model_len": args.prime_opd_teacher_vllm_max_model_len,
        "enforce_eager": args.prime_opd_teacher_vllm_enforce_eager,
        "trust_remote_code": True,
        "tool_call_parser": None,
        "reasoning_parser": args.prime_opd_teacher_vllm_reasoning_parser,
    }
    if args.prime_opd_teacher_tokenizer_path:
        model_config["tokenizer"] = args.prime_opd_teacher_tokenizer_path

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
        "model": model_config,
        "parallel": {
            "tp": args.prime_opd_teacher_vllm_tensor_parallel_size,
            "dp": args.prime_opd_teacher_vllm_data_parallel_size,
        },
        "inference": {
            "gpu_memory_utilization": args.prime_opd_teacher_vllm_gpu_memory_utilization,
            "api_server_count": args.prime_opd_teacher_vllm_api_server_count,
            "data_parallel_rpc_port": args.prime_opd_teacher_vllm_data_parallel_rpc_port,
            "enable_prefix_caching": args.prime_opd_teacher_vllm_enable_prefix_caching,
            "use_deep_gemm": args.prime_opd_teacher_vllm_use_deep_gemm,
            "enable_fp32_lm_head": args.prime_opd_teacher_vllm_enable_fp32_lm_head,
        },
        "vllm_extra": vllm_extra,
    }


def run_logged_subprocess(
    command: list[str],
    env: dict[str, str],
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> int:
    log("Running command: " + " ".join(shlex.quote(part) for part in command))
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Subprocess log: {log_path}")
        log_file = log_path.open("a", encoding="utf-8")
        print(
            f"\n[{datetime.now().isoformat()}] Running command: "
            + " ".join(shlex.quote(part) for part in command),
            file=log_file,
            flush=True,
        )
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
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_file is not None:
                print(line, end="", file=log_file, flush=True)
        return_code = process.wait()
        if log_file is not None:
            print(f"[{datetime.now().isoformat()}] exit_status={return_code}", file=log_file, flush=True)
        return return_code
    finally:
        if log_file is not None:
            log_file.close()


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


def start_logged_process(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("Starting command: " + " ".join(shlex.quote(part) for part in command))
    log(f"Subprocess log: {log_path}")
    log_file = log_path.open("a", encoding="utf-8")
    print(
        f"\n[{datetime.now().isoformat()}] Starting command: "
        + " ".join(shlex.quote(part) for part in command),
        file=log_file,
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_file


def stop_process_group(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    log(f"Stopping {name} process group pid={process.pid}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log(f"Killing {name} process group pid={process.pid}")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def build_distributed_trainer_command(args: argparse.Namespace, trainer_config_path: Path) -> list[str]:
    if args.prime_trainer_num_nodes < 1:
        raise ValueError("--prime_trainer_num_nodes must be at least 1")
    if not 0 <= args.prime_trainer_node_rank < args.prime_trainer_num_nodes:
        raise ValueError(
            "--prime_trainer_node_rank must be in [0, --prime_trainer_num_nodes); "
            f"got rank={args.prime_trainer_node_rank} nodes={args.prime_trainer_num_nodes}"
        )
    if not args.prime_trainer_master_addr:
        raise ValueError("--prime_trainer_master_addr is required for multi-node trainer mode")
    rdzv_id = args.prime_trainer_rdzv_id or os.environ.get("OLMO_RUN_DIR_NAME") or "prime-opd"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.prime_trainer_num_nodes),
        "--nproc-per-node",
        str(args.prime_train_gpus),
        "--node-rank",
        str(args.prime_trainer_node_rank),
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"{args.prime_trainer_master_addr}:{args.prime_trainer_master_port}",
        "--rdzv-id",
        rdzv_id,
        "--rdzv-conf",
        (
            f"timeout={args.prime_trainer_rdzv_timeout},"
            f"read_timeout={args.prime_trainer_rdzv_timeout},"
            f"is_host={'true' if args.prime_trainer_node_rank == 0 else 'false'}"
        ),
        "-m",
        "prime_rl.trainer.rl.train",
        "@",
        str(trainer_config_path),
    ]


def run_distributed_trainer_component(
    args: argparse.Namespace,
    config: dict[str, Any],
    log_dir: Path,
) -> int:
    trainer_config_path = log_dir / "trainer.toml"
    write_prime_trainer_config(trainer_config_path, config)
    trainer_command = build_distributed_trainer_command(args, trainer_config_path)
    env = os.environ.copy()
    env.setdefault("WANDB_SHARED_MODE", "1")
    env["WANDB_SHARED_RUN_ID"] = resolve_wandb_shared_run_id(env, log_dir)
    if args.prime_trainer_node_rank != 0:
        env["WANDB_SHARED_LABEL"] = "trainer"
        return run_logged_subprocess(
            trainer_command,
            env,
            log_path=log_dir / f"trainer_node_{args.prime_trainer_node_rank}.log",
        )

    orchestrator_config_path = log_dir / "orchestrator.toml"
    write_prime_orchestrator_config(orchestrator_config_path, config)
    orchestrator_entrypoint = shutil.which("orchestrator")
    orchestrator_command = (
        [orchestrator_entrypoint, "@", str(orchestrator_config_path)]
        if orchestrator_entrypoint
        else [sys.executable, "-m", "prime_rl.entrypoints.orchestrator", "@", str(orchestrator_config_path)]
    )
    trainer_env = dict(env)
    trainer_env["WANDB_SHARED_LABEL"] = "trainer"
    orchestrator_env = dict(env)
    orchestrator_env["WANDB_SHARED_LABEL"] = "orchestrator"
    trainer_process = orchestrator_process = None
    trainer_log = orchestrator_log = None
    try:
        trainer_process, trainer_log = start_logged_process(
            trainer_command,
            trainer_env,
            log_dir / "trainer_node_0.log",
        )
        orchestrator_process, orchestrator_log = start_logged_process(
            orchestrator_command,
            orchestrator_env,
            log_dir / "orchestrator.log",
        )
        while True:
            trainer_status = trainer_process.poll()
            orchestrator_status = orchestrator_process.poll()
            if trainer_status is not None:
                log(f"Distributed trainer exited with status {trainer_status}")
                return trainer_status
            if orchestrator_status is not None:
                log(f"Orchestrator exited with status {orchestrator_status}")
                return orchestrator_status
            time.sleep(2)
    finally:
        stop_process_group(orchestrator_process, "orchestrator")
        stop_process_group(trainer_process, "distributed trainer")
        if orchestrator_log is not None:
            orchestrator_log.close()
        if trainer_log is not None:
            trainer_log.close()


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


def wait_for_external_http_ready(
    url: str,
    *,
    label: str,
    timeout_s: int,
    poll_s: float = 5.0,
    log_every_s: float = 30.0,
) -> None:
    """Wait for an externally managed Prime-RL service before launching rl.

    In multi-node role-gated runs the policy and teacher inference services are
    started by separate operator commands. Waiting here keeps the long-lived
    trainer role visible as a Python process with periodic logs, instead of
    blocking silently in the shell wrapper before train.py starts.
    """

    deadline = time.monotonic() + max(1, timeout_s)
    last_log = 0.0
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    log(f"{label} ready: {url}")
                    return
                last_error = f"HTTP {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except TimeoutError as exc:
            last_error = str(exc)

        now = time.monotonic()
        if now - last_log >= log_every_s:
            remaining = max(0, int(deadline - now))
            log(f"Waiting for {label}: {url} remaining_s={remaining} last_error={last_error}")
            last_log = now
        time.sleep(max(0.5, poll_s))

    raise TimeoutError(f"Timed out waiting for {label}: {url}; last_error={last_error}")


def start_teacher_inference(
    args: argparse.Namespace,
    log_dir: Path,
    *,
    wait_until_ready: bool = True,
) -> subprocess.Popen | None:
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
    if wait_until_ready:
        wait_for_http_ready(ready_url, process, teacher_log_path, args.prime_opd_teacher_ready_timeout)
        log(f"OPD teacher inference ready: {ready_url}")
    else:
        log(
            "OPD teacher inference launched asynchronously; policy startup and rollout generation "
            f"will overlap teacher readiness at {ready_url}"
        )
    return process


def run_prime_inference_component(args: argparse.Namespace, log_dir: Path, component: str) -> int:
    inference_entrypoint = shutil.which("inference")
    command_prefix = [inference_entrypoint] if inference_entrypoint else [sys.executable, "-m", "prime_rl.entrypoints.inference"]

    if component == "policy_inference":
        inference_dir = log_dir / "policy_inference"
        config = build_prime_policy_inference_config(args, inference_dir)
        config_path = inference_dir / "policy_inference.toml"
        label = "policy"
        gpu_ids = args.prime_policy_gpu_ids
    elif component == "teacher_inference":
        inference_dir = log_dir / "teacher_inference"
        config = build_prime_teacher_inference_config(args, inference_dir)
        config_path = inference_dir / "teacher_inference.toml"
        label = "teacher"
        gpu_ids = args.prime_opd_teacher_gpu_ids
    else:
        raise ValueError(f"Unsupported inference component {component!r}")

    write_prime_inference_config(config_path, config)
    log(f"Prime-RL {label} inference config written: {config_path}")
    log(f"Prime-RL {label} inference log_dir={inference_dir}")
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        log(f"Starting {label} inference on GPU(s) {gpu_ids}")
    command = [*command_prefix, "@", str(config_path)]
    max_restarts = (
        max(0, int(os.environ.get("PRIME_RL_TEACHER_INFERENCE_MAX_RESTARTS", "3")))
        if component == "teacher_inference"
        else 0
    )
    restart_delay_s = max(0.0, float(os.environ.get("PRIME_RL_TEACHER_INFERENCE_RESTART_DELAY_SECONDS", "10")))
    start = time.monotonic()
    restart_count = 0
    while True:
        return_code = run_logged_subprocess(command, env, log_path=inference_dir / f"{label}_inference.log")
        duration_s = time.monotonic() - start
        log(f"Prime-RL {label} inference exit status: {return_code} duration_s={duration_s:.1f}")
        if return_code == 0 or restart_count >= max_restarts:
            return return_code
        restart_count += 1
        log(
            f"Restarting Prime-RL {label} inference after nonzero exit "
            f"restart={restart_count}/{max_restarts} delay_s={restart_delay_s:g}"
        )
        time.sleep(restart_delay_s)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Prime-RL training entrypoint for OLMo3Sink smoke tests")
    parser.add_argument("--backend", default="prime_rl")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_hf_repo", default=DEFAULT_STUDENT_HF_REPO)
    parser.add_argument("--model_hf_revision", default=None)
    parser.add_argument("--model_hf_subdir", default=None)
    parser.add_argument(
        "--hf_assets_dir",
        default=os.environ.get("PRIME_HF_ASSETS_DIR", "/tmp/aimo-proof-pilot-assets"),
        help="Root used to materialize model snapshots and dataset files supplied as HF repos.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=os.environ.get("HF_HOME"),
        help="Optional huggingface_hub cache directory used during automatic downloads.",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--dataset_hf_repo", default=DEFAULT_DATASET_HF_REPO)
    parser.add_argument("--dataset_hf_filename", default=DEFAULT_DATASET_HF_FILENAME)
    parser.add_argument("--dataset_hf_revision", default=None)
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
    parser.add_argument("--prime_lr_scheduler", default="constant", choices=("constant", "linear", "cosine"))
    parser.add_argument("--prime_lr_warmup_steps", type=int, default=10)
    parser.add_argument("--prime_lr_decay_steps", type=int, default=10)
    parser.add_argument("--prime_lr_min", type=float, default=0.0)
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
    parser.add_argument(
        "--prime_component",
        default="full",
        choices=("full", "trainer_orchestrator", "trainer_worker", "policy_inference", "teacher_inference"),
        help=(
            "Prime-RL component to run. 'full' preserves the existing single-node all-in-one launcher. "
            "Use policy_inference, teacher_inference, trainer_orchestrator, and trainer_worker "
            "for manual multi-node role-gated runs."
        ),
    )
    parser.add_argument("--prime_trainer_num_nodes", type=int, default=1)
    parser.add_argument("--prime_trainer_node_rank", type=int, default=0)
    parser.add_argument("--prime_trainer_master_addr", default=None)
    parser.add_argument("--prime_trainer_master_port", type=int, default=29400)
    parser.add_argument("--prime_trainer_rdzv_id", default=None)
    parser.add_argument("--prime_trainer_rdzv_timeout", type=int, default=7200)
    parser.add_argument("--prime_algorithm", default="grpo", choices=("grpo", "opd"))
    parser.add_argument("--prime_env_id", default="math-env")
    parser.add_argument("--prime_env_name", default="math")
    parser.add_argument("--prime_math_dataset_name", default="mikasenghaas/Sanity-Test-R1D-1.5B")
    parser.add_argument("--prime_math_dataset_subset", default="default")
    parser.add_argument("--prime_math_dataset_split", default=None)
    parser.add_argument("--prime_math_min_avg_reward", type=float, default=None)
    parser.add_argument("--prime_math_max_avg_reward", type=float, default=None)
    parser.add_argument("--prime_proof_dataset_path", default=None)
    parser.add_argument("--prime_proof_multi_turn_dataset_path", default=None)
    parser.add_argument(
        "--prime_proof_dataset_mode",
        default="mixed",
        choices=(
            "mixed",
            "single",
            "single_turn",
            "per_turn",
            "hybrid",
            "single_and_multi",
            "mixed_turns",
            "verifiable",
            "verifiable_eval",
            "eval_verifiable",
        ),
    )
    parser.add_argument("--prime_proof_multi_turn_fraction", type=float, default=0.2)
    parser.add_argument(
        "--prime_proof_multi_turn_continue_fraction",
        type=float,
        default=0.25,
    )
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
    parser.add_argument("--prime_proof_selector_top_k", type=int, default=3)
    parser.add_argument("--prime_proof_enable_selector", type=parse_bool, default=True)
    parser.add_argument("--prime_proof_candidate_gate", type=parse_bool, default=False)
    parser.add_argument("--prime_proof_candidate_continue_count", type=int, default=4)
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
    parser.add_argument(
        "--prime_token_batch_size",
        type=int,
        default=None,
        help=(
            "Global prompt-plus-completion token target per optimizer step. This selects "
            "Prime-RL token batching and is mutually exclusive with "
            "--prime_packed_sequences_per_step."
        ),
    )
    parser.add_argument(
        "--prime_packed_sequences_per_step",
        type=int,
        default=0,
        help=(
            "Convenience target for Prime-RL token batching. A positive value sets "
            "token_batch_size=max_seq_length*value; 0 preserves rollout-count batching."
        ),
    )
    parser.add_argument("--prime_max_inflight_rollouts", type=int, default=None)
    parser.add_argument(
        "--prime_max_inflight_questions",
        type=int,
        default=None,
        help=(
            "Cap concurrently active question groups without blocking continuation "
            "turns inside existing proof/verifier/meta/refine rollouts."
        ),
    )
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
    parser.add_argument(
        "--prime_policy_base_url",
        default=None,
        help=(
            "External policy inference base URL(s) for trainer_orchestrator mode, "
            "e.g. http://host:8000/v1 or comma-separated URLs for multiple policy nodes."
        ),
    )
    parser.add_argument(
        "--prime_policy_admin_base_url",
        default=None,
        help=(
            "Optional external policy admin base URL(s). Defaults to policy base URL(s) "
            "inside Prime-RL when unset."
        ),
    )
    parser.add_argument("--prime_policy_dp_rank_count", type=int, default=None)
    parser.add_argument(
        "--prime_weight_broadcast_type",
        default="filesystem",
        choices=("filesystem", "nccl"),
        help=(
            "Prime-RL policy weight-transfer backend. Use 'nccl' to broadcast trainer "
            "weights directly to vLLM workers instead of writing filesystem snapshots."
        ),
    )
    parser.add_argument("--prime_weight_broadcast_port", type=int, default=29501)
    parser.add_argument("--prime_weight_broadcast_timeout", type=int, default=3600)
    parser.add_argument(
        "--prime_weight_broadcast_quantize_in_weight_transfer",
        type=parse_bool,
        default=False,
        help=(
            "Use Prime-RL's quantized NCCL weight-transfer path. Requires "
            "--prime_weight_broadcast_type nccl and custom trainer model support."
        ),
    )
    parser.add_argument("--prime_temperature", type=float, default=0.7)
    parser.add_argument("--prime_top_p", type=float, default=0.95)
    parser.add_argument("--prime_sampling_extra_body", default=None)
    parser.add_argument("--prime_train_gpus", type=int, default=1)
    parser.add_argument("--prime_infer_gpus", type=int, default=1)
    parser.add_argument("--prime_gpus_per_node", type=int, default=2)
    parser.add_argument("--prime_trainer_model_impl", default="hf", choices=("hf", "custom", "auto"))
    parser.add_argument("--prime_trainer_attn", default="olmo3_sink_fa2")
    parser.add_argument("--prime_trainer_fsdp_cpu_offload", type=parse_bool, default=False)
    parser.add_argument("--prime_trainer_optim_cpu_offload", type=parse_bool, default=True)
    parser.add_argument("--prime_trainer_optimization_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--prime_trainer_reduce_dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument(
        "--prime_trainer_dp_replicate",
        type=int,
        default=1,
        help=(
            "Prime-RL trainer model.dp_replicate degree. Set this to the trainer-node count "
            "to use one intra-node FSDP shard group per replicated HSDP island."
        ),
    )
    parser.add_argument("--prime_trainer_context_parallel_size", "--prime_trainer_cp", type=int, default=1)
    parser.add_argument("--prime_trainer_cp_style", default="ulysses", choices=("ring", "ulysses"))
    parser.add_argument("--prime_trainer_fp8", type=parse_bool, default=False)
    parser.add_argument(
        "--prime_trainer_compile",
        type=parse_bool,
        default=True,
        help=(
            "Enable Prime-RL trainer torch.compile. Disable only for smoke tests or "
            "trainer kernels that cannot be traced; vLLM inference remains compiled unless "
            "its own eager flag is set."
        ),
    )
    parser.add_argument("--prime_renderer_reasoning_parser", default="think")
    parser.add_argument("--prime_vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--prime_vllm_data_parallel_size", type=int, default=1)
    parser.add_argument("--prime_policy_port", type=int, default=8000)
    parser.add_argument("--prime_policy_gpu_ids", default=None)
    parser.add_argument("--prime_vllm_max_model_len", type=int, default=8192)
    parser.add_argument("--prime_vllm_dtype", default="bfloat16")
    parser.add_argument("--prime_vllm_enforce_eager", type=parse_bool, default=False)
    parser.add_argument("--prime_vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prime_vllm_api_server_count", type=int, default=1)
    parser.add_argument("--prime_vllm_data_parallel_rpc_port", type=int, default=13345)
    parser.add_argument("--prime_vllm_enable_prefix_caching", type=parse_bool, default=True)
    parser.add_argument("--prime_vllm_use_deep_gemm", type=parse_bool, default=False)
    parser.add_argument(
        "--prime_vllm_enable_fp32_lm_head",
        type=parse_bool,
        default=False,
        help="Use Prime-RL's optional FP32 policy LM-head projection instead of BF16.",
    )
    parser.add_argument("--prime_vllm_quantization", default=None)
    parser.add_argument("--prime_vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--prime_vllm_max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--prime_vllm_reasoning_parser", default="deepseek_v4")
    parser.add_argument("--prime_vllm_extra", default=None)
    parser.add_argument("--prime_opd_teacher_model", default=None)
    parser.add_argument("--prime_opd_teacher_hf_repo", default=DEFAULT_TEACHER_HF_REPO)
    parser.add_argument("--prime_opd_teacher_hf_revision", default=None)
    parser.add_argument("--prime_opd_teacher_hf_subdir", default=None)
    parser.add_argument("--prime_opd_teacher_tokenizer_path", default=None)
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
    parser.add_argument("--prime_opd_teacher_vllm_data_parallel_rpc_port", type=int, default=13345)
    parser.add_argument("--prime_opd_teacher_vllm_enable_prefix_caching", type=parse_bool, default=True)
    parser.add_argument("--prime_opd_teacher_vllm_use_deep_gemm", type=parse_bool, default=False)
    parser.add_argument(
        "--prime_opd_teacher_vllm_enable_fp32_lm_head",
        type=parse_bool,
        default=False,
        help="Use Prime-RL's optional FP32 teacher LM-head projection instead of BF16.",
    )
    parser.add_argument("--prime_opd_teacher_vllm_quantization", default=None)
    parser.add_argument("--prime_opd_teacher_vllm_max_num_seqs", type=int, default=None)
    parser.add_argument("--prime_opd_teacher_vllm_max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--prime_opd_teacher_vllm_reasoning_parser", default="deepseek_v4")
    parser.add_argument("--prime_opd_teacher_vllm_extra", default=None)
    parser.add_argument(
        "--prime_opd_distill_mode",
        default="token_logprobs",
        choices=("token_logprobs", "full_vocab_hidden"),
        help=(
            "OPD teacher signal. token_logprobs keeps the legacy sampled-token reverse-KL path. "
            "full_vocab_hidden requests teacher hidden states and computes full-vocab reverse KL in the trainer."
        ),
    )
    parser.add_argument("--prime_opd_full_vocab_teacher_lm_head_path", default=None)
    parser.add_argument("--prime_opd_full_vocab_teacher_lm_head_key", default=None)
    parser.add_argument("--prime_opd_full_vocab_teacher_hidden_dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--prime_opd_full_vocab_hidden_transport",
        default="inline",
        choices=("inline", "filesystem"),
        help=(
            "Transport for full-vocab teacher hidden states. filesystem writes on the teacher and sends "
            "only shared-filesystem references through the orchestrator/packer."
        ),
    )
    parser.add_argument(
        "--prime_opd_full_vocab_hidden_path",
        default=None,
        help="Absolute shared directory visible to teacher, orchestrator, packer, and trainer.",
    )
    parser.add_argument(
        "--prime_opd_full_vocab_hidden_codec",
        default="raw",
        choices=("raw", "had_int6_blk32"),
        help="Compact selected-row teacher hidden-state representation for filesystem full-vocab OPD.",
    )
    parser.add_argument("--prime_opd_full_vocab_token_chunk_size", type=int, default=64)
    parser.add_argument("--prime_opd_full_vocab_vocab_chunk_size", type=int, default=8192)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--wandb_project", default="olmo3-prime-rl")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument(
        "--prime_wandb_samples_interval",
        type=int,
        default=1,
        help="Prime-RL W&B train sample table logging interval. Default logs every step.",
    )
    parser.add_argument(
        "--prime_wandb_samples_ratio",
        type=float,
        default=None,
        help="Optional Prime-RL W&B train sample sampling ratio. None uses Prime-RL defaults.",
    )
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
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    resolve_runtime_assets(args)

    run_dir_name = os.environ.get("OLMO_RUN_DIR_NAME")
    output_run_dir_name = os.environ.get("OLMO_OUTPUT_RUN_DIR_NAME", run_dir_name)
    log_run_dir_name = os.environ.get("OLMO_LOG_RUN_DIR_NAME", run_dir_name)
    output_dir = make_run_dir(args.output_path, "/tmp/olmo3_prime_rl/output", output_run_dir_name)
    log_dir = make_run_dir(args.logdir, "/tmp/olmo3_prime_rl/logs", log_run_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    prime_rl_dir = Path(args.prime_rl_dir or os.environ.get("PRIME_RL_DIR", "")).expanduser()
    if args.prime_rl_dir:
        os.environ["PRIME_RL_DIR"] = str(prime_rl_dir)
    if prime_rl_dir and str(prime_rl_dir) != "." and prime_rl_dir.exists():
        os.environ["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys(
                [
                    *prime_rl_pythonpath_entries(prime_rl_dir),
                    *os.environ.get("PYTHONPATH", "").split(os.pathsep),
                ]
            )
        )

    os.environ.setdefault("PRIME_RL_OLMO3_SINK", "1")
    os.environ.setdefault("VERL_RLCSD_OLMO3_SINK", "1")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    configure_cuda_toolchain_env()
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    enable_system_nccl_preload_for_transformer_engine(args)

    if args.prime_component in ("policy_inference", "teacher_inference"):
        return run_prime_inference_component(args, log_dir, args.prime_component)

    if args.prime_component == "trainer_orchestrator":
        if not args.prime_policy_base_url:
            raise ValueError("--prime_policy_base_url is required for --prime_component trainer_orchestrator")
        if args.prime_algorithm == "opd" and not args.prime_opd_teacher_base_url:
            raise ValueError("--prime_opd_teacher_base_url is required for OPD trainer_orchestrator mode")
        policy_timeout = int(os.environ.get("PRIME_POLICY_READY_TIMEOUT", "7200"))
        for policy_base_url in parse_csv_list(args.prime_policy_base_url):
            policy_ready_url = policy_base_url.rstrip("/") + "/models"
            wait_for_external_http_ready(policy_ready_url, label=f"policy inference {policy_base_url}", timeout_s=policy_timeout)
        if args.prime_algorithm == "opd":
            teacher_ready_url = args.prime_opd_teacher_base_url.rstrip("/") + "/models"
            teacher_timeout = int(os.environ.get("PRIME_TEACHER_READY_TIMEOUT", str(args.prime_opd_teacher_ready_timeout)))
            wait_for_external_http_ready(teacher_ready_url, label="OPD teacher inference", timeout_s=teacher_timeout)
        args.prime_opd_start_teacher = False

    config_path = log_dir / "prime_rl.toml"
    config = build_prime_rl_config(args, output_dir)
    write_prime_rl_config(config_path, config)
    log(f"Prime-RL config written: {config_path}")
    log(f"output_dir={output_dir}")
    log(f"log_dir={log_dir}")
    orchestrator_config = config["orchestrator"]
    if orchestrator_config.get("token_batch_size") is not None:
        log(
            "Prime-RL token batching enabled after complete environment rollouts: "
            f"token_batch_size={orchestrator_config['token_batch_size']} "
            f"seq_len={args.max_seq_length} "
            f"packed_sequences_target={args.prime_packed_sequences_per_step or 'explicit_tokens'}"
        )
    else:
        log(
            "Prime-RL rollout batching enabled: "
            f"batch_size={orchestrator_config.get('batch_size')}"
        )
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

    if args.prime_component in {"trainer_orchestrator", "trainer_worker"} and args.prime_trainer_num_nodes > 1:
        log(
            "Starting distributed Prime-RL trainer component: "
            f"node_rank={args.prime_trainer_node_rank}/{args.prime_trainer_num_nodes} "
            f"gpus_per_node={args.prime_train_gpus} "
            f"rdzv={args.prime_trainer_master_addr}:{args.prime_trainer_master_port}"
        )
        return run_distributed_trainer_component(args, config, log_dir)

    rl_entrypoint = shutil.which("rl")
    command = [rl_entrypoint, "@", str(config_path)] if rl_entrypoint else [sys.executable, "-m", "prime_rl.entrypoints.rl", "@", str(config_path)]
    start = time.monotonic()
    teacher_process: subprocess.Popen | None = None
    try:
        teacher_process = start_teacher_inference(
            args,
            log_dir,
            wait_until_ready=args.prime_component != "full",
        )
        return_code = run_logged_subprocess(command, os.environ.copy(), log_path=log_dir / "prime_rl_subprocess.log")
        log(f"Prime-RL exit status: {return_code} duration_s={time.monotonic() - start:.1f}")
        return return_code
    finally:
        stop_process(teacher_process, "OPD teacher inference")


if __name__ == "__main__":
    raise SystemExit(main())
