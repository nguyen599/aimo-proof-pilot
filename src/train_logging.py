from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from train_utils import path_name_token, sanitize_slug_part, truncate_slug


# Files that should never be included when syncing logs to Hugging Face.
HF_LOG_IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/.nfs*",
    "**/_shared_cache/**",
    "**/*_shared_cache/**",
    "**/dataset_cache/**",
    "**/hf_dataset_cache/**",
    "**/prepared_raw_dataset/**",
    "**/multinode_prepare_markers/**",
    "**/_shared_olmo_core_dataset/**",
    "**/_shared_olmo_core_checkpoint/**",
    "**/model_and_optim/**",
    "**/checkpoints/**",
    "**/checkpoint*/**",
    "**/step*/**",
    "**/*.arrow",
    "**/*.safetensors",
    "**/*.bin",
    "**/*.pt",
    "**/*.pth",
    "**/*.npy",
    "**/*.npz",
    "**/*.distcp",
    "**/*.memory_snapshot.pickle",
    "**/token_ids_part_*",
    "**/labels_mask_part_*",
]

# Environment keys and value patterns to redact from environment reports.
SENSITIVE_ENV_KEY_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASS|KEY|CREDENTIAL|AUTH|COOKIE|NETRC|PRIVATE)",
    re.IGNORECASE,
)
VALUE_REDACTIONS = [
    re.compile(r"https://[^/@\s]+@"),
    re.compile(r"hf_[A-Za-z0-9_]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]{12,}"),
]

# Lightweight host probes captured once per parent/worker for debugging cluster runs.
ENV_PROBE_COMMANDS = {
    "nvidia_smi": ["nvidia-smi"],
    "nvidia_smi_query": [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,driver_version,cuda_version,power.limit",
        "--format=csv,noheader",
    ],
    "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
    "lscpu": ["lscpu"],
    "free": ["free", "-h"],
    "df": ["df", "-h"],
    "ip_addr": ["ip", "addr"],
    "ip_route": ["ip", "route"],
    "ibv_devices": ["ibv_devices"],
    "ibv_devinfo": ["ibv_devinfo", "-l"],
    "ulimit": ["bash", "-lc", "ulimit -a"],
    "mount_head": ["bash", "-lc", "mount | head -200"],
    "numactl": ["numactl", "--hardware"],
}
DEBUG_PACKAGE_NAMES = ("huggingface_hub", "typer", "click", "transformers", "torch")
NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "urllib3",
    "boto3",
    "botocore",
    "s3transfer",
    "huggingface_hub",
    "huggingface_hub._commit_api",
    "huggingface_hub.file_download",
    "huggingface_hub._upload_large_folder",
    "huggingface_hub.lfs",
    "tqdm",
)
NOISY_THIRD_PARTY_LOGGER_LEVELS = {
    "huggingface_hub._upload_large_folder": logging.CRITICAL,
}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _logging_node_rank_from_args_or_env(args: argparse.Namespace | None = None) -> int | None:
    if args is not None:
        node_rank = node_rank_from_args_or_env(args)
        if node_rank is not None:
            return node_rank
    for name in ("OLMO_LOG_NODE_RANK", "GROUP_RANK", "GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_RANK"):
        node_rank = _env_int(name)
        if node_rank is not None:
            return node_rank
    if _env_int("LOCAL_RANK") is None:
        return _env_int("RANK")
    return None


def _logging_rank_label(args: argparse.Namespace | None = None) -> str | None:
    node_rank = _logging_node_rank_from_args_or_env(args)
    if node_rank is not None:
        return f"node{node_rank}"
    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK")
    if rank is not None:
        return str(rank)
    if local_rank is not None:
        return f"local{local_rank}"
    for name in ("GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_RANK"):
        node_rank = _env_int(name)
        if node_rank is not None:
            return str(node_rank)
    return None


def _shared_train_log_process(args: argparse.Namespace | None = None) -> bool:
    if _env_int("LOCAL_RANK") is not None:
        return False
    node_rank = _logging_node_rank_from_args_or_env(args)
    if node_rank is not None:
        return node_rank == 0
    for name in ("OLMO_LOG_NODE_RANK", "GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_RANK", "RANK"):
        rank = _env_int(name)
        if rank is not None:
            return rank == 0
    return True


def configure_logging(logdir: Path, args: argparse.Namespace | None = None) -> None:
    """Configure stdout and file logging under the caller-provided logdir."""
    logdir.mkdir(parents=True, exist_ok=True)
    rank_label = _logging_rank_label(args)
    if rank_label is None:
        rank_log_file = logdir / "train.log"
    elif rank_label.startswith("node"):
        rank_log_file = logdir / f"train_{rank_label}.log"
    else:
        rank_log_file = logdir / f"train_rank_{rank_label}.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(rank_log_file),
    ]
    shared_log_file = logdir / "train.log"
    if _shared_train_log_process(args) and shared_log_file != rank_log_file:
        handlers.append(logging.FileHandler(shared_log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(funcName)s() %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    suppress_noisy_third_party_loggers()


def suppress_noisy_third_party_loggers() -> None:
    for name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(NOISY_THIRD_PARTY_LOGGER_LEVELS.get(name, logging.WARNING))


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in DEBUG_PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
        except Exception as exc:
            versions[package_name] = f"error:{type(exc).__name__}"
    return versions


def log_dependency_versions() -> None:
    versions = package_versions()
    logging.info(
        "Dependency debug versions: huggingface_hub=%s typer=%s click=%s transformers=%s torch=%s python=%s",
        versions.get("huggingface_hub"),
        versions.get("typer"),
        versions.get("click"),
        versions.get("transformers"),
        versions.get("torch"),
        sys.version.replace("\n", " "),
    )


def flush_log_handlers() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


@contextmanager
def quiet_hf_transfer():
    """Suppress Hugging Face/tqdm transfer progress bars while preserving summary logs."""
    env_keys = ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_VERBOSITY", "TQDM_DISABLE")
    old_env = {key: os.environ.get(key) for key in env_keys}
    old_levels = {
        name: logging.getLogger(name).level
        for name in NOISY_THIRD_PARTY_LOGGERS
    }
    progress_was_disabled: bool | None = None
    enable_progress_bars = None
    try:
        try:
            from huggingface_hub.utils import (
                are_progress_bars_disabled,
                disable_progress_bars,
                enable_progress_bars as _enable_progress_bars,
            )

            progress_was_disabled = bool(are_progress_bars_disabled())
            enable_progress_bars = _enable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_VERBOSITY"] = "warning"
        os.environ["TQDM_DISABLE"] = "1"
        suppress_noisy_third_party_loggers()
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
        if progress_was_disabled is False and enable_progress_bars is not None:
            try:
                enable_progress_bars()
            except Exception:
                pass


def format_elapsed_seconds(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@contextmanager
def hf_transfer_heartbeat(description: str, interval_seconds: int | float = 300):
    """Log compact progress heartbeats while a blocking HF upload is running."""
    interval = max(1.0, float(interval_seconds or 300))
    stop_event = threading.Event()
    started = time.monotonic()

    def run() -> None:
        while not stop_event.wait(interval):
            logging.info(
                "%s still running; elapsed=%s",
                description,
                format_elapsed_seconds(time.monotonic() - started),
            )

    thread = threading.Thread(target=run, name="hf-transfer-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2)


def retry_hf_operation(
    description: str,
    operation,
    *,
    attempts: int | None = None,
    initial_delay_seconds: float | None = None,
    max_delay_seconds: float | None = None,
    abort_if=None,
):
    """Run a Hugging Face network operation with retry/backoff."""
    max_attempts = attempts if attempts is not None else int(os.environ.get("HF_UPLOAD_RETRY_ATTEMPTS", "5"))
    delay = initial_delay_seconds
    if delay is None:
        delay = float(os.environ.get("HF_UPLOAD_RETRY_INITIAL_DELAY_SECONDS", "10"))
    max_delay = max_delay_seconds
    if max_delay is None:
        max_delay = float(os.environ.get("HF_UPLOAD_RETRY_MAX_DELAY_SECONDS", "120"))
    max_attempts = max(1, max_attempts)

    for attempt in range(1, max_attempts + 1):
        if abort_if is not None and abort_if():
            raise RuntimeError(f"{description} aborted before attempt {attempt}.")
        try:
            return operation()
        except Exception:
            if abort_if is not None and abort_if():
                logging.warning("%s aborted after attempt %d/%d.", description, attempt, max_attempts)
                raise
            if attempt >= max_attempts:
                logging.error("%s failed after %d attempts.", description, max_attempts)
                raise
            sleep_for = min(max_delay, delay)
            logging.warning(
                "%s failed on attempt %d/%d; retrying in %.1fs.",
                description,
                attempt,
                max_attempts,
                sleep_for,
                exc_info=True,
            )
            time.sleep(sleep_for)
            delay = min(max_delay, max(1.0, delay * 2))


def redact_value(value: str) -> str:
    redacted = value
    for pattern in VALUE_REDACTIONS:
        if pattern.pattern.startswith("https://"):
            redacted = pattern.sub("https://<redacted>@", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


def redacted_environment() -> dict[str, str]:
    output: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if SENSITIVE_ENV_KEY_RE.search(key):
            output[key] = "<redacted>"
        else:
            output[key] = redact_value(value)
    return output


def truncate_probe_output(value: str, max_chars: int = 40000) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...<truncated {len(value) - max_chars} chars>"


def run_env_probe(name: str, command: list[str], timeout: int) -> dict[str, object]:
    del name
    executable = command[0]
    if shutil.which(executable) is None:
        return {"command": command, "skipped": True, "reason": f"{executable} not found"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "output": truncate_probe_output(redact_value(completed.stdout or "")),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return {
            "command": command,
            "timeout_seconds": timeout,
            "returncode": "timeout",
            "output": truncate_probe_output(redact_value(output)),
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def torch_environment_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"import_error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, object] = {
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_version": None,
    }
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    devices = []
    for index in range(int(info["cuda_device_count"])):
        try:
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / (1024**3), 3),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        except Exception as exc:
            devices.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    info["devices"] = devices
    return info


def node_rank_from_args_or_env(args: argparse.Namespace) -> int | None:
    if hasattr(args, "_resolved_node_rank"):
        return args._resolved_node_rank
    if getattr(args, "node_rank", None) is not None:
        return args.node_rank
    for name in ("OLMO_LOG_NODE_RANK", "GROUP_RANK", "GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID"):
        raw_value = os.environ.get(name)
        if not raw_value:
            continue
        try:
            raw_rank = int(raw_value)
        except ValueError:
            logging.warning("Ignoring non-integer %s=%r.", name, raw_value)
            continue
        if name == "GLOBAL_RANK" and getattr(args, "world_size_mode", "nodes") != "nodes":
            num_gpus = int(getattr(args, "_resolved_num_gpus", None) or getattr(args, "num_gpus", 0) or 0)
            if raw_rank > 0 and num_gpus > 0 and raw_rank % num_gpus == 0:
                return raw_rank // num_gpus
        return raw_rank
    if os.environ.get("LOCAL_RANK") in {None, ""}:
        raw_value = os.environ.get("RANK")
        if raw_value:
            try:
                return int(raw_value)
            except ValueError:
                logging.warning("Ignoring non-integer RANK=%r.", raw_value)
    return None


def primary_wandb_log_process(args: argparse.Namespace) -> bool:
    rank = os.environ.get("RANK")
    if rank not in {None, ""}:
        return rank == "0"
    node_rank = node_rank_from_args_or_env(args)
    return node_rank in {None, 0}


def wandb_rank_metadata(args: argparse.Namespace) -> dict[str, str]:
    node_rank = node_rank_from_args_or_env(args)
    return {
        "rank": os.environ.get("RANK") or "none",
        "local_rank": os.environ.get("LOCAL_RANK") or "none",
        "global_rank": os.environ.get("RANK") or os.environ.get("GLOBAL_RANK") or "none",
        "node_rank": str(node_rank) if node_rank is not None else "none",
    }


def wandb_rank_suffix(args: argparse.Namespace) -> str:
    metadata = wandb_rank_metadata(args)
    if metadata["node_rank"] != "none":
        return f"node{metadata['node_rank']}"
    if metadata["rank"] != "none":
        return metadata["rank"]
    if metadata["global_rank"] != "none":
        return metadata["global_rank"]
    return "none"


def collect_run_environment_info(
    args: argparse.Namespace,
    logdir: Path,
    phase: str,
    launcher: dict[str, object] | None = None,
) -> None:
    """Write a redacted runtime report into logdir for remote cluster debugging."""
    if getattr(args, "collect_env_info", "true") != "true":
        return
    local_rank = _env_int("LOCAL_RANK")
    if local_rank not in {None, 0}:
        logging.info("Skipping environment report on local_rank=%s; local_rank=0 writes the node report.", local_rank)
        return

    logdir.mkdir(parents=True, exist_ok=True)
    rank_metadata = wandb_rank_metadata(args)
    rank_suffix = sanitize_slug_part(wandb_rank_suffix(args))
    phase_slug = sanitize_slug_part(phase)
    report_path = logdir / f"environment_{phase_slug}_rank{rank_suffix}_pid{os.getpid()}.json"
    command_timeout = max(1, int(getattr(args, "env_info_command_timeout", 15)))

    report: dict[str, object] = {
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "os_cpu_count": os.cpu_count(),
            "affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
        "rank": rank_metadata,
        "launcher": launcher or {},
        "paths": {
            "model_path": getattr(args, "model_path", None),
            "dataset_path": getattr(args, "dataset_path", None),
            "output_path": getattr(args, "output_path", None),
            "logdir": getattr(args, "logdir", None),
            "cache_dir": getattr(args, "cache_dir", None),
            "olmo_core_checkpoint_cache": getattr(args, "olmo_core_checkpoint_cache", None),
            "olmo_core_dataset_cache": getattr(args, "olmo_core_dataset_cache", None),
        },
        "parallelism": {
            "tensor_parallel_degree": getattr(args, "tensor_parallel_degree", None),
            "context_parallel_degree": getattr(args, "context_parallel_degree", None),
            "pipeline_parallel_degree": getattr(args, "pipeline_parallel_degree", None),
            "pipeline_schedule": getattr(args, "pipeline_schedule", None),
        },
        "environment": redacted_environment(),
        "torch": torch_environment_info(),
        "python_packages": package_versions(),
        "probes": {},
    }

    probes = report["probes"]
    assert isinstance(probes, dict)
    for name, command in ENV_PROBE_COMMANDS.items():
        probes[name] = run_env_probe(name, command, command_timeout)

    try:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logging.info(
            "Environment report written: %s (phase=%s host=%s rank=%s local_rank=%s node_rank=%s)",
            report_path,
            phase,
            report["hostname"],
            rank_metadata["rank"],
            rank_metadata["local_rank"],
            rank_metadata["node_rank"],
        )
        torch_info = report.get("torch", {})
        if isinstance(torch_info, dict):
            logging.info(
                "Environment summary: torch=%s cuda=%s cuda_available=%s cuda_devices=%s",
                torch_info.get("version"),
                torch_info.get("cuda_version"),
                torch_info.get("cuda_available"),
                torch_info.get("cuda_device_count"),
            )
    except Exception:
        logging.exception("Failed to write environment report for phase=%s", phase)


def primary_hf_log_upload_process(args: argparse.Namespace) -> bool:
    if getattr(args, "internal_backend", None) is not None:
        return False
    return primary_wandb_log_process(args)


def hf_log_upload_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "hf_log_upload", "true") == "true" and bool(getattr(args, "hf_log_repo", "").strip())


def hf_log_path_prefix(args: argparse.Namespace) -> str:
    raw_prefix = getattr(args, "hf_log_path_prefix", "training-logs") or ""
    parts = [sanitize_slug_part(part) for part in raw_prefix.strip("/").split("/") if part.strip()]
    return "/".join(parts)


def hf_log_upload_path(args: argparse.Namespace, status: str) -> str:
    del status
    cached_path = getattr(args, "_hf_log_path_in_repo", None)
    if cached_path:
        return cached_path

    timestamp = getattr(args, "_hf_log_started_at_utc", None)
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        setattr(args, "_hf_log_started_at_utc", timestamp)
    run_name = path_name_token(getattr(args, "output_path", None), path_name_token(getattr(args, "logdir", None), "run"))
    folder_name = truncate_slug(
        f"{timestamp}_{run_name}_rank{wandb_rank_suffix(args)}",
        max_length=180,
    )
    prefix = hf_log_path_prefix(args)
    path_in_repo = f"{prefix}/{folder_name}" if prefix else folder_name
    setattr(args, "_hf_log_path_in_repo", path_in_repo)
    return path_in_repo


def hf_log_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def write_hf_log_metadata(
    args: argparse.Namespace,
    status: str,
    exit_code: int | None,
    path_in_repo: str,
    upload_kind: str,
) -> Path:
    logdir = Path(args.logdir).expanduser().resolve()
    metadata_path = logdir / "hf_log_upload_metadata.json"
    metadata = {
        "status": status,
        "exit_code": exit_code,
        "upload_kind": upload_kind,
        "backend": args.backend,
        "internal_backend": args.internal_backend,
        "model_arch": args.model_arch,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "output_path": args.output_path,
        "logdir": args.logdir,
        "hf_log_repo": args.hf_log_repo,
        "hf_path_in_repo": path_in_repo,
        "hf_log_upload_interval_seconds": getattr(args, "hf_log_upload_interval_seconds", None),
        "started_at_utc": getattr(args, "_hf_log_started_at_utc", None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **wandb_rank_metadata(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def upload_logdir_to_hf(
    args: argparse.Namespace,
    status: str,
    exit_code: int | None,
    upload_kind: str = "final",
) -> None:
    """Upload logdir to a private HF dataset, handling missing deps/tokens gracefully."""
    if not hf_log_upload_enabled(args):
        return
    if not primary_hf_log_upload_process(args):
        logging.info("Skipping HF logdir upload on non-primary or internal worker process.")
        return

    logdir = Path(args.logdir).expanduser().resolve()
    if not logdir.is_dir():
        logging.warning("HF logdir upload skipped; logdir does not exist: %s", logdir)
        return

    repo_id = args.hf_log_repo.strip()
    path_in_repo = hf_log_upload_path(args, status)
    logging.info(
        "Uploading logdir to HF dataset %s/%s status=%s kind=%s: %s",
        repo_id,
        path_in_repo,
        status,
        upload_kind,
        logdir,
    )
    flush_log_handlers()
    write_hf_log_metadata(args, status, exit_code, path_in_repo, upload_kind)
    flush_log_handlers()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logging.warning("HF logdir upload skipped; huggingface_hub is not installed.")
        return

    token = hf_log_token()
    if not token:
        logging.warning(
            "HF logdir upload may fail because no HF token was found. "
            "Set HF_TOKEN or login with huggingface-cli for private repo access."
        )

    try:
        def upload_once():
            api = HfApi(token=token)
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
            logging.info("HF logdir upload ignore patterns: %s", ", ".join(HF_LOG_IGNORE_PATTERNS))
            heartbeat_interval = float(os.environ.get("HF_LOG_UPLOAD_HEARTBEAT_SECONDS", "300"))
            with quiet_hf_transfer(), hf_transfer_heartbeat(
                f"HF logdir upload to {repo_id}/{path_in_repo}",
                heartbeat_interval,
            ):
                return api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(logdir),
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload training logs: {Path(args.logdir).name} ({status}, {upload_kind})",
                    token=token,
                    ignore_patterns=HF_LOG_IGNORE_PATTERNS,
                )

        commit_info = retry_hf_operation(
            f"HF logdir upload to {repo_id}/{path_in_repo}",
            upload_once,
        )
        commit_url = getattr(commit_info, "commit_url", None) or str(commit_info)
        logging.info("Uploaded logdir to HF dataset: %s", commit_url)
    except Exception:
        logging.exception("HF logdir upload failed.")


@dataclass
class PeriodicHFLogUploader:
    """Background uploader that periodically syncs logdir to the same HF path."""

    args: argparse.Namespace
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def interval_seconds(self) -> int:
        return int(getattr(self.args, "hf_log_upload_interval_seconds", 3600) or 0)

    def should_start(self) -> bool:
        return (
            self.interval_seconds > 0
            and hf_log_upload_enabled(self.args)
            and primary_hf_log_upload_process(self.args)
        )

    def start(self) -> None:
        if not self.should_start():
            return
        self._thread = threading.Thread(target=self._run, name="hf-log-periodic-upload", daemon=True)
        self._thread.start()
        logging.info("Periodic HF logdir upload enabled: interval=%ss", self.interval_seconds)

    def _run(self) -> None:
        interval = max(1, self.interval_seconds)
        while not self._stop_event.wait(interval):
            self.upload(status="running", exit_code=None, upload_kind="periodic")

    def upload(self, status: str, exit_code: int | None, upload_kind: str) -> None:
        with self._lock:
            upload_logdir_to_hf(self.args, status=status, exit_code=exit_code, upload_kind=upload_kind)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30)
