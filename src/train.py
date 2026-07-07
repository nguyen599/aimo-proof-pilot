#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _load_env_file() -> None:
    """Load the repo-root .env (or $AIMO_ENV_FILE) into os.environ.

    Format: one KEY=VALUE per line; blank lines and #-comment lines allowed.
    Variables already present in the environment are not overridden."""
    env_file = Path(os.environ.get("AIMO_ENV_FILE") or Path(__file__).resolve().parent.parent / ".env")
    if not env_file.is_file():
        return
    for lineno, raw in enumerate(env_file.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{env_file}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


DEFAULT_SUBMISSIONS_REPO = "https://github.com/nguyen599/aimo-proof-pilot.git"
DEFAULT_OPEN_INSTRUCT_REPO = "https://github.com/nguyen599/open-instruct.git"
DEFAULT_OLMO_CORE_REPO = "https://github.com/nguyen599/OLMo-core.git"
DEFAULT_RLCSD_REPO = "https://github.com/THU-BPM/RLCSD.git"
DEFAULT_VERL_REPO = "https://github.com/nguyen599/verl.git"
DEFAULT_PRIME_RL_REPO = "https://github.com/nguyen599/prime-rl.git"
DEFAULT_MEGATRON_CORE_REPO = "https://github.com/NVIDIA/Megatron-LM.git"
DEFAULT_LIGER_KERNEL_REPO = "https://github.com/linkedin/Liger-Kernel.git"
DEFAULT_SUBMISSIONS_REF = "main"
DEFAULT_OPEN_INSTRUCT_REF = "main"
DEFAULT_OLMO_CORE_REF = "main"
DEFAULT_RLCSD_REF = "main"
DEFAULT_VERL_REF = "main"
DEFAULT_PRIME_RL_REF = "main"
DEFAULT_MEGATRON_CORE_REF = "main"
DEFAULT_LIGER_KERNEL_REF = "main"
DEFAULT_RUNTIME_DIR = "/tmp/aimo-proof-pilot-runtime"
DEFAULT_OPEN_INSTRUCT_RUNTIME_DIR = "/tmp/open-instruct-runtime"
DEFAULT_OLMO_CORE_RUNTIME_DIR = "/tmp/OLMo-core-runtime"
DEFAULT_RLCSD_RUNTIME_DIR = "/tmp/RLCSD-runtime"
DEFAULT_VERL_RUNTIME_DIR = "/tmp/verl-runtime"
DEFAULT_PRIME_RL_RUNTIME_DIR = "/tmp/prime-rl-runtime"
DEFAULT_RUNTIME_FETCH_STATE_DIR = "/tmp/train-runtime-fetch"
DEFAULT_RUNTIME_TRAINING_DEPS_DIR = "/tmp/olmo-train-runtime-deps"
DEFAULT_APEX_WHEEL_REPO = "nguyen599/prebuild-wheels-util"
DEFAULT_APEX_WHEEL_FILE = "torch2.11+cu130/apex-0.1-cp312-cp312-linux_x86_64.whl"
DEFAULT_TRANSFORMER_ENGINE_WHEEL_REPO = "nguyen599/prebuild-wheels-util"
DEFAULT_TRANSFORMER_ENGINE_WHEEL_FILE = (
    "torch2.11+cu130/transformer_engine-2.17.0.dev0-cp312-cp312-linux_x86_64.whl"
)
DEFAULT_RUNTIME_FETCH_TIMEOUT_SECONDS = 1800.0
DEFAULT_RUNTIME_FETCH_POLL_SECONDS = 2.0
DEFAULT_RUNTIME_FETCH_STALE_FAILED_SECONDS = 30.0
DEFAULT_GIT_LOCK_STALE_SECONDS = 300.0
DEFAULT_GIT_RETRY_ATTEMPTS = 5
DEFAULT_GIT_RETRY_BASE_SECONDS = 5.0
DEFAULT_GIT_RETRY_MAX_SECONDS = 60.0
DEFAULT_RUNTIME_DEP_RETRY_ATTEMPTS = 5
DEFAULT_RUNTIME_DEP_RETRY_BASE_SECONDS = 5.0
DEFAULT_RUNTIME_DEP_RETRY_MAX_SECONDS = 60.0
DEFAULT_RUN_DIR_MARKER_TIMEOUT_SECONDS = 300.0
DEFAULT_RUN_DIR_MARKER_POLL_SECONDS = 0.5
DEFAULT_GRPO_RUNTIME_VLLM_VERSION = "0.23.1rc1.dev699+gf5a8d7337"
DEFAULT_GRPO_RUNTIME_VLLM_CUDA_VERSION = "130"
DEFAULT_VLLM_RUNTIME_WHEEL_URL = (
    "https://wheels.vllm.ai/f5a8d73377d0f0a4e00cba172f9fbd0d50471b07/"
    "vllm-0.23.1rc1.dev699%2Bgf5a8d7337-cp38-abi3-manylinux_2_28_x86_64.whl"
)
DEFAULT_GRPO_RUNTIME_REQUIREMENTS = (
    "openenv-core==0.2.1",
    "nltk>=3.9.1",
    "debugpy>=1.8.13",
    "litellm>=1.72.0,<1.75.2",
    # The submitted image already contains a CUDA-compatible vLLM wheel. Install
    # the Python packages GRPO/vLLM expects without replacing the baked binary
    # wheel, because public vLLM wheels may target a different CUDA runtime.
    "anthropic>=0.71.0",
    "compressed-tensors==0.15.0.1",
    "depyf==0.20.0",
    "diskcache==5.6.3",
    "gguf>=0.17.0",
    "ijson",
    "llguidance>=1.3.0,<1.4.0",
    "lm-format-enforcer==0.11.3",
    "mistral_common[image]>=1.11.2",
    "pydantic-extra-types>=2.10.0",
    "pycountry>=24.6.1",
    "model-hosting-container-standards>=0.1.14,<1.0.0",
    "opencv-python-headless>=4.13.0",
    "opentelemetry-semantic-conventions-ai>=0.4.1",
    "outlines_core==0.2.11",
    "partial-json-parser",
    "distlib",
    # FastAPI 0.137 changed router internals in a way that breaks the
    # Prometheus instrumentator used by vLLM 0.21 health checks.
    "fastapi<0.137",
    "prometheus-fastapi-instrumentator>=7.0.0",
    "pybase64",
    "python-json-logger",
    "sentencepiece",
    "setproctitle",
    "tokenspeed-mla==0.1.2",
    "uvloop>=0.21.0",
    "xgrammar>=0.2.0,<1.0.0",
)
DEFAULT_RLCSD_RUNTIME_REQUIREMENTS = (
    # Keep CUDA/torch/transformers packages out of this overlay; vLLM is
    # installed here for RLCSD because the vendored verl imports it directly.
    "hydra-core>=1.3.2",
    "omegaconf>=2.3.0",
    "aiohttp-cors==0.8.1",
    "codetiming",
    "colorful==0.5.8",
    "datasets>=3.0.0",
    "openai>=1.0.0",
    "peft>=0.14.0",
    "markdown==3.10.2",
    "opencensus==0.11.4",
    "opencensus-context==0.1.3",
    "orjson",
    "packaging==25.0",
    f"vllm=={DEFAULT_GRPO_RUNTIME_VLLM_VERSION}",
    "gguf>=0.17.0",
    "pyarrow>=19.0.0",
    "py-spy==0.4.2",
    "pybind11==3.0.4",
    # Use the Ray installed in the image. Installing ray[default] here pulls a
    # dashboard/OpenTelemetry stack that fails on the host image.
    "dill",
    "deepspeed>=0.15.0",
    "pandas",
    "pylatexenc",
    "latex2sympy2_extended",
    "math_verify>=0.7.0",
    "tensorboard",
    "tensorboard-data-server==0.7.2",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "torchdata==0.11.0",
    "pyvers",
    "smart_open==7.6.1",
    "watchdog",
    "werkzeug==3.1.8",
    "wrapt==2.2.1",
)
DEFAULT_VERL_AUTOMODEL_RUNTIME_REQUIREMENTS = (
    "nemo-automodel==0.4.0",
)
DEFAULT_VERL_NVIDIA_RUNTIME_REQUIREMENTS = (
    # cublasmp in the current CUDA 13 / Transformer Engine stack needs the
    # ncclCommQueryProperties symbol, which is present in NCCL 2.29.3.
    "nvidia-nccl-cu13==2.29.3",
)
DEFAULT_PRIME_RL_SOURCE_REQUIREMENTS = (
    # pip does not understand prime-rl's [tool.uv.sources], so direct source
    # dependencies need to be supplied explicitly when installing with pip.
    "torchtitan @ git+https://github.com/pytorch/torchtitan.git@23e4dfc",
    "dion @ git+https://github.com/samsja/dion.git@d891eeb",
    (
        "deep-ep @ "
        "https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.5.0/"
        "deep_ep-1.2.1+29d31c0-cp312-cp312-linux_x86_64.whl"
    ),
    (
        "deep-gemm @ "
        "https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.5.0/"
        "deep_gemm-2.5.0+891d57b-cp312-cp312-linux_x86_64.whl"
    ),
)
DEFAULT_PRIME_RL_RUNTIME_REQUIREMENTS = (
    DEFAULT_VLLM_RUNTIME_WHEEL_URL,
    # This vLLM dev wheel requires FastAPI below 0.137 while depending on
    # Starlette 1.x directly, so keep the resolver on that exact boundary.
    "fastapi>=0.133,<0.137",
    "starlette>=1.0.1,<2.0",
    "prometheus-fastapi-instrumentator>=8.0.0",
    # Prime-RL's W&B monitor imports the historical wandb_gql module. Our
    # runtime source tree provides a compatibility module backed by
    # graphql-core, and these versions match current Prime-RL metadata.
    "graphql-core>=3.2.0",
    "wandb>=0.26.1",
    "wandb-workspaces>=0.4.3",
    "nltk>=3.9.1",
    "jaxtyping>=0.3.2",
    "weave"
)
PROTECTED_RUNTIME_OVERLAY_PACKAGES = (
    "flash_attn",
    "functorch",
    "huggingface_hub",
    "numpy",
    "nvidia",
    "safetensors",
    "tokenizers",
    "torch",
    "torchao",
    "torchaudio",
    "torchdata",
    "torchtext",
    "torchvision",
    "triton",
    "transformer_engine",
)
WRAPPER_REEXEC_ENV = "TRAIN_WRAPPER_REEXECUTED"
PREPARED_ENGINE_ENV = "TRAIN_WRAPPER_PREPARED_ENGINE_PATH"
PREPARED_OPEN_INSTRUCT_ENV = "TRAIN_WRAPPER_PREPARED_OPEN_INSTRUCT_DIR"
PREPARED_OLMO_CORE_ENV = "TRAIN_WRAPPER_PREPARED_OLMO_CORE_DIR"
PREPARED_RLCSD_ENV = "TRAIN_WRAPPER_PREPARED_RLCSD_DIR"
PREPARED_VERL_ENV = "TRAIN_WRAPPER_PREPARED_VERL_DIR"
PREPARED_PRIME_RL_ENV = "TRAIN_WRAPPER_PREPARED_PRIME_RL_DIR"
WRAPPER_LOG_FILE: Path | None = None
NODE_SPECIFIC_RUN_DIR_ARGS = {
    "--node-rank",
    "--node_rank",
}
SENSITIVE_ARG_NAMES = {
    "--github-token",
    "--github_token",
    "--hf-token",
    "--hf_token",
    "--token",
    "--aws-access-key-id",
    "--aws_access_key_id",
    "--aws-secret-access-key",
    "--aws_secret_access_key",
    "--llm-judge-api-key",
    "--llm_judge_api_key",
}
SECRET_VALUE_REDACTIONS = [
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"hf_[A-Za-z0-9_]{16,}"),
    re.compile(r"wandb_v1_[A-Za-z0-9_]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]{12,}"),
    re.compile(r"([?&]X-Amz-Signature=)[A-Fa-f0-9]+"),
    re.compile(r"https://[^/@\s]+@"),
]


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--fetch-update", "--fetch_update", dest="fetch_update", action="store_true", default=None)
    parser.add_argument("--no-fetch-update", "--no_fetch_update", dest="fetch_update", action="store_false")
    parser.add_argument("--submissions-repo", "--submissions_repo", default=None)
    parser.add_argument("--submissions-ref", "--submissions_ref", default=None)
    parser.add_argument("--submissions-runtime-dir", "--submissions_runtime_dir", default=None)
    parser.add_argument("--open-instruct-repo", "--open_instruct_repo", default=None)
    parser.add_argument("--open-instruct-ref", "--open_instruct_ref", default=None)
    parser.add_argument("--open-instruct-runtime-dir", "--open_instruct_runtime_dir", default=None)
    parser.add_argument("--olmo-core-repo", "--olmo_core_repo", default=None)
    parser.add_argument("--olmo-core-ref", "--olmo_core_ref", default=None)
    parser.add_argument("--olmo-core-runtime-dir", "--olmo_core_runtime_dir", default=None)
    parser.add_argument("--rlcsd-repo", "--rlcsd_repo", default=None)
    parser.add_argument("--rlcsd-ref", "--rlcsd_ref", default=None)
    parser.add_argument("--rlcsd-runtime-dir", "--rlcsd_runtime_dir", default=None)
    parser.add_argument("--verl-repo", "--verl_repo", default=None)
    parser.add_argument("--verl-ref", "--verl_ref", default=None)
    parser.add_argument("--verl-runtime-dir", "--verl_runtime_dir", default=None)
    parser.add_argument("--prime-rl-repo", "--prime_rl_repo", default=None)
    parser.add_argument("--prime-rl-ref", "--prime_rl_ref", default=None)
    parser.add_argument("--prime-rl-runtime-dir", "--prime_rl_runtime_dir", default=None)
    parser.add_argument("--runtime-fetch-state-dir", "--runtime_fetch_state_dir", default=None)
    parser.add_argument("--runtime-fetch-timeout", "--runtime_fetch_timeout", type=float, default=None)
    parser.add_argument("--runtime-fetch-poll-interval", "--runtime_fetch_poll_interval", type=float, default=None)
    parser.add_argument(
        "--ensure-runtime-training-deps",
        "--ensure_runtime_training_deps",
        default=None,
    )
    parser.add_argument(
        "--no-ensure-runtime-training-deps",
        "--no_ensure_runtime_training_deps",
        dest="ensure_runtime_training_deps",
        action="store_const",
        const="false",
    )
    parser.add_argument("--runtime-training-deps-dir", "--runtime_training_deps_dir", default=None)
    parser.add_argument("--megatron-core-repo", "--megatron_core_repo", default=None)
    parser.add_argument("--megatron-core-ref", "--megatron_core_ref", default=None)
    parser.add_argument("--liger-kernel-repo", "--liger_kernel_repo", default=None)
    parser.add_argument("--liger-kernel-ref", "--liger_kernel_ref", default=None)
    parser.add_argument("--apex-wheel-repo", "--apex_wheel_repo", default=None)
    parser.add_argument("--apex-wheel-file", "--apex_wheel_file", default=None)
    parser.add_argument("--transformer-engine-wheel-repo", "--transformer_engine_wheel_repo", default=None)
    parser.add_argument("--transformer-engine-wheel-file", "--transformer_engine_wheel_file", default=None)
    parser.add_argument(
        "--self-update-wrapper",
        "--self_update_wrapper",
        dest="self_update_wrapper",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-self-update-wrapper",
        "--no_self_update_wrapper",
        dest="self_update_wrapper",
        action="store_false",
    )
    return parser.parse_known_args(argv)


def wrapper_log_node_label() -> str:
    for env_name in ("OLMO_LOG_NODE_RANK", "GROUP_RANK", "GLOBAL_RANK", "NODE_RANK", "SLURM_NODEID"):
        value = os.environ.get(env_name)
        if value not in {None, ""}:
            return f"node{value}"
    rank = os.environ.get("RANK")
    if rank not in {None, ""}:
        return f"rank{rank}"
    return "none"


def sanitize_slug_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-")
    return cleaned or "value"


def truncate_slug(value: str, max_length: int = 180) -> str:
    if len(value) <= max_length:
        return value
    return value[:max_length].rstrip("-_.+")


def configure_wrapper_file_logging(forwarded_args: list[str]) -> None:
    global WRAPPER_LOG_FILE
    logdir_value = forwarded_option_value(forwarded_args, "--logdir")
    if not logdir_value:
        return
    logdir = Path(logdir_value).expanduser()
    run_dir_name = os.environ.get("OLMO_RUN_DIR_NAME")
    if wrapper_run_dir_enabled(forwarded_args) and run_dir_name:
        logdir = logdir / run_dir_name
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        WRAPPER_LOG_FILE = logdir / "train_wrapper.log"
    except Exception as exc:
        print(
            f"[train.py wrapper] failed to configure wrapper log file under {logdir}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def log(message: str) -> None:
    safe_message = redact_secret(message)
    line = f"[train.py wrapper] {safe_message}"
    print(line, file=sys.stderr, flush=True)
    if WRAPPER_LOG_FILE is None:
        return
    timestamp = datetime.now().strftime("%H:%M:%S,%f")[:-3]
    file_line = f"{timestamp} {wrapper_log_node_label()} pid{os.getpid()} {line}\n"
    try:
        with WRAPPER_LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(file_line)
    except Exception as exc:
        print(
            f"[train.py wrapper] failed to write wrapper log file {WRAPPER_LOG_FILE}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def redact_secret(value: str) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        value = value.replace(token, "<redacted>")
    for pattern in SECRET_VALUE_REDACTIONS:
        if pattern.pattern.startswith("https://"):
            value = pattern.sub("https://<redacted>@", value)
        elif "X-Amz-Signature" in pattern.pattern:
            value = pattern.sub(r"\1<redacted>", value)
        else:
            value = pattern.sub("<redacted>", value)
    return value


def redact_cli_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        if "=" in arg:
            name, value = arg.split("=", 1)
            if name in SENSITIVE_ARG_NAMES or any(token in name.lower() for token in ("token", "secret", "password")):
                redacted.append(f"{name}=<redacted>")
            else:
                redacted.append(f"{name}={redact_secret(value)}")
            continue
        if arg in SENSITIVE_ARG_NAMES or any(token in arg.lower() for token in ("token", "secret", "password")):
            redacted.append(arg)
            skip_next = True
            continue
        redacted.append(redact_secret(arg))
    return redacted


def cli_args_for_log(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in redact_cli_args(args))


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    token = env.get("GITHUB_TOKEN")
    if token:
        # Route every GitHub URL form through token-authenticated https.
        env["GIT_CONFIG_COUNT"] = "3"
        rewrite_key = f"url.https://{token}@github.com/.insteadOf"
        env["GIT_CONFIG_KEY_0"] = rewrite_key
        env["GIT_CONFIG_VALUE_0"] = "https://github.com/"
        env["GIT_CONFIG_KEY_1"] = rewrite_key
        env["GIT_CONFIG_VALUE_1"] = "git@github.com:"
        env["GIT_CONFIG_KEY_2"] = rewrite_key
        env["GIT_CONFIG_VALUE_2"] = "ssh://git@github.com/"
    else:
        # No token: still rewrite ssh forms to anonymous https so public
        # repos and submodules (e.g. prime-rl's deps/*) clone without keys.
        env["GIT_CONFIG_COUNT"] = "2"
        rewrite_key = "url.https://github.com/.insteadOf"
        env["GIT_CONFIG_KEY_0"] = rewrite_key
        env["GIT_CONFIG_VALUE_0"] = "git@github.com:"
        env["GIT_CONFIG_KEY_1"] = rewrite_key
        env["GIT_CONFIG_VALUE_1"] = "ssh://git@github.com/"
    return env


def git_retry_attempts() -> int:
    value = parse_int(os.environ.get("RUNTIME_GIT_RETRY_ATTEMPTS"))
    if value is None:
        return DEFAULT_GIT_RETRY_ATTEMPTS
    return max(1, value)


def git_retry_base_seconds() -> float:
    return max(0.0, parse_float(os.environ.get("RUNTIME_GIT_RETRY_BASE_SECONDS"), DEFAULT_GIT_RETRY_BASE_SECONDS))


def git_retry_max_seconds() -> float:
    return max(0.0, parse_float(os.environ.get("RUNTIME_GIT_RETRY_MAX_SECONDS"), DEFAULT_GIT_RETRY_MAX_SECONDS))


def git_command_is_retryable(args: list[str]) -> bool:
    if not args:
        return False
    return args[0] in {"fetch", "checkout", "remote", "rev-parse"}


def run_git(args: list[str], cwd: Path | None = None, retry: bool | None = None) -> str:
    command = ["git", *args]
    attempts = git_retry_attempts() if (git_command_is_retryable(args) if retry is None else retry) else 1
    last_process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        process = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=git_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode == 0:
            return process.stdout.strip()
        last_process = process
        if attempt >= attempts:
            break
        sleep_seconds = min(git_retry_max_seconds(), git_retry_base_seconds() * (2 ** (attempt - 1)))
        log(
            "Git command failed with exit code %d on attempt %d/%d; retrying in %.1fs: %s\nstderr:\n%s"
            % (
                process.returncode,
                attempt,
                attempts,
                sleep_seconds,
                " ".join(command),
                redact_secret(process.stderr.strip()),
            )
        )
        time.sleep(sleep_seconds)
    safe_command = " ".join(command)
    stdout = last_process.stdout if last_process is not None else ""
    stderr = last_process.stderr if last_process is not None else ""
    returncode = last_process.returncode if last_process is not None else -1
    raise RuntimeError(
        f"Git command failed with exit code {returncode}: {safe_command}\n"
        f"stdout:\n{redact_secret(stdout)}\n"
        f"stderr:\n{redact_secret(stderr)}"
    )


def parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def forwarded_option_value(args: list[str], *names: str) -> str | None:
    value = None
    for idx, arg in enumerate(args):
        for name in names:
            if arg == name and idx + 1 < len(args):
                value = args[idx + 1]
            if arg.startswith(f"{name}="):
                value = arg.split("=", 1)[1]
    return value


def normalized_run_dir_argv(args: list[str]) -> list[str]:
    normalized = []
    skip_next = False
    for arg in args:
        if skip_next:
            normalized.append("<node-rank>")
            skip_next = False
            continue
        if arg in NODE_SPECIFIC_RUN_DIR_ARGS:
            normalized.append(arg)
            skip_next = True
            continue
        if "=" in arg:
            name, _value = arg.split("=", 1)
            if name in NODE_SPECIFIC_RUN_DIR_ARGS:
                normalized.append(f"{name}=<node-rank>")
                continue
        normalized.append(arg)
    return normalized


def wrapper_run_dir_mode(forwarded_args: list[str]) -> str:
    return forwarded_option_value(forwarded_args, "--run_dir_mode", "--run-dir-mode") or "auto"


def wrapper_run_dir_enabled(forwarded_args: list[str]) -> bool:
    if wrapper_run_dir_mode(forwarded_args) == "none":
        return False
    if forwarded_option_value(forwarded_args, "--internal_backend", "--internal-backend"):
        return False
    if os.environ.get("OLMO_SWEEP_CHILD") == "1":
        return False
    operator_mode = forwarded_option_value(forwarded_args, "--operator_mode", "--operator-mode")
    if parse_bool(operator_mode, False):
        return False
    return True


def explicit_wrapper_run_dir_name(forwarded_args: list[str]) -> str | None:
    explicit = forwarded_option_value(forwarded_args, "--run_dir_name", "--run-dir-name") or os.environ.get(
        "OLMO_RUN_DIR_NAME"
    )
    if explicit:
        return truncate_slug(sanitize_slug_part(explicit), max_length=120)
    return None


def timestamp_run_dir_name() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_pid{os.getpid()}"


def wrapper_num_nodes(forwarded_args: list[str]) -> int:
    explicit = parse_int(forwarded_option_value(forwarded_args, "--num_nodes", "--num-nodes"))
    if explicit is not None and explicit > 0:
        return explicit
    world_size = parse_int(os.environ.get("WORLD_SIZE"))
    if world_size is not None and world_size > 0:
        return world_size
    return 1


def wrapper_run_dir_marker_path(forwarded_args: list[str], base_logdir: Path) -> Path:
    payload = {
        "raw_argv": normalized_run_dir_argv(forwarded_args),
        "base_logdir": str(base_logdir),
        "master_addr": forwarded_option_value(forwarded_args, "--master_addr", "--master-addr")
        or os.environ.get("MASTER_ADDR"),
        "master_port": forwarded_option_value(forwarded_args, "--master_port", "--master-port")
        or os.environ.get("MASTER_PORT"),
        "world_size": os.environ.get("WORLD_SIZE"),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    marker_root = Path(os.environ.get("OLMO_RUN_DIR_MARKER_ROOT", str(base_logdir / "_run_dir_markers"))).expanduser()
    return marker_root / f"{fingerprint}.json"


def read_json_file(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def coordinated_wrapper_run_dir_name(forwarded_args: list[str], base_logdir: Path) -> str:
    node_rank = resolve_wrapper_node_rank(forwarded_args)
    num_nodes = wrapper_num_nodes(forwarded_args)
    if node_rank is None or num_nodes <= 1:
        return timestamp_run_dir_name()

    marker_path = wrapper_run_dir_marker_path(forwarded_args, base_logdir)
    min_created_time = time.time() - 15.0
    if node_rank == 0:
        run_dir_name = timestamp_run_dir_name()
        write_json_file(
            marker_path,
            {
                "run_dir_name": run_dir_name,
                "created_time": time.time(),
                "created_utc": datetime.utcnow().isoformat() + "Z",
                "pid": os.getpid(),
            },
        )
        return run_dir_name

    deadline = time.monotonic() + DEFAULT_RUN_DIR_MARKER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        marker = read_json_file(marker_path)
        if marker is not None:
            run_dir_name = marker.get("run_dir_name")
            created_time = marker.get("created_time", 0)
            if isinstance(run_dir_name, str) and isinstance(created_time, (int, float)) and created_time >= min_created_time:
                return run_dir_name
        time.sleep(DEFAULT_RUN_DIR_MARKER_POLL_SECONDS)
    raise TimeoutError(f"Timed out waiting for node_rank=0 wrapper run-dir marker: {marker_path}")


def prepare_wrapper_run_directory(forwarded_args: list[str]) -> None:
    logdir_value = forwarded_option_value(forwarded_args, "--logdir")
    if not logdir_value or not wrapper_run_dir_enabled(forwarded_args):
        return
    run_dir_name = explicit_wrapper_run_dir_name(forwarded_args)
    if run_dir_name is None:
        run_dir_name = coordinated_wrapper_run_dir_name(forwarded_args, Path(logdir_value).expanduser())
    os.environ["OLMO_RUN_DIR_NAME"] = run_dir_name


def resolve_wrapper_node_rank(forwarded_args: list[str]) -> int | None:
    forwarded_rank_value = forwarded_option_value(forwarded_args, "--node_rank", "--node-rank")
    cli_rank = parse_int(forwarded_rank_value)
    if cli_rank is not None and wrapper_num_nodes(forwarded_args) <= 1:
        return cli_rank
    for env_name in ("NODE_RANK", "GLOBAL_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_RANK"):
        rank = parse_int(os.environ.get(env_name))
        if rank is not None:
            if cli_rank is not None and cli_rank != rank:
                log(f"Using {env_name}={rank} for wrapper coordination; forwarded node rank was {cli_rank}")
            return rank
    if cli_rank is not None:
        return cli_rank
    if forwarded_rank_value:
        log(f"Ignoring non-integer node rank value {forwarded_rank_value!r}")
    return None


def wait_for_git_index_lock(runtime_dir: Path, label: str) -> None:
    lock_path = runtime_dir / ".git" / "index.lock"
    if not lock_path.exists():
        return
    stale_seconds = parse_float(os.environ.get("RUNTIME_GIT_LOCK_STALE_SECONDS"), DEFAULT_GIT_LOCK_STALE_SECONDS)
    deadline = time.monotonic() + stale_seconds
    warned = False
    while lock_path.exists():
        try:
            age_seconds = max(0.0, time.time() - lock_path.stat().st_mtime)
        except OSError:
            return
        if age_seconds >= stale_seconds:
            log(f"Removing stale {label} Git lock after {age_seconds:.0f}s: {lock_path}")
            lock_path.unlink(missing_ok=True)
            return
        if time.monotonic() >= deadline:
            break
        if not warned:
            log(f"Waiting for {label} Git lock to clear: {lock_path}")
            warned = True
        time.sleep(min(DEFAULT_RUNTIME_FETCH_POLL_SECONDS, max(0.1, deadline - time.monotonic())))
    if lock_path.exists():
        raise RuntimeError(f"{label} Git lock still exists: {lock_path}")


def ensure_runtime_repo(repo: str, ref: str, runtime_dir: Path, label: str) -> Path:
    if runtime_dir.exists() and not (runtime_dir / ".git").is_dir():
        raise RuntimeError(f"Runtime dir exists but is not a Git checkout: {runtime_dir}")
    if not runtime_dir.exists():
        runtime_dir.parent.mkdir(parents=True, exist_ok=True)
        log(f"Cloning {label} repo into {runtime_dir}")
        run_git(["clone", "--filter=blob:none", "--no-checkout", repo, str(runtime_dir)])

    wait_for_git_index_lock(runtime_dir, label)
    log(f"Fetching {label} ref {ref!r}")
    run_git(["remote", "set-url", "origin", repo], cwd=runtime_dir)
    run_git(["fetch", "--depth", "1", "origin", ref], cwd=runtime_dir)
    run_git(["checkout", "--force", "FETCH_HEAD"], cwd=runtime_dir)
    resolved = run_git(["rev-parse", "--short", "HEAD"], cwd=runtime_dir)
    log(f"Using {label} repo {redact_secret(repo)} at {resolved}")
    return runtime_dir


def prepare_prime_rl_checkout_for_install(prime_rl_dir: Path) -> None:
    if not (prime_rl_dir / ".git").is_dir():
        raise RuntimeError(f"Prime-RL runtime dir is not a Git checkout: {prime_rl_dir}")
    log("Preparing Prime-RL submodules with HTTPS GitHub URLs")
    key = "url.https://github.com/.insteadOf"
    existing = subprocess.run(
        ["git", "config", "--local", "--get-all", key],
        cwd=str(prime_rl_dir),
        env=git_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    existing_values = set(existing.stdout.splitlines()) if existing.returncode == 0 else set()
    for value in ("git@github.com:", "ssh://git@github.com/"):
        if value not in existing_values:
            run_git(["config", "--local", "--add", key, value], cwd=prime_rl_dir, retry=False)
    run_git(["submodule", "sync", "--recursive"], cwd=prime_rl_dir, retry=False)
    run_git(["submodule", "update", "--init", "--recursive"], cwd=prime_rl_dir, retry=False)
    log("Prime-RL submodules are ready")


def baked_engine_path() -> Path:
    return Path(__file__).resolve().with_name("train_engine.py")


def engine_path_for_backend(engine_path: Path, forwarded_args: list[str]) -> Path:
    backend = forwarded_backend(forwarded_args)
    if backend == "prime_rl":
        return engine_path.with_name("train_engine_rl.py")
    if backend == "verl_opd":
        return engine_path.with_name("train_engine_verl.py")
    return engine_path


def build_pythonpath(
    engine_path: Path,
    env: dict[str, str],
    open_instruct_dir: Path | None = None,
    olmo_core_dir: Path | None = None,
    rlcsd_dir: Path | None = None,
    verl_dir: Path | None = None,
    prime_rl_dir: Path | None = None,
) -> str:
    parts = []
    if verl_dir is not None:
        parts.append(str(verl_dir))
    if rlcsd_dir is not None:
        parts.append(str(rlcsd_dir / "third_party" / "verl"))
        parts.append(str(rlcsd_dir))
    if prime_rl_dir is not None:
        parts.append(str(prime_rl_dir / "packages" / "prime-rl-configs" / "src"))
        parts.append(str(prime_rl_dir / "src"))
    if open_instruct_dir is not None:
        parts.append(str(open_instruct_dir))
    if olmo_core_dir is not None:
        parts.append(str(olmo_core_dir / "src"))
    parts.extend(
        [
            str(engine_path.parent),
            str(engine_path.parent.parent),
            "/app",
        ]
    )
    existing = env.get("PYTHONPATH")
    if existing:
        parts.extend(existing.split(os.pathsep))
    return os.pathsep.join(dict.fromkeys(part for part in parts if part))


def prepend_pythonpath(*paths: Path | str) -> None:
    parts = [str(path) for path in paths if str(path)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.extend(existing.split(os.pathsep))
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def prepend_path(*paths: Path | str) -> None:
    parts = [str(path) for path in paths if str(path)]
    existing = os.environ.get("PATH")
    if existing:
        parts.extend(existing.split(os.pathsep))
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def prepend_ld_library_path(*paths: Path | str) -> None:
    parts = [str(path) for path in paths if str(path) and Path(path).exists()]
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        parts.extend(existing.split(os.pathsep))
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def find_system_nccl_library() -> Path | None:
    configured = os.environ.get("PRIME_RL_SYSTEM_NCCL_PATH") or os.environ.get("TRAIN_WRAPPER_SYSTEM_NCCL_PATH")
    candidates = [
        configured,
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/lib/x86_64-linux-gnu/libnccl.so",
        "/usr/local/cuda/lib64/libnccl.so.2",
        "/usr/local/cuda/lib64/libnccl.so",
    ]
    return next((Path(path) for path in candidates if path and Path(path).exists()), None)


def enable_system_nccl_preload_for_wrapper() -> None:
    if not parse_bool(os.environ.get("TRAIN_WRAPPER_PRELOAD_SYSTEM_NCCL"), True):
        return
    nccl_path = find_system_nccl_library()
    if nccl_path is None:
        log("WARNING: could not find a system NCCL library to preload before TE-dependent runtime probes.")
        return

    existing_preload = os.environ.get("LD_PRELOAD", "")
    preload_parts = [part for part in existing_preload.split() if part]
    if str(nccl_path) not in preload_parts:
        os.environ["LD_PRELOAD"] = " ".join([str(nccl_path), *preload_parts])
    prepend_ld_library_path(nccl_path.parent)
    log(f"Preloading system NCCL before TE-dependent runtime probes: {nccl_path}")


def prepend_runtime_library_path(site_dir: Path) -> None:
    prepend_ld_library_path(
        site_dir / "nvidia" / "nccl" / "lib",
        site_dir / "nvidia" / "cublasmp" / "cu13" / "lib",
    )


def exec_engine(
    engine_path: Path,
    forwarded_args: list[str],
    open_instruct_dir: Path | None = None,
    olmo_core_dir: Path | None = None,
    rlcsd_dir: Path | None = None,
    verl_dir: Path | None = None,
    prime_rl_dir: Path | None = None,
) -> None:
    selected_engine_path = engine_path_for_backend(engine_path, forwarded_args)
    if not selected_engine_path.is_file():
        raise FileNotFoundError(f"Could not find train engine: {selected_engine_path}")
    if selected_engine_path != engine_path:
        log(f"Selected backend engine {selected_engine_path} for backend {forwarded_backend(forwarded_args)!r}")
    env = os.environ.copy()
    if open_instruct_dir is not None:
        env["OPEN_INSTRUCT_DIR"] = str(open_instruct_dir)
    if olmo_core_dir is not None:
        env["OLMO_CORE_DIR"] = str(olmo_core_dir)
    if rlcsd_dir is not None:
        env["RLCSD_DIR"] = str(rlcsd_dir)
    if verl_dir is not None:
        env["VERL_DIR"] = str(verl_dir)
    if prime_rl_dir is not None:
        env["PRIME_RL_DIR"] = str(prime_rl_dir)
    runtime_bin = env.get("TRAIN_WRAPPER_RUNTIME_BIN")
    if runtime_bin:
        env["PATH"] = os.pathsep.join(dict.fromkeys([runtime_bin, *env.get("PATH", "").split(os.pathsep)]))
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = build_pythonpath(
        selected_engine_path,
        env,
        open_instruct_dir,
        olmo_core_dir,
        rlcsd_dir,
        verl_dir,
        prime_rl_dir,
    )
    command = [sys.executable, str(selected_engine_path), *forwarded_args]
    os.execvpe(sys.executable, command, env)


def exec_runtime_wrapper(
    wrapper_path: Path,
    raw_args: list[str],
    engine_path: Path,
    open_instruct_dir: Path,
    olmo_core_dir: Path,
    rlcsd_dir: Path,
    verl_dir: Path,
    prime_rl_dir: Path,
) -> None:
    if not wrapper_path.is_file():
        raise FileNotFoundError(f"Could not find runtime train wrapper: {wrapper_path}")
    env = os.environ.copy()
    env[WRAPPER_REEXEC_ENV] = "1"
    env[PREPARED_ENGINE_ENV] = str(engine_path)
    env[PREPARED_OPEN_INSTRUCT_ENV] = str(open_instruct_dir)
    env[PREPARED_OLMO_CORE_ENV] = str(olmo_core_dir)
    env[PREPARED_RLCSD_ENV] = str(rlcsd_dir)
    env[PREPARED_VERL_ENV] = str(verl_dir)
    env[PREPARED_PRIME_RL_ENV] = str(prime_rl_dir)
    command = [sys.executable, str(wrapper_path), *raw_args]
    log(f"Re-executing fetched train wrapper {wrapper_path}")
    os.execvpe(sys.executable, command, env)


def set_prepared_runtime_env(
    engine_path: Path,
    open_instruct_dir: Path,
    olmo_core_dir: Path,
    rlcsd_dir: Path,
    verl_dir: Path,
    prime_rl_dir: Path,
) -> None:
    os.environ[PREPARED_ENGINE_ENV] = str(engine_path)
    os.environ[PREPARED_OPEN_INSTRUCT_ENV] = str(open_instruct_dir)
    os.environ[PREPARED_OLMO_CORE_ENV] = str(olmo_core_dir)
    os.environ[PREPARED_RLCSD_ENV] = str(rlcsd_dir)
    os.environ[PREPARED_VERL_ENV] = str(verl_dir)
    os.environ[PREPARED_PRIME_RL_ENV] = str(prime_rl_dir)
    os.environ["OPEN_INSTRUCT_DIR"] = str(open_instruct_dir)
    os.environ["OLMO_CORE_DIR"] = str(olmo_core_dir)
    os.environ["RLCSD_DIR"] = str(rlcsd_dir)
    os.environ["VERL_DIR"] = str(verl_dir)
    os.environ["PRIME_RL_DIR"] = str(prime_rl_dir)


def resolve_runtime_dir(value: str | None, env_name: str, default: str) -> Path:
    return Path(value or os.environ.get(env_name) or default).expanduser()


def runtime_repo_settings(wrapper_args: argparse.Namespace) -> dict[str, str]:
    return {
        "submissions_repo": wrapper_args.submissions_repo
        or os.environ.get("SUBMISSIONS_REPO")
        or DEFAULT_SUBMISSIONS_REPO,
        "submissions_ref": wrapper_args.submissions_ref or os.environ.get("SUBMISSIONS_REF") or DEFAULT_SUBMISSIONS_REF,
        "submissions_runtime_dir": str(
            resolve_runtime_dir(wrapper_args.submissions_runtime_dir, "SUBMISSIONS_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)
        ),
        "open_instruct_repo": wrapper_args.open_instruct_repo
        or os.environ.get("OPEN_INSTRUCT_REPO")
        or DEFAULT_OPEN_INSTRUCT_REPO,
        "open_instruct_ref": wrapper_args.open_instruct_ref
        or os.environ.get("OPEN_INSTRUCT_REF")
        or DEFAULT_OPEN_INSTRUCT_REF,
        "open_instruct_runtime_dir": str(
            resolve_runtime_dir(
                wrapper_args.open_instruct_runtime_dir,
                "OPEN_INSTRUCT_RUNTIME_DIR",
                DEFAULT_OPEN_INSTRUCT_RUNTIME_DIR,
            )
        ),
        "olmo_core_repo": wrapper_args.olmo_core_repo or os.environ.get("OLMO_CORE_REPO") or DEFAULT_OLMO_CORE_REPO,
        "olmo_core_ref": wrapper_args.olmo_core_ref or os.environ.get("OLMO_CORE_REF") or DEFAULT_OLMO_CORE_REF,
        "olmo_core_runtime_dir": str(
            resolve_runtime_dir(
                wrapper_args.olmo_core_runtime_dir,
                "OLMO_CORE_RUNTIME_DIR",
                DEFAULT_OLMO_CORE_RUNTIME_DIR,
            )
        ),
        "rlcsd_repo": wrapper_args.rlcsd_repo or os.environ.get("RLCSD_REPO") or DEFAULT_RLCSD_REPO,
        "rlcsd_ref": wrapper_args.rlcsd_ref or os.environ.get("RLCSD_REF") or DEFAULT_RLCSD_REF,
        "rlcsd_runtime_dir": str(
            resolve_runtime_dir(
                wrapper_args.rlcsd_runtime_dir,
                "RLCSD_RUNTIME_DIR",
                DEFAULT_RLCSD_RUNTIME_DIR,
            )
        ),
        "verl_repo": wrapper_args.verl_repo or os.environ.get("VERL_REPO") or DEFAULT_VERL_REPO,
        "verl_ref": wrapper_args.verl_ref or os.environ.get("VERL_REF") or DEFAULT_VERL_REF,
        "verl_runtime_dir": str(
            resolve_runtime_dir(
                wrapper_args.verl_runtime_dir,
                "VERL_RUNTIME_DIR",
                DEFAULT_VERL_RUNTIME_DIR,
            )
        ),
        "prime_rl_repo": wrapper_args.prime_rl_repo or os.environ.get("PRIME_RL_REPO") or DEFAULT_PRIME_RL_REPO,
        "prime_rl_ref": wrapper_args.prime_rl_ref or os.environ.get("PRIME_RL_REF") or DEFAULT_PRIME_RL_REF,
        "prime_rl_runtime_dir": str(
            resolve_runtime_dir(
                wrapper_args.prime_rl_runtime_dir,
                "PRIME_RL_RUNTIME_DIR",
                DEFAULT_PRIME_RL_RUNTIME_DIR,
            )
        ),
    }


def fetch_runtime_repos(wrapper_args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    settings = runtime_repo_settings(wrapper_args)
    submissions_repo = settings["submissions_repo"]
    submissions_ref = settings["submissions_ref"]
    submissions_runtime_dir = Path(settings["submissions_runtime_dir"])
    submissions_dir = ensure_runtime_repo(submissions_repo, submissions_ref, submissions_runtime_dir, "submissions")

    open_instruct_repo = settings["open_instruct_repo"]
    open_instruct_ref = settings["open_instruct_ref"]
    open_instruct_runtime_dir = Path(settings["open_instruct_runtime_dir"])
    open_instruct_dir = ensure_runtime_repo(
        open_instruct_repo,
        open_instruct_ref,
        open_instruct_runtime_dir,
        "open-instruct",
    )

    olmo_core_repo = settings["olmo_core_repo"]
    olmo_core_ref = settings["olmo_core_ref"]
    olmo_core_runtime_dir = Path(settings["olmo_core_runtime_dir"])
    olmo_core_dir = ensure_runtime_repo(
        olmo_core_repo,
        olmo_core_ref,
        olmo_core_runtime_dir,
        "OLMo-core",
    )

    rlcsd_repo = settings["rlcsd_repo"]
    rlcsd_ref = settings["rlcsd_ref"]
    rlcsd_runtime_dir = Path(settings["rlcsd_runtime_dir"])
    rlcsd_dir = ensure_runtime_repo(
        rlcsd_repo,
        rlcsd_ref,
        rlcsd_runtime_dir,
        "RLCSD",
    )

    verl_repo = settings["verl_repo"]
    verl_ref = settings["verl_ref"]
    verl_runtime_dir = Path(settings["verl_runtime_dir"])
    verl_dir = ensure_runtime_repo(
        verl_repo,
        verl_ref,
        verl_runtime_dir,
        "VERL",
    )

    prime_rl_repo = settings["prime_rl_repo"]
    prime_rl_ref = settings["prime_rl_ref"]
    prime_rl_runtime_dir = Path(settings["prime_rl_runtime_dir"])
    prime_rl_dir = ensure_runtime_repo(
        prime_rl_repo,
        prime_rl_ref,
        prime_rl_runtime_dir,
        "Prime-RL",
    )

    return submissions_dir / "src" / "train_engine.py", open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir


def runtime_fetch_coordination_id() -> str:
    explicit = os.environ.get("RUNTIME_FETCH_COORDINATION_ID")
    if explicit:
        return f"explicit:{explicit}"
    for env_name in ("PBS_JOBID", "SLURM_JOB_ID", "JOB_ID", "LSB_JOBID"):
        value = os.environ.get(env_name)
        if value:
            return f"{env_name}:{value}"
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "")
    world_size = os.environ.get("WORLD_SIZE", "")
    if master_addr or master_port or world_size:
        return f"distributed:{master_addr}:{master_port}:{world_size}"
    return "single"


def runtime_fetch_state_path(wrapper_args: argparse.Namespace) -> tuple[Path, str]:
    settings = runtime_repo_settings(wrapper_args)
    payload = {
        "coordination_id": runtime_fetch_coordination_id(),
        "settings": settings,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    state_dir = resolve_runtime_dir(
        wrapper_args.runtime_fetch_state_dir,
        "RUNTIME_FETCH_STATE_DIR",
        DEFAULT_RUNTIME_FETCH_STATE_DIR,
    )
    return state_dir / f"{fingerprint}.json", fingerprint


def write_runtime_fetch_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.time()
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def read_runtime_fetch_state(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def runtime_paths_from_state(state: dict[str, object]) -> tuple[Path, Path, Path, Path, Path, Path]:
    return (
        Path(str(state["engine_path"])),
        Path(str(state["open_instruct_dir"])),
        Path(str(state["olmo_core_dir"])),
        Path(str(state["rlcsd_dir"])),
        Path(str(state["verl_dir"])),
        Path(str(state["prime_rl_dir"])),
    )


def validate_runtime_paths(
    engine_path: Path,
    open_instruct_dir: Path,
    olmo_core_dir: Path,
    rlcsd_dir: Path,
    verl_dir: Path,
    prime_rl_dir: Path,
) -> str | None:
    checks = [
        (engine_path.is_file(), f"missing train engine {engine_path}"),
        (engine_path.with_name("train_engine_rl.py").is_file(), f"missing RL train engine {engine_path.with_name('train_engine_rl.py')}"),
        (engine_path.with_name("train_engine_verl.py").is_file(), f"missing VERL train engine {engine_path.with_name('train_engine_verl.py')}"),
        ((open_instruct_dir / "open_instruct").is_dir(), f"missing open-instruct package {open_instruct_dir}"),
        ((olmo_core_dir / "src").is_dir(), f"missing OLMo-core src dir {olmo_core_dir}"),
        ((rlcsd_dir / "src").is_dir(), f"missing RLCSD src dir {rlcsd_dir}"),
        (
            (rlcsd_dir / "third_party" / "verl" / "verl" / "__init__.py").is_file(),
            f"missing RLCSD vendored verl package {rlcsd_dir}",
        ),
        ((verl_dir / "verl" / "__init__.py").is_file(), f"missing VERL package {verl_dir}"),
        ((prime_rl_dir / "src" / "prime_rl").is_dir(), f"missing Prime-RL src package {prime_rl_dir}"),
        (
            (prime_rl_dir / "packages" / "prime-rl-configs" / "src" / "prime_rl").is_dir(),
            f"missing Prime-RL config package {prime_rl_dir}",
        ),
    ]
    missing = [message for ok, message in checks if not ok]
    if missing:
        return "; ".join(missing)
    return None


def same_file_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left.absolute()) == str(right.absolute())


def prepared_runtime_repos_from_env() -> tuple[Path, Path, Path, Path, Path, Path] | None:
    if os.environ.get(WRAPPER_REEXEC_ENV) != "1":
        return None
    engine_path = os.environ.get(PREPARED_ENGINE_ENV)
    open_instruct_dir = os.environ.get(PREPARED_OPEN_INSTRUCT_ENV)
    olmo_core_dir = os.environ.get(PREPARED_OLMO_CORE_ENV)
    rlcsd_dir = os.environ.get(PREPARED_RLCSD_ENV)
    verl_dir = os.environ.get(PREPARED_VERL_ENV)
    prime_rl_dir = os.environ.get(PREPARED_PRIME_RL_ENV)
    if not (engine_path and open_instruct_dir and olmo_core_dir and rlcsd_dir and verl_dir and prime_rl_dir):
        return None
    paths = (
        Path(engine_path),
        Path(open_instruct_dir),
        Path(olmo_core_dir),
        Path(rlcsd_dir),
        Path(verl_dir),
        Path(prime_rl_dir),
    )
    missing = validate_runtime_paths(*paths)
    if missing:
        raise RuntimeError(f"Prepared runtime repo paths are invalid after wrapper re-exec: {missing}")
    return paths


def wrapper_self_update_enabled(wrapper_args: argparse.Namespace) -> bool:
    if wrapper_args.self_update_wrapper is not None:
        return bool(wrapper_args.self_update_wrapper)
    return parse_bool(os.environ.get("TRAIN_WRAPPER_SELF_UPDATE"), True)


def maybe_reexec_runtime_wrapper(
    wrapper_args: argparse.Namespace,
    raw_args: list[str],
    engine_path: Path,
    open_instruct_dir: Path,
    olmo_core_dir: Path,
    rlcsd_dir: Path,
    verl_dir: Path,
    prime_rl_dir: Path,
) -> None:
    if not wrapper_self_update_enabled(wrapper_args):
        log("Wrapper self-update disabled; executing fetched train engine directly.")
        return
    if os.environ.get(WRAPPER_REEXEC_ENV) == "1":
        return
    runtime_wrapper_path = engine_path.with_name("train.py")
    exec_runtime_wrapper(
        runtime_wrapper_path,
        raw_args,
        engine_path,
        open_instruct_dir,
        olmo_core_dir,
        rlcsd_dir,
        verl_dir,
        prime_rl_dir,
    )


def runtime_fetch_timeout(wrapper_args: argparse.Namespace) -> float:
    return parse_float(
        str(wrapper_args.runtime_fetch_timeout) if wrapper_args.runtime_fetch_timeout is not None else None,
        parse_float(os.environ.get("RUNTIME_FETCH_TIMEOUT"), DEFAULT_RUNTIME_FETCH_TIMEOUT_SECONDS),
    )


def runtime_fetch_poll_interval(wrapper_args: argparse.Namespace) -> float:
    return max(
        0.1,
        parse_float(
            str(wrapper_args.runtime_fetch_poll_interval)
            if wrapper_args.runtime_fetch_poll_interval is not None
            else None,
            parse_float(os.environ.get("RUNTIME_FETCH_POLL_INTERVAL"), DEFAULT_RUNTIME_FETCH_POLL_SECONDS),
        ),
    )


def runtime_fetch_stale_failed_seconds() -> float:
    return max(
        0.0,
        parse_float(
            os.environ.get("RUNTIME_FETCH_STALE_FAILED_SECONDS"),
            DEFAULT_RUNTIME_FETCH_STALE_FAILED_SECONDS,
        ),
    )


def runtime_training_deps_enabled(wrapper_args: argparse.Namespace) -> bool:
    if wrapper_args.ensure_runtime_training_deps is not None:
        return parse_bool(str(wrapper_args.ensure_runtime_training_deps), True)
    return parse_bool(os.environ.get("ENSURE_RUNTIME_TRAINING_DEPS"), True)


def runtime_training_deps_settings(wrapper_args: argparse.Namespace) -> dict[str, str]:
    return {
        "base_dir": str(
            resolve_runtime_dir(
                wrapper_args.runtime_training_deps_dir,
                "RUNTIME_TRAINING_DEPS_DIR",
                DEFAULT_RUNTIME_TRAINING_DEPS_DIR,
            )
        ),
        "megatron_repo": wrapper_args.megatron_core_repo
        or os.environ.get("MEGATRON_CORE_REPO")
        or DEFAULT_MEGATRON_CORE_REPO,
        "megatron_ref": wrapper_args.megatron_core_ref
        or os.environ.get("MEGATRON_CORE_REF")
        or DEFAULT_MEGATRON_CORE_REF,
        "liger_repo": wrapper_args.liger_kernel_repo
        or os.environ.get("LIGER_KERNEL_REPO")
        or DEFAULT_LIGER_KERNEL_REPO,
        "liger_ref": wrapper_args.liger_kernel_ref
        or os.environ.get("LIGER_KERNEL_REF")
        or DEFAULT_LIGER_KERNEL_REF,
        "apex_wheel_repo": wrapper_args.apex_wheel_repo
        or os.environ.get("APEX_WHEEL_REPO")
        or DEFAULT_APEX_WHEEL_REPO,
        "apex_wheel_file": wrapper_args.apex_wheel_file
        or os.environ.get("APEX_WHEEL_FILE")
        or DEFAULT_APEX_WHEEL_FILE,
        "transformer_engine_wheel_repo": wrapper_args.transformer_engine_wheel_repo
        or os.environ.get("TRANSFORMER_ENGINE_WHEEL_REPO")
        or DEFAULT_TRANSFORMER_ENGINE_WHEEL_REPO,
        "transformer_engine_wheel_file": wrapper_args.transformer_engine_wheel_file
        or os.environ.get("TRANSFORMER_ENGINE_WHEEL_FILE")
        or DEFAULT_TRANSFORMER_ENGINE_WHEEL_FILE,
        "verl_nvidia_runtime_requirements": verl_nvidia_runtime_requirements_string(),
        "grpo_runtime_requirements": grpo_runtime_requirements_string(),
        "rlcsd_runtime_requirements": rlcsd_runtime_requirements_string(),
        "verl_automodel_runtime_requirements": verl_automodel_runtime_requirements_string(),
        "prime_rl_runtime_requirements": prime_rl_runtime_requirements_string(),
        "prime_rl_source_requirements": prime_rl_source_requirements_string(),
        "prime_rl_runtime_pip_no_deps": str(prime_rl_runtime_pip_no_deps_enabled()).lower(),
        "prime_rl_install_from_git": str(parse_bool(os.environ.get("PRIME_RL_INSTALL_FROM_GIT"), False)).lower(),
        "prime_rl_package_requirement": os.environ.get("PRIME_RL_PACKAGE_REQUIREMENT", "").strip(),
        "grpo_runtime_pip_no_deps": str(grpo_runtime_pip_no_deps_enabled()).lower(),
        "grpo_runtime_vllm_version": os.environ.get(
            "GRPO_RUNTIME_VLLM_VERSION", DEFAULT_GRPO_RUNTIME_VLLM_VERSION
        ).strip(),
        "grpo_runtime_vllm_cuda_version": os.environ.get(
            "GRPO_RUNTIME_VLLM_CUDA_VERSION", DEFAULT_GRPO_RUNTIME_VLLM_CUDA_VERSION
        ).strip(),
        "grpo_runtime_vllm_wheel_url": os.environ.get(
            "GRPO_RUNTIME_VLLM_WHEEL_URL", DEFAULT_VLLM_RUNTIME_WHEEL_URL
        ).strip(),
        "grpo_runtime_vllm_arch": os.environ.get("GRPO_RUNTIME_VLLM_ARCH", "").strip(),
    }


def runtime_training_deps_paths(wrapper_args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    settings = runtime_training_deps_settings(wrapper_args)
    payload = {
        "coordination_id": runtime_fetch_coordination_id(),
        "settings": settings,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    install_root = Path(settings["base_dir"]) / fingerprint
    python_user_site = (
        install_root
        / "userbase"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    return (
        install_root,
        python_user_site,
        install_root / "Megatron-LM",
        install_root / "Liger-Kernel",
        Path(settings["base_dir"]) / "markers" / f"{fingerprint}.json",
    )


def runtime_dependency_env(site_dir: Path, megatron_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    parts = [str(site_dir), str(megatron_dir)]
    system_nccl_path = find_system_nccl_library()
    library_parts = [
        system_nccl_path.parent if system_nccl_path else None,
        site_dir / "nvidia" / "nccl" / "lib",
        site_dir / "nvidia" / "cublasmp" / "cu13" / "lib",
    ]
    existing_library_path = env.get("LD_LIBRARY_PATH")
    if existing_library_path:
        library_parts.extend(Path(part) for part in existing_library_path.split(os.pathsep) if part)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys(str(path) for path in library_parts if path and path.exists())
    )
    if system_nccl_path is not None and parse_bool(env.get("TRAIN_WRAPPER_PRELOAD_SYSTEM_NCCL"), True):
        preload_parts = [part for part in env.get("LD_PRELOAD", "").split() if part]
        if str(system_nccl_path) not in preload_parts:
            env["LD_PRELOAD"] = " ".join([str(system_nccl_path), *preload_parts])
    target_bin_dir = site_dir / "bin"
    userbase_bin_dir = site_dir.parent.parent.parent / "bin"
    env["PATH"] = os.pathsep.join(
        dict.fromkeys([str(target_bin_dir), str(userbase_bin_dir), *env.get("PATH", "").split(os.pathsep)])
    )
    prepared_engine = env.get(PREPARED_ENGINE_ENV)
    if prepared_engine:
        parts.append(str(Path(prepared_engine).parent))
    prepared_open_instruct = env.get(PREPARED_OPEN_INSTRUCT_ENV)
    if prepared_open_instruct:
        parts.append(prepared_open_instruct)
    prepared_olmo_core = env.get(PREPARED_OLMO_CORE_ENV)
    if prepared_olmo_core:
        parts.append(str(Path(prepared_olmo_core) / "src"))
    prepared_verl = env.get(PREPARED_VERL_ENV) or env.get("VERL_DIR")
    if prepared_verl:
        verl_dir = Path(prepared_verl)
        env["VERL_DIR"] = str(verl_dir)
        parts.append(str(verl_dir))
    prepared_rlcsd = env.get(PREPARED_RLCSD_ENV) or env.get("RLCSD_DIR")
    if prepared_rlcsd:
        rlcsd_dir = Path(prepared_rlcsd)
        env["RLCSD_DIR"] = str(rlcsd_dir)
        parts.append(str(rlcsd_dir / "third_party" / "verl"))
        parts.append(str(rlcsd_dir))
    prepared_prime_rl = env.get(PREPARED_PRIME_RL_ENV) or env.get("PRIME_RL_DIR")
    if prepared_prime_rl:
        prime_rl_dir = Path(prepared_prime_rl)
        env["PRIME_RL_DIR"] = str(prime_rl_dir)
        parts.append(str(prime_rl_dir / "packages" / "prime-rl-configs" / "src"))
        parts.append(str(prime_rl_dir / "src"))
    if env.get("PYTHONPATH"):
        parts.extend(env["PYTHONPATH"].split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(part for part in parts if part))
    return env


def runtime_dependency_probe(
    site_dir: Path,
    megatron_dir: Path,
    *,
    module: str | None = None,
) -> tuple[bool, str]:
    if module == "apex":
        source = "import apex; from apex.optimizers import FusedAdam; print(apex.__file__); print(FusedAdam)"
    elif module == "megatron":
        source = (
            "import megatron.core; "
            "from megatron.core.transformer.transformer_config import TransformerConfig; "
            "config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4); "
            "print(megatron.core.__file__); print(config.num_layers, config.hidden_size)"
        )
    elif module == "liger":
        source = (
            "import liger_kernel; "
            "from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm; "
            "print(liger_kernel.__file__); print(LigerMegatronRMSNorm)"
        )
    elif module == "transformer_engine":
        source = (
            "import transformer_engine; "
            "from transformer_engine.pytorch.optimizers import FusedAdam; "
            "print(transformer_engine.__file__); print(FusedAdam)"
        )
    elif module == "grpo":
        source = (
            "import importlib.metadata; import openenv; import nltk; import litellm; import debugpy; "
            "import gguf; import vllm; "
            "import open_instruct.grpo_fast as grpo_fast; "
            "print('openenv', openenv.__file__); "
            "print('nltk', nltk.__version__); "
            "print('litellm', importlib.metadata.version('litellm')); "
            "print('vllm', getattr(vllm, '__version__', '<unknown>')); "
            "print('grpo_fast', grpo_fast.__file__)"
        )
    elif module == "rlcsd":
        source = (
            "import hydra; import omegaconf; import ray; import datasets; import pyarrow; "
            "import pandas; import openai; import codetiming; import tensordict; "
            "import math_verify; import deepspeed; import peft; "
            "import verl.trainer.main_ppo as main_ppo; "
            "print('hydra', hydra.__version__); "
            "print('omegaconf', omegaconf.__version__); "
            "print('ray', ray.__version__); "
            "print('deepspeed', deepspeed.__version__); "
            "print('peft', peft.__version__); "
            "print('verl main_ppo', main_ppo.__file__)"
        )
    elif module == "nemo_automodel":
        source = (
            "import nemo_automodel; "
            "from nemo_automodel.components.quantization.fp8 import FP8Config; "
            "print('nemo_automodel', nemo_automodel.__file__); "
            "print('FP8Config', FP8Config)"
        )
    elif module == "prime_rl":
        source = "print('Prime-RL import preflight skipped')"
    elif module == "prime_rl_env":
        source = "print('Prime-RL environment preflight skipped')"
    else:
        source = (
            "import apex; from apex.optimizers import FusedAdam; "
            "import megatron.core; "
            "from megatron.core.transformer.transformer_config import TransformerConfig; "
            "import liger_kernel; from liger_kernel.megatron.rms_norm import LigerMegatronRMSNorm; "
            "config=TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4); "
            "print('apex', apex.__file__); print('FusedAdam', FusedAdam); "
            "print('megatron.core', megatron.core.__file__); "
            "print('TransformerConfig', config.num_layers, config.hidden_size); "
            "print('liger_kernel', liger_kernel.__file__); print('LigerMegatronRMSNorm', LigerMegatronRMSNorm)"
        )
    process = subprocess.run(
        [sys.executable, "-c", source],
        env=runtime_dependency_env(site_dir, megatron_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    details = "\n".join(part.strip() for part in (process.stdout, process.stderr) if part.strip())
    if process.returncode != 0:
        details = f"returncode={process.returncode}\n{details}".strip()
    return process.returncode == 0, redact_secret(details)


def forwarded_backend(forwarded_args: list[str]) -> str | None:
    return forwarded_option_value(forwarded_args, "--backend")


def grpo_runtime_dependencies_required(forwarded_args: list[str]) -> bool:
    env_override = os.environ.get("ENSURE_GRPO_RUNTIME_DEPS")
    if env_override is not None:
        return parse_bool(env_override, False)
    return forwarded_backend(forwarded_args) == "grpo_fast"


def rlcsd_runtime_dependencies_required(forwarded_args: list[str]) -> bool:
    env_override = os.environ.get("ENSURE_RLCSD_RUNTIME_DEPS")
    if env_override is not None:
        return parse_bool(env_override, False)
    return forwarded_backend(forwarded_args) in {"verl_rlcsd", "verl_opd"}


def prime_rl_runtime_dependencies_required(forwarded_args: list[str]) -> bool:
    env_override = os.environ.get("ENSURE_PRIME_RL_RUNTIME_DEPS")
    if env_override is not None:
        return parse_bool(env_override, False)
    return forwarded_backend(forwarded_args) == "prime_rl"


def transformer_engine_runtime_required(forwarded_args: list[str]) -> bool:
    env_override = os.environ.get("ENSURE_TRANSFORMER_ENGINE_RUNTIME_DEP")
    if env_override is not None:
        return parse_bool(env_override, False)
    optimizer = forwarded_option_value(forwarded_args, "--optimizer")
    if optimizer and optimizer.lower() == "te_fused_adamw":
        return True
    if verl_training_fp8_required(forwarded_args):
        return True
    verl_automodel_optimizer_impl = forwarded_option_value(
        forwarded_args,
        "--verl_automodel_optimizer_impl",
        "--verl-automodel-optimizer-impl",
    )
    if verl_automodel_optimizer_impl and "transformer_engine" in verl_automodel_optimizer_impl:
        return True
    verl_automodel_backend_linear = forwarded_option_value(
        forwarded_args,
        "--verl_automodel_backend_linear",
        "--verl-automodel-backend-linear",
    )
    if verl_automodel_backend_linear and verl_automodel_backend_linear.lower() == "te":
        return True
    attn = forwarded_option_value(
        forwarded_args,
        "--attn_implementation",
        "--attn-implementation",
    )
    return bool(attn and attn.lower() in {"te", "te_attn", "transformer_engine"})


def grpo_runtime_requirements() -> list[str]:
    override = os.environ.get("GRPO_RUNTIME_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_GRPO_RUNTIME_REQUIREMENTS)


def grpo_runtime_requirements_string() -> str:
    return "\n".join(grpo_runtime_requirements())


def rlcsd_runtime_requirements() -> list[str]:
    override = os.environ.get("RLCSD_RUNTIME_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_RLCSD_RUNTIME_REQUIREMENTS)


def rlcsd_runtime_requirements_string() -> str:
    return "\n".join(rlcsd_runtime_requirements())


def verl_training_fp8_required(forwarded_args: list[str]) -> bool:
    env_override = os.environ.get("ENSURE_VERL_AUTOMODEL_RUNTIME_DEPS")
    if env_override is not None:
        return parse_bool(env_override, False)
    if forwarded_backend(forwarded_args) != "verl_opd":
        return False
    return parse_bool(
        forwarded_option_value(forwarded_args, "--verl_training_fp8", "--verl-training-fp8"),
        False,
    )


def verl_automodel_runtime_requirements() -> list[str]:
    override = os.environ.get("VERL_AUTOMODEL_RUNTIME_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_VERL_AUTOMODEL_RUNTIME_REQUIREMENTS)


def verl_automodel_runtime_requirements_string() -> str:
    return "\n".join(verl_automodel_runtime_requirements())


def verl_nvidia_runtime_requirements() -> list[str]:
    override = os.environ.get("VERL_NVIDIA_RUNTIME_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_VERL_NVIDIA_RUNTIME_REQUIREMENTS)


def verl_nvidia_runtime_requirements_string() -> str:
    return "\n".join(verl_nvidia_runtime_requirements())


def prime_rl_runtime_requirements() -> list[str]:
    override = os.environ.get("PRIME_RL_RUNTIME_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_PRIME_RL_RUNTIME_REQUIREMENTS)


def prime_rl_runtime_requirements_string() -> str:
    return "\n".join(prime_rl_runtime_requirements())


def prime_rl_source_requirements() -> list[str]:
    override = os.environ.get("PRIME_RL_SOURCE_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return list(DEFAULT_PRIME_RL_SOURCE_REQUIREMENTS)


def prime_rl_source_requirements_string() -> str:
    return "\n".join(prime_rl_source_requirements())


def prime_rl_runtime_pip_no_deps_enabled() -> bool:
    return parse_bool(os.environ.get("PRIME_RL_RUNTIME_PIP_NO_DEPS"), False)


def prime_rl_build_requirements() -> list[str]:
    override = os.environ.get("PRIME_RL_BUILD_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    return ["hatchling", "editables"]


def prime_rl_config_requirements(prime_rl_dir: Path) -> list[str]:
    override = os.environ.get("PRIME_RL_CONFIG_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    requirements = []
    for relative_path in (
        "deps/pydantic-config",
        "packages/prime-rl-configs",
    ):
        package_dir = prime_rl_dir / relative_path
        if package_dir.is_dir():
            requirements.append(str(package_dir))
    return requirements


def remove_incompatible_prime_rl_config_package() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pydantic_config; raise SystemExit(0 if hasattr(pydantic_config, 'BaseConfig') else 1)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode == 0:
        return
    log("Removing incompatible PyPI pydantic-config package before installing Prime-RL config package")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "pydantic-config",
            "--break-system-packages",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def prime_rl_package_requirement(prime_rl_dir: Path) -> str:
    override = os.environ.get("PRIME_RL_PACKAGE_REQUIREMENT")
    if override:
        return override
    if parse_bool(os.environ.get("PRIME_RL_INSTALL_FROM_GIT"), False):
        repo = os.environ.get("PRIME_RL_REPO", DEFAULT_PRIME_RL_REPO)
        ref = os.environ.get("PRIME_RL_REF", DEFAULT_PRIME_RL_REF)
        return f"git+{repo}@{ref}"
    return "."


def prime_rl_install_requirements(prime_rl_dir: Path) -> list[str]:
    requirements = [
        *prime_rl_source_requirements(),
        prime_rl_package_requirement(prime_rl_dir),
    ]
    return [requirement for requirement in requirements if requirement]


def prime_rl_environment_requirements(prime_rl_dir: Path) -> list[str]:
    override = os.environ.get("PRIME_RL_ENVIRONMENT_REQUIREMENTS")
    if override:
        return [requirement for requirement in shlex.split(override) if requirement]
    requirements = []
    for relative_path in (
        "deps/research-environments/environments/math_env",
    ):
        package_dir = prime_rl_dir / relative_path
        if package_dir.is_dir():
            requirements.append(str(package_dir))
    return requirements


def grpo_runtime_pip_no_deps_enabled() -> bool:
    return parse_bool(os.environ.get("GRPO_RUNTIME_PIP_NO_DEPS"), True)


def grpo_runtime_vllm_wheel_url(settings: dict[str, str]) -> str | None:
    explicit_url = settings.get("grpo_runtime_vllm_wheel_url", "").strip()
    if explicit_url:
        return explicit_url
    version = settings.get("grpo_runtime_vllm_version", "").strip()
    if not version:
        return None
    if version.lower() in {"0", "false", "none", "off", "skip", "disabled"}:
        return None
    if version.startswith(("http://", "https://")):
        return version
    version = version.removeprefix("v")
    cuda_version = settings.get("grpo_runtime_vllm_cuda_version", DEFAULT_GRPO_RUNTIME_VLLM_CUDA_VERSION).strip()
    cuda_version = cuda_version.removeprefix("cu")
    arch = settings.get("grpo_runtime_vllm_arch", "").strip() or os.uname().machine
    if version in {"0.20.0", "0.20.1", "0.20.2"} and cuda_version in {"", "130", "default", "auto"}:
        return (
            f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
            f"vllm-{version}-cp38-abi3-manylinux_2_35_{arch}.whl"
        )
    if version == "0.23.0" and cuda_version in {"", "130", "default", "auto"}:
        return (
            f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
            f"vllm-{version}-cp38-abi3-manylinux_2_28_{arch}.whl"
        )
    return (
        f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
        f"vllm-{version}+cu{cuda_version}-cp38-abi3-manylinux_2_35_{arch}.whl"
    )


def maybe_install_runtime_vllm_override(settings: dict[str, str], site_dir: Path) -> bool:
    wheel_url = grpo_runtime_vllm_wheel_url(settings)
    if not wheel_url:
        return False
    label = settings.get("grpo_runtime_vllm_version", "").strip() or Path(wheel_url).name
    log(f"Installing runtime vLLM override {label} from {wheel_url}")
    install_python_target(wheel_url, site_dir, "vLLM override")
    return True


def remove_runtime_vllm_override(site_dir: Path) -> None:
    if not site_dir.is_dir():
        return
    removed: list[str] = []
    for path in sorted(site_dir.iterdir(), key=lambda item: item.name):
        lower_name = path.name.lower()
        if not (lower_name == "vllm" or lower_name.startswith("vllm-") or lower_name.startswith("vllm_")):
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
        except OSError as exc:
            log(f"WARNING: could not remove runtime vLLM override path {path}: {exc}")
    if removed:
        log("Removed runtime vLLM override after failed import verification: " + ", ".join(removed))


def normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def runtime_overlay_package_matches(path: Path, protected_names: set[str]) -> bool:
    name = path.name
    if name in {"__pycache__", "bin"}:
        return False
    lower_name = name.lower()
    dist_stem = lower_name.removesuffix(".dist-info").removesuffix(".egg-info")
    dist_name_without_version = re.sub(r"-\d.*$", "", dist_stem)
    candidates = {
        normalize_distribution_name(lower_name),
        normalize_distribution_name(dist_stem),
        normalize_distribution_name(dist_name_without_version),
    }
    if lower_name.endswith((".dist-info", ".egg-info")):
        metadata_path = path / "METADATA"
        if metadata_path.is_file():
            try:
                for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.lower().startswith("name:"):
                        candidates.add(normalize_distribution_name(line.split(":", 1)[1].strip()))
                        break
            except OSError:
                pass
    if any(candidate in protected_names for candidate in candidates):
        return True
    return any(
        lower_name == prefix or lower_name.startswith(f"{prefix}-") or lower_name.startswith(f"{prefix}_")
        for prefix in ("nvidia", "cuda")
    )


def remove_protected_runtime_overlay_packages(site_dir: Path) -> None:
    if not site_dir.is_dir():
        return
    protected_names = {normalize_distribution_name(name) for name in PROTECTED_RUNTIME_OVERLAY_PACKAGES}
    removed: list[str] = []
    for path in sorted(site_dir.iterdir(), key=lambda item: item.name):
        if not runtime_overlay_package_matches(path, protected_names):
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
        except OSError as exc:
            log(f"WARNING: could not remove protected runtime overlay package {path}: {exc}")
    if removed:
        preview = ", ".join(removed[:30])
        suffix = "" if len(removed) <= 30 else f", ... (+{len(removed) - 30} more)"
        log(
            "Removed protected runtime overlay packages so the baked image stack remains active: "
            f"{preview}{suffix}"
        )


def install_python_target(source: Path | str, site_dir: Path, label: str, *, no_deps: bool = True) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    if "github.com" in str(source):
        configure_git_https_rewrites_for_runtime_installs()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--ignore-requires-python",
        "--no-input",
        "--no-cache-dir",
        "--upgrade",
        "--target",
        str(site_dir),
        str(source),
    ]
    if no_deps:
        command.insert(command.index("--target"), "--no-deps")
    mode = "without dependency resolution" if no_deps else "with dependency resolution"
    log(f"Installing runtime {label} into {site_dir} {mode}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Runtime {label} install failed with exit code {process.returncode}.\n"
            f"stdout:\n{redact_secret(process.stdout)}\n"
            f"stderr:\n{redact_secret(process.stderr)}"
        )
    if not no_deps:
        remove_protected_runtime_overlay_packages(site_dir)
    log(f"Runtime {label} install complete")


def install_python_global_requirements(
    requirements: list[str],
    label: str,
    *,
    cwd: Path | None = None,
    editable: bool = False,
    no_build_isolation: bool = False,
    no_deps: bool = False,
    upgrade: bool = False,
) -> None:
    if not requirements:
        return
    if any("github.com" in requirement for requirement in requirements):
        configure_git_https_rewrites_for_runtime_installs()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--ignore-requires-python",
        "--no-input",
        "--no-cache-dir",
        "--break-system-packages",
    ]
    if no_deps:
        command.append("--no-deps")
    if upgrade:
        command.append("--upgrade")
    if no_build_isolation:
        command.append("--no-build-isolation")
    for requirement in requirements:
        if editable:
            command.extend(["-e", requirement])
        else:
            command.append(requirement)
    mode_parts = []
    if no_deps:
        mode_parts.append("without dependency resolution")
    if upgrade:
        mode_parts.append("with upgrade")
    mode = f" ({', '.join(mode_parts)})" if mode_parts else ""
    log(f"Installing global runtime {label} requirements{mode}: {' '.join(requirements)}")
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Global runtime {label} requirements install failed with exit code {process.returncode}.\n"
            f"stdout:\n{redact_secret(process.stdout)}\n"
            f"stderr:\n{redact_secret(process.stderr)}"
        )
    log(f"Global runtime {label} requirements install complete")


def configure_git_https_rewrites_for_runtime_installs() -> None:
    rewrites = (
        ("url.https://github.com/.insteadOf", "git@github.com:"),
        ("url.https://github.com/.insteadOf", "ssh://git@github.com/"),
    )
    for key, value in rewrites:
        existing = subprocess.run(
            ["git", "config", "--global", "--get-all", key],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if value in existing.stdout.splitlines():
            continue
        process = subprocess.run(
            ["git", "config", "--global", "--add", key, value],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            log(
                "WARNING: could not configure GitHub HTTPS rewrite for runtime install: "
                f"{key}={value}: {redact_secret(process.stderr.strip())}"
            )


def install_python_requirements(
    requirements: list[str],
    site_dir: Path,
    label: str,
    *,
    no_deps: bool | None = None,
    no_build_isolation: bool = False,
    cwd: Path | None = None,
) -> None:
    if not requirements:
        return
    site_dir.mkdir(parents=True, exist_ok=True)
    if no_deps is None:
        no_deps = grpo_runtime_pip_no_deps_enabled()
    if any("github.com" in requirement for requirement in requirements):
        configure_git_https_rewrites_for_runtime_installs()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--ignore-requires-python",
        "--no-input",
        "--no-cache-dir",
        "--break-system-packages",
        "--target",
        str(site_dir),
        *requirements,
    ]
    if no_build_isolation:
        command.insert(command.index("--target"), "--no-build-isolation")
    if no_deps:
        command.insert(command.index("--target"), "--no-deps")
    mode = "without dependency resolution" if no_deps else "with dependency resolution"
    log(
        f"Installing runtime {label} requirements into {site_dir} {mode}: "
        f"{' '.join(requirements)}"
    )
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Runtime {label} requirements install failed with exit code {process.returncode}.\n"
            f"stdout:\n{redact_secret(process.stdout)}\n"
            f"stderr:\n{redact_secret(process.stderr)}"
        )
    if not no_deps:
        remove_protected_runtime_overlay_packages(site_dir)
    log(f"Runtime {label} requirements install complete")


def remove_grpo_runtime_sitecustomize(site_dir: Path) -> None:
    sitecustomize_path = site_dir / "sitecustomize.py"
    marker = "# OLMO_GRPO_HIDE_BROKEN_TRANSFORMER_ENGINE"
    if not sitecustomize_path.is_file():
        return
    existing = sitecustomize_path.read_text(encoding="utf-8")
    if marker not in existing:
        return
    preserved = existing.split(marker, 1)[0].rstrip()
    if preserved:
        sitecustomize_path.write_text(preserved + "\n", encoding="utf-8")
    else:
        sitecustomize_path.unlink()
    log(f"Removed obsolete GRPO Transformer Engine sitecustomize shim: {sitecustomize_path}")


def patch_runtime_peft_transformer_engine_probe(site_dir: Path) -> None:
    peft_spec = importlib.util.find_spec("peft")
    if peft_spec is None or peft_spec.origin is None:
        raise RuntimeError("Could not locate the installed PEFT package for GRPO runtime compatibility")
    source_dir = Path(peft_spec.origin).resolve().parent
    target_dir = site_dir / "peft"
    if source_dir != target_dir.resolve():
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    import_utils_path = target_dir / "import_utils.py"
    source = import_utils_path.read_text(encoding="utf-8")
    old = """@lru_cache
def is_te_pytorch_available():
    if not is_te_available():
        return False

    import transformer_engine

    return hasattr(transformer_engine, "pytorch")
"""
    new = """@lru_cache
def is_te_pytorch_available():
    if not is_te_available():
        return False

    try:
        import transformer_engine
    except (ImportError, OSError):
        return False

    return hasattr(transformer_engine, "pytorch")
"""
    if old in source:
        import_utils_path.write_text(source.replace(old, new, 1), encoding="utf-8")
    elif "except (ImportError, OSError):" not in source:
        raise RuntimeError(
            f"Could not patch PEFT Transformer Engine availability check in {import_utils_path}"
        )
    log(
        "Prepared GRPO-local PEFT with tolerant Transformer Engine detection: "
        f"{import_utils_path}"
    )


def patch_runtime_openenv_lazy_imports(site_dir: Path) -> None:
    openenv_dir = site_dir / "openenv"
    if not openenv_dir.is_dir():
        return

    patches = {
        openenv_dir / "__init__.py": (
            '"""Runtime-light OpenEnv package init for GRPO imports."""\n'
            "__all__ = []\n"
        ),
        openenv_dir / "core" / "__init__.py": (
            '"""Runtime-light OpenEnv core init for GRPO imports."""\n'
            "__all__ = []\n"
        ),
        openenv_dir / "core" / "env_server" / "__init__.py": (
            '"""Runtime-light OpenEnv env_server init for GRPO imports."""\n'
            "try:\n"
            "    from .types import *  # noqa: F401,F403\n"
            "except Exception:\n"
            "    pass\n"
        ),
    }
    patched = []
    for path, content in patches.items():
        if path.is_file():
            path.write_text(content, encoding="utf-8")
            patched.append(str(path))
    if patched:
        log("Patched runtime OpenEnv lazy imports for GRPO: " + ", ".join(patched))


def patch_runtime_vllm_pr47258() -> None:
    """Apply vLLM PR47258 until the image wheel includes it."""
    try:
        import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle
        import vllm.model_executor.warmup.deep_gemm_warmup as deep_gemm_warmup
    except Exception as exc:
        log(f"WARNING: Could not import vLLM modules for PR47258 runtime patch: {exc}")
        return

    fp8_path = Path(fp8_oracle.__file__)
    warmup_path = Path(deep_gemm_warmup.__file__)
    patched_files: list[str] = []

    try:
        text = fp8_path.read_text(encoding="utf-8")
        original = text
        if "from vllm.config import get_current_vllm_config" not in text:
            text = text.replace(
                "from vllm import envs\n",
                "from vllm import envs\nfrom vllm.config import get_current_vllm_config\n",
                1,
            )
        if "from vllm.utils.deep_gemm import should_auto_disable_deep_gemm" not in text:
            text = text.replace(
                "from vllm.platforms import current_platform\n",
                "from vllm.platforms import current_platform\n"
                "from vllm.utils.deep_gemm import should_auto_disable_deep_gemm\n",
                1,
            )
        helper = '''

def _remove_deep_gemm_if_auto_disabled(
    available_backends: list[Fp8MoeBackend],
) -> None:
    """Drop DeepGEMM MoE backends when the model auto-disables DeepGEMM."""
    model_config = get_current_vllm_config().model_config
    model_type = (
        getattr(model_config.hf_text_config, "model_type", None)
        if model_config is not None
        else None
    )
    if not should_auto_disable_deep_gemm(model_type):
        return
    for backend in (Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM):
        if backend in available_backends:
            available_backends.remove(backend)

'''
        if "def _remove_deep_gemm_if_auto_disabled(" not in text:
            marker = "\ndef select_fp8_moe_backend(\n"
            if marker not in text:
                raise RuntimeError(f"Could not find select_fp8_moe_backend marker in {fp8_path}")
            text = text.replace(marker, helper + marker, 1)
        call = "    _remove_deep_gemm_if_auto_disabled(AVAILABLE_BACKENDS)\n\n"
        if call not in text:
            marker = "    # Handle explicit MARLIN FP8 configuration.\n"
            if marker not in text:
                raise RuntimeError(f"Could not find MARLIN marker in {fp8_path}")
            text = text.replace(marker, call + marker, 1)
        if text != original:
            fp8_path.write_text(text, encoding="utf-8")
            patched_files.append(str(fp8_path))

        text = warmup_path.read_text(encoding="utf-8")
        original = text
        old = '''        and getattr(module.quant_method, "block_quant", False)
        and not getattr(module.quant_method, "use_marlin", True)
    ):
'''
        new = '''        and getattr(module.quant_method, "block_quant", False)
        and not getattr(module.quant_method, "use_marlin", True)
        and getattr(module.quant_method, "use_deep_gemm", False)
    ):
'''
        if new not in text:
            if old not in text:
                raise RuntimeError(f"Could not find FP8 linear warmup marker in {warmup_path}")
            text = text.replace(old, new, 1)
        old = '''    quant_method = module._quant_method
    moe_quant_config = quant_method.get_fused_moe_quant_config(module.routed_experts)
'''
        new = '''    quant_method = module._quant_method
    quant_config = getattr(quant_method, "quant_config", None)
    if getattr(quant_config, "use_deep_gemm", None) is False:
        return False

    moe_quant_config = quant_method.get_fused_moe_quant_config(module.routed_experts)
'''
        if new not in text:
            if old not in text:
                raise RuntimeError(f"Could not find fused MoE warmup marker in {warmup_path}")
            text = text.replace(old, new, 1)
        if text != original:
            warmup_path.write_text(text, encoding="utf-8")
            patched_files.append(str(warmup_path))
    except Exception as exc:
        log(f"WARNING: vLLM PR47258 runtime patch failed: {exc}")
        return

    if patched_files:
        log("Applied vLLM PR47258 runtime patch: " + ", ".join(patched_files))
    else:
        log("vLLM PR47258 runtime patch already present")


def download_runtime_prebuilt_wheel(
    repo_id: str,
    filename: str,
    install_root: Path,
    label: str,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"huggingface_hub is required to hot-install the {label} wheel") from exc

    token = os.environ.get("HF_TOKEN")
    attempts = max(
        1,
        parse_int(os.environ.get("RUNTIME_DEPENDENCY_RETRY_ATTEMPTS"))
        or DEFAULT_RUNTIME_DEP_RETRY_ATTEMPTS,
    )
    base_seconds = max(
        0.0,
        parse_float(
            os.environ.get("RUNTIME_DEPENDENCY_RETRY_BASE_SECONDS"),
            DEFAULT_RUNTIME_DEP_RETRY_BASE_SECONDS,
        ),
    )
    max_seconds = max(
        0.0,
        parse_float(
            os.environ.get("RUNTIME_DEPENDENCY_RETRY_MAX_SECONDS"),
            DEFAULT_RUNTIME_DEP_RETRY_MAX_SECONDS,
        ),
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log(f"Downloading runtime {label} wheel {repo_id}/{filename} attempt {attempt}/{attempts}")
            wheel_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                cache_dir=str(install_root / "hf-cache"),
            )
            return Path(wheel_path)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep_seconds = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
            log(
                "Runtime %s wheel download failed on attempt %d/%d; retrying in %.1fs: %s"
                % (label, attempt, attempts, sleep_seconds, redact_secret(str(exc)))
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"Could not download runtime {label} wheel {repo_id}/{filename} after {attempts} attempts: "
        f"{redact_secret(str(last_error))}"
    )


def download_runtime_apex_wheel(
    repo_id: str,
    filename: str,
    install_root: Path,
) -> Path:
    return download_runtime_prebuilt_wheel(repo_id, filename, install_root, "Apex")


def skip_runtime_apex_liger_bootstrap(
    prime_rl_required: bool,
    grpo_required: bool,
    rlcsd_required: bool,
    verl_automodel_required: bool,
) -> bool:
    return prime_rl_required and not (grpo_required or rlcsd_required or verl_automodel_required)


def prepare_runtime_training_dependencies(
    wrapper_args: argparse.Namespace,
    forwarded_args: list[str],
    site_dir: Path,
    megatron_dir: Path,
    liger_dir: Path,
    install_root: Path,
) -> None:
    settings = runtime_training_deps_settings(wrapper_args)
    grpo_required = grpo_runtime_dependencies_required(forwarded_args)
    rlcsd_required = rlcsd_runtime_dependencies_required(forwarded_args)
    verl_automodel_required = verl_training_fp8_required(forwarded_args)
    prime_rl_required = prime_rl_runtime_dependencies_required(forwarded_args)
    te_required = transformer_engine_runtime_required(forwarded_args)
    cuda_nvidia_runtime_required = verl_automodel_required or prime_rl_required or te_required
    skip_apex_liger_bootstrap = skip_runtime_apex_liger_bootstrap(
        prime_rl_required,
        grpo_required,
        rlcsd_required,
        verl_automodel_required,
    )
    if cuda_nvidia_runtime_required:
        enable_system_nccl_preload_for_wrapper()
        install_python_global_requirements(
            verl_nvidia_runtime_requirements(),
            "NVIDIA CUDA runtime",
            no_deps=True,
            upgrade=True,
        )

    if verl_automodel_required:
        nccl_lib = site_dir / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
        if nccl_lib.exists():
            log(f"VERL FP8 NCCL runtime library already available: {nccl_lib}")
        else:
            install_python_requirements(
                verl_nvidia_runtime_requirements(),
                site_dir,
                "VERL FP8 NVIDIA",
                no_deps=True,
            )
    if grpo_required:
        remove_grpo_runtime_sitecustomize(site_dir)
        patch_runtime_peft_transformer_engine_probe(site_dir)
        patch_runtime_openenv_lazy_imports(site_dir)
    base_stack_required = not prime_rl_required or grpo_required or rlcsd_required or verl_automodel_required or te_required
    if base_stack_required and skip_apex_liger_bootstrap:
        log("Skipping runtime Apex/Liger bootstrap for Prime-RL; these packages are expected from the image.")
    if base_stack_required and not skip_apex_liger_bootstrap:
        apex_ok, apex_details = runtime_dependency_probe(site_dir, megatron_dir, module="apex")
        if apex_ok:
            log("Apex runtime import is already available")
        else:
            if apex_details:
                log(f"Apex runtime import unavailable; installing prebuilt wheel: {apex_details}")
            try:
                apex_wheel = download_runtime_apex_wheel(
                    settings["apex_wheel_repo"],
                    settings["apex_wheel_file"],
                    install_root,
                )
                install_python_target(apex_wheel, site_dir, "Apex")
            except Exception as exc:
                if not grpo_required:
                    raise
                log(f"WARNING: Optional Apex runtime setup failed for grpo_fast: {redact_secret(str(exc))}")

    if te_required:
        te_ok, te_details = runtime_dependency_probe(site_dir, megatron_dir, module="transformer_engine")
        if te_ok:
            log("Transformer Engine runtime import is already available")
        else:
            if te_details:
                log(f"Transformer Engine runtime import unavailable; installing prebuilt wheel: {te_details}")
            te_wheel = download_runtime_prebuilt_wheel(
                settings["transformer_engine_wheel_repo"],
                settings["transformer_engine_wheel_file"],
                install_root,
                "Transformer Engine",
            )
            install_python_target(te_wheel, site_dir, "Transformer Engine")
            te_ok, te_details = runtime_dependency_probe(site_dir, megatron_dir, module="transformer_engine")
            if not te_ok:
                raise RuntimeError(f"Runtime Transformer Engine import verification failed:\n{te_details}")
            log(f"Runtime Transformer Engine import verified:\n{te_details}")

    if base_stack_required:
        megatron_ok, megatron_details = runtime_dependency_probe(site_dir, megatron_dir, module="megatron")
        if megatron_ok:
            log("Megatron Core runtime import is already available")
        else:
            if megatron_details:
                log(f"Megatron Core runtime import unavailable; cloning/installing: {megatron_details}")
            try:
                ensure_runtime_repo(
                    settings["megatron_repo"],
                    settings["megatron_ref"],
                    megatron_dir,
                    "Megatron-LM",
                )
                install_python_target(megatron_dir, site_dir, "Megatron Core")
            except Exception as exc:
                if not grpo_required:
                    raise
                log(f"WARNING: Optional Megatron Core runtime setup failed for grpo_fast: {redact_secret(str(exc))}")

        if not skip_apex_liger_bootstrap:
            liger_ok, liger_details = runtime_dependency_probe(site_dir, megatron_dir, module="liger")
            if liger_ok:
                log("Liger Kernel runtime import is already available")
            else:
                if liger_details:
                    log(f"Liger Kernel runtime import unavailable; cloning/installing: {liger_details}")
                try:
                    ensure_runtime_repo(
                        settings["liger_repo"],
                        settings["liger_ref"],
                        liger_dir,
                        "Liger-Kernel",
                    )
                    install_python_target(liger_dir, site_dir, "Liger Kernel")
                except Exception as exc:
                    if not grpo_required:
                        raise
                    log(f"WARNING: Optional Liger Kernel runtime setup failed for grpo_fast: {redact_secret(str(exc))}")

    if grpo_required:
        for module, label in (
            ("apex", "Apex"),
            ("megatron", "Megatron Core"),
            ("liger", "Liger Kernel"),
        ):
            optional_ok, optional_details = runtime_dependency_probe(site_dir, megatron_dir, module=module)
            if optional_ok:
                log(f"Optional {label} runtime import verified for grpo_fast")
            else:
                log(
                    f"WARNING: Optional {label} runtime import remains unavailable for grpo_fast; "
                    f"continuing without it:\n{optional_details}"
                )
    elif base_stack_required:
        if skip_apex_liger_bootstrap:
            dependencies_ok, details = runtime_dependency_probe(site_dir, megatron_dir, module="megatron")
            if not dependencies_ok:
                raise RuntimeError(f"Runtime Megatron Core import verification failed:\n{details}")
            log(f"Runtime Megatron Core import verified:\n{details}")
        else:
            dependencies_ok, details = runtime_dependency_probe(site_dir, megatron_dir)
            if not dependencies_ok:
                raise RuntimeError(f"Runtime Apex/Megatron/Liger import verification failed:\n{details}")
            log(f"Runtime Apex/Megatron/Liger imports verified:\n{details}")

    if grpo_required:
        grpo_ok, grpo_details = runtime_dependency_probe(site_dir, megatron_dir, module="grpo")
        if grpo_ok:
            log("GRPO runtime imports are already available")
        else:
            if grpo_details:
                log(f"GRPO runtime imports unavailable; installing requirements: {grpo_details}")
            install_python_requirements(grpo_runtime_requirements(), site_dir, "GRPO")
            patch_runtime_openenv_lazy_imports(site_dir)
        vllm_override_installed = maybe_install_runtime_vllm_override(settings, site_dir)
        grpo_ok, grpo_details = runtime_dependency_probe(site_dir, megatron_dir, module="grpo")
        if not grpo_ok and vllm_override_installed:
            log(
                "WARNING: GRPO import failed after installing the requested vLLM override. "
                "Removing the runtime vLLM override and retrying with the baked image vLLM.\n"
                f"{grpo_details}"
            )
            remove_runtime_vllm_override(site_dir)
            grpo_ok, grpo_details = runtime_dependency_probe(site_dir, megatron_dir, module="grpo")
        if not grpo_ok:
            raise RuntimeError(f"Runtime GRPO import verification failed:\n{grpo_details}")
        log(f"Runtime GRPO imports verified:\n{grpo_details}")

    if rlcsd_required:
        rlcsd_ok, rlcsd_details = runtime_dependency_probe(site_dir, megatron_dir, module="rlcsd")
        if rlcsd_ok:
            log("RLCSD/verl runtime imports are already available")
        else:
            if rlcsd_details:
                log(f"RLCSD/verl runtime imports unavailable; installing requirements: {rlcsd_details}")
            install_python_requirements(rlcsd_runtime_requirements(), site_dir, "RLCSD/verl")
            rlcsd_ok, rlcsd_details = runtime_dependency_probe(site_dir, megatron_dir, module="rlcsd")
        if not rlcsd_ok:
            raise RuntimeError(f"Runtime RLCSD/verl import verification failed:\n{rlcsd_details}")
        log(f"Runtime RLCSD/verl imports verified:\n{rlcsd_details}")

    if verl_automodel_required:
        automodel_ok, automodel_details = runtime_dependency_probe(site_dir, megatron_dir, module="nemo_automodel")
        if automodel_ok:
            log("VERL automodel FP8 runtime imports are already available")
        else:
            if automodel_details:
                log(f"VERL automodel FP8 runtime imports unavailable; installing requirements: {automodel_details}")
            install_python_requirements(
                verl_automodel_runtime_requirements(),
                site_dir,
                "VERL automodel FP8",
            )
            automodel_ok, automodel_details = runtime_dependency_probe(
                site_dir,
                megatron_dir,
                module="nemo_automodel",
            )
        if not automodel_ok:
            raise RuntimeError(f"Runtime VERL automodel FP8 import verification failed:\n{automodel_details}")
        log(f"Runtime VERL automodel FP8 imports verified:\n{automodel_details}")

    if prime_rl_required:
        prime_rl_dir_value = os.environ.get(PREPARED_PRIME_RL_ENV) or os.environ.get("PRIME_RL_DIR")
        if not prime_rl_dir_value:
            raise RuntimeError("Prime-RL backend requires PRIME_RL_DIR from runtime fetch or --prime_rl_dir")
        prime_rl_dir = Path(prime_rl_dir_value)
        runtime_requirements = prime_rl_runtime_requirements()
        if runtime_requirements:
            install_python_global_requirements(runtime_requirements, "Prime-RL runtime")
        prepare_prime_rl_checkout_for_install(prime_rl_dir)
        install_python_global_requirements(prime_rl_build_requirements(), "Prime-RL build")
        remove_incompatible_prime_rl_config_package()
        install_python_global_requirements(
            prime_rl_config_requirements(prime_rl_dir),
            "Prime-RL configs",
            editable=True,
            no_build_isolation=True,
        )
        install_python_global_requirements(
            prime_rl_environment_requirements(prime_rl_dir),
            "Prime-RL environments",
            editable=True,
            no_build_isolation=True,
        )
        install_python_global_requirements(
            prime_rl_install_requirements(prime_rl_dir),
            "Prime-RL package",
            no_build_isolation=True,
            cwd=prime_rl_dir,
        )
        install_python_global_requirements(
            verl_nvidia_runtime_requirements(),
            "Prime-RL final NVIDIA CUDA runtime",
            no_deps=True,
            upgrade=True,
        )
        if os.environ.get("PRIME_RL_VLLM_OVERRIDE", "0") == "1":
            patch_runtime_vllm_pr47258()
        if te_required:
            te_ok, te_details = runtime_dependency_probe(
                site_dir,
                megatron_dir,
                module="transformer_engine",
            )
            if not te_ok:
                raise RuntimeError(
                    "Runtime Transformer Engine import verification failed after Prime-RL package install. "
                    "This check runs before Prime-RL starts GPU workers.\n"
                    f"{te_details}"
                )
            log(f"Runtime Transformer Engine import verified after Prime-RL package install:\n{te_details}")
        log("Runtime Prime-RL packages installed; skipping import preflight")


def ensure_runtime_training_dependencies(
    wrapper_args: argparse.Namespace,
    forwarded_args: list[str],
) -> None:
    forwarded_rlcsd_dir = forwarded_option_value(forwarded_args, "--rlcsd_dir", "--rlcsd-dir")
    if forwarded_rlcsd_dir and not os.environ.get("RLCSD_DIR"):
        os.environ["RLCSD_DIR"] = forwarded_rlcsd_dir
    rlcsd_dir_value = os.environ.get(PREPARED_RLCSD_ENV) or os.environ.get("RLCSD_DIR")
    rlcsd_dir = Path(rlcsd_dir_value) if rlcsd_dir_value else None
    verl_dir_value = os.environ.get(PREPARED_VERL_ENV) or os.environ.get("VERL_DIR")
    verl_dir = Path(verl_dir_value) if verl_dir_value else None
    forwarded_prime_rl_dir = forwarded_option_value(forwarded_args, "--prime_rl_dir", "--prime-rl-dir")
    if forwarded_prime_rl_dir and not os.environ.get("PRIME_RL_DIR"):
        os.environ["PRIME_RL_DIR"] = forwarded_prime_rl_dir
    prime_rl_dir_value = os.environ.get(PREPARED_PRIME_RL_ENV) or os.environ.get("PRIME_RL_DIR")
    prime_rl_dir = Path(prime_rl_dir_value) if prime_rl_dir_value else None
    if not runtime_training_deps_enabled(wrapper_args):
        log("Runtime Apex/Megatron/Liger dependency bootstrap disabled")
        return

    install_root, site_dir, megatron_dir, liger_dir, state_path = runtime_training_deps_paths(wrapper_args)
    node_rank = resolve_wrapper_node_rank(forwarded_args)
    rank_label = "none" if node_rank is None else str(node_rank)
    grpo_required = grpo_runtime_dependencies_required(forwarded_args)
    rlcsd_required = rlcsd_runtime_dependencies_required(forwarded_args)
    verl_automodel_required = verl_training_fp8_required(forwarded_args)
    prime_rl_required = prime_rl_runtime_dependencies_required(forwarded_args)
    te_required = transformer_engine_runtime_required(forwarded_args)
    skip_apex_liger_bootstrap = skip_runtime_apex_liger_bootstrap(
        prime_rl_required,
        grpo_required,
        rlcsd_required,
        verl_automodel_required,
    )
    if node_rank in (None, 0):
        base_stack_required = not prime_rl_required or grpo_required or rlcsd_required or verl_automodel_required or te_required
        dependency_parts: list[str] = []
        if base_stack_required:
            if skip_apex_liger_bootstrap:
                dependency_parts.append("Megatron Core")
            else:
                dependency_parts.append("Apex, Megatron Core, Liger Kernel")
        if te_required:
            dependency_parts.append("Transformer Engine")
        if grpo_required:
            dependency_parts.append("GRPO runtime packages")
        if rlcsd_required:
            dependency_parts.append("RLCSD/verl runtime packages")
        if verl_automodel_required:
            dependency_parts.append("VERL automodel FP8 runtime packages")
        if prime_rl_required:
            dependency_parts.append("Prime-RL runtime packages")
        dependency_label = " + ".join(dependency_parts) or "runtime packages"
        log(f"node_rank={rank_label} ensuring runtime {dependency_label}; marker={state_path}")
        write_runtime_fetch_state(
            state_path,
            {
                "status": "installing",
                "node_rank": rank_label,
                "coordination_id": runtime_fetch_coordination_id(),
                "transformer_engine_runtime_dep": te_required,
                "grpo_runtime_deps": grpo_required,
                "rlcsd_runtime_deps": rlcsd_required,
                "verl_automodel_runtime_deps": verl_automodel_required,
                "prime_rl_runtime_deps": prime_rl_required,
                "rlcsd_dir": str(rlcsd_dir) if rlcsd_dir else "",
                "verl_dir": str(verl_dir) if verl_dir else "",
                "prime_rl_dir": str(prime_rl_dir) if prime_rl_dir else "",
            },
        )
        try:
            install_root.mkdir(parents=True, exist_ok=True)
            prepare_runtime_training_dependencies(
                wrapper_args,
                forwarded_args,
                site_dir,
                megatron_dir,
                liger_dir,
                install_root,
            )
        except Exception as exc:
            write_runtime_fetch_state(
                state_path,
                {
                    "status": "failed",
                    "node_rank": rank_label,
                    "coordination_id": runtime_fetch_coordination_id(),
                    "transformer_engine_runtime_dep": te_required,
                    "grpo_runtime_deps": grpo_required,
                    "rlcsd_runtime_deps": rlcsd_required,
                    "verl_automodel_runtime_deps": verl_automodel_required,
                    "prime_rl_runtime_deps": prime_rl_required,
                    "rlcsd_dir": str(rlcsd_dir) if rlcsd_dir else "",
                    "verl_dir": str(verl_dir) if verl_dir else "",
                    "prime_rl_dir": str(prime_rl_dir) if prime_rl_dir else "",
                    "error": redact_secret(str(exc)),
                },
            )
            raise
        write_runtime_fetch_state(
            state_path,
            {
                "status": "ready",
                "node_rank": rank_label,
                "coordination_id": runtime_fetch_coordination_id(),
                "site_dir": str(site_dir),
                "megatron_dir": str(megatron_dir),
                "transformer_engine_runtime_dep": te_required,
                "grpo_runtime_deps": grpo_required,
                "rlcsd_runtime_deps": rlcsd_required,
                "verl_automodel_runtime_deps": verl_automodel_required,
                "prime_rl_runtime_deps": prime_rl_required,
                "rlcsd_dir": str(rlcsd_dir) if rlcsd_dir else "",
                "verl_dir": str(verl_dir) if verl_dir else "",
                "prime_rl_dir": str(prime_rl_dir) if prime_rl_dir else "",
            },
        )
        os.environ["TRAIN_WRAPPER_RUNTIME_BIN"] = str(site_dir / "bin")
        prepend_path(site_dir / "bin", site_dir.parent.parent.parent / "bin")
        prepend_runtime_library_path(site_dir)
        if verl_dir:
            prepend_pythonpath(site_dir, megatron_dir, verl_dir, *( [rlcsd_dir] if rlcsd_dir else [] ))
        elif rlcsd_dir:
            prepend_pythonpath(site_dir, megatron_dir, rlcsd_dir / "third_party" / "verl", rlcsd_dir)
        elif prime_rl_dir:
            prepend_pythonpath(site_dir, megatron_dir, prime_rl_dir / "packages" / "prime-rl-configs" / "src", prime_rl_dir / "src")
        else:
            prepend_pythonpath(site_dir, megatron_dir)
        if verl_dir:
            os.environ[PREPARED_VERL_ENV] = str(verl_dir)
            os.environ["VERL_DIR"] = str(verl_dir)
        if prime_rl_dir:
            os.environ[PREPARED_PRIME_RL_ENV] = str(prime_rl_dir)
            os.environ["PRIME_RL_DIR"] = str(prime_rl_dir)
        return

    timeout = runtime_fetch_timeout(wrapper_args)
    poll_interval = runtime_fetch_poll_interval(wrapper_args)
    deadline = time.monotonic() + timeout
    log(f"node_rank={node_rank} waiting for node_rank=0 runtime dependency marker {state_path}")
    while time.monotonic() < deadline:
        state = read_runtime_fetch_state(state_path)
        if state is None:
            time.sleep(poll_interval)
            continue
        status = str(state.get("status", ""))
        if status == "failed":
            raise RuntimeError(f"node_rank=0 runtime dependency install failed: {state.get('error', '<no error>')}")
        if status == "ready":
            prepared_site = Path(str(state.get("site_dir", site_dir)))
            prepared_megatron = Path(str(state.get("megatron_dir", megatron_dir)))
            prepared_rlcsd_value = str(state.get("rlcsd_dir", "") or "")
            prepared_rlcsd = Path(prepared_rlcsd_value) if prepared_rlcsd_value else rlcsd_dir
            prepared_verl_value = str(state.get("verl_dir", "") or "")
            prepared_verl = Path(prepared_verl_value) if prepared_verl_value else verl_dir
            if prepared_rlcsd:
                os.environ[PREPARED_RLCSD_ENV] = str(prepared_rlcsd)
                os.environ["RLCSD_DIR"] = str(prepared_rlcsd)
            if prepared_verl:
                os.environ[PREPARED_VERL_ENV] = str(prepared_verl)
                os.environ["VERL_DIR"] = str(prepared_verl)
            prepared_prime_rl_value = str(state.get("prime_rl_dir", "") or "")
            prepared_prime_rl = Path(prepared_prime_rl_value) if prepared_prime_rl_value else prime_rl_dir
            if prepared_prime_rl:
                os.environ[PREPARED_PRIME_RL_ENV] = str(prepared_prime_rl)
                os.environ["PRIME_RL_DIR"] = str(prepared_prime_rl)
            state_grpo_required = bool(state.get("grpo_runtime_deps", False))
            state_rlcsd_required = bool(state.get("rlcsd_runtime_deps", False))
            state_verl_automodel_required = bool(state.get("verl_automodel_runtime_deps", False))
            state_prime_rl_required = bool(state.get("prime_rl_runtime_deps", False))
            state_te_required = bool(state.get("transformer_engine_runtime_dep", False))
            if not state_grpo_required and not state_prime_rl_required:
                dependencies_ok, details = runtime_dependency_probe(prepared_site, prepared_megatron)
                if not dependencies_ok:
                    raise RuntimeError(
                        f"Runtime Apex/Megatron/Liger import verification failed on node {node_rank}:\n{details}"
                    )
            if state_te_required:
                te_ok, te_details = runtime_dependency_probe(
                    prepared_site,
                    prepared_megatron,
                    module="transformer_engine",
                )
                if not te_ok:
                    raise RuntimeError(
                        f"Runtime Transformer Engine import verification failed on node {node_rank}:\n{te_details}"
                    )
            if state_grpo_required:
                grpo_ok, grpo_details = runtime_dependency_probe(prepared_site, prepared_megatron, module="grpo")
                if not grpo_ok:
                    raise RuntimeError(
                        f"Runtime GRPO import verification failed on node {node_rank}:\n{grpo_details}"
                    )
            if state_rlcsd_required:
                rlcsd_ok, rlcsd_details = runtime_dependency_probe(prepared_site, prepared_megatron, module="rlcsd")
                if not rlcsd_ok:
                    raise RuntimeError(
                        f"Runtime RLCSD/verl import verification failed on node {node_rank}:\n{rlcsd_details}"
                    )
            if state_verl_automodel_required:
                automodel_ok, automodel_details = runtime_dependency_probe(
                    prepared_site,
                    prepared_megatron,
                    module="nemo_automodel",
                )
                if not automodel_ok:
                    raise RuntimeError(
                        f"Runtime VERL automodel FP8 import verification failed on node {node_rank}:\n{automodel_details}"
                    )
            if state_prime_rl_required:
                log(f"node_rank={node_rank} skipping Prime-RL import preflight")
            os.environ["TRAIN_WRAPPER_RUNTIME_BIN"] = str(prepared_site / "bin")
            prepend_path(prepared_site / "bin", prepared_site.parent.parent.parent / "bin")
            prepend_runtime_library_path(prepared_site)
            if prepared_verl:
                prepend_pythonpath(
                    prepared_site,
                    prepared_megatron,
                    prepared_verl,
                    *( [prepared_rlcsd] if prepared_rlcsd else [] ),
                )
            elif prepared_rlcsd:
                prepend_pythonpath(prepared_site, prepared_megatron, prepared_rlcsd / "third_party" / "verl", prepared_rlcsd)
            elif prepared_prime_rl:
                prepend_pythonpath(
                    prepared_site,
                    prepared_megatron,
                    prepared_prime_rl / "packages" / "prime-rl-configs" / "src",
                    prepared_prime_rl / "src",
                )
            else:
                prepend_pythonpath(prepared_site, prepared_megatron)
            dependency_labels = []
            if state_te_required:
                dependency_labels.append("Transformer Engine")
            if state_grpo_required:
                dependency_labels.append("GRPO")
            if state_rlcsd_required:
                dependency_labels.append("RLCSD/verl")
            if state_verl_automodel_required:
                dependency_labels.append("VERL automodel FP8")
            if state_prime_rl_required:
                dependency_labels.append("Prime-RL")
            if not dependency_labels:
                dependency_labels.append("Apex/Megatron/Liger")
            dependency_label = " + ".join(dependency_labels)
            log(f"node_rank={node_rank} verified runtime {dependency_label} prepared by node_rank=0")
            return
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out after {timeout:.0f}s waiting for node_rank=0 runtime dependency marker {state_path}"
    )


def coordinated_fetch_runtime_repos(
    wrapper_args: argparse.Namespace,
    forwarded_args: list[str],
) -> tuple[Path, Path, Path, Path, Path, Path]:
    node_rank = resolve_wrapper_node_rank(forwarded_args)
    state_path, fingerprint = runtime_fetch_state_path(wrapper_args)
    rank_label = "none" if node_rank is None else str(node_rank)
    if node_rank in (None, 0):
        log(f"node_rank={rank_label} fetching runtime repos; marker={state_path}")
        write_runtime_fetch_state(
            state_path,
            {
                "status": "fetching",
                "fingerprint": fingerprint,
                "node_rank": rank_label,
                "coordination_id": runtime_fetch_coordination_id(),
            },
        )
        try:
            engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir = fetch_runtime_repos(wrapper_args)
            missing = validate_runtime_paths(
                engine_path,
                open_instruct_dir,
                olmo_core_dir,
                rlcsd_dir,
                verl_dir,
                prime_rl_dir,
            )
            if missing:
                raise RuntimeError(missing)
        except Exception as exc:
            write_runtime_fetch_state(
                state_path,
                {
                    "status": "failed",
                    "fingerprint": fingerprint,
                    "node_rank": rank_label,
                    "coordination_id": runtime_fetch_coordination_id(),
                    "error": redact_secret(str(exc)),
                },
            )
            raise
        write_runtime_fetch_state(
            state_path,
            {
                "status": "ready",
                "fingerprint": fingerprint,
                "node_rank": rank_label,
                "coordination_id": runtime_fetch_coordination_id(),
                "engine_path": str(engine_path),
                "open_instruct_dir": str(open_instruct_dir),
                "olmo_core_dir": str(olmo_core_dir),
                "rlcsd_dir": str(rlcsd_dir),
                "verl_dir": str(verl_dir),
                "prime_rl_dir": str(prime_rl_dir),
            },
        )
        log(f"Runtime repo fetch complete; marker={state_path}")
        return engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir

    timeout = runtime_fetch_timeout(wrapper_args)
    poll_interval = runtime_fetch_poll_interval(wrapper_args)
    deadline = time.monotonic() + timeout
    stale_failed_seconds = runtime_fetch_stale_failed_seconds()
    log(f"node_rank={node_rank} waiting for node_rank=0 runtime repo fetch marker {state_path}")
    last_status: str | None = None
    logged_stale_failed = False
    while time.monotonic() < deadline:
        state = read_runtime_fetch_state(state_path)
        if state is None:
            time.sleep(poll_interval)
            continue
        status = str(state.get("status", ""))
        if status != last_status:
            log(f"node_rank={node_rank} saw runtime fetch status={status or 'unknown'}")
            last_status = status
        if state.get("fingerprint") != fingerprint:
            time.sleep(poll_interval)
            continue
        if status == "failed":
            updated_at = parse_float(str(state.get("updated_at", "")), 0.0)
            failed_age = time.time() - updated_at if updated_at else 0.0
            if updated_at and failed_age > stale_failed_seconds:
                if not logged_stale_failed:
                    log(
                        "node_rank=%s ignoring stale failed runtime fetch marker age=%.1fs "
                        "threshold=%.1fs and waiting for a fresh node_rank=0 attempt"
                        % (node_rank, failed_age, stale_failed_seconds)
                    )
                    logged_stale_failed = True
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"node_rank=0 runtime repo fetch failed: {state.get('error', '<no error>')}")
        if status == "ready":
            try:
                engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir = runtime_paths_from_state(state)
            except KeyError:
                time.sleep(poll_interval)
                continue
            missing = validate_runtime_paths(engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir)
            if missing:
                log(f"node_rank={node_rank} waiting for ready runtime paths: {missing}")
                time.sleep(poll_interval)
                continue
            log(f"node_rank={node_rank} using runtime repos prepared by node_rank=0")
            return engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out after {timeout:.0f}s waiting for node_rank=0 runtime repo fetch marker {state_path}"
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    wrapper_args, forwarded_args = parse_args(raw_args)
    prepare_wrapper_run_directory(forwarded_args)
    configure_wrapper_file_logging(forwarded_args)
    if WRAPPER_LOG_FILE is not None:
        log(f"Wrapper log file: {WRAPPER_LOG_FILE}")
    log(f"Raw CLI args: {cli_args_for_log(raw_args)}")
    log(f"Forwarded train_engine args: {cli_args_for_log(forwarded_args)}")
    fetch_update = True if wrapper_args.fetch_update is None else bool(wrapper_args.fetch_update)
    if fetch_update:
        prepared_repos = prepared_runtime_repos_from_env()
        if prepared_repos is not None:
            engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir = prepared_repos
            set_prepared_runtime_env(engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir)
            log(f"Using prepared runtime repos after wrapper self-update: {engine_path.parent.parent}")
            ensure_runtime_training_dependencies(wrapper_args, forwarded_args)
            exec_engine(engine_path, forwarded_args, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir)
            return 0
        engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir = coordinated_fetch_runtime_repos(
            wrapper_args,
            forwarded_args,
        )
        set_prepared_runtime_env(engine_path, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir)
        maybe_reexec_runtime_wrapper(
            wrapper_args,
            raw_args,
            engine_path,
            open_instruct_dir,
            olmo_core_dir,
            rlcsd_dir,
            verl_dir,
            prime_rl_dir,
        )
        ensure_runtime_training_dependencies(wrapper_args, forwarded_args)
        exec_engine(engine_path, forwarded_args, open_instruct_dir, olmo_core_dir, rlcsd_dir, verl_dir, prime_rl_dir)
    else:
        engine_path = baked_engine_path()
        log(f"Using baked train engine {engine_path}")
        ensure_runtime_training_dependencies(wrapper_args, forwarded_args)
        exec_engine(engine_path, forwarded_args)
    return 0


def train(*args: str) -> int:
    return main(list(args))


def cli() -> int:
    try:
        return main()
    except Exception as exc:
        log(f"ERROR: {redact_secret(str(exc))}")
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
