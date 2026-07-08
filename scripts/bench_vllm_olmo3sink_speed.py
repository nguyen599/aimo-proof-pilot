#!/usr/bin/env python3
"""Benchmark OLMo3Sink generation throughput with vLLM offline inference.

Default workload:
  - 8 independent one-GPU vLLM engines (`TP=1, DP=8`)
  - 16 concurrent prompts
  - about 1,024 prompt tokens per request
  - 131,072 total requested output tokens (8,192 per request)

If you need 128k output tokens *per request*, pass
`--max-tokens-per-request 128000 --max-model-len 131072`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import json
import multiprocessing as mp
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_VLLM_RUNTIME_WHEEL_URL = (
    "https://wheels.vllm.ai/f5a8d73377d0f0a4e00cba172f9fbd0d50471b07/"
    "vllm-0.23.1rc1.dev699%2Bgf5a8d7337-cp38-abi3-manylinux_2_28_x86_64.whl"
)
DEFAULT_VLLM_VERSION_FRAGMENT = "0.23.1rc1.dev699+gf5a8d7337"
DEFAULT_VLLM_INSTALL_DIR = "/tmp/olmo3sink_vllm_0_23_1rc1_dev699"
DEFAULT_VLLM_DISABLED_KERNELS = "FlashInferFP8ScaledMMLinearKernel"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--vllm-extra-json must decode to a JSON object")
    return value


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_vllm_pin(args: argparse.Namespace) -> None:
    if not args.install_vllm_wheel:
        return
    target = Path(args.vllm_install_dir).expanduser().resolve()
    if (target / "vllm").exists():
        add_pythonpath(target)
    try:
        current = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        current = None

    expected = args.vllm_version_fragment
    if current and expected in current and not args.force_reinstall_vllm:
        print(f"vllm_pin=current_ok version={current}")
        return

    print(
        "vllm_pin=installing "
        f"current={current!r} expected_fragment={expected!r} wheel={args.vllm_wheel_url}",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "--no-deps",
        "--upgrade",
        "--target",
        str(target),
        args.vllm_wheel_url,
    ]
    subprocess.check_call(command)
    add_pythonpath(target)
    try:
        installed = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        installed = "missing"
    print(f"vllm_pin=installed version={installed}", flush=True)


def add_src_to_path(src_dir: Path) -> None:
    src_text = str(src_dir.resolve())
    add_pythonpath(Path(src_text))


def add_pythonpath(path: Path) -> None:
    path_text = str(path.resolve())
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        if path_text not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = path_text + os.pathsep + existing
    else:
        os.environ["PYTHONPATH"] = path_text


def apply_vllm_env(args: argparse.Namespace) -> None:
    if args.vllm_disabled_kernels is None:
        return
    if args.vllm_disabled_kernels:
        os.environ["VLLM_DISABLED_KERNELS"] = args.vllm_disabled_kernels
    else:
        os.environ.pop("VLLM_DISABLED_KERNELS", None)


def register_olmo3sink(src_dir: Path, skip: bool) -> None:
    if skip:
        return
    add_src_to_path(src_dir)
    install_vllm_plugin_shim(src_dir)
    from olmo3_sink import register_olmo3_sink

    register_olmo3_sink()

    try:
        from vllm import ModelRegistry
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("vLLM is not importable; install vllm before running this benchmark") from exc

    ModelRegistry.register_model(
        "Olmo3SinkForCausalLM",
        "olmo3_sink.vllm_adapter:Olmo3SinkForCausalLM",
    )


def install_vllm_plugin_shim(src_dir: Path) -> None:
    """Expose OLMo3Sink registration to vLLM worker subprocesses.

    vLLM offline inference may spawn worker processes after the parent process
    registers a custom model. In-process `ModelRegistry.register_model()` does
    not cross the process boundary, so we create a tiny temporary distribution
    on `PYTHONPATH` with a `vllm.general_plugins` entry point. vLLM loads these
    plugins in the engine core and worker processes before model resolution.
    """
    src_dir = src_dir.resolve()
    digest = hashlib.sha256(str(src_dir).encode()).hexdigest()[:12]
    plugin_root = Path(os.environ.get("OLMO3SINK_VLLM_PLUGIN_DIR", f"/tmp/olmo3sink_vllm_plugin_{digest}"))
    dist_info = plugin_root / "olmo3sink_vllm_plugin-0.0.0.dist-info"
    plugin_root.mkdir(parents=True, exist_ok=True)
    dist_info.mkdir(parents=True, exist_ok=True)
    (plugin_root / "olmo3sink_vllm_plugin.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import sys",
                f"SRC_DIR = {str(src_dir)!r}",
                "def register():",
                "    if SRC_DIR not in sys.path:",
                "        sys.path.insert(0, SRC_DIR)",
                "    from olmo3_sink import register_olmo3_sink",
                "    register_olmo3_sink()",
                "    from vllm import ModelRegistry",
                "    ModelRegistry.register_model(",
                "        'Olmo3SinkForCausalLM',",
                "        'olmo3_sink.vllm_adapter:Olmo3SinkForCausalLM',",
                "    )",
                "",
            ]
        )
    )
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: olmo3sink-vllm-plugin\n"
        "Version: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.general_plugins]\n"
        "olmo3sink_bench = olmo3sink_vllm_plugin:register\n"
    )
    plugin_text = str(plugin_root)
    if plugin_text not in sys.path:
        sys.path.insert(0, plugin_text)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        if plugin_text not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = plugin_text + os.pathsep + existing
    else:
        os.environ["PYTHONPATH"] = plugin_text
    allowed_plugins = os.environ.get("VLLM_PLUGINS")
    if allowed_plugins is not None:
        names = [name.strip() for name in allowed_plugins.split(",") if name.strip()]
        if "olmo3sink_bench" not in names:
            names.append("olmo3sink_bench")
            os.environ["VLLM_PLUGINS"] = ",".join(names)


def tokenizer_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_prompt(tokenizer: Any, filler_repeats: int, request_id: int) -> str:
    problem = (
        "Prove a difficult olympiad-style inequality. "
        "Write a rigorous proof, check edge cases, and put the final answer in boxed form. "
        f"Benchmark request id {request_id:04d}.\n\n"
    )
    filler = (
        "We need a complete mathematical derivation with clear definitions, "
        "careful estimates, and no skipped algebraic steps. "
    )
    content = problem + filler * filler_repeats
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def build_target_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")

    low = 0
    high = 1
    while tokenizer_len(tokenizer, render_prompt(tokenizer, high, request_id)) < target_tokens:
        high *= 2

    best = render_prompt(tokenizer, low, request_id)
    best_len = tokenizer_len(tokenizer, best)
    while low <= high:
        mid = (low + high) // 2
        candidate = render_prompt(tokenizer, mid, request_id)
        length = tokenizer_len(tokenizer, candidate)
        if length <= target_tokens:
            best = candidate
            best_len = length
            low = mid + 1
        else:
            high = mid - 1

    # Add a tiny amount of neutral padding when the binary-search granularity
    # leaves us noticeably below the requested token count.
    pad = " Therefore"
    while best_len < target_tokens:
        candidate = best + pad
        length = tokenizer_len(tokenizer, candidate)
        if length > target_tokens:
            break
        best = candidate
        best_len = length

    return best, best_len


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable: {exc}"


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sampling_params_kwargs(args: argparse.Namespace, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
    }
    signature = inspect.signature(SamplingParams)
    if "ignore_eos" in signature.parameters:
        kwargs["ignore_eos"] = args.ignore_eos
    if args.force_max_tokens and "min_tokens" in signature.parameters:
        kwargs["min_tokens"] = max_tokens
    return kwargs


def init_llm(args: argparse.Namespace, max_model_len: int, max_num_batched_tokens: int):
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_mode": args.tokenizer_mode,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs or args.batch_size,
        "max_num_batched_tokens": max_num_batched_tokens,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": args.disable_log_stats,
    }
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.disable_custom_all_reduce is not None:
        kwargs["disable_custom_all_reduce"] = args.disable_custom_all_reduce
    kwargs.update(parse_json_object(args.vllm_extra_json))

    print("vllm_engine_kwargs=" + json.dumps(kwargs, default=str, sort_keys=True))
    start = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - start
    print(f"engine_load_seconds={load_seconds:.3f}")
    return llm, load_seconds, kwargs


def run_generate(llm: Any, prompts: list[str], sampling_params: Any) -> tuple[list[Any], float]:
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    seconds = time.perf_counter() - start
    return outputs, seconds


def run_single_process_benchmark(
    args: argparse.Namespace,
    prompts: list[str],
    prompt_lens: list[int],
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    llm, load_seconds, llm_kwargs = init_llm(args, max_model_len, max_num_batched_tokens)

    if args.warmup_tokens > 0:
        warmup_kwargs = sampling_params_kwargs(args, args.warmup_tokens)
        warmup_kwargs["ignore_eos"] = True
        if "min_tokens" in inspect.signature(SamplingParams).parameters:
            warmup_kwargs["min_tokens"] = args.warmup_tokens
        print("running_warmup=true")
        _, warmup_seconds = run_generate(llm, [prompts[0]], SamplingParams(**warmup_kwargs))
        print(f"warmup_seconds={warmup_seconds:.3f}")

    sampling_params = SamplingParams(**sampling_params_kwargs(args, max_tokens))
    round_metrics: list[dict[str, Any]] = []
    for round_idx in range(args.rounds):
        outputs, seconds = run_generate(llm, prompts, sampling_params)
        output_summary = summarize_outputs(outputs)
        total_output_tokens = int(output_summary["output_tokens_total"])
        total_prompt_tokens = sum(prompt_lens)
        metrics = {
            "round": round_idx,
            "seconds": seconds,
            "prompt_tokens_total": total_prompt_tokens,
            "output_tokens_total": total_output_tokens,
            "total_tokens": total_prompt_tokens + total_output_tokens,
            "decode_tokens_per_second": total_output_tokens / seconds if seconds > 0 else 0.0,
            "decode_tokens_per_second_per_request": (
                total_output_tokens / seconds / len(prompts) if seconds > 0 and prompts else 0.0
            ),
            "end_to_end_tokens_per_second": (
                (total_prompt_tokens + total_output_tokens) / seconds if seconds > 0 else 0.0
            ),
            **output_summary,
        }
        round_metrics.append(metrics)
        print("round_metrics=" + json.dumps(metrics, sort_keys=True))
        preview = outputs[0].outputs[0].text[: args.print_preview_chars]
        if preview:
            print(f"round_{round_idx}_first_output_preview={preview!r}")
    return {
        "load_seconds": load_seconds,
        "llm_kwargs": llm_kwargs,
        "rounds": round_metrics,
    }


def rank_bounds(total: int, rank: int, size: int) -> tuple[int, int]:
    floor = total // size
    remainder = total % size
    start = rank * floor + min(rank, remainder)
    end = start + floor + (1 if rank < remainder else 0)
    return start, end


def run_dp_rank(
    args_dict: dict[str, Any],
    prompts: list[str],
    prompt_lens: list[int],
    max_tokens: int,
    max_model_len: int,
    dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    metrics_dir: str,
) -> None:
    args = argparse.Namespace(**args_dict)
    apply_vllm_env(args)
    if args.dp_mode == "vllm-offline":
        os.environ["VLLM_DP_RANK"] = str(dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    else:
        for key in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            os.environ.pop(key, None)
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        tp = max(1, int(args.tensor_parallel_size))
        start_device = dp_rank * tp
        if visible:
            if len(visible) < dp_size * tp:
                raise RuntimeError(
                    "independent DP needs at least DP*TP visible GPUs; "
                    f"got {len(visible)} visible={visible}, dp={dp_size}, tp={tp}"
                )
            selected = visible[start_device : start_device + tp]
        else:
            selected = [str(start_device + offset) for offset in range(tp)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)

    start, end = rank_bounds(len(prompts), dp_rank, dp_size)
    local_prompts = prompts[start:end] or ["Placeholder"]
    local_lens = prompt_lens[start:end] or [1]
    args.batch_size = len(local_prompts)
    args.max_num_seqs = args.max_num_seqs or len(local_prompts)
    local_batched_tokens = args.max_num_batched_tokens or round_up(max(local_lens) * len(local_prompts), 1024)

    register_olmo3sink(Path(args.olmo3sink_src), args.skip_olmo3sink_register)
    print(
        "dp_rank_start="
        + json.dumps(
            {
                "rank": dp_rank,
                "dp_size": dp_size,
                "prompt_start": start,
                "prompt_end": end,
                "num_prompts": len(local_prompts),
                "max_num_batched_tokens": local_batched_tokens,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "dp_mode": args.dp_mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    rank_started = time.perf_counter()
    result = run_single_process_benchmark(
        args,
        local_prompts,
        local_lens,
        max_tokens,
        max_model_len,
        local_batched_tokens,
    )
    result["rank"] = dp_rank
    result["rank_wall_seconds"] = time.perf_counter() - rank_started
    result["prompt_indices"] = [start, end]
    out_path = Path(metrics_dir) / f"rank_{dp_rank}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"dp_rank_done rank={dp_rank} metrics={out_path}", flush=True)


def run_data_parallel_benchmark(
    args: argparse.Namespace,
    prompts: list[str],
    prompt_lens: list[int],
    max_tokens: int,
    max_model_len: int,
) -> dict[str, Any]:
    dp_size = args.data_parallel_size
    metrics_dir = Path(args.dp_output_dir or f"/tmp/olmo3sink_vllm_dp_metrics_{int(time.time())}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dp_master_port = args.dp_master_port or (get_open_port() if args.dp_mode == "vllm-offline" else 0)
    dp_master_ip = args.dp_master_ip
    args_dict = vars(args).copy()

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    started = time.perf_counter()
    for rank in range(dp_size):
        proc = ctx.Process(
            target=run_dp_rank,
            args=(
                args_dict,
                prompts,
                prompt_lens,
                max_tokens,
                max_model_len,
                rank,
                dp_size,
                dp_master_ip,
                dp_master_port,
                str(metrics_dir),
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(args.dp_timeout_seconds)
        if proc.exitcode is None:
            print(f"dp_rank_timeout pid={proc.pid}; killing after {args.dp_timeout_seconds}s", flush=True)
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            print(f"dp_rank_failed pid={proc.pid} exitcode={proc.exitcode}", flush=True)
            exit_code = proc.exitcode
    wall_seconds = time.perf_counter() - started
    if exit_code:
        raise RuntimeError(f"At least one DP rank failed; metrics_dir={metrics_dir}")

    per_rank = []
    for rank in range(dp_size):
        per_rank.append(json.loads((metrics_dir / f"rank_{rank}.json").read_text()))
    output_total = 0
    prompt_total = sum(prompt_lens)
    for item in per_rank:
        for round_item in item.get("rounds", []):
            output_total += int(round_item.get("output_tokens_total", 0))
    aggregate = {
        "data_parallel_size": dp_size,
        "dp_mode": args.dp_mode,
        "wall_seconds": wall_seconds,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "total_tokens": prompt_total + output_total,
        "decode_tokens_per_second": output_total / wall_seconds if wall_seconds > 0 else 0.0,
        "decode_tokens_per_second_per_request": (
            output_total / wall_seconds / len(prompts) if wall_seconds > 0 and prompts else 0.0
        ),
        "end_to_end_tokens_per_second": (
            (prompt_total + output_total) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "metrics_dir": str(metrics_dir),
        "per_rank": per_rank,
    }
    print("dp_aggregate_metrics=" + json.dumps(aggregate, sort_keys=True))
    return aggregate


def summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
    output_lens = [len(item.outputs[0].token_ids) for item in outputs]
    finish_reasons = Counter(str(item.outputs[0].finish_reason) for item in outputs)
    return {
        "num_outputs": len(outputs),
        "output_tokens_total": sum(output_lens),
        "output_tokens_min": min(output_lens) if output_lens else 0,
        "output_tokens_max": max(output_lens) if output_lens else 0,
        "output_tokens_mean": statistics.mean(output_lens) if output_lens else 0.0,
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }


def parse_args() -> argparse.Namespace:
    default_src = repo_root_from_script() / "src"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="OLMo3Sink HF checkpoint path or repo id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to --model.")
    parser.add_argument("--olmo3sink-src", default=str(default_src), help="Path containing the olmo3_sink package.")
    parser.add_argument("--skip-olmo3sink-register", action="store_true", help="Skip local OLMo3Sink/vLLM registration.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--total-output-tokens", type=int, default=131072)
    parser.add_argument("--max-tokens-per-request", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument(
        "--dp-mode",
        choices=("independent", "vllm-offline"),
        default="independent",
        help=(
            "independent starts one regular vLLM engine per DP rank and pins it to a GPU slice. "
            "vllm-offline uses vLLM's offline DP environment variables, which rejects dense models in recent vLLM."
        ),
    )
    parser.add_argument("--dp-master-ip", default="127.0.0.1")
    parser.add_argument("--dp-master-port", type=int, default=0)
    parser.add_argument("--dp-timeout-seconds", type=int, default=7200)
    parser.add_argument("--dp-output-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--enforce-eager", type=str_to_bool, default=False)
    parser.add_argument("--disable-custom-all-reduce", type=str_to_bool, default=None)
    parser.add_argument("--trust-remote-code", type=str_to_bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ignore-eos", type=str_to_bool, default=True)
    parser.add_argument("--force-max-tokens", type=str_to_bool, default=True)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable-log-stats", type=str_to_bool, default=True)
    parser.add_argument("--vllm-extra-json", default=None, help="Extra JSON object merged into vLLM LLM kwargs.")
    parser.add_argument("--install-vllm-wheel", type=str_to_bool, default=True)
    parser.add_argument("--force-reinstall-vllm", type=str_to_bool, default=False)
    parser.add_argument("--vllm-wheel-url", default=DEFAULT_VLLM_RUNTIME_WHEEL_URL)
    parser.add_argument("--vllm-version-fragment", default=DEFAULT_VLLM_VERSION_FRAGMENT)
    parser.add_argument("--vllm-install-dir", default=DEFAULT_VLLM_INSTALL_DIR)
    parser.add_argument(
        "--vllm-disabled-kernels",
        default=os.environ.get("VLLM_DISABLED_KERNELS", DEFAULT_VLLM_DISABLED_KERNELS),
        help=(
            "Comma-separated VLLM_DISABLED_KERNELS value. The default disables the FlashInfer FP8 "
            "linear kernel because this cluster's TileLang libcudart stub lacks cudaDeviceReset."
        ),
    )
    parser.add_argument("--out-json", default=None, help="Optional path for benchmark metrics JSON.")
    parser.add_argument("--print-preview-chars", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.data_parallel_size <= 0:
        raise ValueError("--data-parallel-size must be positive")
    if args.batch_size < args.data_parallel_size:
        raise ValueError("--batch-size should be >= --data-parallel-size so every DP rank gets work")

    apply_vllm_env(args)
    ensure_vllm_pin(args)

    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    prompts: list[str] = []
    prompt_lens: list[int] = []
    for idx in range(args.batch_size):
        prompt, length = build_target_prompt(tokenizer, args.prompt_tokens, idx)
        prompts.append(prompt)
        prompt_lens.append(length)

    max_tokens = args.max_tokens_per_request
    if max_tokens is None:
        max_tokens = round_up(args.total_output_tokens, args.batch_size) // args.batch_size
    max_model_len = args.max_model_len or round_up(max(prompt_lens) + max_tokens + 256, 1024)
    max_num_batched_tokens = args.max_num_batched_tokens or round_up(max(prompt_lens) * args.batch_size, 1024)

    print("benchmark_config=" + json.dumps(
        {
            "batch_size": args.batch_size,
            "prompt_tokens_min": min(prompt_lens),
            "prompt_tokens_max": max(prompt_lens),
            "requested_max_tokens_per_request": max_tokens,
            "requested_output_tokens_total": max_tokens * args.batch_size,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "dp_mode": args.dp_mode,
            "kv_cache_dtype": args.kv_cache_dtype,
            "quantization": args.quantization,
            "force_max_tokens": args.force_max_tokens,
            "ignore_eos": args.ignore_eos,
            "vllm_wheel_url": args.vllm_wheel_url if args.install_vllm_wheel else None,
            "vllm_disabled_kernels": os.environ.get("VLLM_DISABLED_KERNELS"),
        },
        sort_keys=True,
    ))
    print("nvidia_smi_before=\n" + nvidia_smi_snapshot())

    if args.data_parallel_size > 1:
        add_src_to_path(Path(args.olmo3sink_src))
        install_vllm_plugin_shim(Path(args.olmo3sink_src))
        result = run_data_parallel_benchmark(args, prompts, prompt_lens, max_tokens, max_model_len)
        round_metrics = [result]
        load_seconds = None
        llm_kwargs = None
    else:
        register_olmo3sink(Path(args.olmo3sink_src), args.skip_olmo3sink_register)
        result = run_single_process_benchmark(
            args,
            prompts,
            prompt_lens,
            max_tokens,
            max_model_len,
            max_num_batched_tokens,
        )
        round_metrics = result["rounds"]
        load_seconds = result["load_seconds"]
        llm_kwargs = result["llm_kwargs"]

    print("nvidia_smi_after=\n" + nvidia_smi_snapshot())
    final = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "load_seconds": load_seconds,
        "prompt_tokens": prompt_lens,
        "max_tokens_per_request": max_tokens,
        "llm_kwargs": llm_kwargs,
        "data_parallel_size": args.data_parallel_size,
        "rounds": round_metrics,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final, indent=2, sort_keys=True))
        print(f"metrics_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
