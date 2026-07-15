from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "operator_commands" / "prime_rl_sft_8node_per_turn_ctx131072.sh"


def run_preview(
    tmp_path: Path,
    *,
    gpu_name: str,
    compute_cap: str,
    requested: str = "",
    train_nodes: str = "0",
) -> str:
    shared_root = tmp_path / "shared"
    rendezvous_dir = shared_root / "rendezvous"
    rendezvous_dir.mkdir(parents=True)
    for node in train_nodes.split(","):
        (rendezvous_dir / f"node{node}.ip").write_text("127.0.0.1\n")

    env = {
        **os.environ,
        "GLOBAL_RANK": "0",
        "PRIME_SFT_TRAIN_NODES": train_nodes,
        "PRIME_SFT_RUN_NAME": f"test-{tmp_path.name}",
        "PRIME_SFT_SHARED_ROOT": str(shared_root),
        "PRIME_SFT_LOCAL_TMP_ROOT": str(tmp_path / "local"),
        "PRIME_GPU_NAMES_OVERRIDE": gpu_name,
        "PRIME_GPU_COMPUTE_CAPS_OVERRIDE": compute_cap,
        "PRIME_COMMAND_PREVIEW": "1",
    }
    if requested:
        env["PRIME_TRAINER_ATTN"] = requested
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout + result.stderr


def test_hopper_preview_uses_magi_fa3_and_native_override(tmp_path: Path) -> None:
    output = run_preview(tmp_path / "magi", gpu_name="NVIDIA H200", compute_cap="9.0")
    assert "attention=olmo3_sink_fa3" in output
    assert "--prime_algorithm sft" in output
    assert "--prime_component sft_trainer" in output
    assert "--fetch-update" in output
    assert "--prime_sft_global_batch_size 16" in output
    assert "--prime_sft_micro_batch_size 1" in output
    assert "--prime_trainer_optim_cpu_offload true" in output
    assert "micro_batch=1 grad_accum=2 global_batch=16" in output

    native_output = run_preview(
        tmp_path / "native",
        gpu_name="NVIDIA H200",
        compute_cap="9.0",
        requested="olmo3_sink_fa3_native",
    )
    assert "attention=olmo3_sink_fa3_native" in native_output


def test_eight_node_preview_uses_global_batch_128(tmp_path: Path) -> None:
    output = run_preview(
        tmp_path,
        gpu_name="NVIDIA H200",
        compute_cap="9.0",
        train_nodes="0,1,2,3,4,5,6,7",
    )
    assert "--prime_sft_global_batch_size 128" in output
    assert "--prime_sft_micro_batch_size 1" in output
    assert "--prime_trainer_optim_cpu_offload true" in output
    assert "micro_batch=1 grad_accum=2 global_batch=128" in output


def test_blackwell_preview_forces_fa2(tmp_path: Path) -> None:
    output = run_preview(
        tmp_path,
        gpu_name="NVIDIA B200",
        compute_cap="10.0",
        requested="olmo3_sink_fa3_native",
    )
    assert "attention=olmo3_sink_fa2" in output


def test_explicit_batch_and_accumulation_must_agree(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "GLOBAL_RANK": "0",
        "PRIME_SFT_TRAIN_NODES": "0",
        "PRIME_SFT_RUN_NAME": f"test-{tmp_path.name}",
        "PRIME_SFT_SHARED_ROOT": str(tmp_path / "shared"),
        "PRIME_SFT_LOCAL_TMP_ROOT": str(tmp_path / "local"),
        "PRIME_GPU_NAMES_OVERRIDE": "NVIDIA H200",
        "PRIME_GPU_COMPUTE_CAPS_OVERRIDE": "9.0",
        "PRIME_COMMAND_PREVIEW": "1",
        "PRIME_SFT_GLOBAL_BATCH_SIZE": "64",
        "PRIME_SFT_GRAD_ACCUM_STEPS": "2",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "explicit global batch resolves to grad_accum=8, requested 2" in result.stderr
