from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "operator_commands" / "prime_rl_sft_8node_per_turn_ctx131072.sh"


def run_preview(tmp_path: Path, *, gpu_name: str, compute_cap: str, requested: str = "") -> str:
    env = {
        **os.environ,
        "GLOBAL_RANK": "0",
        "PRIME_SFT_TRAIN_NODES": "0",
        "PRIME_SFT_RUN_NAME": f"test-{tmp_path.name}",
        "PRIME_SFT_SHARED_ROOT": str(tmp_path / "shared"),
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
    assert "--prime_sft_micro_batch_size 2" in output
    assert "micro_batch=2 grad_accum=1 global_batch=16" in output

    native_output = run_preview(
        tmp_path / "native",
        gpu_name="NVIDIA H200",
        compute_cap="9.0",
        requested="olmo3_sink_fa3_native",
    )
    assert "attention=olmo3_sink_fa3_native" in native_output


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
    assert "explicit global batch resolves to grad_accum=4, requested 2" in result.stderr
