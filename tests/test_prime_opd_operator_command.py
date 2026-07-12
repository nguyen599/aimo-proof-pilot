from __future__ import annotations

import os
import subprocess
from pathlib import Path

import train
from train import prime_rl_runtime_requirements
from train import prime_rl_source_requirements


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMAND = REPO_ROOT / "operator_commands" / "prime_rl_opd_8node_full_vocab_dpsk_ctx81920_nodes345.sh"


def test_prime_runtime_installs_verifiers_sandbox_dependency(monkeypatch) -> None:
    monkeypatch.delenv("PRIME_RL_RUNTIME_REQUIREMENTS", raising=False)

    requirements = prime_rl_runtime_requirements()

    assert any(requirement.startswith("prime-sandboxes>=") for requirement in requirements)


def test_cuda12_uses_vllm_bundled_deep_gemm(monkeypatch) -> None:
    monkeypatch.delenv("PRIME_RL_SOURCE_REQUIREMENTS", raising=False)
    monkeypatch.setattr(train, "DEFAULT_CUDA_MAJOR", 12)

    requirements = prime_rl_source_requirements()

    assert not any(requirement.startswith("deep-gemm @") for requirement in requirements)


def test_one_node_layout_uses_two_train_four_policy_two_teacher_gpus(tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in ("NODE_RANK", "SLURM_NODEID", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(name, None)
    env.update(
        {
            "GLOBAL_RANK": "0",
            "PRIME_NODE_LAYOUT": "1node",
            "PRIME_COMMAND_PREVIEW": "1",
            "PRIME_3NODE_CLEAN_ROLE_PROCS": "0",
            "PRIME_3NODE_ROLE_LOCK": str(tmp_path / "role.lock"),
            "PRIME_3NODE_TMP_ROOT": str(tmp_path / "runtime"),
            "PRIME_3NODE_RENDEZVOUS_DIR": str(tmp_path / "rendezvous"),
            "PRIME_TRAIN_PYTHON": os.sys.executable,
            "PRIME_TRAIN_ENTRYPOINT": str(REPO_ROOT / "src" / "train.py"),
            "OLMO_RUN_DIR_NAME": "one_node_layout_test",
        }
    )

    result = subprocess.run(
        ["bash", str(COMMAND)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout

    assert "node=0 role=full" in output
    assert "policy_topology tp=1 dp=4" in output
    assert "trainer_gpus=2 teacher_topology tp=2 dp=1 gpu_ids=6,7" in output
    assert "policy GPUs 0-3, trainer GPUs 4-5, teacher GPUs 6,7" in output
    assert "--prime_component full" in output
    assert "--prime_train_gpus 2" in output
    assert "--prime_infer_gpus 4" in output
    assert "--prime_vllm_data_parallel_size 4" in output
    assert "--prime_opd_teacher_vllm_tensor_parallel_size 2" in output
    assert "--prime_temperature 0.7" in output
    assert os.sys.executable in output
    assert str(REPO_ROOT / "src" / "train.py") in output
    assert "disable_custom_all_reduce" not in output
