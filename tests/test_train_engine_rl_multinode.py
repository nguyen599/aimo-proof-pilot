from __future__ import annotations

import tomllib
from pathlib import Path

from train_engine_rl import build_distributed_trainer_command
from train_engine_rl import build_prime_rl_config
from train_engine_rl import parse_args
from train_engine_rl import write_prime_trainer_config


def test_cosine_scheduler_is_written_to_standalone_trainer_config(tmp_path: Path) -> None:
    args, unknown = parse_args(
        [
            "--model_path",
            str(tmp_path / "model"),
            "--prime_lr_scheduler",
            "cosine",
            "--prime_lr_warmup_steps",
            "25",
            "--prime_lr_min",
            "1e-8",
            "--prime_packed_sequences_per_step",
            "0",
        ]
    )
    assert unknown == []
    config = build_prime_rl_config(args, tmp_path / "output")
    config_path = tmp_path / "trainer.toml"
    write_prime_trainer_config(config_path, config)

    loaded = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert loaded["scheduler"] == {
        "type": "cosine",
        "warmup_steps": 25,
        "min_lr": 1e-8,
    }


def test_distributed_trainer_command_uses_requested_node_rank(tmp_path: Path) -> None:
    args, _ = parse_args(
        [
            "--prime_component",
            "trainer_worker",
            "--prime_trainer_num_nodes",
            "2",
            "--prime_trainer_node_rank",
            "1",
            "--prime_trainer_master_addr",
            "10.0.0.1",
            "--prime_trainer_master_port",
            "29400",
            "--prime_train_gpus",
            "8",
        ]
    )
    command = build_distributed_trainer_command(args, tmp_path / "trainer.toml")
    joined = " ".join(command)
    assert "--nnodes 2" in joined
    assert "--nproc-per-node 8" in joined
    assert "--node-rank 1" in joined
    assert "--rdzv-endpoint 10.0.0.1:29400" in joined


def test_hsdp_replication_is_written_to_trainer_model_config(tmp_path: Path) -> None:
    args, unknown = parse_args(
        [
            "--model_path",
            str(tmp_path / "model"),
            "--prime_trainer_dp_replicate",
            "2",
            "--prime_packed_sequences_per_step",
            "0",
        ]
    )
    assert unknown == []

    config = build_prime_rl_config(args, tmp_path / "output")
    config_path = tmp_path / "trainer.toml"
    write_prime_trainer_config(config_path, config)

    loaded = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert loaded["model"]["dp_replicate"] == 2
