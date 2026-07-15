from __future__ import annotations

import tomllib
from pathlib import Path

from train_engine_rl import build_distributed_trainer_command
from train_engine_rl import build_prime_rl_config
from train_engine_rl import build_prime_sft_config
from train_engine_rl import parse_args
from train_engine_rl import run_distributed_sft_component
from train_engine_rl import write_prime_sft_config
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


def test_sft_config_uses_external_hsdp_and_fused_loss(tmp_path: Path) -> None:
    args, unknown = parse_args(
        [
            "--prime_algorithm",
            "sft",
            "--prime_component",
            "sft_trainer",
            "--model_path",
            str(tmp_path / "model"),
            "--max_seq_length",
            "131072",
            "--max_train_steps",
            "1000",
            "--prime_trainer_num_nodes",
            "8",
            "--prime_train_gpus",
            "8",
            "--prime_trainer_master_addr",
            "10.0.0.1",
            "--prime_gpus_per_node",
            "8",
            "--prime_trainer_dp_replicate",
            "8",
            "--prime_trainer_model_impl",
            "custom",
            "--prime_trainer_attn",
            "olmo3_sink_fa3",
            "--prime_trainer_fp8",
            "true",
            "--prime_sft_activation_offloading",
            "true",
            "--prime_sft_activation_offloading_max_inflight",
            "1",
            "--optimizer",
            "te_fused_adamw",
            "--learning_rate",
            "2e-7",
            "--prime_lr_scheduler",
            "cosine",
            "--prime_lr_min",
            "3e-8",
        ]
    )
    assert unknown == []
    config = build_prime_sft_config(
        args,
        tmp_path / "output",
        tmp_path / "train.parquet",
        tmp_path / "validation.parquet",
    )
    config_path = tmp_path / "sft.toml"
    write_prime_sft_config(config_path, config)
    loaded = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert loaded["deployment"] == {
        "type": "multi_node",
        "launcher": "external",
        "gpus_per_node": 8,
        "num_nodes": 8,
        "nodes_per_fsdp_group": 1,
    }
    assert loaded["model"]["dp_replicate"] == 8
    assert loaded["model"]["impl"] == "custom"
    assert loaded["model"]["attn"] == "olmo3_sink_fa3"
    assert loaded["model"]["fp8"] is True
    assert loaded["model"]["ac_offloading"] == {
        "pin_memory": True,
        "max_inflight_activations": 1,
    }
    assert loaded["data"]["batch_size"] == 64
    assert loaded["data"]["seq_len"] == 131072
    assert loaded["data"]["overflow_policy"] == "skip"
    assert loaded["loss_impl"] == "liger_fused"
    assert loaded["scheduler"] == {
        "type": "cosine",
        "warmup_steps": 10,
        "min_lr": 3e-8,
    }

    command = build_distributed_trainer_command(
        args,
        config_path,
        trainer_module="prime_rl.trainer.sft.train",
    )
    assert "prime_rl.trainer.sft.train" in command


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


def test_sft_dry_run_does_not_launch_torchrun(tmp_path: Path, monkeypatch) -> None:
    args, unknown = parse_args(
        [
            "--prime_algorithm",
            "sft",
            "--prime_component",
            "sft_trainer",
            "--model_path",
            str(tmp_path / "model"),
            "--prime_trainer_master_addr",
            "127.0.0.1",
            "--dry_run_prime_rl",
        ]
    )
    assert unknown == []
    config = build_prime_sft_config(
        args,
        tmp_path / "output",
        tmp_path / "train.parquet",
        tmp_path / "validation.parquet",
    )
    monkeypatch.setattr(
        "train_engine_rl.run_logged_subprocess",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torchrun launched")),
    )

    assert run_distributed_sft_component(args, config, tmp_path / "logs") == 0
    assert (tmp_path / "logs" / "sft.toml").is_file()
