from __future__ import annotations

from pathlib import Path

import huggingface_hub

from train_engine_rl import parse_args
from train_engine_rl import resolve_runtime_assets


def write_fake_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_full_component_materializes_hf_models_and_dataset(tmp_path: Path, monkeypatch) -> None:
    snapshot_calls: list[str] = []

    def fake_snapshot_download(*, repo_id: str, local_dir: str, **_kwargs) -> str:
        snapshot_calls.append(repo_id)
        write_fake_model(Path(local_dir))
        return local_dir

    dataset_source = tmp_path / "hub" / "per_turn.parquet"
    dataset_source.parent.mkdir(parents=True)
    dataset_source.write_bytes(b"dataset")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_kwargs: str(dataset_source))

    args, unknown = parse_args(
        [
            "--prime_component",
            "full",
            "--prime_algorithm",
            "opd",
            "--prime_env_id",
            "proof-opd-env",
            "--model_hf_repo",
            "example/student",
            "--prime_opd_teacher_hf_repo",
            "example/teacher",
            "--dataset_hf_repo",
            "example/proofs",
            "--dataset_hf_filename",
            "per_turn.parquet",
            "--hf_assets_dir",
            str(tmp_path / "assets"),
        ]
    )
    assert unknown == []

    resolve_runtime_assets(args)

    assert snapshot_calls == ["example/student", "example/teacher"]
    assert Path(args.model_path, "config.json").is_file()
    assert Path(args.prime_opd_teacher_model, "config.json").is_file()
    assert Path(args.dataset_path).read_bytes() == b"dataset"


def test_policy_component_does_not_download_teacher(tmp_path: Path, monkeypatch) -> None:
    snapshot_calls: list[str] = []

    def fake_snapshot_download(*, repo_id: str, local_dir: str, **_kwargs) -> str:
        snapshot_calls.append(repo_id)
        write_fake_model(Path(local_dir))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    args, _ = parse_args(
        [
            "--prime_component",
            "policy_inference",
            "--prime_algorithm",
            "opd",
            "--model_hf_repo",
            "example/student",
            "--prime_opd_teacher_hf_repo",
            "example/teacher",
            "--hf_assets_dir",
            str(tmp_path / "assets"),
        ]
    )

    resolve_runtime_assets(args)

    assert snapshot_calls == ["example/student"]
    assert args.prime_opd_teacher_model == "example/teacher"
