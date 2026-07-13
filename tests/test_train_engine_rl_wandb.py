from pathlib import Path

from train_engine_rl import resolve_wandb_shared_run_id


def test_shared_wandb_id_is_stable_for_run_name(tmp_path: Path) -> None:
    env = {"OLMO_RUN_DIR_NAME": "prime-opd-example"}

    first = resolve_wandb_shared_run_id(env, tmp_path / "first")
    second = resolve_wandb_shared_run_id(env, tmp_path / "second")

    assert first == second
    assert len(first) == 32


def test_explicit_shared_wandb_id_takes_precedence(tmp_path: Path) -> None:
    env = {
        "OLMO_RUN_DIR_NAME": "prime-opd-example",
        "WANDB_SHARED_RUN_ID": "resume-this-run",
    }

    assert resolve_wandb_shared_run_id(env, tmp_path) == "resume-this-run"
