from __future__ import annotations

from pathlib import Path

from train_engine_rl import build_prime_policy_inference_config
from train_engine_rl import build_prime_rl_config
from train_engine_rl import build_prime_teacher_inference_config
from train_engine_rl import parse_args


def make_args(*extra: str):
    args, unknown = parse_args(
        [
            "--model_path",
            "/models/student",
            "--dataset_path",
            "/data/proofs.csv",
            "--prime_algorithm",
            "opd",
            "--prime_opd_teacher_model",
            "/models/teacher",
            "--prime_env_id",
            "proof-opd-env",
            "--prime_group_size",
            "1",
            *extra,
        ]
    )
    assert unknown == []
    return args


def test_packed_sequence_target_uses_prime_token_batching() -> None:
    args = make_args(
        "--max_seq_length",
        "8192",
        "--prime_packed_sequences_per_step",
        "64",
        "--prime_max_inflight_rollouts",
        "48",
        "--prime_max_inflight_questions",
        "4",
    )

    config = build_prime_rl_config(args, Path("/tmp/output"))

    assert config["orchestrator"]["batch_size"] is None
    assert config["orchestrator"]["token_batch_size"] == 8192 * 64
    assert config["orchestrator"]["max_inflight_questions"] == 4
    assert config["train_envs"][0]["group_size"] == 1


def test_default_keeps_rollout_count_batching() -> None:
    args = make_args("--prime_batch_size", "2")

    config = build_prime_rl_config(args, Path("/tmp/output"))

    assert config["orchestrator"]["batch_size"] == 2
    assert config["orchestrator"]["token_batch_size"] is None


def test_token_batching_requires_explicit_inflight_limit() -> None:
    args = make_args(
        "--prime_packed_sequences_per_step",
        "64",
    )

    try:
        build_prime_rl_config(args, Path("/tmp/output"))
    except ValueError as exc:
        assert "prime_max_inflight_rollouts" in str(exc)
    else:
        raise AssertionError("token batching should require an in-flight rollout limit")


def test_candidate_gate_is_forwarded_for_eight_member_groups() -> None:
    args = make_args(
        "--prime_group_size",
        "8",
        "--prime_proof_candidate_gate",
        "true",
        "--prime_proof_candidate_continue_count",
        "4",
    )

    config = build_prime_rl_config(args, Path("/tmp/output"))
    env = config["train_envs"][0]

    assert env["group_size"] == 8
    assert env["args"]["candidate_gate_enabled"] is True
    assert env["args"]["candidate_continue_count"] == 4


def test_inference_lm_heads_default_to_bfloat16_projection() -> None:
    args = make_args()

    combined = build_prime_rl_config(args, Path("/tmp/output"))
    policy = build_prime_policy_inference_config(args, Path("/tmp/policy"))
    teacher = build_prime_teacher_inference_config(args, Path("/tmp/teacher"))

    assert combined["inference"]["enable_fp32_lm_head"] is False
    assert policy["inference"]["enable_fp32_lm_head"] is False
    assert teacher["inference"]["enable_fp32_lm_head"] is False


def test_fp32_inference_lm_heads_remain_explicitly_available() -> None:
    args = make_args(
        "--prime_vllm_enable_fp32_lm_head",
        "true",
        "--prime_opd_teacher_vllm_enable_fp32_lm_head",
        "true",
    )

    policy = build_prime_policy_inference_config(args, Path("/tmp/policy"))
    teacher = build_prime_teacher_inference_config(args, Path("/tmp/teacher"))

    assert policy["inference"]["enable_fp32_lm_head"] is True
    assert teacher["inference"]["enable_fp32_lm_head"] is True
