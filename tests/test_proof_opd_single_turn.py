from __future__ import annotations

import asyncio
import json

import pytest
from verifiers.clients.client import Client
from verifiers.types import Response
from verifiers.types import ResponseMessage
from verifiers.types import ResponseTokens
from verifiers.types import Usage

from proof_opd_env import ProofOPDSingleTurnEnv
from proof_opd_env import load_environment
from proof_opd_env import normalize_single_turn_dataset_rows
from proof_opd_env import record_policy_token_metrics


class RecordingClient(Client):
    def __init__(self) -> None:
        self.call_count = 0
        self.prompts = []

    def setup_client(self, config):
        return None

    async def to_native_tool(self, tool):
        return None

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        return None

    async def raise_from_native_response(self, response):
        return None

    async def from_native_response(self, response):
        return response

    async def close(self) -> None:
        return None

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs) -> Response:
        self.call_count += 1
        self.prompts.append(prompt)
        return Response(
            id="single-turn-response",
            created=0,
            model=model,
            usage=Usage(
                prompt_tokens=4,
                reasoning_tokens=2,
                completion_tokens=3,
                total_tokens=7,
            ),
            message=ResponseMessage(
                content="<evaluation>valid</evaluation><score>1</score>",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=ResponseTokens(
                    prompt_ids=[1, 2, 3, 4],
                    prompt_mask=[1, 1, 1, 1],
                    completion_ids=[5, 6, 7],
                    completion_mask=[1, 1, 1],
                    completion_logprobs=[-0.1, -0.2, -0.3],
                ),
                tool_calls=None,
            ),
        )


def source_row(stage: str, index: int = 0) -> dict:
    messages = [
        {"role": "system", "content": f"system-{stage}"},
        {"role": "user", "content": f"ready prompt {stage}"},
    ]
    return {
        "pipeline": "opd_v2",
        "run_id": "run-1",
        "problem_id": f"problem-{index}",
        "problem": "Prove that 1 = 1.",
        "stage": stage,
        "candidate_id": f"candidate-{index}",
        "verifier_idx": index,
        "messages_json": json.dumps(messages),
    }


def test_single_turn_normalization_preserves_prompts_and_filters_stages() -> None:
    rows = [source_row(stage, index) for index, stage in enumerate(
        ["prove", "verify", "select", "refine", "plain", "meta"]
    )]

    normalized = normalize_single_turn_dataset_rows(rows, max_examples=None)

    assert [row["stage"] for row in normalized] == ["prove", "verify", "select", "refine"]
    for source, row in zip(rows[:4], normalized, strict=True):
        assert row["prompt"] == json.loads(source["messages_json"])
        assert row["info"]["stage"] == source["stage"]
        assert row["task_type"] == "opd_single_turn"


def test_single_turn_normalization_rejects_missing_messages() -> None:
    with pytest.raises(ValueError, match="zero usable rows"):
        normalize_single_turn_dataset_rows(
            [{"stage": "prove", "messages_json": "not-json"}],
            max_examples=None,
        )


def test_single_turn_factory_uses_direct_prompt_dataset(tmp_path) -> None:
    path = tmp_path / "per_turn.jsonl"
    rows = [source_row("prove"), source_row("verify", 1)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    env = load_environment(str(path), dataset_mode="single")

    assert isinstance(env, ProofOPDSingleTurnEnv)
    assert env.max_turns == 1
    assert len(env.dataset) == 2
    assert env.dataset[0]["prompt"] == json.loads(rows[0]["messages_json"])
    assert env.dataset[0]["info"]["stage"] == "prove"


def test_single_turn_rollout_makes_exactly_one_model_call(tmp_path) -> None:
    path = tmp_path / "per_turn.jsonl"
    row = source_row("verify")
    path.write_text(json.dumps(row), encoding="utf-8")
    env = load_environment(str(path), dataset_mode="single")
    client = RecordingClient()

    state = asyncio.run(env.rollout(env.dataset[0], client=client, model="test-model"))

    assert client.call_count == 1
    assert [message.model_dump(exclude_none=True) for message in client.prompts[0]] == json.loads(
        row["messages_json"]
    )
    assert len(state["trajectory"]) == 1
    assert state["info"]["proof_opd_trace"]["stage"] == "verify"
    assert state["info"]["proof_opd_trace"]["stage_records"][0]["raw_chars"] > 0
    assert "generated_tokens" not in state["info"]["proof_opd_trace"]
    assert "generated_tokens" not in state["info"]["proof_opd_trace"]["stage_records"][0]

    asyncio.run(env.rubric.score_rollout(state))
    assert state["metrics"]["proof_opd_policy_generated_tokens"] == 3
    assert state["metrics"]["proof_opd_policy_reasoning_tokens"] == 2
    assert state["metrics"]["proof_opd_verify_policy_generated_tokens"] == 3
    assert "proof_opd_prove_policy_generated_tokens" not in state["metrics"]


def test_policy_token_metrics_accumulate_across_stages() -> None:
    state = {}
    record_policy_token_metrics(
        state,
        "proof",
        {"prompt_tokens": 10, "generated_tokens": 20, "reasoning_tokens": 5, "total_tokens": 30},
    )
    record_policy_token_metrics(
        state,
        "verifier",
        {"prompt_tokens": 30, "generated_tokens": 7, "reasoning_tokens": 2, "total_tokens": 37},
    )
    record_policy_token_metrics(
        state,
        "verifier",
        {"prompt_tokens": 31, "generated_tokens": 8, "reasoning_tokens": 3, "total_tokens": 39},
    )

    assert state["metrics"]["proof_opd_policy_generated_tokens"] == 35
    assert state["metrics"]["proof_opd_proof_policy_generated_tokens"] == 20
    assert state["metrics"]["proof_opd_verifier_policy_generated_tokens"] == 15
    assert state["metrics"]["proof_opd_verifier_policy_reasoning_tokens"] == 5
