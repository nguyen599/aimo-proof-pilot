from __future__ import annotations

import asyncio
import logging
import re

from datasets import Dataset
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.trajectories import trace_to_samples
from verifiers.clients.client import Client
from verifiers.types import Response
from verifiers.types import ResponseMessage
from verifiers.v1.legacy import rollout_output_to_trace

from proof_opd_env import ProofOPDEnv
from proof_opd_env import ProofOPDRubric
from proof_opd_env import normalize_dataset_rows
from proof_opd_env import parse_generation_response


class StageClient(Client):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._config = None
        self.call_count = 0

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
        prompt_text = "\n".join(str(getattr(message, "content", "")) for message in prompt)
        if "## Instruction" in prompt_text:
            content = (
                "<think>verify</think>\n"
                "Here is my evaluation of the solution:\nThe proof is correct.\n"
                "Based on my evaluation, the final overall score should be:\n\\boxed{1}"
            )
        else:
            content = generation_output(1.0, f"call-{self.call_count}")
        return Response(
            id=f"response-{self.call_count}",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=content,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )


class SelectorStageClient(StageClient):
    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs) -> Response:
        self.call_count += 1
        prompt_text = "\n".join(str(getattr(message, "content", "")) for message in prompt)
        if "<selected_id>ID</selected_id>" in prompt_text:
            content = "<think>select the strongest candidate</think><selected_id>1</selected_id>"
        elif 'assess whether this "solution evaluation" is reasonable' in prompt_text:
            content = (
                '<think>check the evaluation</think>\nHere is my analysis of the "solution evaluation":\n'
                "The evaluation is reasonable.\n"
                'Based on my analysis, I rate the "solution evaluation" as:\n\\boxed{1}'
            )
        elif "## Instruction" in prompt_text:
            content = (
                "<think>verify</think>\n"
                "Here is my evaluation of the solution:\nThe proof has a minor gap.\n"
                "Based on my evaluation, the final overall score should be:\n\\boxed{0.5}"
            )
        else:
            content = generation_output(1.0, f"call-{self.call_count}")
        return Response(
            id=f"response-{self.call_count}",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=content,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )


def generation_output(score: float, label: str) -> str:
    return (
        f"<think>reason about {label}</think>\n"
        f"## Solution\nProof candidate {label}.\n"
        "## Self Evaluation\n"
        "Here is my evaluation of the solution: the proof was checked.\n"
        f"\\boxed{{{score:g}}}"
    )


def make_env(*, continue_count: int = 4, num_verifiers: int = 2) -> ProofOPDEnv:
    rows = normalize_dataset_rows(
        [{"problem": "Prove that 1=1."}],
        problem_column="auto",
        solution_column="auto",
        max_examples=None,
    )
    dataset = Dataset.from_list(rows)
    return ProofOPDEnv(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=ProofOPDRubric(),
        message_type="chat",
        refine_rounds=1,
        num_verifiers=num_verifiers,
        refine_review_n=1,
        enable_meta_verification=True,
        candidate_gate_enabled=True,
        candidate_continue_count=continue_count,
    )


def proof_state(group_id: str, candidate_index: int, output: str, logprob: float) -> dict:
    return {
        "input": {
            "problem": "Prove that 1=1.",
            "info": {
                "candidate_group_id": group_id,
                "candidate_group_size": 8,
                "candidate_index": candidate_index,
                "candidate_continue_count": 4,
            },
        },
        "proof_opd_stage": "proof",
        "proof_opd_current_round": 0,
        "proof_opd_rounds": [],
        "proof_opd_stage_records": [],
        "proof_opd_verify_index": 0,
        "proof_opd_verifier_results": [],
        "proof_opd_pending_verifier_result": None,
        "proof_opd_selector_candidates": [],
        "proof_opd_selector": None,
        "proof_opd_reward_payload": None,
        "trajectory": [
            {
                "completion": [{"role": "assistant", "content": output}],
                "response": {"finish_reason": "stop"},
                "tokens": {
                    "completion_logprobs": [logprob, logprob],
                },
            }
        ],
    }


def install_group(env: ProofOPDEnv, group_id: str, expected: int, continue_count: int) -> None:
    env._candidate_gate_groups[group_id] = {
        "condition": asyncio.Condition(),
        "expected": expected,
        "continue_count": continue_count,
        "records": {},
        "selected": None,
        "ranks": {},
        "selector_records": {},
        "selector_candidates": None,
        "selector_leader": None,
    }


def test_candidate_gate_selects_four_of_eight() -> None:
    async def run() -> tuple[list[bool], list[dict]]:
        env = make_env(continue_count=4)
        group_id = "group-eight"
        install_group(env, group_id, expected=8, continue_count=4)
        states = [
            proof_state(
                group_id,
                candidate_index,
                generation_output(1.0 if candidate_index < 4 else 0.5, str(candidate_index)),
                -0.1 - candidate_index,
            )
            for candidate_index in range(8)
        ]
        selected = await asyncio.gather(
            *(
                env._candidate_gate_after_proof(
                    state,
                    parse_generation_response(state["trajectory"][-1]["completion"][0]["content"]),
                    "",
                )
                for state in states
            )
        )
        return list(selected), states

    selected, states = asyncio.run(run())

    assert selected == [True, True, True, True, False, False, False, False]
    assert [state["input"]["info"]["candidate_rank"] for state in states] == list(range(1, 9))


def test_unselected_candidate_stops_after_proof_without_verifier() -> None:
    async def run() -> tuple[ProofOPDEnv, dict, dict, list, list]:
        env = make_env(continue_count=1)
        group_id = "group-two"
        install_group(env, group_id, expected=2, continue_count=1)
        selected_state = proof_state(group_id, 0, generation_output(1.0, "selected"), -0.1)
        proof_only_state = proof_state(group_id, 1, generation_output(0.5, "proof-only"), -1.0)
        selected_messages, proof_only_messages = await asyncio.gather(
            env._advance_after_completion(selected_state),
            env._advance_after_completion(proof_only_state),
        )
        return env, selected_state, proof_only_state, selected_messages, proof_only_messages

    env, selected_state, proof_only_state, selected_messages, proof_only_messages = asyncio.run(run())

    assert selected_state["proof_opd_stage"] == "verifier"
    assert len(selected_messages) == 1
    assert "evaluate the quality" in selected_messages[0].content
    assert proof_only_messages == []
    assert proof_only_state["final_env_response"] == []
    assert proof_only_state["proof_opd_reward_payload"]["reason"] == "candidate_gate_proof_only"
    assert proof_only_state["proof_opd_verifier_results"] == []
    assert env.requires_group_rollouts is True


def test_perfect_round_stops_without_refinement_or_selector() -> None:
    env = make_env(continue_count=1, num_verifiers=1)
    generation = parse_generation_response(generation_output(1.0, "perfect"))
    state = {
        "input": {"problem": "Prove that 1=1.", "info": {}},
        "proof_opd_current_round": 0,
        "proof_opd_rounds": [],
        "proof_opd_stage_records": [],
        "proof_opd_verify_index": 0,
        "proof_opd_generation": generation,
        "proof_opd_verifier_results": [
            {
                "verify_index": 0,
                "verifier": {"evaluation": "The proof is correct.", "raw_output": "", "score": 1.0},
                "verifier_valid": True,
                "proof_score": 1.0,
                "verifier_evaluation": "The proof is correct.",
                "meta": None,
                "meta_valid": False,
                "meta_score": 1.0,
                "meta_invalid_reason": "skipped_perfect_verifier",
                "meta_analysis": "",
            }
        ],
        "proof_opd_selector_candidates": [],
    }

    messages = asyncio.run(env._after_verifier_result(state))

    assert messages == []
    assert state["final_env_response"] == []
    assert state["proof_opd_reward_payload"]["reward"] == 1.0
    assert state["proof_opd_reward_payload"]["selector_invalid_reason"] == "skipped_early_stop_reward"


def test_run_group_generates_eight_proofs_but_only_four_verifier_calls() -> None:
    env = make_env(continue_count=4, num_verifiers=1)
    env.refine_rounds = 0
    client = StageClient()
    row = dict(env.get_dataset()[0])

    outputs = asyncio.run(
        env.run_group(
            group_inputs=[dict(row) for _ in range(8)],
            client=client,
            model="test-model",
            sampling_args={},
            state_columns=["trajectory"],
        )
    )

    trajectory_lengths = sorted(len(output["trajectory"]) for output in outputs)
    selected = [bool(output["info"].get("candidate_selected")) for output in outputs]
    assert trajectory_lengths == [1, 1, 1, 1, 2, 2, 2, 3]
    assert sum(selected) == 4
    assert client.call_count == 13


def test_selected_candidate_trains_every_stage_including_selector() -> None:
    env = make_env(continue_count=4, num_verifiers=1)
    env.refine_rounds = 0
    client = SelectorStageClient()
    row = dict(env.get_dataset()[0])

    outputs = asyncio.run(
        env.run_group(
            group_inputs=[dict(row) for _ in range(8)],
            client=client,
            model="test-model",
            sampling_args={},
            state_columns=["trajectory"],
        )
    )

    selected_outputs = [output for output in outputs if output["info"].get("candidate_selected")]
    proof_only_outputs = [output for output in outputs if not output["info"].get("candidate_selected")]
    assert len(selected_outputs) == 4
    assert len(proof_only_outputs) == 4
    assert sorted(len(output["trajectory"]) for output in selected_outputs) == [3, 3, 3, 4]
    assert all(len(output["trajectory"]) == 1 for output in proof_only_outputs)
    assert client.call_count == 17

    selected = next(output for output in selected_outputs if len(output["trajectory"]) == 4)
    selector_prompt = "\n".join(
        str(getattr(message, "content", "")) for message in selected["trajectory"][-1]["prompt"]
    )
    assert selector_prompt.count("### Candidate ") == 3
    selector_proofs = re.findall(r"### Candidate \d+\nProof candidate ([^.\n]+)", selector_prompt)
    assert len(selector_proofs) == 3
    assert len(set(selector_proofs)) == 3
    for step_index, step in enumerate(selected["trajectory"]):
        step["tokens"] = {
            "prompt_ids": [1000 + step_index],
            "completion_ids": [2000 + step_index, 3000 + step_index],
            "completion_logprobs": [-0.1, -0.2],
        }
    trace = rollout_output_to_trace(selected, task_idx=0)
    samples = trace_to_samples(trace, env_name="proof_math")

    assert trace.num_branches == 4
    assert len(samples) == 4
    assert all(sum(sample.mask) == 2 for sample in samples)
    assert [sample.token_ids[-2:] for sample in samples] == [
        [2000, 3000],
        [2001, 3001],
        [2002, 3002],
        [2003, 3003],
    ]
    for sample in samples:
        stamp_loss_routing(sample, "ref_kl")
        assert sample.ref_kl_weights == [float(value) for value in sample.mask]
        assert sample.rl_weights == [0.0] * len(sample.token_ids)
