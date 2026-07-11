from __future__ import annotations

import asyncio

from proof_opd_env import ProofOPDEnv
from proof_opd_env import parse_selector_response


def make_env() -> ProofOPDEnv:
    env = ProofOPDEnv.__new__(ProofOPDEnv)
    env.selector_top_k = 3
    env.require_closed_think = True
    return env


def candidate_round(
    round_index: int,
    *,
    format_score: float,
    verifier_meta_reward: float,
    reward: float,
) -> dict:
    return {
        "round_index": round_index,
        "format_score": format_score,
        "verifier_meta_reward": verifier_meta_reward,
        "reward": reward,
        "proof": f"proof {round_index}",
        "stage_records": [],
    }


def test_selector_parser_uses_last_well_formed_id() -> None:
    parsed = parse_selector_response(
        "<think>compare candidates</think>\n"
        "<selected_id>1</selected_id>\n"
        "<selected_id>Candidate 2</selected_id>"
    )

    assert parsed["selected_id"] == 2
    assert parsed["closed_thinking"] is True
    assert parsed["visible_output"].endswith("<selected_id>Candidate 2</selected_id>")


def test_selector_candidates_use_requested_reward_formula_and_top_three() -> None:
    env = make_env()
    state = {
        "proof_opd_rounds": [
            candidate_round(0, format_score=0.7, verifier_meta_reward=0.5, reward=0.99),
            candidate_round(1, format_score=1.0, verifier_meta_reward=0.9, reward=0.1),
            candidate_round(2, format_score=1.0, verifier_meta_reward=0.8, reward=0.1),
            candidate_round(3, format_score=1.0, verifier_meta_reward=0.6, reward=0.1),
        ]
    }

    candidates = env._rank_selector_candidates(state)

    assert [candidate["round_index"] for candidate in candidates] == [1, 2, 3]
    assert [candidate["candidate_id"] for candidate in candidates] == [1, 2, 3]
    assert [candidate["preselection_score"] for candidate in candidates] == [0.9, 0.8, 0.6]


def test_selector_prompt_contains_only_the_top_ranked_proofs() -> None:
    env = make_env()
    state = {
        "input": {"problem": "choose a proof"},
        "proof_opd_rounds": [
            candidate_round(0, format_score=1.0, verifier_meta_reward=0.1, reward=0.1),
            candidate_round(1, format_score=1.0, verifier_meta_reward=0.9, reward=0.9),
            candidate_round(2, format_score=1.0, verifier_meta_reward=0.8, reward=0.8),
            candidate_round(3, format_score=1.0, verifier_meta_reward=0.7, reward=0.7),
        ],
    }

    messages = env._next_selector_prompt(state)
    prompt = messages[0].content

    assert "### Candidate 1\nproof 1" in prompt
    assert "### Candidate 2\nproof 2" in prompt
    assert "### Candidate 3\nproof 3" in prompt
    assert "proof 0" not in prompt


def test_valid_selector_uses_selected_candidate_reward() -> None:
    env = make_env()
    rounds = [
        candidate_round(0, format_score=1.0, verifier_meta_reward=0.9, reward=0.9),
        candidate_round(1, format_score=1.0, verifier_meta_reward=0.8, reward=0.8),
        candidate_round(2, format_score=1.0, verifier_meta_reward=0.6, reward=0.6),
    ]
    state = {
        "proof_opd_rounds": rounds,
        "proof_opd_selector_candidates": env._rank_selector_candidates({"proof_opd_rounds": rounds}),
        "proof_opd_stage_records": [{"stage": "selector"}],
    }
    selector = parse_selector_response("<think>candidate two is strongest</think><selected_id>2</selected_id>")

    payload = env._finalize_selector(state, selector, "", finish_reason="stop")

    assert payload["selected_round_index"] == 1
    assert payload["reward"] == 0.8
    assert payload["selector_selected_id"] == 2
    assert payload["selector_valid"] is True
    assert payload["selector_fallback_used"] is False
    assert payload["best_round_reward"] == 0.9
    assert state["proof_opd_selector"]["selected_round_index"] == 1


def test_invalid_selector_falls_back_to_highest_preselection_score() -> None:
    env = make_env()
    rounds = [
        candidate_round(0, format_score=1.0, verifier_meta_reward=0.4, reward=0.4),
        candidate_round(1, format_score=1.0, verifier_meta_reward=0.9, reward=0.9),
    ]
    candidates = env._rank_selector_candidates({"proof_opd_rounds": rounds})
    state = {
        "proof_opd_rounds": rounds,
        "proof_opd_selector_candidates": candidates,
        "proof_opd_stage_records": [{"stage": "selector"}],
    }
    selector = parse_selector_response("<think>unfinished")
    invalid_reason = env._selector_invalid_reason(selector, False, len(candidates))

    payload = env._finalize_selector(state, selector, invalid_reason)

    assert invalid_reason == "missing_closed_think"
    assert payload["selected_round_index"] == 1
    assert payload["reward"] == 0.9
    assert payload["selector_selected_id"] == 1
    assert payload["selector_valid"] is False
    assert payload["selector_fallback_used"] is True


def test_selector_stage_is_processed_before_environment_stops() -> None:
    env = make_env()
    rounds = [
        candidate_round(0, format_score=1.0, verifier_meta_reward=0.9, reward=0.9),
        candidate_round(1, format_score=1.0, verifier_meta_reward=0.8, reward=0.8),
    ]
    state = {
        "proof_opd_stage": "selector",
        "proof_opd_current_round": 1,
        "proof_opd_verify_index": 0,
        "proof_opd_rounds": rounds,
        "proof_opd_selector_candidates": env._rank_selector_candidates({"proof_opd_rounds": rounds}),
        "proof_opd_stage_records": [],
        "trajectory": [
            {
                "completion": [
                    {
                        "role": "assistant",
                        "content": "<think>choose the second</think><selected_id>2</selected_id>",
                    }
                ],
                "response": {"finish_reason": "stop"},
            }
        ],
    }

    response = asyncio.run(env._advance_after_completion(state))

    assert response == []
    assert state["final_env_response"] == []
    assert state["proof_opd_reward_payload"]["selected_round_index"] == 1
    assert state["proof_opd_stage_records"][-1]["stage"] == "selector"
