from __future__ import annotations

import asyncio

from proof_opd_env import ProofOPDEnv


def make_env(*, num_verifiers: int = 2, refine_review_n: int = 2) -> ProofOPDEnv:
    env = ProofOPDEnv.__new__(ProofOPDEnv)
    env.enable_meta_verification = True
    env.require_closed_think = True
    env.num_verifiers = num_verifiers
    env.refine_review_n = refine_review_n
    return env


def verifier_output(score: float) -> str:
    return (
        "<think>check the proof</think>\n"
        "Here is my evaluation of the solution:\n"
        "The proof has been checked.\n\n"
        "Based on my evaluation, I rate the solution as:\n"
        f"\\boxed{{{score:g}}}"
    )


def verifier_state(score: float) -> dict:
    return {
        "proof_opd_stage": "verifier",
        "proof_opd_current_round": 0,
        "proof_opd_verify_index": 0,
        "proof_opd_generation": {"proof": "candidate proof"},
        "proof_opd_verifier_results": [],
        "proof_opd_pending_verifier_result": None,
        "proof_opd_stage_records": [],
        "input": {"problem": "prove the claim"},
        "trajectory": [
            {
                "completion": [{"role": "assistant", "content": verifier_output(score)}],
                "response": {"finish_reason": "stop"},
            }
        ],
    }


def verifier_result(
    verify_index: int,
    proof_score: float,
    *,
    meta_score: float = 1.0,
    perfect_skip: bool = False,
) -> dict:
    return {
        "verify_index": verify_index,
        "verifier_valid": True,
        "proof_score": proof_score,
        "verifier_evaluation": f"review {verify_index}",
        "meta_valid": not perfect_skip,
        "meta_score": meta_score,
        "meta_invalid_reason": "skipped_perfect_verifier" if perfect_skip else "",
        "meta_analysis": f"meta {verify_index}" if not perfect_skip else "",
    }


def test_perfect_verifier_skips_meta_with_neutral_multiplier() -> None:
    env = make_env()
    state = verifier_state(1.0)

    messages = asyncio.run(env._advance_after_completion(state))

    result = state["proof_opd_verifier_results"][0]
    assert state["proof_opd_stage"] == "verifier"
    assert "## Solution" in messages[0].content
    assert result["meta_valid"] is False
    assert result["meta_invalid_reason"] == "skipped_perfect_verifier"
    assert result["meta_score_effective"] == 1.0
    assert result["reward_term"] == 1.0


def test_nonperfect_verifier_still_runs_meta() -> None:
    env = make_env()
    state = verifier_state(0.5)

    messages = asyncio.run(env._advance_after_completion(state))

    assert state["proof_opd_stage"] == "meta"
    assert state["proof_opd_verifier_results"] == []
    assert state["proof_opd_pending_verifier_result"]["proof_score"] == 0.5
    assert "## Solution Evaluation" in messages[0].content


def test_refinement_reviews_prefer_low_scores_then_high_meta_scores() -> None:
    env = make_env(refine_review_n=2)
    payload = {
        "verifier_results": [
            verifier_result(0, 1.0, perfect_skip=True),
            verifier_result(1, 0.5, meta_score=0.5),
            verifier_result(2, 0.5, meta_score=1.0),
        ]
    }

    reviews = env._select_refinement_reviews(payload)

    assert reviews[0].startswith("Verifier #3")
    assert reviews[1].startswith("Verifier #2")


def test_refinement_reviews_use_perfect_scores_only_to_fill_the_limit() -> None:
    env = make_env(refine_review_n=2)
    payload = {
        "verifier_results": [
            verifier_result(0, 1.0, perfect_skip=True),
            verifier_result(1, 0.5, meta_score=0.5),
            verifier_result(2, 1.0, perfect_skip=True),
        ]
    }

    reviews = env._select_refinement_reviews(payload)

    assert reviews[0].startswith("Verifier #2")
    assert reviews[1].startswith("Verifier #1")
    assert "Meta-verifier score: not run (neutral 1.000)" in reviews[1]
