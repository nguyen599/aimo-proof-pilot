from proof_opd_env import ProofOPDEnv


def make_env() -> ProofOPDEnv:
    return ProofOPDEnv.__new__(ProofOPDEnv)


def test_continuation_stages_preempt_new_proof_generation() -> None:
    env = make_env()
    late_proof = {
        "proof_opd_stage": "proof",
        "proof_opd_question_sequence": 100,
    }
    early_verifier = {
        "proof_opd_stage": "verifier",
        "proof_opd_question_sequence": 1,
    }

    assert env._request_priority(early_verifier) < env._request_priority(late_proof)


def test_earlier_question_wins_within_the_same_stage() -> None:
    env = make_env()
    early_meta = {
        "proof_opd_stage": "meta",
        "proof_opd_question_sequence": 3,
    }
    late_meta = {
        "proof_opd_stage": "meta",
        "proof_opd_question_sequence": 9,
    }

    assert env._request_priority(early_meta) < env._request_priority(late_meta)
