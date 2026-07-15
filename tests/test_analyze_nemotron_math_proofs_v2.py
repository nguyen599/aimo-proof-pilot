from __future__ import annotations

import json

from scripts.analyze_nemotron_math_proofs_v2 import analyze, tokenized_length


def test_tokenized_length_accepts_plain_and_mapping_results() -> None:
    assert tokenized_length([1, 2, 3]) == 3
    assert tokenized_length([[1, 2, 3]]) == 3
    assert tokenized_length({"input_ids": [1, 2, 3]}) == 3
    assert tokenized_length({"input_ids": [[1, 2, 3]]}) == 3


def test_analyze_counts_reasoning_content_as_assistant_target(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "subset": "proof",
                "license": "cc-by-4.0",
                "source": "AoPS",
                "dataset": "Nemotron-Math-Proofs-v2",
                "uuid": "row-1",
                "problem": "Prove it.",
                "tools": [],
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "reasoning_content": "private reasoning",
                        "content": "## Solution\nAnswer.\n\\boxed{1}",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze(
        source,
        tokenizer_path=None,
        token_sample_size=0,
        seq_len=131072,
    )

    lengths = report["character_length_stats"]["proof"]
    assert lengths["assistant_reasoning_chars"]["max"] == len("private reasoning")
    assert lengths["assistant_final_chars"]["max"] == len(
        "## Solution\nAnswer.\n\\boxed{1}"
    )
    assert lengths["assistant_target_chars"]["max"] == (
        len("private reasoning") + len("## Solution\nAnswer.\n\\boxed{1}")
    )
    assert report["format_counts"]["proof"]["has_reasoning_content"] == 1
