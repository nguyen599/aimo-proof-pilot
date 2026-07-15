from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prime_sft import build_per_turn_sft_cache


def write_source(path: Path) -> None:
    rows = []
    for problem_id, stage in [
        ("p1", "prove"),
        ("p1", "verify"),
        ("p2", "select"),
        ("p3", "refine"),
        ("p4", "plain"),
    ]:
        rows.append(
            {
                "stage": stage,
                "problem_id": problem_id,
                "problem": f"problem text {problem_id}",
                "nm_uuid": f"uuid-{problem_id}",
                "messages_json": json.dumps(
                    [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": f"question {problem_id}"},
                    ]
                ),
                "reasoning_content": f"reasoning {stage}",
                "content": f"answer {stage}",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_per_turn_conversion_preserves_targets_and_groups_validation(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "per_turn.parquet"
    write_source(source_path)

    train_path, validation_path, manifest_path = build_per_turn_sft_cache(
        source_path,
        tmp_path / "cache",
        validation_problem_count=1,
        validation_seed=7,
        seq_len=32,
        batch_size=2,
    )

    train_rows = pq.read_table(train_path).to_pylist()
    validation_rows = pq.read_table(validation_path).to_pylist()
    all_rows = train_rows + validation_rows
    assert len(all_rows) == 4
    assert {row["stage"] for row in all_rows} == {"prove", "verify", "select", "refine"}
    for row in all_rows:
        assistant = row["messages"][-1]
        assert assistant["role"] == "assistant"
        assert assistant["reasoning_content"] == f"reasoning {row['stage']}"
        assert assistant["content"] == f"answer {row['stage']}"

    train_problem_ids = {row["problem_id"] for row in train_rows}
    validation_problem_ids = {row["problem_id"] for row in validation_rows}
    assert train_problem_ids.isdisjoint(validation_problem_ids)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sum(manifest["row_counts"]["train"].values()) == len(train_rows)
    assert sum(manifest["row_counts"]["validation"].values()) == len(validation_rows)


def test_per_turn_cache_is_reused(tmp_path: Path) -> None:
    source_path = tmp_path / "per_turn.parquet"
    write_source(source_path)
    first = build_per_turn_sft_cache(
        source_path, tmp_path / "cache", validation_problem_count=1
    )
    second = build_per_turn_sft_cache(
        source_path, tmp_path / "cache", validation_problem_count=1
    )
    assert first == second


def write_nemotron_source(path: Path) -> None:
    rows = [
        ("proof", "new problem", "## Solution\nA complete proof."),
        ("verification", "new problem", "Here is my evaluation.\\n\\boxed{1}"),
        ("meta-verification", "new problem", "Here is my analysis.\\n\\boxed{1}"),
        ("proof", "problem text p1", "## Solution\nA held-out overlap."),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for index, (subset, problem, assistant) in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "uuid": f"nemotron-{index}",
                        "problem": problem,
                        "messages": [
                            {"role": "user", "content": f"prompt for {problem}"},
                            {
                                "role": "assistant",
                                "reasoning_content": f"reasoning {index}",
                                "content": assistant,
                            },
                        ],
                        "tools": [],
                        "subset": subset,
                    }
                )
                + "\n"
            )


def test_nemotron_messages_are_mixed_without_reformatting(tmp_path: Path) -> None:
    source_path = tmp_path / "per_turn.parquet"
    nemotron_path = tmp_path / "nemotron.jsonl"
    write_source(source_path)
    write_nemotron_source(nemotron_path)

    train_path, validation_path, manifest_path = build_per_turn_sft_cache(
        source_path,
        tmp_path / "cache",
        nemotron_source_path=nemotron_path,
        validation_problem_count=0,
        batch_size=2,
    )

    train_rows = pq.read_table(train_path).to_pylist()
    assert pq.read_table(validation_path).num_rows == 0
    nemotron_rows = [
        row
        for row in train_rows
        if row["source_dataset"] == "nvidia/Nemotron-Math-Proofs-v2"
    ]
    assert len(nemotron_rows) == 4
    assert {row["stage"] for row in nemotron_rows} == {"prove", "verify", "meta"}
    assert nemotron_rows[0]["messages"][-1]["reasoning_content"] == "reasoning 0"
    assert (
        nemotron_rows[0]["messages"][-1]["content"] == "## Solution\nA complete proof."
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_counts_by_source"]["nvidia/Nemotron-Math-Proofs-v2"] == {
        "meta": 1,
        "prove": 2,
        "verify": 1,
    }


def test_nemotron_rows_matching_validation_problems_are_excluded(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "per_turn.parquet"
    nemotron_path = tmp_path / "nemotron.jsonl"
    write_source(source_path)
    write_nemotron_source(nemotron_path)

    train_path, _, manifest_path = build_per_turn_sft_cache(
        source_path,
        tmp_path / "cache",
        nemotron_source_path=nemotron_path,
        validation_problem_count=4,
        batch_size=2,
    )

    train_rows = pq.read_table(train_path).to_pylist()
    assert not any(
        row["source_dataset"] == "nvidia/Nemotron-Math-Proofs-v2"
        and row["problem"] == "problem text p1"
        for row in train_rows
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["nemotron_validation_overlap_counts"] == {"proof": 1}
