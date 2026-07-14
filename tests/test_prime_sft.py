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


def test_per_turn_conversion_preserves_targets_and_groups_validation(tmp_path: Path) -> None:
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
    first = build_per_turn_sft_cache(source_path, tmp_path / "cache", validation_problem_count=1)
    second = build_per_turn_sft_cache(source_path, tmp_path / "cache", validation_problem_count=1)
    assert first == second
