#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


EXPECTED_SUBSETS = {"proof", "verification", "meta-verification"}
SCORE_PATTERN = re.compile(r"\\boxed\s*\{\s*(?:0(?:\.5)?|1)\s*\}")


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def summarize(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": round(fmean(values), 2) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def tokenized_length(value: Any) -> int:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("tokenizer result mapping does not contain input_ids")
        value = value["input_ids"]
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise TypeError("tokenizer returned a scalar input_ids value")
        return int(shape[-1])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"unsupported tokenizer result: {type(value).__name__}")
    if (
        value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise ValueError(
                f"expected one tokenized sample, got batch size {len(value)}"
            )
        value = value[0]
    return len(value)


def update_token_sample(
    sample_heap: list[tuple[int, int, str, list[dict[str, Any]]]],
    *,
    sample_size: int,
    source_index: int,
    subset: str,
    uuid: str,
    messages: list[dict[str, Any]],
) -> None:
    if sample_size <= 0:
        return
    key = int.from_bytes(hashlib.sha256(uuid.encode("utf-8")).digest()[:8], "big")
    item = (-key, source_index, subset, messages)
    if len(sample_heap) < sample_size:
        heapq.heappush(sample_heap, item)
    elif key < -sample_heap[0][0]:
        heapq.heapreplace(sample_heap, item)


def analyze(
    source_path: Path,
    *,
    tokenizer_path: str | None,
    token_sample_size: int,
    seq_len: int,
) -> dict[str, Any]:
    subset_counts: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    message_counts: Counter[int] = Counter()
    field_counts: Counter[str] = Counter()
    format_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {
            "prompt_chars": [],
            "assistant_reasoning_chars": [],
            "assistant_final_chars": [],
            "assistant_target_chars": [],
            "total_chars": [],
        }
    )
    licenses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    unique_uuids: set[str] = set()
    unique_problem_hashes: set[str] = set()
    duplicate_uuids = 0
    invalid_messages = 0
    nonempty_tools = 0
    blank_lines = 0
    token_sample: list[tuple[int, int, str, list[dict[str, Any]]]] = []

    with source_path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                blank_lines += 1
                continue
            row = json.loads(line)
            field_counts.update(row.keys())
            subset = str(row.get("subset"))
            subset_counts[subset] += 1
            licenses[str(row.get("license"))] += 1
            sources[str(row.get("source"))] += 1
            datasets[str(row.get("dataset"))] += 1
            if row.get("tools"):
                nonempty_tools += 1

            uuid = str(row.get("uuid"))
            if uuid in unique_uuids:
                duplicate_uuids += 1
            unique_uuids.add(uuid)
            problem = row.get("problem")
            if isinstance(problem, str):
                normalized_problem = " ".join(problem.split())
                unique_problem_hashes.add(
                    hashlib.sha256(normalized_problem.encode()).hexdigest()
                )

            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                invalid_messages += 1
                continue
            valid_messages = True
            roles: list[str] = []
            prompt_chars = 0
            assistant_reasoning_chars = 0
            assistant_final_chars = 0
            for message in messages:
                if not isinstance(message, dict):
                    valid_messages = False
                    break
                role = message.get("role")
                content = message.get("content")
                reasoning_content = message.get("reasoning_content")
                if role not in {
                    "system",
                    "user",
                    "assistant",
                    "tool",
                } or not isinstance(content, str):
                    valid_messages = False
                    break
                if reasoning_content is not None and not isinstance(
                    reasoning_content, str
                ):
                    valid_messages = False
                    break
                roles.append(role)
                if role == "assistant":
                    reasoning_content = reasoning_content or ""
                    assistant_reasoning_chars += len(reasoning_content)
                    assistant_final_chars += len(content)
                    format_counts[subset]["has_reasoning_content"] += bool(
                        reasoning_content.strip()
                    )
                    format_counts[subset]["has_think_tag"] += "<think>" in content
                    format_counts[subset]["has_end_think_tag"] += "</think>" in content
                    format_counts[subset]["has_boxed_score"] += bool(
                        SCORE_PATTERN.search(content)
                    )
                    format_counts[subset]["starts_solution_heading"] += (
                        content.lstrip().startswith("## Solution")
                    )
                    format_counts[subset]["starts_evaluation_phrase"] += (
                        "Here is my evaluation" in content[:160]
                    )
                    format_counts[subset]["starts_analysis_phrase"] += (
                        "Here is my analysis" in content[:160]
                    )
                else:
                    prompt_chars += len(content)
            if not valid_messages or "assistant" not in roles:
                invalid_messages += 1
                continue

            role_sequences["->".join(roles)] += 1
            message_counts[len(messages)] += 1
            lengths[subset]["prompt_chars"].append(prompt_chars)
            lengths[subset]["assistant_reasoning_chars"].append(
                assistant_reasoning_chars
            )
            lengths[subset]["assistant_final_chars"].append(assistant_final_chars)
            lengths[subset]["assistant_target_chars"].append(
                assistant_reasoning_chars + assistant_final_chars
            )
            lengths[subset]["total_chars"].append(
                prompt_chars + assistant_reasoning_chars + assistant_final_chars
            )
            update_token_sample(
                token_sample,
                sample_size=token_sample_size,
                source_index=source_index,
                subset=subset,
                uuid=uuid,
                messages=messages,
            )

    report: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "row_count": sum(subset_counts.values()),
        "blank_lines": blank_lines,
        "subset_counts": dict(sorted(subset_counts.items())),
        "unexpected_subsets": sorted(set(subset_counts) - EXPECTED_SUBSETS),
        "unique_problem_count": len(unique_problem_hashes),
        "unique_uuid_count": len(unique_uuids),
        "duplicate_uuid_count": duplicate_uuids,
        "invalid_message_row_count": invalid_messages,
        "nonempty_tools_row_count": nonempty_tools,
        "role_sequence_counts": dict(sorted(role_sequences.items())),
        "message_count_histogram": {
            str(key): value for key, value in sorted(message_counts.items())
        },
        "field_presence_counts": dict(sorted(field_counts.items())),
        "license_counts": dict(sorted(licenses.items())),
        "source_counts": dict(sorted(sources.items())),
        "dataset_counts": dict(sorted(datasets.items())),
        "format_counts": {
            subset: dict(sorted(counts.items()))
            for subset, counts in sorted(format_counts.items())
        },
        "character_length_stats": {
            subset: {name: summarize(values) for name, values in subset_lengths.items()}
            for subset, subset_lengths in sorted(lengths.items())
        },
    }

    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        token_lengths: dict[str, list[int]] = defaultdict(list)
        tokenization_errors: list[dict[str, Any]] = []
        for _, source_index, subset, messages in sorted(token_sample, reverse=True):
            try:
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                token_lengths[subset].append(tokenized_length(token_ids))
            except Exception as exc:
                tokenization_errors.append(
                    {
                        "source_index": source_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        all_token_lengths = [
            value for values in token_lengths.values() for value in values
        ]
        report["tokenizer_sample"] = {
            "tokenizer_path": tokenizer_path,
            "requested_sample_size": token_sample_size,
            "successful_sample_size": len(all_token_lengths),
            "errors": tokenization_errors[:20],
            "length_stats": {
                "all": summarize(all_token_lengths),
                **{
                    subset: summarize(values)
                    for subset, values in sorted(token_lengths.items())
                },
            },
            "over_seq_len": {
                "seq_len": seq_len,
                "count": sum(value > seq_len for value in all_token_lengths),
                "fraction": (
                    round(
                        sum(value > seq_len for value in all_token_lengths)
                        / len(all_token_lengths),
                        6,
                    )
                    if all_token_lengths
                    else None
                ),
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--token-sample-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=131072)
    args = parser.parse_args()
    report = analyze(
        args.source,
        tokenizer_path=args.tokenizer,
        token_sample_size=args.token_sample_size,
        seq_len=args.seq_len,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
