from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PER_TURN_CACHE_VERSION = 1
DEFAULT_PER_TURN_STAGES = ("prove", "verify", "select", "refine")


def parse_stage_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    stages = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not stages:
        raise ValueError("At least one SFT stage must be selected")
    return stages


def _source_fingerprint(source_path: Path, settings: dict[str, Any]) -> str:
    stat = source_path.stat()
    digest = hashlib.sha256()
    digest.update(str(source_path.resolve()).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    with source_path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:24]


def _validation_problem_ids(
    source_path: Path,
    stages: set[str],
    count: int,
    seed: int,
) -> set[str]:
    problem_ids: set[str] = set()
    parquet = pq.ParquetFile(source_path)
    for batch in parquet.iter_batches(batch_size=4096, columns=["stage", "problem_id"]):
        for stage, problem_id in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if stage in stages:
                if problem_id is None:
                    raise ValueError("per_turn.parquet contains a selected row with null problem_id")
                problem_ids.add(str(problem_id))
    if not problem_ids:
        raise ValueError(f"No rows matched SFT stages: {sorted(stages)}")
    ranked = sorted(
        problem_ids,
        key=lambda problem_id: hashlib.sha256(f"{seed}:{problem_id}".encode("utf-8")).digest(),
    )
    return set(ranked[: min(count, len(ranked))])


def _output_schema() -> pa.Schema:
    message = pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string()),
            pa.field("reasoning_content", pa.string()),
        ]
    )
    return pa.schema(
        [
            pa.field("messages", pa.list_(message), nullable=False),
            pa.field("stage", pa.string(), nullable=False),
            pa.field("problem_id", pa.string(), nullable=False),
            pa.field("source_index", pa.int64(), nullable=False),
            pa.field("prompt_tokens", pa.int64()),
            pa.field("completion_tokens", pa.int64()),
            pa.field("finish_reason", pa.string()),
        ]
    )


def _normalize_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    try:
        prompt_messages = json.loads(row["messages_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid messages_json at source row {source_index}") from exc
    if not isinstance(prompt_messages, list) or not prompt_messages:
        raise ValueError(f"messages_json must be a nonempty list at source row {source_index}")

    messages: list[dict[str, str | None]] = []
    for message_index, message in enumerate(prompt_messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Message {message_index} is not an object at source row {source_index}"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError(
                f"Invalid role/content in message {message_index} at source row {source_index}"
            )
        messages.append({"role": role, "content": content, "reasoning_content": None})

    reasoning = row.get("reasoning_content")
    content = row.get("content")
    if not isinstance(reasoning, str) or not isinstance(content, str):
        raise ValueError(f"Missing assistant reasoning/content at source row {source_index}")
    if not reasoning.strip() and not content.strip():
        raise ValueError(f"Empty assistant target at source row {source_index}")
    messages.append(
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
        }
    )
    return {
        "messages": messages,
        "stage": row["stage"],
        "problem_id": str(row["problem_id"]),
        "source_index": source_index,
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "finish_reason": row.get("finish_reason"),
    }


def build_per_turn_sft_cache(
    source_path: Path,
    cache_root: Path,
    *,
    stages: Sequence[str] = DEFAULT_PER_TURN_STAGES,
    validation_problem_count: int = 33,
    validation_seed: int = 34521,
    seq_len: int = 131072,
    batch_size: int = 256,
) -> tuple[Path, Path, Path]:
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"SFT source parquet does not exist: {source_path}")
    if validation_problem_count < 0:
        raise ValueError("validation_problem_count must be nonnegative")
    selected_stages = parse_stage_names(stages)
    settings = {
        "version": PER_TURN_CACHE_VERSION,
        "stages": selected_stages,
        "validation_problem_count": validation_problem_count,
        "validation_seed": validation_seed,
        "seq_len": seq_len,
    }
    fingerprint = _source_fingerprint(source_path, settings)
    cache_dir = cache_root.expanduser().resolve() / fingerprint
    train_path = cache_dir / "train.parquet"
    validation_path = cache_dir / "validation.parquet"
    manifest_path = cache_dir / "manifest.json"
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        return train_path, validation_path, manifest_path

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = cache_root / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    schema = _output_schema()
    train_writer = pq.ParquetWriter(temporary_dir / "train.parquet", schema, compression="zstd")
    validation_writer = pq.ParquetWriter(
        temporary_dir / "validation.parquet", schema, compression="zstd"
    )
    validation_ids = _validation_problem_ids(
        source_path,
        set(selected_stages),
        validation_problem_count,
        validation_seed,
    )
    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    estimated_overflow: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    columns = [
        "stage",
        "problem_id",
        "messages_json",
        "reasoning_content",
        "content",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
    ]
    source_index = 0
    try:
        try:
            parquet = pq.ParquetFile(source_path)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                output_rows: dict[str, list[dict[str, Any]]] = {
                    "train": [],
                    "validation": [],
                }
                for row in batch.to_pylist():
                    row_index = source_index
                    source_index += 1
                    stage = row.get("stage")
                    if stage not in selected_stages:
                        continue
                    normalized = _normalize_row(row, row_index)
                    split = (
                        "validation"
                        if normalized["problem_id"] in validation_ids
                        else "train"
                    )
                    output_rows[split].append(normalized)
                    split_counts[split][stage] += 1
                    prompt_tokens = normalized.get("prompt_tokens")
                    completion_tokens = normalized.get("completion_tokens")
                    if (
                        isinstance(prompt_tokens, int)
                        and isinstance(completion_tokens, int)
                        and prompt_tokens + completion_tokens > seq_len
                    ):
                        estimated_overflow[split][stage] += 1
                for split, rows in output_rows.items():
                    if not rows:
                        continue
                    table = pa.Table.from_pylist(rows, schema=schema)
                    (validation_writer if split == "validation" else train_writer).write_table(
                        table
                    )
        finally:
            train_writer.close()
            validation_writer.close()
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    if not split_counts["train"]:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise ValueError("Per-turn SFT conversion produced an empty training split")
    manifest = {
        "cache_version": PER_TURN_CACHE_VERSION,
        "fingerprint": fingerprint,
        "source_path": str(source_path),
        "source_size": source_path.stat().st_size,
        "settings": settings,
        "validation_problem_ids": sorted(validation_ids),
        "row_counts": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "estimated_overflow_counts": {
            split: dict(sorted(counts.items())) for split, counts in estimated_overflow.items()
        },
    }
    (temporary_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        temporary_dir.replace(cache_dir)
    except OSError:
        if not manifest_path.is_file():
            raise
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return train_path, validation_path, manifest_path


def prepare_distributed_per_turn_sft_cache(
    source_path: Path,
    cache_root: Path,
    *,
    node_rank: int,
    coordination_id: str,
    timeout_seconds: int,
    log: Callable[[str], None],
    **build_kwargs: Any,
) -> tuple[Path, Path, Path]:
    selected_stages = parse_stage_names(build_kwargs.get("stages", DEFAULT_PER_TURN_STAGES))
    settings = {
        "version": PER_TURN_CACHE_VERSION,
        "stages": selected_stages,
        "validation_problem_count": build_kwargs.get("validation_problem_count", 33),
        "validation_seed": build_kwargs.get("validation_seed", 34521),
        "seq_len": build_kwargs.get("seq_len", 131072),
    }
    fingerprint = _source_fingerprint(source_path.expanduser().resolve(), settings)
    cache_dir = cache_root.expanduser().resolve() / fingerprint
    paths = (cache_dir / "train.parquet", cache_dir / "validation.parquet", cache_dir / "manifest.json")
    if all(path.is_file() for path in paths):
        log(f"Using normalized per-turn SFT cache: {cache_dir}")
        return paths

    status_dir = cache_root.expanduser().resolve() / ".coordination"
    status_dir.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.sha256(coordination_id.encode("utf-8")).hexdigest()[:16]
    success_path = status_dir / f"{fingerprint}-{safe_id}.ready"
    error_path = status_dir / f"{fingerprint}-{safe_id}.error"
    if node_rank == 0:
        success_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)
        try:
            log(f"Normalizing per-turn SFT dataset from {source_path}")
            paths = build_per_turn_sft_cache(source_path, cache_root, **build_kwargs)
            success_path.write_text(str(paths[2]) + "\n", encoding="utf-8")
            log(f"Normalized per-turn SFT cache ready: {paths[2].parent}")
            return paths
        except Exception as exc:
            error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise

    deadline = time.monotonic() + timeout_seconds
    log(f"Waiting for trainer node 0 to normalize per-turn SFT data: {cache_dir}")
    while time.monotonic() < deadline:
        if error_path.is_file():
            raise RuntimeError(error_path.read_text(encoding="utf-8").strip())
        if success_path.is_file() and all(path.is_file() for path in paths):
            return paths
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for normalized per-turn SFT cache: {cache_dir}")
