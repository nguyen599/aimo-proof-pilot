from __future__ import annotations

import ast
import atexit
import asyncio
import contextlib
import csv
import glob
import hashlib
import json
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
import nest_asyncio

# Apply the patch to allow nested event loops
nest_asyncio.apply()

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

# The same model is used for proof generation, verification, meta-verification,
# refinement, and final candidate selection.
DEFAULT_API_KEY = "vllm-local"
DEFAULT_SERVED_MODEL_NAME = "proof-model"
DEFAULT_PROBLEM_TIMEOUT_SECONDS = 13800
DEFAULT_SELECTION_RESERVE_SECONDS = 900

DEFAULT_VLLM_EXTRA_ARGS = (
    "--generation-config vllm --kv-cache-dtype fp8 --max_num_batched_tokens 16384 --uvicorn-log-level warning"
)


class CFG:
    model_path = Path("/kaggle/input/models/nguyennguyen599/allenai-olmo-3.1-32b-think-aimo-proof-pilot/transformers/sft-v2-ckpt-1000-nvfp4-w4a16/1")
    input_csv = Path("/kaggle/input/competitions/ai-mathematical-olympiad-proof-pilot/test.csv")

    output_csv = Path("/kaggle/working/submission.csv")
    logdir = Path("/kaggle/working/proof_logs")

    num_ctx = 65536
    max_new_tokens = 64000
    thinking_budget_enabled = False
    proof_generation_thinking_budgets = [
        58500,
        58500,
        58500,
        58500,
        58500,
        58500,
        # 59000,
        # 59000,
        59000,
        59500,
        59500,
        59500,
        59500,
        60000,
        60000,
        60000,
    ]
    thinking_budget_force_text = "\nWe should now write the final solution due time limit.\n</think>\n\n## Solution\n"
    verifier_thinking_budget_tokens = 36000
    verifier_thinking_budget_force_text = (
        "\nWe should now write the final evaluation due time limit.\n</think>\n\n"
        "Here is my evaluation of the solution:\n"
    )
    meta_thinking_budget_tokens = 36000
    meta_thinking_budget_force_text = (
        "\nWe should now write the final analysis.\n</think>\n\n"
        'Here is my analysis of the "solution evaluation":\n'
    )
    verifier_max_new_tokens = 50000
    meta_max_new_tokens = 50000
    selector_max_new_tokens = 50000
    selector_max_candidate_chars = 32000

    temperature = 1.0
    proof_generation_temperatures = [0.7, 0.7, 0.6, 0.6, 0.7, 0.7, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0]
    top_p = 0.95
    top_k = -1
    min_new_tokens = 0
    min_p: Optional[float] = 0.01

    num_gpus = 1
    gpus = ""
    tensor_parallel_size = 0
    dtype = "auto"
    gpu_memory_utilization = 0.97
    max_num_seqs = 32
    max_concurrent_requests = 32
    max_concurrent_problems = 1

    pipelines_per_problem = 14
    # Last N candidates use a shorter proof-only prompt. This keeps some
    # candidates verifyable when full proof+self-evaluation generations hit
    # the context/output limit.
    proof_only_candidate_count = 0
    skip_self_score_zero = True
    wait_for_all_generations_before_verify = False
    verify_candidate_limit_while_generating = 2
    verify_request_limit_while_generating = 8
    verify_n = 4
    meta_n = 0
    meta_policy = "all-reviews" # low-only, all-reviews
    strict_pass_meta = True
    refine_rounds = 0
    refine_review_n = 2
    min_valid_low = 1
    problem_timeout_seconds = DEFAULT_PROBLEM_TIMEOUT_SECONDS
    selection_reserve_seconds = DEFAULT_SELECTION_RESERVE_SECONDS
    selection_temperature = 0.0
    selector_mode = "score" # llm, score
    selector_min_final_score = 0.5

    decomposition_strategy_enabled = False
    decomposition_fallback_to_direct = False
    decomposition_candidate_count = 0
    decomposition_max_sublemmas = 4
    architect_guided_json = False
    proof_architect_max_new_tokens = 8192
    sublemma_max_new_tokens = 40000
    proof_assembler_max_new_tokens = 50000

    vllm_extra_args = DEFAULT_VLLM_EXTRA_ARGS
    stream_interval = 100
    host = "127.0.0.1"
    port = 8000
    api_key = DEFAULT_API_KEY
    served_model_name = DEFAULT_SERVED_MODEL_NAME
    server_timeout = 3600
    no_serve = False
    base_url = ""
    stream_vllm = True
    stream_vllm_server_log = True
    max_rows = 0
    mock_llm = False
    verbose = True
QUESTION_COLUMN_CANDIDATES = ("problem", "question", "prompt", "theorem", "statement", "natural_problem")
ID_COLUMN_CANDIDATES = ("id", "problem_id", "question_id", "uuid", "question_uuid")
GOLD_PROOF_COLUMN_NAMES = {
    "solution",
    "proof",
    "reference_solution",
    "reference proof",
    "reference_proof",
    "gold_solution",
    "gold proof",
    "gold_proof",
    "official_solution",
    "official proof",
    "official_proof",
    "rubrics",
    "grading_scheme", # usamo
}
EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""

_SCORE_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}|"
    r"\bboxed\s*\{\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\}",
    flags=re.IGNORECASE,
)
_FALLBACK_SCORE_PATTERN = re.compile(
    r"(?:final\s+overall\s+score|score|rating)\s*(?:should\s+be|is|:)?\s*"
    r"\**\s*(0(?:\.5)?|1(?:\.0)?|0\.0)\s*\**\s*\.?\s*$",
    flags=re.IGNORECASE,
)
_SELECTED_INDEX_PATTERN = re.compile(r"selected[_\s-]*index\s*[:=]\s*(\d+)", flags=re.IGNORECASE)
_THINK_BLOCK_PATTERN = re.compile(r"(?is)<think>.*?</think>\s*")
_VERIFIER_EVALUATION_MARKERS = (
    "Here is my evaluation of the solution:",
    "Here is my evaluation of the candidate solution against the reference proof:",
)
_VERIFIER_EVALUATION_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?detailed[ \t]+evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?solution[ \t]+evaluation[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
)
_META_ANALYSIS_MARKERS = (
    'Here is my analysis of the "solution evaluation":',
    "Here is my analysis of the solution evaluation:",
)
_META_ANALYSIS_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?detailed[ \t]+analysis[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?analysis[ \t]*:?(?:\*\*)?[ \t]*$"
    ),
)
_VERIFIER_SCORE_TRAILER_PATTERN = re.compile(
    r"(?is)\n+\s*(?:\*\*)?Based on my evaluation[\s,:-]+.*?(?:final overall score|score).*",
)
_META_SCORE_TRAILER_PATTERN = re.compile(
    r"(?is)\n+\s*(?:\*\*)?Based on my analysis[\s,:-]+.*?(?:solution evaluation|rate|score).*",
)
_VISIBLE_OUTPUT_MARKERS = (
    "## Solution",
    "## Self Evaluation",
    "Based on my evaluation",
    "Based on my analysis",
    "Here is my evaluation",
    "Here is my analysis",
    "SELECTED_INDEX",
)
_HEADER_SUFFIX_PATTERN = r"[ \t]*:?[ \t]*$"
_DEFAULT_STAGE_TOKEN_LIMITS: dict[str, int] = {}
MAX_SUBMISSION_ANSWER_CHARS = 20_000
DEFAULT_FALLBACK_ANSWER = "0"
MAX_FORWARDED_EVALUATION_CHARS = 12_000
MAX_FORWARDED_META_ANALYSIS_CHARS = 8_000
MAX_SELECTOR_EVALUATION_EXCERPT_CHARS = 1_200
MAX_FORWARDED_SUBLEMMA_CHARS = 12_000
REPETITION_GUARD_RECENT_TOKENS = 500
REPETITION_GUARD_DUPLICATE_LINE_THRESHOLD = 8
_CLIPPED_TEXT_MARKER = "\n\n[... clipped middle reasoning before forwarding ...]\n\n"
ARCHITECT_GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_summary": {"type": "string"},
        "definitions_and_notation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["symbol", "meaning"],
            },
        },
        "chosen_strategy": {
            "type": "object",
            "properties": {
                "high_level_idea": {"type": "string"},
                "why_this_strategy_should_work": {"type": "string"},
                "main_bottleneck": {"type": "string"},
            },
            "required": [
                "high_level_idea",
                "why_this_strategy_should_work",
                "main_bottleneck",
            ],
        },
        "rejected_or_risky_strategies": {"type": "array", "items": {"type": "object"}},
        "dependency_graph": {"type": "array", "items": {"type": "object"}},
        "sublemma_tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "lemma_id": {"type": "string"},
                    "short_name": {"type": "string"},
                    "statement": {"type": "string"},
                    "role_in_solution": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "allowed_previous_lemmas": {"type": "array", "items": {"type": "string"}},
                    "suggested_proof_methods": {"type": "array", "items": {"type": "string"}},
                    "forbidden_shortcuts": {"type": "array", "items": {"type": "string"}},
                    "counterexample_tests": {"type": "array", "items": {"type": "string"}},
                    "expected_output_contract": {"type": "string"},
                },
                "required": [
                    "lemma_id",
                    "short_name",
                    "statement",
                    "role_in_solution",
                    "assumptions",
                    "allowed_previous_lemmas",
                    "suggested_proof_methods",
                    "forbidden_shortcuts",
                    "counterexample_tests",
                    "expected_output_contract",
                ],
            },
        },
        "final_assembly_plan": {"type": "array", "items": {"type": "object"}},
        "risk_register": {"type": "array", "items": {"type": "object"}},
    },
    "required": [
        "problem_summary",
        "definitions_and_notation",
        "chosen_strategy",
        "rejected_or_risky_strategies",
        "dependency_graph",
        "sublemma_tasks",
        "final_assembly_plan",
        "risk_register",
    ],
}


@dataclass
class SamplingConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_new_tokens: int
    min_p: Optional[float]


@dataclass
class VLLMConfig:
    model_path: str
    served_model_name: str
    host: str
    port: int
    api_key: str
    num_ctx: int
    dtype: str
    gpu_memory_utilization: float
    max_num_seqs: int
    tensor_parallel_size: int
    vllm_extra_args: str
    logdir: Path
    stream_interval: int = 10
    stream_server_logs: bool = True


@dataclass
class PipelineConfig:
    proof_max_new_tokens: int
    default_temperature: float
    proof_generation_temperatures: list[float]
    proof_only_candidate_count: int
    skip_self_score_zero: bool
    wait_for_all_generations_before_verify: bool
    thinking_budget_enabled: bool
    proof_generation_thinking_budgets: list[int]
    thinking_budget_force_text: str
    verifier_thinking_budget_tokens: int
    verifier_thinking_budget_force_text: str
    meta_thinking_budget_tokens: int
    meta_thinking_budget_force_text: str
    verify_candidate_limit_while_generating: int
    verify_request_limit_while_generating: int
    verify_n: int
    meta_n: int
    meta_policy: str
    strict_pass_meta: bool
    refine_rounds: int
    refine_review_n: int
    min_valid_low: int
    selector_max_candidate_chars: int
    selection_temperature: float
    selector_mode: str
    selector_min_final_score: float
    decomposition_strategy_enabled: bool
    decomposition_fallback_to_direct: bool
    decomposition_candidate_count: int
    decomposition_max_sublemmas: int
    proof_architect_max_new_tokens: int
    sublemma_max_new_tokens: int
    proof_assembler_max_new_tokens: int
    gold_eval_selection: str


@dataclass
class InputRecord:
    id: Any
    question: str
    source_file: str
    source_path: Path
    source_stem: str
    row_index: int
    question_column: str
    gold_proof: Optional[str] = None
    gold_proof_column: Optional[str] = None


def setup_logging(logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


class PipelineProgress:
    _position_lock = threading.Lock()
    _next_position = 0
    _free_positions: list[int] = []

    def __init__(
        self,
        enabled: bool,
        problem_id: Any,
        pipeline_count: int = 0,
    ) -> None:
        self.enabled = enabled
        self.problem_id = problem_id
        self.pipeline_count = max(1, int(pipeline_count or 1))
        self._lock = threading.Lock()
        self._streams: dict[str, dict[str, Any]] = {}
        self._bars: dict[str, tqdm] = {}
        self._bar_state: dict[str, dict[str, Any]] = {}
        self._bar_positions: dict[str, int] = {}

    @classmethod
    def _allocate_positions(cls, width: int) -> int:
        with cls._position_lock:
            if width == 1 and cls._free_positions:
                return cls._free_positions.pop()
            base = cls._next_position
            cls._next_position += width
            return base

    @classmethod
    def _release_positions(cls, base: int, width: int = 1) -> None:
        with cls._position_lock:
            for position in range(base, base + width):
                cls._free_positions.append(position)

    @staticmethod
    def _candidate_index(detail: str) -> Optional[int]:
        match = re.search(r"\bcandidate=(\d+)\b", detail)
        return int(match.group(1)) if match else None

    @staticmethod
    def _detail_value(detail: str, key: str) -> Optional[str]:
        match = re.search(rf"\b{re.escape(key)}=([A-Za-z0-9_.-]+)\b", detail or "")
        return match.group(1) if match else None

    def _stream_key_and_desc(self, stage: str, detail: str) -> tuple[str, str]:
        candidate = self._candidate_index(detail)
        candidate_label = f"P{candidate}" if candidate is not None else "P0"
        key_parts = [candidate_label, stage]
        desc_parts = [str(self.problem_id), candidate_label, stage]
        for detail_key, label in (
            ("round", "r"),
            ("verifier", "v"),
            ("meta", "m"),
            ("lemma", "lemma"),
        ):
            value = self._detail_value(detail, detail_key)
            if value is None:
                continue
            key_parts.append(f"{label}{value}")
            desc_parts.append(f"{label}{value}")
        key = "|".join(key_parts)
        desc = " ".join(desc_parts)
        return key, desc

    def _bar_for_stream(
        self,
        bar_key: str,
        desc: str,
        stage: str,
        max_tokens: int,
    ) -> tqdm:
        state = self._bar_state.get(bar_key)
        bar = self._bars.get(bar_key)
        if bar is None or state is None or (state["stage"] != stage and int(state["active"]) == 0):
            if bar is not None:
                bar.close()
                old_position = self._bar_positions.pop(bar_key, None)
                if old_position is not None:
                    self._release_positions(old_position)
            position = self._allocate_positions(1)
            bar = tqdm(
                total=max_tokens,
                desc=desc,
                unit="tok",
                position=position,
                leave=True,
                miniters=1,
                mininterval=0.5,
                disable=not self.enabled,
            )
            self._bars[bar_key] = bar
            self._bar_positions[bar_key] = position
            self._bar_state[bar_key] = {
                "stage": stage,
                "active": 0,
            }
        else:
            bar.total = int(bar.total or 0) + max_tokens
            bar.refresh()
        return bar

    def complete(
        self,
        stage: str,
        success: bool,
        detail: str = "",
        latency_s: Optional[float] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        parts = [detail, f"ok={success}"]
        if latency_s is not None:
            parts.append(f"latency={latency_s:.1f}s")
        if completion_tokens is not None:
            parts.append(f"tokens={completion_tokens}")
        summary = " ".join(part for part in parts if part)
        logging.debug(
            "[problem=%s] stage=%s status=complete %s",
            self.problem_id,
            stage,
            summary,
        )

    def stream_start(
        self,
        stream_id: str,
        stage: str,
        detail: str,
        max_tokens: int,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            bar_key, desc = self._stream_key_and_desc(stage, detail)
            bar = self._bar_for_stream(bar_key, desc, stage, max_tokens)
            self._bar_state[bar_key]["active"] += 1
            self._streams[stream_id] = {
                "bar_key": bar_key,
                "bar": bar,
                "max_tokens": max_tokens,
                "tokens": 0,
            }

    def stream_advance(self, stream_id: str, new_tokens: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return
            new_tokens = max(0, int(new_tokens))
            stream["tokens"] += new_tokens
            stream["bar"].update(new_tokens)

    def stream_finish(self, stream_id: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                return
            bar_key = stream["bar_key"]
            if bar_key is not None:
                state = self._bar_state[bar_key]
                bar = stream["bar"]
                bar.total = max(int(stream["max_tokens"]), int(bar.total or 0))
                bar.refresh()
                state["active"] = max(0, int(state["active"]) - 1)
                if int(state["active"]) == 0:
                    bar.close()
                    self._bars.pop(bar_key, None)
                    self._bar_state.pop(bar_key, None)
                    position = self._bar_positions.pop(bar_key, None)
                    if position is not None:
                        self._release_positions(position)
            del self._streams[stream_id]

    def log(self, message: str, *args: Any) -> None:
        if self.enabled:
            logging.debug("[problem=%s] " + message, self.problem_id, *args)

    def close(self) -> None:
        with self._lock:
            for bar in self._bars.values():
                bar.close()
            self._streams.clear()
            self._bars.clear()
            for position in self._bar_positions.values():
                self._release_positions(position)
            self._bar_positions.clear()
            self._bar_state.clear()


def build_deepseek_proof_generation_prompt(
    question: str,
    use_tool: bool = False,
) -> str:
    tool_note = ""
    if use_tool:
        tool_note = "\nYou may use tools if available, but the final response must be a standalone proof.\n"

    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
```
{tool_note}
In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.
- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation // Your evaluation of your own solution above should start with this exact same markdown title

Here is my evaluation of the solution: // Your analysis should start with this exact same phrase
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above

---

Here is your task input:

## Problem
{question}"""


def build_deepseek_proof_only_generation_prompt(
    question: str,
    use_tool: bool = False,
) -> str:
    tool_note = ""
    if use_tool:
        tool_note = "\nYou may use tools if available, but the final response must be a standalone proof.\n"

    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
```
{tool_note}
For this attempt, focus only on producing the best final proof you can within the token budget. Do not write a self-evaluation section.

Your final response should be in the following format:

## Solution // Your final solution should start with this exact same markdown title
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

---

Here is your task input:

## Problem
{question}"""


def build_proof_architect_prompt(question: str) -> str:
    return f"""You are a proof architect for olympiad-level mathematics.

Do NOT solve the problem. Decompose it into a small plan for parallel proof workers.
Keep the output short and structured. Do not use JSON.

Rules:
- Give the most robust strategy, not every possible strategy.
- Return 4 lemma blocks unless fewer are genuinely enough.
- Prefer independent lemmas so workers can prove them in parallel.
- Make every lemma exact and minimal.
- Do not assume monotonicity, injectivity, periodicity, maximality, independence, or extremality unless one lemma proves it.
- If the problem defines iteration on a finite set, explicitly account for cycles/periods; do not assume forward orbits are monotone without set inclusion.
- No final proof, no long reasoning, no repeated text.

Use exactly this compact format:

## Strategy
One or two concise sentences describing the proof route.

## Risks
- One risky hidden assumption or failure mode.
- Another risky hidden assumption or failure mode.

### Lemma L1: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L2: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L3: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L4: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

## Assembly
1. Use L1-L4 to write the final proof.

Problem:
```txt
{question}
```"""


def build_proof_architect_repair_prompt(question: str, bad_architect_output: str) -> str:
    clipped_output, _ = clip_middle_text(str(bad_architect_output or ""), 12_000)
    return f"""Convert the malformed proof-architect draft below into a compact lemma plan.

Rules:
- Do not solve the problem.
- Do not use JSON.
- Use the exact block format below.
- Include exactly 4 lemma blocks unless fewer are genuinely sufficient.
- Prefer independent lemmas that can be proved in parallel.
- If the draft is repetitive or unusable, ignore the repetition and create a fresh conservative decomposition from the problem.

Format:
## Strategy
One or two concise sentences.

## Risks
- risk
- risk

### Lemma L1: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L2: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L3: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

### Lemma L4: short name
Statement: exact statement to prove.
Role: how this lemma is used in the final solution.
Depends: none
Tests: small examples or edge cases to check.

## Assembly
1. Use the lemmas to write the final proof.

Problem:
```txt
{question}
```

Malformed draft:
```txt
{clipped_output}
```"""


def build_sublemma_prover_prompt(
    question: str,
    sublemma_task: dict[str, Any],
    allowed_previous_lemmas: str,
) -> str:
    return f"""You are a rigorous olympiad proof assistant.

You will receive:

1. the original problem,
2. one sub-lemma task,
3. optionally, previously proved lemmas that this task is allowed to use.

Your job is to prove the assigned sub-lemma completely and rigorously.

You must not solve the whole original problem unless it is necessary for the sub-lemma. Focus only on the assigned lemma.

Evaluation standard:

```txt
- Score 1: completely correct proof, all details justified.
- Score 0.5: generally correct but minor omissions.
- Score 0: fatal error, false claim, or severe gap.
```

Rules:

* Prove exactly the lemma statement given.
* Use only the original problem assumptions and the allowed previous lemmas.
* Do not rely on unproved named theorems unless you prove the exact version needed.
* Check edge cases and small examples when relevant.
* If the lemma is false or too strong, do not fake a proof. Instead return a counterexample or explain the gap and propose the weakest corrected lemma that seems true.
* Avoid vague phrases such as "clearly," "obviously," or "it is easy to see" unless the step is genuinely immediate.
* Be especially careful with monotonicity versus cardinality, existence versus construction, pairwise reachability versus global ordering, finite periodic behavior, hidden divisibility assumptions, and equality cases in inequalities.

Return your answer in this exact format:

```md
## Status
PROVED / DISPROVED / NEEDS_WEAKER_STATEMENT

## Lemma
Restate the exact lemma.

## Proof or Counterexample
Give the complete proof. If false, give a counterexample.

## Notes for Final Assembler
State precisely what the final proof is allowed to use from this result.
If the lemma was weakened, state the corrected usable version.
```

Original problem:

```txt
{question}
```

Assigned sub-lemma task:

```json
{json.dumps(sublemma_task, ensure_ascii=False, indent=2)}
```

Allowed previous lemmas, if any:

```txt
{allowed_previous_lemmas or "None."}
```"""


def build_final_assembly_prompt(
    question: str,
    analyst_json: dict[str, Any],
    proved_sublemmas: str,
) -> str:
    return f"""You are the final proof writer.

Your task is to write the final solution to the original problem using the provided proved sub-lemmas.

The final solution will be evaluated by this standard:

```txt
- Score 1: completely correct, all steps properly proved and clearly demonstrated.
- Score 0.5: generally correct but with minor omissions or minor errors.
- Score 0: fatal errors, severe omissions, or not actually solving the problem.

Referencing known results is not enough unless the solution also proves them or the proved sub-lemmas already prove them.
```

You must write a complete, polished proof.

Rules:

* Use only sub-lemmas whose status is PROVED, unless a weakened corrected lemma is explicitly proved and sufficient.
* Do not use any lemma marked DISPROVED or NEEDS_WEAKER_STATEMENT unless the corrected weaker version is explicitly proved.
* If a necessary lemma is missing, prove the missing step yourself in the final solution, but only if it is short and rigorous.
* Do not introduce a new major claim without proof.
* Do not include a self-evaluation section.
* Do not mention the multi-agent process.
* The final response must start with exactly:

```md
## Solution
```

Before writing the final proof, internally verify:

1. Every selected object is well-defined.
2. Every existence claim is justified.
3. Every inequality has the correct direction.
4. Every induction or extremal argument has a valid base case and transition.
5. Every finite process terminates.
6. Every use of pairwise conditions is converted correctly into a global structure.
7. No false monotonicity or injectivity assumption is used.
8. The final conclusion exactly matches the problem statement.

Original problem:

```txt
{question}
```

Proof plan from analyst:

```json
{json.dumps(analyst_json, ensure_ascii=False, indent=2)}
```

Proved sub-lemmas:

```txt
{proved_sublemmas}
```

Now write the final solution."""


def build_deepseek_proof_verification_prompt(question: str, proof: str) -> str:
    return f"""## Instruction

Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the above criteria

---

Here is your task input:

## Problem
{question}

## Solution
{proof}"""


def build_deepseek_gold_proof_evaluation_prompt(question: str, proof: str, gold_proof: str) -> str:
    return f"""## Instruction

Your task is to evaluate the quality of a candidate solution to a problem. You are also given a reference/gold proof. Use the reference proof as ground truth for the intended theorem, key ideas, and missing obligations, but do not require the candidate solution to match the reference proof word-for-word.

Please evaluate the candidate solution according to this rubric:
- If the candidate solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1.
- If the candidate solution is generally correct, but has minor gaps, unclear justifications, or small repairable omissions, then the score is 0.5.
- If the candidate solution does not solve the problem, contains fatal errors, proves a different claim, or omits a central argument from the gold proof without replacement, then the score is 0.

When comparing against the gold proof:
- Check whether the candidate establishes every essential claim needed for the problem.
- Accept a different method if it is rigorous and complete.
- Penalize unsupported claims, missing boundary cases, non-reversible transformations, or unjustified appeals to known results.
- If the gold proof itself appears incomplete, still use it as context but judge the candidate by mathematical correctness.

Your response should be in the following format:

Here is my evaluation of the candidate solution against the reference proof:
... your detailed evaluation here.

Based on my evaluation, the final overall score should be:
\\boxed{{...}}

where ... must be exactly one of 0, 0.5, or 1.

---

Here is your task input:

## Problem
{question}

## Candidate Solution
{proof}

## Reference / Gold Proof
{gold_proof}"""


def build_deepseek_meta_verification_prompt(question: str, proof: str, proof_analysis: str) -> str:
    proof_analysis, _ = clip_middle_text(
        proof_analysis,
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    return f"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
{EVALUATION_RUBRIC}
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

Step Restatement: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

Defect Analysis: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

Expression Analysis: Whether the "solution evaluation"'s expressions are accurate.

Score Analysis: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.

- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.
**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:

- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**.

In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:

- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with \(0\)
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be \(0.5\)

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:

- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with \(0.5\); if no, your rating should be \(1\)

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
... // Your analysis here.

Based on my analysis, I rate the "solution evaluation" as:
\\boxed{{...}} // where ... should be a numerical rating of the "solution evaluation" (0, 0.5, or 1, and nothing else) based on the criteria above.

---

Here is your task input:

## Problem
{question}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""


def build_deepseek_proof_refinement_prompt(
    question: str,
    proof: str,
    proof_analyses: list[str],
) -> str:
    analyses = "\n\n".join(f"### Evaluation {idx + 1}\n{analysis}" for idx, analysis in enumerate(proof_analyses))
    generation_prompt = build_deepseek_proof_generation_prompt(question)
    return f"""{generation_prompt}

## Candidate Solution(s) to Refine
Here is a solution sample along with correctness evaluation(s). Provide a better solution by solving issues mentioned in the evaluations, reusing promising ideas from the solution, or both.

### Candidate Solution
{proof}

{analyses}

## Final Instruction
Your final response must follow the format above, including a `## Solution` section followed by a `## Self Evaluation` section."""


def format_selector_verifier_summaries(candidate: dict[str, Any]) -> str:
    summaries = candidate.get("verifier_score_summaries") or []
    if not summaries:
        return "Verifier/meta score summaries: none"
    lines = ["Verifier/meta score summaries:"]
    for summary in summaries:
        meta_scores = summary.get("meta_scores") or []
        meta_scores_text = ", ".join(f"{score:.3g}" for score in meta_scores) or "none"
        evaluation = str(summary.get("evaluation") or "").strip()
        evaluation, _ = clip_middle_text(evaluation, MAX_SELECTOR_EVALUATION_EXCERPT_CHARS)
        lines.append(
            "- verifier {verifier_index}: verifier_score={verifier_score:.3g}, "
            "meta_factor={meta_factor:.3g}, weighted_score={weighted_score:.3g}, "
            "meta_source={meta_source}, meta_scores=[{meta_scores}]".format(
                verifier_index=summary.get("verifier_index"),
                verifier_score=float(summary.get("verifier_score", 0.0)),
                meta_factor=float(summary.get("meta_factor", 0.0)),
                weighted_score=float(summary.get("weighted_score", 0.0)),
                meta_source=summary.get("meta_source"),
                meta_scores=meta_scores_text,
            )
        )
        if evaluation:
            lines.append(f"  evaluation_excerpt: {evaluation}")
    return "\n".join(lines)


def format_selected_verifier_scores(candidate: Optional[dict[str, Any]]) -> str:
    if not candidate:
        return "Verifier scores: none"
    summaries = candidate.get("verifier_score_summaries") or []
    if not summaries:
        return "Verifier scores: none"
    lines = ["Verifier scores:"]
    for summary in summaries:
        lines.append(
            "- verifier {verifier_index}: score={verifier_score:.3g}, "
            "meta_factor={meta_factor:.3g}, weighted={weighted_score:.3g}".format(
                verifier_index=summary.get("verifier_index"),
                verifier_score=float(summary.get("verifier_score", 0.0)),
                meta_factor=float(summary.get("meta_factor", 0.0)),
                weighted_score=float(summary.get("weighted_score", 0.0)),
            )
        )
    return "\n".join(lines)


def print_selected_solution_summary(
    *,
    problem_id: Any,
    selected_idx: Any,
    proof: Any,
    final_score: Any,
    final_status: Any,
    candidate: Optional[dict[str, Any]],
) -> None:
    raw_proof = str(proof or "").strip()
    submitted_proof = format_submission_answer(raw_proof)
    summary = (
        "\n"
        f"===== Selected Solution Summary id={problem_id} candidate={selected_idx} =====\n"
        f"final_score: {final_score}\n"
        f"final_status: {final_status}\n"
        f"generation_mode: {candidate.get('generation_mode') if candidate else None}\n"
        f"self_score: {candidate.get('self_score') if candidate else None}\n"
        f"raw_proof_chars: {len(raw_proof)}\n"
        f"submitted_proof_chars: {len(submitted_proof)}\n"
        f"{format_selected_verifier_scores(candidate)}\n"
        "----- Submitted proof text -----\n"
        f"{submitted_proof}\n"
        "===== End Selected Solution Summary ====="
    )
    print(summary, flush=True)


def build_selection_prompt(
    question: str,
    candidates: list[dict[str, Any]],
    max_candidate_chars: int,
    use_gold_scores: bool,
) -> str:
    candidate_blocks = []
    for idx, candidate in enumerate(candidates):
        proof = str(candidate.get("proof_solution") or "")
        if len(proof) > max_candidate_chars:
            proof = proof[:max_candidate_chars] + "\n...[truncated for selection prompt]"
        gold_lines = ""
        if use_gold_scores:
            gold_lines = (
                f"gold_eval_score: {candidate.get('gold_eval_score')}\n"
                f"gold_eval_status: {candidate.get('gold_eval_status')}\n"
            )
        candidate_blocks.append(
            f"""### Candidate {idx}
final_score: {candidate.get("final_score")}
final_status: {candidate.get("final_status")}
{gold_lines}{format_selector_verifier_summaries(candidate)}

Proof:
{proof}"""
        )
    joined = "\n\n".join(candidate_blocks)
    gold_instruction = ""
    if use_gold_scores:
        gold_instruction = "When gold_eval_score is present, prefer the highest gold-evaluated complete proof."
    return f"""You are choosing the best proof candidate for a math problem.

Do not synthesize a new proof. Choose exactly one candidate index. Prefer a complete, rigorous proof over a shorter proof. Use verifier status and critiques as advisory signals, but inspect the proof yourself.
{gold_instruction}

Output exactly:
SELECTED_INDEX: <integer>
REASON: <brief reason>

## Problem
{question}

## Candidates
{joined}"""


def extract_boxed_score(text: str) -> Optional[float]:
    if not text:
        return None
    score_text = text[-2000:]
    matches = list(_SCORE_PATTERN.finditer(score_text))
    if matches:
        raw = next(group for group in matches[-1].groups() if group is not None)
    else:
        fallback_matches = list(_FALLBACK_SCORE_PATTERN.finditer(score_text.strip()))
        if not fallback_matches:
            return None
        raw = fallback_matches[-1].group(1)
    value = float(raw)
    if value in {0.0, 0.5, 1.0}:
        return value
    return None


def strip_reasoning_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK_PATTERN.sub("", text or "")
    orphan_close = re.search(r"(?is)</think>\s*", cleaned)
    if orphan_close is not None:
        first_marker = min(
            (
                idx
                for marker in _VISIBLE_OUTPUT_MARKERS
                if (idx := cleaned.lower().find(marker.lower())) >= 0
            ),
            default=None,
        )
        if first_marker is None or orphan_close.start() < first_marker:
            prefix = cleaned[: orphan_close.start()]
            suffix = cleaned[orphan_close.end() :]
            if "<think" not in prefix.lower():
                cleaned = suffix
    return cleaned.strip()


def has_closed_thinking_block(text: str) -> bool:
    return re.search(r"(?is)</think>\s*", text or "") is not None


def output_after_last_thinking_block(text: str) -> tuple[str, bool]:
    raw = str(text or "")
    matches = list(re.finditer(r"(?is)</think>\s*", raw))
    if not matches:
        return strip_reasoning_blocks(raw), False
    return raw[matches[-1].end() :].strip(), True


def log_generation_visible_output(
    stage: str,
    text: str,
    *,
    problem_id: Any,
    attempt_idx: int,
    round_idx: int,
) -> None:
    visible, found_end_think = output_after_last_thinking_block(text)
    print(
        f"Visible output after thinking stage={stage} problem={problem_id} candidate={attempt_idx} round={round_idx} "
        f"end_think_found={found_end_think} chars={len(visible)}:\n{visible}",
    )


def log_verifier_tail_output(
    stage: str,
    text: str,
    *,
    attempt_idx: int,
    round_idx: int,
    verifier_index: Optional[int] = None,
    meta_index: Optional[int] = None,
    max_chars: int = 500,
) -> None:
    raw = str(text or "")
    tail = raw[-max_chars:]
    logging.info(
        "Raw output tail stage=%s candidate=%d round=%d verifier=%s meta=%s "
        "raw_chars=%d tail_chars=%d:\n%s",
        stage,
        attempt_idx,
        round_idx,
        verifier_index,
        meta_index,
        len(raw),
        len(tail),
        tail,
    )


def safe_path_component(value: Any, default: str = "unknown") -> str:
    raw = str(value if value is not None else default).strip() or default
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or default


def parse_detail_int(detail: str, key: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(key)}=(\d+)\b", detail or "")
    return int(match.group(1)) if match else None


def stage_file_alias(stage: str) -> str:
    return {
        "proof_generation": "proof_gen",
        "proof_architect": "architect",
        "proof_architect_repair": "architect_repair",
        "sublemma_proof": "sublemma",
        "proof_assembly": "assembly",
        "proof_refine": "proof_refine",
        "proof_verify": "verify",
        "proof_meta_verify": "meta",
        "proof_gold_eval": "gold_eval",
    }.get(stage, safe_path_component(stage, "llm"))


def llm_call_filename(stage: str, detail: str) -> str:
    candidate = parse_detail_int(detail, "candidate")
    round_idx = parse_detail_int(detail, "round")
    verifier_idx = parse_detail_int(detail, "verifier")
    meta_idx = parse_detail_int(detail, "meta")
    lemma_match = re.search(r"\blemma=([A-Za-z0-9_.-]+)\b", detail or "")
    lemma_id = lemma_match.group(1) if lemma_match else None
    alias = stage_file_alias(stage)
    parts: list[str] = []
    if candidate is not None:
        parts.append(f"cand_{candidate}")
        parts.append(alias)
    else:
        parts.append(alias)
    if round_idx is not None:
        parts.append(f"r{round_idx}")
    if verifier_idx is not None:
        parts.append(f"v{verifier_idx}")
    if meta_idx is not None:
        parts.append(f"m{meta_idx}")
    if lemma_id is not None:
        parts.append(safe_path_component(lemma_id, "lemma"))
    if len(parts) == 1 and detail:
        detail_hash = hashlib.sha256(detail.encode("utf-8")).hexdigest()[:8]
        parts.append(detail_hash)
    return "_".join(parts) + ".txt"


def extract_after_marker(text: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        match = re.search(re.escape(marker), text, flags=re.IGNORECASE)
        if match:
            return text[match.end() :].strip()
    return ""


def extract_after_last_marker(text: str, markers: tuple[str, ...]) -> tuple[str, bool]:
    best_match: Optional[re.Match[str]] = None
    for marker in markers:
        for match in re.finditer(re.escape(marker), text, flags=re.IGNORECASE):
            if best_match is None or match.start() > best_match.start():
                best_match = match
    if best_match is None:
        return "", False
    return text[best_match.end() :].strip(), True


def extract_after_last_section_marker(
    text: str,
    markers: tuple[str, ...],
    patterns: tuple[re.Pattern[str], ...],
) -> tuple[str, bool, str]:
    best_match: Optional[re.Match[str]] = None
    marker_kind = ""
    for marker in markers:
        for match in re.finditer(re.escape(marker), text, flags=re.IGNORECASE):
            if best_match is None or match.start() > best_match.start():
                best_match = match
                marker_kind = marker
    for pattern in patterns:
        for match in pattern.finditer(text):
            if best_match is None or match.start() > best_match.start():
                best_match = match
                marker_kind = pattern.pattern
    if best_match is None:
        return "", False, ""
    return text[best_match.end() :].strip(), True, marker_kind


def clip_middle_text(text: str, max_chars: int) -> tuple[str, bool]:
    cleaned = str(text or "").strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned, False
    marker = _CLIPPED_TEXT_MARKER
    if max_chars <= len(marker) + 20:
        return cleaned[-max_chars:], True
    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return f"{cleaned[:head_chars]}{marker}{cleaned[-tail_chars:]}", True


def merge_streamed_token_ids(existing: list[int], incoming: list[int]) -> tuple[list[int], int]:
    if len(incoming) >= len(existing) and incoming[: len(existing)] == existing:
        return list(incoming), len(incoming) - len(existing)
    merged = list(existing)
    merged.extend(incoming)
    return merged, len(incoming)


def detect_duplicate_segment_in_recent_tokens(
    tokenizer: Any,
    token_ids: list[int],
    *,
    recent_token_count: int = REPETITION_GUARD_RECENT_TOKENS,
    duplicate_threshold: int = REPETITION_GUARD_DUPLICATE_LINE_THRESHOLD,
) -> Optional[tuple[str, int, str]]:
    if tokenizer is None or not token_ids:
        return None
    recent_ids = token_ids[-max(1, int(recent_token_count)) :]
    try:
        text = tokenizer.decode(recent_ids, skip_special_tokens=False)
    except Exception:
        logging.debug("Repetition guard decode failed", exc_info=True)
        return None

    def check_segments(segments: list[str], *, min_chars: int, kind: str) -> Optional[tuple[str, int, str]]:
        counts: dict[str, tuple[str, int]] = {}
        for raw_segment in segments:
            segment = raw_segment.strip()
            if len(segment) < min_chars:
                continue
            normalized = re.sub(r"\s+", " ", segment).strip().lower()
            if not normalized:
                continue
            original, count = counts.get(normalized, (segment, 0))
            count += 1
            if count > duplicate_threshold:
                return original, count, kind
            counts[normalized] = (original, count)
        return None

    line_repetition = check_segments(str(text or "").split("\n"), min_chars=8, kind="line")
    if line_repetition is not None:
        return line_repetition

    sentence_repetition = check_segments(
        re.split(r"(?<=[.!?])\s+", str(text or "")),
        min_chars=24,
        kind="sentence",
    )
    if sentence_repetition is not None:
        return sentence_repetition

    return None


def trim_score_trailer(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub("", text or "").strip()


def _header_matches(text: str, header: str) -> list[re.Match[str]]:
    header_pattern = re.escape(header.strip()).replace(r"\ ", r"[ \t]+")
    return list(re.finditer(rf"(?im)^[ \t]*{header_pattern}{_HEADER_SUFFIX_PATTERN}", text))


def _extract_generation_sections(text: str) -> tuple[str, str, bool, bool]:
    solution_headers = _header_matches(text, "## Solution")
    evaluation_headers = _header_matches(text, "## Self Evaluation")
    if not solution_headers:
        return "", "", False, False

    solution_header = solution_headers[-1]
    following_evaluation = next(
        (match for match in evaluation_headers if match.start() > solution_header.end()),
        None,
    )
    if following_evaluation is None:
        return text[solution_header.end() :].strip(), "", True, False

    proof = text[solution_header.end() : following_evaluation.start()].strip()
    self_evaluation = text[following_evaluation.end() :].strip()
    return proof, self_evaluation, True, True


def parse_generation_response(
    text: str,
    *,
    require_self_evaluation: bool = True,
) -> dict[str, Any]:
    text = strip_reasoning_blocks(text)
    proof, self_evaluation, has_solution_section, has_self_evaluation_section = (
        _extract_generation_sections(text)
    )
    self_score = extract_boxed_score(self_evaluation)
    is_valid = bool(has_solution_section and proof)
    if require_self_evaluation:
        is_valid = bool(is_valid and has_self_evaluation_section)
    return {
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": has_solution_section,
        "has_self_evaluation_section": has_self_evaluation_section,
        "requires_self_evaluation": require_self_evaluation,
        "is_valid_candidate_response": is_valid,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    raw = strip_reasoning_blocks(text)
    fence_match = re.search(r"(?is)```(?:json)?\s*(\{.*\})\s*```", raw)
    if fence_match:
        raw = fence_match.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in architect response")
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import json_repair  # type: ignore[import-not-found]

            parsed = json_repair.loads(raw)
        except Exception:
            parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Architect response parsed to {type(parsed).__name__}, not object")
    return parsed


def split_architect_list_value(value: str) -> list[str]:
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "no", "nothing"}:
        return []
    lines = [
        line.strip(" \t-*")
        for line in cleaned.splitlines()
        if line.strip(" \t-*")
    ]
    if len(lines) > 1:
        return lines
    return [
        part.strip()
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", cleaned)
        if part.strip() and part.strip().lower() not in {"none", "n/a"}
    ]


def parse_architect_block_fields(block: str) -> dict[str, str]:
    field_names = {
        "statement",
        "role",
        "depends",
        "dependencies",
        "tests",
        "counterexample tests",
        "assumptions",
        "methods",
        "forbidden",
    }
    fields: dict[str, list[str]] = {}
    current: Optional[str] = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        match = re.match(r"^\s*([A-Za-z][A-Za-z ]{0,40})\s*:\s*(.*)$", line)
        if match and match.group(1).strip().lower() in field_names:
            current = match.group(1).strip().lower()
            fields.setdefault(current, []).append(match.group(2).strip())
            continue
        if current and line.strip() and not re.match(r"^\s*#{1,6}\s+", line):
            fields[current].append(line.strip())
    return {
        key: "\n".join(part for part in values if part).strip()
        for key, values in fields.items()
    }


def parse_architect_block_output(text: str, max_sublemmas: int) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    lemma_matches = list(
        re.finditer(
            r"(?im)^\s*#{2,4}\s*Lemma\s+([A-Za-z0-9_.-]+)\s*:?\s*(.*?)\s*$",
            visible,
        )
    )
    if not lemma_matches:
        raise ValueError("No lemma blocks found in architect output")

    strategy_match = re.search(
        r"(?ims)^\s*#{1,3}\s*Strategy\s*$\s*(.*?)(?=^\s*#{1,4}\s*(?:Risks|Lemma|Assembly)\b|\Z)",
        visible,
        flags=0,
    )
    strategy_text = strategy_match.group(1).strip() if strategy_match else ""
    risks_match = re.search(
        r"(?ims)^\s*#{1,3}\s*Risks\s*$\s*(.*?)(?=^\s*#{1,4}\s*(?:Lemma|Assembly)\b|\Z)",
        visible,
        flags=0,
    )
    risk_lines = []
    if risks_match:
        risk_lines = [
            line.strip(" \t-*")
            for line in risks_match.group(1).splitlines()
            if line.strip(" \t-*")
        ]
    assembly_match = re.search(
        r"(?ims)^\s*#{1,3}\s*Assembly\s*$\s*(.*)$",
        visible,
        flags=0,
    )
    assembly_text = assembly_match.group(1).strip() if assembly_match else ""

    tasks: list[dict[str, Any]] = []
    dependency_graph: list[dict[str, Any]] = []
    for idx, match in enumerate(lemma_matches[: max(0, max_sublemmas)]):
        lemma_id = normalize_lemma_id(match.group(1), f"L{idx + 1}")
        short_name = (match.group(2) or "").strip() or lemma_id
        start = match.end()
        end = lemma_matches[idx + 1].start() if idx + 1 < len(lemma_matches) else len(visible)
        block = visible[start:end]
        assembly_header = re.search(r"(?im)^\s*#{1,3}\s*Assembly\s*$", block)
        if assembly_header:
            block = block[: assembly_header.start()]
        fields = parse_architect_block_fields(block)
        statement = fields.get("statement", "").strip()
        if not statement:
            fallback_statement, _ = clip_middle_text(block.strip(), 1200)
            statement = fallback_statement
        if not statement:
            continue
        depends = split_architect_list_value(
            fields.get("depends") or fields.get("dependencies") or ""
        )
        tests = split_architect_list_value(
            fields.get("tests") or fields.get("counterexample tests") or ""
        )
        tasks.append(
            {
                "lemma_id": lemma_id,
                "short_name": short_name,
                "statement": statement,
                "role_in_solution": fields.get("role", "").strip(),
                "assumptions": split_architect_list_value(fields.get("assumptions", "")),
                "allowed_previous_lemmas": [
                    normalize_lemma_id(dep, "")
                    for dep in depends
                    if normalize_lemma_id(dep, "")
                ],
                "suggested_proof_methods": split_architect_list_value(fields.get("methods", "")),
                "forbidden_shortcuts": split_architect_list_value(fields.get("forbidden", "")),
                "counterexample_tests": tests,
                "expected_output_contract": (
                    "The prover must either prove the lemma completely or report that it is "
                    "false/too strong and suggest a corrected version."
                ),
            }
        )
        dependency_graph.append(
            {
                "lemma_id": lemma_id,
                "depends_on": depends,
                "used_by": [],
                "short_name": short_name,
            }
        )
    if not tasks:
        raise ValueError("No usable lemma statements found in architect output")
    return {
        "problem_summary": "",
        "definitions_and_notation": [],
        "chosen_strategy": {
            "high_level_idea": strategy_text,
            "why_this_strategy_should_work": "",
            "main_bottleneck": "",
        },
        "rejected_or_risky_strategies": [],
        "dependency_graph": dependency_graph,
        "sublemma_tasks": tasks,
        "final_assembly_plan": [
            {
                "step": 1,
                "uses": [task["lemma_id"] for task in tasks],
                "description": assembly_text or "Use the proved lemmas to assemble the final proof.",
            }
        ],
        "risk_register": [
            {
                "risk": risk,
                "why_it_is_dangerous": "",
                "how_the_sublemmas_address_it": "",
            }
            for risk in risk_lines
        ],
    }


def parse_architect_output(text: str, max_sublemmas: int) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except Exception:
        return parse_architect_block_output(text, max_sublemmas)


def parse_sublemma_status(text: str) -> str:
    visible = strip_reasoning_blocks(text)
    status_match = re.search(
        r"(?is)##\s*Status\s*\n\s*(PROVED|DISPROVED|NEEDS_WEAKER_STATEMENT)\b",
        visible,
    )
    if status_match:
        return status_match.group(1).upper()
    fallback_match = re.search(r"\b(PROVED|DISPROVED|NEEDS_WEAKER_STATEMENT)\b", visible)
    if fallback_match:
        return fallback_match.group(1).upper()
    return "UNKNOWN"


def parse_assembled_solution(text: str) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    headers = _header_matches(visible, "## Solution")
    if not headers:
        proof = visible.strip()
        has_solution_section = False
    else:
        proof = visible[headers[-1].start() :].strip()
        has_solution_section = True
    return {
        "proof": proof,
        "self_evaluation": "",
        "self_score": None,
        "has_solution_section": has_solution_section,
        "has_self_evaluation_section": False,
        "requires_self_evaluation": False,
        "is_valid_candidate_response": bool(proof),
    }


def normalize_lemma_id(value: Any, fallback: str) -> str:
    raw = str(value or fallback).strip() or fallback
    return safe_path_component(raw, fallback)


def analyst_sublemma_tasks(analyst_json: dict[str, Any], max_sublemmas: int) -> list[dict[str, Any]]:
    tasks = analyst_json.get("sublemma_tasks")
    if not isinstance(tasks, list):
        return []
    normalized = []
    seen: set[str] = set()
    for idx, task in enumerate(tasks[: max(0, max_sublemmas)]):
        if not isinstance(task, dict):
            continue
        copied = dict(task)
        lemma_id = normalize_lemma_id(copied.get("lemma_id"), f"L{idx + 1}")
        if lemma_id in seen:
            lemma_id = f"{lemma_id}_{idx + 1}"
        copied["lemma_id"] = lemma_id
        seen.add(lemma_id)
        allowed = copied.get("allowed_previous_lemmas")
        if not isinstance(allowed, list):
            allowed = copied.get("depends_on", [])
        copied["allowed_previous_lemmas"] = [
            normalize_lemma_id(item, "")
            for item in (allowed if isinstance(allowed, list) else [])
            if str(item or "").strip()
        ]
        normalized.append(copied)
    return normalized


def order_sublemma_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = {task["lemma_id"]: task for task in tasks}
    ordered: list[dict[str, Any]] = []
    proved_or_scheduled: set[str] = set()
    while remaining:
        ready_ids = [
            lemma_id
            for lemma_id, task in remaining.items()
            if all(dep in proved_or_scheduled or dep not in remaining for dep in task.get("allowed_previous_lemmas", []))
        ]
        if not ready_ids:
            ordered.extend(remaining.pop(lemma_id) for lemma_id in sorted(remaining))
            break
        for lemma_id in sorted(ready_ids):
            task = remaining.pop(lemma_id)
            ordered.append(task)
            proved_or_scheduled.add(lemma_id)
    return ordered


def sublemma_task_waves(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    remaining = {task["lemma_id"]: task for task in tasks}
    completed: set[str] = set()
    waves: list[list[dict[str, Any]]] = []
    while remaining:
        ready_ids = [
            lemma_id
            for lemma_id, task in remaining.items()
            if all(dep in completed or dep not in remaining for dep in task.get("allowed_previous_lemmas", []))
        ]
        if not ready_ids:
            waves.append([remaining.pop(lemma_id) for lemma_id in sorted(remaining)])
            break
        wave = [remaining.pop(lemma_id) for lemma_id in sorted(ready_ids)]
        waves.append(wave)
        completed.update(task["lemma_id"] for task in wave)
    return waves


def format_allowed_previous_lemmas(
    allowed_ids: list[str],
    lemma_results: dict[str, dict[str, Any]],
) -> str:
    blocks = []
    for lemma_id in allowed_ids:
        result = lemma_results.get(lemma_id)
        if not result:
            continue
        lemma_text, _ = clip_middle_text(
            str(result.get("text", "")).strip(),
            MAX_FORWARDED_SUBLEMMA_CHARS,
        )
        blocks.append(
            f"### {lemma_id}\nStatus: {result.get('status')}\n\n{lemma_text}"
        )
    return "\n\n".join(blocks)


def format_proved_sublemmas(lemma_results: dict[str, dict[str, Any]]) -> str:
    blocks = []
    for lemma_id, result in lemma_results.items():
        status = result.get("status")
        if status not in {"PROVED", "NEEDS_WEAKER_STATEMENT"}:
            continue
        lemma_text, _ = clip_middle_text(
            str(result.get("text", "")).strip(),
            MAX_FORWARDED_SUBLEMMA_CHARS,
        )
        blocks.append(f"### {lemma_id}\nStatus: {status}\n\n{lemma_text}")
    return "\n\n".join(blocks)


def require_valid_candidate_response(
    parsed: dict[str, Any],
    *,
    problem_id: Any,
    attempt_idx: int,
    stage: str,
    round_idx: int,
) -> bool:
    if parsed.get("is_valid_candidate_response"):
        return True
    logging.warning(
        "Invalid candidate output problem=%s candidate=%d stage=%s round=%d "
        "has_solution=%s proof_chars=%d requires_self_evaluation=%s "
        "has_self_evaluation=%s self_evaluation_chars=%d self_score=%s; "
        "skipping this output without raising",
        problem_id,
        attempt_idx,
        stage,
        round_idx,
        parsed.get("has_solution_section"),
        len(str(parsed.get("proof") or "")),
        parsed.get("requires_self_evaluation"),
        parsed.get("has_self_evaluation_section"),
        len(str(parsed.get("self_evaluation") or "")),
        parsed.get("self_score"),
    )
    return False


def parse_verifier_response(text: str) -> dict[str, Any]:
    visible_text = strip_reasoning_blocks(text)
    evaluation, marker_found, marker_kind = extract_after_last_section_marker(
        visible_text,
        _VERIFIER_EVALUATION_MARKERS,
        _VERIFIER_EVALUATION_PATTERNS,
    )
    if evaluation:
        evaluation = trim_score_trailer(evaluation, _VERIFIER_SCORE_TRAILER_PATTERN)
    else:
        evaluation = trim_score_trailer(visible_text, _VERIFIER_SCORE_TRAILER_PATTERN)
    if not marker_found and len(evaluation) > MAX_FORWARDED_EVALUATION_CHARS:
        evaluation = evaluation[-MAX_FORWARDED_EVALUATION_CHARS:]
        clipped = True
    else:
        evaluation, clipped = clip_middle_text(
            evaluation,
            MAX_FORWARDED_EVALUATION_CHARS,
        )
    return {
        "evaluation": evaluation.strip(),
        "score": extract_boxed_score(visible_text),
        "evaluation_marker_found": marker_found,
        "evaluation_marker": marker_kind,
        "evaluation_raw_chars": len(visible_text),
        "evaluation_forwarded_chars": len(evaluation),
        "evaluation_clipped": clipped,
    }


def parse_meta_verifier_response(text: str) -> dict[str, Any]:
    visible_text = strip_reasoning_blocks(text)
    analysis, marker_found, marker_kind = extract_after_last_section_marker(
        visible_text,
        _META_ANALYSIS_MARKERS,
        _META_ANALYSIS_PATTERNS,
    )
    if analysis:
        analysis = trim_score_trailer(analysis, _META_SCORE_TRAILER_PATTERN)
    else:
        analysis = trim_score_trailer(visible_text, _META_SCORE_TRAILER_PATTERN)
    if not marker_found and len(analysis) > MAX_FORWARDED_META_ANALYSIS_CHARS:
        analysis = analysis[-MAX_FORWARDED_META_ANALYSIS_CHARS:]
        clipped = True
    else:
        analysis, clipped = clip_middle_text(
            analysis,
            MAX_FORWARDED_META_ANALYSIS_CHARS,
        )
    return {
        "analysis": analysis.strip(),
        "score": extract_boxed_score(visible_text),
        "analysis_marker_found": marker_found,
        "analysis_marker": marker_kind,
        "analysis_raw_chars": len(visible_text),
        "analysis_forwarded_chars": len(analysis),
        "analysis_clipped": clipped,
    }


def parse_selected_index(text: str, candidate_count: int) -> Optional[int]:
    match = _SELECTED_INDEX_PATTERN.search(text or "")
    if not match:
        return None
    selected = int(match.group(1))
    if 0 <= selected < candidate_count:
        return selected
    return None


def summarize_meta_votes(meta_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_votes = sum(1 for result in meta_results if result.get("score") == 1.0)
    half_votes = sum(1 for result in meta_results if result.get("score") == 0.5)
    invalid_votes = sum(1 for result in meta_results if result.get("score") == 0.0)
    parsed_votes = valid_votes + half_votes + invalid_votes
    threshold = (len(meta_results) // 2) + 1 if meta_results else 1
    return {
        "valid_votes": valid_votes,
        "half_votes": half_votes,
        "invalid_votes": invalid_votes,
        "parsed_votes": parsed_votes,
        "threshold": threshold,
        "validated": valid_votes >= threshold,
    }


def compute_meta_summary_by_verifier(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for verifier in verifier_results:
        try:
            verifier_index = int(verifier.get("verifier_index"))
        except (TypeError, ValueError):
            continue
        summaries[verifier_index] = summarize_meta_votes(
            meta_results_by_verifier.get(verifier_index, [])
        )
    return summaries


def compute_strict_pass(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    *,
    require_meta: bool,
) -> dict[str, Any]:
    verifier_count = len(verifier_results)
    parsed_verifier_count = sum(1 for result in verifier_results if result.get("score") is not None)
    pass_count = sum(1 for result in verifier_results if result.get("score") == 1.0)
    all_verifiers_passed = (
        verifier_count > 0
        and parsed_verifier_count == verifier_count
        and pass_count == verifier_count
    )
    meta_summaries = compute_meta_summary_by_verifier(verifier_results, meta_results_by_verifier)
    meta_valid_count = sum(1 for summary in meta_summaries.values() if summary.get("validated"))
    meta_checked_count = sum(
        1 for summary in meta_summaries.values() if summary.get("parsed_votes", 0) > 0
    )
    strict_pass = all_verifiers_passed
    if require_meta:
        strict_pass = (
            all_verifiers_passed
            and verifier_count > 0
            and meta_checked_count == verifier_count
            and meta_valid_count == verifier_count
        )
    return {
        "strict_pass": strict_pass,
        "all_verifiers_passed": all_verifiers_passed,
        "meta_valid_count": meta_valid_count,
        "meta_checked_count": meta_checked_count,
        "meta_summary_by_verifier": meta_summaries,
    }


def coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    return None


def verifier_meta_factor(
    verifier_index: int,
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    meta_n: int,
) -> tuple[float, list[float], str]:
    if meta_n <= 0:
        return 1.0, [], "meta_disabled"
    meta_scores = [
        score
        for score in (
            coerce_score(result.get("score"))
            for result in meta_results_by_verifier.get(verifier_index, [])
        )
        if score is not None
    ]
    if meta_scores:
        return sum(meta_scores) / len(meta_scores), meta_scores, "parsed_meta"
    return 0.6, [], "missing_meta_default"


def build_verifier_score_summaries(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    meta_n: int,
) -> list[dict[str, Any]]:
    summaries = []
    for idx, verifier in enumerate(verifier_results):
        verifier_index = int(verifier.get("verifier_index", idx))
        verifier_score = coerce_score(verifier.get("score"))
        if verifier_score is None:
            continue
        meta_factor, meta_scores, meta_source = verifier_meta_factor(
            verifier_index,
            meta_results_by_verifier,
            meta_n,
        )
        weighted_score = verifier_score * meta_factor
        evaluation, evaluation_clipped = clip_middle_text(
            str(verifier.get("evaluation") or "").strip(),
            MAX_FORWARDED_EVALUATION_CHARS,
        )
        summaries.append(
            {
                "verifier_index": verifier_index,
                "verifier_score": verifier_score,
                "meta_scores": meta_scores,
                "meta_factor": meta_factor,
                "meta_source": meta_source,
                "weighted_score": weighted_score,
                "evaluation": evaluation,
                "evaluation_clipped": evaluation_clipped,
            }
        )
    return summaries


def aggregate_proof_label(
    verifier_results: list[dict[str, Any]],
    meta_results_by_verifier: dict[int, list[dict[str, Any]]],
    min_valid_low: int,
    strict_pass_meta: bool = False,
    meta_n: int = 0,
) -> dict[str, Any]:
    validated_critiques = []
    low_scores_seen = 0
    score_summaries = build_verifier_score_summaries(
        verifier_results,
        meta_results_by_verifier,
        meta_n=meta_n,
    )

    for idx, verifier in enumerate(verifier_results):
        verifier_index = int(verifier.get("verifier_index", idx))
        score = coerce_score(verifier.get("score"))
        if score is None or score >= 1.0:
            continue
        low_scores_seen += 1
        meta_summary = summarize_meta_votes(meta_results_by_verifier.get(verifier_index, []))
        critique = {
            "verifier_index": verifier_index,
            "score": score,
            "evaluation": verifier.get("evaluation", ""),
            "meta_summary": meta_summary,
        }
        if meta_n <= 0:
            critique["meta_summary"] = {
                **meta_summary,
                "validated": True,
                "validation_source": "meta_disabled",
            }
            validated_critiques.append(critique)
        elif meta_summary["validated"]:
            validated_critiques.append(critique)

    strict_pass = compute_strict_pass(
        verifier_results,
        meta_results_by_verifier,
        require_meta=strict_pass_meta,
    )
    weighted_scores = [summary["weighted_score"] for summary in score_summaries]
    final_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else None
    if final_score is None:
        final_status = "needs_review"
    elif len(validated_critiques) >= min_valid_low and final_score <= 0.5:
        final_status = "validated_low_score"
    elif final_score > 0.5:
        if strict_pass["strict_pass"]:
            final_status = "strict_pass" if strict_pass_meta else "all_verifiers_passed"
        elif strict_pass["all_verifiers_passed"] and (not strict_pass_meta or final_score >= 1.0):
            final_status = "all_verifiers_passed"
        else:
            final_status = "weighted_score_pass"
    else:
        final_status = "weighted_score_low"

    return {
        "final_score": final_score,
        "final_status": final_status,
        "validated_critiques": validated_critiques,
        "verifier_score_summaries": score_summaries,
        "low_scores_seen": low_scores_seen,
        **strict_pass,
    }


def response_usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


class VLLMServer:
    def __init__(self, cfg: VLLMConfig, port: int, gpu_group: str, index: int) -> None:
        self.cfg = cfg
        self.port = port
        self.gpu_group = gpu_group
        self.index = index
        self.client_host = "127.0.0.1" if cfg.host == "0.0.0.0" else cfg.host
        self.base_url = f"http://{self.client_host}:{port}/v1"
        self.health_url = f"http://{self.client_host}:{port}/health"
        self._proc: Optional[subprocess.Popen] = None
        self._log_file: Any = None
        self._log_path: Optional[Path] = None
        self._stopped = False

    @property
    def tag(self) -> str:
        return f"vllm[{self.index}] port={self.port} gpus={self.gpu_group}"

    def is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((self.client_host, self.port)) == 0

    def build_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--api-key",
            self.cfg.api_key,
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--max-num-seqs",
            str(self.cfg.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--host",
            self.cfg.host,
            "--port",
            str(self.port),
            "--dtype",
            self.cfg.dtype,
            "--max-model-len",
            str(self.cfg.num_ctx),
            "--stream-interval",
            str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
            "--trust-remote-code",
        ]
        extra = shlex.split(self.cfg.vllm_extra_args or "")
        if extra:
            cmd.extend(extra)
        return cmd

    def start(self) -> None:
        if self.is_port_open():
            logging.info("Port %s is active. Reusing existing vLLM server.", self.port)
            return
        cmd = self.build_command()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu_group}
        log_path = self.cfg.logdir / f"vllm_server_{self.index}.log"
        self._log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        logging.info("Launching %s: %s", self.tag, " ".join(shlex.quote(part) for part in cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: int) -> None:
        start = time.time()
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                code = None if self._proc is None else self._proc.returncode
                log_tail = ""
                if self._log_path is not None and self._log_path.exists():
                    lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    log_tail = "\n".join(lines[-80:])
                detail = (
                    f"\nLast lines from {self._log_path}:\n{log_tail}"
                    if log_tail
                    else f"\nCheck {self._log_path} for the underlying vLLM error."
                )
                raise RuntimeError(
                    f"{self.tag} exited before health check passed with code {code}{detail}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as response:
                    if response.status == 200:
                        logging.info("%s ready after %.1fs", self.tag, time.time() - start)
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() - start > timeout:
                self.stop()
                raise TimeoutError(f"{self.tag} did not become healthy within {timeout}s")
            time.sleep(3)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if self._log_file is not None:
            self._log_file.close()


class ServerManager:
    def __init__(
        self,
        cfg: VLLMConfig,
        gpu_groups: list[str],
        no_serve: bool,
        base_url: str,
        server_timeout: int,
    ) -> None:
        self.cfg = cfg
        self.gpu_groups = gpu_groups
        self.no_serve = no_serve
        self.base_url = base_url
        self.server_timeout = server_timeout
        self.servers: list[VLLMServer] = []
        self.urls: list[str] = []
        self._stopped = False

    def start(self) -> None:
        if self.no_serve:
            if self.base_url.strip():
                self.urls = [url.strip() for url in self.base_url.split(",") if url.strip()]
            else:
                self.urls = [f"http://{self.cfg.host}:{self.cfg.port + i}/v1" for i in range(len(self.gpu_groups))]
            logging.info("Using existing vLLM endpoints: %s", self.urls)
            return

        self.servers = [
            VLLMServer(self.cfg, self.cfg.port + idx, gpu_group, idx)
            for idx, gpu_group in enumerate(self.gpu_groups)
        ]
        atexit.register(self.stop)
        previous_handlers = {}

        if threading.current_thread() is threading.main_thread():
            previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

            def handle_signal(signum: int, frame: Any) -> None:
                self.stop()
                previous = previous_handlers.get(signum)
                if callable(previous):
                    previous(signum, frame)
                raise KeyboardInterrupt

            for sig in previous_handlers:
                signal.signal(sig, handle_signal)
        for server in self.servers:
            server.start()
        for server in self.servers:
            server.wait_ready(self.server_timeout)
        self.urls = [server.base_url for server in self.servers]

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for server in self.servers:
            server.stop()


class ChatScheduler:
    def __init__(
        self,
        base_urls: list[str],
        api_key: str,
        model: str,
        sampling: SamplingConfig,
        max_concurrent_requests: int,
        mock_llm: bool = False,
        stage_max_new_tokens: Optional[dict[str, int]] = None,
        request_timeout_seconds: float = 900.0,
        stream_responses: bool = True,
        context_length: int = 0,
        context_margin_tokens: int = 64,
        tokenizer: Any = None,
        llm_call_logdir: Optional[Path] = None,
        architect_guided_json: bool = True,
        stream_interval_tokens: int = 100,
    ) -> None:
        if not base_urls and not mock_llm:
            raise ValueError("At least one vLLM base URL is required")
        self.base_urls = base_urls or ["mock://local"]
        self.model = model
        self.sampling = sampling
        self.mock_llm = mock_llm
        self.stream_responses = stream_responses
        self.context_length = max(0, int(context_length or 0))
        self.context_margin_tokens = max(0, int(context_margin_tokens))
        self.tokenizer = tokenizer
        self.llm_call_logdir = llm_call_logdir
        self.architect_guided_json = bool(architect_guided_json)
        self.stream_interval_tokens = max(1, int(stream_interval_tokens or 1))
        self.stage_max_new_tokens = {
            **_DEFAULT_STAGE_TOKEN_LIMITS,
            **(stage_max_new_tokens or {}),
        }
        self._clients = [
            OpenAI(base_url=url, api_key=api_key, timeout=request_timeout_seconds, max_retries=2)
            for url in self.base_urls
            if not mock_llm
        ]
        self._counter = 0
        self._counter_lock = asyncio.Lock()
        self._prompt_preview_lock = threading.Lock()
        self._previewed_prompt_hashes: set[tuple[str, str]] = set()
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    @staticmethod
    def _token_ids_to_list(token_ids: Any) -> list[int]:
        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", [])
        elif hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _render_completion_prompt(
        self,
        messages: list[dict[str, str]],
        assistant_prefix: Optional[str],
    ) -> tuple[str, list[int]]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for completion prompt rendering")
        template_kwargs = {
            "add_generation_prompt": assistant_prefix is None,
            "continue_final_message": assistant_prefix is not None,
        }
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **template_kwargs,
        )
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        return str(rendered_prompt), self._token_ids_to_list(token_ids)

    def _adjust_max_tokens_for_context(
        self,
        stage: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> int:
        if self.context_length <= 0:
            return max_tokens
        adjusted = self.context_length - prompt_tokens - self.context_margin_tokens
        adjusted = max(1, min(max_tokens, adjusted))
        if adjusted < max_tokens:
            logging.warning(
                "Reducing max_tokens for stage=%s from %d to %d before request "
                "(prompt_tokens=%d context=%d margin=%d)",
                stage,
                max_tokens,
                adjusted,
                prompt_tokens,
                self.context_length,
                self.context_margin_tokens,
            )
        return adjusted

    def _maybe_log_stage_prompt(
        self,
        stage: str,
        detail: str,
        rendered_prompt: str,
        prompt_tokens: int,
    ) -> None:
        prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()[:6]
        prompt_key = (stage, prompt_hash)
        with self._prompt_preview_lock:
            if prompt_key in self._previewed_prompt_hashes:
                return
            self._previewed_prompt_hashes.add(prompt_key)
        prompt_preview = rendered_prompt
        logging.info(
            "Chat-template prompt stage=%s detail=%s hash=%s chars=%d tokens=%d preview_chars=%d:\n%s",
            stage,
            detail or "-",
            prompt_hash,
            len(rendered_prompt),
            prompt_tokens,
            len(prompt_preview),
            prompt_preview,
        )

    def _llm_call_path(
        self,
        progress: Optional[PipelineProgress],
        stage: str,
        detail: str,
    ) -> Optional[Path]:
        if self.llm_call_logdir is None or progress is None:
            return None
        question_dir = self.llm_call_logdir / safe_path_component(progress.problem_id, "question")
        return question_dir / llm_call_filename(stage, detail)

    def _write_llm_call_input(
        self,
        path: Optional[Path],
        stage: str,
        detail: str,
        rendered_prompt: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        f"stage: {stage}",
                        f"detail: {detail or '-'}",
                        f"prompt_tokens: {prompt_tokens}",
                        f"max_tokens: {max_tokens}",
                        "",
                        "===== INPUT PROMPT =====",
                        rendered_prompt,
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        except OSError:
            logging.exception("Failed to write LLM call input debug file: %s", path)

    def _append_llm_call_output(
        self,
        path: Optional[Path],
        response: dict[str, Any],
    ) -> None:
        if path is None:
            return
        try:
            with open(path, "a", encoding="utf-8") as file_obj:
                file_obj.write("\n===== OUTPUT =====\n")
                file_obj.write(f"success: {response.get('success')}\n")
                file_obj.write(f"error: {response.get('error')}\n")
                file_obj.write(f"finish_reason: {response.get('finish_reason')}\n")
                file_obj.write(
                    "usage: "
                    + json.dumps(response.get("usage") or {}, ensure_ascii=False, default=str)
                    + "\n"
                )
                file_obj.write(f"server_url: {response.get('server_url')}\n")
                file_obj.write(f"latency_s: {response.get('latency_s')}\n\n")
                file_obj.write(str(response.get("text") or ""))
                file_obj.write("\n")
        except OSError:
            logging.exception("Failed to append LLM call output debug file: %s", path)

    def _append_llm_call_input_segment(
        self,
        path: Optional[Path],
        title: str,
        prompt_ids: list[int],
        max_tokens: int,
    ) -> None:
        if path is None:
            return
        try:
            decoded_prompt = self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
            )
        except Exception:
            logging.exception("Failed to decode LLM continuation prompt for debug file: %s", path)
            decoded_prompt = f"[decode failed; prompt_token_count={len(prompt_ids)}]"
        try:
            with open(path, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"\n===== {title} =====\n")
                file_obj.write(f"prompt_tokens: {len(prompt_ids)}\n")
                file_obj.write(f"max_tokens: {max_tokens}\n\n")
                file_obj.write(decoded_prompt)
                file_obj.write("\n")
        except OSError:
            logging.exception("Failed to append LLM continuation input debug file: %s", path)

    async def call(
        self,
        stage: str,
        prompt: str,
        temperature: Optional[float] = None,
        progress: Optional[PipelineProgress] = None,
        detail: str = "",
        thinking_budget_tokens: Optional[int] = None,
        thinking_budget_force_text: str = "",
    ) -> dict[str, Any]:

        response: Optional[dict[str, Any]] = None
        stream_id: Optional[str] = None
        try:
            async with self._semaphore:
                async with self._counter_lock:
                    index = self._counter % len(self.base_urls)
                    request_index = self._counter
                    self._counter += 1
                if self.mock_llm:
                    response = self._mock_response(stage, prompt, index)
                else:
                    if progress is not None and self.stream_responses:
                        stream_id = f"{stage}-{request_index}"
                    response = await asyncio.to_thread(
                        self._call_sync,
                        stage,
                        prompt,
                        index,
                        temperature,
                        progress,
                        stream_id,
                        detail,
                        thinking_budget_tokens,
                        thinking_budget_force_text,
                    )
        except Exception:
            if progress is not None:
                progress.complete(stage, False, detail)
            raise
        finally:
            if progress is not None and stream_id is not None:
                progress.stream_finish(stream_id)
        if response is None:
            raise RuntimeError(f"LLM call at stage={stage} returned no response")
        if progress is not None:
            usage = response.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            progress.complete(
                stage,
                bool(response.get("success")),
                detail,
                response.get("latency_s"),
                int(completion_tokens) if completion_tokens is not None else None,
            )
        return response

    def _call_sync(
        self,
        stage: str,
        prompt: str,
        index: int,
        temperature: Optional[float],
        progress: Optional[PipelineProgress] = None,
        stream_id: Optional[str] = None,
        detail: str = "",
        thinking_budget_tokens: Optional[int] = None,
        thinking_budget_force_text: str = "",
    ) -> dict[str, Any]:
        client = self._clients[index]
        call_log_path: Optional[Path] = None
        extra_body: dict[str, Any] = {}
        if self.sampling.top_k > 0:
            extra_body["top_k"] = self.sampling.top_k
        if self.sampling.min_p is not None:
            extra_body["min_p"] = self.sampling.min_p
        if self.sampling.min_new_tokens > 0:
            extra_body["min_tokens"] = self.sampling.min_new_tokens
        if stage in {"proof_architect", "proof_architect_repair"}:
            extra_body["repetition_penalty"] = 1.08
            extra_body["top_k"] = 40
            if self.architect_guided_json:
                extra_body["guided_json"] = ARCHITECT_GUIDED_JSON_SCHEMA
        if stage in {"proof_verify"}:
            extra_body["repetition_penalty"] = 1.05
        assistant_prefix = None
        messages = [{"role": "user", "content": prompt}]
        started = time.time()
        rendered_prompt, prompt_ids = self._render_completion_prompt(messages, assistant_prefix)
        prompt_tokens = len(prompt_ids)
        self._maybe_log_stage_prompt(stage, detail, rendered_prompt, prompt_tokens)
        configured_max_tokens = self.stage_max_new_tokens.get(stage, self.sampling.max_new_tokens)
        max_tokens = self._adjust_max_tokens_for_context(
            stage,
            prompt_tokens,
            configured_max_tokens,
        )
        call_log_path = self._llm_call_path(progress, stage, detail)
        self._write_llm_call_input(
            call_log_path,
            stage,
            detail,
            rendered_prompt,
            prompt_tokens,
            max_tokens,
        )
        if self.stream_responses:
            extra_body["return_token_ids"] = True
        try:
            request_extra_body = dict(extra_body)
            if (
                "min_tokens" in request_extra_body
                and int(request_extra_body["min_tokens"]) > max_tokens
            ):
                request_extra_body["min_tokens"] = max_tokens
            if self.stream_responses:
                if progress is not None and stream_id is not None:
                    progress.stream_start(stream_id, stage, detail, max_tokens)

                stream_segment_idx = 0

                def stream_completion(
                    active_prompt_ids: list[int],
                    active_max_tokens: int,
                    stop_after_tokens: Optional[int] = None,
                ) -> dict[str, Any]:
                    nonlocal stream_segment_idx
                    stream_segment_idx += 1
                    if stream_segment_idx > 1:
                        self._append_llm_call_input_segment(
                            call_log_path,
                            f"CONTINUATION INPUT PROMPT {stream_segment_idx}",
                            active_prompt_ids,
                            active_max_tokens,
                        )
                    active_extra_body = dict(request_extra_body)
                    if (
                        "min_tokens" in active_extra_body
                        and int(active_extra_body["min_tokens"]) > active_max_tokens
                    ):
                        active_extra_body["min_tokens"] = active_max_tokens
                    stream = client.completions.create(
                        model=self.model,
                        prompt=active_prompt_ids,
                        temperature=self.sampling.temperature if temperature is None else temperature,
                        top_p=self.sampling.top_p,
                        max_tokens=active_max_tokens,
                        stream=True,
                        extra_body=active_extra_body or None,
                    )
                    finish_reason: Optional[str] = None
                    usage: dict[str, Any] = {}
                    streamed_tokens = 0
                    generated_token_ids: list[int] = []
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    stopped_by_budget = False
                    stopped_by_repetition = False
                    repetition_line: Optional[str] = None
                    repetition_kind: Optional[str] = None
                    repetition_count = 0
                    last_repetition_check_tokens = 0
                    repetition_guard_enabled = bool(thinking_budget_force_text) and stage == "proof_verify"
                    try:
                        for chunk in stream:
                            chunk_usage = response_usage_to_dict(getattr(chunk, "usage", None))
                            if chunk_usage:
                                usage = chunk_usage
                            choices = getattr(chunk, "choices", None) or []
                            if not choices:
                                continue
                            choice = choices[0]
                            choice_finish_reason = getattr(choice, "finish_reason", None)
                            if choice_finish_reason is not None:
                                finish_reason = choice_finish_reason
                            content = getattr(choice, "text", None)
                            choice_extra = getattr(choice, "model_extra", None) or {}
                            reasoning = choice_extra.get("reasoning_content") or choice_extra.get("reasoning")
                            if content:
                                content_parts.append(content)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            token_ids = getattr(choice, "token_ids", None)
                            if token_ids is None:
                                token_ids = choice_extra.get("token_ids")
                            if token_ids:
                                token_id_list = self._token_ids_to_list(token_ids)
                                generated_token_ids, new_tokens = merge_streamed_token_ids(
                                    generated_token_ids,
                                    token_id_list,
                                )
                                if new_tokens:
                                    streamed_tokens += new_tokens
                                    if progress is not None and stream_id is not None:
                                        progress.stream_advance(stream_id, new_tokens)
                                    if (
                                        repetition_guard_enabled
                                        and finish_reason is None
                                        and streamed_tokens - last_repetition_check_tokens
                                        >= self.stream_interval_tokens
                                    ):
                                        last_repetition_check_tokens = streamed_tokens
                                        repetition = detect_duplicate_segment_in_recent_tokens(
                                            self.tokenizer,
                                            generated_token_ids,
                                        )
                                        if repetition is not None:
                                            repetition_line, repetition_count, repetition_kind = repetition
                                            stopped_by_repetition = True
                                            logging.warning(
                                                "Repetition guard triggered stage=%s detail=%s "
                                                "streamed=%d repeated_%s_count=%d text=%r",
                                                stage,
                                                detail,
                                                streamed_tokens,
                                                repetition_kind,
                                                repetition_count,
                                                repetition_line[:200],
                                            )
                                            break
                            if (
                                stop_after_tokens is not None
                                and finish_reason is None
                                and streamed_tokens >= stop_after_tokens
                            ):
                                stopped_by_budget = True
                                break
                            if stopped_by_repetition:
                                break
                    finally:
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                    content_text = "".join(content_parts)
                    reasoning_text = "".join(reasoning_parts)
                    text_parts: list[str] = []
                    if reasoning_text:
                        text_parts.append(f"<think>\n{reasoning_text}\n</think>\n\n")
                    text_parts.append(content_text)
                    return {
                        "text": "".join(text_parts),
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "streamed_tokens": streamed_tokens,
                        "generated_token_ids": generated_token_ids,
                        "stopped_by_budget": stopped_by_budget,
                        "stopped_by_repetition": stopped_by_repetition,
                        "repetition_line": repetition_line,
                        "repetition_kind": repetition_kind,
                        "repetition_count": repetition_count,
                    }

                budget_tokens: Optional[int] = None
                if (
                    thinking_budget_tokens is not None
                    and thinking_budget_force_text
                    and max_tokens > 1
                ):
                    budget_tokens = min(max(1, int(thinking_budget_tokens)), max_tokens - 1)

                first_segment = stream_completion(
                    prompt_ids,
                    max_tokens,
                    stop_after_tokens=budget_tokens,
                )
                text_parts = [first_segment["text"]]
                generated_token_ids = list(first_segment["generated_token_ids"])
                streamed_tokens = int(first_segment["streamed_tokens"])
                finish_reason = first_segment["finish_reason"]
                usage = first_segment["usage"]
                repetition_guard_applied = bool(first_segment.get("stopped_by_repetition"))
                thinking_budget_applied = bool(first_segment["stopped_by_budget"]) or repetition_guard_applied
                thinking_stop_reason = (
                    "repetition_guard" if repetition_guard_applied else "token_budget"
                )
                thinking_budget_skipped_closed = bool(
                    first_segment["stopped_by_budget"]
                    and has_closed_thinking_block(first_segment["text"])
                )

                if thinking_budget_applied and thinking_budget_skipped_closed:
                    remaining_tokens = max_tokens - streamed_tokens
                    logging.info(
                        "Skipped thinking budget force text stage=%s detail=%s "
                        "reason=%s budget=%s streamed=%d remaining=%d because </think> was already generated",
                        stage,
                        detail,
                        thinking_stop_reason,
                        budget_tokens,
                        streamed_tokens,
                        remaining_tokens,
                    )
                    if remaining_tokens > 0:
                        continuation_prompt_ids = prompt_ids + generated_token_ids
                        second_segment = stream_completion(
                            continuation_prompt_ids,
                            remaining_tokens,
                        )
                        text_parts.append(second_segment["text"])
                        generated_token_ids.extend(second_segment["generated_token_ids"])
                        streamed_tokens += int(second_segment["streamed_tokens"])
                        finish_reason = second_segment["finish_reason"]
                        if second_segment["usage"]:
                            usage = second_segment["usage"]
                    else:
                        finish_reason = "thinking_budget_reached"
                elif thinking_budget_applied:
                    intervention_text = thinking_budget_force_text
                    append_intervention_to_output = True
                    force_token_ids = self.tokenizer.encode(
                        intervention_text,
                        add_special_tokens=False,
                    )
                    force_token_ids = self._token_ids_to_list(force_token_ids)
                    if (
                        append_intervention_to_output
                        and force_token_ids
                        and progress is not None
                        and stream_id is not None
                    ):
                        progress.stream_advance(stream_id, len(force_token_ids))
                    if append_intervention_to_output:
                        text_parts.append(intervention_text)
                    remaining_tokens = max_tokens - streamed_tokens - len(force_token_ids)
                    logging.info(
                        "Applied thinking intervention stage=%s detail=%s reason=%s "
                        "visible=%s budget=%s streamed=%d force_tokens=%d remaining=%d "
                        "repetition_kind=%s repeated_text=%r",
                        stage,
                        detail,
                        thinking_stop_reason,
                        append_intervention_to_output,
                        budget_tokens,
                        streamed_tokens,
                        len(force_token_ids),
                        remaining_tokens,
                        first_segment.get("repetition_kind"),
                        first_segment.get("repetition_line"),
                    )
                    if remaining_tokens > 0:
                        continuation_prompt_ids = prompt_ids + generated_token_ids + force_token_ids
                        second_segment = stream_completion(
                            continuation_prompt_ids,
                            remaining_tokens,
                        )
                        text_parts.append(second_segment["text"])
                        if append_intervention_to_output:
                            generated_token_ids.extend(force_token_ids)
                        generated_token_ids.extend(second_segment["generated_token_ids"])
                        streamed_tokens += int(second_segment["streamed_tokens"])
                        if append_intervention_to_output:
                            streamed_tokens += len(force_token_ids)
                        finish_reason = second_segment["finish_reason"]
                        if second_segment["usage"]:
                            usage = second_segment["usage"]
                    else:
                        if append_intervention_to_output:
                            generated_token_ids.extend(force_token_ids)
                            streamed_tokens += len(force_token_ids)
                        finish_reason = "thinking_budget_reached"

                usage["completion_tokens"] = streamed_tokens
                usage["total_tokens"] = prompt_tokens + streamed_tokens
                usage["requested_max_tokens"] = max_tokens
                usage["estimated_prompt_tokens"] = prompt_tokens
                usage["thinking_budget_tokens"] = budget_tokens
                usage["thinking_budget_applied"] = thinking_budget_applied
                usage["thinking_budget_stop_reason"] = (
                    thinking_stop_reason if thinking_budget_applied else None
                )
                usage["repetition_guard_applied"] = repetition_guard_applied
                usage["repetition_guard_line"] = first_segment.get("repetition_line")
                usage["repetition_guard_kind"] = first_segment.get("repetition_kind")
                usage["repetition_guard_count"] = first_segment.get("repetition_count")
                usage["thinking_budget_force_skipped_closed"] = thinking_budget_skipped_closed
                text = (assistant_prefix or "") + "".join(text_parts)
                result = {
                    "stage": stage,
                    "success": True,
                    "error": None,
                    "text": text,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "server_url": self.base_urls[index],
                    "latency_s": time.time() - started,
                }
                self._append_llm_call_output(call_log_path, result)
                return result
            if thinking_budget_tokens is not None:
                logging.warning(
                    "Thinking budget for stage=%s requires streaming token IDs; "
                    "running a single-call request because stream_vllm is disabled.",
                    stage,
                )
            response = client.completions.create(
                model=self.model,
                prompt=prompt_ids,
                temperature=self.sampling.temperature if temperature is None else temperature,
                top_p=self.sampling.top_p,
                max_tokens=max_tokens,
                extra_body=request_extra_body or None,
            )
            choice = response.choices[0]
            text = (assistant_prefix or "") + (choice.text or "")
            usage = response_usage_to_dict(getattr(response, "usage", None))
            usage["requested_max_tokens"] = max_tokens
            usage["estimated_prompt_tokens"] = prompt_tokens
            result = {
                "stage": stage,
                "success": True,
                "error": None,
                "text": text,
                "finish_reason": choice.finish_reason,
                "usage": usage,
                "server_url": self.base_urls[index],
                "latency_s": time.time() - started,
            }
            self._append_llm_call_output(call_log_path, result)
            return result
        except Exception as exc:
            logging.exception("LLM call failed at stage=%s server=%s", stage, self.base_urls[index])
            result = {
                "stage": stage,
                "success": False,
                "error": repr(exc),
                "text": "",
                "finish_reason": None,
                "usage": {},
                "server_url": self.base_urls[index],
                "latency_s": time.time() - started,
            }
            self._append_llm_call_output(call_log_path, result)
            return result

    def _mock_response(self, stage: str, prompt: str, index: int) -> dict[str, Any]:
        if stage == "selector":
            text = "SELECTED_INDEX: 0\nREASON: mock selector chooses the first candidate."
        elif stage in {"proof_architect", "proof_architect_repair"}:
            text = (
                "## Strategy\nUse one mock lemma and assemble it.\n\n"
                "## Risks\n- Mock-only risk.\n\n"
                "### Lemma L1: Mock lemma\n"
                "Statement: A structurally valid mock lemma.\n"
                "Role: Used to validate decomposition plumbing.\n"
                "Depends: none\n"
                "Tests: trivial example.\n\n"
                "## Assembly\n1. Use L1 to assemble the mock proof."
            )
        elif stage == "sublemma_proof":
            text = (
                "## Status\nPROVED\n\n"
                "## Lemma\nA structurally valid mock lemma.\n\n"
                "## Proof or Counterexample\nThis validates the lemma-prover path.\n\n"
                "## Notes for Final Assembler\nThe final proof may use the mock lemma."
            )
        elif stage == "proof_assembly":
            text = "## Solution\nThis is a mock assembled proof produced for runner validation."
        elif stage in {"proof_generation", "proof_refine"}:
            text = (
                "## Solution\n"
                "This is a mock proof produced for runner validation.\n\n"
                "## Self Evaluation\n"
                "Here is my evaluation of the solution:\nThe mock output is structurally valid.\n\n"
                "Based on my evaluation, the final overall score should be:\n\\boxed{1}"
            )
        elif stage == "proof_meta_verify":
            text = 'Here is my analysis of the "solution evaluation":\nThe critique is reasonable.\n\n\\boxed{1}'
        else:
            text = "Here is my evaluation of the solution:\nThe proof is acceptable for a mock run.\n\n\\boxed{1}"
        return {
            "stage": stage,
            "success": True,
            "error": None,
            "text": text,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "server_url": self.base_urls[index],
            "latency_s": 0.0,
        }


def make_output(stage: str, response: dict[str, Any], parsed: dict[str, Any], **extra: Any) -> dict[str, Any]:
    output = {
        "stage": stage,
        "success": response.get("success", False),
        "error": response.get("error"),
        "text": response.get("text", ""),
        "parsed": parsed,
        "finish_reason": response.get("finish_reason"),
        "usage": response.get("usage", {}),
        "server_url": response.get("server_url"),
        "latency_s": response.get("latency_s"),
    }
    output.update(extra)
    return output


def score_sort_value(candidate: dict[str, Any]) -> float:
    score = candidate.get("final_score")
    if score is None:
        return -1.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return -1.0


def should_run_meta_verification(verifier: dict[str, Any], cfg: PipelineConfig) -> bool:
    if cfg.meta_n <= 0:
        return False
    score = verifier.get("score")
    if cfg.meta_policy == "all-reviews":
        return score is not None
    return score is not None and score < 1.0


def format_refinement_critique(critique: dict[str, Any]) -> str:
    score = critique.get("score")
    evaluation, _ = clip_middle_text(
        str(critique.get("evaluation") or "").strip(),
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    if score is None:
        score_text = "Verifier score: unknown"
    else:
        score_text = f"Verifier score: {score}"
    return f"{score_text}\n\nHere is my evaluation of the solution:\n{evaluation}".strip()


async def cancel_pending_tasks(tasks: list[asyncio.Task[Any]] | set[asyncio.Task[Any]]) -> int:
    pending = [task for task in tasks if not task.done()]
    if not pending:
        return 0
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return len(pending)


class VerificationThrottle:
    def __init__(
        self,
        total_generations: int,
        candidate_limit_while_generating: int,
        request_limit_while_generating: int,
    ) -> None:
        self.remaining_generations = max(0, int(total_generations))
        self.candidate_limit = max(0, int(candidate_limit_while_generating))
        self.request_limit = max(0, int(request_limit_while_generating))
        self.active_candidates = 0
        self.active_requests = 0
        self.generations: dict[int, dict[str, Any]] = {}
        self.generation_filter_cache: Optional[dict[str, Any]] = None
        self._condition = asyncio.Condition()

    def _generation_open(self) -> bool:
        return self.remaining_generations > 0

    async def mark_generation_done(
        self,
        attempt_idx: int,
        generation: Optional[dict[str, Any]] = None,
    ) -> None:
        async with self._condition:
            if generation is not None:
                self.generations[int(attempt_idx)] = generation
            self.remaining_generations = max(0, self.remaining_generations - 1)
            self._condition.notify_all()

    async def wait_for_all_generations(self) -> None:
        async with self._condition:
            while self._generation_open():
                await self._condition.wait()

    def valid_generations(self) -> list[dict[str, Any]]:
        return [
            self.generations[index]
            for index in sorted(self.generations)
        ]

    @contextlib.asynccontextmanager
    async def candidate_slot(self, attempt_idx: int, round_idx: int) -> Any:
        acquired = False
        async with self._condition:
            while (
                self._generation_open()
                and self.candidate_limit > 0
                and self.active_candidates >= self.candidate_limit
            ):
                await self._condition.wait()
            if self._generation_open() and self.candidate_limit > 0:
                self.active_candidates += 1
                acquired = True
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self.active_candidates = max(0, self.active_candidates - 1)
                    self._condition.notify_all()

    @contextlib.asynccontextmanager
    async def request_slot(self, stage: str, detail: str) -> Any:
        acquired = False
        async with self._condition:
            while (
                self._generation_open()
                and self.request_limit > 0
                and self.active_requests >= self.request_limit
            ):
                await self._condition.wait()
            if self._generation_open() and self.request_limit > 0:
                self.active_requests += 1
                acquired = True
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self.active_requests = max(0, self.active_requests - 1)
                    self._condition.notify_all()


def gold_score_sort_value(candidate: dict[str, Any]) -> float:
    score = candidate.get("gold_eval_score")
    if score is None:
        return -1.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return -1.0


def resolve_thinking_budget_tokens(attempt_idx: int, cfg: PipelineConfig) -> Optional[int]:
    if not cfg.thinking_budget_enabled or not cfg.proof_generation_thinking_budgets:
        return None
    index = min(max(0, attempt_idx), len(cfg.proof_generation_thinking_budgets) - 1)
    raw_budget = int(cfg.proof_generation_thinking_budgets[index])
    min_budget = max(1, int(cfg.proof_max_new_tokens) - 5000)
    return max(raw_budget, min_budget)


def resolve_proof_generation_temperature(attempt_idx: int, cfg: PipelineConfig) -> float:
    if not cfg.proof_generation_temperatures:
        return float(cfg.default_temperature)
    if 0 <= attempt_idx < len(cfg.proof_generation_temperatures):
        return float(cfg.proof_generation_temperatures[attempt_idx])
    return float(cfg.default_temperature)


def is_proof_only_candidate(attempt_idx: int, total_candidates: int, cfg: PipelineConfig) -> bool:
    proof_only_count = max(0, min(int(cfg.proof_only_candidate_count), int(total_candidates)))
    if proof_only_count <= 0:
        return False
    return attempt_idx >= max(0, int(total_candidates) - proof_only_count)


def is_length_stopped_proof_only_generation(initial_generation: dict[str, Any]) -> bool:
    generation_output = initial_generation.get("generation_output") or {}
    return (
        initial_generation.get("generation_mode") == "proof_only"
        and generation_output.get("finish_reason") == "length"
    )


def is_zero_self_scored_generation(initial_generation: dict[str, Any]) -> bool:
    if initial_generation.get("generation_mode") == "proof_only":
        return False
    parsed = initial_generation.get("generation_parsed") or {}
    score = coerce_score(parsed.get("self_score"))
    return score == 0.0


def filter_generations_before_verification(
    initial_generations: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return generation candidates for selection, verifier-eligible candidates, and skip records."""
    if len(initial_generations) <= 1:
        return list(initial_generations), list(initial_generations), []

    selection_pool = list(initial_generations)
    skipped: list[dict[str, Any]] = []

    complete_or_normal = [
        generation
        for generation in selection_pool
        if not is_length_stopped_proof_only_generation(generation)
    ]
    if complete_or_normal:
        for generation in selection_pool:
            if is_length_stopped_proof_only_generation(generation):
                skipped.append(
                    {
                        "attempt_idx": generation.get("attempt_idx"),
                        "reason": "proof_only_finish_reason_length",
                        "proof_chars": len(str(generation.get("proof") or "")),
                    }
                )
        selection_pool = complete_or_normal

    if cfg.skip_self_score_zero:
        nonzero_or_unscored = [
            generation
            for generation in selection_pool
            if not is_zero_self_scored_generation(generation)
        ]
        if nonzero_or_unscored:
            for generation in selection_pool:
                if is_zero_self_scored_generation(generation):
                    skipped.append(
                        {
                            "attempt_idx": generation.get("attempt_idx"),
                            "reason": "self_score_zero",
                            "proof_chars": len(str(generation.get("proof") or "")),
                        }
                    )
            selection_pool = nonzero_or_unscored

    verifier_pool = [
        generation
        for generation in selection_pool
        if not is_length_stopped_proof_only_generation(generation)
    ]
    return selection_pool, verifier_pool, skipped


def make_generation_only_candidate(
    initial_generation: dict[str, Any],
    final_status: str,
) -> dict[str, Any]:
    parsed = initial_generation.get("generation_parsed") or {}
    proof = str(initial_generation.get("proof") or "")
    return {
        "attempt_idx": initial_generation.get("attempt_idx"),
        "generation_mode": initial_generation.get("generation_mode"),
        "proof_generation_output": initial_generation.get("generation_output"),
        "proof_verify_output": [],
        "proof_meta_verify_output": [],
        "proof_refine_output": [],
        "proof_solution": proof,
        "self_evaluation": parsed.get("self_evaluation"),
        "self_score": parsed.get("self_score"),
        "validated_critiques": [],
        "verifier_score_summaries": [],
        "final_score": parsed.get("self_score"),
        "final_status": final_status,
        "low_scores_seen": 0,
        "strict_pass": False,
        "all_verifiers_passed": False,
        "meta_valid_count": 0,
        "meta_checked_count": 0,
        "meta_summary_by_verifier": {},
        "success": True,
        "generation_only": True,
    }


async def maybe_wait_all_candidates(
    initial_generation: dict[str, Any],
    cfg: PipelineConfig,
    throttle: Optional[VerificationThrottle],
    progress: Optional[PipelineProgress] = None,
) -> dict[str, Any]:
    if throttle is None or not cfg.wait_for_all_generations_before_verify:
        return {"action": "verify"}

    await throttle.wait_for_all_generations()
    cache_created = False
    async with throttle._condition:
        if throttle.generation_filter_cache is None:
            initial_generations = throttle.valid_generations()
            selection_generations, verifier_generations, skipped = (
                filter_generations_before_verification(initial_generations, cfg)
            )
            selected_generation = None
            selected_status = None
            if len(selection_generations) == 1:
                selected_generation = selection_generations[0]
                selected_status = (
                    "single_valid_generation"
                    if len(initial_generations) == 1
                    else "single_candidate_after_generation_filter"
                )
            elif not verifier_generations and selection_generations:
                selected_generation = selection_generations[
                    select_generation_fallback_index(selection_generations)
                ]
                selected_status = "no_verifier_eligible_generation_fallback"
            throttle.generation_filter_cache = {
                "selection_count": len(selection_generations),
                "verifier_attempts": {
                    int(generation.get("attempt_idx") or 0)
                    for generation in verifier_generations
                },
                "skipped_by_attempt": {
                    int(item.get("attempt_idx")): item
                    for item in skipped
                    if item.get("attempt_idx") is not None
                },
                "selected_attempt": (
                    int(selected_generation.get("attempt_idx") or 0)
                    if selected_generation is not None
                    else None
                ),
                "selected_status": selected_status,
                "selected_generation": selected_generation,
                "skipped_count": len(skipped),
            }
            cache_created = True
        cache = dict(throttle.generation_filter_cache)

    if progress is not None and cache_created:
        progress.log(
            "stage=generation_filter status=complete selection=%d verifier=%d skipped=%d",
            cache["selection_count"],
            len(cache["verifier_attempts"]),
            cache["skipped_count"],
        )
        for skipped in cache["skipped_by_attempt"].values():
            progress.log(
                "candidate=%s stage=generation_filter status=skip reason=%s proof_chars=%s",
                skipped.get("attempt_idx"),
                skipped.get("reason"),
                skipped.get("proof_chars"),
            )

    attempt_idx = int(initial_generation.get("attempt_idx") or 0)
    if cache["selected_attempt"] is not None:
        if attempt_idx == cache["selected_attempt"]:
            return {
                "action": "verify",
                "selected_status": cache["selected_status"],
            }
        return {"action": "skip", "reason": "not_selected_after_generation_filter"}
    if attempt_idx in cache["skipped_by_attempt"]:
        return {
            "action": "skip",
            "reason": cache["skipped_by_attempt"][attempt_idx].get("reason"),
        }
    if attempt_idx not in cache["verifier_attempts"]:
        return {"action": "skip", "reason": "not_verifier_eligible_after_generation_filter"}
    return {"action": "verify"}


async def generate_single_attempt(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
) -> Optional[dict[str, Any]]:
    proof_only = is_proof_only_candidate(attempt_idx, total_candidates, cfg)
    prompt_builder = (
        build_deepseek_proof_only_generation_prompt
        if proof_only
        else build_deepseek_proof_generation_prompt
    )
    generation_mode = "proof_only" if proof_only else "proof_with_self_evaluation"
    if progress is not None:
        progress.log(
            "candidate=%d stage=generation status=start mode=%s",
            attempt_idx,
            generation_mode,
        )
    generation_response = await scheduler.call(
        "proof_generation",
        prompt_builder(
            question,
        ),
        progress=progress,
        detail=f"candidate={attempt_idx} round=0 mode={generation_mode}",
        temperature=resolve_proof_generation_temperature(attempt_idx, cfg),
        thinking_budget_tokens=resolve_thinking_budget_tokens(attempt_idx, cfg),
        thinking_budget_force_text=cfg.thinking_budget_force_text,
    )
    log_generation_visible_output(
        "proof_generation",
        generation_response.get("text", ""),
        problem_id=problem_id,
        attempt_idx=attempt_idx,
        round_idx=0,
    )
    generation_parsed = parse_generation_response(
        generation_response.get("text", ""),
        require_self_evaluation=not proof_only,
    )
    generation_parsed["generation_mode"] = generation_mode
    if not require_valid_candidate_response(
        generation_parsed,
        problem_id=problem_id,
        attempt_idx=attempt_idx,
        stage="proof_generation",
        round_idx=0,
    ):
        return None
    proof = generation_parsed["proof"]
    generation_output = make_output("proof_generation", generation_response, generation_parsed)
    if progress is not None:
        progress.log(
            "candidate=%d stage=generation status=parsed mode=%s has_solution=%s "
            "has_self_evaluation=%s proof_chars=%d self_score=%s",
            attempt_idx,
            generation_mode,
            generation_parsed["has_solution_section"],
            generation_parsed["has_self_evaluation_section"],
            len(proof),
            generation_parsed.get("self_score"),
        )
    return {
        "attempt_idx": attempt_idx,
        "generation_output": generation_output,
        "generation_parsed": generation_parsed,
        "generation_mode": generation_mode,
        "proof": proof,
    }


async def generate_decomposed_attempt(
    question: str,
    attempt_idx: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
) -> Optional[dict[str, Any]]:
    generation_mode = "decompose_sublemmas"
    if progress is not None:
        progress.log(
            "candidate=%d stage=decomposition status=start",
            attempt_idx,
        )
    architect_response = await scheduler.call(
        "proof_architect",
        build_proof_architect_prompt(question),
        progress=progress,
        detail=f"candidate={attempt_idx} round=0",
        temperature=0.0,
    )
    architect_output = make_output(
        "proof_architect",
        architect_response,
        {},
    )
    try:
        analyst_json = parse_architect_output(
            architect_response.get("text", ""),
            cfg.decomposition_max_sublemmas,
        )
    except Exception as exc:
        logging.warning(
            "Architect plan parse failed problem=%s candidate=%d: %s; retrying repair",
            problem_id,
            attempt_idx,
            exc,
        )
        repair_response = await scheduler.call(
            "proof_architect_repair",
            build_proof_architect_repair_prompt(
                question,
                architect_response.get("text", ""),
            ),
            progress=progress,
            detail=f"candidate={attempt_idx} round=0",
            temperature=0.0,
        )
        try:
            analyst_json = parse_architect_output(
                repair_response.get("text", ""),
                cfg.decomposition_max_sublemmas,
            )
        except Exception as repair_exc:
            logging.warning(
                "Architect repair plan parse failed problem=%s candidate=%d: %s",
                problem_id,
                attempt_idx,
                repair_exc,
            )
            return None
        architect_output = make_output(
            "proof_architect_repair",
            repair_response,
            {"analyst_json": analyst_json, "repaired_from_failed_architect": True},
            original_architect_output=architect_output,
        )
    else:
        architect_output = make_output(
            "proof_architect",
            architect_response,
            {"analyst_json": analyst_json},
        )
    tasks = order_sublemma_tasks(
        analyst_sublemma_tasks(analyst_json, cfg.decomposition_max_sublemmas)
    )
    if not tasks:
        logging.warning(
            "Architect returned no usable sublemma tasks problem=%s candidate=%d",
            problem_id,
            attempt_idx,
        )
        return None

    lemma_results: dict[str, dict[str, Any]] = {}
    sublemma_outputs: list[dict[str, Any]] = []

    async def prove_sublemma(task: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        lemma_id = str(task["lemma_id"])
        allowed_previous = format_allowed_previous_lemmas(
            list(task.get("allowed_previous_lemmas") or []),
            lemma_results,
        )
        response = await scheduler.call(
            "sublemma_proof",
            build_sublemma_prover_prompt(question, task, allowed_previous),
            progress=progress,
            detail=f"candidate={attempt_idx} round=0 lemma={lemma_id}",
            temperature=0.4,
        )
        status = parse_sublemma_status(response.get("text", ""))
        result = {
            "lemma_id": lemma_id,
            "status": status,
            "text": response.get("text", ""),
            "task": task,
        }
        output = make_output(
            "sublemma_proof",
            response,
            {"lemma_id": lemma_id, "status": status},
            lemma_id=lemma_id,
        )
        return lemma_id, result, output

    for wave_idx, wave in enumerate(sublemma_task_waves(tasks)):
        if progress is not None:
            progress.log(
                "candidate=%d stage=sublemma_wave status=start wave=%d count=%d",
                attempt_idx,
                wave_idx,
                len(wave),
            )
        wave_results = await asyncio.gather(
            *[prove_sublemma(task) for task in wave],
            return_exceptions=True,
        )
        for item in wave_results:
            if isinstance(item, Exception):
                logging.error(
                    "Sublemma proof failed problem=%s candidate=%d",
                    problem_id,
                    attempt_idx,
                    exc_info=(type(item), item, item.__traceback__),
                )
                continue
            lemma_id, result, output = item
            lemma_results[lemma_id] = result
            sublemma_outputs.append(output)
            status = result.get("status")
            if progress is not None:
                progress.log(
                    "candidate=%d stage=sublemma status=complete lemma=%s lemma_status=%s",
                    attempt_idx,
                    lemma_id,
                    status,
                )

    proved_sublemmas = format_proved_sublemmas(lemma_results)
    if not proved_sublemmas.strip():
        logging.warning(
            "No proved sublemmas available for assembly problem=%s candidate=%d",
            problem_id,
            attempt_idx,
        )
        return None
    assembly_response = await scheduler.call(
        "proof_assembly",
        build_final_assembly_prompt(question, analyst_json, proved_sublemmas),
        progress=progress,
        detail=f"candidate={attempt_idx} round=0 sublemmas={len(lemma_results)}",
        temperature=0.3,
    )
    log_generation_visible_output(
        "proof_assembly",
        assembly_response.get("text", ""),
        problem_id=problem_id,
        attempt_idx=attempt_idx,
        round_idx=0,
    )
    generation_parsed = parse_assembled_solution(assembly_response.get("text", ""))
    generation_parsed["generation_mode"] = generation_mode
    if not require_valid_candidate_response(
        generation_parsed,
        problem_id=problem_id,
        attempt_idx=attempt_idx,
        stage="proof_assembly",
        round_idx=0,
    ):
        return None
    proof = generation_parsed["proof"]
    generation_output = make_output(
        "proof_assembly",
        assembly_response,
        generation_parsed,
        architect_output=architect_output,
        sublemma_outputs=sublemma_outputs,
        lemma_statuses={
            lemma_id: result.get("status")
            for lemma_id, result in lemma_results.items()
        },
    )
    if progress is not None:
        progress.log(
            "candidate=%d stage=decomposition status=assembled proof_chars=%d sublemmas=%d",
            attempt_idx,
            len(proof),
            len(lemma_results),
        )
    return {
        "attempt_idx": attempt_idx,
        "generation_output": generation_output,
        "generation_parsed": generation_parsed,
        "generation_mode": generation_mode,
        "proof": proof,
    }


async def run_verification_round(
    question: str,
    proof: str,
    attempt_idx: int,
    round_idx: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    progress: Optional[PipelineProgress] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    verifier_results_by_index: dict[int, dict[str, Any]] = {}
    verifier_outputs_by_index: dict[int, dict[str, Any]] = {}
    meta_results_by_verifier: dict[int, list[dict[str, Any]]] = {}
    proof_meta_verify_output: list[dict[str, Any]] = []
    task_info: dict[asyncio.Task[Any], dict[str, Any]] = {}
    pending: set[asyncio.Task[Any]] = set()
    cancelled_count = 0
    early_stop = False

    def add_task(coro: Any, info: dict[str, Any]) -> None:
        task = asyncio.create_task(coro)
        pending.add(task)
        task_info[task] = info

    async def verification_call(stage: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        detail = str(kwargs.get("detail") or "")
        if throttle is None:
            return await scheduler.call(stage, prompt, **kwargs)
        async with throttle.request_slot(stage, detail):
            return await scheduler.call(stage, prompt, **kwargs)

    def sorted_verifier_results() -> list[dict[str, Any]]:
        return [
            verifier_results_by_index[index]
            for index in sorted(verifier_results_by_index)
        ]

    def current_aggregation() -> dict[str, Any]:
        return aggregate_proof_label(
            sorted_verifier_results(),
            meta_results_by_verifier,
            cfg.min_valid_low,
            strict_pass_meta=cfg.strict_pass_meta and cfg.meta_n > 0,
            meta_n=cfg.meta_n,
        )

    def enough_validated_critiques() -> bool:
        aggregation = current_aggregation()
        return (
            round_idx < cfg.refine_rounds
            and aggregation.get("final_score") is not None
            and aggregation.get("final_score") < 1.0
            and len(aggregation.get("validated_critiques") or []) >= max(1, cfg.refine_review_n)
        )

    def finalize_verifier(
        verifier_index: int,
        response: dict[str, Any],
        parsed: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        verifier_result = {
            "verifier_index": verifier_index,
            "success": response.get("success", False),
            "error": response.get("error"),
            **parsed,
        }
        verifier_results_by_index[verifier_index] = verifier_result
        verifier_outputs_by_index[verifier_index] = output
        if parsed.get("evaluation_clipped") or not parsed.get("evaluation_marker_found"):
            logging.info(
                "Verifier evaluation extracted candidate=%d round=%d verifier=%d "
                "marker_found=%s raw_chars=%s forwarded_chars=%s clipped=%s score=%s",
                attempt_idx,
                round_idx,
                verifier_index,
                parsed.get("evaluation_marker_found"),
                parsed.get("evaluation_raw_chars"),
                parsed.get("evaluation_forwarded_chars"),
                parsed.get("evaluation_clipped"),
                parsed.get("score"),
            )
        if not should_run_meta_verification(verifier_result, cfg):
            return
        for meta_idx in range(cfg.meta_n):
            add_task(
                verification_call(
                    "proof_meta_verify",
                    build_deepseek_meta_verification_prompt(
                        question,
                        proof,
                        verifier_result.get("evaluation", ""),
                    ),
                    progress=progress,
                    detail=(
                        f"candidate={attempt_idx} round={round_idx} "
                        f"verifier={verifier_index} meta={meta_idx}"
                    ),
                    thinking_budget_tokens=(
                        cfg.meta_thinking_budget_tokens if cfg.thinking_budget_enabled else None
                    ),
                    thinking_budget_force_text=cfg.meta_thinking_budget_force_text,
                ),
                {
                    "kind": "meta",
                    "verifier_index": verifier_index,
                    "meta_index": meta_idx,
                },
            )

    verifier_prompts = [
        build_deepseek_proof_verification_prompt(question, proof)
        for _ in range(cfg.verify_n)
    ]
    for verifier_idx, prompt in enumerate(verifier_prompts):
        add_task(
            verification_call(
                "proof_verify",
                prompt,
                progress=progress,
                detail=f"candidate={attempt_idx} round={round_idx} verifier={verifier_idx}",
                thinking_budget_tokens=(
                    cfg.verifier_thinking_budget_tokens if cfg.thinking_budget_enabled else None
                ),
                thinking_budget_force_text=cfg.verifier_thinking_budget_force_text,
            ),
            {"kind": "verifier", "verifier_index": verifier_idx},
        )

    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                info = task_info.pop(task)
                kind = info["kind"]
                try:
                    response = task.result()
                except Exception as exc:
                    logging.exception(
                        "Verification task failed candidate=%d round=%d kind=%s info=%s",
                        attempt_idx,
                        round_idx,
                        kind,
                        info,
                    )
                    response = {
                        "success": False,
                        "error": repr(exc),
                        "text": "",
                        "finish_reason": None,
                        "usage": {},
                        "server_url": None,
                        "latency_s": None,
                    }

                if kind == "verifier":
                    verifier_index = int(info["verifier_index"])
                    log_verifier_tail_output(
                        "proof_verify",
                        response.get("text", ""),
                        attempt_idx=attempt_idx,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                    )
                    parsed = parse_verifier_response(response.get("text", ""))
                    output = make_output(
                        "proof_verify",
                        response,
                        parsed,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                    )
                    finalize_verifier(verifier_index, response, parsed, output)

                elif kind == "meta":
                    verifier_index = int(info["verifier_index"])
                    meta_idx = int(info["meta_index"])
                    log_verifier_tail_output(
                        "proof_meta_verify",
                        response.get("text", ""),
                        attempt_idx=attempt_idx,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                        meta_index=meta_idx,
                    )
                    parsed = parse_meta_verifier_response(response.get("text", ""))
                    meta_result = {
                        "meta_index": meta_idx,
                        "verifier_index": verifier_index,
                        "success": response.get("success", False),
                        "error": response.get("error"),
                        **parsed,
                    }
                    meta_output = make_output(
                        "proof_meta_verify",
                        response,
                        parsed,
                        round_idx=round_idx,
                        verifier_index=verifier_index,
                        meta_index=meta_idx,
                    )
                    meta_results_by_verifier.setdefault(verifier_index, []).append(meta_result)
                    proof_meta_verify_output.append(meta_output)

                if enough_validated_critiques():
                    print("early stopping verification round due to enough validated critiques")
                    early_stop = True
            if early_stop:
                cancelled_count += await cancel_pending_tasks(pending)
                pending.clear()
    finally:
        if pending:
            cancelled_count += await cancel_pending_tasks(pending)

    verifier_results = sorted_verifier_results()
    verifier_outputs = [
        verifier_outputs_by_index[index]
        for index in sorted(verifier_outputs_by_index)
    ]
    final_aggregation = current_aggregation()
    final_aggregation["verification_early_stop"] = early_stop
    final_aggregation["verification_cancelled_tasks"] = cancelled_count
    return (
        verifier_results,
        verifier_outputs,
        meta_results_by_verifier,
        proof_meta_verify_output,
        final_aggregation,
    )


async def run_single_attempt(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    initial_generation: Optional[dict[str, Any]] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> dict[str, Any]:
    if initial_generation is None:
        initial_generation = await generate_single_attempt(
            question,
            attempt_idx,
            total_candidates,
            scheduler,
            cfg,
            problem_id=problem_id,
            progress=progress,
        )
    if initial_generation is None:
        return {
            "attempt_idx": attempt_idx,
            "skipped": True,
            "skip_reason": "invalid_generation",
            "success": False,
        }
    generation_output = initial_generation["generation_output"]
    generation_parsed = initial_generation["generation_parsed"]
    proof = initial_generation["proof"]

    proof_verify_output: list[dict[str, Any]] = []
    proof_meta_verify_output: list[dict[str, Any]] = []
    proof_refine_output: list[dict[str, Any]] = []
    final_aggregation: dict[str, Any] = {
        "final_score": None,
        "final_status": "not_verified",
        "validated_critiques": [],
        "low_scores_seen": 0,
    }
    latest_generation_parsed = generation_parsed
    best_proof = proof
    best_generation_parsed = latest_generation_parsed
    best_aggregation = dict(final_aggregation)
    best_round_idx = -1
    latest_verified_round_idx = -1

    def aggregation_score_value(aggregation: dict[str, Any]) -> float:
        score = aggregation.get("final_score")
        if score is None:
            return -1.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return -1.0

    for round_idx in range(cfg.refine_rounds + 1):
        if progress is not None:
            progress.log(
                "candidate=%d round=%d/%d stage=verification status=start verify_n=%d proof_chars=%d",
                attempt_idx,
                round_idx,
                cfg.refine_rounds,
                cfg.verify_n,
                len(proof),
            )
        if throttle is not None:
            verification_context = throttle.candidate_slot(attempt_idx, round_idx)
        else:
            verification_context = contextlib.nullcontext()
        async with verification_context:
            (
                verifier_results,
                verifier_outputs,
                meta_results_by_verifier,
                meta_outputs,
                final_aggregation,
            ) = await run_verification_round(
                question,
                proof,
                attempt_idx,
                round_idx,
                scheduler,
                cfg,
                progress=progress,
                throttle=throttle,
            )
        proof_verify_output.extend(verifier_outputs)
        proof_meta_verify_output.extend(meta_outputs)
        latest_verified_round_idx = round_idx
        if (
            best_round_idx < 0
            or aggregation_score_value(final_aggregation) >= aggregation_score_value(best_aggregation)
        ):
            best_proof = proof
            best_generation_parsed = latest_generation_parsed
            best_aggregation = dict(final_aggregation)
            best_round_idx = round_idx
        if progress is not None:
            score_counts = {
                score: sum(1 for result in verifier_results if result.get("score") == score)
                for score in (0.0, 0.5, 1.0)
            }
            progress.log(
                "candidate=%d round=%d stage=verification status=complete scores=%s "
                "validated_critiques=%d final_status=%s",
                attempt_idx,
                round_idx,
                score_counts,
                len(final_aggregation.get("validated_critiques") or []),
                final_aggregation.get("final_status"),
            )
        should_refine = (
            round_idx < cfg.refine_rounds
            and final_aggregation.get("final_score") is not None
            and final_aggregation.get("final_score") < 1.0
            and final_aggregation.get("validated_critiques")
        )
        if not should_refine:
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=refinement status=skip final_score=%s "
                    "final_status=%s validated_critiques=%d refine_rounds=%d",
                    attempt_idx,
                    round_idx,
                    final_aggregation.get("final_score"),
                    final_aggregation.get("final_status"),
                    len(final_aggregation.get("validated_critiques") or []),
                    cfg.refine_rounds,
                )
            break

        ranked_critiques = sorted(
            final_aggregation["validated_critiques"],
            key=lambda critique: (
                float(critique.get("score") if critique.get("score") is not None else 1.0),
                int(critique.get("verifier_index") or 0),
            ),
        )
        selected_critiques = ranked_critiques[: max(1, cfg.refine_review_n)]
        critiques = [format_refinement_critique(critique) for critique in selected_critiques]
        if progress is not None:
            progress.log(
                "candidate=%d round=%d/%d stage=refinement status=start reviews=%d",
                attempt_idx,
                round_idx + 1,
                cfg.refine_rounds,
                len(critiques),
            )
        refinement_response = await scheduler.call(
            "proof_refine",
            build_deepseek_proof_refinement_prompt(question, proof, critiques),
            progress=progress,
            detail=f"candidate={attempt_idx} round={round_idx + 1} reviews={len(critiques)}",
        )
        log_generation_visible_output(
            "proof_refine",
            refinement_response.get("text", ""),
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            round_idx=round_idx + 1,
        )
        refinement_parsed = parse_generation_response(refinement_response.get("text", ""))
        if not require_valid_candidate_response(
            refinement_parsed,
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            stage="proof_refine",
            round_idx=round_idx + 1,
        ):
            proof_refine_output.append(
                make_output(
                    "proof_refine",
                    refinement_response,
                    refinement_parsed,
                    round_idx=round_idx + 1,
                    used_critiques=critiques,
                    invalid=True,
                )
            )
            if progress is not None:
                progress.log(
                    "candidate=%d round=%d stage=refinement status=invalid_keep_previous",
                    attempt_idx,
                    round_idx + 1,
                )
            break
        proof = refinement_parsed["proof"]
        latest_generation_parsed = refinement_parsed
        proof_refine_output.append(
            make_output(
                "proof_refine",
                refinement_response,
                refinement_parsed,
                round_idx=round_idx + 1,
                used_critiques=critiques,
            )
        )
        if progress is not None:
            progress.log(
                "candidate=%d round=%d stage=refinement status=complete has_solution=%s "
                "has_self_evaluation=%s proof_chars=%d self_score=%s",
                attempt_idx,
                round_idx + 1,
                refinement_parsed["has_solution_section"],
                refinement_parsed["has_self_evaluation_section"],
                len(proof),
                refinement_parsed.get("self_score"),
            )

    rollback_from_round = None
    if best_round_idx >= 0 and best_round_idx != latest_verified_round_idx:
        rollback_from_round = latest_verified_round_idx
        if progress is not None:
            progress.log(
                "candidate=%d stage=refinement status=rollback selected_round=%d "
                "rejected_round=%d selected_score=%s rejected_score=%s",
                attempt_idx,
                best_round_idx,
                latest_verified_round_idx,
                best_aggregation.get("final_score"),
                final_aggregation.get("final_score"),
            )
        proof = best_proof
        latest_generation_parsed = best_generation_parsed
        final_aggregation = best_aggregation

    if progress is not None:
        progress.log(
            "candidate=%d status=complete final_score=%s final_status=%s refinements=%d "
            "selected_round=%d rollback_from_round=%s",
            attempt_idx,
            final_aggregation.get("final_score"),
            final_aggregation.get("final_status"),
            len(proof_refine_output),
            best_round_idx,
            rollback_from_round,
        )
    return {
        "attempt_idx": attempt_idx,
        "generation_mode": initial_generation.get("generation_mode"),
        "proof_generation_output": generation_output,
        "proof_verify_output": proof_verify_output,
        "proof_meta_verify_output": proof_meta_verify_output,
        "proof_refine_output": proof_refine_output,
        "proof_solution": proof,
        "self_evaluation": latest_generation_parsed.get("self_evaluation"),
        "self_score": latest_generation_parsed.get("self_score"),
        "validated_critiques": final_aggregation.get("validated_critiques", []),
        "verifier_score_summaries": final_aggregation.get("verifier_score_summaries", []),
        "final_score": final_aggregation.get("final_score"),
        "final_status": final_aggregation.get("final_status"),
        "low_scores_seen": final_aggregation.get("low_scores_seen"),
        "strict_pass": final_aggregation.get("strict_pass", False),
        "all_verifiers_passed": final_aggregation.get("all_verifiers_passed", False),
        "meta_valid_count": final_aggregation.get("meta_valid_count", 0),
        "meta_checked_count": final_aggregation.get("meta_checked_count", 0),
        "meta_summary_by_verifier": final_aggregation.get("meta_summary_by_verifier", {}),
        "selected_verification_round": best_round_idx,
        "rollback_from_round": rollback_from_round,
        "success": final_aggregation.get("final_status") != "needs_review",
    }


async def run_candidate_pipeline(
    question: str,
    attempt_idx: int,
    total_candidates: int,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    throttle: Optional[VerificationThrottle] = None,
) -> dict[str, Any]:
    try:
        initial_generation = None
        attempted_decomposition = False
        try:
            if (
                cfg.decomposition_strategy_enabled
                and attempt_idx < cfg.decomposition_candidate_count
            ):
                attempted_decomposition = True
                initial_generation = await generate_decomposed_attempt(
                    question,
                    attempt_idx,
                    scheduler,
                    cfg,
                    problem_id=problem_id,
                    progress=progress,
                )
                if (
                    initial_generation is None
                    and cfg.decomposition_fallback_to_direct
                ):
                    logging.warning(
                        "Decomposition generation failed; falling back to direct generation "
                        "problem=%s candidate=%d",
                        problem_id,
                        attempt_idx,
                    )
            if (
                initial_generation is None
                and (
                    not attempted_decomposition
                    or cfg.decomposition_fallback_to_direct
                )
            ):
                initial_generation = await generate_single_attempt(
                    question,
                    attempt_idx,
                    total_candidates,
                    scheduler,
                    cfg,
                    problem_id=problem_id,
                    progress=progress,
                )
        finally:
            if throttle is not None:
                await throttle.mark_generation_done(attempt_idx, initial_generation)

        if initial_generation is None:
            return {
                "attempt_idx": attempt_idx,
                "candidate": None,
                "error": "invalid_generation",
            }
        wait_decision = await maybe_wait_all_candidates(
            initial_generation,
            cfg,
            throttle,
            progress=progress,
        )
        if wait_decision["action"] == "candidate":
            return {
                "attempt_idx": attempt_idx,
                "candidate": wait_decision["candidate"],
                "error": None,
            }
        if wait_decision["action"] == "skip":
            return {
                "attempt_idx": attempt_idx,
                "candidate": None,
                "error": wait_decision.get("reason", "skipped_after_generation_filter"),
            }
        if progress is not None and wait_decision.get("selected_status"):
            progress.log(
                "candidate=%d stage=generation_filter status=%s action=verify",
                attempt_idx,
                wait_decision.get("selected_status"),
            )
        candidate = await run_single_attempt(
            question,
            attempt_idx,
            total_candidates,
            scheduler,
            cfg,
            problem_id=problem_id,
            progress=progress,
            initial_generation=initial_generation,
            throttle=throttle,
        )
        return {"attempt_idx": attempt_idx, "candidate": candidate, "error": None}
    except Exception as exc:
        logging.exception("Pipeline failed id=%s attempt=%d", problem_id, attempt_idx)
        return {"attempt_idx": attempt_idx, "candidate": None, "error": repr(exc)}


def select_generation_fallback_index(initial_generations: list[dict[str, Any]]) -> int:
    return max(
        range(len(initial_generations)),
        key=lambda idx: (
            score_sort_value(make_generation_only_candidate(initial_generations[idx], "generation_only_fallback")),
            len(str(initial_generations[idx].get("proof") or "")),
        ),
    )


async def run_streaming_candidates(
    question: str,
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    pipelines_per_problem: int,
    *,
    problem_id: Any = "problem",
    progress: Optional[PipelineProgress] = None,
    timeout_s: Optional[float] = None,
) -> dict[str, Any]:
    verification_throttle = VerificationThrottle(
        pipelines_per_problem,
        cfg.verify_candidate_limit_while_generating,
        cfg.verify_request_limit_while_generating,
    )
    pipeline_tasks = [
        asyncio.create_task(
            run_candidate_pipeline(
                question,
                attempt_idx,
                pipelines_per_problem,
                scheduler,
                cfg,
                problem_id=problem_id,
                progress=progress,
                throttle=verification_throttle,
            )
        )
        for attempt_idx in range(pipelines_per_problem)
    ]
    candidates: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    strict_pass_candidate: Optional[dict[str, Any]] = None
    cancelled_count = 0
    try:
        iterator = (
            asyncio.as_completed(pipeline_tasks, timeout=max(0.1, timeout_s))
            if timeout_s is not None
            else asyncio.as_completed(pipeline_tasks)
        )
        for future in iterator:
            try:
                result = await future
            except (TimeoutError, asyncio.TimeoutError):
                raise
            except Exception as exc:
                logging.exception("Pipeline task failed id=%s", problem_id)
                failed_attempts.append({"attempt_idx": None, "error": repr(exc)})
                continue
            if result["candidate"] is None:
                failed_attempts.append(
                    {"attempt_idx": result["attempt_idx"], "error": result["error"]}
                )
                continue
            candidate = result["candidate"]
            candidates.append(candidate)
            if progress is not None:
                progress.log(
                    "candidate=%d status=pipeline_collected final_score=%s final_status=%s "
                    "strict_pass=%s completed=%d/%d",
                    result["attempt_idx"],
                    candidate.get("final_score"),
                    candidate.get("final_status"),
                    candidate.get("strict_pass"),
                    len(candidates),
                    pipelines_per_problem,
                )
            if candidate.get("strict_pass"):
                strict_pass_candidate = candidate
                cancelled_count += await cancel_pending_tasks(pipeline_tasks)
                if progress is not None:
                    progress.log(
                        "stage=pipeline status=early_stop candidate=%d cancelled=%d",
                        candidate.get("attempt_idx"),
                        cancelled_count,
                    )
                break
    except (TimeoutError, asyncio.TimeoutError):
        cancelled_count += await cancel_pending_tasks(pipeline_tasks)
        if progress is not None:
            progress.log(
                "stage=pipeline status=timeout completed=%d cancelled=%d budget_s=%.1f",
                len(candidates),
                cancelled_count,
                float(timeout_s or 0.0),
            )

    return {
        "candidates": candidates,
        "initial_generations": [],
        "failed_attempts": failed_attempts,
        "skipped_generations": [],
        "strict_pass_candidate": strict_pass_candidate,
        "cancelled_count": cancelled_count,
    }


async def run_gold_evaluation(
    question: str,
    candidate: dict[str, Any],
    gold_proof: Optional[str],
    scheduler: ChatScheduler,
    progress: Optional[PipelineProgress] = None,
) -> None:
    if not gold_proof:
        candidate["proof_gold_eval_output"] = None
        candidate["gold_eval_score"] = None
        candidate["gold_eval_status"] = "skipped_no_gold"
        return

    response = await scheduler.call(
        "proof_gold_eval",
        build_deepseek_gold_proof_evaluation_prompt(
            question,
            str(candidate.get("proof_solution") or ""),
            gold_proof,
        ),
        progress=progress,
        detail=f"candidate={candidate.get('attempt_idx', 0)}",
    )
    parsed = parse_verifier_response(response.get("text", ""))
    candidate["proof_gold_eval_output"] = make_output("proof_gold_eval", response, parsed)
    candidate["gold_eval_score"] = parsed.get("score")
    candidate["gold_eval_status"] = "evaluated" if response.get("success") else "error"


async def run_gold_evaluations(
    question: str,
    candidates: list[dict[str, Any]],
    gold_proof: Optional[str],
    scheduler: ChatScheduler,
    progress: Optional[PipelineProgress] = None,
) -> None:
    if not candidates:
        return
    if not gold_proof:
        for candidate in candidates:
            candidate["proof_gold_eval_output"] = None
            candidate["gold_eval_score"] = None
            candidate["gold_eval_status"] = "skipped_no_gold"
        return
    await asyncio.gather(
        *[
            run_gold_evaluation(question, candidate, gold_proof, scheduler, progress)
            for candidate in candidates
        ]
    )


async def select_best_candidate(
    question: str,
    candidates: list[dict[str, Any]],
    scheduler: ChatScheduler,
    cfg: PipelineConfig,
    progress: Optional[PipelineProgress] = None,
) -> tuple[int, dict[str, Any]]:
    if not candidates:
        return 0, {"success": False, "error": "no candidates", "text": ""}

    use_gold_scores = cfg.gold_eval_selection == "use_in_selector" and any(
        candidate.get("gold_eval_score") is not None for candidate in candidates
    )
    if cfg.selector_mode == "score":
        if use_gold_scores:
            selected_idx = max(
                range(len(candidates)),
                key=lambda idx: (
                    gold_score_sort_value(candidates[idx]),
                    score_sort_value(candidates[idx]),
                    len(str(candidates[idx].get("proof_solution") or "")),
                ),
            )
        else:
            selected_idx = fallback_candidate_index(candidates)
        return selected_idx, selector_fallback_output(
            selected_idx,
            "selector_mode_score",
            len(candidates),
        )

    prompt = build_selection_prompt(
        question,
        candidates,
        cfg.selector_max_candidate_chars,
        use_gold_scores,
    )
    response = await scheduler.call(
        "selector",
        prompt,
        temperature=cfg.selection_temperature,
        progress=progress,
        detail=f"candidates={len(candidates)}",
    )
    selected_idx = parse_selected_index(response.get("text", ""), len(candidates))
    if selected_idx is None:
        if use_gold_scores:
            selected_idx = max(
                range(len(candidates)),
                key=lambda idx: (
                    gold_score_sort_value(candidates[idx]),
                    score_sort_value(candidates[idx]),
                    len(str(candidates[idx].get("proof_solution") or "")),
                ),
            )
        else:
            selected_idx = fallback_candidate_index(candidates)
        response["fallback_reason"] = "selector_parse_failed"
    return selected_idx, make_output("selector", response, {"selected_index": selected_idx})


async def process_problem(
    problem_id: Any,
    question: str,
    gold_proof: Optional[str],
    gold_proof_column: Optional[str],
    scheduler: ChatScheduler,
    pipeline_cfg: PipelineConfig,
    pipelines_per_problem: int,
    progress: Optional[PipelineProgress] = None,
) -> dict[str, Any]:
    pipeline_result = await run_streaming_candidates(
        question,
        scheduler,
        pipeline_cfg,
        pipelines_per_problem,
        problem_id=problem_id,
        progress=progress,
    )
    failed_attempts = pipeline_result["failed_attempts"]
    failed_attempts.extend(pipeline_result["skipped_generations"])
    candidates = pipeline_result["candidates"]
    strict_pass_candidate = pipeline_result["strict_pass_candidate"]
    cancelled_count = pipeline_result["cancelled_count"]

    if progress is not None:
        progress.log(
            "stage=pipeline status=complete valid=%d failed=%d cancelled=%d early_stop=%s",
            len(candidates),
            len(failed_attempts),
            cancelled_count,
            strict_pass_candidate is not None,
        )

    await run_gold_evaluations(question, candidates, gold_proof, scheduler, progress)

    if not candidates:
        selected = None
        proof = DEFAULT_FALLBACK_ANSWER
        selected_idx = -1
        selector_output = {"success": False, "error": "all pipeline attempts failed", "text": ""}
        final_score = None
        final_status = "all_attempts_failed"
        gold_eval_score = None
        gold_eval_status = "skipped_no_candidate"
    elif strict_pass_candidate is not None:
        selected = strict_pass_candidate
        selected_idx = selected.get("attempt_idx", candidates.index(selected))
        proof = selected.get("proof_solution", "")
        final_score = selected.get("final_score")
        final_status = selected.get("final_status")
        gold_eval_score = selected.get("gold_eval_score")
        gold_eval_status = selected.get("gold_eval_status")
        selector_output = {
            "success": True,
            "fallback_reason": "strict_pass_early_stop",
            "selected_index": selected_idx,
        }
    else:
        selection_pool, threshold_passed = candidate_selection_pool(candidates, pipeline_cfg)
        if threshold_passed:
            selected_candidate_idx, selector_output = await select_best_candidate(
                question,
                selection_pool,
                scheduler,
                pipeline_cfg,
                progress=progress,
            )
        else:
            selected_candidate_idx = fallback_candidate_index(selection_pool)
            selector_output = selector_fallback_output(
                selected_candidate_idx,
                "no_candidates_above_selector_min_final_score",
                len(selection_pool),
            )
        selected = selection_pool[selected_candidate_idx]
        selected_idx = selected.get("attempt_idx", selected_candidate_idx)
        proof = selected.get("proof_solution", "")
        final_score = selected.get("final_score")
        final_status = selected.get("final_status")
        gold_eval_score = selected.get("gold_eval_score")
        gold_eval_status = selected.get("gold_eval_status")

    print_selected_solution_summary(
        problem_id=problem_id,
        selected_idx=selected_idx,
        proof=proof,
        final_score=final_score,
        final_status=final_status,
        candidate=selected,
    )

    return {
        "id": problem_id,
        "problem": question,
        "proof": proof,
        "prediction": proof,
        "answer": format_submission_answer(proof),
        "selected_pipeline": selected_idx,
        "final_score": final_score,
        "final_status": final_status,
        "gold_eval_score": gold_eval_score,
        "gold_eval_status": gold_eval_status,
        "gold_proof_column": gold_proof_column,
        "candidates": candidates,
        "failed_attempts": failed_attempts,
        "selector_output": selector_output,
        "early_stop": strict_pass_candidate is not None,
        "strict_pass_candidate": (
            strict_pass_candidate.get("attempt_idx") if strict_pass_candidate is not None else None
        ),
    }


def write_debug_row(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def format_submission_answer(value: Any) -> str:
    answer = str(value or "").strip()
    if not answer:
        answer = DEFAULT_FALLBACK_ANSWER
    return answer[:MAX_SUBMISSION_ANSWER_CHARS]


def write_output_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "answer"]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                "id": row.get("id"),
                "answer": format_submission_answer(
                    row.get("answer", row.get("prediction", DEFAULT_FALLBACK_ANSWER))
                ),
            }
            for row in rows
        )


def is_text_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def normalize_column_name(column: str) -> str:
    return column.strip().lower().replace("-", "_").replace(" ", "_")


def find_named_column(columns: list[str], candidates: tuple[str, ...] | set[str]) -> Optional[str]:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        found = normalized.get(normalize_column_name(candidate))
        if found is not None:
            return found
    return None


def detect_question_column(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested question column {requested!r} is not present.")
        return requested
    found = find_named_column(list(df.columns), QUESTION_COLUMN_CANDIDATES)
    if found is None:
        raise ValueError(
            "Could not auto-detect a question column. "
            f"Tried: {', '.join(QUESTION_COLUMN_CANDIDATES)}"
        )
    return found


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_named_column(list(df.columns), ID_COLUMN_CANDIDATES)


def detect_gold_proof_column(
    df: pd.DataFrame,
    question_column: str,
    id_column: Optional[str],
    requested: str,
) -> Optional[str]:
    if requested == "none":
        return None
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested gold solution column {requested!r} is not present.")
        return requested

    excluded = {question_column}
    if id_column:
        excluded.add(id_column)
    for column in df.columns:
        if column in excluded:
            continue
        normalized = column.strip().lower()
        normalized_sep = normalize_column_name(column)
        is_gold_name = (
            normalized in GOLD_PROOF_COLUMN_NAMES
            or normalized_sep in GOLD_PROOF_COLUMN_NAMES
            or ("solution" in normalized_sep and "new_answer" not in normalized_sep)
            or "proof" in normalized_sep
        )
        if not is_gold_name:
            continue
        if any(is_text_value(value) for value in df[column].head(20).tolist()):
            return column
    return None


def resolve_input_paths(input_csv: str) -> list[Path]:
    paths: list[Path] = []
    for raw_part in input_csv.split(","):
        part = raw_part.strip()
        if not part:
            continue
        candidate = Path(part).expanduser()
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.csv")))
            continue
        matches = sorted(Path(match) for match in glob.glob(str(candidate)))
        if matches:
            paths.extend(match for match in matches if match.is_file())
        elif candidate.is_file():
            paths.append(candidate)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError(f"No input CSV files matched {input_csv!r}")
    return deduped


def load_input_records(
    input_csv: str,
    question_column: str,
    gold_solution_column: str,
) -> list[InputRecord]:
    paths = resolve_input_paths(input_csv)
    records: list[InputRecord] = []
    stem_counts: dict[str, int] = {}

    for path in paths:
        df = pd.read_csv(path)
        detected_question_column = detect_question_column(df, question_column)
        id_column = detect_id_column(df)
        detected_gold_column = detect_gold_proof_column(
            df,
            detected_question_column,
            id_column,
            gold_solution_column,
        )
        base_stem = path.stem
        stem_counts[base_stem] = stem_counts.get(base_stem, 0) + 1
        source_stem = base_stem if stem_counts[base_stem] == 1 else f"{base_stem}_{stem_counts[base_stem]}"
        logging.info(
            "Loaded source %s rows=%d question_column=%s id_column=%s gold_column=%s",
            path,
            len(df),
            detected_question_column,
            id_column,
            detected_gold_column,
        )

        for row_index, row in df.iterrows():
            question = row.get(detected_question_column)
            if not is_text_value(question):
                logging.warning("Skipping %s row=%s with empty question", path, row_index)
                continue
            problem_id = row.get(id_column) if id_column else f"{source_stem}-{row_index}"
            if not is_text_value(problem_id):
                problem_id = f"{source_stem}-{row_index}"
            gold_proof = None
            if detected_gold_column:
                raw_gold = row.get(detected_gold_column)
                if is_text_value(raw_gold):
                    gold_proof = str(raw_gold).strip()
            records.append(
                InputRecord(
                    id=problem_id,
                    question=str(question).strip(),
                    source_file=str(path),
                    source_path=path,
                    source_stem=source_stem,
                    row_index=int(row_index),
                    question_column=detected_question_column,
                    gold_proof=gold_proof,
                    gold_proof_column=detected_gold_column if gold_proof else None,
                )
            )
    if not records:
        raise ValueError("No usable input rows were loaded.")
    return records


def resolve_output_paths(records: list[InputRecord], output_csv: Path) -> dict[str, Path]:
    source_stems = list(dict.fromkeys(record.source_stem for record in records))
    if len(source_stems) == 1 and output_csv.suffix.lower() == ".csv":
        return {source_stems[0]: output_csv}
    return {source_stem: output_csv / f"{source_stem}.csv" for source_stem in source_stems}


def resolve_debug_paths(records: list[InputRecord], logdir: Path) -> dict[str, Path]:
    return {
        source_stem: logdir / f"debug_{source_stem}.jsonl"
        for source_stem in dict.fromkeys(record.source_stem for record in records)
    }


def output_row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "answer": format_submission_answer(
            result.get(
                "answer",
                result.get("prediction", result.get("proof", DEFAULT_FALLBACK_ANSWER)),
            )
        ),
        "proof": result.get("proof"),
        "prediction": result.get("prediction"),
        "selected_pipeline": result.get("selected_pipeline"),
        "final_score": result.get("final_score"),
        "final_status": result.get("final_status"),
        "gold_eval_score": result.get("gold_eval_score"),
        "gold_eval_status": result.get("gold_eval_status"),
        "gold_proof_column": result.get("gold_proof_column"),
        "source_file": result.get("source_file"),
        "row_index": result.get("row_index"),
    }


def run_async(
    records: list[InputRecord],
    scheduler: ChatScheduler,
    pipeline_cfg: PipelineConfig,
    pipelines_per_problem: int,
    output_paths: dict[str, Path],
    debug_paths: dict[str, Path],
    logdir: Path,
    benchmark_mode: bool,
    max_concurrent_problems: int,
    verbose: bool,
) -> None:
    output_rows_by_source: dict[str, list[dict[str, Any]]] = {
        source_stem: [] for source_stem in output_paths
    }
    start_time = time.time()
    failures: list[dict[str, Any]] = []

    async def process_record(record: InputRecord) -> dict[str, Any]:
        started = time.time()
        progress = PipelineProgress(verbose, record.id, pipelines_per_problem)
        try:
            result = await process_problem(
                record.id,
                record.question,
                record.gold_proof,
                record.gold_proof_column,
                scheduler,
                pipeline_cfg,
                pipelines_per_problem,
                progress=progress,
            )
        finally:
            progress.close()
        result.update(
            {
                "source_file": record.source_file,
                "source_stem": record.source_stem,
                "row_index": record.row_index,
                "question_column": record.question_column,
                "elapsed_s": time.time() - started,
            }
        )
        return result

    def persist_result(result: dict[str, Any]) -> None:
        source_stem = str(result["source_stem"])
        debug_path = debug_paths[source_stem]
        write_debug_row(debug_path, result)
        output_rows_by_source[source_stem].append(output_row_from_result(result))
        output_rows_by_source[source_stem].sort(key=lambda row: int(row.get("row_index") or 0))
        write_output_csv(output_paths[source_stem], output_rows_by_source[source_stem])

    async def run_all() -> None:
        if benchmark_mode:
            limit = max_concurrent_problems
            if limit <= 0:
                limit = max(1, min(len(records), scheduler.max_concurrent_requests))
            problem_semaphore = asyncio.Semaphore(limit)

            async def bounded(record: InputRecord) -> dict[str, Any]:
                async with problem_semaphore:
                    return await process_record(record)

            tasks = [asyncio.create_task(bounded(record)) for record in records]
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    persist_result(result)
                    logging.info(
                        "Finished id=%s source=%s selected=%s status=%s score=%s gold=%s elapsed=%.1fs",
                        result["id"],
                        result["source_stem"],
                        result["selected_pipeline"],
                        result["final_status"],
                        result["final_score"],
                        result.get("gold_eval_score"),
                        result["elapsed_s"],
                    )
                except Exception as exc:
                    logging.exception("Benchmark task failed")
                    failures.append({"error": repr(exc)})
        else:
            for idx, record in enumerate(records):
                print(
                    f"Processing row {idx+1}/{len(records)} source={record.source_stem} id={record.id}",
                )
                try:
                    result = await process_record(record)
                    persist_result(result)
                    logging.info(
                        "Finished id=%s source=%s selected=%s status=%s score=%s gold=%s elapsed=%.1fs",
                        result["id"],
                        result["source_stem"],
                        result["selected_pipeline"],
                        result["final_status"],
                        result["final_score"],
                        result.get("gold_eval_score"),
                        result["elapsed_s"],
                    )
                except Exception as exc:
                    logging.exception("Sequential task failed source=%s id=%s", record.source_stem, record.id)
                    failures.append({"source_stem": record.source_stem, "id": record.id, "error": repr(exc)})

    asyncio.run(run_all())
    elapsed = time.time() - start_time
    completed = sum(len(rows) for rows in output_rows_by_source.values())
    gold_evaluated = sum(
        1
        for rows in output_rows_by_source.values()
        for row in rows
        if row.get("gold_eval_status") == "evaluated"
    )
    summary = {
        "benchmark_mode": benchmark_mode,
        "records_total": len(records),
        "records_completed": completed,
        "records_failed": len(failures),
        "elapsed_s": elapsed,
        "rows_per_second": completed / elapsed if elapsed > 0 else None,
        "pipelines_per_problem": pipelines_per_problem,
        "candidate_count": completed * pipelines_per_problem,
        "gold_eval_completed_rows": gold_evaluated,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "debug_paths": {key: str(value) for key, value in debug_paths.items()},
        "failures": failures,
    }
    with open(logdir / "benchmark_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, default=str)

def fallback_candidate_index(candidates: list[dict[str, Any]]) -> int:
    return max(
        range(len(candidates)),
        key=lambda idx: (
            score_sort_value(candidates[idx]),
            len(str(candidates[idx].get("proof_solution") or "")),
        ),
    )


def selector_fallback_output(
    selected_idx: int,
    reason: str,
    candidate_count: int,
) -> dict[str, Any]:
    return {
        "stage": "selector",
        "success": True,
        "error": None,
        "text": "",
        "parsed": {"selected_index": selected_idx},
        "finish_reason": None,
        "usage": {},
        "server_url": None,
        "latency_s": None,
        "selected_index": selected_idx,
        "fallback_reason": reason,
        "candidate_count": candidate_count,
    }


def candidate_selection_pool(
    candidates: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], bool]:
    eligible_candidates = [
        candidate
        for candidate in candidates
        if score_sort_value(candidate) > cfg.selector_min_final_score
    ]
    if eligible_candidates:
        return eligible_candidates, True
    return candidates, False


def load_simple_input(input_csv: Path) -> tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(input_csv)
    problem_column = detect_question_column(df, "auto")
    return df, problem_column, detect_id_column(df)


def write_simple_output(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": row.get("id"),
                "answer": format_submission_answer(
                    row.get("answer", row.get("prediction", DEFAULT_FALLBACK_ANSWER))
                ),
            }
            for row in rows
        ]
    ).to_csv(output_csv, index=False)


def load_counting_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    logging.info("Loaded tokenizer for prompt token counting path=%s", model_path)
    return tokenizer


class ProofRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        logdir: Path,
        gpu_group: str,
        tensor_parallel_size: int,
        num_ctx: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_concurrent_requests: int,
        pipelines_per_problem: int,
        proof_only_candidate_count: int,
        skip_self_score_zero: bool,
        wait_for_all_generations_before_verify: bool,
        verify_candidate_limit_while_generating: int,
        verify_request_limit_while_generating: int,
        verify_n: int,
        meta_n: int,
        meta_policy: str,
        strict_pass_meta: bool,
        refine_rounds: int,
        refine_review_n: int,
        min_valid_low: int,
        problem_timeout_seconds: int,
        selection_reserve_seconds: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_new_tokens: int,
        min_p: Optional[float],
        proof_max_new_tokens: int,
        proof_generation_temperatures: list[float],
        thinking_budget_enabled: bool,
        proof_generation_thinking_budgets: list[int],
        thinking_budget_force_text: str,
        verifier_thinking_budget_tokens: int,
        verifier_thinking_budget_force_text: str,
        meta_thinking_budget_tokens: int,
        meta_thinking_budget_force_text: str,
        verifier_max_new_tokens: int,
        meta_max_new_tokens: int,
        selector_max_new_tokens: int,
        selector_max_candidate_chars: int,
        selection_temperature: float,
        selector_mode: str,
        selector_min_final_score: float,
        decomposition_strategy_enabled: bool,
        decomposition_fallback_to_direct: bool,
        decomposition_candidate_count: int,
        decomposition_max_sublemmas: int,
        architect_guided_json: bool,
        proof_architect_max_new_tokens: int,
        sublemma_max_new_tokens: int,
        proof_assembler_max_new_tokens: int,
        vllm_extra_args: str,
        stream_interval: int,
        host: str,
        port: int,
        api_key: str,
        served_model_name: str,
        server_timeout: int,
        no_serve: bool,
        base_url: str,
        mock_llm: bool,
        stream_vllm: bool,
        stream_vllm_server_log: bool,
        verbose: bool,
    ) -> None:
        self.logdir = logdir
        self.gpu_group = gpu_group
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.pipelines_per_problem = max(1, pipelines_per_problem)
        self.problem_timeout_seconds = max(60, problem_timeout_seconds)
        self.selection_reserve_seconds = max(30, selection_reserve_seconds)
        self.api_key = api_key
        self.served_model_name = served_model_name
        self.no_serve = no_serve
        self.base_url = base_url
        self.mock_llm = mock_llm
        self.stream_vllm = stream_vllm
        self.architect_guided_json = bool(architect_guided_json)
        self.verbose = verbose
        self.tokenizer = None if mock_llm else load_counting_tokenizer(model_path)
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._manager: Optional[ServerManager] = None
        self._scheduler: Optional[ChatScheduler] = None
        self._loop = asyncio.new_event_loop()

        extra_args = shlex.split(vllm_extra_args)
        self.vllm_config = VLLMConfig(
            model_path=str(model_path),
            served_model_name=served_model_name,
            host=host,
            port=port,
            api_key=api_key,
            num_ctx=num_ctx,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            vllm_extra_args=shlex.join(extra_args),
            logdir=logdir,
            stream_interval=stream_interval,
            stream_server_logs=stream_vllm_server_log,
        )
        self.sampling = SamplingConfig(
            max_new_tokens=proof_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
            min_p=min_p,
        )
        self.stage_token_limits = {
            "proof_generation": proof_max_new_tokens,
            "proof_architect": proof_architect_max_new_tokens,
            "proof_architect_repair": proof_architect_max_new_tokens,
            "sublemma_proof": sublemma_max_new_tokens,
            "proof_assembly": proof_assembler_max_new_tokens,
            "proof_refine": proof_max_new_tokens,
            "proof_verify": verifier_max_new_tokens,
            "proof_meta_verify": meta_max_new_tokens,
            "selector": selector_max_new_tokens,
        }
        normalized_meta_policy = meta_policy.strip().lower().replace("_", "-")
        if normalized_meta_policy not in {"all-reviews", "low-only"}:
            raise ValueError("--meta_policy must be either 'all-reviews' or 'low-only'")
        normalized_selector_mode = selector_mode.strip().lower()
        if normalized_selector_mode not in {"llm", "score"}:
            raise ValueError("selector_mode must be either 'llm' or 'score'")
        self.pipeline_config = PipelineConfig(
            proof_max_new_tokens=proof_max_new_tokens,
            default_temperature=float(temperature),
            proof_generation_temperatures=[
                float(value) for value in proof_generation_temperatures
            ],
            proof_only_candidate_count=max(0, int(proof_only_candidate_count)),
            skip_self_score_zero=bool(skip_self_score_zero),
            wait_for_all_generations_before_verify=bool(
                wait_for_all_generations_before_verify
            ),
            thinking_budget_enabled=thinking_budget_enabled,
            proof_generation_thinking_budgets=[
                int(value) for value in proof_generation_thinking_budgets
            ],
            thinking_budget_force_text=thinking_budget_force_text,
            verifier_thinking_budget_tokens=max(0, int(verifier_thinking_budget_tokens)),
            verifier_thinking_budget_force_text=verifier_thinking_budget_force_text,
            meta_thinking_budget_tokens=max(0, int(meta_thinking_budget_tokens)),
            meta_thinking_budget_force_text=meta_thinking_budget_force_text,
            verify_candidate_limit_while_generating=max(
                0,
                int(verify_candidate_limit_while_generating),
            ),
            verify_request_limit_while_generating=max(
                0,
                int(verify_request_limit_while_generating),
            ),
            verify_n=max(0, verify_n),
            meta_n=max(0, meta_n),
            meta_policy=normalized_meta_policy,
            strict_pass_meta=strict_pass_meta,
            refine_rounds=max(0, refine_rounds),
            refine_review_n=max(1, refine_review_n),
            min_valid_low=max(1, min_valid_low),
            selector_max_candidate_chars=max(1000, selector_max_candidate_chars),
            selection_temperature=selection_temperature,
            selector_mode=normalized_selector_mode,
            selector_min_final_score=float(selector_min_final_score),
            decomposition_strategy_enabled=bool(decomposition_strategy_enabled),
            decomposition_fallback_to_direct=bool(decomposition_fallback_to_direct),
            decomposition_candidate_count=max(0, int(decomposition_candidate_count)),
            decomposition_max_sublemmas=max(1, int(decomposition_max_sublemmas)),
            proof_architect_max_new_tokens=max(1000, int(proof_architect_max_new_tokens)),
            sublemma_max_new_tokens=max(1000, int(sublemma_max_new_tokens)),
            proof_assembler_max_new_tokens=max(1000, int(proof_assembler_max_new_tokens)),
            gold_eval_selection="report_only",
        )
        self.server_timeout = server_timeout
        atexit.register(self.close)

    def ensure_ready(self) -> None:
        if self._scheduler is not None:
            return
        with self._init_lock:
            if self._scheduler is not None:
                return
            self.logdir.mkdir(parents=True, exist_ok=True)
            manager = ServerManager(
                self.vllm_config,
                ["external"] if self.no_serve or self.mock_llm else [self.gpu_group],
                no_serve=self.no_serve or self.mock_llm,
                base_url=self.base_url,
                server_timeout=self.server_timeout,
            )
            manager.start()
            self._manager = manager
            self._scheduler = ChatScheduler(
                base_urls=manager.urls,
                api_key=self.api_key,
                model=self.served_model_name,
                sampling=self.sampling,
                max_concurrent_requests=self.max_concurrent_requests,
                mock_llm=self.mock_llm,
                stage_max_new_tokens=self.stage_token_limits,
                request_timeout_seconds=float(self.problem_timeout_seconds),
                stream_responses=self.stream_vllm,
                context_length=self.vllm_config.num_ctx,
                tokenizer=self.tokenizer,
                llm_call_logdir=self.logdir / "llm_calls",
                architect_guided_json=self.architect_guided_json,
                stream_interval_tokens=self.vllm_config.stream_interval,
            )

    async def solve_problem(
        self,
        problem_id: Any,
        question: str,
    ) -> dict[str, Any]:
        if self._scheduler is None:
            raise RuntimeError("Inference runtime was not initialized")
        started = time.monotonic()
        progress = PipelineProgress(
            self.verbose,
            problem_id,
            self.pipelines_per_problem,
        )
        progress.log(
            "status=start candidates=%d verify_n=%d refine_rounds=%d question_chars=%d",
            self.pipelines_per_problem,
            self.pipeline_config.verify_n,
            self.pipeline_config.refine_rounds,
            len(question),
        )
        attempt_budget = max(30, self.problem_timeout_seconds - self.selection_reserve_seconds)
        try:
            pipeline_result = await run_streaming_candidates(
                question,
                self._scheduler,
                self.pipeline_config,
                self.pipelines_per_problem,
                problem_id=problem_id,
                progress=progress,
                timeout_s=float(attempt_budget),
            )
            candidates = pipeline_result["candidates"]
            strict_pass_candidate = pipeline_result["strict_pass_candidate"]
            cancelled_count = pipeline_result["cancelled_count"]
            errors = [
                f"attempt={item.get('attempt_idx')} error={item.get('error', item.get('reason'))}"
                for item in (
                    list(pipeline_result["failed_attempts"])
                    + list(pipeline_result["skipped_generations"])
                )
            ]
            failed = len(pipeline_result["failed_attempts"]) + len(
                pipeline_result["skipped_generations"]
            )
            progress.log(
                "stage=pipeline status=complete valid=%d failed=%d cancelled=%d early_stop=%s",
                len(candidates),
                max(0, failed),
                cancelled_count,
                strict_pass_candidate is not None,
            )
            if not candidates:
                remaining_attempt_budget = max(
                    0.0,
                    attempt_budget - (time.monotonic() - started),
                )
                result = {
                    "id": problem_id,
                    "answer": DEFAULT_FALLBACK_ANSWER,
                    "prediction": DEFAULT_FALLBACK_ANSWER,
                    "selected_pipeline": -1,
                    "final_score": None,
                    "final_status": "all_attempts_failed",
                    "candidate_count": 0,
                    "elapsed_s": time.monotonic() - started,
                    "selector_output": {
                        "success": False,
                        "error": (
                            "No proof candidate completed generation/verification/refinement "
                            f"within {remaining_attempt_budget:.1f}s"
                        ),
                    },
                    "errors": errors,
                    "early_stop": False,
                    "strict_pass_candidate": None,
                }
                progress.log(
                    "status=complete selected=-1 final_status=all_attempts_failed "
                    "fallback_answer=%s elapsed=%.1fs",
                    DEFAULT_FALLBACK_ANSWER,
                    result["elapsed_s"],
                )
                print_selected_solution_summary(
                    problem_id=problem_id,
                    selected_idx=-1,
                    proof=DEFAULT_FALLBACK_ANSWER,
                    final_score=None,
                    final_status="all_attempts_failed",
                    candidate=None,
                )
                return result

            selection_pool, threshold_passed = candidate_selection_pool(
                candidates,
                self.pipeline_config,
            )
            selected_index = fallback_candidate_index(selection_pool)
            selector_output: dict[str, Any] = selector_fallback_output(
                selected_index,
                "time_budget",
                len(selection_pool),
            )
            if strict_pass_candidate is not None:
                if strict_pass_candidate in selection_pool:
                    selected_index = selection_pool.index(strict_pass_candidate)
                else:
                    selection_pool = [strict_pass_candidate]
                    selected_index = 0
                selector_output = {
                    "success": True,
                    "fallback_reason": "strict_pass_early_stop",
                    "selected_index": selected_index,
                }
            remaining = self.problem_timeout_seconds - (time.monotonic() - started)
            if strict_pass_candidate is None and not threshold_passed:
                selector_output = selector_fallback_output(
                    selected_index,
                    "no_candidates_above_selector_min_final_score",
                    len(selection_pool),
                )
            elif strict_pass_candidate is None and remaining > 20:
                try:
                    selected_index, selector_output = await asyncio.wait_for(
                        select_best_candidate(
                            question,
                            selection_pool,
                            self._scheduler,
                            self.pipeline_config,
                            progress=progress,
                        ),
                        timeout=remaining,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logging.warning(
                        "Selector timed out for id=%s; using score fallback",
                        problem_id,
                    )

            selected = selection_pool[selected_index]
            prediction = str(selected.get("proof_solution") or "").strip()
            print_selected_solution_summary(
                problem_id=problem_id,
                selected_idx=selected.get("attempt_idx", selected_index),
                proof=prediction,
                final_score=selected.get("final_score"),
                final_status=selected.get("final_status"),
                candidate=selected,
            )
            result = {
                "id": problem_id,
                "answer": format_submission_answer(prediction),
                "prediction": prediction,
                "selected_pipeline": selected.get("attempt_idx", selected_index),
                "final_score": selected.get("final_score"),
                "final_status": selected.get("final_status"),
                "candidate_count": len(candidates),
                "elapsed_s": time.monotonic() - started,
                "selector_output": selector_output,
                "errors": errors,
                "early_stop": strict_pass_candidate is not None,
                "strict_pass_candidate": (
                    strict_pass_candidate.get("attempt_idx")
                    if strict_pass_candidate is not None
                    else None
                ),
            }
            progress.log(
                "status=complete selected=%s final_score=%s final_status=%s elapsed=%.1fs",
                result["selected_pipeline"],
                result["final_score"],
                result["final_status"],
                result["elapsed_s"],
            )
            return result
        finally:
            progress.close()

    def predict(self, row_id: Any, problem: Any) -> Any:
        import polars as pl

        with self._predict_lock:
            problem_id = row_id.item()
            question = str(problem.item())
            self.ensure_ready()
            result = self._loop.run_until_complete(self.solve_problem(problem_id, question))
            return pl.DataFrame(
                {
                    "id": [problem_id],
                    "answer": [format_submission_answer(result.get("answer"))],
                }
            )

    def close(self) -> None:
        if self._manager is not None:
            self._manager.stop()
            self._manager = None
        if not self._loop.is_closed() and not self._loop.is_running():
            self._loop.close()


def run_simple_csv(
    runtime: ProofRuntime,
    input_csv: Path,
    output_csv: Path,
    max_rows: int,
    max_concurrent_problems: int,
) -> None:
    df, problem_column, id_column = load_simple_input(input_csv)
    if max_rows > 0:
        df = df.head(max_rows)
    runtime.ensure_ready()
    output_rows: dict[int, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(max(1, max_concurrent_problems))

    def persist_outputs() -> None:
        rows = [output_rows[index] for index in sorted(output_rows)]
        write_simple_output(output_csv, rows)

    async def process_row(row_position: int, row_idx: int, row: pd.Series) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            problem_id = row[id_column] if id_column is not None else row_idx
            question = str(row[problem_column])
            logging.info(
                "Processing row %d/%d id=%s",
                row_position + 1,
                len(df),
                problem_id,
            )
            started = time.monotonic()
            try:
                result = await runtime.solve_problem(problem_id, question)
                output_row = {
                    "id": problem_id,
                    "answer": format_submission_answer(
                        result.get("answer", result.get("prediction", DEFAULT_FALLBACK_ANSWER))
                    ),
                    "prediction": result.get("prediction", ""),
                    "final_status": result.get("final_status"),
                    "final_score": result.get("final_score"),
                    "selected_pipeline": result.get("selected_pipeline"),
                    "elapsed_s": result.get("elapsed_s"),
                    "error": "",
                }
                write_debug_row(runtime.logdir / "results.jsonl", result)
            except Exception as exc:
                logging.exception("Failed row=%s id=%s", row_idx, problem_id)
                output_row = {
                    "id": problem_id,
                    "answer": DEFAULT_FALLBACK_ANSWER,
                    "prediction": DEFAULT_FALLBACK_ANSWER,
                    "final_status": "error",
                    "final_score": None,
                    "selected_pipeline": None,
                    "elapsed_s": time.monotonic() - started,
                    "error": repr(exc),
                }
            return row_position, output_row

    async def run_rows() -> None:
        tasks = [
            asyncio.create_task(process_row(row_position, int(row_idx), row))
            for row_position, (row_idx, row) in enumerate(df.iterrows())
        ]
        for task in asyncio.as_completed(tasks):
            row_idx, output_row = await task
            output_rows[row_idx] = output_row
            persist_outputs()
            print(
                f"Wrote row {len(output_rows)}/{len(df)} id={output_row['id']} "
                f"status={output_row['final_status']} elapsed={output_row['elapsed_s'] or 0.0}",
            )

    runtime._loop.run_until_complete(run_rows())


def run(cfg: type[CFG] = CFG) -> None:
    setup_logging(cfg.logdir)
    selected_gpus = [item.strip() for item in cfg.gpus.split(",") if item.strip()]
    if not selected_gpus:
        selected_gpus = [str(index) for index in range(max(1, cfg.num_gpus))]
    resolved_tp = cfg.tensor_parallel_size if cfg.tensor_parallel_size > 0 else len(selected_gpus)
    if resolved_tp != len(selected_gpus):
        raise ValueError(
            "--tensor_parallel_size must equal the number of selected GPUs for the single vLLM server"
        )
    logging.info(
        "Inference runtime: stream_vllm=%s stream_vllm_server_log=%s verbose=%s "
        "meta_policy=%s strict_pass_meta=%s max_concurrent_problems=%s "
        "decomposition_enabled=%s decomposition_candidates=%s architect_guided_json=%s "
        "architect_max_tokens=%s",
        cfg.stream_vllm,
        cfg.stream_vllm_server_log,
        cfg.verbose,
        cfg.meta_policy,
        cfg.strict_pass_meta,
        cfg.max_concurrent_problems,
        cfg.decomposition_strategy_enabled,
        cfg.decomposition_candidate_count,
        cfg.architect_guided_json,
        cfg.proof_architect_max_new_tokens,
    )
    runtime = ProofRuntime(
        model_path=cfg.model_path,
        logdir=cfg.logdir,
        gpu_group=",".join(selected_gpus),
        tensor_parallel_size=resolved_tp,
        num_ctx=cfg.num_ctx,
        dtype=cfg.dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        max_concurrent_requests=cfg.max_concurrent_requests,
        pipelines_per_problem=cfg.pipelines_per_problem,
        proof_only_candidate_count=cfg.proof_only_candidate_count,
        skip_self_score_zero=cfg.skip_self_score_zero,
        wait_for_all_generations_before_verify=cfg.wait_for_all_generations_before_verify,
        verify_candidate_limit_while_generating=cfg.verify_candidate_limit_while_generating,
        verify_request_limit_while_generating=cfg.verify_request_limit_while_generating,
        verify_n=cfg.verify_n,
        meta_n=cfg.meta_n,
        meta_policy=cfg.meta_policy,
        strict_pass_meta=cfg.strict_pass_meta,
        refine_rounds=cfg.refine_rounds,
        refine_review_n=cfg.refine_review_n,
        min_valid_low=cfg.min_valid_low,
        problem_timeout_seconds=cfg.problem_timeout_seconds,
        selection_reserve_seconds=cfg.selection_reserve_seconds,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_new_tokens=cfg.min_new_tokens,
        min_p=cfg.min_p,
        proof_max_new_tokens=cfg.max_new_tokens,
        proof_generation_temperatures=list(cfg.proof_generation_temperatures),
        thinking_budget_enabled=cfg.thinking_budget_enabled,
        proof_generation_thinking_budgets=list(cfg.proof_generation_thinking_budgets),
        thinking_budget_force_text=cfg.thinking_budget_force_text,
        verifier_thinking_budget_tokens=cfg.verifier_thinking_budget_tokens,
        verifier_thinking_budget_force_text=cfg.verifier_thinking_budget_force_text,
        meta_thinking_budget_tokens=cfg.meta_thinking_budget_tokens,
        meta_thinking_budget_force_text=cfg.meta_thinking_budget_force_text,
        verifier_max_new_tokens=cfg.verifier_max_new_tokens,
        meta_max_new_tokens=cfg.meta_max_new_tokens,
        selector_max_new_tokens=cfg.selector_max_new_tokens,
        selector_max_candidate_chars=cfg.selector_max_candidate_chars,
        selection_temperature=cfg.selection_temperature,
        selector_mode=cfg.selector_mode,
        selector_min_final_score=cfg.selector_min_final_score,
        decomposition_strategy_enabled=cfg.decomposition_strategy_enabled,
        decomposition_fallback_to_direct=cfg.decomposition_fallback_to_direct,
        decomposition_candidate_count=cfg.decomposition_candidate_count,
        decomposition_max_sublemmas=cfg.decomposition_max_sublemmas,
        architect_guided_json=cfg.architect_guided_json,
        proof_architect_max_new_tokens=cfg.proof_architect_max_new_tokens,
        sublemma_max_new_tokens=cfg.sublemma_max_new_tokens,
        proof_assembler_max_new_tokens=cfg.proof_assembler_max_new_tokens,
        vllm_extra_args=cfg.vllm_extra_args,
        stream_interval=cfg.stream_interval,
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
        served_model_name=cfg.served_model_name,
        server_timeout=cfg.server_timeout,
        no_serve=cfg.no_serve,
        base_url=cfg.base_url,
        mock_llm=cfg.mock_llm,
        stream_vllm=cfg.stream_vllm,
        stream_vllm_server_log=cfg.stream_vllm_server_log,
        verbose=cfg.verbose,
    )
    try:
        run_simple_csv(
            runtime,
            cfg.input_csv,
            cfg.output_csv,
            cfg.max_rows,
            cfg.max_concurrent_problems,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    run()
