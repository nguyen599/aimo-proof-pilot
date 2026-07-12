#!/usr/bin/env python
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf


LOGGER = logging.getLogger(__name__)

try:
    from open_instruct.math_utils import hendrycks_is_equiv
    from open_instruct.math_utils import is_equiv
    from open_instruct.math_utils import last_boxed_only_string
    from open_instruct.math_utils import normalize_final_answer
    from open_instruct.math_utils import remove_boxed
except Exception:  # pragma: no cover - open-instruct is optional in local env tests.
    hendrycks_is_equiv = None
    is_equiv = None
    last_boxed_only_string = None
    normalize_final_answer = None
    remove_boxed = None

EVALUATION_RUBRIC = """Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""

# Forward verifier/meta text without clipping by default. Set these env vars
# only when a deployment needs a hard context-window guard.
MAX_FORWARDED_EVALUATION_CHARS = int(os.environ.get("PROOF_OPD_MAX_FORWARDED_EVALUATION_CHARS", "0") or "0")
MAX_FORWARDED_META_ANALYSIS_CHARS = int(os.environ.get("PROOF_OPD_MAX_FORWARDED_META_ANALYSIS_CHARS", "0") or "0")
MAX_WANDB_TRACE_TEXT_CHARS = 4_000_000
HEADER_SUFFIX_PATTERN = r"(?:\s*//[^\n]*)?\s*$"
BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}")
SELECTED_ID_PATTERN = re.compile(
    r"<selected_id>\s*(?:candidate\s*|[cr])?(\d+)\s*</selected_id>",
    re.IGNORECASE,
)
DEFAULT_MIX_SEED = 34521
CANDIDATE_LOGPROB_FLOOR = -1e30
VERIFIABLE_ANSWER_MATCH_METHOD_IDS = {
    "missing_prediction": 0.0,
    "missing_gold": 0.0,
    "no_match": 1.0,
    "normalized_exact": 2.0,
    "math_equiv": 3.0,
}
SINGLE_TURN_STAGES = frozenset({"prove", "verify", "select", "refine"})

VERIFIABLE_BOXED_INSTRUCTION = """For this problem, the final answer is directly checkable. In your `## Solution` section, include the final answer exactly once in the form `\\boxed{...}` before your `## Self Evaluation` section. The boxed value should contain only the final answer, not explanatory text."""


def build_deepseek_proof_generation_prompt(question: str) -> str:
    return f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
{EVALUATION_RUBRIC}
```

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


def build_deepseek_verifiable_generation_prompt(question: str) -> str:
    return build_deepseek_proof_generation_prompt(f"{question}\n\n{VERIFIABLE_BOXED_INSTRUCTION}")


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


def build_deepseek_meta_verification_prompt(question: str, proof: str, proof_analysis: str) -> str:
    proof_analysis, _ = clip_middle_text(proof_analysis, MAX_FORWARDED_EVALUATION_CHARS)
    return f"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

```
{EVALUATION_RUBRIC}
```

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":
1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.
2. You need to analyze the content of the "solution evaluation" from three aspects: Step Restatement, Defect Analysis, Expression Analysis, and Score Analysis.
3. The most important part is Defect Analysis: check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable.

You should rate the "solution evaluation" with:
- 1 if the evaluation's defect analysis and final score are reasonable.
- 0.5 if the evaluation is generally useful but has minor issues.
- 0 if the evaluation is misleading, ignores major issues, fabricates defects, or gives an unreasonable final score.

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


def build_deepseek_proof_refinement_prompt(question: str, proof: str, proof_analyses: list[str]) -> str:
    analyses = "\n\n".join(f"### Evaluation {idx + 1}\n{analysis}" for idx, analysis in enumerate(proof_analyses))
    return f"""{build_deepseek_proof_generation_prompt(question)}

## Candidate Solution(s) to Refine
Here is a solution sample along with correctness evaluation(s). Provide a better solution by solving issues mentioned in the evaluations, reusing promising ideas from the solution, or both.

### Candidate Solution
{proof}

{analyses}

## Final Instruction
Your final response must follow the format above, including a `## Solution` section followed by a `## Self Evaluation` section."""


def build_deepseek_selector_prompt(problem: str, selection_bundle: list[str]) -> str:
    selection_bundle = "\n\n".join(
        f"### Candidate {idx + 1}\n{candidate}" for idx, candidate in enumerate(selection_bundle)
    )
    return f"""===SYSTEM===
You are choosing the final submission for an olympiad-style proof problem.

You are given several candidate proofs, each with an ID. Do not solve from scratch and do not write any new mathematics. Decide which single candidate is most likely to be a complete and correct proof.

Use this priority:
1. Prefer a proof with no identified fatal gap.
2. Prefer a proof whose key lemmas are explicitly proved.
3. Prefer a proof that exactly proves the problem statement.
4. If all complete-looking proofs have fatal gaps, choose the strongest rigorous partial proof.

Output ONLY the XML format below -- no text outside the tag.
===USER===
Problem:
{problem}

Candidate proofs:
{selection_bundle}

Respond with ONLY the chosen candidate's ID:

<selected_id>ID</selected_id>
"""


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return default if not text else text not in {"0", "false", "no", "off"}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def strip_reasoning_blocks(text: str) -> str:
    visible = re.sub(r"(?is)<think>.*?</think>", "", text or "")
    visible = re.sub(r"(?is)^.*?</think>", "", visible)
    return visible.strip()


def has_closed_thinking(text: str) -> bool:
    return "</think>" in str(text or "").lower()


def coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.strip("$` .,;:")
    if text in {"1", "1.0", "correct"}:
        return 1.0
    if text in {"0.5", ".5", "1/2", "half", "partial"}:
        return 0.5
    if text in {"0", "0.0", "incorrect"}:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    if number in {0.0, 0.5, 1.0}:
        return number
    return None


def extract_boxed_score(text: str) -> float | None:
    scores = [coerce_score(match.group(1)) for match in BOXED_PATTERN.finditer(text or "")]
    scores = [score for score in scores if score is not None]
    if scores:
        return scores[-1]
    fallback = re.findall(r"(?i)\b(?:score|rating)[^0-9]{0,40}(0\.5|1/2|1(?:\.0)?|0(?:\.0)?)\b", text or "")
    if fallback:
        return coerce_score(fallback[-1])
    return None


def extract_verifiable_boxed_answer(text: str) -> str:
    text = str(text or "")
    self_eval_headers = header_matches(strip_reasoning_blocks(text), "## Self Evaluation")
    if self_eval_headers:
        text = strip_reasoning_blocks(text)[: self_eval_headers[0].start()]
    if last_boxed_only_string is not None and remove_boxed is not None:
        try:
            boxed = last_boxed_only_string(text)
            if boxed:
                return str(remove_boxed(boxed)).strip()
        except Exception:
            LOGGER.debug("Open-Instruct boxed answer extraction failed", exc_info=True)
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def normalize_verifiable_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    boxed = extract_verifiable_boxed_answer(text)
    if boxed:
        text = boxed
    if normalize_final_answer is not None:
        math_logger = logging.getLogger("open_instruct.math_utils")
        was_disabled = math_logger.disabled
        math_logger.disabled = True
        try:
            normalized = normalize_final_answer(text)
            if normalized:
                text = normalized
        except Exception:
            LOGGER.debug("Open-Instruct answer normalization failed", exc_info=True)
        finally:
            math_logger.disabled = was_disabled
    text = text.strip().strip("$` .,;:")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace(",", "")
    return text.lower()


def check_verifiable_answer(predicted: Any, gold: Any) -> tuple[bool, str]:
    predicted_text = str(predicted or "").strip()
    gold_text = str(gold or "").strip()
    if not predicted_text:
        return False, "missing_prediction"
    if not gold_text:
        return False, "missing_gold"
    normalized_predicted = normalize_verifiable_answer(predicted_text)
    normalized_gold = normalize_verifiable_answer(gold_text)
    if normalized_predicted == normalized_gold:
        return True, "normalized_exact"
    if not re.search(r"[\\{}^_=]", normalized_predicted + normalized_gold):
        return False, "no_match"
    for equiv_fn in (is_equiv, hendrycks_is_equiv):
        if equiv_fn is None:
            continue
        math_logger = logging.getLogger("open_instruct.math_utils")
        was_disabled = math_logger.disabled
        math_logger.disabled = True
        try:
            if bool(equiv_fn(predicted_text, gold_text)):
                return True, "math_equiv"
        except Exception:
            LOGGER.debug("Math equivalence check failed", exc_info=True)
        finally:
            math_logger.disabled = was_disabled
    return False, "no_match"


def clip_middle_text(text: str, max_chars: int) -> tuple[str, bool]:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n\n...[clipped {len(text) - max_chars} chars]...\n\n" + text[-tail:], True


def clipped_trace_text(text: Any, max_chars: int = MAX_WANDB_TRACE_TEXT_CHARS) -> str:
    clipped, _ = clip_middle_text(str(text or ""), max_chars)
    return clipped


def header_matches(text: str, header: str) -> list[re.Match[str]]:
    header_pattern = re.escape(header.strip()).replace(r"\ ", r"[ \t]+")
    return list(re.finditer(rf"(?im)^[ \t]*{header_pattern}{HEADER_SUFFIX_PATTERN}", text or ""))


def parse_generation_response(text: str) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    closed_thinking = has_closed_thinking(text)
    solution_headers = header_matches(visible, "## Solution")
    evaluation_headers = header_matches(visible, "## Self Evaluation")
    if not solution_headers:
        proof = ""
        self_evaluation = ""
        has_solution = False
        has_self_eval = False
    else:
        solution_header = solution_headers[-1]
        following_eval = next((m for m in evaluation_headers if m.start() > solution_header.end()), None)
        has_solution = True
        has_self_eval = following_eval is not None
        if following_eval is None:
            proof = visible[solution_header.end() :].strip()
            self_evaluation = ""
        else:
            proof = visible[solution_header.end() : following_eval.start()].strip()
            self_evaluation = visible[following_eval.end() :].strip()
    self_score = extract_boxed_score(self_evaluation)
    return {
        "raw_output": text or "",
        "raw_chars": len(text or ""),
        "closed_thinking": closed_thinking,
        "proof": proof,
        "self_evaluation": self_evaluation,
        "self_score": self_score,
        "has_solution_section": has_solution,
        "has_self_evaluation_section": has_self_eval,
        "format_ok": bool(has_solution and proof and has_self_eval and self_score is not None),
    }


def parse_selector_response(text: str) -> dict[str, Any]:
    visible = strip_reasoning_blocks(text)
    matches = SELECTED_ID_PATTERN.findall(visible)
    return {
        "raw_output": text or "",
        "raw_chars": len(text or ""),
        "visible_output": visible,
        "closed_thinking": has_closed_thinking(text),
        "selected_id": int(matches[-1]) if matches else None,
    }


def extract_marked_section(text: str, markers: tuple[str, ...], max_chars: int) -> tuple[str, bool]:
    visible = strip_reasoning_blocks(text)
    lower = visible.lower()
    start = -1
    marker_len = 0
    for marker in markers:
        idx = lower.rfind(marker.lower())
        if idx > start:
            start = idx
            marker_len = len(marker)
    section = visible[start + marker_len :].strip() if start >= 0 else visible.strip()
    score_idx = section.lower().rfind("based on")
    if score_idx > 0:
        section = section[:score_idx].strip()
    return clip_middle_text(section, max_chars)


def parse_verifier_response(text: str) -> dict[str, Any]:
    evaluation, clipped = extract_marked_section(
        text,
        ("Here is my evaluation of the solution:", "Here is my evaluation"),
        MAX_FORWARDED_EVALUATION_CHARS,
    )
    return {
        "raw_output": text or "",
        "evaluation": evaluation,
        "score": extract_boxed_score(text),
        "evaluation_clipped": clipped,
        "raw_chars": len(text or ""),
        "closed_thinking": has_closed_thinking(text),
    }


def verifier_invalid_reason(verifier: dict[str, Any]) -> str:
    if verifier.get("score") is None:
        return "missing_or_invalid_boxed_score"
    if not str(verifier.get("evaluation") or "").strip():
        return "empty_verifier_evaluation"
    return ""


def is_valid_verifier_response(verifier: dict[str, Any]) -> bool:
    return verifier_invalid_reason(verifier) == ""


def parse_meta_verifier_response(text: str) -> dict[str, Any]:
    analysis, clipped = extract_marked_section(
        text,
        ('Here is my analysis of the "solution evaluation":', "Here is my analysis"),
        MAX_FORWARDED_META_ANALYSIS_CHARS,
    )
    return {
        "raw_output": text or "",
        "analysis": analysis,
        "score": extract_boxed_score(text),
        "analysis_clipped": clipped,
        "raw_chars": len(text or ""),
        "closed_thinking": has_closed_thinking(text),
    }


def as_message_list(messages: Any) -> list[Any]:
    if messages is None:
        return []
    if isinstance(messages, list):
        return list(messages)
    return [messages]


def message_signature(message: Any) -> Any:
    if isinstance(message, dict):
        return message
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return repr(message)


def matching_prompt_tail(prompt: list[Any], previous_full: list[Any]) -> list[Any]:
    if len(prompt) < len(previous_full):
        return prompt
    prompt_prefix = [message_signature(message) for message in prompt[: len(previous_full)]]
    previous_signature = [message_signature(message) for message in previous_full]
    if prompt_prefix == previous_signature:
        return prompt[len(previous_full) :]
    return prompt


def render_full_stage_completion(state: Any) -> list[Any]:
    trajectory = state.get("trajectory") if isinstance(state, dict) else None
    if not trajectory:
        return []

    first_step = trajectory[0]
    first_prompt = as_message_list(first_step.get("prompt") if isinstance(first_step, dict) else None)
    first_completion = as_message_list(first_step.get("completion") if isinstance(first_step, dict) else None)
    previous_full = first_prompt + first_completion
    rendered_completion = list(first_completion)

    for step in trajectory[1:]:
        if not isinstance(step, dict):
            continue
        prompt = as_message_list(step.get("prompt"))
        completion = as_message_list(step.get("completion"))
        rendered_completion.extend(matching_prompt_tail(prompt, previous_full))
        rendered_completion.extend(completion)
        previous_full = prompt + completion

    final_response = state.get("final_env_response") if isinstance(state, dict) else None
    rendered_completion.extend(as_message_list(final_response))
    return rendered_completion


def trajectory_step_text(step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    return completion_to_text(step.get("completion"))


def trajectory_step_finish_reason(step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    response = step.get("response")
    message = getattr(response, "message", None)
    finish_reason = getattr(message, "finish_reason", None)
    if finish_reason:
        return str(finish_reason)
    if isinstance(response, dict):
        for key in ("finish_reason", "finishReasons", "stop_reason"):
            if response.get(key):
                return str(response[key])
        response_message = response.get("message")
        if isinstance(response_message, dict):
            for key in ("finish_reason", "finishReasons", "stop_reason"):
                if response_message.get(key):
                    return str(response_message[key])
    return ""


def trajectory_step_is_truncated(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if bool(step.get("is_truncated")):
        return True
    tokens = step.get("tokens")
    if isinstance(tokens, dict) and bool(tokens.get("is_truncated")):
        return True
    response = step.get("response")
    message = getattr(response, "message", None)
    if bool(getattr(message, "is_truncated", False)):
        return True
    finish_reason = trajectory_step_finish_reason(step).lower()
    return finish_reason in {"length", "max_tokens", "token_limit"}


def trajectory_step_token_counts(step: Any) -> dict[str, int]:
    counts = {
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    if not isinstance(step, dict):
        return counts

    response = step.get("response")
    response_message = getattr(response, "message", None)
    if response_message is None and isinstance(response, dict):
        response_message = response.get("message")
    response_tokens = (
        response_message.get("tokens")
        if isinstance(response_message, dict)
        else getattr(response_message, "tokens", None)
    )

    def token_ids(value: Any, key: str) -> Any:
        return value.get(key) if isinstance(value, dict) else getattr(value, key, None)

    raw_prompt_ids = token_ids(response_tokens, "prompt_ids")
    raw_completion_ids = token_ids(response_tokens, "completion_ids")
    if isinstance(raw_prompt_ids, list):
        counts["prompt_tokens"] = len(raw_prompt_ids)
    if isinstance(raw_completion_ids, list):
        counts["generated_tokens"] = len(raw_completion_ids)

    # Parsed trajectory IDs are bounded by max_seq_len, so use them only when
    # the provider did not retain the original response token arrays.
    tokens = step.get("tokens")
    if isinstance(tokens, dict):
        prompt_ids = tokens.get("prompt_ids")
        completion_ids = tokens.get("completion_ids")
        if counts["prompt_tokens"] == 0 and isinstance(prompt_ids, list):
            counts["prompt_tokens"] = len(prompt_ids)
        if counts["generated_tokens"] == 0 and isinstance(completion_ids, list):
            counts["generated_tokens"] = len(completion_ids)

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    def usage_value(key: str) -> int:
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    if counts["prompt_tokens"] == 0:
        counts["prompt_tokens"] = usage_value("prompt_tokens")
    if counts["generated_tokens"] == 0:
        counts["generated_tokens"] = usage_value("completion_tokens")
    counts["reasoning_tokens"] = usage_value("reasoning_tokens")
    counts["total_tokens"] = counts["prompt_tokens"] + counts["generated_tokens"]
    if counts["total_tokens"] == 0:
        counts["total_tokens"] = usage_value("total_tokens")
    return counts


def json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part)
    return str(content)


def message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def message_to_text(message: Any) -> tuple[str | None, str]:
    role = message_field(message, "role")
    content_text = message_content_to_text(message_field(message, "content")).strip()
    reasoning_text = message_content_to_text(message_field(message, "reasoning_content")).strip()
    if str(role or "") == "assistant" and reasoning_text:
        if "</think>" in reasoning_text.lower():
            text = reasoning_text
            if content_text:
                text = f"{text}\n\n{content_text}"
        else:
            text = f"{reasoning_text}</think>{content_text}"
        return role, text.strip()
    return role, content_text


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return str(completion or "")
    assistant_texts: list[str] = []
    all_texts: list[str] = []
    for message in completion:
        role, text = message_to_text(message)
        if not text:
            continue
        all_texts.append(text)
        if role == "assistant":
            assistant_texts.append(text)
    return "\n\n".join(assistant_texts or all_texts)


def log_llm_input(stage: str, prompt: str, *, state: Any | None = None) -> None:
    if not parse_bool(os.environ.get("PROOF_OPD_LOG_LLM_INPUTS"), True):
        return
    max_chars = int(os.environ.get("PROOF_OPD_LOG_LLM_INPUT_MAX_CHARS", "0") or "0")
    shown, clipped = clip_middle_text(prompt, max_chars) if max_chars > 0 else (prompt, False)
    input_payload = state.get("input", {}) if isinstance(state, dict) else {}
    LOGGER.info(
        "Proof-OPD LLM input stage=%s task_id=%s source_index=%s chars=%d clipped=%s\n%s",
        stage,
        input_payload.get("task_id"),
        input_payload.get("source_index"),
        len(prompt),
        clipped,
        shown,
    )


def read_dataset_rows(
    dataset_path: str | Path,
    *,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Proof-OPD dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "train", "rows", "examples"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"JSON dataset must contain a list of rows: {path}")
        return [dict(row) for row in data]

    import pandas as pd

    if suffix == ".parquet":
        selected_columns = columns
        if selected_columns is not None:
            import pyarrow.parquet as pq

            available_columns = set(pq.ParquetFile(path).schema.names)
            selected_columns = [column for column in selected_columns if column in available_columns]
        frame = pd.read_parquet(path, columns=selected_columns)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported Proof-OPD dataset extension: {suffix}")
    return frame.to_dict(orient="records")


def resolve_column(row: dict[str, Any], requested: str, candidates: list[str]) -> str | None:
    if requested and requested != "auto":
        return requested if requested in row else None
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        if candidate in row:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def extract_problem_from_messages(value: Any) -> str:
    messages = json_loads_maybe(value)
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = message_content_to_text(message.get("content")).strip()
        match = re.search(r"(?ims)^##[ \t]+Problem[ \t]*\n(?P<problem>.*)$", text)
        return match.group("problem").strip() if match else text
    return ""


def normalize_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    problem_column: str,
    solution_column: str,
    max_examples: int | None,
    task_type: str = "proof",
    answer_column: str = "auto",
    dataset_label: str = "proof_math",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    is_verifiable = task_type == "verifiable"
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        problem_key = resolve_column(row, problem_column, ["problem", "question", "Problem", "Question"])
        solution_key = resolve_column(row, solution_column, ["solution", "Solution", "answer", "Answer"])
        problem = cell_to_text(row.get(problem_key)) if problem_key else ""
        if not problem and "messages" in row:
            problem = extract_problem_from_messages(row.get("messages"))
        if not problem:
            continue
        solution = cell_to_text(row.get(solution_key)) if solution_key else ""
        answer_key = (
            resolve_column(row, answer_column, ["answer", "final_answer", "gold_answer", "Answer"])
            if is_verifiable
            else None
        )
        gold_answer = cell_to_text(row.get(answer_key)) if is_verifiable and answer_key else ""
        if is_verifiable and not gold_answer:
            continue
        task_id = str(row.get("task_id") or row.get("id") or row.get("problem_id") or index)
        answer = {"problem": problem, "task_type": task_type}
        if solution:
            answer["solution"] = solution
        if gold_answer:
            answer["gold_answer"] = gold_answer
        prompt = (
            build_deepseek_verifiable_generation_prompt(problem)
            if is_verifiable
            else build_deepseek_proof_generation_prompt(problem)
        )
        normalized.append(
            {
                "question": prompt,
                "problem": problem,
                "solution": solution,
                "answer": json.dumps(answer, ensure_ascii=False),
                "dataset": dataset_label,
                "task_id": task_id,
                "source_index": index,
                "task_type": task_type,
                "info": {
                    "stage": "proof_generation",
                    "task_type": task_type,
                    "task_id": task_id,
                    "source_index": index,
                },
            }
        )
        if is_verifiable:
            normalized[-1]["gold_answer"] = gold_answer
            normalized[-1]["info"]["gold_answer"] = gold_answer
        if max_examples is not None and max_examples > 0 and len(normalized) >= max_examples:
            break
    if not normalized:
        raise ValueError("Proof-OPD dataset produced zero usable rows.")
    return normalized


def normalize_single_turn_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int | None,
) -> list[dict[str, Any]]:
    """Normalize pre-rendered OPD stage prompts without modifying their messages."""

    normalized: list[dict[str, Any]] = []
    skipped_stages: dict[str, int] = {}
    invalid_messages = 0
    for source_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        stage = cell_to_text(row.get("stage")).lower()
        if stage not in SINGLE_TURN_STAGES:
            skipped_stages[stage or "<missing>"] = skipped_stages.get(stage or "<missing>", 0) + 1
            continue

        messages = json_loads_maybe(row.get("messages_json"))
        if not isinstance(messages, list) or not messages:
            invalid_messages += 1
            continue
        if any(
            not isinstance(message, dict)
            or not cell_to_text(message.get("role"))
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            invalid_messages += 1
            continue

        # Copy only the chat fields accepted by Verifiers. The source prompts are
        # otherwise preserved byte-for-byte and receive no additional template.
        prompt = [
            {"role": str(message["role"]), "content": message["content"]}
            for message in messages
        ]
        problem = cell_to_text(row.get("problem"))
        problem_id = cell_to_text(row.get("problem_id"))
        run_id = cell_to_text(row.get("run_id"))
        candidate_id = cell_to_text(row.get("candidate_id"))
        verifier_idx = cell_to_text(row.get("verifier_idx"))
        identity_parts = [
            part
            for part in (run_id, problem_id, stage, candidate_id, verifier_idx, str(source_index))
            if part
        ]
        task_id = ":".join(identity_parts)
        info = {
            "stage": stage,
            "task_type": "opd_single_turn",
            "task_id": task_id,
            "source_index": source_index,
            "pipeline": cell_to_text(row.get("pipeline")),
            "run_id": run_id,
            "problem_id": problem_id,
            "candidate_id": candidate_id,
            "verifier_idx": verifier_idx,
        }
        answer = {
            "problem": problem,
            "stage": stage,
            "task_type": "opd_single_turn",
        }
        normalized.append(
            {
                "prompt": prompt,
                "problem": problem,
                "answer": json.dumps(answer, ensure_ascii=False),
                "dataset": "proof_opd_single_turn",
                "task_id": task_id,
                "source_index": source_index,
                "task_type": "opd_single_turn",
                "stage": stage,
                "info": info,
            }
        )
        if max_examples is not None and max_examples > 0 and len(normalized) >= max_examples:
            break

    if not normalized:
        raise ValueError(
            "Single-turn Proof-OPD dataset produced zero usable rows; expected "
            "messages_json and stage in prove, verify, select, or refine."
        )
    LOGGER.info(
        "Normalized single-turn Proof-OPD rows: usable=%d invalid_messages=%d skipped_stages=%s",
        len(normalized),
        invalid_messages,
        skipped_stages,
    )
    return normalized


def _clone_dataset_row(row: dict[str, Any], repeat_index: int = 0) -> dict[str, Any]:
    cloned = dict(row)
    info = dict(json_loads_maybe(cloned.get("info")) or {})
    if repeat_index > 0:
        original_task_id = str(cloned.get("task_id") or info.get("task_id") or "")
        repeated_task_id = f"{original_task_id}:repeat{repeat_index}"
        cloned["task_id"] = repeated_task_id
        cloned["source_repeat"] = repeat_index
        info["original_task_id"] = original_task_id
        info["task_id"] = repeated_task_id
        info["source_repeat"] = repeat_index
    cloned["info"] = info
    return cloned


def _take_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not rows:
        raise ValueError("Cannot take rows from an empty dataset.")
    selected: list[dict[str, Any]] = []
    for idx in range(count):
        source = rows[idx % len(rows)]
        selected.append(_clone_dataset_row(source, idx // len(rows)))
    return selected


def normalize_boxed_training_rows(
    rows: list[dict[str, Any]],
    *,
    problem_column: str,
    solution_column: str,
    answer_column: str = "auto",
    max_examples: int | None = None,
    dataset_label: str = "proof_math",
) -> list[dict[str, Any]]:
    """Normalize answerable rows for training without preserving answer labels.

    These rows are used only to expose boxed-answer prompting patterns in the
    train mix. Boxed-answer accuracy belongs to the separate eval environment,
    so the train rows are plain proof tasks and intentionally omit gold answers.
    """

    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        problem_key = resolve_column(row, problem_column, ["problem", "question", "Problem", "Question"])
        answer_key = resolve_column(row, answer_column, ["answer", "final_answer", "gold_answer", "Answer"])
        solution_key = resolve_column(row, solution_column, ["solution", "Solution", "answer", "Answer"])
        problem = cell_to_text(row.get(problem_key)) if problem_key else ""
        if not problem and "messages" in row:
            problem = extract_problem_from_messages(row.get("messages"))
        if not problem:
            continue
        if not answer_key or not cell_to_text(row.get(answer_key)):
            continue
        solution = cell_to_text(row.get(solution_key)) if solution_key else ""
        task_id = str(row.get("task_id") or row.get("id") or row.get("problem_id") or index)
        answer = {"problem": problem, "task_type": "proof"}
        if solution:
            answer["solution"] = solution
        normalized.append(
            {
                "question": build_deepseek_verifiable_generation_prompt(problem),
                "problem": problem,
                "solution": solution,
                "answer": json.dumps(answer, ensure_ascii=False),
                "dataset": dataset_label,
                "task_id": task_id,
                "source_index": index,
                "task_type": "proof",
                "info": {
                    "stage": "proof_generation",
                    "task_type": "proof",
                    "task_id": task_id,
                    "source_index": index,
                    "boxed_training_prompt": True,
                },
            }
        )
        if max_examples is not None and max_examples > 0 and len(normalized) >= max_examples:
            break
    if not normalized:
        raise ValueError("Boxed training dataset produced zero usable rows.")
    return normalized


def normalize_mixed_dataset_rows(
    proof_rows: list[dict[str, Any]],
    *,
    problem_column: str,
    solution_column: str,
    max_examples: int | None,
    verifiable_rows: list[dict[str, Any]] | None = None,
    verifiable_fraction: float = 0.2,
    verifiable_answer_column: str = "auto",
    mix_seed: int = DEFAULT_MIX_SEED,
) -> list[dict[str, Any]]:
    proof_normalized = normalize_dataset_rows(
        proof_rows,
        problem_column=problem_column,
        solution_column=solution_column,
        max_examples=None,
        task_type="proof",
        dataset_label="proof_math",
    )
    if not verifiable_rows or verifiable_fraction <= 0:
        count = max_examples if max_examples is not None and max_examples > 0 else len(proof_normalized)
        return [_clone_dataset_row(row) for row in proof_normalized[:count]]

    boxed_train_normalized = normalize_boxed_training_rows(
        verifiable_rows,
        problem_column=problem_column,
        solution_column=solution_column,
        answer_column=verifiable_answer_column,
        dataset_label="proof_math",
    )
    final_count = max_examples if max_examples is not None and max_examples > 0 else len(proof_normalized)
    fraction = max(0.0, min(1.0, float(verifiable_fraction)))
    verifiable_count = int(round(final_count * fraction))
    if fraction > 0.0 and verifiable_count == 0 and final_count > 0:
        verifiable_count = 1
    verifiable_count = min(final_count, verifiable_count)
    proof_count = final_count - verifiable_count
    verifiable_train_rows = _take_rows(boxed_train_normalized, verifiable_count)
    mixed = _take_rows(proof_normalized, proof_count) + verifiable_train_rows
    random.Random(int(mix_seed)).shuffle(mixed)
    return mixed


class ProofOPDRubric(vf.Rubric):
    def __init__(self) -> None:
        super().__init__()
        self.add_reward_func(self.proof_opd_reward)
        self.add_metric(self.proof_opd_format_score)
        self.add_metric(self.proof_opd_proof_score)
        self.add_metric(self.proof_opd_meta_score)
        self.add_metric(self.proof_opd_round_index)
        self.add_metric(self.proof_opd_selector_valid)

    async def proof_opd_reward(self, state: Any, **_: Any) -> float:
        payload = state.get("proof_opd_reward_payload") if isinstance(state, dict) else None
        return float((payload or {}).get("reward", 0.0) or 0.0)

    async def proof_opd_format_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "format_score")

    async def proof_opd_proof_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "proof_score")

    async def proof_opd_meta_score(self, state: Any, **_: Any) -> float:
        return self._metric(state, "meta_score")

    async def proof_opd_round_index(self, state: Any, **_: Any) -> float:
        return self._metric(state, "selected_round_index", -1.0)

    async def proof_opd_selector_valid(self, state: Any, **_: Any) -> float:
        return self._metric(state, "selector_valid")

    @staticmethod
    def _metric(state: Any, key: str, default: float = 0.0) -> float:
        payload = state.get("proof_opd_reward_payload") if isinstance(state, dict) else None
        value = (payload or {}).get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class ProofOPDSingleTurnRubric(ProofOPDRubric):
    async def score_rollout(self, state: vf.State) -> None:
        await super().score_rollout(state)
        info = dict(state.get("info") or {})
        stage = str(info.get("stage") or state.get("proof_opd_stage") or "single").lower()
        if stage not in SINGLE_TURN_STAGES:
            stage = "single"
        trajectory = state.get("trajectory") or []
        counts = trajectory_step_token_counts(trajectory[-1] if trajectory else None)
        metrics = dict(state.get("metrics") or {})
        for name, value in counts.items():
            metrics[f"proof_opd_policy_{name}"] = float(value)
            metrics[f"proof_opd_{stage}_policy_{name}"] = float(value)
        state["metrics"] = metrics


class ProofOPDSingleTurnEnv(vf.SingleTurnEnv):
    """Run one pre-rendered OPD stage prompt and terminate immediately."""

    async def setup_state(self, state: vf.State) -> None:
        info = dict(state.get("info") or {})
        state["proof_opd_stage"] = str(info.get("stage") or state.get("stage") or "single")

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        messages = await super().get_prompt_messages(state)
        stage = str(state.get("proof_opd_stage") or "single")
        log_llm_input(f"single_{stage}", completion_to_text(messages), state=state)
        return messages

    @vf.cleanup(priority=-10)
    async def attach_wandb_trace(self, state: vf.State) -> None:
        info = dict(state.get("info") or {})
        stage = str(state.get("proof_opd_stage") or info.get("stage") or "single")
        raw_output, is_truncated, finish_reason = "", False, ""
        token_counts = trajectory_step_token_counts(None)
        trajectory = state.get("trajectory") or []
        if trajectory:
            raw_output = trajectory_step_text(trajectory[-1])
            is_truncated = trajectory_step_is_truncated(trajectory[-1])
            finish_reason = trajectory_step_finish_reason(trajectory[-1])
            token_counts = trajectory_step_token_counts(trajectory[-1])
        info["proof_opd_trace"] = {
            "task_id": info.get("task_id") or state.get("task_id"),
            "source_index": info.get("source_index") or state.get("source_index"),
            "task_type": "opd_single_turn",
            "stage": stage,
            "problem": clipped_trace_text(state.get("problem", "")),
            "reward": float(state.get("reward") or 0.0),
            "finish_reason": finish_reason,
            "is_truncated": is_truncated,
            **token_counts,
            "stage_records": [
                {
                    "stage": stage,
                    "round_index": 0,
                    "verify_index": 0,
                    "raw_chars": len(raw_output),
                    "raw_output_excerpt": clipped_trace_text(raw_output),
                    "is_truncated": is_truncated,
                    "finish_reason": finish_reason,
                    "source": "single_turn_dataset",
                    **token_counts,
                }
            ],
            "raw_output_excerpt": clipped_trace_text(raw_output),
        }
        state["info"] = info


class ProofOPDEnv(vf.MultiTurnEnv):
    _PRIORITY_STAGE_OFFSETS = {
        "selector": -5_000_000_000,
        "meta": -4_000_000_000,
        "verifier": -3_000_000_000,
        "refine": -2_000_000_000,
        "proof": 0,
    }

    def __init__(
        self,
        *,
        refine_rounds: int = 1,
        num_verifiers: int = 4,
        refine_review_n: int = 2,
        enable_meta_verification: bool = True,
        partial_format_score: float = 0.7,
        refine_early_stop_reward: float = 0.95,
        selector_top_k: int = 3,
        require_closed_think: bool | str = True,
        candidate_gate_enabled: bool | str = False,
        candidate_continue_count: int = 4,
        **kwargs: Any,
    ) -> None:
        self.refine_rounds = max(0, int(refine_rounds))
        self.num_verifiers = max(1, int(num_verifiers))
        self.refine_review_n = max(1, int(refine_review_n))
        self.enable_meta_verification = bool(enable_meta_verification)
        self.partial_format_score = clamp01(float(partial_format_score))
        self.refine_early_stop_reward = clamp01(float(refine_early_stop_reward))
        self.selector_top_k = max(1, int(selector_top_k))
        self.require_closed_think = parse_bool(require_closed_think, True)
        self.candidate_gate_enabled = parse_bool(candidate_gate_enabled, False)
        self.candidate_continue_count = max(1, int(candidate_continue_count))
        self._candidate_gate_groups: dict[str, dict[str, Any]] = {}
        self._question_sequence = 0
        turns_per_round = 1 + self.num_verifiers * (2 if self.enable_meta_verification else 1)
        super().__init__(max_turns=turns_per_round * (self.refine_rounds + 1) + 2, **kwargs)

    @property
    def requires_group_rollouts(self) -> bool:
        return self.candidate_gate_enabled or super().requires_group_rollouts

    async def _run_group_states(
        self,
        group_inputs: list[Any],
        client: Any,
        model: str,
        sampling_args: Any,
    ) -> list[Any]:
        if not self.candidate_gate_enabled or len(group_inputs) <= 1:
            return await super()._run_group_states(group_inputs, client, model, sampling_args)

        group_id = uuid.uuid4().hex
        question_sequence = self._question_sequence
        self._question_sequence += 1
        group = {
            "condition": asyncio.Condition(),
            "expected": len(group_inputs),
            "continue_count": min(self.candidate_continue_count, len(group_inputs)),
            "records": {},
            "selected": None,
            "ranks": {},
            "selector_records": {},
            "selector_candidates": None,
            "selector_leader": None,
        }
        self._candidate_gate_groups[group_id] = group
        prepared_inputs: list[Any] = []
        for candidate_index, raw_input in enumerate(group_inputs):
            candidate_input = dict(raw_input)
            info = dict(json_loads_maybe(candidate_input.get("info")) or {})
            info.update(
                {
                    "candidate_gate_enabled": True,
                    "candidate_group_id": group_id,
                    "candidate_group_size": len(group_inputs),
                    "candidate_index": candidate_index,
                    "candidate_continue_count": group["continue_count"],
                    "question_sequence": question_sequence,
                }
            )
            candidate_input["info"] = info
            prepared_inputs.append(candidate_input)

        LOGGER.info(
            "Proof-OPD candidate group started: group=%s candidates=%d continue=%d",
            group_id,
            len(group_inputs),
            group["continue_count"],
        )
        try:
            return await super()._run_group_states(
                prepared_inputs,
                client,
                model,
                sampling_args,
            )
        finally:
            self._candidate_gate_groups.pop(group_id, None)

    async def setup_state(self, state: vf.State) -> None:
        state["proof_opd_stage"] = "proof"
        state["proof_opd_current_round"] = 0
        state["proof_opd_rounds"] = []
        state["proof_opd_stage_records"] = []
        state["proof_opd_verify_index"] = 0
        state["proof_opd_verifier_results"] = []
        state["proof_opd_pending_verifier_result"] = None
        state["proof_opd_selector_candidates"] = []
        state["proof_opd_selector"] = None
        state["proof_opd_reward_payload"] = None
        info = self._input_info(state)
        state["proof_opd_question_sequence"] = int(
            info.get("question_sequence", self._question_sequence)
        )

    async def get_model_response(
        self,
        state: vf.State,
        prompt: vf.Messages,
        client: Any = None,
        model: str | None = None,
        tool_defs: Any = None,
        sampling_args: Any = None,
    ) -> Any:
        request_args = dict(sampling_args or state.get("sampling_args") or {})
        extra_body = dict(request_args.get("extra_body") or {})
        stage = str(state.get("proof_opd_stage") or "proof")
        question_sequence = int(state.get("proof_opd_question_sequence", 0))
        priority = self._request_priority(state)
        sampling_extra = dict(extra_body.get("extra_args") or {})
        sampling_extra["prime_priority"] = priority
        extra_body["extra_args"] = sampling_extra
        request_args["extra_body"] = extra_body
        LOGGER.info(
            "Proof-OPD scheduling request stage=%s question_sequence=%d priority=%d",
            stage,
            question_sequence,
            priority,
        )
        return await super().get_model_response(
            state,
            prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=request_args,
        )

    def _request_priority(self, state: vf.State) -> int:
        stage = str(state.get("proof_opd_stage") or "proof")
        question_sequence = int(state.get("proof_opd_question_sequence", 0))
        return self._PRIORITY_STAGE_OFFSETS.get(stage, 0) + question_sequence

    def _input_value(self, state: vf.State, key: str) -> str:
        value = state.get("input", {}).get(key)
        return str(value or "").strip()

    def _problem(self, state: vf.State) -> str:
        return self._input_value(state, "problem") or self._input_value(state, "question")

    def _input_info(self, state: vf.State) -> dict[str, Any]:
        info = state.get("input", {}).get("info") if isinstance(state, dict) else None
        info = json_loads_maybe(info)
        return dict(info) if isinstance(info, dict) else {}

    def _update_input_info(self, state: vf.State, **updates: Any) -> None:
        input_row = state.get("input")
        if not isinstance(input_row, dict):
            return
        info = dict(json_loads_maybe(input_row.get("info")) or {})
        info.update(updates)
        input_row["info"] = info

    @staticmethod
    def _mean_completion_logprob(state: vf.State) -> float:
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return CANDIDATE_LOGPROB_FLOOR
        step = trajectory[-1]
        tokens = step.get("tokens") if isinstance(step, dict) else None
        raw_logprobs = tokens.get("completion_logprobs") if isinstance(tokens, dict) else None
        values: list[float] = []
        for raw_value in raw_logprobs or []:
            if isinstance(raw_value, dict):
                raw_value = raw_value.get("logprob")
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                continue
        return sum(values) / len(values) if values else CANDIDATE_LOGPROB_FLOOR

    @staticmethod
    def _candidate_gate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(record.get("valid")),
            float(record.get("format_score", 0.0) or 0.0),
            float(record.get("self_score", -1.0) if record.get("self_score") is not None else -1.0),
            float(record.get("mean_completion_logprob", CANDIDATE_LOGPROB_FLOOR)),
            -int(record.get("candidate_index", 0)),
        )

    def _finalize_candidate_gate_locked(self, group: dict[str, Any]) -> None:
        if group.get("selected") is not None:
            return
        ranked = sorted(
            group["records"].values(),
            key=self._candidate_gate_sort_key,
            reverse=True,
        )
        eligible = [record for record in ranked if record.get("valid")]
        selected = {
            int(record["candidate_index"])
            for record in eligible[: int(group["continue_count"])]
        }
        group["selected"] = selected
        group["ranks"] = {
            int(record["candidate_index"]): rank
            for rank, record in enumerate(ranked, start=1)
        }

    async def _submit_candidate_gate_record(
        self,
        state: vf.State,
        record: dict[str, Any],
        *,
        wait_for_selection: bool,
    ) -> bool:
        info = self._input_info(state)
        group_id = str(info.get("candidate_group_id") or "")
        if not self.candidate_gate_enabled or not group_id:
            return True
        group = self._candidate_gate_groups.get(group_id)
        if group is None:
            LOGGER.warning("Proof-OPD candidate group missing: group=%s", group_id)
            return True

        candidate_index = int(info.get("candidate_index", -1))
        record = dict(record)
        record["candidate_index"] = candidate_index
        condition: asyncio.Condition = group["condition"]
        async with condition:
            group["records"].setdefault(candidate_index, record)
            state["proof_opd_candidate_gate_submitted"] = True
            if len(group["records"]) >= int(group["expected"]):
                self._finalize_candidate_gate_locked(group)
                condition.notify_all()
            while wait_for_selection and group.get("selected") is None:
                await condition.wait()

            if group.get("selected") is None:
                return False
            selected = candidate_index in group["selected"]
            rank = int(group["ranks"].get(candidate_index, 0))
            self._update_input_info(
                state,
                candidate_selected=selected,
                candidate_rank=rank,
                candidate_selection_valid=bool(record.get("valid")),
                candidate_selection_format_score=float(record.get("format_score", 0.0) or 0.0),
                candidate_selection_self_score=record.get("self_score"),
                candidate_selection_mean_logprob=record.get("mean_completion_logprob"),
            )
            LOGGER.info(
                "Proof-OPD candidate gate: group=%s candidate=%d rank=%d selected=%s "
                "valid=%s format=%.3f self_score=%s mean_logprob=%s",
                group_id,
                candidate_index,
                rank,
                selected,
                bool(record.get("valid")),
                float(record.get("format_score", 0.0) or 0.0),
                record.get("self_score"),
                record.get("mean_completion_logprob"),
            )
            return selected

    async def _candidate_gate_after_proof(
        self,
        state: vf.State,
        parsed: dict[str, Any],
        invalid_reason: str,
    ) -> bool:
        return await self._submit_candidate_gate_record(
            state,
            {
                "valid": not bool(invalid_reason),
                "format_score": self._format_score(parsed),
                "self_score": parsed.get("self_score"),
                "mean_completion_logprob": self._mean_completion_logprob(state),
                "proof_chars": len(str(parsed.get("proof") or "")),
                "invalid_reason": invalid_reason,
            },
            wait_for_selection=True,
        )

    def _proof_only_candidate_payload(
        self,
        state: vf.State,
        parsed: dict[str, Any],
        *,
        is_truncated: bool,
        finish_reason: str,
    ) -> dict[str, Any]:
        info = self._input_info(state)
        return {
            "round_index": 0,
            "selected_round_index": 0,
            "reward": 0.0,
            "format_score": self._format_score(parsed),
            "format_ok": parsed.get("format_ok", False),
            "proof_score": None,
            "meta_score": None,
            "num_verifiers": self.num_verifiers,
            "num_verifier_results": 0,
            "valid_verifier_count": 0,
            "valid_meta_count": 0,
            "verifier_reward_terms": [],
            "verifier_meta_reward": None,
            "verifier_results": [],
            "self_score": parsed.get("self_score"),
            "proof_chars": len(str(parsed.get("proof") or "")),
            "self_evaluation_chars": len(str(parsed.get("self_evaluation") or "")),
            "closed_thinking": parsed.get("closed_thinking", False),
            "is_truncated": is_truncated,
            "finish_reason": finish_reason,
            "reason": "candidate_gate_proof_only",
            "generation_raw_output": parsed.get("raw_output", ""),
            "proof": parsed.get("proof", ""),
            "self_evaluation": parsed.get("self_evaluation", ""),
            "candidate_selected": False,
            "candidate_rank": info.get("candidate_rank"),
            "candidate_index": info.get("candidate_index"),
            "candidate_group_size": info.get("candidate_group_size"),
            "stage_records": list(state.get("proof_opd_stage_records") or []),
        }

    @vf.cleanup(priority=50)
    async def release_candidate_gate_on_failed_rollout(self, state: vf.State) -> None:
        if state.get("proof_opd_candidate_gate_submitted"):
            return
        info = self._input_info(state)
        if not info.get("candidate_group_id"):
            return
        await self._submit_candidate_gate_record(
            state,
            {
                "valid": False,
                "format_score": 0.0,
                "self_score": None,
                "mean_completion_logprob": CANDIDATE_LOGPROB_FLOOR,
                "proof_chars": 0,
                "invalid_reason": "rollout_failed_before_candidate_gate",
            },
            wait_for_selection=False,
        )

    def _finalize_group_selector_locked(self, group: dict[str, Any]) -> None:
        if group.get("selector_candidates") is not None:
            return

        ranked: list[dict[str, Any]] = []
        for record in group.get("selector_records", {}).values():
            valid_rounds = [
                dict(payload)
                for payload in record.get("rounds", [])
                if str(payload.get("proof") or "").strip()
                and float(payload.get("format_score", 0.0) or 0.0) > 0.0
            ]
            if not valid_rounds:
                continue
            best_payload = max(
                valid_rounds,
                key=lambda payload: (
                    self._selector_score(payload),
                    float(payload.get("format_score", 0.0) or 0.0),
                    float(payload.get("verifier_meta_reward", 0.0) or 0.0),
                    int(payload.get("round_index", 0)),
                ),
            )
            ranked.append(
                {
                    "source_candidate_index": int(record["candidate_index"]),
                    "source_candidate_rank": int(record.get("candidate_rank", 0)),
                    "round_index": int(best_payload.get("round_index", 0)),
                    "preselection_score": self._selector_score(best_payload),
                    "format_score": clamp01(float(best_payload.get("format_score", 0.0) or 0.0)),
                    "verifier_meta_reward": clamp01(
                        float(best_payload.get("verifier_meta_reward", 0.0) or 0.0)
                    ),
                    "proof": str(best_payload.get("proof") or "").strip(),
                    "_payload": best_payload,
                }
            )

        ranked.sort(
            key=lambda candidate: (
                float(candidate["preselection_score"]),
                float(candidate["format_score"]),
                float(candidate["verifier_meta_reward"]),
                -int(candidate["source_candidate_rank"]),
                -int(candidate["source_candidate_index"]),
            ),
            reverse=True,
        )
        candidates = ranked[: self.selector_top_k]
        for candidate_id, candidate in enumerate(candidates, start=1):
            candidate["candidate_id"] = candidate_id

        selected = set(group.get("selected") or set())
        group["selector_candidates"] = candidates
        group["selector_leader"] = (
            int(candidates[0]["source_candidate_index"])
            if candidates
            else (min(selected) if selected else None)
        )

    async def _submit_group_selector_record(
        self,
        state: vf.State,
        *,
        failed: bool = False,
        wait_for_group: bool = True,
    ) -> tuple[bool, list[dict[str, Any]]] | None:
        info = self._input_info(state)
        group_id = str(info.get("candidate_group_id") or "")
        if not self.candidate_gate_enabled or not group_id or not info.get("candidate_selected"):
            return None
        group = self._candidate_gate_groups.get(group_id)
        if group is None:
            LOGGER.warning("Proof-OPD selector group missing: group=%s", group_id)
            return None

        candidate_index = int(info.get("candidate_index", -1))
        condition: asyncio.Condition = group["condition"]
        async with condition:
            selector_records = group.setdefault("selector_records", {})
            selector_records.setdefault(
                candidate_index,
                {
                    "candidate_index": candidate_index,
                    "candidate_rank": int(info.get("candidate_rank") or 0),
                    "rounds": [] if failed else [dict(payload) for payload in state.get("proof_opd_rounds") or []],
                    "failed": bool(failed),
                },
            )
            state["proof_opd_group_selector_submitted"] = True
            expected = len(group.get("selected") or set())
            if len(selector_records) >= expected:
                self._finalize_group_selector_locked(group)
                condition.notify_all()
            while wait_for_group and group.get("selector_candidates") is None:
                await condition.wait()

            if group.get("selector_candidates") is None:
                return None
            candidates = list(group["selector_candidates"])
            is_leader = candidate_index == group.get("selector_leader")
            LOGGER.info(
                "Proof-OPD group selector ready: group=%s candidate=%d leader=%s pool=%d",
                group_id,
                candidate_index,
                is_leader,
                len(candidates),
            )
            return is_leader, candidates

    @vf.cleanup(priority=40)
    async def release_group_selector_on_failed_rollout(self, state: vf.State) -> None:
        if state.get("proof_opd_group_selector_submitted"):
            return
        info = self._input_info(state)
        if not info.get("candidate_group_id") or not info.get("candidate_selected"):
            return
        await self._submit_group_selector_record(state, failed=True, wait_for_group=False)

    def _effective_meta_score(self, result: dict[str, Any]) -> float:
        if not self.enable_meta_verification:
            return 1.0
        if not result.get("verifier_valid"):
            return 0.0
        if result.get("meta_invalid_reason") == "skipped_perfect_verifier":
            return 1.0
        if result.get("meta_valid"):
            return clamp01(float(result.get("meta_score") or 0.0))
        return 0.5

    def _effective_verifier_score(self, result: dict[str, Any]) -> float:
        if not result.get("verifier_valid"):
            return 0.0
        return clamp01(float(result.get("proof_score") or 0.0))

    def _verifier_reward_term(self, result: dict[str, Any]) -> float:
        return self._effective_verifier_score(result) * self._effective_meta_score(result)

    def _summarize_verifier_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        terms = [self._verifier_reward_term(result) for result in results]
        valid_verifiers = [result for result in results if result.get("verifier_valid")]
        valid_meta = [result for result in results if result.get("meta_valid")]
        return {
            "num_verifiers": self.num_verifiers,
            "num_verifier_results": len(results),
            "valid_verifier_count": len(valid_verifiers),
            "valid_meta_count": len(valid_meta),
            "verifier_reward_terms": terms,
            "proof_score": sum(self._effective_verifier_score(result) for result in results) / max(len(results), 1),
            "meta_score": sum(self._effective_meta_score(result) for result in results) / max(len(results), 1),
            "verifier_meta_reward": sum(terms) / max(len(terms), 1),
        }

    def _select_refinement_reviews(self, payload: dict[str, Any]) -> list[str]:
        results = list(payload.get("verifier_results") or [])
        valid = [
            result
            for result in results
            if result.get("verifier_valid") and str(result.get("verifier_evaluation") or "").strip()
        ]
        valid.sort(
            key=lambda result: (
                self._effective_verifier_score(result),
                -self._effective_meta_score(result),
                int(result.get("verify_index", 0)),
            )
        )
        selected = valid[: self.refine_review_n]
        analyses: list[str] = []
        for result in selected:
            if result.get("meta_invalid_reason") == "skipped_perfect_verifier":
                meta_score_label = "not run (neutral 1.000)"
            else:
                meta_score_label = f"{self._effective_meta_score(result):.3f}"
            parts = [
                f"Verifier #{int(result.get('verify_index', 0)) + 1}",
                f"Verifier score: {result.get('proof_score')}",
                f"Meta-verifier score: {meta_score_label}",
                "",
                "Verifier evaluation:",
                str(result.get("verifier_evaluation") or "").strip(),
            ]
            meta_analysis = str(result.get("meta_analysis") or "").strip()
            if meta_analysis:
                parts.extend(["", "Meta-verifier analysis:", meta_analysis])
            analyses.append("\n".join(parts).strip())
        return analyses

    @staticmethod
    def _selector_score(payload: dict[str, Any]) -> float:
        format_score = float(payload.get("format_score", 0.0) or 0.0)
        verifier_meta_reward = float(payload.get("verifier_meta_reward", 0.0) or 0.0)
        return clamp01(format_score * verifier_meta_reward)

    def _rank_selector_candidates(self, state: vf.State) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for round_list_index, payload in enumerate(state.get("proof_opd_rounds") or []):
            proof = str(payload.get("proof") or "").strip()
            if not proof or float(payload.get("format_score", 0.0) or 0.0) <= 0.0:
                continue
            ranked.append(
                {
                    "round_list_index": round_list_index,
                    "round_index": int(payload.get("round_index", round_list_index)),
                    "preselection_score": self._selector_score(payload),
                    "format_score": clamp01(float(payload.get("format_score", 0.0) or 0.0)),
                    "verifier_meta_reward": clamp01(
                        float(payload.get("verifier_meta_reward", 0.0) or 0.0)
                    ),
                    "proof": proof,
                    "_payload": dict(payload),
                }
            )
        ranked.sort(
            key=lambda candidate: (
                float(candidate["preselection_score"]),
                float(candidate["format_score"]),
                float(candidate["verifier_meta_reward"]),
                int(candidate["round_index"]),
            ),
            reverse=True,
        )
        top_candidates = ranked[: self.selector_top_k]
        for candidate_id, candidate in enumerate(top_candidates, start=1):
            candidate["candidate_id"] = candidate_id
        return top_candidates

    def _build_wandb_trace(self, state: vf.State) -> dict[str, Any]:
        payload = dict(state.get("proof_opd_reward_payload") or {})
        generation = dict(state.get("proof_opd_generation") or {})
        verifier = dict(state.get("proof_opd_verifier") or {})
        meta = dict(state.get("proof_opd_meta") or {})
        selector = dict(state.get("proof_opd_selector") or {})
        info = self._input_info(state)
        trace = {
            "task_id": info.get("task_id") or self._input_value(state, "task_id"),
            "source_index": info.get("source_index") or self._input_value(state, "source_index"),
            "task_type": payload.get("task_type") or info.get("task_type") or self._input_value(state, "task_type"),
            "candidate_group_id": info.get("candidate_group_id"),
            "candidate_group_size": info.get("candidate_group_size"),
            "candidate_index": info.get("candidate_index"),
            "candidate_rank": info.get("candidate_rank"),
            "candidate_selected": info.get("candidate_selected"),
            "candidate_continue_count": info.get("candidate_continue_count"),
            "candidate_selection_format_score": info.get("candidate_selection_format_score"),
            "candidate_selection_self_score": info.get("candidate_selection_self_score"),
            "candidate_selection_mean_logprob": info.get("candidate_selection_mean_logprob"),
            "problem": clipped_trace_text(self._problem(state)),
            "reward": payload.get("reward", 0.0),
            "format_score": payload.get("format_score", 0.0),
            "format_ok": payload.get("format_ok", False),
            "proof_score": payload.get("proof_score"),
            "meta_score": payload.get("meta_score"),
            "verifier_meta_reward": payload.get("verifier_meta_reward"),
            "num_verifiers": payload.get("num_verifiers", self.num_verifiers),
            "num_verifier_results": payload.get("num_verifier_results", 0),
            "valid_verifier_count": payload.get("valid_verifier_count", 0),
            "valid_meta_count": payload.get("valid_meta_count", 0),
            "verifier_reward_terms": payload.get("verifier_reward_terms", []),
            "verifier_results": payload.get("verifier_results", []),
            "selected_refinement_verify_indices": payload.get("selected_refinement_verify_indices", []),
            "self_score": payload.get("self_score"),
            "selected_round_index": payload.get("selected_round_index", payload.get("round_index", -1)),
            "final_round_reward": payload.get("final_round_reward"),
            "best_round_reward": payload.get("best_round_reward"),
            "refine_rounds_used": payload.get("refine_rounds_used", 0),
            "selector_top_k": payload.get("selector_top_k", self.selector_top_k),
            "selector_valid": payload.get("selector_valid", False),
            "selector_fallback_used": payload.get("selector_fallback_used", False),
            "selector_invalid_reason": payload.get("selector_invalid_reason", ""),
            "selector_selected_id": payload.get("selector_selected_id"),
            "selector_selected_round_index": payload.get("selector_selected_round_index"),
            "selector_selected_preselection_score": payload.get("selector_selected_preselection_score"),
            "selector_candidates": payload.get(
                "selector_candidates",
                list(state.get("proof_opd_selector_candidates") or []),
            ),
            "reason": payload.get("reason", ""),
            "finish_reason": payload.get("finish_reason", ""),
            "is_truncated": payload.get("is_truncated", False),
            "closed_thinking": payload.get("closed_thinking", generation.get("closed_thinking", False)),
            "proof_chars": payload.get("proof_chars", len(str(generation.get("proof") or ""))),
            "self_evaluation_chars": payload.get(
                "self_evaluation_chars",
                len(str(generation.get("self_evaluation") or "")),
            ),
            "verifier_evaluation_chars": payload.get(
                "verifier_evaluation_chars",
                len(str(verifier.get("evaluation") or "")),
            ),
            "meta_analysis_chars": payload.get("meta_analysis_chars", len(str(meta.get("analysis") or ""))),
            "verifier_invalid_reason": payload.get("verifier_invalid_reason", verifier.get("invalid_reason", "")),
            "meta_invalid_reason": payload.get("meta_invalid_reason", meta.get("invalid_reason", "")),
            "stage_records": payload.get("stage_records", list(state.get("proof_opd_stage_records") or [])),
            "proof_raw_output_excerpt": clipped_trace_text(
                payload.get("generation_raw_output", generation.get("raw_output", ""))
            ),
            "proof_excerpt": clipped_trace_text(payload.get("proof", generation.get("proof", ""))),
            "self_evaluation_excerpt": clipped_trace_text(
                payload.get("self_evaluation", generation.get("self_evaluation", ""))
            ),
            "verifier_raw_output_excerpt": clipped_trace_text(verifier.get("raw_output", "")),
            "verifier_evaluation_excerpt": clipped_trace_text(
                payload.get("verifier_evaluation", verifier.get("evaluation", ""))
            ),
            "meta_raw_output_excerpt": clipped_trace_text(meta.get("raw_output", "")),
            "meta_analysis_excerpt": clipped_trace_text(payload.get("meta_analysis", meta.get("analysis", ""))),
            "selector_raw_output_excerpt": clipped_trace_text(
                payload.get("selector_raw_output", selector.get("raw_output", ""))
            ),
            "selector_visible_output": clipped_trace_text(
                payload.get("selector_visible_output", selector.get("visible_output", ""))
            ),
        }
        return trace

    def _format_score(self, parsed: dict[str, Any]) -> float:
        if not parsed.get("has_solution_section") or not str(parsed.get("proof") or "").strip():
            return 0.0
        if parsed.get("format_ok"):
            return 1.0
        return self.partial_format_score

    def _generation_invalid_reason(self, parsed: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not parsed.get("closed_thinking"):
            return "missing_closed_think"
        if not parsed.get("has_solution_section") or not str(parsed.get("proof") or "").strip():
            return "missing_solution_section_or_empty_proof"
        return ""

    def _verifier_invalid_reason(self, verifier: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not verifier.get("closed_thinking"):
            return "missing_closed_think"
        return verifier_invalid_reason(verifier)

    def _meta_invalid_reason(self, meta: dict[str, Any], is_truncated: bool) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not meta.get("closed_thinking"):
            return "missing_closed_think"
        if meta.get("score") is None:
            return "missing_or_invalid_boxed_score"
        if not str(meta.get("analysis") or "").strip():
            return "empty_meta_analysis"
        return ""

    def _selector_invalid_reason(
        self,
        selector: dict[str, Any],
        is_truncated: bool,
        candidate_count: int,
    ) -> str:
        if is_truncated:
            return "truncated_or_length_finish"
        if self.require_closed_think and not selector.get("closed_thinking"):
            return "missing_closed_think"
        selected_id = selector.get("selected_id")
        if selected_id is None:
            return "missing_or_invalid_selected_id"
        if not 1 <= int(selected_id) <= candidate_count:
            return "selected_id_out_of_range"
        return ""

    def _invalid_generation_payload(
        self,
        parsed: dict[str, Any],
        round_idx: int,
        reason: str,
        *,
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> dict[str, Any]:
        return {
            "round_index": round_idx,
            "selected_round_index": round_idx,
            "reward": 0.0,
            "format_score": 0.0,
            "format_ok": False,
            "proof_score": 0.0,
            "meta_score": 0.0 if self.enable_meta_verification else 1.0,
            "num_verifiers": self.num_verifiers,
            "num_verifier_results": 0,
            "valid_verifier_count": 0,
            "valid_meta_count": 0,
            "verifier_reward_terms": [],
            "verifier_meta_reward": 0.0,
            "verifier_results": [],
            "self_score": parsed.get("self_score"),
            "proof_chars": len(str(parsed.get("proof") or "")),
            "closed_thinking": parsed.get("closed_thinking", False),
            "is_truncated": is_truncated,
            "finish_reason": finish_reason,
            "reason": reason,
            "raw_output": parsed.get("raw_output", ""),
            "proof": parsed.get("proof", ""),
            "self_evaluation": parsed.get("self_evaluation", ""),
        }

    def _record_stage(
        self,
        state: vf.State,
        *,
        stage: str,
        parsed: dict[str, Any],
        invalid_reason: str = "",
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> None:
        trajectory = state.get("trajectory") or []
        token_counts = trajectory_step_token_counts(trajectory[-1] if trajectory else None)
        state.setdefault("proof_opd_stage_records", []).append(
            {
                "stage": stage,
                "round_index": int(state.get("proof_opd_current_round", 0)),
                "verify_index": int(state.get("proof_opd_verify_index", 0)),
                "raw_chars": int(parsed.get("raw_chars") or 0),
                "raw_output_excerpt": clipped_trace_text(parsed.get("raw_output", "")),
                "closed_thinking": bool(parsed.get("closed_thinking")),
                "is_truncated": bool(is_truncated),
                "finish_reason": finish_reason,
                "invalid_reason": invalid_reason,
                **token_counts,
            }
        )

    def _last_step_status(self, state: vf.State) -> tuple[str, bool, str]:
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return "", False, ""
        step = trajectory[-1]
        return trajectory_step_text(step), trajectory_step_is_truncated(step), trajectory_step_finish_reason(step)

    def _stop(self, state: vf.State) -> vf.Messages:
        state["final_env_response"] = []
        return []

    def _finalize_round(self, state: vf.State) -> dict[str, Any]:
        round_idx = int(state.get("proof_opd_current_round", 0))
        generation = dict(state.get("proof_opd_generation") or {})
        verifier_results = list(state.get("proof_opd_verifier_results") or [])
        primary_result = verifier_results[0] if verifier_results else {}
        verifier = dict(primary_result.get("verifier") or state.get("proof_opd_verifier") or {})
        meta = dict(primary_result.get("meta") or state.get("proof_opd_meta") or {})
        format_score = self._format_score(generation)
        verifier_summary = self._summarize_verifier_results(verifier_results)
        reward = clamp01(format_score * float(verifier_summary["verifier_meta_reward"]))
        selected_refinement_reviews = self._select_refinement_reviews({"verifier_results": verifier_results})
        selected_refinement_verify_indices = []
        for review in selected_refinement_reviews:
            match = re.search(r"Verifier #(\d+)", review)
            if match:
                selected_refinement_verify_indices.append(int(match.group(1)) - 1)
        payload = {
            "round_index": round_idx,
            "reward": reward,
            "format_score": format_score,
            "format_ok": generation.get("format_ok", False),
            **verifier_summary,
            "selected_refinement_verify_indices": selected_refinement_verify_indices,
            "self_score": generation.get("self_score"),
            "proof_chars": len(str(generation.get("proof") or "")),
            "self_evaluation_chars": len(str(generation.get("self_evaluation") or "")),
            "verifier_evaluation_chars": sum(len(str(result.get("verifier_evaluation") or "")) for result in verifier_results),
            "meta_analysis_chars": sum(len(str(result.get("meta_analysis") or "")) for result in verifier_results),
            "generation_raw_output": generation.get("raw_output", ""),
            "verifier_raw_output": verifier.get("raw_output", ""),
            "meta_raw_output": meta.get("raw_output", ""),
            "proof": generation.get("proof", ""),
            "self_evaluation": generation.get("self_evaluation", ""),
            "verifier_evaluation": verifier.get("evaluation", ""),
            "verifier_invalid_reason": primary_result.get("verifier_invalid_reason", verifier.get("invalid_reason", "")),
            "meta_analysis": meta.get("analysis", ""),
            "meta_invalid_reason": primary_result.get("meta_invalid_reason", meta.get("invalid_reason", "")),
            "verifier_results": verifier_results,
            "selected_refinement_reviews": selected_refinement_reviews,
            "stage_records": list(state.get("proof_opd_stage_records") or []),
        }
        rounds = state.setdefault("proof_opd_rounds", [])
        rounds.append(payload)
        best_idx = max(range(len(rounds)), key=lambda idx: float(rounds[idx].get("reward", 0.0) or 0.0))
        selected = dict(rounds[best_idx])
        selected["selected_round_index"] = best_idx
        selected["final_round_reward"] = reward
        selected["best_round_reward"] = float(rounds[best_idx].get("reward", 0.0) or 0.0)
        selected["refine_rounds_used"] = max(0, len(rounds) - 1)
        state["proof_opd_reward_payload"] = selected
        LOGGER.debug("Proof-OPD round scored: %s", json.dumps(selected, ensure_ascii=False)[:4000])
        return selected

    def _public_selector_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in candidate.items() if key not in {"_payload", "proof"}}
            for candidate in candidates
        ]

    def _next_selector_prompt(
        self,
        state: vf.State,
        candidates: list[dict[str, Any]] | None = None,
    ) -> vf.Messages:
        candidates = list(candidates) if candidates is not None else self._rank_selector_candidates(state)
        state["proof_opd_selector_candidates"] = candidates
        if not candidates:
            payload = dict(state.get("proof_opd_reward_payload") or {})
            payload.update(
                {
                    "selector_top_k": self.selector_top_k,
                    "selector_candidates": [],
                    "selector_valid": False,
                    "selector_fallback_used": False,
                    "selector_invalid_reason": "no_valid_candidates",
                }
            )
            state["proof_opd_reward_payload"] = payload
            return self._stop(state)

        rounds = state.get("proof_opd_rounds") or []
        selection_bundle = []
        for candidate in candidates:
            proof = str(candidate.get("proof") or "").strip()
            if not proof and "round_list_index" in candidate:
                proof = str(rounds[int(candidate["round_list_index"])].get("proof") or "").strip()
            selection_bundle.append(proof)
        prompt = build_deepseek_selector_prompt(self._problem(state), selection_bundle)
        log_llm_input("selector", prompt, state=state)
        state["proof_opd_stage"] = "selector"
        return [vf.UserMessage(content=prompt)]

    def _finalize_selector(
        self,
        state: vf.State,
        selector: dict[str, Any],
        invalid_reason: str,
        *,
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> dict[str, Any]:
        candidates = list(state.get("proof_opd_selector_candidates") or [])
        if not candidates:
            return dict(state.get("proof_opd_reward_payload") or {})

        requested_id = selector.get("selected_id")
        selected_id = int(requested_id) if not invalid_reason else int(candidates[0]["candidate_id"])
        selected_candidate = next(
            candidate for candidate in candidates if int(candidate["candidate_id"]) == selected_id
        )
        rounds = state.get("proof_opd_rounds") or []
        selected_payload = selected_candidate.get("_payload")
        if isinstance(selected_payload, dict):
            selected = dict(selected_payload)
        else:
            selected = dict(rounds[int(selected_candidate["round_list_index"])])
        selected["reward"] = self._selector_score(selected)
        selected["selected_round_index"] = int(selected_candidate["round_index"])
        if "source_candidate_index" in selected_candidate:
            selected["selector_source_candidate_index"] = int(selected_candidate["source_candidate_index"])
            selected["selector_source_candidate_rank"] = int(selected_candidate.get("source_candidate_rank", 0))
            selected["final_round_reward"] = float(
                selected.get("final_round_reward", self._selector_score(selected)) or 0.0
            )
            selected["best_round_reward"] = max(
                float(candidate.get("preselection_score", 0.0) or 0.0) for candidate in candidates
            )
            selected.setdefault("refine_rounds_used", int(selected_candidate.get("round_index", 0)))
        else:
            selected["final_round_reward"] = self._selector_score(rounds[-1])
            selected["best_round_reward"] = max(self._selector_score(payload) for payload in rounds)
            selected["refine_rounds_used"] = max(0, len(rounds) - 1)
        selected["selector_top_k"] = self.selector_top_k
        selected["selector_candidates"] = self._public_selector_candidates(candidates)
        selected["selector_valid"] = not bool(invalid_reason)
        selected["selector_fallback_used"] = bool(invalid_reason)
        selected["selector_invalid_reason"] = invalid_reason
        selected["selector_requested_id"] = requested_id
        selected["selector_selected_id"] = selected_id
        selected["selector_selected_round_index"] = int(selected_candidate["round_index"])
        selected["selector_selected_preselection_score"] = float(selected_candidate["preselection_score"])
        selected["selector_raw_output"] = selector.get("raw_output", "")
        selected["selector_visible_output"] = selector.get("visible_output", "")
        selected["selector_closed_thinking"] = selector.get("closed_thinking", False)
        selected["selector_is_truncated"] = is_truncated
        selected["selector_finish_reason"] = finish_reason
        selected["stage_records"] = list(state.get("proof_opd_stage_records") or [])

        selector_state = dict(selector)
        selector_state.update(
            {
                "valid": not bool(invalid_reason),
                "invalid_reason": invalid_reason,
                "fallback_used": bool(invalid_reason),
                "requested_id": requested_id,
                "selected_id": selected_id,
                "selected_round_index": int(selected_candidate["round_index"]),
                "is_truncated": is_truncated,
                "finish_reason": finish_reason,
            }
        )
        state["proof_opd_selector"] = selector_state
        state["proof_opd_reward_payload"] = selected
        LOGGER.info("Proof-OPD selector finalized: %s", json.dumps(selected, ensure_ascii=False)[:4000])
        return selected

    def _should_refine(self, state: vf.State, payload: dict[str, Any]) -> bool:
        rounds = state.get("proof_opd_rounds") or []
        if len(rounds) > self.refine_rounds:
            return False
        return float(payload.get("reward", 0.0) or 0.0) < self.refine_early_stop_reward

    def _maybe_attach_invalid_generation_eval_metrics(
        self,
        state: vf.State,
        payload: dict[str, Any],
        parsed: dict[str, Any],
    ) -> None:
        return None

    def _maybe_score_valid_generation_for_eval(
        self,
        state: vf.State,
        parsed: dict[str, Any],
        round_idx: int,
        *,
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> dict[str, Any] | None:
        return None

    def _start_verification_round(self, state: vf.State) -> vf.Messages:
        state["proof_opd_verify_index"] = 0
        state["proof_opd_verifier_results"] = []
        state["proof_opd_pending_verifier_result"] = None
        return self._next_verifier_prompt(state)

    def _next_verifier_prompt(self, state: vf.State) -> vf.Messages:
        proof = str((state.get("proof_opd_generation") or {}).get("proof") or "")
        prompt = build_deepseek_proof_verification_prompt(self._problem(state), proof)
        verify_index = int(state.get("proof_opd_verify_index", 0))
        log_llm_input(f"verifier_{verify_index + 1}_of_{self.num_verifiers}", prompt, state=state)
        state["proof_opd_stage"] = "verifier"
        return [vf.UserMessage(content=prompt)]

    def _append_verifier_result(self, state: vf.State, result: dict[str, Any]) -> None:
        result["meta_score_effective"] = self._effective_meta_score(result)
        result["verifier_score_effective"] = self._effective_verifier_score(result)
        result["reward_term"] = self._verifier_reward_term(result)
        state.setdefault("proof_opd_verifier_results", []).append(result)
        state["proof_opd_verifier"] = result.get("verifier")
        state["proof_opd_meta"] = result.get("meta")
        state["proof_opd_pending_verifier_result"] = None

    async def _finish_candidate(self, state: vf.State, payload: dict[str, Any]) -> vf.Messages:
        group_result = await self._submit_group_selector_record(state)
        if group_result is not None:
            is_leader, candidates = group_result
            if is_leader:
                return self._next_selector_prompt(state, candidates)
            payload.update(
                {
                    "selector_top_k": self.selector_top_k,
                    "selector_candidates": self._public_selector_candidates(candidates),
                    "selector_valid": False,
                    "selector_fallback_used": False,
                    "selector_invalid_reason": "group_selector_delegated",
                }
            )
            state["proof_opd_reward_payload"] = payload
            return self._stop(state)

        if float(payload.get("reward", 0.0) or 0.0) >= self.refine_early_stop_reward:
            payload.update(
                {
                    "selector_valid": False,
                    "selector_fallback_used": False,
                    "selector_invalid_reason": "skipped_early_stop_reward",
                }
            )
            state["proof_opd_reward_payload"] = payload
            return self._stop(state)
        return self._next_selector_prompt(state)

    async def _after_verifier_result(self, state: vf.State) -> vf.Messages:
        state["proof_opd_verify_index"] = int(state.get("proof_opd_verify_index", 0)) + 1
        if int(state["proof_opd_verify_index"]) < self.num_verifiers:
            return self._next_verifier_prompt(state)
        payload = self._finalize_round(state)
        if self._should_refine(state, payload):
            return self._next_refinement_prompt(state, payload)
        return await self._finish_candidate(state, payload)

    async def env_response(self, messages: vf.Messages, state: vf.State, **_: Any) -> vf.Messages:
        return []

    async def _advance_after_completion(self, state: vf.State) -> vf.Messages:
        stage = str(state.get("proof_opd_stage") or "proof")
        text, is_truncated, finish_reason = self._last_step_status(state)
        problem = self._problem(state)
        round_idx = int(state.get("proof_opd_current_round", 0))

        if stage in {"proof", "refine"}:
            parsed = parse_generation_response(text)
            state["proof_opd_generation"] = parsed
            invalid_reason = self._generation_invalid_reason(parsed, is_truncated)
            self._record_stage(
                state,
                stage=stage,
                parsed=parsed,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            candidate_selected = True
            if stage == "proof":
                candidate_selected = await self._candidate_gate_after_proof(
                    state,
                    parsed,
                    invalid_reason,
                )
            if invalid_reason:
                payload = self._invalid_generation_payload(
                    parsed,
                    round_idx,
                    invalid_reason,
                    is_truncated=is_truncated,
                    finish_reason=finish_reason,
                )
                payload["stage_records"] = list(state.get("proof_opd_stage_records") or [])
                self._maybe_attach_invalid_generation_eval_metrics(state, payload, parsed)
                state.setdefault("proof_opd_rounds", []).append(payload)
                state["proof_opd_reward_payload"] = payload
                LOGGER.info("Proof-OPD invalid generation: %s", json.dumps(payload, ensure_ascii=False))
                if self._rank_selector_candidates(state):
                    return await self._finish_candidate(state, payload)
                return self._stop(state)
            if stage == "proof" and not candidate_selected:
                payload = self._proof_only_candidate_payload(
                    state,
                    parsed,
                    is_truncated=is_truncated,
                    finish_reason=finish_reason,
                )
                state.setdefault("proof_opd_rounds", []).append(payload)
                state["proof_opd_reward_payload"] = payload
                LOGGER.info(
                    "Proof-OPD candidate stopped after proof-only stage: %s",
                    json.dumps(payload, ensure_ascii=False)[:4000],
                )
                return self._stop(state)
            payload = self._maybe_score_valid_generation_for_eval(
                state,
                parsed,
                round_idx,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            if payload is not None:
                state.setdefault("proof_opd_rounds", []).append(payload)
                state["proof_opd_reward_payload"] = payload
                LOGGER.info("Proof-OPD verifiable eval generation scored: %s", json.dumps(payload, ensure_ascii=False)[:4000])
                return self._stop(state)
            return self._start_verification_round(state)

        if stage == "verifier":
            verifier = parse_verifier_response(text)
            invalid_reason = self._verifier_invalid_reason(verifier, is_truncated)
            self._record_stage(
                state,
                stage="verifier",
                parsed=verifier,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            if invalid_reason:
                verifier["invalid_reason"] = invalid_reason
                LOGGER.info(
                    "Proof-OPD skipping meta verifier: invalid verifier output "
                    "reason=%s score=%s evaluation_chars=%d raw_chars=%d truncated=%s finish_reason=%s",
                    invalid_reason,
                    verifier.get("score"),
                    len(str(verifier.get("evaluation") or "")),
                    int(verifier.get("raw_chars") or 0),
                    is_truncated,
                    finish_reason,
                )
            result = {
                "verify_index": int(state.get("proof_opd_verify_index", 0)),
                "verifier": verifier,
                "verifier_valid": not bool(invalid_reason),
                "verifier_invalid_reason": invalid_reason,
                "proof_score": verifier.get("score") if not invalid_reason else 0.0,
                "verifier_evaluation": verifier.get("evaluation", ""),
                "verifier_raw_output": verifier.get("raw_output", ""),
                "verifier_closed_thinking": verifier.get("closed_thinking", False),
                "verifier_is_truncated": is_truncated,
                "verifier_finish_reason": finish_reason,
            }
            state["proof_opd_verifier"] = verifier
            state["proof_opd_pending_verifier_result"] = result
            verifier_score = self._effective_verifier_score(result)
            if self.enable_meta_verification and not invalid_reason and verifier_score < 1.0:
                proof = str((state.get("proof_opd_generation") or {}).get("proof") or "")
                prompt = build_deepseek_meta_verification_prompt(problem, proof, verifier["evaluation"])
                log_llm_input(
                    f"meta_verifier_{int(state.get('proof_opd_verify_index', 0)) + 1}_of_{self.num_verifiers}",
                    prompt,
                    state=state,
                )
                state["proof_opd_stage"] = "meta"
                return [vf.UserMessage(content=prompt)]
            if not invalid_reason and self.enable_meta_verification:
                LOGGER.info(
                    "Proof-OPD skipping meta verifier: verifier score is 1 "
                    "verify_index=%d evaluation_chars=%d",
                    int(state.get("proof_opd_verify_index", 0)),
                    len(str(verifier.get("evaluation") or "")),
                )
            result["meta"] = None
            result["meta_valid"] = False
            if invalid_reason:
                result["meta_invalid_reason"] = "skipped_invalid_verifier"
            elif self.enable_meta_verification:
                result["meta_invalid_reason"] = "skipped_perfect_verifier"
            else:
                result["meta_invalid_reason"] = "disabled"
            result["meta_score"] = 0.0 if invalid_reason else 1.0
            result["meta_analysis"] = ""
            result["meta_raw_output"] = ""
            self._append_verifier_result(state, result)
            return await self._after_verifier_result(state)

        if stage == "meta":
            meta = parse_meta_verifier_response(text)
            invalid_reason = self._meta_invalid_reason(meta, is_truncated)
            if invalid_reason:
                meta["invalid_reason"] = invalid_reason
                LOGGER.info(
                    "Proof-OPD meta verifier invalid: reason=%s score=%s analysis_chars=%d raw_chars=%d "
                    "truncated=%s finish_reason=%s",
                    invalid_reason,
                    meta.get("score"),
                    len(str(meta.get("analysis") or "")),
                    int(meta.get("raw_chars") or 0),
                    is_truncated,
                    finish_reason,
                )
            self._record_stage(
                state,
                stage="meta",
                parsed=meta,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            state["proof_opd_meta"] = meta
            result = dict(state.get("proof_opd_pending_verifier_result") or {})
            result.update(
                {
                    "meta": meta,
                    "meta_valid": not bool(invalid_reason),
                    "meta_invalid_reason": invalid_reason,
                    "meta_score": meta.get("score") if not invalid_reason else 0.5,
                    "meta_analysis": meta.get("analysis", ""),
                    "meta_raw_output": meta.get("raw_output", ""),
                    "meta_closed_thinking": meta.get("closed_thinking", False),
                    "meta_is_truncated": is_truncated,
                    "meta_finish_reason": finish_reason,
                }
            )
            self._append_verifier_result(state, result)
            return await self._after_verifier_result(state)

        if stage == "selector":
            selector = parse_selector_response(text)
            candidates = list(state.get("proof_opd_selector_candidates") or [])
            invalid_reason = self._selector_invalid_reason(selector, is_truncated, len(candidates))
            self._record_stage(
                state,
                stage="selector",
                parsed=selector,
                invalid_reason=invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            self._finalize_selector(
                state,
                selector,
                invalid_reason,
                is_truncated=is_truncated,
                finish_reason=finish_reason,
            )
            return self._stop(state)

        return self._stop(state)

    def _next_refinement_prompt(self, state: vf.State, payload: dict[str, Any]) -> vf.Messages:
        state["proof_opd_current_round"] = int(state.get("proof_opd_current_round", 0)) + 1
        state["proof_opd_stage"] = "refine"
        state["proof_opd_generation"] = None
        state["proof_opd_verifier"] = None
        state["proof_opd_meta"] = None
        state["proof_opd_verify_index"] = 0
        state["proof_opd_verifier_results"] = []
        state["proof_opd_pending_verifier_result"] = None
        analyses = self._select_refinement_reviews(payload)
        if not analyses:
            analyses = [str(payload.get("verifier_evaluation") or "No verifier analysis was available.")]
        prompt = build_deepseek_proof_refinement_prompt(
            self._problem(state),
            str(payload.get("proof") or ""),
            analyses,
        )
        log_llm_input("refinement", prompt, state=state)
        return [vf.UserMessage(content=prompt)]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state.get("trajectory") or []) == 0:
            state["proof_opd_stage"] = "proof"
            log_llm_input("proof_generation", completion_to_text(state["prompt"]), state=state)
            return state["prompt"]
        return await self._advance_after_completion(state)

    @vf.cleanup(priority=0)
    async def render_full_completion_trace(self, state: vf.State) -> None:
        state["completion"] = render_full_stage_completion(state)

    @vf.cleanup(priority=-10)
    async def attach_wandb_trace(self, state: vf.State) -> None:
        info = dict(self._input_info(state))
        if state.get("proof_opd_reward_payload") is not None:
            info["proof_opd_trace"] = self._build_wandb_trace(state)
        state["info"] = info


class ProofOPDVerifiableEvalEnv(ProofOPDEnv):
    """Eval-only env that scores boxed final answers instead of train reward."""

    def _build_wandb_trace(self, state: vf.State) -> dict[str, Any]:
        trace = super()._build_wandb_trace(state)
        payload = dict(state.get("proof_opd_reward_payload") or {})
        info = self._input_info(state)
        trace.update(
            {
                "gold_answer": payload.get("gold_answer")
                or info.get("gold_answer")
                or self._input_value(state, "gold_answer"),
                "boxed_answer": payload.get("boxed_answer", ""),
                "boxed_present": payload.get("boxed_present", 0.0),
                "verifiable_accuracy": payload.get("verifiable_accuracy", 0.0),
                "answer_match_method": payload.get("answer_match_method", ""),
            }
        )
        return trace

    def _attach_answer_metrics(
        self,
        state: vf.State,
        payload: dict[str, Any],
        generation: dict[str, Any],
    ) -> dict[str, Any]:
        info = self._input_info(state)
        gold_answer = str(info.get("gold_answer") or self._input_value(state, "gold_answer") or "").strip()
        payload["task_type"] = "verifiable"
        payload["gold_answer"] = gold_answer
        if not gold_answer:
            payload["boxed_answer"] = ""
            payload["boxed_present"] = 0.0
            payload["verifiable_accuracy"] = 0.0
            payload["answer_match_method"] = "missing_gold"
            payload["answer_match_method_id"] = VERIFIABLE_ANSWER_MATCH_METHOD_IDS["missing_gold"]
            return payload

        boxed_answer = extract_verifiable_boxed_answer(str(generation.get("proof") or ""))
        is_correct, method = check_verifiable_answer(boxed_answer, gold_answer)
        payload["boxed_answer"] = boxed_answer
        payload["boxed_present"] = 1.0 if boxed_answer else 0.0
        payload["verifiable_accuracy"] = 1.0 if is_correct else 0.0
        payload["answer_match_method"] = method
        payload["answer_match_method_id"] = VERIFIABLE_ANSWER_MATCH_METHOD_IDS.get(method, 1.0)
        return payload

    def _maybe_attach_invalid_generation_eval_metrics(
        self,
        state: vf.State,
        payload: dict[str, Any],
        parsed: dict[str, Any],
    ) -> None:
        self._attach_answer_metrics(state, payload, parsed)

    def _maybe_score_valid_generation_for_eval(
        self,
        state: vf.State,
        parsed: dict[str, Any],
        round_idx: int,
        *,
        is_truncated: bool = False,
        finish_reason: str = "",
    ) -> dict[str, Any] | None:
        if round_idx < self.refine_rounds:
            return None
        payload = {
            "round_index": round_idx,
            "selected_round_index": round_idx,
            "reward": 0.0,
            "format_score": self._format_score(parsed),
            "format_ok": parsed.get("format_ok", False),
            "proof_score": None,
            "meta_score": None,
            "num_verifiers": 0,
            "num_verifier_results": 0,
            "valid_verifier_count": 0,
            "valid_meta_count": 0,
            "verifier_reward_terms": [],
            "verifier_meta_reward": 0.0,
            "verifier_results": [],
            "self_score": parsed.get("self_score"),
            "proof_chars": len(str(parsed.get("proof") or "")),
            "self_evaluation_chars": len(str(parsed.get("self_evaluation") or "")),
            "verifier_evaluation_chars": 0,
            "meta_analysis_chars": 0,
            "closed_thinking": parsed.get("closed_thinking", False),
            "is_truncated": is_truncated,
            "finish_reason": finish_reason,
            "reason": "verifiable_eval_generation_only",
            "generation_raw_output": parsed.get("raw_output", ""),
            "proof": parsed.get("proof", ""),
            "self_evaluation": parsed.get("self_evaluation", ""),
            "verifier_evaluation": "",
            "verifier_invalid_reason": "",
            "meta_analysis": "",
            "meta_invalid_reason": "",
            "stage_records": list(state.get("proof_opd_stage_records") or []),
        }
        self._attach_answer_metrics(state, payload, parsed)
        payload["reward"] = max(0.0, float(payload.get("verifiable_accuracy", 0.0) or 0.0))
        return payload

    def _should_refine(self, state: vf.State, payload: dict[str, Any]) -> bool:
        rounds = state.get("proof_opd_rounds") or []
        return len(rounds) <= self.refine_rounds


def load_environment(
    dataset_path: str,
    problem_column: str = "auto",
    solution_column: str = "auto",
    max_examples: int | None = None,
    dataset_mode: str = "mixed",
    verifiable_dataset_path: str | None = None,
    verifiable_fraction: float = 0.2,
    verifiable_answer_column: str = "auto",
    mix_seed: int = DEFAULT_MIX_SEED,
    enable_meta_verification: bool | str = True,
    num_verifiers: int = 4,
    partial_format_score: float = 0.7,
    require_closed_think: bool | str = True,
    refine_rounds: int = 1,
    refine_review_n: int = 2,
    refine_early_stop_reward: float = 0.95,
    selector_top_k: int = 3,
    candidate_gate_enabled: bool | str = False,
    candidate_continue_count: int = 4,
    **_: Any,
) -> vf.Environment:
    normalized_mode = str(dataset_mode or "mixed").strip().lower()
    use_single_turn = normalized_mode in {"single", "single_turn", "per_turn"}
    use_verifiable_eval = normalized_mode in {"verifiable", "verifiable_eval", "eval_verifiable"}
    single_turn_columns = [
        "pipeline",
        "run_id",
        "problem_id",
        "problem",
        "stage",
        "candidate_id",
        "verifier_idx",
        "messages_json",
    ]
    proof_rows = read_dataset_rows(
        dataset_path,
        columns=single_turn_columns if use_single_turn else None,
    )
    if use_single_turn:
        rows = normalize_single_turn_dataset_rows(
            proof_rows,
            max_examples=max_examples,
        )
    elif use_verifiable_eval:
        rows = normalize_dataset_rows(
            proof_rows,
            problem_column=problem_column,
            solution_column=solution_column,
            max_examples=max_examples,
            task_type="verifiable",
            answer_column=verifiable_answer_column,
            dataset_label="proof_math_verifiable",
        )
    else:
        verifiable_rows = read_dataset_rows(verifiable_dataset_path) if verifiable_dataset_path else None
        rows = normalize_mixed_dataset_rows(
            proof_rows,
            problem_column=problem_column,
            solution_column=solution_column,
            max_examples=max_examples,
            verifiable_rows=verifiable_rows,
            verifiable_fraction=float(verifiable_fraction),
            verifiable_answer_column=verifiable_answer_column,
            mix_seed=int(mix_seed),
        )
    task_counts: dict[str, int] = {}
    for row in rows:
        task_counts[str(row.get("task_type") or "proof")] = task_counts.get(str(row.get("task_type") or "proof"), 0) + 1
    LOGGER.info(
        "Loaded Proof-OPD dataset: mode=%s proof_path=%s verifiable_path=%s rows=%d task_counts=%s",
        normalized_mode,
        dataset_path,
        verifiable_dataset_path,
        len(rows),
        task_counts,
    )
    dataset = Dataset.from_list(rows)
    if use_single_turn:
        return ProofOPDSingleTurnEnv(
            dataset=dataset,
            eval_dataset=dataset,
            rubric=ProofOPDSingleTurnRubric(),
            message_type="chat",
        )
    env_cls = ProofOPDVerifiableEvalEnv if use_verifiable_eval else ProofOPDEnv
    return env_cls(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=ProofOPDRubric(),
        message_type="chat",
        refine_rounds=int(refine_rounds),
        num_verifiers=int(num_verifiers),
        refine_review_n=int(refine_review_n),
        enable_meta_verification=parse_bool(enable_meta_verification, True),
        partial_format_score=float(partial_format_score),
        require_closed_think=parse_bool(require_closed_think, True),
        refine_early_stop_reward=float(refine_early_stop_reward),
        selector_top_k=int(selector_top_k),
        candidate_gate_enabled=(
            parse_bool(candidate_gate_enabled, False) and not use_verifiable_eval
        ),
        candidate_continue_count=int(candidate_continue_count),
    )
