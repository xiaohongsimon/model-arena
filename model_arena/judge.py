# model_arena/judge.py
import json
import random
import re
import time
from typing import Optional
import httpx
from model_arena.types import TaskSpec

# Judge model config — Opus via API
JUDGE_ENDPOINT = "https://api.anthropic.com/v1/messages"
JUDGE_MODEL = "claude-opus-4-6"
JUDGE_MAX_TOKENS = 1024


def randomize_labels(model_ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Assign random A/B/C labels to model IDs.
    Returns (model_id -> label, label -> model_id)."""
    labels_pool = [chr(ord("A") + i) for i in range(len(model_ids))]
    shuffled = list(model_ids)
    random.shuffle(shuffled)
    id_to_label = {}
    label_to_id = {}
    for model_id, label in zip(shuffled, labels_pool):
        id_to_label[model_id] = label
        label_to_id[label] = model_id
    return id_to_label, label_to_id


def build_judge_prompt(
    task_spec: TaskSpec,
    user_prompt: str,
    label_outputs: dict[str, str],
) -> str:
    """Build the judge prompt with anonymous labeled outputs."""
    outputs_text = "\n\n".join(
        f"=== Output {label} ===\n{output}"
        for label, output in sorted(label_outputs.items())
    )
    return f"""You are judging model outputs for quality. Rubric version: {task_spec.judge_rubric_version}

## Task
{user_prompt}

## Outputs
{outputs_text}

## Instructions
Compare the outputs and respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "winner": "<label or null if tie/all_poor>",
  "scores": {{"<label>": <float 0.0-5.0>, ...}},
  "all_poor": <true if all scores below {task_spec.quality_floor}>,
  "comment": "<brief justification>",
  "judge_rubric_version": "{task_spec.judge_rubric_version}"
}}

Score each output 0.0-5.0. Pick the winner (highest score), or null if tied or all poor."""


def parse_judge_response(raw: str) -> tuple[Optional[dict], Optional[str]]:
    """Parse judge response JSON. Handles markdown code blocks."""
    text = raw.strip()
    # Strip markdown code block
    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def validate_judge_schema(result: dict, valid_labels: list[str]) -> list[str]:
    """Validate judge response against expected schema."""
    errors = []
    # Check required fields
    for field in ("winner", "scores", "all_poor", "comment", "judge_rubric_version"):
        if field not in result:
            errors.append(f"Missing field: {field}")
    if errors:
        return errors

    # Winner validation
    winner = result["winner"]
    all_poor = result.get("all_poor", False)
    if winner is None:
        # null winner is only valid when all_poor is True
        if not all_poor:
            errors.append("winner is null but all_poor is False")
    elif winner not in valid_labels:
        errors.append(f"winner '{winner}' not in valid labels {valid_labels}")

    # Scores validation
    scores = result.get("scores", {})
    for label in valid_labels:
        if label not in scores:
            errors.append(f"Missing score for label {label}")
    for label, score in scores.items():
        if not isinstance(score, (int, float)):
            errors.append(f"Score for {label} must be numeric, got {type(score)}")
        elif score < 0.0 or score > 5.0:
            errors.append(f"Score for {label} out of range [0.0, 5.0]: {score}")

    return errors


async def judge_outputs(
    task_spec: TaskSpec,
    user_prompt: str,
    label_outputs: dict[str, str],
    api_key: str,
) -> dict:
    """Call Opus judge. Returns dict with parsed result or error info.

    Return keys: status ('completed'|'judge_error'), parsed (dict|None),
    raw_response (str), latency_ms (int).
    """
    valid_labels = sorted(label_outputs.keys())
    prompt = build_judge_prompt(task_spec, user_prompt, label_outputs)
    raw = ""

    for attempt in range(2):  # original + 1 repair retry
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    JUDGE_ENDPOINT,
                    json={
                        "model": JUDGE_MODEL,
                        "max_tokens": JUDGE_MAX_TOKENS,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["content"][0]["text"]
                latency_ms = int((time.monotonic() - start) * 1000)

        except Exception as e:
            return {
                "status": "judge_error",
                "parsed": None,
                "raw_response": str(e),
                "latency_ms": int((time.monotonic() - start) * 1000),
            }

        parsed, parse_error = parse_judge_response(raw)
        if parsed is not None:
            schema_errors = validate_judge_schema(parsed, valid_labels)
            if not schema_errors:
                return {
                    "status": "completed",
                    "parsed": parsed,
                    "raw_response": raw,
                    "latency_ms": latency_ms,
                }

        # Repair retry: send error feedback
        if attempt == 0:
            error_detail = parse_error or "; ".join(schema_errors)
            prompt = f"""Your previous response had errors: {error_detail}

Original response:
{raw}

Please fix and respond with ONLY valid JSON matching the schema. Labels are: {valid_labels}"""
            continue

    return {
        "status": "judge_error",
        "parsed": None,
        "raw_response": raw,
        "latency_ms": latency_ms,
    }
