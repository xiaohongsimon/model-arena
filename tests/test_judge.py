import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
from model_arena.judge import (
    build_judge_prompt,
    parse_judge_response,
    validate_judge_schema,
    randomize_labels,
    judge_outputs,
)
from model_arena.types import TaskSpec


@pytest.fixture
def task_spec():
    return TaskSpec(
        task_type="signal_analysis",
        system_prompt="You are an expert.",
        judge_rubric_version="signal_analysis_v1",
    )


def test_randomize_labels():
    model_ids = ["qwen-27b", "glm-9b", "minimax-m2"]
    labels, mapping = randomize_labels(model_ids)
    assert set(labels.values()) == {"A", "B", "C"}
    assert set(labels.keys()) == set(model_ids)
    # mapping is label -> model_id (inverse)
    assert len(mapping) == 3


def test_build_judge_prompt(task_spec):
    outputs = {"A": "output one", "B": "output two"}
    prompt = build_judge_prompt(task_spec, "user question", outputs)
    assert "output one" in prompt
    assert "output two" in prompt
    assert "signal_analysis_v1" in prompt
    # Should NOT contain model IDs
    assert "qwen" not in prompt.lower()


def test_validate_judge_schema_valid():
    result = {
        "winner": "A",
        "scores": {"A": 4.2, "B": 3.1},
        "all_poor": False,
        "comment": "A is better.",
        "judge_rubric_version": "v1",
    }
    errors = validate_judge_schema(result, valid_labels=["A", "B"])
    assert errors == []


def test_validate_judge_schema_winner_not_in_labels():
    result = {
        "winner": "C",
        "scores": {"A": 4.0, "B": 3.0},
        "all_poor": False,
        "comment": "test",
        "judge_rubric_version": "v1",
    }
    errors = validate_judge_schema(result, valid_labels=["A", "B"])
    assert any("winner" in e for e in errors)


def test_validate_judge_schema_score_out_of_range():
    result = {
        "winner": "A",
        "scores": {"A": 6.0, "B": 3.0},
        "all_poor": False,
        "comment": "test",
        "judge_rubric_version": "v1",
    }
    errors = validate_judge_schema(result, valid_labels=["A", "B"])
    assert any("score" in e.lower() for e in errors)


def test_validate_judge_schema_null_winner_with_all_poor():
    result = {
        "winner": None,
        "scores": {"A": 1.0, "B": 1.5},
        "all_poor": True,
        "comment": "All outputs are poor.",
        "judge_rubric_version": "v1",
    }
    errors = validate_judge_schema(result, valid_labels=["A", "B"])
    assert errors == []


def test_validate_judge_schema_missing_label_score():
    result = {
        "winner": "A",
        "scores": {"A": 4.0},  # missing B
        "all_poor": False,
        "comment": "test",
        "judge_rubric_version": "v1",
    }
    errors = validate_judge_schema(result, valid_labels=["A", "B"])
    assert any("B" in e for e in errors)


def test_parse_judge_response_valid_json():
    raw = json.dumps({
        "winner": "A",
        "scores": {"A": 4.0, "B": 3.0},
        "all_poor": False,
        "comment": "A wins.",
        "judge_rubric_version": "v1",
    })
    result, error = parse_judge_response(raw)
    assert result is not None
    assert error is None
    assert result["winner"] == "A"


def test_parse_judge_response_json_in_markdown():
    raw = '```json\n{"winner": "A", "scores": {"A": 4.0, "B": 3.0}, "all_poor": false, "comment": "test", "judge_rubric_version": "v1"}\n```'
    result, error = parse_judge_response(raw)
    assert result is not None
    assert error is None


def test_parse_judge_response_invalid():
    result, error = parse_judge_response("not json at all")
    assert result is None
    assert error is not None
