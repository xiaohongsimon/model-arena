# tests/test_integration.py
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from model_arena import compete, compete_sync
from model_arena.types import TaskSpec, ArenaResult


@pytest.fixture
def task_spec():
    return TaskSpec(
        task_type="test_task",
        system_prompt="You are a test expert.",
        judge_rubric_version="test_v1",
        default_n=2,
    )


@pytest.fixture
def mock_config(tmp_path):
    """Create a temporary models.yaml and set env var."""
    import yaml
    data = {
        "models": {
            "model-a": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "test-a",
                "api_key": "key",
                "timeout_s": 60,
                "max_input_tokens": 100000,
                "max_output_tokens": 4000,
                "status": "active",
            },
            "model-b": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "test-b",
                "api_key": "key",
                "timeout_s": 60,
                "max_input_tokens": 100000,
                "max_output_tokens": 4000,
                "status": "active",
            },
        }
    }
    path = tmp_path / "models.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    db_path = str(tmp_path / "arena.db")
    return str(path), db_path


def test_compete_sync_success(task_spec, mock_config):
    config_path, db_path = mock_config

    # Mock inference: both models return success
    async def mock_run_inference(cfg, sys_prompt, user_prompt):
        return {
            "model_id": cfg.model_id,
            "status": "success",
            "output": f"output from {cfg.model_id}",
            "latency_ms": 500,
        }

    # Mock judge: model-a wins
    judge_result = {
        "status": "completed",
        "parsed": {
            "winner": "A",
            "scores": {"A": 4.5, "B": 3.2},
            "all_poor": False,
            "comment": "A is better.",
            "judge_rubric_version": "test_v1",
        },
        "raw_response": "{}",
        "latency_ms": 1000,
    }

    with patch("model_arena.pool.run_inference", side_effect=mock_run_inference), \
         patch("model_arena.judge.judge_outputs", return_value=judge_result), \
         patch.dict(os.environ, {"ARENA_CONFIG": config_path, "ARENA_DB": db_path, "ARENA_JUDGE_API_KEY": "test-key"}):

        result = compete_sync(
            task_spec=task_spec,
            user_prompt="test question",
        )

        assert isinstance(result, ArenaResult)
        assert result.best_output is not None
        assert result.winner_score == 4.5
        assert result.degraded is False
        assert result.bandit_updated is True


def test_compete_sync_all_fail_degraded(task_spec, mock_config):
    config_path, db_path = mock_config

    async def mock_run_inference(cfg, sys_prompt, user_prompt):
        return {
            "model_id": cfg.model_id,
            "status": "timeout",
            "latency_ms": 60000,
            "error_type": "TimeoutError",
            "error_message": "timeout",
        }

    with patch("model_arena.pool.run_inference", side_effect=mock_run_inference), \
         patch.dict(os.environ, {"ARENA_CONFIG": config_path, "ARENA_DB": db_path, "ARENA_JUDGE_API_KEY": "test-key"}):

        result = compete_sync(
            task_spec=task_spec,
            user_prompt="test question",
        )

        assert result.best_output is None
        assert result.degraded is True
        assert result.bandit_updated is False


def test_compete_sync_single_success_fallback(task_spec, mock_config):
    config_path, db_path = mock_config

    call_count = 0

    async def mock_run_inference(cfg, sys_prompt, user_prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "model_id": cfg.model_id,
                "status": "success",
                "output": "only one succeeded",
                "latency_ms": 500,
            }
        return {
            "model_id": cfg.model_id,
            "status": "timeout",
            "latency_ms": 60000,
            "error_type": "TimeoutError",
            "error_message": "timeout",
        }

    with patch("model_arena.pool.run_inference", side_effect=mock_run_inference), \
         patch.dict(os.environ, {"ARENA_CONFIG": config_path, "ARENA_DB": db_path, "ARENA_JUDGE_API_KEY": "test-key"}):

        result = compete_sync(
            task_spec=task_spec,
            user_prompt="test question",
        )

        assert result.best_output == "only one succeeded"
        assert result.degraded is True
        assert result.fallback_used is True
        assert result.bandit_updated is False
