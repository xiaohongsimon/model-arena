# tests/test_smoke.py
"""End-to-end smoke test: real config shape, mocked HTTP."""
import asyncio
import json
import os
import yaml
import pytest
from unittest.mock import AsyncMock, patch
from model_arena import compete_sync
from model_arena.types import TaskSpec
from model_arena.config import validate_config


@pytest.fixture
def real_config(tmp_path):
    """Mirror the actual models.yaml shape."""
    data = {
        "models": {
            "qwen-27b": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "qwen3.5-27b-distilled",
                "api_key": "omlx-xxx",
                "timeout_s": 90,
                "max_input_tokens": 64000,
                "max_output_tokens": 4000,
                "status": "active",
            },
            "glm-9b": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "glm-4.7-flash",
                "api_key": "omlx-xxx",
                "timeout_s": 60,
                "max_input_tokens": 32000,
                "max_output_tokens": 4000,
                "status": "active",
            },
            "minimax-m2": {
                "endpoint": "http://127.0.0.1:8002/v1",
                "model_name": "minimax-m2.5",
                "api_key": "omlx-xxx",
                "timeout_s": 60,
                "max_input_tokens": 128000,
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


def test_validate_real_config(real_config):
    config_path, _ = real_config
    errors = validate_config(config_path)
    assert errors == [], f"Config validation errors: {errors}"


def test_full_flow_3_models(real_config):
    config_path, db_path = real_config

    spec = TaskSpec(
        task_type="signal_analysis",
        system_prompt="You are an expert.",
        judge_rubric_version="signal_analysis_v1",
        default_n=3,
    )

    call_counter = {"n": 0}

    async def mock_run_inference(cfg, sys_prompt, user_prompt):
        call_counter["n"] += 1
        return {
            "model_id": cfg.model_id,
            "status": "success",
            "output": f"Analysis from {cfg.model_id}: signal is bullish.",
            "latency_ms": 800 + call_counter["n"] * 200,
        }

    judge_response = {
        "status": "completed",
        "parsed": {
            "winner": "B",
            "scores": {"A": 3.5, "B": 4.2, "C": 2.8},
            "all_poor": False,
            "comment": "B captures the core signal most precisely.",
            "judge_rubric_version": "signal_analysis_v1",
        },
        "raw_response": "{}",
        "latency_ms": 2000,
    }

    with patch("model_arena.pool.run_inference", side_effect=mock_run_inference), \
         patch("model_arena.judge.judge_outputs", return_value=judge_response), \
         patch.dict(os.environ, {"ARENA_CONFIG": config_path, "ARENA_DB": db_path, "ARENA_JUDGE_API_KEY": "test"}):

        result = compete_sync(task_spec=spec, user_prompt="分析这条推文的信号价值")

        assert result.best_output is not None
        assert "signal" in result.best_output.lower() or "Analysis" in result.best_output
        assert result.winner_score == 4.2
        assert result.degraded is False
        assert result.bandit_updated is True
        assert len(result.outputs) == 3
        assert len(result.failures) == 0
        assert call_counter["n"] == 3  # all 3 models were called
