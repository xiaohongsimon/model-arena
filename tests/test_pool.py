import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
from model_arena.pool import estimate_tokens, filter_eligible, run_inference
from model_arena.config import ModelConfig


def _make_config(model_id, max_input_tokens=32000, status="active", timeout_s=60):
    return ModelConfig(
        model_id=model_id,
        endpoint="http://127.0.0.1:8002/v1",
        model_name=f"test-{model_id}",
        api_key="key",
        timeout_s=timeout_s,
        max_input_tokens=max_input_tokens,
        max_output_tokens=4000,
        status=status,
    )


def test_estimate_tokens():
    # Rough: ~4 chars per token for English, ~2 chars per token for Chinese
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("a" * 400) >= 50  # at least ~100 tokens
    assert estimate_tokens("") == 0


def test_filter_eligible_removes_short_context():
    models = {
        "big": _make_config("big", max_input_tokens=100000),
        "small": _make_config("small", max_input_tokens=100),
    }
    prompt_tokens = 500  # exceeds small's limit
    eligible, skipped = filter_eligible(models, prompt_tokens)
    assert "big" in eligible
    assert "small" not in eligible
    assert "small" in skipped


def test_filter_eligible_all_eligible():
    models = {
        "a": _make_config("a", max_input_tokens=100000),
        "b": _make_config("b", max_input_tokens=100000),
    }
    eligible, skipped = filter_eligible(models, 500)
    assert len(eligible) == 2
    assert len(skipped) == 0


def test_run_inference_success():
    async def go():
        config = _make_config("m1")

        mock_response = {
            "choices": [{"message": {"content": "analysis result"}}]
        }

        async def mock_post(url, **kwargs):
            mock = AsyncMock()
            mock.status_code = 200
            mock.json = lambda: mock_response
            mock.raise_for_status = lambda: None
            return mock

        with patch("model_arena.pool.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.post = mock_post
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client_instance

            result = await run_inference(
                config, "system prompt", "user prompt"
            )
            assert result["status"] == "success"
            assert result["output"] == "analysis result"
            assert result["latency_ms"] >= 0

    asyncio.run(go())


def test_run_inference_timeout():
    async def go():
        config = _make_config("m1", timeout_s=1)

        async def mock_post(url, **kwargs):
            import httpx
            raise httpx.ReadTimeout("timeout")

        with patch("model_arena.pool.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.post = mock_post
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client_instance

            result = await run_inference(
                config, "system prompt", "user prompt"
            )
            assert result["status"] == "timeout"

    asyncio.run(go())


def test_run_inference_empty_output():
    async def go():
        config = _make_config("m1")

        mock_response = {
            "choices": [{"message": {"content": ""}}]
        }

        async def mock_post(url, **kwargs):
            mock = AsyncMock()
            mock.status_code = 200
            mock.json = lambda: mock_response
            mock.raise_for_status = lambda: None
            return mock

        with patch("model_arena.pool.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.post = mock_post
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = client_instance

            result = await run_inference(
                config, "system prompt", "user prompt"
            )
            assert result["status"] == "invalid_output"

    asyncio.run(go())
