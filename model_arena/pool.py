import asyncio
import time
from typing import Optional
import httpx
from model_arena.config import ModelConfig

# Concurrency gate: limits parallel arena rounds
_semaphore: Optional[asyncio.Semaphore] = None
MAX_CONCURRENT_ROUNDS = 2


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROUNDS)
    return _semaphore


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token English, ~2 chars/token Chinese."""
    if not text:
        return 0
    # Count Chinese characters
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def filter_eligible(
    models: dict[str, ModelConfig], prompt_tokens: int
) -> tuple[dict[str, ModelConfig], list[str]]:
    """Filter models by context window. Returns (eligible, skipped_ids)."""
    eligible = {}
    skipped = []
    for model_id, cfg in models.items():
        if prompt_tokens > cfg.max_input_tokens:
            skipped.append(model_id)
        else:
            eligible[model_id] = cfg
    return eligible, skipped


async def run_inference(
    config: ModelConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Call a single model. Returns dict with status, output, latency_ms, etc."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=config.timeout_s) as client:
            resp = await client.post(
                f"{config.endpoint}/chat/completions",
                json={
                    "model": config.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": config.max_output_tokens,
                },
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            output = data["choices"][0]["message"]["content"]
            latency_ms = int((time.monotonic() - start) * 1000)

            if not output or not output.strip():
                return {
                    "model_id": config.model_id,
                    "status": "invalid_output",
                    "output": output,
                    "latency_ms": latency_ms,
                    "error_type": "EmptyOutput",
                    "error_message": "Model returned empty output",
                }

            return {
                "model_id": config.model_id,
                "status": "success",
                "output": output,
                "latency_ms": latency_ms,
            }

    except httpx.TimeoutException:
        return {
            "model_id": config.model_id,
            "status": "timeout",
            "latency_ms": int((time.monotonic() - start) * 1000),
            "error_type": "TimeoutError",
            "error_message": f"exceeded {config.timeout_s}s",
        }
    except Exception as e:
        return {
            "model_id": config.model_id,
            "status": "api_error",
            "latency_ms": int((time.monotonic() - start) * 1000),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


async def run_all(
    configs: list[ModelConfig],
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """Run inference on all selected models concurrently, behind semaphore."""
    sem = _get_semaphore()
    async with sem:
        tasks = [
            run_inference(cfg, system_prompt, user_prompt)
            for cfg in configs
        ]
        return await asyncio.gather(*tasks)
