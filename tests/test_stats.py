import asyncio
import pytest
from model_arena.storage import ArenaStorage
from model_arena.stats import list_models, get_stats


def _run(coro):
    """Run a coroutine synchronously, compatible with Python 3.10+."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_arena.db")


@pytest.fixture
def storage(db_path):
    s = ArenaStorage(db_path)
    _run(s.init())
    yield s
    _run(s.close())


def _seed_data(storage):
    """Insert sample runs and attempts."""
    async def go():
        for i in range(3):
            await storage.save_run({
                "id": f"run-{i}", "task_type": "signal_analysis",
                "judge_rubric_version": "v1", "prompt_hash": f"h{i}",
                "metadata_json": "{}", "requested_n": 2, "eligible_n": 2,
                "selected_n": 2, "success_count": 2, "judge_status": "completed",
                "degraded": 0, "all_poor": 0, "fallback_used": 0,
                "bandit_updated": 1, "status": "completed",
            })
            await storage.save_attempts([
                {"run_id": f"run-{i}", "model_id": "m1", "label": "A",
                 "status": "success", "latency_ms": 1000 + i * 100,
                 "output": "x", "score": 4.0, "pairwise_wins": 1.0,
                 "pairwise_losses": 0.0, "is_winner": 1},
                {"run_id": f"run-{i}", "model_id": "m2", "label": "B",
                 "status": "success", "latency_ms": 2000 + i * 100,
                 "output": "y", "score": 3.0, "pairwise_wins": 0.0,
                 "pairwise_losses": 1.0, "is_winner": 0},
            ])
    _run(go())


def test_list_models(storage):
    _seed_data(storage)

    async def go():
        result = await list_models(storage, "signal_analysis")
        assert "m1" in result
        assert "m2" in result
        assert result["m1"]["total_wins"] == 3.0
        assert result["m2"]["total_losses"] == 3.0
    _run(go())


def test_get_stats(storage):
    _seed_data(storage)

    async def go():
        stats = await get_stats(storage, "signal_analysis")
        assert stats["total_runs"] == 3
        assert stats["degraded_runs"] == 0
        assert "models" in stats
    _run(go())


def test_get_stats_empty(storage):
    async def go():
        stats = await get_stats(storage, "nonexistent")
        assert stats["total_runs"] == 0
    _run(go())
