import pytest
import asyncio
from model_arena.storage import ArenaStorage


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_arena.db")


def _run(coro):
    """Run a coroutine synchronously, compatible with Python 3.10+."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def storage(db_path):
    s = ArenaStorage(db_path)
    _run(s.init())
    yield s
    _run(s.close())


def test_init_creates_tables(storage):
    async def check():
        rows = await storage.fetch_all("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = [r[0] for r in rows]
        assert "arena_runs" in names
        assert "arena_attempts" in names
    _run(check())


def test_save_and_load_run(storage):
    async def go():
        run_data = {
            "id": "run-001",
            "task_type": "signal_analysis",
            "judge_rubric_version": "v1",
            "prompt_hash": "abc123",
            "metadata_json": "{}",
            "requested_n": 3,
            "eligible_n": 3,
            "selected_n": 3,
            "success_count": 2,
            "judge_status": "completed",
            "degraded": 0,
            "all_poor": 0,
            "fallback_used": 0,
            "bandit_updated": 1,
            "status": "completed",
        }
        await storage.save_run(run_data)
        row = await storage.get_run("run-001")
        assert row is not None
        assert row["task_type"] == "signal_analysis"
        assert row["status"] == "completed"
    _run(go())


def test_save_and_load_attempts(storage):
    async def go():
        # Save run first
        run_data = {
            "id": "run-002",
            "task_type": "test",
            "judge_rubric_version": "v1",
            "prompt_hash": "xyz",
            "metadata_json": "{}",
            "requested_n": 2,
            "eligible_n": 2,
            "selected_n": 2,
            "success_count": 1,
            "judge_status": "completed",
            "degraded": 0,
            "all_poor": 0,
            "fallback_used": 0,
            "bandit_updated": 0,
            "status": "completed",
        }
        await storage.save_run(run_data)

        attempts = [
            {
                "run_id": "run-002",
                "model_id": "m1",
                "label": "A",
                "status": "success",
                "latency_ms": 1200,
                "output": "output text",
                "score": 4.0,
                "pairwise_wins": 1.0,
                "pairwise_losses": 0.0,
                "is_winner": 1,
            },
            {
                "run_id": "run-002",
                "model_id": "m2",
                "label": "B",
                "status": "timeout",
                "latency_ms": 60000,
                "is_winner": 0,
            },
        ]
        await storage.save_attempts(attempts)
        rows = await storage.get_attempts("run-002")
        assert len(rows) == 2
        assert rows[0]["model_id"] in ("m1", "m2")
    _run(go())


def test_get_bandit_history(storage):
    async def go():
        # Create run + attempts with pairwise data
        await storage.save_run({
            "id": "run-003", "task_type": "signal_analysis",
            "judge_rubric_version": "v1", "prompt_hash": "h",
            "metadata_json": "{}", "requested_n": 2, "eligible_n": 2,
            "selected_n": 2, "success_count": 2, "judge_status": "completed",
            "degraded": 0, "all_poor": 0, "fallback_used": 0,
            "bandit_updated": 1, "status": "completed",
        })
        await storage.save_attempts([
            {"run_id": "run-003", "model_id": "m1", "label": "A",
             "status": "success", "latency_ms": 100, "output": "x",
             "score": 4.0, "pairwise_wins": 1.0, "pairwise_losses": 0.0,
             "is_winner": 1},
            {"run_id": "run-003", "model_id": "m2", "label": "B",
             "status": "success", "latency_ms": 100, "output": "y",
             "score": 3.0, "pairwise_wins": 0.0, "pairwise_losses": 1.0,
             "is_winner": 0},
        ])

        records = await storage.get_bandit_history("signal_analysis")
        assert len(records) == 2
        # records: list of (model_id, pairwise_wins, pairwise_losses)
        m1_rec = [r for r in records if r[0] == "m1"][0]
        assert m1_rec[1] == 1.0  # wins
        assert m1_rec[2] == 0.0  # losses
    _run(go())
