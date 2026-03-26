# tests/test_types.py
from model_arena.types import TaskSpec, ArenaResult, ModelOutput, ModelFailure


def test_taskspec_defaults():
    spec = TaskSpec(
        task_type="signal_analysis",
        system_prompt="You are an expert.",
        judge_rubric_version="v1",
    )
    assert spec.task_type == "signal_analysis"
    assert spec.default_n == 3
    assert spec.fallback_policy == "best_single_success"
    assert spec.quality_floor == 2.0
    assert spec.tie_epsilon == 0.3


def test_taskspec_frozen():
    spec = TaskSpec(
        task_type="test",
        system_prompt="test",
        judge_rubric_version="v1",
    )
    try:
        spec.task_type = "other"
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_model_output():
    out = ModelOutput(
        model_id="qwen-27b",
        label="A",
        output="analysis result",
        score=4.2,
        pairwise_wins=1.5,
        pairwise_losses=0.5,
        latency_ms=1200,
    )
    assert out.model_id == "qwen-27b"
    assert out.score == 4.2


def test_model_failure():
    fail = ModelFailure(
        model_id="glm-9b",
        status="timeout",
        error_type="TimeoutError",
        error_message="exceeded 60s",
        latency_ms=60000,
    )
    assert fail.status == "timeout"
    assert fail.model_id == "glm-9b"


def test_arena_result_fields():
    result = ArenaResult(
        run_id="run-001",
        best_output="winning text",
        winner_model_id="qwen-27b",
        winner_score=4.2,
        judge_comment="B captures the core signal.",
        outputs=[],
        failures=[],
        degraded=False,
        degraded_reason=None,
        fallback_used=False,
        bandit_updated=True,
    )
    assert result.run_id == "run-001"
    assert result.best_output == "winning text"
    assert result.degraded is False


def test_arena_result_all_failed():
    result = ArenaResult(
        run_id="run-002",
        best_output=None,
        winner_model_id=None,
        winner_score=None,
        judge_comment="",
        outputs=[],
        failures=[],
        degraded=True,
        degraded_reason="all models failed",
        fallback_used=False,
        bandit_updated=False,
    )
    assert result.best_output is None
    assert result.degraded is True
