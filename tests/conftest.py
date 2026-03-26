# tests/conftest.py
import pytest
from model_arena.types import TaskSpec


@pytest.fixture
def sample_task_spec():
    return TaskSpec(
        task_type="signal_analysis",
        system_prompt="You are a signal analysis expert.",
        judge_rubric_version="signal_analysis_v1",
        default_n=3,
    )
