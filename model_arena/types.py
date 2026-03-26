# model_arena/types.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    system_prompt: str
    judge_rubric_version: str
    default_n: int = 3
    fallback_policy: str = "best_single_success"  # raise | best_single_success
    quality_floor: float = 2.0
    tie_epsilon: float = 0.3


@dataclass
class ModelOutput:
    model_id: str
    label: str
    output: str
    score: Optional[float] = None
    pairwise_wins: float = 0.0
    pairwise_losses: float = 0.0
    latency_ms: int = 0


@dataclass
class ModelFailure:
    model_id: str
    status: str  # timeout | oom | api_error | invalid_output | skipped_context_limit
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int = 0


@dataclass
class ArenaResult:
    run_id: str
    best_output: Optional[str]
    winner_model_id: Optional[str]
    winner_score: Optional[float]
    judge_comment: str
    outputs: list[ModelOutput]
    failures: list[ModelFailure]
    degraded: bool
    degraded_reason: Optional[str]
    fallback_used: bool
    bandit_updated: bool
