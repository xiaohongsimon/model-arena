# Model Arena

Local multi-model best-of-N competition framework. Multiple local models compete on the same task, Opus judges, Thompson Sampling MAB routes future traffic to winners.

## How It Works

```
TaskSpec (task_type + rubric)
       |
       v
  +---------+      +----------+      +-------+      +--------+
  | Config  | ---> |  Bandit  | ---> | Pool  | ---> | Judge  |
  | models  |      | Thompson |      | N并行  |      | Opus   |
  | .yaml   |      | Sampling |      | 推理   |      | 评审    |
  +---------+      +----------+      +-------+      +--------+
                        ^                                |
                        |         pairwise updates       |
                        +--------------------------------+
                                      |
                                   Storage
                                  (SQLite)
```

## Quick Start

```python
from model_arena import compete_sync
from model_arena.types import TaskSpec

SIGNAL_ANALYSIS = TaskSpec(
    task_type="signal_analysis",
    system_prompt="你是一个信号分析专家...",
    judge_rubric_version="signal_analysis_v1",
    default_n=3,
)

result = compete_sync(
    task_spec=SIGNAL_ANALYSIS,
    user_prompt="分析这条推文的信号价值...",
)

print(result.best_output)       # winning output
print(result.winner_model_id)   # which model won
print(result.winner_score)      # judge score
```

## Architecture

| Module | File | Responsibility |
|--------|------|----------------|
| types | `types.py` | TaskSpec, ArenaResult, ModelOutput, ModelFailure |
| config | `config.py` | models.yaml loading (mtime-cached) + validation |
| bandit | `bandit.py` | Thompson Sampling with pairwise updates |
| storage | `storage.py` | SQLite WAL persistence |
| pool | `pool.py` | Parallel inference + concurrency gate |
| judge | `judge.py` | Opus judge protocol + schema validation + repair retry |
| stats | `stats.py` | list_models(), get_stats() observability |

## Relationship to multi-model-orchestrate Skill

This project and the `multi-model-orchestrate` CC skill serve **different layers**:

| | multi-model-orchestrate | model-arena |
|---|---|---|
| **What** | CC skill for human-in-the-loop multi-model collaboration | Library for automated model competition + routing |
| **Models** | Opus + Codex + Local 27B (fixed roles) | N local models (dynamic pool, interchangeable) |
| **Selection** | Manual preset (debate/best-of-n/verify) | Automatic via Thompson Sampling MAB |
| **Judge** | Opus as arbiter in the loop | Opus as post-hoc scorer |
| **Learning** | No — each run is independent | Yes — pairwise stats accumulate, better models get more traffic |
| **Use case** | Spec design, code review, content writing | Production inference routing (Prism analysis, translation, etc.) |

**They complement each other:**
- `multi-model-orchestrate` helps **design** the TaskSpec and judge rubric
- `model-arena` **executes** the competition at runtime, learning which models are best per task

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARENA_CONFIG` | `./models.yaml` | Path to model registry |
| `ARENA_DB` | `./arena.db` | Path to SQLite database |
| `ARENA_JUDGE_API_KEY` | (empty) | Anthropic API key for Opus judge |

### models.yaml

```yaml
models:
  qwen-27b:
    endpoint: http://127.0.0.1:8002/v1
    model_name: qwen3.5-27b-distilled
    api_key: omlx-xxx
    timeout_s: 90
    max_input_tokens: 64000
    max_output_tokens: 4000
    status: active    # active | retired
```

Adding a model = adding an entry (starts with neutral Beta(1,1) prior).
Retiring = set `status: retired` (stops selection, history preserved).

## Observability

```python
from model_arena.stats import list_models, get_stats

# Per-model MAB stats for a task type
list_models(storage, "signal_analysis")

# Aggregate run stats (win rates, failure rates, degradation)
get_stats(storage, "signal_analysis", since_days=30)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

49 tests, <0.1s. All HTTP calls mocked in tests.
