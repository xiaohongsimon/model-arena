# Model Arena

Local multi-model best-of-N competition framework with Thompson Sampling MAB routing.

## Project Structure

```
model_arena/
  __init__.py   - compete() / compete_sync() orchestrator
  types.py      - TaskSpec, ArenaResult, ModelOutput, ModelFailure
  config.py     - models.yaml loading (mtime-cached) + validation
  bandit.py     - Thompson Sampling with pairwise updates
  storage.py    - SQLite WAL persistence (arena_runs + arena_attempts)
  pool.py       - Parallel inference + concurrency gate
  judge.py      - Opus judge protocol + schema validation + repair retry
  stats.py      - list_models(), get_stats() observability
```

## Key Design Decisions

- **Pairwise MAB** (not score-based): scale-invariant, only relative order matters
- **Per task_type routing**: each task type has independent bandit stats
- **Sync + async API**: `compete_sync()` for Prism's ThreadPoolExecutor, `compete()` for async callers
- **Config hot-reload**: mtime check per call, no restart needed
- **Bandit stats computed on-the-fly**: no cached mutable state, ~4 runs/day is trivially fast
- **Judge labels randomized per run**: prevents position bias

## Testing

```bash
pytest tests/ -v          # 49 tests, <0.1s
```

All tests use mocked HTTP. No real model inference in test suite.

## Environment Variables

- `ARENA_CONFIG` — path to models.yaml (default: `./models.yaml`)
- `ARENA_DB` — path to SQLite DB (default: `./arena.db`)
- `ARENA_JUDGE_API_KEY` — Anthropic API key for Opus judge

## Integration Points

- **Prism**: `compete_sync()` in `prism/pipeline/analyze.py` for incremental analysis
- **model-debate skill**: complementary — skill designs specs, arena executes at runtime
