from model_arena.storage import ArenaStorage
from model_arena.bandit import compute_stats


async def list_models(storage: ArenaStorage, task_type: str) -> dict[str, dict]:
    """List all models with their MAB stats for a given task type."""
    records = await storage.get_bandit_history(task_type)
    stats = compute_stats(records)

    result = {}
    for model_id, (wins, losses) in stats.items():
        total = wins + losses
        result[model_id] = {
            "total_wins": wins,
            "total_losses": losses,
            "total_rounds": total,
            "win_rate": wins / total if total > 0 else 0.0,
        }
    return result


async def get_stats(storage: ArenaStorage, task_type: str, since_days: int = 30) -> dict:
    """Get aggregate stats for a task type."""
    runs = await storage.fetch_all(
        """SELECT COUNT(*) as total,
                  SUM(degraded) as degraded,
                  SUM(fallback_used) as fallback,
                  SUM(all_poor) as all_poor
           FROM arena_runs
           WHERE task_type = ?
             AND ts >= datetime('now', ? || ' days')""",
        (task_type, f"-{since_days}"),
    )
    row = runs[0] if runs else None
    total = row[0] if row else 0
    degraded = row[1] if row and row[1] else 0
    fallback = row[2] if row and row[2] else 0
    all_poor = row[3] if row and row[3] else 0

    models = await list_models(storage, task_type)

    return {
        "task_type": task_type,
        "total_runs": total,
        "degraded_runs": degraded,
        "fallback_runs": fallback,
        "all_poor_runs": all_poor,
        "models": models,
    }
