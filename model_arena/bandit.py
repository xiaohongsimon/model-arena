import random
from typing import Optional


def pairwise_updates(scores: dict[str, float], tie_epsilon: float) -> tuple[dict[str, float], dict[str, float]]:
    """Convert judge scores into pairwise win/loss credits."""
    wins = {m: 0.0 for m in scores}
    losses = {m: 0.0 for m in scores}
    models = list(scores)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            delta = scores[a] - scores[b]
            if abs(delta) <= tie_epsilon:
                wins[a] += 0.5
                wins[b] += 0.5
                losses[a] += 0.5
                losses[b] += 0.5
            elif delta > 0:
                wins[a] += 1.0
                losses[b] += 1.0
            else:
                wins[b] += 1.0
                losses[a] += 1.0
    return wins, losses


def select_models(
    eligible: list[str],
    n: int,
    history: dict[str, tuple[float, float]],
) -> list[str]:
    """Thompson Sampling selection. history maps model_id -> (total_wins, total_losses)."""
    if len(eligible) <= n:
        return list(eligible)

    samples: list[tuple[float, str]] = []
    for model_id in eligible:
        total_wins, total_losses = history.get(model_id, (0.0, 0.0))
        alpha = 1.0 + total_wins
        beta = 1.0 + total_losses
        sample = random.betavariate(alpha, beta)
        samples.append((sample, model_id))

    samples.sort(reverse=True)
    return [model_id for _, model_id in samples[:n]]


def compute_stats(records: list[tuple[str, float, float]]) -> dict[str, tuple[float, float]]:
    """Aggregate pairwise records into (total_wins, total_losses) per model."""
    stats: dict[str, tuple[float, float]] = {}
    for model_id, pw, pl in records:
        if model_id in stats:
            old_w, old_l = stats[model_id]
            stats[model_id] = (old_w + pw, old_l + pl)
        else:
            stats[model_id] = (pw, pl)
    return stats
