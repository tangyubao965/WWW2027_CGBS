from __future__ import annotations

from typing import Dict, List, Sequence


def hits_at_k(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    k: int,
) -> float:
    """Compute Hits@K."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    if len(targets) == 0:
        return 0.0

    hit_count = 0
    for pred_list, target in zip(predictions, targets):
        if target in pred_list[:k]:
            hit_count += 1

    return hit_count / len(targets)


def mrr_at_k(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    k: int,
) -> float:
    """Compute MRR@K."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    if len(targets) == 0:
        return 0.0

    total_rr = 0.0

    for pred_list, target in zip(predictions, targets):
        reciprocal_rank = 0.0
        for rank, pred in enumerate(pred_list[:k], start=1):
            if pred == target:
                reciprocal_rank = 1.0 / rank
                break
        total_rr += reciprocal_rank

    return total_rr / len(targets)


def evaluate_retrieval(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    ks: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Compute retrieval metrics."""
    results: Dict[str, float] = {}

    for k in ks:
        results[f"Hits@{k}"] = hits_at_k(predictions, targets, k)
        results[f"MRR@{k}"] = mrr_at_k(predictions, targets, k)

    return results


def format_metrics(metrics: Dict[str, float], scale: float = 100.0) -> str:
    """Format metrics for logging."""
    parts = []
    for key, value in metrics.items():
        parts.append(f"{key}: {value * scale:.2f}")
    return " | ".join(parts)