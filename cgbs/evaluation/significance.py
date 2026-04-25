from __future__ import annotations

import random
from typing import Callable, Sequence


def paired_randomization_test(
    system_a_scores: Sequence[float],
    system_b_scores: Sequence[float],
    num_samples: int = 10000,
    seed: int = 42,
) -> float:
    """
    Paired randomization test.

    Args:
        system_a_scores:
            Per-query scores of system A.
        system_b_scores:
            Per-query scores of system B.
        num_samples:
            Number of random swaps.
        seed:
            Random seed.

    Returns:
        p-value. A small p-value suggests the difference is significant.
    """
    if len(system_a_scores) != len(system_b_scores):
        raise ValueError("Two score lists must have the same length.")

    if len(system_a_scores) == 0:
        raise ValueError("Score lists are empty.")

    rng = random.Random(seed)

    diffs = [a - b for a, b in zip(system_a_scores, system_b_scores)]
    observed = abs(sum(diffs) / len(diffs))

    count = 0
    for _ in range(num_samples):
        randomized_sum = 0.0
        for diff in diffs:
            if rng.random() < 0.5:
                randomized_sum += diff
            else:
                randomized_sum -= diff

        randomized = abs(randomized_sum / len(diffs))
        if randomized >= observed:
            count += 1

    return (count + 1) / (num_samples + 1)


def per_query_hits_at_k(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> list[float]:
    """Compute per-query Hits@K scores."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    scores = []
    for pred_list, target in zip(predictions, targets):
        scores.append(1.0 if target in pred_list[:k] else 0.0)
    return scores


def per_query_mrr_at_k(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> list[float]:
    """Compute per-query reciprocal ranks at K."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    scores = []
    for pred_list, target in zip(predictions, targets):
        rr = 0.0
        for rank, pred in enumerate(pred_list[:k], start=1):
            if pred == target:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return scores


def significance_test_for_metric(
    predictions_a: Sequence[Sequence[str]],
    predictions_b: Sequence[Sequence[str]],
    targets: Sequence[str],
    metric: str = "mrr",
    k: int = 20,
    num_samples: int = 10000,
    seed: int = 42,
) -> float:
    """
    Run paired randomization test for Hits@K or MRR@K.
    """
    metric = metric.lower()

    if metric in {"mrr", "mrr@k"}:
        scores_a = per_query_mrr_at_k(predictions_a, targets, k)
        scores_b = per_query_mrr_at_k(predictions_b, targets, k)
    elif metric in {"hits", "hit", "hits@k"}:
        scores_a = per_query_hits_at_k(predictions_a, targets, k)
        scores_b = per_query_hits_at_k(predictions_b, targets, k)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return paired_randomization_test(
        system_a_scores=scores_a,
        system_b_scores=scores_b,
        num_samples=num_samples,
        seed=seed,
    )