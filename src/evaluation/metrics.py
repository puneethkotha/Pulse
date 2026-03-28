"""
Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K.

All functions accept ranked lists of item IDs and relevant item ID sets.
"""

import math
from typing import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K items that are relevant."""
    if k == 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized DCG at K."""
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal_dcg


def compute_all_metrics(
    ranked: Sequence[int],
    relevant: set[int],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """
    Compute all configured metrics for a single query.

    Returns a flat dict like {"precision@5": 0.4, "ndcg@10": 0.6, ...}
    """
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return metrics


def aggregate_metrics(per_user: list[dict[str, float]]) -> dict[str, float]:
    """
    Average per-user metrics across all users.

    Users with no ground-truth items are skipped.
    """
    if not per_user:
        return {}
    keys = per_user[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in per_user]
        result[key] = round(sum(values) / len(values), 6)
    return result
