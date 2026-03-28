"""
Full evaluation harness.

Compares BM25, two-tower retrieval, and two-tower + re-ranker.
Saves results to artifacts/metrics/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.data.schema import ItemFeatures, UserFeatures

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/metrics")
KS = (5, 10, 20)


def build_user_ground_truth(test_df: pd.DataFrame, min_rating: float = 4.0) -> dict[int, set[int]]:
    """
    Map user_id -> set of item_ids rated >= min_rating in the test set.
    """
    gt: dict[int, set[int]] = {}
    for row in test_df[test_df["rating"] >= min_rating].itertuples(index=False):
        gt.setdefault(row.user_id, set()).add(row.item_id)
    return gt


def build_user_seen_items(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[int, set[int]]:
    """Items a user has already seen (train + val) — excluded from retrieved results."""
    seen: dict[int, set[int]] = {}
    for df in (train_df, val_df):
        for row in df.itertuples(index=False):
            seen.setdefault(row.user_id, set()).add(row.item_id)
    return seen


def evaluate_bm25(
    bm25_retriever,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_features: dict[int, ItemFeatures],
    k_eval: int = 20,
    ks: tuple = KS,
    max_users: Optional[int] = None,
) -> dict[str, float]:
    from src.models.bm25_baseline import build_user_query

    ground_truth = build_user_ground_truth(test_df)
    seen_items = build_user_seen_items(train_df, val_df)

    all_interactions = [
        type("I", (), {"user_id": r.user_id, "item_id": r.item_id, "rating": r.rating})()
        for r in train_df.itertuples(index=False)
    ]

    users = list(ground_truth.keys())
    if max_users:
        users = users[:max_users]

    per_user = []
    t0 = time.perf_counter()
    for user_id in users:
        relevant = ground_truth[user_id]
        if not relevant:
            continue
        query = build_user_query(user_id, all_interactions, item_features)
        exclude = seen_items.get(user_id, set())
        ranked_with_scores = bm25_retriever.retrieve(query, k=k_eval, exclude_item_ids=exclude)
        ranked = [iid for iid, _ in ranked_with_scores]
        per_user.append(compute_all_metrics(ranked, relevant, ks))

    elapsed = time.perf_counter() - t0
    result = aggregate_metrics(per_user)
    result["num_users_evaluated"] = len(per_user)
    result["avg_latency_ms"] = round(elapsed / max(len(per_user), 1) * 1000, 3)
    return result


def evaluate_two_tower(
    retriever,
    two_tower_model,
    user_id_map: dict[int, int],
    item_id_map: dict[int, int],
    user_feat_matrix: np.ndarray,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    k_eval: int = 20,
    ks: tuple = KS,
    max_users: Optional[int] = None,
) -> dict[str, float]:
    ground_truth = build_user_ground_truth(test_df)
    seen_items = build_user_seen_items(train_df, val_df)

    users = [u for u in ground_truth if u in user_id_map]
    if max_users:
        users = users[:max_users]

    per_user = []
    t0 = time.perf_counter()

    two_tower_model.eval()
    with torch.no_grad():
        for user_id in users:
            relevant = ground_truth[user_id]
            if not relevant:
                continue

            user_idx = user_id_map[user_id]
            user_feat = torch.tensor(user_feat_matrix[user_idx], dtype=torch.float32).unsqueeze(0)
            user_idx_t = torch.tensor([user_idx], dtype=torch.long)
            user_emb = two_tower_model.encode_user(user_idx_t, user_feat).cpu().numpy()[0]

            exclude = seen_items.get(user_id, set())
            candidates = retriever.retrieve(user_emb, k=k_eval + len(exclude))

            ranked = [iid for iid, _ in candidates if iid not in exclude][:k_eval]
            per_user.append(compute_all_metrics(ranked, relevant, ks))

    elapsed = time.perf_counter() - t0
    result = aggregate_metrics(per_user)
    result["num_users_evaluated"] = len(per_user)
    result["avg_latency_ms"] = round(elapsed / max(len(per_user), 1) * 1000, 3)
    return result


def evaluate_two_tower_plus_reranker(
    retriever,
    two_tower_model,
    rerank_model,
    user_id_map: dict[int, int],
    item_id_map: dict[int, int],
    user_feat_matrix: np.ndarray,
    item_features: dict[int, ItemFeatures],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate_pool: int = 100,
    k_eval: int = 20,
    ks: tuple = KS,
    max_users: Optional[int] = None,
) -> dict[str, float]:
    from src.models.feature_builder import build_candidate_features

    ground_truth = build_user_ground_truth(test_df)
    seen_items = build_user_seen_items(train_df, val_df)

    # Build user interaction history for genre features
    user_train_interactions: dict[int, list] = {}
    for row in train_df.itertuples(index=False):
        user_train_interactions.setdefault(row.user_id, []).append(
            type("I", (), {"user_id": row.user_id, "item_id": row.item_id, "rating": row.rating})()
        )

    users = [u for u in ground_truth if u in user_id_map]
    if max_users:
        users = users[:max_users]

    per_user = []
    t0 = time.perf_counter()

    two_tower_model.eval()
    rerank_model.eval()

    with torch.no_grad():
        for user_id in users:
            relevant = ground_truth[user_id]
            if not relevant:
                continue

            user_idx = user_id_map[user_id]
            user_feat = torch.tensor(user_feat_matrix[user_idx], dtype=torch.float32).unsqueeze(0)
            user_idx_t = torch.tensor([user_idx], dtype=torch.long)
            user_emb = two_tower_model.encode_user(user_idx_t, user_feat).cpu().numpy()[0]

            exclude = seen_items.get(user_id, set())
            candidates = retriever.retrieve(user_emb, k=candidate_pool + len(exclude))
            candidates = [(iid, s) for iid, s in candidates if iid not in exclude][:candidate_pool]

            if not candidates:
                continue

            pairs = [(user_id, iid, sim, rank) for rank, (iid, sim) in enumerate(candidates)]
            feats = build_candidate_features(
                pairs, item_features, user_train_interactions
            )
            feat_t = torch.tensor(feats, dtype=torch.float32)
            scores = rerank_model(feat_t).cpu().numpy()

            ranked_pairs = sorted(
                zip([c[0] for c in candidates], scores),
                key=lambda x: x[1], reverse=True,
            )
            ranked = [iid for iid, _ in ranked_pairs[:k_eval]]
            per_user.append(compute_all_metrics(ranked, relevant, ks))

    elapsed = time.perf_counter() - t0
    result = aggregate_metrics(per_user)
    result["num_users_evaluated"] = len(per_user)
    result["avg_latency_ms"] = round(elapsed / max(len(per_user), 1) * 1000, 3)
    return result


def save_results(results: dict, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save evaluation results as JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    lines = ["# Evaluation Results\n"]
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        lines.append(f"\n## {model_name}\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")

    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved evaluation results to %s", output_dir)
