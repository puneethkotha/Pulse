"""
/recommend endpoint.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies import app_state
from src.models.feature_builder import build_candidate_features
from src.data.schema import RecommendationResult

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    latency_ms: float
    retrieval_source: str
    online_features_used: bool


@router.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(..., description="MovieLens user ID"),
    k: int = Query(10, ge=1, le=100),
):
    t_start = time.perf_counter()
    state = app_state

    if state.retriever is None or state.item_features is None:
        raise HTTPException(status_code=503, detail="Retrieval system not ready.")

    if state.user_id_map is None or user_id not in state.user_id_map:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    cfg = state.config.get("recommendation", {})
    candidate_pool = cfg.get("candidate_pool_size", 100)
    use_reranker = cfg.get("use_reranker", True)

    # --- Get online features ---
    online_features = None
    online_features_used = False
    if state.feature_store and state.feature_store.available:
        online_feat = state.feature_store.get_user_features(user_id)
        if online_feat is not None:
            online_features = {user_id: online_feat}
            online_features_used = True

    # --- Encode user ---
    user_idx = state.user_id_map[user_id]
    user_feat_vec = torch.tensor(
        state.user_feat_matrix[user_idx],
        dtype=torch.float32,
    ).unsqueeze(0)
    user_idx_t = torch.tensor([user_idx], dtype=torch.long)

    with torch.no_grad():
        user_emb = state.two_tower_model.encode_user(user_idx_t, user_feat_vec)
    user_emb_np = user_emb.cpu().numpy()[0]

    # --- FAISS retrieval ---
    candidates = state.retriever.retrieve(user_emb_np, k=candidate_pool)

    if not candidates:
        return RecommendResponse(
            user_id=user_id,
            recommendations=[],
            latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieval_source="faiss",
            online_features_used=online_features_used,
        )

    # --- Re-rank ---
    if use_reranker and state.rerank_model is not None:
        pairs = [
            (user_id, iid, sim, rank)
            for rank, (iid, sim) in enumerate(candidates)
        ]
        feats = build_candidate_features(
            user_item_pairs=pairs,
            item_features=state.item_features,
            user_train_interactions={},      # empty — offline data not loaded at serve time
            online_features=online_features,
        )
        feat_t = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            scores = state.rerank_model(feat_t).cpu().numpy()

        ranked = sorted(
            zip([c[0] for c in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        ranked = candidates

    # --- Build response ---
    results = []
    for item_id, score in ranked[:k]:
        feat = state.item_features.get(item_id, {})
        retrieval_score = next((s for iid, s in candidates if iid == item_id), 0.0)
        results.append({
            "item_id": item_id,
            "score": round(float(score), 6),
            "retrieval_score": round(float(retrieval_score), 6),
            "title": getattr(feat, "title", ""),
            "genres": getattr(feat, "genres", []),
            "avg_rating": getattr(feat, "avg_rating", 0.0),
        })

    latency_ms = (time.perf_counter() - t_start) * 1000

    return RecommendResponse(
        user_id=user_id,
        recommendations=results,
        latency_ms=round(latency_ms, 2),
        retrieval_source="faiss+reranker" if (use_reranker and state.rerank_model) else "faiss",
        online_features_used=online_features_used,
    )
