"""
/embedding endpoint — return the two-tower item embedding for a given item_id.
"""

from fastapi import APIRouter, HTTPException, Query
from src.api.dependencies import app_state

router = APIRouter()


@router.get("/embedding")
def get_embedding(item_id: int = Query(..., description="MovieLens item ID")) -> dict:
    state = app_state

    if state.item_features is None:
        raise HTTPException(status_code=503, detail="Item features not loaded.")
    if item_id not in state.item_features:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")

    embedding = None
    if state.item_embeddings is not None and state.item_id_map is not None:
        idx = state.item_id_map.get(item_id)
        if idx is not None and idx < len(state.item_embeddings):
            embedding = state.item_embeddings[idx].tolist()

    feat = state.item_features[item_id]
    return {
        "item_id": item_id,
        "title": feat.title,
        "genres": feat.genres,
        "avg_rating": feat.avg_rating,
        "embedding": embedding,
        "embedding_dim": len(embedding) if embedding else None,
    }
