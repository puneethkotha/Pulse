"""
FastAPI endpoint tests using TestClient.

These tests mock the AppState to avoid needing a trained model or running Redis.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_state():
    """Build a minimal AppState mock that satisfies endpoint logic."""
    from src.data.schema import ItemFeatures, UserFeatures, NUM_GENRES

    state = MagicMock()

    item_feat = ItemFeatures(
        item_id=1, title="Test Movie (2000)", year=2000,
        genres=["Action"], genre_vector=[0] * NUM_GENRES,
        avg_rating=4.0, num_ratings=200,
    )
    state.item_features = {1: item_feat, 2: item_feat}
    state.user_features = {
        1: UserFeatures(user_id=1, gender=0, age_bucket=2, occupation=5, zip_prefix=9)
    }
    state.user_id_map = {1: 0}
    state.item_id_map = {1: 0, 2: 1}
    state.user_feat_matrix = np.zeros((1, 30), dtype=np.float32)
    state.item_feat_matrix = np.zeros((2, 21), dtype=np.float32)
    state.item_embeddings = np.zeros((2, 64), dtype=np.float32)
    state.eval_metrics = {
        "bm25": {"ndcg@10": 0.12},
        "two_tower": {"ndcg@10": 0.19},
        "two_tower_reranker": {"ndcg@10": 0.22},
    }
    state.config = {"recommendation": {"candidate_pool_size": 10, "use_reranker": False}}
    state.feature_store = MagicMock()
    state.feature_store.available = False
    state.rerank_model = None

    # Mock two-tower encode_user
    import torch
    state.two_tower_model = MagicMock()
    state.two_tower_model.encode_user.return_value = torch.zeros(1, 64)

    # Mock retriever
    state.retriever = MagicMock()
    state.retriever.retrieve.return_value = [(1, 0.9), (2, 0.7)]

    return state


@pytest.fixture
def client():
    from src.api.main import app
    import src.api.dependencies as deps

    mock_state = make_mock_state()

    # Patch load() to be a no-op so the lifespan does not try to import FAISS
    # or load files from disk during unit tests.
    mock_state.load = MagicMock()

    with patch.object(deps, "app_state", mock_state), \
         patch("src.api.main.app_state", mock_state), \
         patch("src.api.routers.recommend.app_state", mock_state), \
         patch("src.api.routers.metrics.app_state", mock_state), \
         patch("src.api.routers.embeddings.app_state", mock_state), \
         patch("src.api.routers.events.app_state", mock_state):
        with TestClient(app) as c:
            yield c, mock_state


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        c, state = client
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestRecommendEndpoint:
    def test_valid_user_returns_recommendations(self, client):
        c, state = client
        resp = c.get("/recommend", params={"user_id": 1, "k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) == 2

    def test_unknown_user_returns_404(self, client):
        c, state = client
        resp = c.get("/recommend", params={"user_id": 9999, "k": 5})
        assert resp.status_code == 404

    def test_recommendations_have_required_fields(self, client):
        c, state = client
        resp = c.get("/recommend", params={"user_id": 1, "k": 2})
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        for rec in recs:
            assert "item_id" in rec
            assert "score" in rec
            assert "title" in rec
            assert "genres" in rec

    def test_latency_ms_present(self, client):
        c, state = client
        resp = c.get("/recommend", params={"user_id": 1, "k": 2})
        assert "latency_ms" in resp.json()


class TestMetricsEndpoint:
    def test_returns_metrics(self, client):
        c, state = client
        resp = c.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "bm25" in data
        assert "two_tower" in data


class TestEmbeddingEndpoint:
    def test_valid_item_returns_embedding(self, client):
        c, state = client
        resp = c.get("/embedding", params={"item_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["item_id"] == 1
        assert "title" in data
        assert "genres" in data

    def test_unknown_item_returns_404(self, client):
        c, state = client
        resp = c.get("/embedding", params={"item_id": 9999})
        assert resp.status_code == 404


class TestEventEndpoint:
    def test_valid_event_accepted(self, client):
        c, state = client
        resp = c.post("/event", json={
            "user_id": 1,
            "item_id": 1,
            "event_type": "rate",
            "rating": 4.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "event_id" in data

    def test_invalid_event_type_rejected(self, client):
        c, state = client
        resp = c.post("/event", json={
            "user_id": 1,
            "item_id": 1,
            "event_type": "invalid",
        })
        assert resp.status_code == 422

    def test_invalid_rating_rejected(self, client):
        c, state = client
        resp = c.post("/event", json={
            "user_id": 1,
            "item_id": 1,
            "event_type": "rate",
            "rating": 10.0,
        })
        assert resp.status_code == 422

    def test_view_event_no_rating(self, client):
        c, state = client
        resp = c.post("/event", json={
            "user_id": 1,
            "item_id": 1,
            "event_type": "view",
        })
        assert resp.status_code == 200
