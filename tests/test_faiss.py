"""Tests for FAISS index builder and retriever."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.faiss_index import (
    build_flat_index,
    build_ivf_index,
    save_index,
    load_index,
    FaissRetriever,
)


def make_embeddings(n: int = 100, dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    embs = rng.randn(n, dim).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


class TestFlatIndex:
    def test_build_and_search(self):
        embs = make_embeddings(50, 32)
        index = build_flat_index(embs.copy())
        assert index.ntotal == 50

        query = embs[0].reshape(1, -1)
        import faiss
        faiss.normalize_L2(query)
        scores, indices = index.search(query, k=5)
        # top-1 should be the query itself
        assert indices[0][0] == 0

    def test_scores_in_cosine_range(self):
        embs = make_embeddings(20, 16)
        index = build_flat_index(embs.copy())
        query = embs[5].reshape(1, -1)
        import faiss
        faiss.normalize_L2(query)
        scores, _ = index.search(query, k=10)
        assert np.all(scores <= 1.01)
        assert np.all(scores >= -1.01)


class TestIVFIndex:
    def test_build_with_enough_items(self):
        embs = make_embeddings(500, 32)
        index = build_ivf_index(embs.copy(), n_centroids=32, n_probe=8)
        assert index.ntotal == 500

    def test_fallback_to_flat_when_small(self):
        embs = make_embeddings(10, 32)
        index = build_ivf_index(embs.copy(), n_centroids=64)
        # Should fall back to flat — either type is valid
        assert index.ntotal == 10


class TestSaveLoad:
    def test_round_trip(self):
        embs = make_embeddings(30, 16)
        index = build_flat_index(embs.copy())
        internal_to_raw = {i: i * 10 for i in range(30)}

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            map_path = Path(tmpdir) / "map.json"
            save_index(index, internal_to_raw, idx_path, map_path)

            loaded_index, loaded_map = load_index(idx_path, map_path)
            assert loaded_index.ntotal == 30
            assert loaded_map[5] == 50


class TestFaissRetriever:
    def test_retrieve_returns_k_results(self):
        embs = make_embeddings(100, 32)
        index = build_flat_index(embs.copy())
        internal_to_raw = {i: i + 1000 for i in range(100)}
        retriever = FaissRetriever(index, internal_to_raw)

        results = retriever.retrieve(embs[0], k=10)
        assert len(results) == 10

    def test_retrieve_returns_raw_ids(self):
        embs = make_embeddings(50, 32)
        index = build_flat_index(embs.copy())
        internal_to_raw = {i: i + 500 for i in range(50)}
        retriever = FaissRetriever(index, internal_to_raw)

        results = retriever.retrieve(embs[0], k=5)
        returned_ids = [iid for iid, _ in results]
        for rid in returned_ids:
            assert rid >= 500

    def test_scores_are_sorted_descending(self):
        embs = make_embeddings(50, 32)
        index = build_flat_index(embs.copy())
        internal_to_raw = {i: i for i in range(50)}
        retriever = FaissRetriever(index, internal_to_raw)

        results = retriever.retrieve(embs[0], k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
