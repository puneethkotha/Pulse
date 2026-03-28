"""Tests for ranking evaluation metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    dcg_at_k,
    compute_all_metrics,
    aggregate_metrics,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial(self):
        assert precision_at_k([1, 4, 2, 5], {1, 2, 3}, k=4) == pytest.approx(0.5)

    def test_k_larger_than_list(self):
        assert precision_at_k([1, 2], {1, 2, 3}, k=5) == pytest.approx(2 / 5)

    def test_k_zero(self):
        assert precision_at_k([1, 2], {1, 2}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero(self):
        assert recall_at_k([4, 5], {1, 2}, k=2) == 0.0

    def test_partial(self):
        assert recall_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_empty_relevant(self):
        assert recall_at_k([1, 2], set(), k=2) == 0.0


class TestNDCGAtK:
    def test_perfect(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_zero(self):
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_order_matters(self):
        ndcg_top = ndcg_at_k([1, 4, 2], {1, 2}, k=3)
        ndcg_bottom = ndcg_at_k([4, 1, 2], {1, 2}, k=3)
        assert ndcg_top > ndcg_bottom

    def test_single_relevant(self):
        # Relevant item at rank 1 should give 1.0
        assert ndcg_at_k([1], {1}, k=1) == pytest.approx(1.0)

    def test_empty_relevant(self):
        assert ndcg_at_k([1, 2], set(), k=2) == 0.0


class TestComputeAllMetrics:
    def test_keys_present(self):
        result = compute_all_metrics([1, 2, 3, 4, 5], {1, 3, 5}, ks=(5, 10))
        expected_keys = {"precision@5", "recall@5", "ndcg@5", "precision@10", "recall@10", "ndcg@10"}
        assert expected_keys == set(result.keys())

    def test_values_in_range(self):
        result = compute_all_metrics([1, 2, 3], {1, 2}, ks=(3,))
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestAggregateMetrics:
    def test_average(self):
        per_user = [
            {"precision@5": 0.4, "ndcg@5": 0.6},
            {"precision@5": 0.6, "ndcg@5": 0.8},
        ]
        result = aggregate_metrics(per_user)
        assert result["precision@5"] == pytest.approx(0.5)
        assert result["ndcg@5"] == pytest.approx(0.7)

    def test_empty(self):
        assert aggregate_metrics([]) == {}
