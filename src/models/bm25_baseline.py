"""
BM25 retrieval baseline using item title and genre text.

Each item is represented as a text document: "<title> <genre1> <genre2> ...".
A query is built from the genres of items the user has highly rated.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.schema import ItemFeatures

logger = logging.getLogger(__name__)

BM25_CACHE = Path("models/bm25_index.pkl")


def build_corpus(item_features: dict[int, ItemFeatures]) -> tuple[list[int], list[list[str]]]:
    """
    Build item_ids list and tokenized corpus.

    Returns
    -------
    item_ids : ordered list of raw item IDs
    corpus   : corresponding tokenized documents
    """
    item_ids = sorted(item_features.keys())
    corpus = []
    for iid in item_ids:
        feat = item_features[iid]
        # simple whitespace tokenization on title + genres
        tokens = feat.title.lower().split() + [g.lower() for g in feat.genres]
        corpus.append(tokens)
    return item_ids, corpus


def build_bm25_index(
    item_features: dict[int, ItemFeatures],
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple["BM25Okapi", list[int]]:
    """Build and return a BM25 index over all items."""
    item_ids, corpus = build_corpus(item_features)
    bm25 = BM25Okapi(corpus, k1=k1, b=b)
    return bm25, item_ids


def save_bm25(bm25: "BM25Okapi", item_ids: list[int], path: Path = BM25_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": item_ids}, f)


def load_bm25(path: Path = BM25_CACHE) -> tuple["BM25Okapi", list[int]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["item_ids"]


def build_user_query(
    user_id: int,
    train_interactions: list,           # list of Interaction
    item_features: dict[int, ItemFeatures],
    min_rating: float = 4.0,
    max_tokens: int = 50,
) -> list[str]:
    """
    Build a BM25 query for a user from their high-rated item genres/titles.
    """
    tokens: list[str] = []
    for interaction in train_interactions:
        if interaction.user_id != user_id:
            continue
        if interaction.rating < min_rating:
            continue
        feat = item_features.get(interaction.item_id)
        if feat is None:
            continue
        tokens.extend([g.lower() for g in feat.genres])
        if len(tokens) >= max_tokens:
            break
    return tokens if tokens else ["movie"]  # fallback


class BM25Retriever:
    """
    Retriever backed by a BM25 index.
    """

    def __init__(
        self,
        bm25: "BM25Okapi",
        item_ids: list[int],
    ) -> None:
        self._bm25 = bm25
        self._item_ids = item_ids

    def retrieve(
        self,
        query_tokens: list[str],
        k: int = 100,
        exclude_item_ids: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        """
        Return top-K (item_id, bm25_score) pairs.

        exclude_item_ids : items to skip (e.g., already interacted with)
        """
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[tuple[int, float]] = []
        for idx in ranked_indices:
            if len(results) >= k:
                break
            iid = self._item_ids[idx]
            if exclude_item_ids and iid in exclude_item_ids:
                continue
            results.append((iid, float(scores[idx])))
        return results

    @classmethod
    def from_disk(cls, path: Path = BM25_CACHE) -> "BM25Retriever":
        bm25, item_ids = load_bm25(path)
        return cls(bm25, item_ids)
