"""
Re-ranker feature builder.

Constructs the 44-dimensional feature vector for each (user, candidate_item) pair.

Feature layout (must match RERANKER_INPUT_DIM in reranker.py):
  [0]      embedding_similarity
  [1]      genre_overlap
  [2]      item_popularity (log-normalized)
  [3]      item_avg_rating (normalized)
  [4]      item_year_norm
  [5:23]   item_genre_vector (18 dims)
  [23]     user_avg_rating (normalized)
  [24]     user_num_ratings (log-normalized)
  [25:43]  user_genre_pref_vector (18 dims, normalized)
  [43]     retrieval_rank_norm  (rank in candidate list / candidate_pool_size)
Total: 44
"""

import math
from typing import Optional

import numpy as np

from src.data.schema import ItemFeatures, NUM_GENRES, GENRE_LIST
from src.streaming.schema import UserOnlineFeatures

FEATURE_DIM = 44


def _log_norm(x: float, max_val: float) -> float:
    return math.log1p(x) / math.log1p(max_val) if max_val > 0 else 0.0


def build_candidate_features(
    user_item_pairs: list[tuple[int, int, float, int]],
    # (user_id, item_id, embedding_similarity, rank_in_candidate_list)
    item_features: dict[int, ItemFeatures],
    user_train_interactions: dict[int, list],   # user_id -> list[Interaction]
    online_features: Optional[dict[int, UserOnlineFeatures]] = None,
    max_item_ratings: int = 3428,               # set from dataset stats
    max_user_ratings: int = 2314,               # set from dataset stats
    candidate_pool_size: int = 100,
) -> np.ndarray:
    """
    Build feature matrix for a batch of (user, item) candidate pairs.

    Parameters
    ----------
    user_item_pairs : list of (user_id, item_id, embedding_sim, candidate_rank)
    item_features   : item metadata
    user_train_interactions : training interactions per user (for genre preferences)
    online_features : optional real-time features from Redis
    max_item_ratings : max item rating count in the dataset (for normalization)
    max_user_ratings : max user rating count in the dataset (for normalization)
    candidate_pool_size : denominator for rank normalization

    Returns
    -------
    features : np.ndarray of shape (len(user_item_pairs), FEATURE_DIM)
    """
    features = np.zeros((len(user_item_pairs), FEATURE_DIM), dtype=np.float32)

    for i, (user_id, item_id, sim, rank) in enumerate(user_item_pairs):
        item_feat = item_features.get(item_id)
        if item_feat is None:
            continue

        # --- item features ---
        features[i, 0] = float(sim)
        features[i, 2] = _log_norm(item_feat.num_ratings, max_item_ratings)
        features[i, 3] = item_feat.avg_rating / 5.0
        # year normalized (1919-2000 range based on MovieLens 1M)
        year = item_feat.year if item_feat.year > 0 else 1960
        features[i, 4] = (year - 1919) / max(2000 - 1919, 1)
        # genre vector
        features[i, 5:5 + NUM_GENRES] = item_feat.genre_vector
        # rank in candidate list
        features[i, 43] = rank / max(candidate_pool_size, 1)

        # --- user features (from online store if available, else offline) ---
        online = online_features.get(user_id) if online_features else None

        if online is not None:
            avg_rating = online.avg_rating
            num_ratings = online.num_ratings
            genre_counts = online.genre_counts
        else:
            # derive from training interactions
            interactions = user_train_interactions.get(user_id, [])
            ratings = [inter.rating for inter in interactions]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
            num_ratings = len(ratings)
            genre_counts: dict[str, int] = {}
            for inter in interactions:
                ifeats = item_features.get(inter.item_id)
                if ifeats:
                    for g in ifeats.genres:
                        genre_counts[g] = genre_counts.get(g, 0) + 1

        features[i, 23] = avg_rating / 5.0
        features[i, 24] = _log_norm(num_ratings, max_user_ratings)

        # user genre preference (normalized)
        total_genre_count = max(sum(genre_counts.values()), 1)
        genre_pref = np.array(
            [genre_counts.get(g, 0) / total_genre_count for g in GENRE_LIST],
            dtype=np.float32,
        )
        features[i, 25:25 + NUM_GENRES] = genre_pref

        # genre overlap
        item_genres = set(item_feat.genres)
        user_genres = set(g for g, c in genre_counts.items() if c > 0)
        if user_genres:
            overlap = len(item_genres & user_genres) / len(user_genres)
        else:
            overlap = 0.0
        features[i, 1] = overlap

    return features
