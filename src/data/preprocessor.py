"""
Data preprocessing, feature engineering, and train/val/test splitting.

Split strategy: user-aware temporal split.
For each user, interactions are sorted by timestamp.
- Train: earliest 80% of interactions
- Val:   next 10%
- Test:  final 10%

Users with fewer than min_interactions are dropped entirely.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import (
    Interaction, UserFeatures, ItemFeatures, Split,
    AGE_BUCKETS, NUM_AGE_BUCKETS, NUM_OCCUPATIONS, NUM_GENRES
)


PROCESSED_DIR = Path("data/processed")


def filter_cold_users(
    ratings_df: pd.DataFrame,
    min_interactions: int = 5,
) -> pd.DataFrame:
    """Remove users with fewer than min_interactions ratings."""
    counts = ratings_df.groupby("user_id")["item_id"].count()
    valid_users = counts[counts >= min_interactions].index
    return ratings_df[ratings_df["user_id"].isin(valid_users)].copy()


def temporal_user_split(
    ratings_df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each user's interactions temporally into train / val / test.

    Returns three DataFrames with the same columns as ratings_df.
    """
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []

    for _, group in ratings_df.groupby("user_id"):
        n = len(group)
        i_val = max(1, int(n * train_frac))
        i_test = max(i_val + 1, int(n * (train_frac + val_frac)))

        train_rows.append(group.iloc[:i_val])
        val_rows.append(group.iloc[i_val:i_test])
        test_rows.append(group.iloc[i_test:])

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.concat(test_rows, ignore_index=True),
    )


def df_to_interactions(df: pd.DataFrame) -> list[Interaction]:
    """Convert a ratings DataFrame to a list of Interaction objects."""
    return [
        Interaction(
            user_id=int(row.user_id),
            item_id=int(row.item_id),
            rating=float(row.rating),
            timestamp=int(row.timestamp),
        )
        for row in df.itertuples(index=False)
    ]


def build_user_id_map(user_features: dict[int, UserFeatures]) -> dict[int, int]:
    """Map raw user_ids to contiguous 0-indexed integers."""
    return {uid: idx for idx, uid in enumerate(sorted(user_features.keys()))}


def build_item_id_map(item_features: dict[int, ItemFeatures]) -> dict[int, int]:
    """Map raw item_ids to contiguous 0-indexed integers."""
    return {iid: idx for idx, iid in enumerate(sorted(item_features.keys()))}


def normalize_year(
    item_features: dict[int, ItemFeatures],
    min_year: int = 1919,
    max_year: int = 2000,
) -> dict[int, float]:
    """Return item_id -> normalized year in [0, 1]."""
    result = {}
    span = max(max_year - min_year, 1)
    for iid, feat in item_features.items():
        y = feat.year if feat.year > 0 else min_year
        result[iid] = (y - min_year) / span
    return result


def normalize_popularity(item_features: dict[int, ItemFeatures]) -> dict[int, float]:
    """Return item_id -> log-normalized popularity score in [0, 1]."""
    counts = {iid: feat.num_ratings for iid, feat in item_features.items()}
    max_log = np.log1p(max(counts.values(), default=1))
    return {
        iid: float(np.log1p(c) / max_log)
        for iid, c in counts.items()
    }


def build_item_feature_matrix(
    item_features: dict[int, ItemFeatures],
    item_id_map: dict[int, int],
) -> np.ndarray:
    """
    Build a dense feature matrix for all items.

    Shape: (num_items, feature_dim)
    Features: [genre_vector (18), normalized_year (1), log_popularity (1), avg_rating/5 (1)]
    Total: 21 dims
    """
    n = len(item_id_map)
    feat_dim = NUM_GENRES + 3  # 18 genre + year + pop + avg_rating
    matrix = np.zeros((n, feat_dim), dtype=np.float32)

    norm_year = normalize_year(item_features)
    norm_pop = normalize_popularity(item_features)

    for iid, idx in item_id_map.items():
        feat = item_features[iid]
        # genre multi-hot
        matrix[idx, :NUM_GENRES] = feat.genre_vector
        # year normalized
        matrix[idx, NUM_GENRES] = norm_year.get(iid, 0.0)
        # popularity
        matrix[idx, NUM_GENRES + 1] = norm_pop.get(iid, 0.0)
        # avg rating normalized to [0,1]
        matrix[idx, NUM_GENRES + 2] = feat.avg_rating / 5.0

    return matrix


def build_user_feature_matrix(
    user_features: dict[int, UserFeatures],
    user_id_map: dict[int, int],
) -> np.ndarray:
    """
    Build a dense feature matrix for all users.

    Shape: (num_users, feature_dim)
    Features: [gender (1), age_bucket OHE (7), occupation OHE (21), zip_prefix (1)]
    Total: 30 dims
    """
    n = len(user_id_map)
    feat_dim = 1 + NUM_AGE_BUCKETS + NUM_OCCUPATIONS + 1
    matrix = np.zeros((n, feat_dim), dtype=np.float32)

    for uid, idx in user_id_map.items():
        feat = user_features[uid]
        matrix[idx, 0] = float(feat.gender)
        if 0 <= feat.age_bucket < NUM_AGE_BUCKETS:
            matrix[idx, 1 + feat.age_bucket] = 1.0
        occ = max(0, min(feat.occupation, NUM_OCCUPATIONS - 1))
        matrix[idx, 1 + NUM_AGE_BUCKETS + occ] = 1.0
        matrix[idx, 1 + NUM_AGE_BUCKETS + NUM_OCCUPATIONS] = feat.zip_prefix / 9.0

    return matrix


def preprocess_and_save(
    ratings_df: pd.DataFrame,
    user_features: dict[int, UserFeatures],
    item_features: dict[int, ItemFeatures],
    min_interactions: int = 5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    output_dir: Path = PROCESSED_DIR,
) -> dict:
    """
    Full preprocessing pipeline.

    Saves all artifacts to output_dir and returns a dict of paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter cold users
    filtered = filter_cold_users(ratings_df, min_interactions)

    # Only keep users and items present after filtering
    valid_users = set(filtered["user_id"].unique())
    valid_items = set(filtered["item_id"].unique())
    user_features = {uid: v for uid, v in user_features.items() if uid in valid_users}
    item_features = {iid: v for iid, v in item_features.items() if iid in valid_items}

    # Temporal split
    train_df, val_df, test_df = temporal_user_split(filtered, train_frac, val_frac)

    # ID maps
    user_id_map = build_user_id_map(user_features)
    item_id_map = build_item_id_map(item_features)

    # Feature matrices
    item_feat_matrix = build_item_feature_matrix(item_features, item_id_map)
    user_feat_matrix = build_user_feature_matrix(user_features, user_id_map)

    # Persist splits as DataFrames
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    # Persist ID maps and features
    with open(output_dir / "user_id_map.pkl", "wb") as f:
        pickle.dump(user_id_map, f)
    with open(output_dir / "item_id_map.pkl", "wb") as f:
        pickle.dump(item_id_map, f)
    with open(output_dir / "user_features.pkl", "wb") as f:
        pickle.dump(user_features, f)
    with open(output_dir / "item_features.pkl", "wb") as f:
        pickle.dump(item_features, f)

    np.save(output_dir / "item_feat_matrix.npy", item_feat_matrix)
    np.save(output_dir / "user_feat_matrix.npy", user_feat_matrix)

    stats = {
        "num_users": len(user_id_map),
        "num_items": len(item_id_map),
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
        "total_interactions": len(filtered),
        "item_feature_dim": item_feat_matrix.shape[1],
        "user_feature_dim": user_feat_matrix.shape[1],
    }

    import json
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def load_processed(output_dir: Path = PROCESSED_DIR) -> dict:
    """Load all preprocessed artifacts from disk."""
    import json

    train_df = pd.read_parquet(output_dir / "train.parquet")
    val_df = pd.read_parquet(output_dir / "val.parquet")
    test_df = pd.read_parquet(output_dir / "test.parquet")

    with open(output_dir / "user_id_map.pkl", "rb") as f:
        user_id_map = pickle.load(f)
    with open(output_dir / "item_id_map.pkl", "rb") as f:
        item_id_map = pickle.load(f)
    with open(output_dir / "user_features.pkl", "rb") as f:
        user_features = pickle.load(f)
    with open(output_dir / "item_features.pkl", "rb") as f:
        item_features = pickle.load(f)

    item_feat_matrix = np.load(output_dir / "item_feat_matrix.npy")
    user_feat_matrix = np.load(output_dir / "user_feat_matrix.npy")

    with open(output_dir / "stats.json") as f:
        stats = json.load(f)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "user_features": user_features,
        "item_features": item_features,
        "item_feat_matrix": item_feat_matrix,
        "user_feat_matrix": user_feat_matrix,
        "stats": stats,
    }
