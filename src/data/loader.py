"""
MovieLens 1M dataset loader.

Downloads and parses the raw dataset files. All files are expected at
data/raw/ml-1m/ after running scripts/download_data.sh.
"""

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.schema import (
    Interaction, UserFeatures, ItemFeatures,
    GENRE_LIST, GENRE_TO_IDX, AGE_BUCKETS
)

RAW_DATA_DIR = Path("data/raw/ml-1m")

RATINGS_FILE = RAW_DATA_DIR / "ratings.dat"
USERS_FILE = RAW_DATA_DIR / "users.dat"
MOVIES_FILE = RAW_DATA_DIR / "movies.dat"


def _parse_year(title: str) -> int:
    """Extract 4-digit year from a title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)$", title.strip())
    if match:
        return int(match.group(1))
    return 0


def load_ratings(path: Path = RATINGS_FILE) -> pd.DataFrame:
    """
    Load ratings.dat into a DataFrame.

    Columns: user_id, item_id, rating, timestamp
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    return df


def load_users(path: Path = USERS_FILE) -> pd.DataFrame:
    """
    Load users.dat into a DataFrame.

    Columns: user_id, gender, age, occupation, zip_code
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"user_id": int, "age": int, "occupation": int},
    )
    return df


def load_movies(path: Path = MOVIES_FILE) -> pd.DataFrame:
    """
    Load movies.dat into a DataFrame.

    Columns: item_id, title, genres (as list)
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genres_str"],
        encoding="latin-1",
        dtype={"item_id": int},
    )
    df["genres"] = df["genres_str"].apply(lambda g: g.split("|"))
    df["year"] = df["title"].apply(_parse_year)
    df.drop(columns=["genres_str"], inplace=True)
    return df


def build_user_features(users_df: pd.DataFrame) -> dict[int, UserFeatures]:
    """Convert users DataFrame to a user_id -> UserFeatures mapping."""
    features: dict[int, UserFeatures] = {}
    for row in users_df.itertuples(index=False):
        gender_enc = 0 if row.gender == "M" else 1
        age_enc = AGE_BUCKETS.get(row.age, 0)
        zip_prefix = int(str(row.zip_code).replace("-", "")[0]) if row.zip_code else 0
        features[row.user_id] = UserFeatures(
            user_id=row.user_id,
            gender=gender_enc,
            age_bucket=age_enc,
            occupation=row.occupation,
            zip_prefix=zip_prefix,
        )
    return features


def build_item_features(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> dict[int, ItemFeatures]:
    """
    Build item_id -> ItemFeatures mapping.

    Computes avg_rating and num_ratings from the full ratings set.
    """
    stats = (
        ratings_df.groupby("item_id")["rating"]
        .agg(avg_rating="mean", num_ratings="count")
        .reset_index()
    )
    movies = movies_df.merge(stats, on="item_id", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0.0)
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)

    features: dict[int, ItemFeatures] = {}
    for row in movies.itertuples(index=False):
        genre_vector = [0] * len(GENRE_LIST)
        for g in row.genres:
            if g in GENRE_TO_IDX:
                genre_vector[GENRE_TO_IDX[g]] = 1
        features[row.item_id] = ItemFeatures(
            item_id=row.item_id,
            title=row.title,
            year=row.year,
            genres=list(row.genres),
            genre_vector=genre_vector,
            avg_rating=round(float(row.avg_rating), 4),
            num_ratings=int(row.num_ratings),
        )
    return features


def load_all() -> tuple[pd.DataFrame, dict[int, UserFeatures], dict[int, ItemFeatures]]:
    """
    Load and return all raw data.

    Returns
    -------
    ratings_df : DataFrame with columns [user_id, item_id, rating, timestamp]
    user_features : dict mapping user_id -> UserFeatures
    item_features : dict mapping item_id -> ItemFeatures
    """
    ratings_df = load_ratings()
    users_df = load_users()
    movies_df = load_movies()

    user_features = build_user_features(users_df)
    item_features = build_item_features(movies_df, ratings_df)

    return ratings_df, user_features, item_features
