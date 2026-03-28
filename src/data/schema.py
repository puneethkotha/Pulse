"""
Data schemas and type definitions for the ranking pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class UserFeatures:
    user_id: int
    gender: int              # 0=M, 1=F
    age_bucket: int          # encoded age group
    occupation: int          # 0-20
    zip_prefix: int          # first digit of zip code


@dataclass
class ItemFeatures:
    item_id: int
    title: str
    year: int                # release year extracted from title
    genres: list[str]        # list of genre strings
    genre_vector: list[int]  # multi-hot genre encoding
    avg_rating: float
    num_ratings: int


@dataclass
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass
class Split:
    """Train/val/test split result."""
    train: list[Interaction]
    val: list[Interaction]
    test: list[Interaction]
    user_features: dict[int, UserFeatures]
    item_features: dict[int, ItemFeatures]


@dataclass
class RecommendationResult:
    item_id: int
    score: float
    title: str
    genres: list[str]
    avg_rating: float
    retrieval_score: float
    rerank_score: Optional[float] = None


GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_LIST)}
NUM_GENRES = len(GENRE_LIST)

AGE_BUCKETS = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
NUM_AGE_BUCKETS = len(AGE_BUCKETS)
NUM_OCCUPATIONS = 21
