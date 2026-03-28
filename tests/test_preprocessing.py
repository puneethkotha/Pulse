"""Tests for data preprocessing pipeline."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.schema import UserFeatures, ItemFeatures, GENRE_LIST, NUM_GENRES
from src.data.preprocessor import (
    filter_cold_users,
    temporal_user_split,
    build_user_id_map,
    build_item_id_map,
    build_item_feature_matrix,
    build_user_feature_matrix,
)


def make_ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def make_user_features(user_ids):
    return {
        uid: UserFeatures(uid, gender=0, age_bucket=2, occupation=1, zip_prefix=9)
        for uid in user_ids
    }


def make_item_features(item_ids):
    return {
        iid: ItemFeatures(
            item_id=iid, title=f"Movie {iid} (2000)",
            year=2000, genres=["Action", "Drama"],
            genre_vector=[0] * NUM_GENRES,
            avg_rating=3.5, num_ratings=100,
        )
        for iid in item_ids
    }


class TestFilterColdUsers:
    def test_removes_low_interaction_users(self):
        df = make_ratings([
            (1, 1, 4.0, 1000), (1, 2, 3.0, 1001), (1, 3, 5.0, 1002),
            (1, 4, 4.0, 1003), (1, 5, 3.0, 1004),  # 5 interactions
            (2, 1, 3.0, 2000),  # only 1 interaction
        ])
        result = filter_cold_users(df, min_interactions=5)
        assert set(result["user_id"].unique()) == {1}

    def test_keeps_users_meeting_threshold(self):
        df = make_ratings([
            (1, i, 4.0, i) for i in range(5)
        ] + [
            (2, i, 3.0, i) for i in range(5)
        ])
        result = filter_cold_users(df, min_interactions=5)
        assert set(result["user_id"].unique()) == {1, 2}


class TestTemporalSplit:
    def make_user_df(self, user_id, n_items=20):
        return make_ratings([
            (user_id, i, 4.0, i * 100)
            for i in range(n_items)
        ])

    def test_split_sizes(self):
        df = self.make_user_df(1, n_items=20)
        train, val, test = temporal_user_split(df, train_frac=0.8, val_frac=0.1)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == 20

    def test_temporal_ordering_respected(self):
        df = self.make_user_df(1, n_items=20)
        train, val, test = temporal_user_split(df, train_frac=0.8, val_frac=0.1)
        max_train_ts = train["timestamp"].max()
        min_val_ts = val["timestamp"].min()
        assert max_train_ts <= min_val_ts

    def test_no_data_leakage_between_splits(self):
        df = self.make_user_df(1, n_items=20)
        train, val, test = temporal_user_split(df)
        train_ids = set(zip(train["user_id"], train["item_id"]))
        val_ids = set(zip(val["user_id"], val["item_id"]))
        test_ids = set(zip(test["user_id"], test["item_id"]))
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestFeatureMatrices:
    def test_item_feature_matrix_shape(self):
        item_ids = list(range(1, 11))
        item_features = make_item_features(item_ids)
        item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        matrix = build_item_feature_matrix(item_features, item_id_map)
        assert matrix.shape == (10, NUM_GENRES + 3)

    def test_user_feature_matrix_shape(self):
        user_ids = list(range(1, 6))
        user_features = make_user_features(user_ids)
        user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        matrix = build_user_feature_matrix(user_features, user_id_map)
        # 1 (gender) + 7 (age) + 21 (occ) + 1 (zip) = 30
        assert matrix.shape == (5, 30)

    def test_item_feature_values_in_range(self):
        item_ids = [1, 2, 3]
        item_features = make_item_features(item_ids)
        item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        matrix = build_item_feature_matrix(item_features, item_id_map)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
