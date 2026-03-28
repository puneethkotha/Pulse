"""Tests for Kafka producer/consumer and online feature updates."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming.schema import InteractionEvent, UserOnlineFeatures
from src.streaming.consumer import update_user_features_in_place
from src.data.schema import ItemFeatures, NUM_GENRES


def make_event(user_id=1, item_id=10, event_type="rate", rating=4.0):
    return InteractionEvent(
        event_id="test-uuid",
        event_type=event_type,
        user_id=user_id,
        item_id=item_id,
        rating=rating if event_type == "rate" else None,
        timestamp=time.time(),
    )


def make_item_features(item_id, genres=("Action", "Drama")):
    return ItemFeatures(
        item_id=item_id,
        title=f"Movie {item_id} (2000)",
        year=2000,
        genres=list(genres),
        genre_vector=[0] * NUM_GENRES,
        avg_rating=3.5,
        num_ratings=100,
    )


class TestInteractionEvent:
    def test_serialization_round_trip(self):
        event = make_event()
        json_str = event.to_json()
        recovered = InteractionEvent.from_json(json_str)
        assert recovered.user_id == event.user_id
        assert recovered.item_id == event.item_id
        assert recovered.rating == event.rating
        assert recovered.event_type == event.event_type

    def test_validation_passes_for_rate(self):
        event = make_event(event_type="rate", rating=4.0)
        event.validate()  # should not raise

    def test_validation_fails_for_invalid_type(self):
        event = make_event()
        event.event_type = "invalid"
        with pytest.raises(ValueError, match="event_type"):
            event.validate()

    def test_validation_fails_missing_rating_for_rate(self):
        event = make_event(event_type="rate", rating=None)
        with pytest.raises(ValueError):
            event.validate()

    def test_validation_fails_out_of_range_rating(self):
        event = make_event(event_type="rate", rating=6.0)
        with pytest.raises(ValueError):
            event.validate()

    def test_view_event_no_rating(self):
        event = make_event(event_type="view", rating=None)
        event.validate()  # should not raise


class TestUserOnlineFeatures:
    def test_empty_initialization(self):
        feat = UserOnlineFeatures.empty(user_id=42)
        assert feat.user_id == 42
        assert feat.num_events == 0
        assert feat.avg_rating == 0.0
        assert feat.last_item_ids == []

    def test_serialization_round_trip(self):
        feat = UserOnlineFeatures(
            user_id=1, num_events=5, num_ratings=3,
            avg_rating=4.2, last_item_ids=[10, 20, 30],
            genre_counts={"Action": 2, "Drama": 1},
            last_updated=time.time(),
        )
        json_str = feat.to_json()
        recovered = UserOnlineFeatures.from_json(json_str)
        assert recovered.user_id == feat.user_id
        assert recovered.num_events == feat.num_events
        assert recovered.genre_counts == feat.genre_counts


class TestUpdateUserFeatures:
    def test_creates_new_features_for_unknown_user(self):
        event = make_event(user_id=99, item_id=10, event_type="view")
        item_feats = {10: make_item_features(10)}
        result = update_user_features_in_place(None, event, item_feats)
        assert result.user_id == 99
        assert result.num_events == 1

    def test_increments_event_count(self):
        existing = UserOnlineFeatures.empty(user_id=1)
        event = make_event(user_id=1, item_id=10, event_type="view")
        item_feats = {10: make_item_features(10)}
        result = update_user_features_in_place(existing, event, item_feats)
        assert result.num_events == 1

    def test_updates_rating_average(self):
        existing = UserOnlineFeatures.empty(user_id=1)
        event = make_event(user_id=1, item_id=10, event_type="rate", rating=4.0)
        item_feats = {10: make_item_features(10)}
        result = update_user_features_in_place(existing, event, item_feats)
        assert result.num_ratings == 1
        assert result.avg_rating == pytest.approx(4.0)

    def test_running_average_is_correct(self):
        existing = UserOnlineFeatures(
            user_id=1, num_events=1, num_ratings=1,
            avg_rating=4.0, last_item_ids=[],
            genre_counts={}, last_updated=time.time(),
        )
        event = make_event(user_id=1, item_id=10, event_type="rate", rating=2.0)
        item_feats = {10: make_item_features(10)}
        result = update_user_features_in_place(existing, event, item_feats)
        assert result.avg_rating == pytest.approx(3.0)

    def test_updates_genre_counts(self):
        existing = UserOnlineFeatures.empty(user_id=1)
        event = make_event(user_id=1, item_id=10, event_type="view")
        item_feats = {10: make_item_features(10, genres=("Action", "Drama"))}
        result = update_user_features_in_place(existing, event, item_feats)
        assert result.genre_counts.get("Action", 0) == 1
        assert result.genre_counts.get("Drama", 0) == 1

    def test_last_items_ring_buffer(self):
        from src.streaming.consumer import MAX_LAST_ITEMS
        existing = UserOnlineFeatures.empty(user_id=1)
        existing.last_item_ids = list(range(MAX_LAST_ITEMS))
        event = make_event(user_id=1, item_id=999, event_type="view")
        item_feats = {999: make_item_features(999)}
        result = update_user_features_in_place(existing, event, item_feats)
        assert len(result.last_item_ids) == MAX_LAST_ITEMS
        assert 999 in result.last_item_ids
        assert 0 not in result.last_item_ids  # oldest dropped
