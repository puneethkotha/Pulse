"""
Kafka consumer that updates rolling user features in Redis.

Reads InteractionEvent messages from the configured topic and
incrementally updates UserOnlineFeatures for each user.
"""

import logging
import time
from typing import Optional

from src.streaming.schema import InteractionEvent, UserOnlineFeatures
from src.data.schema import ItemFeatures
from src.features.online_store import OnlineFeatureStore

logger = logging.getLogger(__name__)

MAX_LAST_ITEMS = 20   # ring buffer length
DEFAULT_TOPIC = "user-interactions"


def update_user_features_in_place(
    existing: Optional[UserOnlineFeatures],
    event: InteractionEvent,
    item_features: dict[int, ItemFeatures],
) -> UserOnlineFeatures:
    """
    Apply a single event to produce updated UserOnlineFeatures.

    Creates a new UserOnlineFeatures if none exists for the user.
    """
    if existing is None:
        features = UserOnlineFeatures.empty(event.user_id)
    else:
        features = existing

    features.num_events += 1
    features.last_updated = time.time()

    # update last items ring buffer
    features.last_item_ids = (features.last_item_ids + [event.item_id])[-MAX_LAST_ITEMS:]

    # update genre counts from the interacted item
    item_feat = item_features.get(event.item_id)
    if item_feat:
        for genre in item_feat.genres:
            features.genre_counts[genre] = features.genre_counts.get(genre, 0) + 1

    # update rating stats
    if event.event_type == "rate" and event.rating is not None:
        old_total = features.avg_rating * features.num_ratings
        features.num_ratings += 1
        features.avg_rating = (old_total + event.rating) / features.num_ratings

    return features


def run_consumer(
    bootstrap_servers: str = "localhost:9092",
    topic: str = DEFAULT_TOPIC,
    group_id: str = "feature-updater",
    feature_store: Optional[OnlineFeatureStore] = None,
    item_features: Optional[dict[int, ItemFeatures]] = None,
    max_messages: Optional[int] = None,
) -> int:
    """
    Consume events from Kafka and update Redis feature store.

    Parameters
    ----------
    max_messages : int, optional
        Stop after processing this many messages (useful for testing).

    Returns the number of messages processed.
    """
    from kafka import KafkaConsumer

    if feature_store is None:
        feature_store = OnlineFeatureStore()
    if item_features is None:
        item_features = {}

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v.decode("utf-8"),
    )

    logger.info("Kafka consumer started. Listening on topic '%s'...", topic)
    count = 0

    for message in consumer:
        try:
            event = InteractionEvent.from_json(message.value)
            event.validate()

            existing = feature_store.get_user_features(event.user_id)
            updated = update_user_features_in_place(existing, event, item_features or {})
            feature_store.set_user_features(updated)

            count += 1
            if count % 100 == 0:
                logger.info("Processed %d messages.", count)

            if max_messages is not None and count >= max_messages:
                break

        except Exception as exc:
            logger.warning("Failed to process message: %s", exc)

    consumer.close()
    logger.info("Consumer finished. Total messages processed: %d", count)
    return count
