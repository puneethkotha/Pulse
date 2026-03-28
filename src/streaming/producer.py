"""
Kafka producer that simulates user interaction events from MovieLens data.

This is a simulation layer — it replays historical MovieLens interactions
as if they were live events arriving in real time. It is clearly separated
from the offline training dataset and does not modify any training data.
"""

import json
import logging
import random
import time
import uuid
from pathlib import Path

import pandas as pd

from src.streaming.schema import InteractionEvent

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "user-interactions"


def get_kafka_producer(bootstrap_servers: str = "localhost:9092"):
    """Create and return a KafkaProducer. Raises if Kafka is unavailable."""
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v.encode("utf-8"),
    )


def simulate_events_from_ratings(
    ratings_df: pd.DataFrame,
    bootstrap_servers: str = "localhost:9092",
    topic: str = DEFAULT_TOPIC,
    rate_per_second: float = 10.0,
    max_events: int = 10000,
    seed: int = 42,
) -> int:
    """
    Replay MovieLens ratings as simulated interaction events.

    Randomly samples from the ratings DataFrame and publishes
    InteractionEvent messages to Kafka.

    Returns the number of events published.
    """
    random.seed(seed)
    producer = get_kafka_producer(bootstrap_servers)

    sampled = ratings_df.sample(n=min(max_events, len(ratings_df)), random_state=seed)
    delay = 1.0 / rate_per_second
    count = 0

    for row in sampled.itertuples(index=False):
        event = InteractionEvent(
            event_id=str(uuid.uuid4()),
            event_type="rate",
            user_id=int(row.user_id),
            item_id=int(row.item_id),
            rating=float(row.rating),
            timestamp=float(row.timestamp),
            session_id=f"sim-{row.user_id}",
        )
        producer.send(topic, value=event.to_json())
        count += 1

        if count % 100 == 0:
            logger.info("Published %d events...", count)

        time.sleep(delay)

    producer.flush()
    logger.info("Done. Published %d events to topic '%s'.", count, topic)
    return count


def simulate_random_events(
    user_ids: list[int],
    item_ids: list[int],
    bootstrap_servers: str = "localhost:9092",
    topic: str = DEFAULT_TOPIC,
    rate_per_second: float = 5.0,
    num_events: int = 500,
    seed: int = 0,
) -> int:
    """
    Publish randomly generated events (useful for stress testing).

    Uses real user/item IDs from the dataset but randomizes pairings.
    Clearly synthetic — not derived from real interaction sequences.
    """
    random.seed(seed)
    producer = get_kafka_producer(bootstrap_servers)
    delay = 1.0 / rate_per_second
    event_types = ["view", "click", "rate"]
    count = 0

    for _ in range(num_events):
        etype = random.choice(event_types)
        rating = round(random.uniform(1.0, 5.0), 1) if etype == "rate" else None
        event = InteractionEvent(
            event_id=str(uuid.uuid4()),
            event_type=etype,
            user_id=random.choice(user_ids),
            item_id=random.choice(item_ids),
            rating=rating,
            timestamp=time.time(),
            session_id=f"rand-{count}",
        )
        producer.send(topic, value=event.to_json())
        count += 1
        time.sleep(delay)

    producer.flush()
    return count
