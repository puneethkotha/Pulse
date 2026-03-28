"""
Event schema for the streaming interaction layer.

Events are produced to Kafka and consumed to update Redis feature store.
This schema is intentionally kept separate from the offline training schema
so the two layers remain independently deployable.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
import time


EVENT_TYPES = ("view", "click", "rate", "skip")


@dataclass
class InteractionEvent:
    """
    A single user-item interaction event.

    Fields
    ------
    event_id : str
        UUID for deduplication.
    event_type : str
        One of: view, click, rate, skip.
    user_id : int
        MovieLens user identifier.
    item_id : int
        MovieLens item identifier.
    rating : Optional[float]
        Explicit rating 1-5, present only for event_type == 'rate'.
    timestamp : float
        Unix epoch seconds (float for sub-second resolution).
    session_id : Optional[str]
        Optional session grouping identifier.
    """
    event_id: str
    event_type: str
    user_id: int
    item_id: int
    rating: Optional[float]
    timestamp: float
    session_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "InteractionEvent":
        data = json.loads(raw)
        return cls(**data)

    def validate(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: {self.event_type}")
        if self.event_type == "rate" and self.rating is None:
            raise ValueError("rating is required for event_type='rate'")
        if self.rating is not None and not (1.0 <= self.rating <= 5.0):
            raise ValueError(f"rating must be in [1, 5], got {self.rating}")


@dataclass
class UserOnlineFeatures:
    """
    Rolling features maintained per user in Redis.

    Updated incrementally by the Kafka consumer.
    """
    user_id: int
    num_events: int
    num_ratings: int
    avg_rating: float
    last_item_ids: list[int]          # last N items interacted with
    genre_counts: dict[str, int]      # genre -> interaction count
    last_updated: float               # unix timestamp

    def to_json(self) -> str:
        return json.dumps({
            "user_id": self.user_id,
            "num_events": self.num_events,
            "num_ratings": self.num_ratings,
            "avg_rating": self.avg_rating,
            "last_item_ids": self.last_item_ids,
            "genre_counts": self.genre_counts,
            "last_updated": self.last_updated,
        })

    @classmethod
    def from_json(cls, raw: str | bytes) -> "UserOnlineFeatures":
        data = json.loads(raw)
        return cls(**data)

    @classmethod
    def empty(cls, user_id: int) -> "UserOnlineFeatures":
        return cls(
            user_id=user_id,
            num_events=0,
            num_ratings=0,
            avg_rating=0.0,
            last_item_ids=[],
            genre_counts={},
            last_updated=time.time(),
        )
