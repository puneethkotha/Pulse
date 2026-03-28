"""
Redis-backed online feature store interface.

Provides get/set operations for UserOnlineFeatures.
If Redis is unavailable, all reads return None and writes are silently skipped,
so the offline recommender continues to function without the streaming layer.
"""

import logging
from typing import Optional

import redis

from src.streaming.schema import UserOnlineFeatures

logger = logging.getLogger(__name__)

FEATURE_KEY_PREFIX = "user_features:"
DEFAULT_TTL = 3600  # seconds


class OnlineFeatureStore:
    """
    Thin wrapper around Redis for reading and writing user online features.

    Parameters
    ----------
    host : str
    port : int
    db : int
    ttl : int
        Time-to-live for feature keys in seconds.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = DEFAULT_TTL,
    ) -> None:
        self._ttl = ttl
        self._client: Optional[redis.Redis] = None
        try:
            client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
            client.ping()
            self._client = client
            logger.info("Connected to Redis at %s:%d", host, port)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis unavailable (%s). Online features disabled.", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_user_features(self, user_id: int) -> Optional[UserOnlineFeatures]:
        """Return online features for user_id, or None if unavailable."""
        if not self._client:
            return None
        try:
            raw = self._client.get(f"{FEATURE_KEY_PREFIX}{user_id}")
            if raw is None:
                return None
            return UserOnlineFeatures.from_json(raw)
        except Exception as exc:
            logger.warning("Redis read failed for user %d: %s", user_id, exc)
            return None

    def set_user_features(self, features: UserOnlineFeatures) -> bool:
        """Persist user features. Returns True on success."""
        if not self._client:
            return False
        try:
            key = f"{FEATURE_KEY_PREFIX}{features.user_id}"
            self._client.setex(key, self._ttl, features.to_json())
            return True
        except Exception as exc:
            logger.warning("Redis write failed for user %d: %s", features.user_id, exc)
            return False

    def delete_user_features(self, user_id: int) -> bool:
        """Remove features for a user."""
        if not self._client:
            return False
        try:
            self._client.delete(f"{FEATURE_KEY_PREFIX}{user_id}")
            return True
        except Exception as exc:
            logger.warning("Redis delete failed for user %d: %s", user_id, exc)
            return False

    def flush_all(self) -> None:
        """Clear all feature keys (testing only)."""
        if self._client:
            self._client.flushdb()
