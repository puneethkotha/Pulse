"""
/event endpoint — receive and process interaction events.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from src.api.dependencies import app_state
from src.streaming.schema import InteractionEvent, EVENT_TYPES

logger = logging.getLogger(__name__)
router = APIRouter()


class EventRequest(BaseModel):
    user_id: int
    item_id: int
    event_type: str
    rating: Optional[float] = None
    session_id: Optional[str] = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        return v

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (1.0 <= v <= 5.0):
            raise ValueError("rating must be between 1.0 and 5.0")
        return v


class EventResponse(BaseModel):
    event_id: str
    status: str
    feature_store_updated: bool


@router.post("/event", response_model=EventResponse)
def post_event(req: EventRequest):
    event = InteractionEvent(
        event_id=str(uuid.uuid4()),
        event_type=req.event_type,
        user_id=req.user_id,
        item_id=req.item_id,
        rating=req.rating,
        timestamp=time.time(),
        session_id=req.session_id,
    )

    try:
        event.validate()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    feature_store_updated = False
    state = app_state
    if state.feature_store and state.feature_store.available:
        from src.streaming.consumer import update_user_features_in_place
        item_features = state.item_features or {}
        existing = state.feature_store.get_user_features(req.user_id)
        updated = update_user_features_in_place(existing, event, item_features)
        feature_store_updated = state.feature_store.set_user_features(updated)

    return EventResponse(
        event_id=event.event_id,
        status="accepted",
        feature_store_updated=feature_store_updated,
    )
