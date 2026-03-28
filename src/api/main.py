"""
FastAPI inference service for the PinRanker recommendation system.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import recommend, events, metrics, embeddings
from src.api.dependencies import AppState, app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading recommendation system artifacts...")
    app_state.load()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="PinRanker",
    description="Real-time embedding-based ranking system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
app.include_router(events.router)
app.include_router(metrics.router)
app.include_router(embeddings.router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "faiss_loaded": app_state.retriever is not None,
        "redis_available": app_state.feature_store.available if app_state.feature_store else False,
        "reranker_loaded": app_state.rerank_model is not None,
    }
