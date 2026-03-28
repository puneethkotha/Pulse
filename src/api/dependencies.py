"""
Shared application state and loaded artifacts for the FastAPI service.

All heavy resources (FAISS index, models, feature store) are loaded once
at startup and shared across requests via the AppState singleton.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from src.indexing.faiss_index import FaissRetriever
from src.features.online_store import OnlineFeatureStore
from src.models.reranker import RerankModel, RERANKER_INPUT_DIM
from src.models.two_tower import TwoTowerModel
from src.data.schema import ItemFeatures, UserFeatures

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
CONFIG_PATH = Path("configs/api_config.yaml")


class AppState:
    """Holds all loaded artifacts shared across API requests."""

    def __init__(self) -> None:
        self.retriever: Optional[FaissRetriever] = None
        self.rerank_model: Optional[RerankModel] = None
        self.two_tower_model: Optional[TwoTowerModel] = None
        self.feature_store: Optional[OnlineFeatureStore] = None
        self.item_features: Optional[dict[int, ItemFeatures]] = None
        self.user_features: Optional[dict[int, UserFeatures]] = None
        self.user_id_map: Optional[dict[int, int]] = None
        self.item_id_map: Optional[dict[int, int]] = None
        self.user_feat_matrix: Optional[np.ndarray] = None
        self.item_feat_matrix: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.eval_metrics: Optional[dict] = None
        self.config: dict = {}

    def load(self) -> None:
        """Load all artifacts from disk. Called once at startup."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self.config = yaml.safe_load(f)

        self._load_processed_data()
        self._load_faiss_index()
        self._load_models()
        self._load_feature_store()
        self._load_eval_metrics()

    def _load_processed_data(self) -> None:
        try:
            with open(PROCESSED_DIR / "item_features.pkl", "rb") as f:
                self.item_features = pickle.load(f)
            with open(PROCESSED_DIR / "user_features.pkl", "rb") as f:
                self.user_features = pickle.load(f)
            with open(PROCESSED_DIR / "user_id_map.pkl", "rb") as f:
                self.user_id_map = pickle.load(f)
            with open(PROCESSED_DIR / "item_id_map.pkl", "rb") as f:
                self.item_id_map = pickle.load(f)
            self.user_feat_matrix = np.load(PROCESSED_DIR / "user_feat_matrix.npy")
            self.item_feat_matrix = np.load(PROCESSED_DIR / "item_feat_matrix.npy")
            logger.info(
                "Loaded %d items, %d users from processed data.",
                len(self.item_features), len(self.user_features),
            )
        except FileNotFoundError as e:
            logger.error("Processed data not found: %s. Run the preprocessing pipeline first.", e)

    def _load_faiss_index(self) -> None:
        try:
            faiss_cfg = self.config.get("faiss", {})
            idx_path = Path(faiss_cfg.get("index_path", "indexes/faiss/item_index.faiss"))
            id_map_path = Path(faiss_cfg.get("id_map_path", "indexes/faiss/item_id_map.json"))
            self.retriever = FaissRetriever.from_disk(idx_path, id_map_path)
            logger.info("FAISS index loaded.")
        except Exception as e:
            logger.warning("FAISS index not loaded: %s", e)

    def _load_models(self) -> None:
        # Two-tower
        tt_path = MODEL_DIR / "two_tower" / "model.pt"
        emb_path = Path("data/embeddings/item_embeddings.npy")
        if tt_path.exists():
            try:
                checkpoint = torch.load(tt_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                num_users = checkpoint["num_users"]
                num_items = checkpoint["num_items"]
                user_feat_dim = checkpoint["user_feat_dim"]
                item_feat_dim = checkpoint["item_feat_dim"]
                self.two_tower_model = TwoTowerModel(
                    num_users=num_users,
                    num_items=num_items,
                    user_feat_dim=user_feat_dim,
                    item_feat_dim=item_feat_dim,
                    embedding_dim=cfg.get("embedding_dim", 64),
                    hidden_dims=cfg.get("hidden_dims", [256, 128]),
                    output_dim=cfg.get("output_dim", 64),
                    dropout=cfg.get("dropout", 0.2),
                )
                self.two_tower_model.load_state_dict(checkpoint["model_state_dict"])
                self.two_tower_model.eval()
                logger.info("Two-tower model loaded.")
            except Exception as e:
                logger.warning("Two-tower model failed to load: %s", e)

        if emb_path.exists():
            self.item_embeddings = np.load(emb_path)
            logger.info("Item embeddings loaded, shape: %s", self.item_embeddings.shape)

        # Re-ranker
        rr_path = MODEL_DIR / "reranker" / "model.pt"
        if rr_path.exists():
            try:
                checkpoint = torch.load(rr_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                self.rerank_model = RerankModel(
                    input_dim=RERANKER_INPUT_DIM,
                    hidden_dims=cfg.get("hidden_dims", [128, 64, 32]),
                    dropout=cfg.get("dropout", 0.2),
                )
                self.rerank_model.load_state_dict(checkpoint["model_state_dict"])
                self.rerank_model.eval()
                logger.info("Re-ranker model loaded.")
            except Exception as e:
                logger.warning("Re-ranker model failed to load: %s", e)

    def _load_feature_store(self) -> None:
        redis_cfg = self.config.get("redis", {})
        self.feature_store = OnlineFeatureStore(
            host=redis_cfg.get("host", "localhost"),
            port=redis_cfg.get("port", 6379),
            db=redis_cfg.get("db", 0),
            ttl=redis_cfg.get("feature_ttl_seconds", 3600),
        )

    def _load_eval_metrics(self) -> None:
        metrics_path = Path("artifacts/metrics/evaluation_results.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.eval_metrics = json.load(f)


# Module-level singleton — imported by routers
app_state = AppState()
