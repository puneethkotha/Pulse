"""
Generate offline demo artifacts for the frontend.

Reads from processed data and evaluation artifacts and writes
pre-computed JSON files to artifacts/offline_demo/ and
frontend/public/offline_data/.

These files allow the frontend to display real results when the
backend is not running. They must be generated after running the
full training and evaluation pipeline.

Usage:
    python scripts/generate_offline_demo.py
"""

import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OFFLINE_DIR = Path("artifacts/offline_demo")
FRONTEND_DATA_DIR = Path("frontend/public/offline_data")


def load_state():
    import pickle
    import numpy as np

    state = {}
    with open("data/processed/item_features.pkl", "rb") as f:
        state["item_features"] = pickle.load(f)
    with open("data/processed/user_features.pkl", "rb") as f:
        state["user_features"] = pickle.load(f)
    with open("data/processed/user_id_map.pkl", "rb") as f:
        state["user_id_map"] = pickle.load(f)
    with open("data/processed/item_id_map.pkl", "rb") as f:
        state["item_id_map"] = pickle.load(f)
    state["user_feat_matrix"] = np.load("data/processed/user_feat_matrix.npy")
    state["item_feat_matrix"] = np.load("data/processed/item_feat_matrix.npy")

    if Path("data/embeddings/item_embeddings.npy").exists():
        state["item_embeddings"] = np.load("data/embeddings/item_embeddings.npy")

    if Path("artifacts/metrics/evaluation_results.json").exists():
        with open("artifacts/metrics/evaluation_results.json") as f:
            state["eval_metrics"] = json.load(f)

    if Path("artifacts/metrics/latency_report.json").exists():
        with open("artifacts/metrics/latency_report.json") as f:
            state["latency_report"] = json.load(f)

    return state


def generate_sample_recommendations(state, n_users: int = 20, k: int = 10) -> list[dict]:
    """
    Generate pre-computed recommendations for a sample of users.

    Uses the FAISS retriever and re-ranker if available, otherwise
    falls back to popularity-sorted items.
    """
    from src.indexing.faiss_index import FaissRetriever
    from src.models.two_tower import TwoTowerModel
    from src.models.reranker import RerankModel, RERANKER_INPUT_DIM
    from src.models.feature_builder import build_candidate_features

    item_features = state["item_features"]
    user_id_map = state["user_id_map"]
    user_feat_matrix = state["user_feat_matrix"]

    # Sample users
    all_user_ids = list(user_id_map.keys())
    random.seed(42)
    sample_users = random.sample(all_user_ids, min(n_users, len(all_user_ids)))

    retriever = None
    two_tower_model = None
    rerank_model = None

    try:
        retriever = FaissRetriever.from_disk()
        ckpt = torch.load("models/two_tower/model.pt", map_location="cpu")
        cfg = ckpt["config"]
        two_tower_model = TwoTowerModel(
            num_users=ckpt["num_users"], num_items=ckpt["num_items"],
            user_feat_dim=ckpt["user_feat_dim"], item_feat_dim=ckpt["item_feat_dim"],
            embedding_dim=cfg.get("user_embedding_dim", 64),
            hidden_dims=cfg.get("tower_hidden_dims", [256, 128]),
            output_dim=cfg.get("output_dim", 64),
        )
        two_tower_model.load_state_dict(ckpt["model_state_dict"])
        two_tower_model.eval()
        logger.info("Two-tower model loaded for offline demo generation.")
    except Exception as e:
        logger.warning("Could not load two-tower model: %s. Using popularity fallback.", e)

    try:
        rr_ckpt = torch.load("models/reranker/model.pt", map_location="cpu")
        rr_cfg = rr_ckpt.get("config", {})
        rerank_model = RerankModel(
            input_dim=RERANKER_INPUT_DIM,
            hidden_dims=rr_cfg.get("hidden_dims", [128, 64, 32]),
        )
        rerank_model.load_state_dict(rr_ckpt["model_state_dict"])
        rerank_model.eval()
    except Exception as e:
        logger.warning("Could not load re-ranker: %s", e)

    # Popularity fallback list (sorted by num_ratings)
    popular_items = sorted(
        item_features.items(), key=lambda x: x[1].num_ratings, reverse=True
    )[:100]

    results = []
    for user_id in sample_users:
        user_idx = user_id_map[user_id]
        recs = []

        if retriever is not None and two_tower_model is not None:
            try:
                u_feat = torch.tensor(user_feat_matrix[user_idx], dtype=torch.float32).unsqueeze(0)
                u_idx = torch.tensor([user_idx], dtype=torch.long)
                with torch.no_grad():
                    u_emb = two_tower_model.encode_user(u_idx, u_feat).numpy()[0]
                candidates = retriever.retrieve(u_emb, k=50)

                if rerank_model is not None:
                    pairs = [(user_id, iid, sim, rank) for rank, (iid, sim) in enumerate(candidates)]
                    feats = build_candidate_features(pairs, item_features, {})
                    feat_t = torch.tensor(feats, dtype=torch.float32)
                    with torch.no_grad():
                        scores = rerank_model(feat_t).numpy()
                    ranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: x[1], reverse=True)
                else:
                    ranked = candidates

                for item_id, score in ranked[:k]:
                    feat = item_features.get(item_id)
                    if feat:
                        recs.append({
                            "item_id": item_id,
                            "score": round(float(score), 6),
                            "title": feat.title,
                            "genres": feat.genres,
                            "avg_rating": feat.avg_rating,
                        })
            except Exception as e:
                logger.warning("Failed to generate recs for user %d: %s", user_id, e)

        if not recs:
            # popularity fallback
            for item_id, feat in popular_items[:k]:
                recs.append({
                    "item_id": item_id,
                    "score": feat.num_ratings / max(f.num_ratings for _, f in popular_items),
                    "title": feat.title,
                    "genres": feat.genres,
                    "avg_rating": feat.avg_rating,
                })

        results.append({"user_id": user_id, "recommendations": recs})

    return results


def main():
    logger.info("Loading state...")
    state = load_state()

    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluation metrics
    if "eval_metrics" in state:
        metrics_out = {
            "source": "offline_pipeline",
            "note": "Metrics computed on MovieLens 1M held-out test set",
            "results": state["eval_metrics"],
        }
        with open(OFFLINE_DIR / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info("Saved evaluation metrics.")

    # Latency report
    if "latency_report" in state:
        with open(OFFLINE_DIR / "latency.json", "w") as f:
            json.dump(state["latency_report"], f, indent=2)
        logger.info("Saved latency report.")

    # Dataset stats
    stats_path = Path("data/processed/stats.json")
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        with open(OFFLINE_DIR / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    # Sample item catalog
    item_features = state["item_features"]
    catalog = [
        {
            "item_id": iid,
            "title": feat.title,
            "genres": feat.genres,
            "year": feat.year,
            "avg_rating": feat.avg_rating,
            "num_ratings": feat.num_ratings,
        }
        for iid, feat in sorted(
            item_features.items(), key=lambda x: x[1].num_ratings, reverse=True
        )[:500]
    ]
    with open(OFFLINE_DIR / "item_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Saved item catalog (%d items).", len(catalog))

    # Sample recommendations
    logger.info("Generating sample recommendations...")
    sample_recs = generate_sample_recommendations(state, n_users=20, k=10)
    with open(OFFLINE_DIR / "sample_recommendations.json", "w") as f:
        json.dump(sample_recs, f, indent=2)
    logger.info("Saved recommendations for %d users.", len(sample_recs))

    # Copy to frontend public dir
    import shutil
    for fname in ["metrics.json", "latency.json", "dataset_stats.json",
                  "item_catalog.json", "sample_recommendations.json"]:
        src = OFFLINE_DIR / fname
        if src.exists():
            shutil.copy(src, FRONTEND_DATA_DIR / fname)

    logger.info("Offline demo data written to %s and %s", OFFLINE_DIR, FRONTEND_DATA_DIR)


if __name__ == "__main__":
    main()
