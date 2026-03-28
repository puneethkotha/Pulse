"""
Full evaluation pipeline: BM25, two-tower, two-tower + re-ranker.

Usage:
    python scripts/evaluate.py [--max-users 500]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import load_processed
from src.models.bm25_baseline import build_bm25_index, BM25Retriever
from src.models.two_tower import TwoTowerModel
from src.models.reranker import RerankModel, RERANKER_INPUT_DIM
from src.indexing.faiss_index import FaissRetriever
from src.evaluation.evaluator import (
    evaluate_bm25,
    evaluate_two_tower,
    evaluate_two_tower_plus_reranker,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_two_tower(device):
    ckpt = torch.load("models/two_tower/model.pt", map_location=device)
    cfg = ckpt["config"]
    model = TwoTowerModel(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        user_feat_dim=ckpt["user_feat_dim"],
        item_feat_dim=ckpt["item_feat_dim"],
        embedding_dim=cfg.get("user_embedding_dim", 64),
        hidden_dims=cfg.get("tower_hidden_dims", [256, 128]),
        output_dim=cfg.get("output_dim", 64),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def load_reranker(device):
    ckpt = torch.load("models/reranker/model.pt", map_location=device)
    cfg = ckpt.get("config", {})
    model = RerankModel(
        input_dim=RERANKER_INPUT_DIM,
        hidden_dims=cfg.get("hidden_dims", [128, 64, 32]),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit evaluation to this many test users (for speed)")
    args = parser.parse_args()

    device = torch.device("cpu")  # evaluation runs on CPU

    logger.info("Loading processed data...")
    data = load_processed()
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    user_id_map = data["user_id_map"]
    item_id_map = data["item_id_map"]
    user_feat_matrix = data["user_feat_matrix"]
    item_features = data["item_features"]

    results = {}

    # --- BM25 ---
    logger.info("Building BM25 index...")
    bm25, bm25_item_ids = build_bm25_index(item_features)
    bm25_retriever = BM25Retriever(bm25, bm25_item_ids)
    logger.info("Evaluating BM25...")
    results["bm25"] = evaluate_bm25(
        bm25_retriever, train_df, val_df, test_df, item_features,
        max_users=args.max_users,
    )
    logger.info("BM25: %s", {k: v for k, v in results["bm25"].items() if "@" in k})

    # --- Two-tower ---
    logger.info("Loading two-tower model and FAISS index...")
    two_tower_model = load_two_tower(device)
    retriever = FaissRetriever.from_disk()

    logger.info("Evaluating two-tower retrieval...")
    results["two_tower"] = evaluate_two_tower(
        retriever, two_tower_model,
        user_id_map, item_id_map,
        user_feat_matrix,
        test_df, train_df, val_df,
        max_users=args.max_users,
    )
    logger.info("Two-tower: %s", {k: v for k, v in results["two_tower"].items() if "@" in k})

    # --- Two-tower + Re-ranker ---
    logger.info("Loading re-ranker...")
    rerank_model = load_reranker(device)

    logger.info("Evaluating two-tower + re-ranker...")
    results["two_tower_reranker"] = evaluate_two_tower_plus_reranker(
        retriever, two_tower_model, rerank_model,
        user_id_map, item_id_map,
        user_feat_matrix, item_features,
        test_df, train_df, val_df,
        max_users=args.max_users,
    )
    logger.info(
        "Two-tower+reranker: %s",
        {k: v for k, v in results["two_tower_reranker"].items() if "@" in k}
    )

    save_results(results)
    logger.info("Evaluation complete. Results saved to artifacts/metrics/")


if __name__ == "__main__":
    main()
