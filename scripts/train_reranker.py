"""
Re-ranker training script.

Generates training data from two-tower retrieval candidates,
then trains a pointwise re-ranker with binary cross-entropy loss
(relevant items = positive, retrieved non-relevant = negative).

Usage:
    python scripts/train_reranker.py
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import load_processed
from src.models.two_tower import TwoTowerModel
from src.models.reranker import RerankModel, RERANKER_INPUT_DIM
from src.models.feature_builder import build_candidate_features
from src.indexing.faiss_index import FaissRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/reranker")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_two_tower(device) -> TwoTowerModel:
    ckpt = torch.load("models/two_tower/model.pt", map_location=device)
    model = TwoTowerModel(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        user_feat_dim=ckpt["user_feat_dim"],
        item_feat_dim=ckpt["item_feat_dim"],
        embedding_dim=ckpt["config"].get("user_embedding_dim", 64),
        hidden_dims=ckpt["config"].get("tower_hidden_dims", [256, 128]),
        output_dim=ckpt["config"].get("output_dim", 64),
        dropout=ckpt["config"].get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate_training_examples(
    train_df,
    val_df,
    two_tower_model,
    retriever,
    user_id_map,
    item_id_map,
    user_feat_matrix,
    item_feat_matrix,
    item_features,
    device,
    candidate_pool: int = 50,
    max_users: int = 2000,
    seed: int = 42,
):
    """
    For each sampled user, retrieve candidates, label them (1 if in val positives, 0 otherwise),
    and build feature vectors.

    Returns (features_array, labels_array).
    """
    random.seed(seed)

    # Build val ground truth (items rated >= 4)
    val_positives: dict[int, set[int]] = {}
    for row in val_df[val_df["rating"] >= 4.0].itertuples(index=False):
        val_positives.setdefault(row.user_id, set()).add(row.item_id)

    # User interaction history for genre features
    user_train_interactions: dict[int, list] = {}
    for row in train_df.itertuples(index=False):
        user_train_interactions.setdefault(row.user_id, []).append(
            type("I", (), {"user_id": row.user_id, "item_id": row.item_id, "rating": row.rating})()
        )

    # Seen items (train)
    user_seen: dict[int, set[int]] = {}
    for row in train_df.itertuples(index=False):
        user_seen.setdefault(row.user_id, set()).add(row.item_id)

    users = [u for u in val_positives if u in user_id_map and val_positives[u]]
    random.shuffle(users)
    users = users[:max_users]

    all_feats = []
    all_labels = []
    item_feat_t = torch.tensor(item_feat_matrix, dtype=torch.float32)

    with torch.no_grad():
        for user_id in users:
            uid = user_id_map[user_id]
            u_feat = torch.tensor(user_feat_matrix[uid], dtype=torch.float32).unsqueeze(0).to(device)
            u_idx = torch.tensor([uid], dtype=torch.long).to(device)
            u_emb = two_tower_model.encode_user(u_idx, u_feat).cpu().numpy()[0]

            seen = user_seen.get(user_id, set())
            candidates = retriever.retrieve(u_emb, k=candidate_pool + len(seen))
            candidates = [(iid, s) for iid, s in candidates if iid not in seen][:candidate_pool]
            if not candidates:
                continue

            positives = val_positives[user_id]
            pairs = [(user_id, iid, sim, rank) for rank, (iid, sim) in enumerate(candidates)]
            feats = build_candidate_features(pairs, item_features, user_train_interactions)
            labels = np.array([1.0 if iid in positives else 0.0 for _, (iid, _) in enumerate(candidates)], dtype=np.float32)

            all_feats.append(feats)
            all_labels.append(labels)

    if not all_feats:
        raise RuntimeError("No training examples generated for re-ranker.")

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


def main():
    with open("configs/training_config.yaml") as f:
        cfg = yaml.safe_load(f)["reranker"]

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training re-ranker on device: %s", device)

    data = load_processed()
    train_df = data["train_df"]
    val_df = data["val_df"]
    user_id_map = data["user_id_map"]
    item_id_map = data["item_id_map"]
    user_feat_matrix = data["user_feat_matrix"]
    item_feat_matrix = data["item_feat_matrix"]
    item_features = data["item_features"]

    logger.info("Loading two-tower model and FAISS index...")
    two_tower_model = load_two_tower(device)
    retriever = FaissRetriever.from_disk()

    logger.info("Generating re-ranker training examples (this may take a few minutes)...")
    X, y = generate_training_examples(
        train_df, val_df,
        two_tower_model, retriever,
        user_id_map, item_id_map,
        user_feat_matrix, item_feat_matrix,
        item_features, device,
    )
    logger.info("Generated %d examples. Positive rate: %.3f", len(X), y.mean())

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)["reranker"]

    model = RerankModel(
        input_dim=RERANKER_INPUT_DIM,
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.BCELoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= max(len(val_loader), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, MODEL_DIR / "model.pt")
            logger.info("  Saved best re-ranker (val_loss=%.4f)", best_val_loss)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss}, f, indent=2)

    logger.info("Re-ranker training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
