"""
Two-tower retrieval model training script.

Usage:
    python scripts/train_two_tower.py [--config configs/training_config.yaml]
"""

import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import load_processed
from src.models.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/two_tower")
EMBEDDINGS_DIR = Path("data/embeddings")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_user_item_pairs(df, user_id_map, item_id_map):
    """Return (user_idx, item_idx) pairs from a ratings DataFrame."""
    pairs = []
    for row in df.itertuples(index=False):
        uid = user_id_map.get(row.user_id)
        iid = item_id_map.get(row.item_id)
        if uid is not None and iid is not None:
            pairs.append((uid, iid))
    return pairs


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        user_feat_matrix: np.ndarray,
        item_feat_matrix: np.ndarray,
    ):
        self.pairs = pairs
        self.user_feats = torch.tensor(user_feat_matrix, dtype=torch.float32)
        self.item_feats = torch.tensor(item_feat_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_idx, item_idx = self.pairs[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            self.user_feats[user_idx],
            torch.tensor(item_idx, dtype=torch.long),
            self.item_feats[item_idx],
        )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        user_idx, user_feats, item_idx, item_feats = [b.to(device) for b in batch]
        user_emb, item_emb = model(user_idx, user_feats, item_idx, item_feats)
        loss = model.in_batch_loss(user_emb, item_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        user_idx, user_feats, item_idx, item_feats = [b.to(device) for b in batch]
        user_emb, item_emb = model(user_idx, user_feats, item_idx, item_feats)
        loss = model.in_batch_loss(user_emb, item_emb)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def extract_item_embeddings(
    model: TwoTowerModel,
    item_feat_matrix: np.ndarray,
    item_id_map: dict[int, int],
    device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract embeddings for all items. Returns (num_items, emb_dim)."""
    model.eval()
    all_item_ids = sorted(item_id_map.values())
    item_feats = torch.tensor(item_feat_matrix, dtype=torch.float32)
    embeddings = []

    for start in range(0, len(all_item_ids), batch_size):
        batch_idx = all_item_ids[start:start + batch_size]
        idx_t = torch.tensor(batch_idx, dtype=torch.long).to(device)
        feats_t = item_feats[batch_idx].to(device)
        emb = model.encode_item(idx_t, feats_t)
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["two_tower"]

    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)["two_tower"]

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Load data
    data = load_processed()
    train_df = data["train_df"]
    val_df = data["val_df"]
    user_id_map = data["user_id_map"]
    item_id_map = data["item_id_map"]
    user_feat_matrix = data["user_feat_matrix"]
    item_feat_matrix = data["item_feat_matrix"]

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    user_feat_dim = user_feat_matrix.shape[1]
    item_feat_dim = item_feat_matrix.shape[1]

    logger.info(
        "Dataset: %d users, %d items, %d train, %d val",
        num_users, num_items, len(train_df), len(val_df)
    )

    train_pairs = build_user_item_pairs(train_df, user_id_map, item_id_map)
    val_pairs = build_user_item_pairs(val_df, user_id_map, item_id_map)

    train_ds = TwoTowerDataset(train_pairs, user_feat_matrix, item_feat_matrix)
    val_ds = TwoTowerDataset(val_pairs, user_feat_matrix, item_feat_matrix)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        embedding_dim=model_cfg["user_embedding_dim"],
        hidden_dims=model_cfg["tower_hidden_dims"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg["temperature"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    logger.info("Starting training for up to %d epochs.", cfg["num_epochs"])
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(
            "Epoch %d/%d  train=%.4f  val=%.4f  time=%.1fs",
            epoch, cfg["num_epochs"], train_loss, val_loss, elapsed
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": model_cfg,
                "num_users": num_users,
                "num_items": num_items,
                "user_feat_dim": user_feat_dim,
                "item_feat_dim": item_feat_dim,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }
            torch.save(checkpoint, MODEL_DIR / "model.pt")
            logger.info("  Saved best model (val_loss=%.4f)", best_val_loss)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    # Save training history
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss}, f, indent=2)

    # Extract and save item embeddings
    logger.info("Extracting item embeddings...")
    best_ckpt = torch.load(MODEL_DIR / "model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    item_embeddings = extract_item_embeddings(model, item_feat_matrix, item_id_map, device)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "item_embeddings.npy", item_embeddings)
    logger.info("Item embeddings saved: shape %s", item_embeddings.shape)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
