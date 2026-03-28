"""
Run the full data preprocessing pipeline.

Usage:
    python scripts/preprocess.py
"""

import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_all
from src.data.preprocessor import preprocess_and_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    with open("configs/training_config.yaml") as f:
        cfg = yaml.safe_load(f)["two_tower"]

    logger.info("Loading MovieLens 1M data...")
    ratings_df, user_features, item_features = load_all()

    logger.info(
        "Raw dataset: %d ratings, %d users, %d items",
        len(ratings_df),
        ratings_df["user_id"].nunique(),
        ratings_df["item_id"].nunique(),
    )

    logger.info("Preprocessing and splitting...")
    stats = preprocess_and_save(
        ratings_df=ratings_df,
        user_features=user_features,
        item_features=item_features,
        min_interactions=cfg["min_user_interactions"],
        train_frac=cfg["train_split"],
        val_frac=cfg["val_split"],
    )

    logger.info("Preprocessing complete:")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
