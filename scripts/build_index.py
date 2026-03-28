"""
Build the FAISS ANN index from saved item embeddings.

Usage:
    python scripts/build_index.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.faiss_index import build_ivf_index, save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/embeddings/item_embeddings.npy")
ITEM_ID_MAP_PATH = Path("data/processed/item_id_map.pkl")
INDEX_DIR = Path("indexes/faiss")


def main():
    logger.info("Loading item embeddings from %s", EMBEDDINGS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    logger.info("Embeddings shape: %s", embeddings.shape)

    with open(ITEM_ID_MAP_PATH, "rb") as f:
        item_id_map = pickle.load(f)  # raw_item_id -> internal_idx

    # Invert: internal_idx -> raw_item_id
    internal_to_raw = {v: k for k, v in item_id_map.items()}

    n_items = embeddings.shape[0]
    n_centroids = min(256, n_items // 4)
    logger.info("Building IVF index with %d centroids over %d items.", n_centroids, n_items)

    index = build_ivf_index(embeddings, n_centroids=n_centroids, n_probe=32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_index(
        index,
        internal_to_raw,
        index_path=INDEX_DIR / "item_index.faiss",
        id_map_path=INDEX_DIR / "item_id_map.json",
    )

    logger.info("FAISS index built and saved. Total items indexed: %d", index.ntotal)


if __name__ == "__main__":
    main()
