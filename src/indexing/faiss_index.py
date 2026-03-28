"""
FAISS index builder and ANN searcher for item embeddings.

Supports flat exact search and IVF approximate search.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_DIR = Path("indexes/faiss")
INDEX_PATH = INDEX_DIR / "item_index.faiss"
ID_MAP_PATH = INDEX_DIR / "item_id_map.json"


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product index (assumes L2-normalized embeddings)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def build_ivf_index(
    embeddings: np.ndarray,
    n_centroids: int = 256,
    n_probe: int = 32,
) -> faiss.IndexIVFFlat:
    """
    Build an IVF approximate index for faster large-scale search.

    Falls back to flat if embeddings count < n_centroids.
    """
    n, dim = embeddings.shape
    if n < n_centroids:
        logger.warning(
            "Item count (%d) < n_centroids (%d), falling back to flat index.",
            n, n_centroids
        )
        return build_flat_index(embeddings)

    embs = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(embs)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
    index.train(embs)
    index.add(embs)
    index.nprobe = n_probe
    return index


def save_index(
    index: faiss.Index,
    internal_to_raw: dict[int, int],
    index_path: Path = INDEX_PATH,
    id_map_path: Path = ID_MAP_PATH,
) -> None:
    """Persist FAISS index and the internal-index -> raw-item-id map."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(id_map_path, "w") as f:
        # keys as strings for JSON compatibility
        json.dump({str(k): v for k, v in internal_to_raw.items()}, f)
    logger.info("Saved FAISS index to %s (%d items)", index_path, index.ntotal)


def load_index(
    index_path: Path = INDEX_PATH,
    id_map_path: Path = ID_MAP_PATH,
) -> tuple[faiss.Index, dict[int, int]]:
    """Load FAISS index and id map from disk."""
    index = faiss.read_index(str(index_path))
    with open(id_map_path) as f:
        raw_map = json.load(f)
    internal_to_raw = {int(k): int(v) for k, v in raw_map.items()}
    logger.info("Loaded FAISS index from %s (%d items)", index_path, index.ntotal)
    return index, internal_to_raw


class FaissRetriever:
    """
    ANN retriever backed by a FAISS index.

    Parameters
    ----------
    index : faiss.Index
    internal_to_raw : dict mapping FAISS internal index -> raw item_id
    """

    def __init__(
        self,
        index: faiss.Index,
        internal_to_raw: dict[int, int],
    ) -> None:
        self._index = index
        self._internal_to_raw = internal_to_raw

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
    ) -> list[tuple[int, float]]:
        """
        Return top-K (item_id, score) pairs for a query embedding.

        query_embedding : shape (dim,) or (1, dim), float32, will be L2-normalized.
        """
        emb = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(emb)
        scores, indices = self._index.search(emb, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            raw_id = self._internal_to_raw.get(int(idx))
            if raw_id is not None:
                results.append((raw_id, float(score)))
        return results

    @classmethod
    def from_disk(
        cls,
        index_path: Path = INDEX_PATH,
        id_map_path: Path = ID_MAP_PATH,
    ) -> "FaissRetriever":
        index, internal_to_raw = load_index(index_path, id_map_path)
        return cls(index, internal_to_raw)
