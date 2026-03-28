"""
Re-ranking model.

Takes a set of candidate (user, item) pairs with rich features
and outputs a relevance score used to re-order them.

Input features per candidate:
  - embedding_similarity (1): cosine similarity from two-tower retrieval
  - genre_overlap (1): fraction of user's genre preference matching item
  - item_popularity (1): log-normalized num_ratings
  - item_avg_rating (1): avg rating / 5.0
  - item_year_norm (1): normalized release year
  - item_genre_vector (18): multi-hot genre encoding
  - user_avg_rating (1): user's historical avg rating / 5.0
  - user_num_ratings (1): log-normalized
  - genre_pref_vector (18): user genre interaction counts, normalized
  Total: 44 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


RERANKER_INPUT_DIM = 44  # must stay in sync with feature_builder.py


class RerankModel(nn.Module):
    """
    Pointwise re-ranker: maps feature vector to relevance score in [0, 1].
    """

    def __init__(
        self,
        input_dim: int = RERANKER_INPUT_DIM,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, input_dim) float tensor

        Returns
        -------
        scores : (N,) float tensor in [0, 1]
        """
        return self.net(x).squeeze(-1)
