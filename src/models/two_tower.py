"""
Two-tower retrieval model.

Architecture:
  - User tower: user_id embedding + user side features -> L2-normalized output
  - Item tower: item_id embedding + item side features -> L2-normalized output

Training uses in-batch negatives with temperature-scaled dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP(nn.Module):
    """Simple feed-forward block with ReLU and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    """
    Encodes a user into a fixed-dim embedding.

    Parameters
    ----------
    num_users : int
    user_feat_dim : int
        Dimension of side features (gender, age, occupation, zip prefix).
    embedding_dim : int
        Learnable user embedding size.
    hidden_dims : list[int]
    output_dim : int
        Final L2-normalized embedding dimension.
    dropout : float
    """

    def __init__(
        self,
        num_users: int,
        user_feat_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        self.mlp = MLP(
            input_dim=embedding_dim + user_feat_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_idx: torch.Tensor,       # (B,) long
        user_feats: torch.Tensor,     # (B, user_feat_dim) float
    ) -> torch.Tensor:
        emb = self.user_embedding(user_idx)               # (B, emb_dim)
        x = torch.cat([emb, user_feats], dim=-1)          # (B, emb_dim + feat_dim)
        out = self.mlp(x)                                  # (B, output_dim)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes an item into a fixed-dim embedding.

    Parameters
    ----------
    num_items : int
    item_feat_dim : int
        Dimension of side features (genre, year, popularity, avg_rating).
    embedding_dim : int
    hidden_dims : list[int]
    output_dim : int
    dropout : float
    """

    def __init__(
        self,
        num_items: int,
        item_feat_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.mlp = MLP(
            input_dim=embedding_dim + item_feat_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        item_idx: torch.Tensor,       # (B,) long
        item_feats: torch.Tensor,     # (B, item_feat_dim) float
    ) -> torch.Tensor:
        emb = self.item_embedding(item_idx)
        x = torch.cat([emb, item_feats], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model combining user and item towers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feat_dim: int,
        item_feat_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 64,
        dropout: float = 0.2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        self.user_tower = UserTower(
            num_users=num_users,
            user_feat_dim=user_feat_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            item_feat_dim=item_feat_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        user_feats: torch.Tensor,
        item_idx: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (user_embeddings, item_embeddings), both L2-normalized.
        """
        user_emb = self.user_tower(user_idx, user_feats)
        item_emb = self.item_tower(item_idx, item_feats)
        return user_emb, item_emb

    def in_batch_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch negative contrastive loss.

        Treats other items in the batch as negatives for each user.
        Similarity is temperature-scaled cosine (dot product of normalized vecs).
        """
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def encode_user(
        self,
        user_idx: torch.Tensor,
        user_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(user_idx, user_feats)

    def encode_item(
        self,
        item_idx: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_idx, item_feats)
