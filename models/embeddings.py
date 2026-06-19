from __future__ import annotations

import torch
from torch import nn


class MetadataEmbeddings(nn.Module):
    """Modality, phase, visit and dataset embeddings added to volume tokens."""

    def __init__(
        self,
        embed_dim: int,
        max_modalities: int = 16,
        max_phases: int = 32,
        max_visits: int = 16,
        max_datasets: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.modality = nn.Embedding(max_modalities, embed_dim)
        self.phase = nn.Embedding(max_phases, embed_dim)
        self.visit = nn.Embedding(max_visits, embed_dim)
        self.dataset = nn.Embedding(max_datasets, embed_dim)
        self.relative_time = nn.Sequential(nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_id: torch.Tensor,
        phase_id: torch.Tensor,
        visit_id: torch.Tensor,
        dataset_id: torch.Tensor,
        relative_time: torch.Tensor,
    ) -> torch.Tensor:
        b, v = modality_id.shape
        visit = self.visit(visit_id).unsqueeze(1).expand(b, v, -1)
        dataset = self.dataset(dataset_id).unsqueeze(1).expand(b, v, -1)
        rel = self.relative_time(relative_time.unsqueeze(-1))
        meta = self.modality(modality_id) + self.phase(phase_id.clamp_min(0)) + visit + dataset + rel
        return self.dropout(self.norm(meta))


class SpatialPositionEmbedding(nn.Module):
    """Learnable spatial position embedding for 3D patch tokens."""

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pos_embed[:, : tokens.shape[1]]
