from __future__ import annotations

import torch
from torch import nn


class DCETemporalEncoder(nn.Module):
    """Patch-level temporal encoder for arbitrary DCE phases.

    Input ``patch_tokens`` has shape ``[B, V, P, D]`` where ``V`` is the number
    of volumes/phases and ``P`` is the number of spatial patches per volume.
    Temporal modeling is applied patch-wise across phases, so each spatial
    location sees its own DCE evolution instead of only a volume-level mean
    feature.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        max_phases: int = 32,
        ffn_ratio: float = 4.0,
        patch_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.patch_chunk_size = max(1, int(patch_chunk_size))
        ffn_dim = max(embed_dim, int(round(float(embed_dim) * float(ffn_ratio))))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.phase_order_head = nn.Linear(embed_dim, max_phases)
        self.phase_completion_head = nn.Linear(embed_dim, embed_dim)
        self.future_phase_head = nn.Linear(embed_dim, embed_dim)
        self.patch_attn_score = nn.Linear(embed_dim, 1)
        self.phase_attn_score = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens: torch.Tensor, dce_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 4:
            raise ValueError(f"Expected [B, V, P, D] patch tokens, got {tuple(patch_tokens.shape)}")

        b, v, p, d = patch_tokens.shape
        phase_valid = dce_mask.bool()
        empty_rows = phase_valid.sum(dim=1, keepdim=True) == 0
        first_position = torch.zeros_like(phase_valid)
        first_position[:, :1] = True
        safe_phase_valid = phase_valid | (empty_rows & first_position)
        key_padding_mask = ~safe_phase_valid

        temporal_tokens = patch_tokens.new_empty((b, v, p, d))
        chunk_size = min(self.patch_chunk_size, p)
        for start in range(0, p, chunk_size):
            end = min(start + chunk_size, p)
            current = end - start
            chunk_tokens = patch_tokens[:, :, start:end, :]
            chunk_tokens = chunk_tokens.permute(0, 2, 1, 3).reshape(b * current, v, d)
            chunk_mask = key_padding_mask.unsqueeze(1).expand(b, current, v).reshape(b * current, v)
            encoded_chunk = self.encoder(chunk_tokens, src_key_padding_mask=chunk_mask)
            encoded_chunk = encoded_chunk.reshape(b, current, v, d).permute(0, 2, 1, 3)
            temporal_tokens[:, :, start:end, :] = encoded_chunk

        contextual_tokens = torch.where(
            phase_valid.unsqueeze(-1).unsqueeze(-1),
            temporal_tokens,
            patch_tokens,
        )

        patch_scores = self.patch_attn_score(contextual_tokens).squeeze(-1)
        patch_scores = patch_scores.masked_fill(~phase_valid.unsqueeze(-1), -1e4)
        patch_attention = torch.softmax(patch_scores, dim=-1) * phase_valid.unsqueeze(-1).float()
        phase_features = torch.sum(contextual_tokens * patch_attention.unsqueeze(-1), dim=2)

        phase_scores = self.phase_attn_score(phase_features).squeeze(-1).masked_fill(~safe_phase_valid, -1e4)
        attention = torch.softmax(phase_scores, dim=1)
        attention = attention * (~empty_rows).float()
        pooled = torch.sum(phase_features * attention.unsqueeze(-1), dim=1)
        return {
            "dynamic_feature": pooled,
            "encoded": phase_features,
            "phase_features": phase_features,
            "contextual_tokens": contextual_tokens,
            "temporal_attention": attention,
            "phase_order_logits": self.phase_order_head(phase_features),
            "phase_completion": self.phase_completion_head(phase_features),
            "future_phase": self.future_phase_head(phase_features),
        }
