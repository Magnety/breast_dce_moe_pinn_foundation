from __future__ import annotations

import torch
from torch import nn


class DCETemporalEncoder(nn.Module):
    """Temporal encoder for arbitrary DCE phases.

    Input ``volume_features`` has shape ``[B, V, D]``. Only records where
    ``dce_mask`` is true participate in temporal modeling; missing phases are
    represented by the mask and never assumed to exist.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, depth: int = 2, max_phases: int = 32) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.phase_order_head = nn.Linear(embed_dim, max_phases)
        self.phase_completion_head = nn.Linear(embed_dim, embed_dim)
        self.future_phase_head = nn.Linear(embed_dim, embed_dim)
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, volume_features: torch.Tensor, dce_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        empty_rows = dce_mask.sum(dim=1, keepdim=True) == 0
        first_position = torch.zeros_like(dce_mask)
        first_position[:, :1] = True
        safe_dce_mask = dce_mask | (empty_rows & first_position)
        key_padding_mask = ~safe_dce_mask
        encoded = self.encoder(volume_features, src_key_padding_mask=key_padding_mask)
        scores = self.attn_score(encoded).squeeze(-1).masked_fill(~safe_dce_mask, -1e4)
        attention = torch.softmax(scores, dim=1)
        attention = attention * (~empty_rows).float()
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        return {
            "dynamic_feature": pooled,
            "encoded": encoded,
            "temporal_attention": attention,
            "phase_order_logits": self.phase_order_head(encoded),
            "phase_completion": self.phase_completion_head(encoded),
            "future_phase": self.future_phase_head(encoded),
        }
