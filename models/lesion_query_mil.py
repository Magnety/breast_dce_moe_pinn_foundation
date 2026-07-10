from __future__ import annotations

import torch
from torch import nn


class LesionQueryMIL(nn.Module):
    """ROI-free lesion query module using learnable cross-attention queries."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_queries: int = 8,
        num_heads: int = 8,
        topk: int = 16,
        pool_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.topk = topk
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        pool_hidden = max(embed_dim, int(round(float(embed_dim) * float(pool_ratio))))
        self.pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, pool_hidden),
            nn.GELU(),
            nn.Linear(pool_hidden, embed_dim),
            nn.GELU(),
        )

    def forward(self, patch_tokens: torch.Tensor, patch_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b = patch_tokens.shape[0]
        queries = self.queries.expand(b, -1, -1)
        lesion_tokens, weights = self.cross_attn(
            queries,
            patch_tokens,
            patch_tokens,
            key_padding_mask=~patch_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        lesion_tokens = self.norm(lesion_tokens + queries)
        heatmap = weights.mean(dim=1)
        suspicious_score = heatmap.max(dim=1).values.masked_fill(~patch_mask, -1.0)
        k = min(self.topk, suspicious_score.shape[1])
        topk_idx = suspicious_score.topk(k=k, dim=1).indices
        topk_features = torch.gather(patch_tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
        pooled = self.pool(torch.cat([lesion_tokens, topk_features], dim=1).mean(dim=1))
        entropy = -(heatmap.clamp_min(1e-8) * heatmap.clamp_min(1e-8).log()).sum(dim=-1).mean()
        sparsity = heatmap.abs().mean()
        return {
            "lesion_tokens": lesion_tokens,
            "attention_heatmap": heatmap,
            "topk_features": topk_features,
            "pooled": pooled,
            "losses": {"attention_entropy": entropy, "attention_sparsity": sparsity},
        }
