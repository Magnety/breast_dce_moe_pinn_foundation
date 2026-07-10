from __future__ import annotations

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    """Prediction heads for pCR/HER2/ER/PR/HR/subtype/survival MVP."""

    def __init__(self, embed_dim: int = 256, subtype_classes: int = 4) -> None:
        super().__init__()
        self.binary_heads = nn.ModuleDict(
            {task: _mlp_head(embed_dim, 1) for task in ("pCR", "HER2", "ER", "PR", "HR")}
        )
        self.subtype = _mlp_head(embed_dim, subtype_classes)
        self.survival = _mlp_head(embed_dim, 1)

    def forward(self, fused: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = {task: head(fused[task]).squeeze(-1) for task, head in self.binary_heads.items() if task in fused}
        if "molecular_subtype" in fused:
            outputs["molecular_subtype"] = self.subtype(fused["molecular_subtype"])
        if "survival" in fused:
            outputs["survival_risk"] = self.survival(fused["survival"]).squeeze(-1)
        return outputs


def _mlp_head(embed_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, out_dim))
