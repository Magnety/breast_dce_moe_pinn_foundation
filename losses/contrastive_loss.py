from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)
