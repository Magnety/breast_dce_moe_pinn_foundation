from __future__ import annotations

import torch
import torch.nn.functional as F


def mae_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(pred, target, reduction="none")
    weight = mask.float()
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
