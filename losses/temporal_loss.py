from __future__ import annotations

import torch
import torch.nn.functional as F


def phase_order_loss(logits: torch.Tensor, phase_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & (phase_id > 0)
    target = phase_id.long().clamp(0, logits.shape[-1] - 1)
    loss = F.cross_entropy(logits.flatten(0, -2), target.reshape(-1), reduction="none")
    loss = loss.reshape_as(target)
    weight = valid.float()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
