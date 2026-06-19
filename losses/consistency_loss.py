from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_consistency_loss(full_logits: torch.Tensor, dropped_logits: torch.Tensor) -> torch.Tensor:
    """Missingness robustness loss for modality/phase dropout augmentation."""

    return F.mse_loss(torch.sigmoid(full_logits), torch.sigmoid(dropped_logits))
