from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.losses.survival_loss import cox_partial_likelihood_loss

TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival_time", "survival_event", "survival")
TASK_TO_INDEX = {task: idx for idx, task in enumerate(TASKS)}


class MaskedMultitaskLoss(nn.Module):
    """Multi-task supervised loss with explicit missing-label masks."""

    def __init__(
        self,
        task_weights: dict[str, float] | None = None,
        subtype_consistency_weight: float = 0.05,
        hr_consistency_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.task_weights = task_weights or {"pCR": 1.0, "HER2": 1.0, "ER": 0.5, "PR": 0.5, "HR": 0.5}
        self.subtype_consistency_weight = subtype_consistency_weight
        self.hr_consistency_weight = hr_consistency_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor] | torch.Tensor,
        label_mask: dict[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        device = next(iter(predictions.values())).device
        total = torch.zeros((), device=device)
        for task in ("pCR", "HER2", "ER", "PR", "HR"):
            if task not in predictions:
                continue
            target = _task_value(labels, task, device)
            mask = _task_value(label_mask, task, device)
            if target is None or mask is None:
                continue
            loss = _masked_bce(predictions[task], target, mask)
            losses[task] = loss
            total = total + float(self.task_weights.get(task, 1.0)) * loss
        subtype_target = _task_value(labels, "molecular_subtype", device)
        subtype_mask = _task_value(label_mask, "molecular_subtype", device)
        if "molecular_subtype" in predictions and subtype_target is not None and subtype_mask is not None:
            logits = predictions["molecular_subtype"]
            mask = (subtype_mask > 0).float()
            target = subtype_target.long().clamp(0, logits.shape[-1] - 1)
            per_sample = F.cross_entropy(logits, target, reduction="none")
            loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
            losses["molecular_subtype"] = loss
            total = total + float(self.task_weights.get("molecular_subtype", 0.5)) * loss
        survival_time = _task_value(labels, "survival_time", device)
        survival_event = _task_value(labels, "survival_event", device)
        survival_mask = _task_value(label_mask, "survival", device)
        if (
            "survival_risk" in predictions
            and survival_time is not None
            and survival_event is not None
            and survival_mask is not None
        ):
            loss = cox_partial_likelihood_loss(
                predictions["survival_risk"],
                survival_time,
                survival_event,
                survival_mask,
            )
            losses["survival"] = loss
            total = total + float(self.task_weights.get("survival", 0.5)) * loss
        consistency = self._consistency_losses(predictions)
        losses.update(consistency)
        total = total + self.hr_consistency_weight * consistency["hr_consistency"]
        total = total + self.subtype_consistency_weight * consistency["subtype_consistency"]
        losses["supervised_total"] = total
        return total, losses

    def _consistency_losses(self, predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        zero = next(iter(predictions.values())).mean() * 0.0
        hr_loss = zero
        subtype_loss = zero
        if all(task in predictions for task in ("ER", "PR", "HR")):
            er = torch.sigmoid(predictions["ER"])
            pr = torch.sigmoid(predictions["PR"])
            hr_target = torch.maximum(er, pr).detach()
            hr_loss = F.binary_cross_entropy_with_logits(predictions["HR"], hr_target)
        if all(task in predictions for task in ("ER", "PR", "HER2", "molecular_subtype")):
            subtype_prob = torch.softmax(predictions["molecular_subtype"], dim=-1)
            er = torch.sigmoid(predictions["ER"]).detach()
            pr = torch.sigmoid(predictions["PR"]).detach()
            her2 = torch.sigmoid(predictions["HER2"]).detach()
            hr = torch.maximum(er, pr)
            luminal = hr * (1.0 - her2)
            her2_pos = her2
            triple_neg = (1.0 - er) * (1.0 - pr) * (1.0 - her2)
            target = torch.stack([luminal, luminal * her2, her2_pos, triple_neg], dim=-1)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            subtype_loss = F.kl_div(subtype_prob.clamp_min(1e-6).log(), target, reduction="batchmean")
        return {"hr_consistency": hr_loss, "subtype_consistency": subtype_loss}


def _task_value(container: dict[str, torch.Tensor] | torch.Tensor, task: str, device: torch.device) -> torch.Tensor | None:
    if isinstance(container, dict):
        value = container.get(task)
        return None if value is None else value.to(device, non_blocking=True)
    index = TASK_TO_INDEX.get(task)
    if index is None or container.shape[-1] <= index:
        return None
    return container[..., index].to(device, non_blocking=True)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    loss = F.binary_cross_entropy_with_logits(logits, target.float().clamp(0, 1), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
