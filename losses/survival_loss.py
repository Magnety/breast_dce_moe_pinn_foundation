from __future__ import annotations

import torch


def cox_partial_likelihood_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative Cox partial likelihood with censoring support."""

    if mask is None:
        mask = torch.ones_like(time)
    valid = (mask > 0) & torch.isfinite(time) & torch.isfinite(event)
    safe_time = torch.where(valid, time, time.new_full(time.shape, float("-inf")))
    order = torch.argsort(safe_time, descending=True)
    risk = risk[order]
    event = event[order].float().clamp_min(0.0)
    event_weight = event * valid[order].float()
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -(risk - log_cumsum) * event_weight
    return loss.sum() / event_weight.sum().clamp_min(1.0)
