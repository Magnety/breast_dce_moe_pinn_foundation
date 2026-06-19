from __future__ import annotations

import torch


def weighted_pinn_loss(losses: dict[str, torch.Tensor], weights: dict[str, float] | None = None) -> torch.Tensor:
    weights = weights or {
        "pinn_signal": 1.0,
        "pinn_ode": 0.1,
        "pinn_param_range": 0.01,
        "pinn_aif": 0.01,
    }
    total = None
    for name, weight in weights.items():
        if name not in losses:
            continue
        value = float(weight) * losses[name]
        total = value if total is None else total + value
    if total is None:
        raise ValueError("No PINN loss keys were present.")
    return total
