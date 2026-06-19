from __future__ import annotations

import torch
from torch import nn


class MultimodalMoEFusion(nn.Module):
    """Task-aware mixture-of-experts fusion for missing-robust multimodal MRI."""

    TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival")

    def __init__(self, embed_dim: int = 256, num_experts: int = 7, top_k: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.task_to_id = {task: idx for idx, task in enumerate(self.TASKS)}
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
                for _ in range(num_experts)
            ]
        )
        self.task_embed = nn.Embedding(len(self.TASKS), embed_dim)
        self.gate = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_experts))

    def forward(self, feature: torch.Tensor, tasks: tuple[str, ...] | list[str]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        expert_outputs = torch.stack([expert(feature) for expert in self.experts], dim=1)
        fused: dict[str, torch.Tensor] = {}
        weights_by_task: dict[str, torch.Tensor] = {}
        for task in tasks:
            task_ids = feature.new_full((feature.shape[0],), self.task_to_id.get(task, 0), dtype=torch.long)
            task_feature = self.task_embed(task_ids)
            logits = self.gate(torch.cat([feature, task_feature], dim=-1))
            top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
            sparse_weights = torch.zeros_like(logits)
            # softmax is promoted to fp32 under autocast even when ``top_values`` is fp16, so cast back
            # to the buffer dtype before scattering.
            sparse_weights.scatter_(1, top_indices, torch.softmax(top_values, dim=-1).to(sparse_weights.dtype))
            fused[task] = torch.sum(expert_outputs * sparse_weights.unsqueeze(-1), dim=1)
            weights_by_task[task] = sparse_weights
        all_weights = torch.stack(list(weights_by_task.values()), dim=1)
        balance = all_weights.mean(dim=(0, 1)).var()
        sparse = all_weights.abs().mean()
        return {"fused": fused, "expert_weights": weights_by_task, "losses": {"moe_balance": balance, "moe_sparse": sparse}}
