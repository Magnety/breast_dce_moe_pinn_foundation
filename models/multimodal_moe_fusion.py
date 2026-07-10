from __future__ import annotations

import torch
from torch import nn


def _make_expert(embed_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, embed_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class MultimodalMoEFusion(nn.Module):
    """Task-aware MoE with shared experts and a pCR-specific routed expert.

    Non-pCR tasks route over a shared expert pool. The pCR branch first reads
    the fused auxiliary-task representations, builds a task-conditioned context,
    and then routes over ``shared experts + one dedicated pCR expert``. This
    lets HER2/ER/PR/HR/subtype/survival experts explicitly refine pCR routing
    without hard-wiring auxiliary task heads into the pCR classifier.
    """

    TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival")
    PCR_TASK = "pCR"
    def __init__(
        self,
        embed_dim: int = 256,
        num_experts: int = 7,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_hidden_ratio: float = 1.0,
        gate_hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_shared_experts = max(1, int(num_experts))
        self.num_experts = self.num_shared_experts
        self.top_k = min(max(1, int(top_k)), self.num_shared_experts + 1)
        self.task_to_id = {task: idx for idx, task in enumerate(self.TASKS)}
        expert_hidden = max(embed_dim, int(round(float(embed_dim) * float(expert_hidden_ratio))))
        gate_hidden = max(embed_dim, int(round(float(embed_dim) * float(gate_hidden_ratio))))

        self.shared_experts = nn.ModuleList(
            [_make_expert(embed_dim, expert_hidden, dropout) for _ in range(self.num_shared_experts)]
        )
        self.pcr_expert = _make_expert(embed_dim, expert_hidden, dropout)

        self.task_embed = nn.Embedding(len(self.TASKS), embed_dim)
        self.shared_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_shared_experts),
        )
        self.pcr_context_router = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.pcr_condition_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, embed_dim),
            nn.GELU(),
        )
        self.pcr_gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_shared_experts + 1),
        )

    def forward(self, feature: torch.Tensor, tasks: tuple[str, ...] | list[str]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        shared_outputs = torch.stack([expert(feature) for expert in self.shared_experts], dim=1)
        fused: dict[str, torch.Tensor] = {}
        weights_by_task: dict[str, torch.Tensor] = {}
        task_context_weights: dict[str, torch.Tensor] = {}

        requested_tasks = [task for task in tasks if task in self.task_to_id]
        non_pcr_tasks = [task for task in requested_tasks if task != self.PCR_TASK]
        for task in non_pcr_tasks:
            task_feature = self._task_feature(feature, task)
            logits = self.shared_gate(torch.cat([feature, task_feature], dim=-1))
            fused[task], weights_by_task[task] = self._route(shared_outputs, logits)

        if self.PCR_TASK in requested_tasks:
            pcr_task_feature = self._task_feature(feature, self.PCR_TASK)
            aux_context, aux_weights = self._build_pcr_context(
                feature=feature,
                pcr_task_feature=pcr_task_feature,
                auxiliary_tasks=non_pcr_tasks,
                auxiliary_fused=fused,
            )
            task_context_weights[self.PCR_TASK] = aux_weights
            pcr_input = self.pcr_condition_proj(torch.cat([feature, pcr_task_feature, aux_context], dim=-1))
            pcr_specific_output = self.pcr_expert(pcr_input)
            pcr_outputs = torch.cat([shared_outputs, pcr_specific_output.unsqueeze(1)], dim=1)
            pcr_logits = self.pcr_gate(torch.cat([feature, pcr_task_feature, aux_context], dim=-1))
            fused[self.PCR_TASK], weights_by_task[self.PCR_TASK] = self._route(pcr_outputs, pcr_logits)

        balance = self._shared_balance(feature, weights_by_task)
        sparse = self._sparse_penalty(feature, weights_by_task)
        pcr_usage = self._pcr_specific_usage(feature, weights_by_task)
        pcr_context_entropy = self._pcr_context_entropy(feature, task_context_weights)
        return {
            "fused": fused,
            "expert_weights": weights_by_task,
            "task_context_weights": task_context_weights,
            "losses": {
                "moe_balance": balance,
                "moe_sparse": sparse,
                "moe_pcr_specific_usage": pcr_usage,
                "moe_pcr_context_entropy": pcr_context_entropy,
            },
        }

    def _task_feature(self, feature: torch.Tensor, task: str) -> torch.Tensor:
        task_ids = feature.new_full((feature.shape[0],), self.task_to_id.get(task, 0), dtype=torch.long)
        return self.task_embed(task_ids)

    def _route(self, expert_outputs: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_top_k = min(self.top_k, int(expert_outputs.shape[1]))
        top_values, top_indices = torch.topk(logits, k=current_top_k, dim=-1)
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(1, top_indices, torch.softmax(top_values, dim=-1).to(sparse_weights.dtype))
        fused = torch.sum(expert_outputs * sparse_weights.unsqueeze(-1), dim=1)
        return fused, sparse_weights

    def _build_pcr_context(
        self,
        *,
        feature: torch.Tensor,
        pcr_task_feature: torch.Tensor,
        auxiliary_tasks: list[str],
        auxiliary_fused: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not auxiliary_tasks:
            return torch.zeros_like(feature), feature.new_zeros((feature.shape[0], 0))

        aux_stack = torch.stack([auxiliary_fused[task] for task in auxiliary_tasks], dim=1)
        aux_task_ids = feature.new_tensor([self.task_to_id[task] for task in auxiliary_tasks], dtype=torch.long)
        aux_task_embed = self.task_embed(aux_task_ids).unsqueeze(0).expand(feature.shape[0], -1, -1)
        feature_expanded = feature.unsqueeze(1).expand(-1, aux_stack.shape[1], -1)
        pcr_expanded = pcr_task_feature.unsqueeze(1).expand(-1, aux_stack.shape[1], -1)
        router_input = torch.cat([aux_stack, feature_expanded, pcr_expanded, aux_task_embed], dim=-1)
        aux_logits = self.pcr_context_router(router_input).squeeze(-1)
        aux_weights = torch.softmax(aux_logits, dim=1)
        aux_context = torch.sum(aux_stack * aux_weights.unsqueeze(-1), dim=1)
        return aux_context, aux_weights

    def _shared_balance(self, feature: torch.Tensor, weights_by_task: dict[str, torch.Tensor]) -> torch.Tensor:
        shared_weights: list[torch.Tensor] = []
        for task, weights in weights_by_task.items():
            if task == self.PCR_TASK:
                shared_weights.append(weights[:, : self.num_shared_experts])
            else:
                shared_weights.append(weights)
        if not shared_weights:
            return feature.mean() * 0.0
        stacked = torch.stack(shared_weights, dim=1)
        return stacked.mean(dim=(0, 1)).var()

    def _sparse_penalty(self, feature: torch.Tensor, weights_by_task: dict[str, torch.Tensor]) -> torch.Tensor:
        if not weights_by_task:
            return feature.mean() * 0.0
        means = [weights.abs().mean(dim=1) for weights in weights_by_task.values()]
        return torch.stack(means, dim=1).mean()

    def _pcr_specific_usage(self, feature: torch.Tensor, weights_by_task: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = weights_by_task.get(self.PCR_TASK)
        if weights is None or weights.shape[1] <= self.num_shared_experts:
            return feature.mean() * 0.0
        return weights[:, -1].mean()

    def _pcr_context_entropy(self, feature: torch.Tensor, task_context_weights: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = task_context_weights.get(self.PCR_TASK)
        if weights is None or weights.numel() == 0:
            return feature.mean() * 0.0
        stabilized = weights.clamp_min(1e-8)
        return -(stabilized * stabilized.log()).sum(dim=1).mean()
