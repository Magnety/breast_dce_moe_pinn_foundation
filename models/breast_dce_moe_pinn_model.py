from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.collate_variable_modalities import MODALITY_VOCAB
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.dce_temporal_encoder import DCETemporalEncoder
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.embeddings import MetadataEmbeddings, SpatialPositionEmbedding
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.lesion_query_mil import LesionQueryMIL
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.multimodal_moe_fusion import MultimodalMoEFusion
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.multitask_heads import MultiTaskHeads
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.patch_embed_3d import PatchEmbed3D
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.pinn_hemodynamic import PINNHemodynamicModule


class BreastDCEMoEPINNModel(nn.Module):
    """Breast-DCE-MoE-PINN Foundation Model MVP.

    Forward accepts a collated batch and supports ``mode="pretrain"``,
    ``"finetune"`` and ``"infer"``. DCE phases are modeled as a variable
    sequence, not as fixed input channels.
    """

    TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival")

    def __init__(
        self,
        image_size: tuple[int, int, int] = (64, 128, 128),
        patch_size: tuple[int, int, int] = (8, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        encoder_depth: int = 2,
        temporal_depth: int = 2,
        lesion_queries: int = 8,
        moe_top_k: int = 2,
        subtype_classes: int = 4,
        mask_ratio: float = 0.4,
        channels_last_3d: bool = False,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.channels_last_3d = channels_last_3d
        self.patch_embed = PatchEmbed3D(image_size, patch_size, in_channels, embed_dim)
        self.spatial_embed = SpatialPositionEmbedding(self.patch_embed.num_patches, embed_dim)
        self.meta_embed = MetadataEmbeddings(embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_depth)
        self.temporal_encoder = DCETemporalEncoder(embed_dim, num_heads=num_heads, depth=temporal_depth)
        self.lesion_mil = LesionQueryMIL(embed_dim, num_queries=lesion_queries, num_heads=num_heads)
        self.pinn = PINNHemodynamicModule(embed_dim)
        self.fusion = nn.Sequential(nn.LayerNorm(embed_dim * 4), nn.Linear(embed_dim * 4, embed_dim), nn.GELU())
        self.moe = MultimodalMoEFusion(embed_dim, top_k=moe_top_k)
        self.heads = MultiTaskHeads(embed_dim, subtype_classes=subtype_classes)
        self.mae_decoder = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2])

    def forward(self, batch: dict[str, Any], mode: str = "finetune") -> dict[str, Any]:
        volumes = batch["volumes"]
        b, v, c, d, h, w = volumes.shape
        available = batch["modality_available_mask"].bool()
        flat = volumes.reshape(b * v, c, d, h, w)
        available_flat = available.reshape(-1)
        valid_indices = batch.get("valid_volume_indices")
        if valid_indices is None:
            valid_indices = torch.arange(b * v, device=flat.device, dtype=torch.long)
        else:
            valid_indices = valid_indices.to(device=flat.device, non_blocking=True)
        if valid_indices.numel() == 0:
            valid_indices = torch.zeros((1,), device=flat.device, dtype=torch.long)
        flat_valid = flat.index_select(0, valid_indices)
        if self.channels_last_3d and flat.is_cuda:
            flat_valid = flat_valid.contiguous(memory_format=torch.channels_last_3d)
        meta = self.meta_embed(
            batch["modality_id"],
            batch["phase_id"].clamp(max=31),
            batch["visit_id"].clamp(max=15),
            batch["dataset_id"].clamp(max=15),
            batch["relative_time"],
        )
        tokens_valid = self.patch_embed(flat_valid)
        tokens_valid = self.spatial_embed(tokens_valid)
        tokens_valid = tokens_valid + meta.reshape(b * v, -1).index_select(0, valid_indices).unsqueeze(1)
        encoded_valid = self.encoder(tokens_valid)
        encoded = encoded_valid.new_zeros((b * v, encoded_valid.shape[1], encoded_valid.shape[2]))
        encoded.index_copy_(0, valid_indices, encoded_valid)
        encoded = encoded.reshape(b, v, encoded.shape[1], encoded.shape[2])
        volume_features = encoded_valid.new_zeros((b * v, encoded_valid.shape[2]))
        volume_features.index_copy_(0, valid_indices, encoded_valid.mean(dim=1))
        volume_features = volume_features.reshape(b, v, encoded_valid.shape[2])
        volume_features = volume_features * available.unsqueeze(-1)
        global_feature = volume_features.sum(dim=1) / available.sum(dim=1, keepdim=True).clamp_min(1)

        dce_mask = available & (batch["modality_id"] == MODALITY_VOCAB["DCE"])
        temporal = self.temporal_encoder(volume_features, dce_mask)
        dce_curve = self._dce_curve(volumes, dce_mask)
        pinn = self.pinn(temporal["dynamic_feature"], dce_curve, batch["relative_time"], dce_mask)

        patch_tokens = encoded.reshape(b, v * encoded.shape[2], encoded.shape[3])
        patch_mask = available.unsqueeze(-1).expand(-1, -1, encoded.shape[2]).reshape(b, -1)
        lesion = self.lesion_mil(patch_tokens, patch_mask)

        foundation_feature = self.fusion(
            torch.cat([global_feature, temporal["dynamic_feature"], lesion["pooled"], pinn["feature"]], dim=-1)
        )
        moe = self.moe(foundation_feature, self.TASKS)
        predictions = self.heads(moe["fused"])
        losses = {**pinn["losses"], **lesion["losses"], **moe["losses"]}
        # DDP 友好：所有任务 head / MoE expert / temporal 辅助头每步都参与
        # 反向 graph，否则不同 mode（pretrain 不用 heads；finetune 不用所有 task）
        # 会导致 unused-parameter 集合不一致，触发 NCCL all_reduce 死锁。
        # 用 0 系数把模型全部可训练参数挂到 graph 上，开销近似 0。
        param_anchor = self._param_anchor(foundation_feature)
        losses["_param_anchor"] = param_anchor
        if mode == "pretrain":
            losses.update(self._pretrain_losses(flat_valid, tokens_valid, available_flat.index_select(0, valid_indices)))
            losses["total"] = (
                losses["mae"]
                + 0.1 * losses["phase_order"]
                + losses["pinn_signal"]
                + 0.1 * losses["pinn_ode"]
                + 0.01 * losses["moe_balance"]
                + param_anchor
            )
        return {
            "predictions": predictions,
            "losses": losses,
            "aux": {
                "temporal_attention": temporal["temporal_attention"],
                "lesion_attention": lesion["attention_heatmap"],
                "expert_weights": moe["expert_weights"],
                "hemodynamic": pinn["parameter_maps"],
            },
        }

    def _param_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        """0 系数的全参数 anchor，让 DDP 每步看到的 used-set 都是全集。

        放在 reference 设备/类型上，便于直接加进任意 loss 张量。
        """
        anchor = reference.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                anchor = anchor + param.float().sum() * 0.0
        return anchor

    def _pretrain_losses(self, flat_volumes: torch.Tensor, flat_tokens: torch.Tensor, available: torch.Tensor) -> dict[str, torch.Tensor]:
        patch_targets = self.patch_embed.patchify(flat_volumes)
        pred = self.mae_decoder(flat_tokens[:, : patch_targets.shape[1]])
        random_mask = torch.rand(patch_targets.shape[:2], device=flat_volumes.device) < self.mask_ratio
        random_mask = random_mask & available.unsqueeze(1)
        per_patch = F.mse_loss(pred, patch_targets, reduction="none").mean(dim=-1)
        mask = random_mask.float()
        mae = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        phase_order = flat_tokens.mean() * 0.0
        return {"mae": mae, "phase_order": phase_order}

    @staticmethod
    def _dce_curve(volumes: torch.Tensor, dce_mask: torch.Tensor) -> torch.Tensor:
        curve = volumes.mean(dim=(2, 3, 4, 5))
        valid = dce_mask.float()
        valid_count = valid.sum(dim=1, keepdim=True)
        baseline = curve * valid
        first = baseline.sum(dim=1, keepdim=True) / valid_count.clamp_min(1.0)
        relative = (curve - first) / (first.abs() + 1e-6)
        return relative * (valid_count > 0).float()
