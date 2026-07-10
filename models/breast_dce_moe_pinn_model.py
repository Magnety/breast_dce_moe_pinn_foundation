from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.collate_variable_modalities import MODALITY_VOCAB
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.losses.temporal_loss import phase_order_loss
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.dce_temporal_encoder import DCETemporalEncoder
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.embeddings import MetadataEmbeddings, SpatialPositionEmbedding
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.lesion_query_mil import LesionQueryMIL
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.multimodal_moe_fusion import MultimodalMoEFusion
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.multitask_heads import MultiTaskHeads
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.patch_embed_3d import PatchEmbed3D
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.pinn_hemodynamic import PINNHemodynamicModule

_MODALITY_NAMES = {value: key for key, value in MODALITY_VOCAB.items()}


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
        patch_size: tuple[int, int, int] = (16, 16, 4),
        in_channels: int = 1,
        embed_dim: int = 512,
        num_heads: int = 8,
        encoder_depth: int = 2,
        temporal_depth: int = 2,
        lesion_queries: int = 8,
        moe_top_k: int = 2,
        moe_num_experts: int = 7,
        encoder_mlp_ratio: float = 4.0,
        temporal_mlp_ratio: float = 4.0,
        temporal_patch_chunk_size: int = 128,
        moe_hidden_ratio: float = 1.0,
        moe_gate_hidden_ratio: float = 1.0,
        lesion_pool_ratio: float = 1.0,
        fusion_hidden_ratio: float = 1.0,
        pinn_hidden_dim: int = 128,
        subtype_classes: int = 4,
        mask_ratio: float = 0.4,
        channels_last_3d: bool = False,
        use_param_anchor: bool = True,
        pinn_curve_low_quantile: float = 0.10,
        pinn_curve_high_quantile: float = 0.90,
        pinn_curve_margin_ratio: float = 0.15,
        pinn_curve_upper_quantile: float = 0.995,
        pinn_curve_max_samples: int = 16384,
        pinn_curve_min_voxels: int = 512,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.channels_last_3d = channels_last_3d
        # ``use_param_anchor``：DDP static_graph 兼容性所需的 0 系数全参 anchor。
        # 多卡时它保证每个 rank 看到的 used-set 一致，避免 DDP reducer 死锁；
        # 单卡时它只增加 forward/backward graph 开销而不带任何收益，由 solver
        # 根据 world_size + ddp_static_graph 决定是否启用。
        self.use_param_anchor = bool(use_param_anchor)
        self.pinn_curve_low_quantile = float(min(max(pinn_curve_low_quantile, 0.0), 1.0))
        self.pinn_curve_high_quantile = float(min(max(pinn_curve_high_quantile, 0.0), 1.0))
        self.pinn_curve_margin_ratio = float(max(pinn_curve_margin_ratio, 0.0))
        self.pinn_curve_upper_quantile = float(min(max(pinn_curve_upper_quantile, 0.0), 1.0))
        self.pinn_curve_high_quantile = max(self.pinn_curve_high_quantile, self.pinn_curve_low_quantile)
        self.pinn_curve_upper_quantile = max(self.pinn_curve_upper_quantile, self.pinn_curve_high_quantile)
        self.pinn_curve_max_samples = max(1, int(pinn_curve_max_samples))
        self.pinn_curve_min_voxels = max(1, int(pinn_curve_min_voxels))
        self.patch_embed = PatchEmbed3D(image_size, patch_size, in_channels, embed_dim)
        self.spatial_embed = SpatialPositionEmbedding(self.patch_embed.num_patches, embed_dim)
        self.meta_embed = MetadataEmbeddings(embed_dim)
        encoder_ffn_dim = max(embed_dim, int(round(float(embed_dim) * float(encoder_mlp_ratio))))
        fusion_hidden_dim = max(embed_dim, int(round(float(embed_dim) * float(fusion_hidden_ratio))))
        layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            encoder_ffn_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_depth)
        self.temporal_encoder = DCETemporalEncoder(
            embed_dim,
            num_heads=num_heads,
            depth=temporal_depth,
            ffn_ratio=temporal_mlp_ratio,
            patch_chunk_size=temporal_patch_chunk_size,
        )
        self.lesion_mil = LesionQueryMIL(
            embed_dim,
            num_queries=lesion_queries,
            num_heads=num_heads,
            pool_ratio=lesion_pool_ratio,
        )
        self.pinn = PINNHemodynamicModule(embed_dim, hidden_dim=pinn_hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.moe = MultimodalMoEFusion(
            embed_dim,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            expert_hidden_ratio=moe_hidden_ratio,
            gate_hidden_ratio=moe_gate_hidden_ratio,
        )
        self.heads = MultiTaskHeads(embed_dim, subtype_classes=subtype_classes)
        self.mae_decoder = nn.Linear(embed_dim, in_channels * patch_size[0] * patch_size[1] * patch_size[2])

    def forward(
        self,
        batch: dict[str, Any],
        mode: str = "finetune",
        *,
        collect_visuals: bool = False,
        max_visualized_volumes: int = 4,
    ) -> dict[str, Any]:
        volumes = batch["volumes"]
        b, v, c, d, h, w = volumes.shape
        available = batch["modality_available_mask"].bool()
        flat = volumes.reshape(b * v, c, d, h, w)
        available_flat = available.reshape(-1)
        valid_indices = batch.get("valid_volume_indices")
        dense_valid = bool(batch.get("valid_volume_indices_is_dense", False)) or valid_indices is None
        if not dense_valid:
            valid_indices = valid_indices.to(device=flat.device, non_blocking=True)
            if valid_indices.numel() == 0:
                valid_indices = torch.zeros((1,), device=flat.device, dtype=torch.long)
            flat_valid = flat.index_select(0, valid_indices)
        else:
            flat_valid = flat
        if self.channels_last_3d and flat.is_cuda:
            flat_valid = flat_valid.contiguous(memory_format=torch.channels_last_3d)
        meta = self.meta_embed(
            batch["modality_id"],
            batch["phase_id"].clamp(max=31),
            batch["visit_id"].clamp(max=15),
            batch["dataset_id"].clamp(max=15),
            batch["relative_time"],
        )
        meta_flat = meta.reshape(b * v, -1)
        if not dense_valid:
            meta_flat = meta_flat.index_select(0, valid_indices)
        tokens_valid = self.patch_embed(flat_valid)
        tokens_valid = self.spatial_embed(tokens_valid)
        tokens_valid = tokens_valid + meta_flat.unsqueeze(1)
        encoded_valid = self.encoder(tokens_valid)
        if dense_valid:
            encoded = encoded_valid.reshape(b, v, encoded_valid.shape[1], encoded_valid.shape[2])
            volume_features = encoded_valid.mean(dim=1).reshape(b, v, encoded_valid.shape[2])
        else:
            encoded = encoded_valid.new_zeros((b * v, encoded_valid.shape[1], encoded_valid.shape[2]))
            encoded.index_copy_(0, valid_indices, encoded_valid)
            encoded = encoded.reshape(b, v, encoded.shape[1], encoded.shape[2])
            volume_features = encoded_valid.new_zeros((b * v, encoded_valid.shape[2]))
            volume_features.index_copy_(0, valid_indices, encoded_valid.mean(dim=1))
            volume_features = volume_features.reshape(b, v, encoded_valid.shape[2])
        raw_volume_features = volume_features * available.unsqueeze(-1)

        dce_mask = batch.get("temporal_dce_mask")
        if dce_mask is None:
            dce_mask = available & (batch["modality_id"] == MODALITY_VOCAB["DCE"])
        else:
            dce_mask = dce_mask.bool() & available
        temporal = self.temporal_encoder(encoded, dce_mask)
        temporal_phase_features = temporal.get("phase_features")
        if torch.is_tensor(temporal_phase_features):
            volume_features = torch.where(dce_mask.unsqueeze(-1), temporal_phase_features, raw_volume_features)
        else:
            volume_features = raw_volume_features
        global_feature = volume_features.sum(dim=1) / available.sum(dim=1, keepdim=True).clamp_min(1)
        dce_curve, pinn_voxel_mask = self._dce_curve(volumes, dce_mask)
        pinn = self.pinn(temporal["dynamic_feature"], dce_curve, batch["relative_time"], dce_mask)

        contextual_tokens = temporal.get("contextual_tokens")
        patch_source = contextual_tokens if torch.is_tensor(contextual_tokens) else encoded
        patch_tokens = patch_source.reshape(b, v * encoded.shape[2], encoded.shape[3])
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
        # 单卡训练不需要这个 anchor —— 它只会增加 graph 开销。
        if self.use_param_anchor:
            param_anchor = self._param_anchor(foundation_feature)
        else:
            param_anchor = foundation_feature.sum() * 0.0
        losses["_param_anchor"] = param_anchor
        pretrain_visuals: dict[str, Any] | None = None
        if mode == "pretrain":
            pretrain_available = available_flat if dense_valid else available_flat.index_select(0, valid_indices)
            pretrain_losses = self._pretrain_losses(
                flat_valid,
                tokens_valid,
                pretrain_available,
                return_visual_payload=collect_visuals,
                max_visualized_volumes=max_visualized_volumes,
            )
            pretrain_visuals = pretrain_losses.pop("_visuals", None)
            losses.update(pretrain_losses)
            losses["phase_order"] = phase_order_loss(temporal["phase_order_logits"], batch["phase_id"], dce_mask)
            losses["total"] = (
                losses["mae"]
                + 0.1 * losses["phase_order"]
                + losses["pinn_signal"]
                + 0.1 * losses["pinn_ode"]
                + 0.01 * losses["pinn_param_range"]
                + 0.01 * losses["pinn_aif"]
                + 0.01 * losses["moe_balance"]
                + param_anchor
            )
        aux: dict[str, Any] = {
            "temporal_attention": temporal["temporal_attention"],
            "phase_order_logits": temporal["phase_order_logits"],
            "lesion_attention": lesion["attention_heatmap"],
            "expert_weights": moe["expert_weights"],
            "task_context_weights": moe.get("task_context_weights", {}),
            "hemodynamic": pinn["parameter_maps"],
            "pred_curve": pinn["pred_curve"],
            "dce_curve": dce_curve,
            "dce_mask": dce_mask,
            "pinn_voxel_mask": pinn_voxel_mask,
        }
        if mode == "pretrain" and collect_visuals and pretrain_visuals is not None:
            aux["pretrain_visuals"] = self._build_pretrain_visual_payload(
                pretrain_visuals=pretrain_visuals,
                batch=batch,
                volumes=volumes,
                dce_mask=dce_mask,
                temporal_attention=temporal["temporal_attention"],
                phase_order_logits=temporal["phase_order_logits"],
                pred_curve=pinn["pred_curve"],
                aif_curve=pinn["aif_curve"],
                dce_curve=dce_curve,
                pinn_voxel_mask=pinn_voxel_mask,
                hemodynamic=pinn["parameter_maps"],
                dense_valid=dense_valid,
                valid_indices=valid_indices,
            )
        return {
            "predictions": predictions,
            "losses": losses,
            "aux": aux,
            "features": {
                "foundation_feature": foundation_feature,
                "global_feature": global_feature,
                "dynamic_feature": temporal["dynamic_feature"],
                "lesion_feature": lesion["pooled"],
                "pinn_feature": pinn["feature"],
                "volume_features": volume_features,
                "raw_volume_features": raw_volume_features,
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

    def _pretrain_losses(
        self,
        flat_volumes: torch.Tensor,
        flat_tokens: torch.Tensor,
        available: torch.Tensor,
        *,
        return_visual_payload: bool = False,
        max_visualized_volumes: int = 4,
    ) -> dict[str, Any]:
        # 旧实现把 100% patches 都 decode 一遍再用 mask 过滤，浪费 ~60% 算力
        # 和显存。这里改成只 decode masked patches：每个 volume 选固定数量
        # ``num_mask`` 个 patch，mae_decoder 输入 [N*num_mask, D] 而不是
        # [N*P, D]。固定数量是为了避免 ``bool(tensor.any())`` 这类 per-step
        # CPU-GPU 同步 —— 那种同步会让 GPU 在每个 step 等 CPU。
        patch_targets = self.patch_embed.patchify(flat_volumes)
        n, p, target_dim = patch_targets.shape

        # Python 端纯 shape 判断，不会触发 CUDA sync。
        if n == 0 or p == 0:
            zero = flat_tokens.mean() * 0.0
            if return_visual_payload:
                return {"mae": zero, "phase_order": zero, "_visuals": None}
            return {"mae": zero, "phase_order": zero}

        num_mask = max(1, min(p, int(round(float(p) * float(self.mask_ratio)))))

        # 每个 volume 随机选 num_mask 个 patch（用 topk(largest=False) 保证
        # 每行 num_mask 个，不重复）。
        scores = torch.rand((n, p), device=flat_volumes.device)
        mask_idx = scores.topk(k=num_mask, dim=1, largest=False).indices

        row_idx = torch.arange(n, device=flat_volumes.device).unsqueeze(1).expand(-1, num_mask)
        token_grid = flat_tokens[:, :p]
        masked_tokens = token_grid[row_idx, mask_idx]      # [n, num_mask, embed_dim]
        masked_targets = patch_targets[row_idx, mask_idx]  # [n, num_mask, patch_voxels]

        pred = self.mae_decoder(masked_tokens.reshape(-1, masked_tokens.shape[-1]))
        pred = pred.reshape(n, num_mask, target_dim)

        per_patch = F.mse_loss(
            pred.float(),
            masked_targets.float(),
            reduction="none",
        ).mean(dim=-1)

        # available mask 在 volume 维度起作用；padded volume 不参与 loss。
        weights = available.bool().view(n, 1).float()
        denom = weights.sum().mul(float(num_mask)).clamp_min(1.0)
        mae = (per_patch * weights).sum() / denom

        phase_order = flat_tokens.mean() * 0.0
        if not return_visual_payload:
            return {"mae": mae, "phase_order": phase_order}

        visual_count = max(0, min(int(max_visualized_volumes), n))
        visuals: dict[str, Any] | None = None
        if visual_count > 0:
            vis_targets = patch_targets[:visual_count]
            vis_mask_idx = mask_idx[:visual_count]
            vis_pred = pred[:visual_count]
            vis_rows = torch.arange(visual_count, device=flat_volumes.device).unsqueeze(1).expand(-1, num_mask)

            masked_patch_grid = vis_targets.clone()
            masked_patch_grid[vis_rows, vis_mask_idx] = 0.0
            reconstructed_patch_grid = vis_targets.clone()
            reconstructed_patch_grid[vis_rows, vis_mask_idx] = vis_pred
            mask_patch_grid = vis_targets.new_zeros((visual_count, p, target_dim))
            mask_patch_grid[vis_rows, vis_mask_idx] = 1.0

            target_shape = tuple(int(dim_size) for dim_size in flat_volumes.shape[1:])
            original = flat_volumes[:visual_count]
            masked_input = self.patch_embed.unpatchify(masked_patch_grid, output_shape=target_shape)
            reconstruction = self.patch_embed.unpatchify(reconstructed_patch_grid, output_shape=target_shape)
            mask_volume = self.patch_embed.unpatchify(mask_patch_grid, output_shape=target_shape)
            visuals = {
                "original": original,
                "masked_input": masked_input,
                "reconstruction": reconstruction,
                "absolute_error": (reconstruction - original).abs(),
                "mask": mask_volume,
            }
        return {"mae": mae, "phase_order": phase_order, "_visuals": visuals}

    def _build_pretrain_visual_payload(
        self,
        *,
        pretrain_visuals: dict[str, Any],
        batch: dict[str, Any],
        volumes: torch.Tensor,
        dce_mask: torch.Tensor,
        temporal_attention: torch.Tensor,
        phase_order_logits: torch.Tensor,
        pred_curve: torch.Tensor,
        aif_curve: torch.Tensor,
        dce_curve: torch.Tensor,
        pinn_voxel_mask: torch.Tensor,
        hemodynamic: dict[str, torch.Tensor],
        dense_valid: bool,
        valid_indices: torch.Tensor | None,
    ) -> dict[str, Any]:
        visual_count = int(pretrain_visuals["original"].shape[0]) if torch.is_tensor(pretrain_visuals.get("original")) else 0
        flat_modality = batch["modality_id"].reshape(-1)
        flat_phase = batch["phase_id"].reshape(-1)
        flat_rel_time = batch["relative_time"].reshape(-1)
        if dense_valid:
            selected = torch.arange(flat_modality.shape[0], device=flat_modality.device)[:visual_count]
        else:
            selected = valid_indices[:visual_count] if valid_indices is not None else torch.zeros((0,), device=flat_modality.device, dtype=torch.long)
        volume_labels = []
        for index in selected.tolist():
            modality_name = _MODALITY_NAMES.get(int(flat_modality[index].item()), "unknown")
            phase_value = int(flat_phase[index].item())
            rel_time = float(flat_rel_time[index].item())
            volume_labels.append(
                {
                    "modality": modality_name,
                    "phase_id": phase_value,
                    "relative_time": rel_time,
                    "label": f"{modality_name} phase={phase_value} t={rel_time:.2f}",
                }
            )

        phase_logits = phase_order_logits[:1]
        phase_probs = phase_logits.softmax(dim=-1)
        phase_prediction = phase_probs.argmax(dim=-1)
        phase_confidence = phase_probs.max(dim=-1).values
        phase_order = {
            "target_phase_id": batch["phase_id"][:1],
            "predicted_phase_id": phase_prediction,
            "confidence": phase_confidence,
            "temporal_attention": temporal_attention[:1],
            "relative_time": batch["relative_time"][:1],
            "dce_mask": dce_mask[:1],
            "dce_volumes": volumes[:1],
        }
        pinn = {
            "relative_time": batch["relative_time"][:1],
            "phase_id": batch["phase_id"][:1],
            "target_curve": dce_curve[:1],
            "pred_curve": pred_curve[:1],
            "aif_curve": aif_curve[:1],
            "confidence": phase_confidence,
            "dce_mask": dce_mask[:1],
            "voxel_mask": pinn_voxel_mask[:1],
            "parameters": {key: value[:1] for key, value in hemodynamic.items()},
        }
        identifiers = {
            "patient_id": batch.get("patient_id", [None])[0] if isinstance(batch.get("patient_id"), list) else None,
            "sample_id": batch.get("sample_id", [None])[0] if isinstance(batch.get("sample_id"), list) else None,
            "dataset_name": batch.get("dataset_name", [None])[0] if isinstance(batch.get("dataset_name"), list) else None,
            "visit_timepoint": batch.get("visit_timepoint", [None])[0] if isinstance(batch.get("visit_timepoint"), list) else None,
        }
        pretrain_visuals["volume_labels"] = volume_labels
        pretrain_visuals["phase_order"] = phase_order
        pretrain_visuals["pinn"] = pinn
        pretrain_visuals["identifiers"] = identifiers
        return pretrain_visuals

    def _dce_curve(self, volumes: torch.Tensor, dce_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_volumes = torch.where(torch.isfinite(volumes), volumes, torch.zeros_like(volumes))
        phase_valid = dce_mask.bool()
        spatial_mask = self._build_pinn_voxel_mask(volumes, phase_valid)

        # 先在空间上去掉空气/异常体素，再按 phase 聚合全局 DCE 曲线，
        # 避免背景体素把 PINN 目标曲线压平。
        flat_volumes = safe_volumes.reshape(safe_volumes.shape[0], safe_volumes.shape[1], -1)
        flat_finite = torch.isfinite(volumes).reshape(volumes.shape[0], volumes.shape[1], -1)
        flat_spatial_mask = spatial_mask.reshape(spatial_mask.shape[0], -1)
        curve_weights = phase_valid.unsqueeze(-1) & flat_finite & flat_spatial_mask.unsqueeze(1)
        weights = curve_weights.float()

        curve = (flat_volumes * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        valid = phase_valid.float()
        valid_count = valid.sum(dim=1, keepdim=True)
        has_valid = valid_count > 0

        # 以当前样本中“第一有效 DCE 相位”作为增强基线，而不是对所有有效相位
        # 取均值。这样 target_curve 才符合标准 enhancement curve 的定义。
        first_index = phase_valid.float().argmax(dim=1, keepdim=True)
        first = curve.gather(dim=1, index=first_index)
        first = torch.where(has_valid, first, torch.zeros_like(first))
        relative = (curve - first) / (first.abs() + 1e-6)
        return relative * has_valid.float(), spatial_mask

    def _build_pinn_voxel_mask(self, volumes: torch.Tensor, dce_mask: torch.Tensor) -> torch.Tensor:
        phase_valid = dce_mask.bool().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        finite = torch.isfinite(volumes)
        phase_finite = phase_valid & finite
        voxel_valid_counts = phase_finite.sum(dim=1)
        baseline = torch.where(phase_finite, volumes, torch.zeros_like(volumes)).sum(dim=1)
        baseline = baseline / voxel_valid_counts.clamp_min(1).to(volumes.dtype)
        spatial_valid = voxel_valid_counts > 0

        flat_baseline = baseline.reshape(baseline.shape[0], -1)
        flat_valid = spatial_valid.reshape(spatial_valid.shape[0], -1)
        flat_mask = torch.zeros_like(flat_valid)

        for sample_idx in range(flat_baseline.shape[0]):
            sample_valid = flat_valid[sample_idx]
            sample_values = flat_baseline[sample_idx][sample_valid]
            if sample_values.numel() == 0:
                continue
            # 用抽样 quantile 做自适应阈值，去掉极低信号空气区和极亮异常区。
            stats_values = self._subsample_pinn_curve_values(sample_values)
            low = torch.quantile(stats_values, self.pinn_curve_low_quantile)
            high = torch.quantile(stats_values, self.pinn_curve_high_quantile)
            upper = torch.quantile(stats_values, self.pinn_curve_upper_quantile)
            lower = low + (high - low) * self.pinn_curve_margin_ratio

            sample_mask = sample_valid & (flat_baseline[sample_idx] >= lower) & (flat_baseline[sample_idx] <= upper)
            if int(sample_mask.sum().item()) < self.pinn_curve_min_voxels:
                relaxed_mask = sample_valid & (flat_baseline[sample_idx] >= low) & (flat_baseline[sample_idx] <= upper)
                sample_mask = relaxed_mask if int(relaxed_mask.sum().item()) >= int(sample_mask.sum().item()) else sample_mask
            if int(sample_mask.sum().item()) < self.pinn_curve_min_voxels:
                sample_mask = sample_valid
            flat_mask[sample_idx] = sample_mask

        return flat_mask.reshape_as(spatial_valid)

    def _subsample_pinn_curve_values(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= self.pinn_curve_max_samples:
            return values
        step = max(1, values.numel() // self.pinn_curve_max_samples)
        return values[::step][: self.pinn_curve_max_samples]
