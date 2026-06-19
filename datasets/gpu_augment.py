"""GPU-side batch augmentation for variable-modality breast MRI.

CPU 端 augmentation（``RandomAugmentation3D``）在 worker 里跑 numpy 随机操作，
对 ``(B, V, 1, D, H, W)`` 这种 batch 是 dataloader 主要的瓶颈：每个样本要做
12 次 ``np.random.normal(size=(64,128,128))`` 就足以把 GPU 饿死。

把同样的几何/强度增强搬到 GPU，在 batch tensor 已经落到设备之后做，
单步开销只有几十微秒，而且天然按 ``available`` mask 跳过 padded slot。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GPUBatchAugment:
    flip_prob: float = 0.5
    noise_prob: float = 0.2
    noise_std: float = 0.03
    scale_prob: float = 0.3
    scale_low: float = 0.9
    scale_high: float = 1.1
    shift_prob: float = 0.3
    shift_low: float = -0.05
    shift_high: float = 0.05

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        volumes = batch.get("volumes")
        if not torch.is_tensor(volumes) or volumes.numel() == 0:
            return batch
        # volumes: [B, V, C, D, H, W]; available: [B, V] bool
        available = batch.get("modality_available_mask")
        device = volumes.device
        b, v = volumes.shape[:2]

        # 所有随机数一次性在 GPU 上生成；广播到 volumes 上做就地修改。
        # 仅对 available 的 slot 应用增强；padded slot 的值反正会被下游 mask 屏蔽，
        # 但保持 0 让模型 forward 数值更稳。
        if available is None:
            avail_f = volumes.new_ones((b, v))
        else:
            avail_f = available.to(device=device, dtype=volumes.dtype)
        # broadcast shape for per-(B,V) coefficients
        coef_shape = (b, v, 1, 1, 1, 1)

        # 1) 强度缩放：每 (B,V) 独立采样
        if self.scale_prob > 0:
            do_scale = (torch.rand((b, v), device=device) < self.scale_prob).to(volumes.dtype)
            scale = (
                torch.rand((b, v), device=device) * (self.scale_high - self.scale_low)
                + self.scale_low
            )
            scale = scale * do_scale + (1.0 - do_scale)
            scale = scale * avail_f + (1.0 - avail_f)  # padded slot 不变
            volumes = volumes * scale.view(coef_shape)

        # 2) 强度平移
        if self.shift_prob > 0:
            do_shift = (torch.rand((b, v), device=device) < self.shift_prob).to(volumes.dtype)
            shift = (
                torch.rand((b, v), device=device) * (self.shift_high - self.shift_low)
                + self.shift_low
            )
            shift = shift * do_shift * avail_f
            volumes = volumes + shift.view(coef_shape)

        # 3) 高斯噪声：randn_like 在 GPU 几乎免费
        if self.noise_prob > 0 and self.noise_std > 0:
            do_noise = (torch.rand((b, v), device=device) < self.noise_prob).to(volumes.dtype)
            do_noise = (do_noise * avail_f).view(coef_shape)
            noise = torch.randn_like(volumes) * (self.noise_std * do_noise)
            volumes = volumes + noise

        # 4) 翻转：H/W 维各自决定。整 batch 共用一次决定（或者按样本独立）；
        #    样本独立翻转需要 gather，开销不值得，按 batch-level 即可。
        #    用 CPU 的 random（python 端）决定，避免在 GPU 上做 .item() 触发 sync。
        if self.flip_prob > 0:
            import random as _random  # 延迟 import，函数体内
            if _random.random() < self.flip_prob:
                volumes = torch.flip(volumes, dims=(-2,))  # H
            if _random.random() < self.flip_prob:
                volumes = torch.flip(volumes, dims=(-1,))  # W

        batch["volumes"] = volumes
        return batch


def build_gpu_augment(config: dict[str, Any] | None) -> GPUBatchAugment | None:
    if not config:
        return None
    cfg = dict(config)
    if not bool(cfg.get("enabled", True)):
        return None
    scale_range = tuple(cfg.get("scale_range", (0.9, 1.1)))
    shift_range = tuple(cfg.get("shift_range", (-0.05, 0.05)))
    return GPUBatchAugment(
        flip_prob=float(cfg.get("flip_prob", 0.5)),
        noise_prob=float(cfg.get("noise_prob", 0.2)),
        noise_std=float(cfg.get("noise_std", 0.03)),
        scale_prob=float(cfg.get("scale_prob", 0.3)),
        scale_low=float(scale_range[0]),
        scale_high=float(scale_range[1]),
        shift_prob=float(cfg.get("shift_prob", 0.3)),
        shift_low=float(shift_range[0]),
        shift_high=float(shift_range[1]),
    )
