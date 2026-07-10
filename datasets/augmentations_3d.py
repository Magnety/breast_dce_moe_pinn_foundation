from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RandomAugmentation3D:
    """Lightweight 3D augmentation for variable-modality breast MRI volumes."""

    flip_prob: float = 0.5
    noise_prob: float = 0.2
    noise_std: float = 0.03
    scale_prob: float = 0.3
    scale_range: tuple[float, float] = (0.9, 1.1)
    shift_prob: float = 0.3
    shift_range: tuple[float, float] = (-0.05, 0.05)

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        # volume shape: [D, H, W]
        if np.random.rand() < self.flip_prob:
            axis = int(np.random.choice([1, 2]))
            volume = np.flip(volume, axis=axis).copy()
        if np.random.rand() < self.scale_prob:
            low, high = self.scale_range
            volume = volume * np.random.uniform(low, high)
        if np.random.rand() < self.shift_prob:
            low, high = self.shift_range
            volume = volume + np.random.uniform(low, high)
        if np.random.rand() < self.noise_prob:
            volume = volume + np.random.normal(0.0, self.noise_std, size=volume.shape)
        return volume.astype(np.float32)


def build_augmentation(config: dict[str, Any] | None, enabled: bool) -> RandomAugmentation3D | None:
    if not enabled:
        return None
    cfg = config or {}
    return RandomAugmentation3D(
        flip_prob=float(cfg.get("flip_prob", 0.5)),
        noise_prob=float(cfg.get("noise_prob", 0.2)),
        noise_std=float(cfg.get("noise_std", 0.03)),
        scale_prob=float(cfg.get("scale_prob", 0.3)),
        scale_range=tuple(cfg.get("scale_range", (0.9, 1.1))),
        shift_prob=float(cfg.get("shift_prob", 0.3)),
        shift_range=tuple(cfg.get("shift_range", (-0.05, 0.05))),
    )
