from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def robust_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Percentile normalize a 3D medical image without assuming scanner scale."""

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(finite, (1, 99))
    volume = np.clip(volume, low, high)
    mean = float(volume.mean())
    std = float(volume.std())
    return ((volume - mean) / (std + eps)).astype(np.float32)


def center_crop_or_pad(volume: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Center crop/pad ``volume`` to ``target_shape``.

    Shapes are interpreted as ``[D, H, W]`` inside this package.
    """

    target = np.asarray(tuple(int(v) for v in target_shape), dtype=np.int64)
    current = np.asarray(volume.shape, dtype=np.int64)
    result = np.zeros(tuple(target), dtype=volume.dtype)
    src_start = np.maximum(0, (current - target) // 2)
    dst_start = np.maximum(0, (target - current) // 2)
    copy_shape = np.minimum(current, target)
    src_slices = tuple(slice(int(s), int(s + c)) for s, c in zip(src_start, copy_shape))
    dst_slices = tuple(slice(int(s), int(s + c)) for s, c in zip(dst_start, copy_shape))
    result[dst_slices] = volume[src_slices]
    return result
