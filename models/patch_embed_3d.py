from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PatchEmbed3D(nn.Module):
    """Shared 3D patch embedding for all modalities.

    Input shape: ``[B, C, D, H, W]``.
    Output shape: ``[B, N, embed_dim]`` where ``N`` is the 3D patch count.
    """

    def __init__(
        self,
        image_size: Sequence[int] = (64, 128, 128),
        patch_size: Sequence[int] = (16, 16, 4),
        in_channels: int = 1,
        embed_dim: int = 512,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.grid_size = tuple(max(1, i // p) for i, p in zip(self.image_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to flattened patch targets, shape ``[B, N, patch_voxels]``."""

        b, c, d, h, w = x.shape
        pd, ph, pw = self.patch_size
        d_trim, h_trim, w_trim = (d // pd) * pd, (h // ph) * ph, (w // pw) * pw
        x = x[:, :, :d_trim, :h_trim, :w_trim]
        x = x.reshape(b, c, d_trim // pd, pd, h_trim // ph, ph, w_trim // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, c * pd * ph * pw)
        return x

    def unpatchify(
        self,
        patches: torch.Tensor,
        output_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Reconstruct ``[B, C, D, H, W]`` volumes from flattened patches."""

        if patches.ndim != 3:
            raise ValueError(f"Expected [B, N, patch_voxels] patches, got {tuple(patches.shape)}")
        b, n, dim = patches.shape
        pd, ph, pw = self.patch_size
        gd, gh, gw = self.grid_size
        expected_patches = gd * gh * gw
        if n != expected_patches:
            raise ValueError(f"Expected {expected_patches} patches, got {n}")
        patch_voxels = pd * ph * pw
        if dim % patch_voxels != 0:
            raise ValueError(
                f"Patch dimension {dim} is incompatible with patch volume {patch_voxels}"
            )
        c = dim // patch_voxels
        x = patches.reshape(b, gd, gh, gw, c, pd, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, gd * pd, gh * ph, gw * pw)
        if output_shape is None:
            return x
        target = tuple(int(v) for v in output_shape)
        if len(target) == 3:
            target = (c, *target)
        if len(target) != 4:
            raise ValueError(f"output_shape must be [D, H, W] or [C, D, H, W], got {target}")
        _, d, h, w = target
        return x[:, :, :d, :h, :w]
