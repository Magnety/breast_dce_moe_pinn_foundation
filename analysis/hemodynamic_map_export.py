from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.pinn_hemodynamic import PINNHemodynamicModule


def export_hemodynamic_maps(
    dce_sequence: torch.Tensor,
    output_dir: str | Path,
    affine: np.ndarray | None = None,
    save_nifti: bool = True,
) -> dict[str, Path]:
    """Export Ktrans/ve/vp/kep/IAUC/washin/washout/peak maps.

    ``dce_sequence`` shape is ``[T, D, H, W]``. If nibabel is not installed,
    maps are still written as ``.npy`` files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module = PINNHemodynamicModule()
    maps = module.voxel_level_export(dce_sequence.float().cpu())
    paths: dict[str, Path] = {}
    nib = None
    if save_nifti:
        try:
            import nibabel as nib  # type: ignore[assignment]
        except ModuleNotFoundError:
            nib = None
    for name, tensor in maps.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        npy_path = output_dir / f"{name}.npy"
        np.save(npy_path, array)
        paths[name] = npy_path
        if nib is not None:
            nii_path = output_dir / f"{name}.nii.gz"
            nib.save(nib.Nifti1Image(array, affine if affine is not None else np.eye(4)), str(nii_path))
            paths[name] = nii_path
    return paths
