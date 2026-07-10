from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.analysis.hemodynamic_map_export import export_hemodynamic_maps


def main() -> int:
    parser = argparse.ArgumentParser(description="Export approximate DCE hemodynamic parameter maps.")
    parser.add_argument("--dce-npy", required=True, help="Path to a [T,D,H,W] DCE sequence .npy file.")
    parser.add_argument("--output-dir", default="outputs/dce_moe_pinn_maps")
    parser.add_argument("--no-nifti", action="store_true")
    args = parser.parse_args()
    sequence = torch.from_numpy(np.load(Path(args.dce_npy))).float()
    paths = export_hemodynamic_maps(sequence, args.output_dir, save_nifti=not args.no_nifti)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
