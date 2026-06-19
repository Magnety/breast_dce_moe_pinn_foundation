from __future__ import annotations

import argparse
from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Breast-DCE-MoE-PINN inference.")
    parser.add_argument("--config", default=CONFIG_DIR / "infer_export_maps.yaml")
    parser.add_argument("--output-csv", default="outputs/dce_moe_pinn_inference/predictions.csv")
    args = parser.parse_args()
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.trainers.inferencer import Inferencer

    Inferencer(load_config(args.config)).run(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
