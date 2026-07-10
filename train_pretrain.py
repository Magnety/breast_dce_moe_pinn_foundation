from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain Breast-DCE-MoE-PINN foundation model.")
    parser.add_argument("--config", default=CONFIG_DIR / "pretrain_dce_moe_pinn.yaml")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.solver import BreastDCEMoEPINNSolver
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.seed import set_seed

    set_seed(int(config.get("seed", 2026)))
    BreastDCEMoEPINNSolver(config, mode="pretrain").run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
