from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.dataset_preprocess import optional_int, print_dataset_summary, run_dataset_from_args


DATASET_ID = "advanced_mri_breast_lesions"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the Advanced-MRI-Breast-Lesions dataset.")
    parser.add_argument("--mode", choices=("plan", "multimodal", "all"), default="all")
    parser.add_argument("--run-id", default="full_v4_advanced_mri_breast_lesions")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--plan-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-patients", type=optional_int, default=None)
    parser.add_argument("--limit", type=optional_int, default=None)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(160, 160, 96), metavar=("X", "Y", "Z"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", default=False)
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--write-previews", dest="write_previews", action="store_true")
    preview_group.add_argument("--no-previews", dest="write_previews", action="store_false")
    parser.set_defaults(write_previews=False)
    args = parser.parse_args(argv)

    print_dataset_summary(run_dataset_from_args(DATASET_ID, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

