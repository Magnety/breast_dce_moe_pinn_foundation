from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.dataset_preprocess import (  # noqa: E402
    optional_int,
    print_dataset_summary,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.ispy1 import DATASET_ID, build_adapter  # noqa: E402
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.multimodal_dataset import (  # noqa: E402
    build_multimodal_dataset,
    resolve_preprocessing_workers,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the I-SPY 1 breast MRI dataset.")
    parser.add_argument("--mode", choices=("plan", "multimodal", "all"), default="all")
    parser.add_argument("--run-id", default="full_v4_ispy1")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--plan-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-patients", type=optional_int, default=None)
    parser.add_argument("--limit", type=optional_int, default=None)
    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default=(160, 160, 96),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", default=False)
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--write-previews", dest="write_previews", action="store_true")
    preview_group.add_argument("--no-previews", dest="write_previews", action="store_false")
    parser.set_defaults(write_previews=False)
    args = parser.parse_args(argv)

    print_dataset_summary(run_ispy1_from_args(args))
    return 0


def run_ispy1_from_args(args: Any) -> dict[str, Any]:
    adapter = build_adapter(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        label_root=args.label_root,
    )
    plan_dir = (
        args.plan_dir
        or PROJECT_ROOT / "outputs" / "local" / "preprocess_realdata_plan" / args.run_id
    )
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "outputs" / "local" / "multimodal_dataset" / args.run_id
    )

    summary: dict[str, Any] = {"dataset_id": DATASET_ID}
    if args.mode in {"plan", "all"}:
        summary["plan"] = build_realdata_preprocessing_plan(
            plan_dir,
            max_patients=args.max_patients,
            adapters=[adapter],
        )

    if args.mode in {"multimodal", "all"}:
        required = ("cases.csv", "series.csv", "masks.csv", "labels.csv", "preprocessing_tasks.csv")
        if not all((plan_dir / name).exists() for name in required):
            summary["plan"] = build_realdata_preprocessing_plan(
                plan_dir,
                max_patients=args.max_patients,
                adapters=[adapter],
            )
        summary["multimodal"] = build_multimodal_dataset(
            output_dir=output_dir,
            plan_dir=plan_dir,
            max_patients=args.max_patients,
            limit=args.limit,
            target_shape=tuple(args.target_shape),
            overwrite=args.overwrite,
            workers=resolve_preprocessing_workers(args.workers),
            write_previews=args.write_previews,
        )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
