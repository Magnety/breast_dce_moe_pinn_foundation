from __future__ import annotations

from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets import build_dataset_adapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.multimodal_dataset import build_multimodal_dataset, resolve_preprocessing_workers
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "full", "all"}:
        return None
    return int(text)


def run_dataset_from_args(dataset_id: str, args: Any) -> dict[str, Any]:
    adapter = build_dataset_adapter(
        dataset_id,
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        label_root=args.label_root,
    )
    plan_dir = args.plan_dir or project_path("outputs", "local", "preprocess_realdata_plan", args.run_id)
    output_dir = args.output_dir or project_path("outputs", "local", "multimodal_dataset", args.run_id)

    summary: dict[str, Any] = {"dataset_id": dataset_id}
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


def print_dataset_summary(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset_id']}")
    plan = summary.get("plan")
    if plan:
        print(f"Plan cases: {plan['case_count']}")
        print(f"Plan series: {plan['series_count']}")
        print(f"Plan labels: {plan['label_count']}")
        print(f"Plan output: {plan['output_dir']}")
    multimodal = summary.get("multimodal")
    if multimodal:
        print(f"Workers: {multimodal.get('workers')}")
        print(f"Write previews: {multimodal.get('write_previews')}")
        print(f"Timepoints: {multimodal['timepoint_rows']}")
        print(f"Training rows: {multimodal['training_rows']}")
        print(f"Image files: {multimodal['image_rows']}")
        print(f"DCE phase files: {multimodal['dce_phase_files']}")
        print(f"Failures: {multimodal['failure_rows']}")
        print(f"Output: {multimodal['output_dir']}")

