from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters import build_default_adapters
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import AdapterResult, DatasetAdapter, LabelRecord, SeriesRecord
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings

CASE_FIELDS = [
    "dataset_id",
    "subject_id",
    "study_count",
    "series_count",
    "mask_count",
    "available_roles",
    "has_label",
    "source_root",
    "notes",
]

SERIES_FIELDS = [
    "dataset_id",
    "subject_id",
    "study_uid",
    "study_date",
    "series_uid",
    "series_description",
    "modality",
    "series_role",
    "timepoint",
    "image_count",
    "source_path",
    "relative_path",
    "sop_class_name",
    "manufacturer",
    "local_exists",
    "notes",
]

MASK_FIELDS = [
    "dataset_id",
    "subject_id",
    "study_uid",
    "study_date",
    "series_uid",
    "mask_type",
    "source_path",
    "relative_path",
    "image_count",
    "notes",
]

LABEL_FIELDS = ["dataset_id", "subject_id", "source_file", "key_column", "labels", "notes"]

TASK_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "primary_dce_series_uid",
    "primary_dce_path",
    "t2_series_uid",
    "dwi_series_uid",
    "adc_series_uid",
    "mask_series_uids",
    "roles_available",
    "label_available",
    "pcr_status",
    "task_status",
    "reason",
]


def build_realdata_preprocessing_plan(
    output_dir: str | Path,
    max_patients: int | None = None,
    adapters: list[DatasetAdapter] | None = None,
) -> dict[str, Any]:
    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters = adapters or build_default_adapters()
    results: list[AdapterResult] = []
    for adapter in progress_iter(adapters, total=len(adapters), desc="Scan datasets", unit="dataset"):
        results.append(adapter.scan(max_patients=max_patients))

    cases = [case.to_dict() for result in results for case in result.cases]
    series = [record.to_dict() for result in results for record in result.series]
    masks = [record.to_dict() for result in results for record in result.masks]
    labels = [record.to_dict() for result in results for record in result.labels]
    issues = [dict(issue, dataset_id=result.dataset_id) for result in results for issue in result.issues]
    tasks = build_tasks(
        [record for result in results for record in result.series],
        [record for result in results for record in result.labels],
    )

    atomic_write_csv(out_dir / "cases.csv", cases, CASE_FIELDS)
    atomic_write_csv(out_dir / "series.csv", series, SERIES_FIELDS)
    atomic_write_csv(out_dir / "masks.csv", masks, MASK_FIELDS)
    atomic_write_csv(out_dir / "labels.csv", labels, LABEL_FIELDS)
    atomic_write_csv(out_dir / "issues.csv", issues, ["dataset_id", "severity", "category", "message", "status"])
    atomic_write_csv(out_dir / "preprocessing_tasks.csv", tasks, TASK_FIELDS)

    summary = {
        "output_dir": str(out_dir),
        "dataset_count": len(results),
        "case_count": len(cases),
        "series_count": len(series),
        "mask_count": len(masks),
        "label_count": len(labels),
        "task_count": len(tasks),
        "case_count_by_dataset": dict(Counter(row["dataset_id"] for row in cases)),
        "series_roles": dict(Counter(row["series_role"] for row in series)),
        "task_status_counts": dict(Counter(row["task_status"] for row in tasks)),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def build_tasks(series: list[SeriesRecord], labels: list[LabelRecord]) -> list[dict[str, Any]]:
    label_map = {(label.dataset_id, label.subject_id): label for label in labels}
    grouped: dict[tuple[str, str, str, str], list[SeriesRecord]] = defaultdict(list)
    for record in series:
        grouped[(record.dataset_id, record.subject_id, record.timepoint, record.study_uid)].append(record)

    tasks: list[dict[str, Any]] = []
    grouped_items = sorted(grouped.items())
    for (dataset_id, subject_id, timepoint, study_uid), records in progress_iter(
        grouped_items,
        total=len(grouped_items),
        desc="Build tasks",
        unit="study",
    ):
        by_role: dict[str, list[SeriesRecord]] = defaultdict(list)
        for record in records:
            by_role[record.series_role].append(record)
        dce = _pick_first(by_role, ["DCE", "DCE_PRE", "DCE_POST", "DCE_PE", "DCE_SER"])
        masks = [record for record in records if record.series_role in {"mask", "mask_seg"}]
        label = label_map.get((dataset_id, subject_id))
        pcr_status = extract_pcr_status(label)
        if dce is None:
            status = "blocked_no_dce"
            reason = "No DCE-like series found for this subject/timepoint."
        elif not dce.local_exists:
            status = "blocked_missing_source"
            reason = "Primary DCE source path does not exist locally."
        else:
            status = "ready_for_conversion"
            reason = ""
        tasks.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "timepoint": timepoint,
                "study_uid": study_uid,
                "primary_dce_series_uid": dce.series_uid if dce else "",
                "primary_dce_path": dce.source_path if dce else "",
                "t2_series_uid": _uid(_pick_first(by_role, ["T2"])),
                "dwi_series_uid": _uid(_pick_first(by_role, ["DWI"])),
                "adc_series_uid": _uid(_pick_first(by_role, ["ADC"])),
                "mask_series_uids": [record.series_uid for record in masks],
                "roles_available": sorted(by_role.keys()),
                "label_available": label is not None,
                "pcr_status": pcr_status,
                "task_status": status,
                "reason": reason,
            }
        )
    return tasks


def extract_pcr_status(label: LabelRecord | None) -> str:
    if label is None:
        return "unknown"
    for key in ("是否PCR", "pCR", "PCR", "Pathologic Response to Neoadjuvant Therapy"):
        if key in label.labels and str(label.labels[key]).strip() != "":
            value = str(label.labels[key]).strip().lower()
            if value in {"是", "1", "1.0", "true", "yes", "pcr", "pathologic complete response"}:
                return "1"
            if value in {"否", "0", "0.0", "false", "no", "non-pcr", "non pcr"}:
                return "0"
            return str(label.labels[key]).strip()
    return "unknown"


def _pick_first(by_role: dict[str, list[SeriesRecord]], roles: list[str]) -> SeriesRecord | None:
    candidates: list[SeriesRecord] = []
    for role in roles:
        candidates.extend(by_role.get(role, []))
    if not candidates:
        return None
    return sorted(candidates, key=lambda record: (-record.image_count, record.series_description))[0]


def _uid(record: SeriesRecord | None) -> str:
    return record.series_uid if record else ""


def add_preprocess_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.dce_pattern_analysis import add_dce_pattern_analysis_parser
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.label_optimization import add_label_optimization_parser
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.multimodal_dataset import add_multimodal_parser
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.sequence_analysis import add_sequence_analysis_parser
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.unified_dataset import add_unified_parser

    parser = subparsers.add_parser("preprocess", help="Build real-data preprocessing plans.")
    preprocess_subparsers = parser.add_subparsers(dest="preprocess_command", required=True)
    plan = preprocess_subparsers.add_parser("plan-realdata", help="Analyze real datasets and write preprocessing plan tables.")
    plan.add_argument("--output-dir", default="outputs/local/preprocess_realdata_plan")
    plan.add_argument("--max-patients", type=int)
    plan.set_defaults(func=run_plan_from_args)

    add_sequence_analysis_parser(preprocess_subparsers)
    add_dce_pattern_analysis_parser(preprocess_subparsers)

    convert = preprocess_subparsers.add_parser(
        "convert-nifti",
        help="Convert ready primary DCE tasks from a real-data plan to NIfTI.",
    )
    convert.add_argument("--tasks-csv", default="outputs/local/preprocess_realdata_plan/preprocessing_tasks.csv")
    convert.add_argument("--output-root", default="outputs/local/nifti")
    convert.add_argument("--limit", type=int)
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(func=run_convert_from_args)

    add_unified_parser(preprocess_subparsers)
    add_multimodal_parser(preprocess_subparsers)
    add_label_optimization_parser(preprocess_subparsers)
    return parser


def run_plan_from_args(args: argparse.Namespace) -> int:
    summary = build_realdata_preprocessing_plan(args.output_dir, max_patients=args.max_patients)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_convert_from_args(args: argparse.Namespace) -> int:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.convert import convert_tasks_to_nifti

    summary = convert_tasks_to_nifti(
        tasks_csv=args.tasks_csv,
        output_root=args.output_root,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
