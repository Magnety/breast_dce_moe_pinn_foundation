from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.dce_metadata import analyze_series_temporal_metadata
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings


SEQUENCE_FIELDS = [
    "dataset_id",
    "series_role",
    "series_description",
    "modality",
    "sop_class_name",
    "series_count",
    "subject_count",
    "study_count",
    "image_count_total",
    "local_exists_count",
    "example_source_path",
]

DCE_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "series_uid",
    "series_role",
    "series_description",
    "source_path",
    "image_count",
    "acquisition_time_min",
    "acquisition_time_max",
    "contrast_bolus_start_time",
    "temporal_position_count",
    "number_of_temporal_positions",
    "unique_slice_positions_sampled",
    "repeated_slice_positions_sampled",
    "contrast_phase",
    "phase_evidence",
]

DATASET_FIELDS = [
    "dataset_id",
    "subjects",
    "studies",
    "series",
    "masks",
    "label_records",
    "roles",
    "dce_like_series",
    "dce_pre_series",
    "dce_post_series",
    "dce_unknown_phase_series",
    "with_injection_time",
    "notes",
]

LABEL_SOURCE_FIELDS = [
    "dataset_id",
    "source_file",
    "key_column",
    "record_count",
    "subject_count",
    "columns_sample",
    "pcr_candidate_columns",
    "receptor_candidate_columns",
    "response_candidate_columns",
]

MASK_FIELDS = [
    "dataset_id",
    "mask_type",
    "mask_count",
    "subject_count",
    "example_source_path",
    "notes",
]

NEEDS_REVIEW_FIELDS = ["dataset_id", "category", "subject_id", "timepoint", "series_uid", "message"]

DCE_SOURCE_ROLES = {"DCE", "DCE_PRE", "DCE_POST"}
DCE_RELATED_ROLES = {*DCE_SOURCE_ROLES, "DCE_PE", "DCE_SER"}


def analyze_realdata_sequences(
    output_dir: str | Path,
    plan_dir: str | Path | None = None,
    max_patients: int | None = None,
    max_subjects_per_dataset: int | None = 80,
    max_dce_series_per_dataset: int | None = None,
    dce_header_sample_files: int = 48,
) -> dict[str, Any]:
    """Write a read-only empirical sequence/label/mask analysis for all adapters."""

    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _ensure_plan(plan_dir, out_dir, max_patients=max_patients)

    series_rows = _read_csv_rows(plan_path / "series.csv")
    label_rows = _read_csv_rows(plan_path / "labels.csv")
    mask_rows = _read_csv_rows(plan_path / "masks.csv")
    case_rows = _read_csv_rows(plan_path / "cases.csv")
    original_counts = {
        "series_rows": len(series_rows),
        "label_rows": len(label_rows),
        "mask_rows": len(mask_rows),
        "case_rows": len(case_rows),
    }
    selected_subjects = _selected_subjects_by_dataset(series_rows, max_subjects_per_dataset)
    if selected_subjects is not None:
        series_rows = _filter_rows_to_subjects(series_rows, selected_subjects)
        label_rows = _filter_rows_to_subjects(label_rows, selected_subjects)
        mask_rows = _filter_rows_to_subjects(mask_rows, selected_subjects)
        case_rows = _filter_rows_to_subjects(case_rows, selected_subjects)

    sequence_rows = _sequence_summary_rows(series_rows)
    label_source_rows = _label_source_rows(label_rows)
    mask_summary_rows = _mask_summary_rows(mask_rows)
    dce_rows, dce_review_rows = _dce_analysis_rows(
        series_rows,
        max_dce_series_per_dataset=max_dce_series_per_dataset,
        dce_header_sample_files=dce_header_sample_files,
    )
    dataset_rows = _dataset_summary_rows(case_rows, series_rows, label_rows, mask_rows, dce_rows)
    needs_review_rows = [*dce_review_rows, *_general_review_rows(series_rows, label_rows)]

    atomic_write_csv(out_dir / "dataset_summary.csv", dataset_rows, DATASET_FIELDS)
    atomic_write_csv(out_dir / "sequence_summary.csv", sequence_rows, SEQUENCE_FIELDS)
    atomic_write_csv(out_dir / "dce_phase_analysis.csv", dce_rows, DCE_FIELDS)
    atomic_write_csv(out_dir / "label_sources.csv", label_source_rows, LABEL_SOURCE_FIELDS)
    atomic_write_csv(out_dir / "mask_summary.csv", mask_summary_rows, MASK_FIELDS)
    atomic_write_csv(out_dir / "needs_review.csv", needs_review_rows, NEEDS_REVIEW_FIELDS)
    report = _render_markdown(dataset_rows, sequence_rows, dce_rows, label_source_rows, mask_summary_rows, needs_review_rows)
    atomic_write_text(out_dir / "README.md", report)

    summary = {
        "output_dir": str(out_dir),
        "plan_dir": str(plan_path),
        "max_subjects_per_dataset": max_subjects_per_dataset,
        "sampled_subjects_by_dataset": {dataset_id: len(subjects) for dataset_id, subjects in (selected_subjects or {}).items()},
        "original_counts": original_counts,
        "dataset_count": len(dataset_rows),
        "series_rows": len(series_rows),
        "sequence_summary_rows": len(sequence_rows),
        "dce_rows": len(dce_rows),
        "label_source_rows": len(label_source_rows),
        "mask_summary_rows": len(mask_summary_rows),
        "needs_review_rows": len(needs_review_rows),
        "datasets": {row["dataset_id"]: row for row in dataset_rows},
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def _ensure_plan(plan_dir: str | Path | None, out_dir: Path, max_patients: int | None) -> Path:
    if plan_dir is not None:
        plan_path = Path(plan_dir).expanduser().resolve(strict=False)
        if not (plan_path / "series.csv").exists():
            raise FileNotFoundError(f"Missing series.csv under plan directory: {plan_path}")
        return plan_path
    plan_path = out_dir / "realdata_plan"
    build_realdata_preprocessing_plan(plan_path, max_patients=max_patients)
    return plan_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_subjects_by_dataset(
    rows: list[dict[str, str]],
    max_subjects_per_dataset: int | None,
) -> dict[str, set[str]] | None:
    if max_subjects_per_dataset is None:
        return None
    by_dataset: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: (item.get("dataset_id", ""), item.get("subject_id", ""))):
        dataset_id = row.get("dataset_id", "")
        subject_id = row.get("subject_id", "")
        key = (dataset_id, subject_id)
        if not dataset_id or not subject_id or key in seen:
            continue
        seen.add(key)
        if len(by_dataset[dataset_id]) < max_subjects_per_dataset:
            by_dataset[dataset_id].append(subject_id)
    return {dataset_id: set(subjects) for dataset_id, subjects in by_dataset.items()}


def _filter_rows_to_subjects(
    rows: list[dict[str, str]],
    selected_subjects: dict[str, set[str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("subject_id", "") in selected_subjects.get(row.get("dataset_id", ""), set())
    ]


def _sequence_summary_rows(series_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in series_rows:
        key = (
            row.get("dataset_id", ""),
            row.get("series_role", ""),
            row.get("series_description", ""),
            row.get("modality", ""),
            row.get("sop_class_name", ""),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (dataset_id, role, description, modality, sop), rows in sorted(grouped.items()):
        output.append(
            {
                "dataset_id": dataset_id,
                "series_role": role,
                "series_description": description,
                "modality": modality,
                "sop_class_name": sop,
                "series_count": len(rows),
                "subject_count": len({row.get("subject_id", "") for row in rows}),
                "study_count": len({row.get("study_uid", "") for row in rows}),
                "image_count_total": sum(_as_int(row.get("image_count")) or 0 for row in rows),
                "local_exists_count": sum(row.get("local_exists") == "True" for row in rows),
                "example_source_path": rows[0].get("source_path", "") if rows else "",
            }
        )
    return output


def _dce_analysis_rows(
    series_rows: list[dict[str, str]],
    max_dce_series_per_dataset: int | None,
    dce_header_sample_files: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    output: list[dict[str, Any]] = []
    review: list[dict[str, str]] = []
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in series_rows:
        if row.get("series_role") in DCE_RELATED_ROLES and row.get("local_exists") == "True":
            by_dataset[row.get("dataset_id", "")].append(row)
    selected_rows: list[tuple[str, dict[str, str]]] = []
    for dataset_id, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda row: (row.get("subject_id", ""), row.get("timepoint", ""), _path_number(row.get("source_path", ""))))
        if max_dce_series_per_dataset is not None:
            selected = rows[:max_dce_series_per_dataset]
        else:
            selected = rows
        selected_rows.extend((dataset_id, row) for row in selected)

    for dataset_id, row in progress_iter(
        selected_rows,
        total=len(selected_rows),
        desc="Analyze DCE headers",
        unit="series",
    ):
        if row.get("series_role") in DCE_SOURCE_ROLES:
            summary = analyze_series_temporal_metadata(
                row.get("source_path", ""),
                series_uid=row.get("series_uid", ""),
                series_description=row.get("series_description", ""),
                max_files=dce_header_sample_files,
            ).to_dict()
        else:
            summary = {
                "acquisition_time_min": "",
                "acquisition_time_max": "",
                "contrast_bolus_start_time": "",
                "temporal_position_count": 0,
                "number_of_temporal_positions": "",
                "unique_slice_positions_sampled": 0,
                "repeated_slice_positions_sampled": False,
                "contrast_phase": "derived",
                "phase_evidence": "parametric/derived DCE map, not an original phase",
            }
        out = {field: "" for field in DCE_FIELDS}
        out.update({key: row.get(key, "") for key in out})
        out.update(
            {
                "dataset_id": row.get("dataset_id", ""),
                "subject_id": row.get("subject_id", ""),
                "timepoint": row.get("timepoint", ""),
                "study_uid": row.get("study_uid", ""),
                "series_uid": row.get("series_uid", ""),
                "series_role": row.get("series_role", ""),
                "series_description": row.get("series_description", ""),
                "source_path": row.get("source_path", ""),
                "image_count": row.get("image_count", ""),
                **{key: summary.get(key, "") for key in DCE_FIELDS if key in summary},
            }
        )
        output.append(out)
        if row.get("series_role") in DCE_SOURCE_ROLES and out.get("contrast_phase") == "unknown":
            review.append(
                {
                    "dataset_id": dataset_id,
                    "category": "dce_phase_unknown",
                    "subject_id": row.get("subject_id", ""),
                    "timepoint": row.get("timepoint", ""),
                    "series_uid": row.get("series_uid", ""),
                    "message": out.get("phase_evidence", "DCE source phase could not be determined"),
                }
            )
    return output, review


def _label_source_rows(label_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        grouped[(row.get("dataset_id", ""), row.get("source_file", ""), row.get("key_column", ""))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset_id, source_file, key_column), rows in sorted(grouped.items()):
        labels = [_json_object(row.get("labels", "")) for row in rows]
        columns = sorted({key for payload in labels for key in payload.keys()})
        output.append(
            {
                "dataset_id": dataset_id,
                "source_file": source_file,
                "key_column": key_column,
                "record_count": len(rows),
                "subject_count": len({row.get("subject_id", "") for row in rows}),
                "columns_sample": columns[:80],
                "pcr_candidate_columns": _columns_matching(columns, ("pcr", "pathologic response", "是否pcr")),
                "receptor_candidate_columns": _columns_matching(columns, ("er", "pr", "her2", "分子分型")),
                "response_candidate_columns": _columns_matching(columns, ("response", "rcb", "miller", "疗效", "新辅后")),
            }
        )
    return output


def _mask_summary_rows(mask_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mask_rows:
        grouped[(row.get("dataset_id", ""), row.get("mask_type", ""))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset_id, mask_type), rows in sorted(grouped.items()):
        output.append(
            {
                "dataset_id": dataset_id,
                "mask_type": mask_type,
                "mask_count": len(rows),
                "subject_count": len({row.get("subject_id", "") for row in rows}),
                "example_source_path": rows[0].get("source_path", "") if rows else "",
                "notes": rows[0].get("notes", "") if rows else "",
            }
        )
    return output


def _dataset_summary_rows(
    case_rows: list[dict[str, str]],
    series_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    mask_rows: list[dict[str, str]],
    dce_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    datasets = sorted({row.get("dataset_id", "") for row in [*case_rows, *series_rows, *label_rows, *mask_rows]})
    output: list[dict[str, Any]] = []
    for dataset_id in datasets:
        dataset_series = [row for row in series_rows if row.get("dataset_id") == dataset_id]
        dataset_dce = [row for row in dce_rows if row.get("dataset_id") == dataset_id]
        roles = Counter(row.get("series_role", "") for row in dataset_series)
        output.append(
            {
                "dataset_id": dataset_id,
                "subjects": len({row.get("subject_id", "") for row in dataset_series}),
                "studies": len({row.get("study_uid", "") for row in dataset_series}),
                "series": len(dataset_series),
                "masks": sum(1 for row in mask_rows if row.get("dataset_id") == dataset_id),
                "label_records": sum(1 for row in label_rows if row.get("dataset_id") == dataset_id),
                "roles": dict(roles),
                "dce_like_series": sum(roles.get(role, 0) for role in DCE_RELATED_ROLES),
                "dce_pre_series": sum(row.get("contrast_phase") == "pre" for row in dataset_dce),
                "dce_post_series": sum(row.get("contrast_phase") == "post" for row in dataset_dce),
                "dce_unknown_phase_series": sum(row.get("contrast_phase") == "unknown" for row in dataset_dce),
                "with_injection_time": sum(bool(row.get("contrast_bolus_start_time")) for row in dataset_dce),
                "notes": _dataset_note(dataset_id, roles),
            }
        )
    return output


def _general_review_rows(series_rows: list[dict[str, str]], label_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in series_rows:
        if row.get("series_role") == "unknown":
            rows.append(
                {
                    "dataset_id": row.get("dataset_id", ""),
                    "category": "unknown_series_role",
                    "subject_id": row.get("subject_id", ""),
                    "timepoint": row.get("timepoint", ""),
                    "series_uid": row.get("series_uid", ""),
                    "message": row.get("series_description", ""),
                }
            )
    by_dataset = Counter(row.get("dataset_id", "") for row in label_rows)
    for dataset_id in sorted({row.get("dataset_id", "") for row in series_rows}):
        if by_dataset.get(dataset_id, 0) == 0:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "category": "label_missing_or_not_available",
                    "subject_id": "",
                    "timepoint": "",
                    "series_uid": "",
                    "message": "No label table is configured or matched for this dataset.",
                }
            )
    return rows


def _render_markdown(
    dataset_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    dce_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    review_rows: list[dict[str, str]],
) -> str:
    lines = ["# Real Data Sequence Analysis", ""]
    lines.append("This report is read-only. Archive files are treated as already extracted and are not processed.")
    lines.extend(["", "## Dataset Summary", ""])
    for row in dataset_rows:
        lines.append(
            f"- `{row['dataset_id']}`: subjects `{row['subjects']}`, studies `{row['studies']}`, "
            f"series `{row['series']}`, labels `{row['label_records']}`, masks `{row['masks']}`, "
            f"DCE-like `{row['dce_like_series']}`; pre/post/unknown sampled "
            f"`{row['dce_pre_series']}/{row['dce_post_series']}/{row['dce_unknown_phase_series']}`."
        )
    lines.extend(["", "## Dominant Sequences By Dataset", ""])
    for dataset_id in sorted({row["dataset_id"] for row in sequence_rows}):
        lines.append(f"### `{dataset_id}`")
        rows = [row for row in sequence_rows if row["dataset_id"] == dataset_id]
        rows = sorted(rows, key=lambda row: int(row["series_count"]), reverse=True)[:12]
        for row in rows:
            desc = row["series_description"] or "<blank>"
            lines.append(f"- `{row['series_role']}` `{desc}`: `{row['series_count']}` series")
        lines.append("")
    lines.extend(["## DCE Phase Evidence", ""])
    for dataset_id in sorted({row["dataset_id"] for row in dce_rows}):
        rows = [row for row in dce_rows if row["dataset_id"] == dataset_id]
        counts = Counter(row.get("contrast_phase", "") for row in rows)
        injected = sum(bool(row.get("contrast_bolus_start_time")) for row in rows)
        lines.append(f"- `{dataset_id}`: sampled `{len(rows)}` DCE-related series, phase counts `{dict(counts)}`, injection-time rows `{injected}`.")
    lines.extend(["", "## Label Sources", ""])
    if label_rows:
        for row in label_rows:
            lines.append(f"- `{row['dataset_id']}` `{Path(row['source_file']).name}` key `{row['key_column']}` records `{row['record_count']}`")
    else:
        lines.append("- No configured label sources were matched.")
    lines.extend(["", "## Mask Sources", ""])
    if mask_rows:
        for row in mask_rows:
            lines.append(f"- `{row['dataset_id']}` `{row['mask_type']}`: `{row['mask_count']}`")
    else:
        lines.append("- No mask metadata was found.")
    lines.extend(["", "## Needs Review", ""])
    if review_rows:
        for row in review_rows[:80]:
            lines.append(f"- `{row['dataset_id']}` `{row['category']}` {row.get('subject_id','')} {row.get('message','')}")
        if len(review_rows) > 80:
            lines.append(f"- ... {len(review_rows) - 80} more rows in `needs_review.csv`")
    else:
        lines.append("- No review rows generated.")
    lines.append("")
    return "\n".join(lines)


def _columns_matching(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    output = []
    for column in columns:
        lowered = column.lower()
        if any(token.lower() in lowered for token in tokens):
            output.append(column)
    return output[:40]


def _json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _dataset_note(dataset_id: str, roles: Counter[str]) -> str:
    if dataset_id == "acrin_contralateral":
        return "Clinical labels are in the sibling ACRIN-6667 clinical folder and matched by cn -> ACRIN subject id."
    if roles.get("mask_seg", 0):
        return "Contains DICOM SEG mask metadata."
    if roles.get("DCE", 0) or roles.get("DCE_PRE", 0) or roles.get("DCE_POST", 0):
        return "Contains DCE source series; verify phase evidence before kinetic analysis."
    return ""


def _path_number(value: str) -> float:
    name = Path(value).name
    import re

    match = re.match(r"(\d+(?:\.\d+)?)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 1e12
    return 1e12


def _as_int(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def add_sequence_analysis_parser(preprocess_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = preprocess_subparsers.add_parser(
        "analyze-sequences",
        help="Read-only empirical sequence/label/mask/DCE phase analysis for real datasets.",
    )
    parser.add_argument("--output-dir", default="outputs/local/sequence_analysis/local_audit")
    parser.add_argument("--plan-dir")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--max-subjects-per-dataset", type=int, default=80)
    parser.add_argument("--max-dce-series-per-dataset", type=int, default=None)
    parser.add_argument("--dce-header-sample-files", type=int, default=48)
    parser.set_defaults(func=run_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = analyze_realdata_sequences(
        output_dir=args.output_dir,
        plan_dir=args.plan_dir,
        max_patients=args.max_patients,
        max_subjects_per_dataset=args.max_subjects_per_dataset,
        max_dce_series_per_dataset=args.max_dce_series_per_dataset,
        dce_header_sample_files=args.dce_header_sample_files,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
