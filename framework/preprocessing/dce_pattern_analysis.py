from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.classification import classify_series_role
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.convert import DicomTemporalGrouping, dicom_temporal_grouping
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings


DCE_ORIGINAL_ROLES = {"DCE", "DCE_PRE", "DCE_POST"}
DCE_MAP_ROLES = {"DCE_PE", "DCE_SER"}
DCE_RELATED_ROLES = {*DCE_ORIGINAL_ROLES, *DCE_MAP_ROLES, "derived"}

DCE_SEQUENCE_PATTERN_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "series_uid",
    "source_series_role",
    "analysis_series_role",
    "reclassified_series_role",
    "series_description",
    "modality",
    "sop_class_name",
    "manufacturer",
    "source_path",
    "image_count",
    "local_exists",
    "reference_slice_count",
    "temporal_method",
    "temporal_evidence",
    "group_count",
    "group_sizes",
    "total_files",
    "unique_slice_positions",
    "repeated_slice_positions",
    "organization_type",
    "needs_review",
    "review_reason",
]

DCE_STUDY_PATTERN_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "organization_type",
    "original_series_count",
    "derived_map_count",
    "multiphase_original_series_count",
    "single_volume_original_series_count",
    "expected_dce_phase_count",
    "grouping_methods",
    "series_descriptions",
    "needs_review",
    "review_reason",
    "example_source_path",
]

DCE_PATTERN_SUMMARY_FIELDS = [
    "dataset_id",
    "organization_type",
    "study_count",
    "series_count",
    "original_series_count",
    "derived_map_count",
    "example_subject_id",
    "example_study_uid",
]


def analyze_dce_sequence_patterns(
    output_dir: str | Path,
    plan_dir: str | Path | None = None,
    max_patients: int | None = None,
    max_subjects_per_dataset: int | None = 80,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Write a full read-only DCE organization analysis from the real-data plan."""

    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _ensure_plan(plan_dir, out_dir, max_patients=max_patients)
    series_rows = _read_csv_rows(plan_path / "series.csv")
    annotated_rows = [_annotate_analysis_role(row) for row in series_rows]
    selected_subjects = _selected_subjects_by_dataset(annotated_rows, max_subjects_per_dataset)
    if selected_subjects is not None:
        annotated_rows = _filter_rows_to_subjects(annotated_rows, selected_subjects)

    study_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in annotated_rows:
        study_groups[_study_key(row)].append(row)

    selected: list[tuple[dict[str, str], int]] = []
    for rows in study_groups.values():
        reference_counts = _reference_slice_counts(rows)
        for row in rows:
            if not _is_dce_related(row):
                continue
            key = row.get("series_uid") or row.get("source_path", "")
            selected.append((row, reference_counts.get(key, 0)))
    selected = sorted(selected, key=lambda item: _row_sort_key(item[0]))
    if limit is not None:
        selected = selected[:limit]

    sequence_rows = _analyze_sequence_rows(selected, workers=max(workers, 1))
    sequence_rows = sorted(sequence_rows, key=_row_sort_key)

    sequence_by_study: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sequence_rows:
        sequence_by_study[_study_key(row)].append(row)
    study_rows = [_study_pattern_row(rows) for _, rows in sorted(sequence_by_study.items())]
    summary_rows = _summary_rows(study_rows, sequence_rows)

    atomic_write_csv(out_dir / "dce_sequence_patterns.csv", sequence_rows, DCE_SEQUENCE_PATTERN_FIELDS)
    atomic_write_csv(out_dir / "dce_study_patterns.csv", study_rows, DCE_STUDY_PATTERN_FIELDS)
    atomic_write_csv(out_dir / "dce_pattern_summary.csv", summary_rows, DCE_PATTERN_SUMMARY_FIELDS)
    atomic_write_text(out_dir / "README.md", _render_markdown(summary_rows, study_rows))

    summary = {
        "output_dir": str(out_dir),
        "plan_dir": str(plan_path),
        "max_subjects_per_dataset": max_subjects_per_dataset,
        "sampled_subjects_by_dataset": {dataset_id: len(subjects) for dataset_id, subjects in (selected_subjects or {}).items()},
        "series_rows_total": len(series_rows),
        "series_rows": len(annotated_rows),
        "dce_sequence_rows": len(sequence_rows),
        "dce_study_rows": len(study_rows),
        "needs_review_studies": sum(_truthy(row.get("needs_review")) for row in study_rows),
        "patterns": dict(Counter(row.get("organization_type", "") for row in study_rows)),
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


def _analyze_sequence_rows(selected: list[tuple[dict[str, str], int]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(selected) <= 1:
        return [
            _sequence_pattern_row(row, reference_slice_count)
            for row, reference_slice_count in progress_iter(
                selected,
                total=len(selected),
                desc="Analyze DCE patterns",
                unit="series",
            )
        ]

    output: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_sequence_pattern_worker, row, reference_slice_count)
            for row, reference_slice_count in selected
        ]
        for future in progress_iter(
            as_completed(futures),
            total=len(futures),
            desc=f"Analyze DCE patterns x{workers}",
            unit="series",
        ):
            output.append(future.result())
    return output


def _sequence_pattern_worker(row: dict[str, str], reference_slice_count: int) -> dict[str, Any]:
    suppress_warnings()
    return _sequence_pattern_row(row, reference_slice_count)


def _annotate_analysis_role(row: dict[str, str]) -> dict[str, str]:
    output = dict(row)
    reclassified = classify_series_role(
        output.get("series_description", ""),
        modality=output.get("modality", ""),
        sop_class_name=output.get("sop_class_name", ""),
        source_path=output.get("source_path", "") or output.get("relative_path", ""),
        dataset_id=output.get("dataset_id", ""),
    )
    source_role = output.get("series_role", "") or reclassified
    analysis_role = source_role
    if reclassified in DCE_RELATED_ROLES:
        analysis_role = reclassified
    if source_role in DCE_ORIGINAL_ROLES and _derived_dce_role(output) != "":
        analysis_role = _derived_dce_role(output)
    output["source_series_role"] = source_role
    output["reclassified_series_role"] = reclassified
    output["analysis_series_role"] = analysis_role
    return output


def _is_dce_related(row: dict[str, str]) -> bool:
    role = row.get("analysis_series_role", "")
    if role in {*DCE_ORIGINAL_ROLES, *DCE_MAP_ROLES}:
        return True
    return role == "derived" and _looks_like_dce_derived(row)


def _is_dce_reference_map(row: dict[str, str]) -> bool:
    role = row.get("analysis_series_role", "")
    return role in DCE_MAP_ROLES or (role == "derived" and _looks_like_dce_derived(row))


def _sequence_pattern_row(row: dict[str, str], reference_slice_count: int) -> dict[str, Any]:
    role = row.get("analysis_series_role", "")
    output = {field: "" for field in DCE_SEQUENCE_PATTERN_FIELDS}
    output.update({key: row.get(key, "") for key in output})
    output.update(
        {
            "source_series_role": row.get("source_series_role", row.get("series_role", "")),
            "analysis_series_role": role,
            "reclassified_series_role": row.get("reclassified_series_role", ""),
            "reference_slice_count": reference_slice_count or "",
        }
    )

    if not _truthy(row.get("local_exists")):
        output.update(
            {
                "organization_type": "missing_local_source",
                "needs_review": True,
                "review_reason": "series path is not available locally",
            }
        )
        return output

    if role not in DCE_ORIGINAL_ROLES:
        output.update(
            {
                "organization_type": "derived_dce_map",
                "temporal_method": "not_applicable",
                "temporal_evidence": "parametric/subtraction/map series is not an original DCE phase source",
                "group_count": 0,
            }
        )
        return output

    try:
        grouping = dicom_temporal_grouping(
            Path(row.get("source_path", "")),
            row.get("series_uid", ""),
            reference_slice_count=reference_slice_count or None,
        )
    except Exception as exc:
        output.update(
            {
                "organization_type": "read_failed",
                "temporal_method": "read_failed",
                "temporal_evidence": str(exc),
                "group_count": 1,
                "needs_review": True,
                "review_reason": "DICOM headers could not be read for temporal grouping",
            }
        )
        return output

    output.update(_grouping_columns(grouping))
    output.update(_series_organization_columns(row, grouping))
    return output


def _grouping_columns(grouping: DicomTemporalGrouping) -> dict[str, Any]:
    return {
        "temporal_method": grouping.method,
        "temporal_evidence": grouping.evidence,
        "group_count": grouping.group_count,
        "group_sizes": json.dumps(grouping.group_sizes, ensure_ascii=False),
        "total_files": grouping.total_files,
        "unique_slice_positions": grouping.unique_slice_positions,
        "repeated_slice_positions": grouping.repeated_slice_positions,
        "reference_slice_count": grouping.reference_slice_count or "",
    }


def _series_organization_columns(row: dict[str, str], grouping: DicomTemporalGrouping) -> dict[str, Any]:
    if grouping.group_count > 1:
        if grouping.method == "reference_slice_blocks":
            organization = "single_series_multiphase_by_reference_blocks"
            review_reason = ""
        elif grouping.method == "repeated_slice_positions":
            organization = "single_series_multiphase_by_repeated_slices"
            review_reason = ""
        elif grouping.method in {"temporal_position", "acquisition_number", "acquisition_datetime", "acquisition", "content_time", "trigger_time"}:
            organization = "single_series_multiphase_by_header"
            review_reason = ""
        else:
            organization = "single_series_multiphase"
            review_reason = ""
        phase_count = grouping.group_count
        if phase_count > 8:
            return {
                "organization_type": organization,
                "needs_review": True,
                "review_reason": f"large temporal group count detected: {phase_count}",
            }
        return {"organization_type": organization, "needs_review": False, "review_reason": review_reason}

    if _as_int(row.get("image_count")) and grouping.reference_slice_count:
        image_count = _as_int(row.get("image_count")) or 0
        if image_count > grouping.reference_slice_count and image_count % grouping.reference_slice_count != 0:
            return {
                "organization_type": "single_volume_or_ambiguous",
                "needs_review": True,
                "review_reason": "image count is larger than reference slice count but not divisible by it",
            }
    return {
        "organization_type": "single_volume_or_ambiguous",
        "needs_review": True,
        "review_reason": "only one temporal volume was detected for this original DCE series",
    }


def _study_pattern_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    original_rows = [row for row in rows if row.get("analysis_series_role") in DCE_ORIGINAL_ROLES]
    derived_rows = [row for row in rows if row.get("analysis_series_role") not in DCE_ORIGINAL_ROLES]
    multiphase_rows = [_as_int(row.get("group_count")) or 0 for row in original_rows if (_as_int(row.get("group_count")) or 0) > 1]
    single_rows = [row for row in original_rows if (_as_int(row.get("group_count")) or 0) <= 1]
    expected_phase_count = sum(max(_as_int(row.get("group_count")) or 1, 1) for row in original_rows)

    if not original_rows and derived_rows:
        organization = "derived_only"
        needs_review = True
        reason = "DCE-related derived maps exist, but no original DCE source series was found"
    elif not original_rows:
        organization = "no_original_dce"
        needs_review = True
        reason = "no original DCE source series was found"
    elif multiphase_rows and len(original_rows) == 1:
        organization = "single_series_multiphase"
        needs_review = any(_truthy(row.get("needs_review")) for row in original_rows)
        reason = _join_reasons(original_rows)
    elif multiphase_rows and len(original_rows) > 1:
        organization = "mixed_single_series_and_cross_series"
        needs_review = True
        reason = "both multi-phase original series and separate original DCE series were found in the same study"
    elif len(original_rows) > 1:
        organization = "cross_series_temporal"
        needs_review = any(_truthy(row.get("needs_review")) for row in original_rows)
        reason = _join_reasons(original_rows)
    else:
        organization = "single_volume_or_ambiguous"
        needs_review = True
        reason = _join_reasons(original_rows) or "only one original DCE volume was detected"

    return {
        "dataset_id": first.get("dataset_id", ""),
        "subject_id": first.get("subject_id", ""),
        "timepoint": first.get("timepoint", ""),
        "study_uid": first.get("study_uid", ""),
        "organization_type": organization,
        "original_series_count": len(original_rows),
        "derived_map_count": len(derived_rows),
        "multiphase_original_series_count": len(multiphase_rows),
        "single_volume_original_series_count": len(single_rows),
        "expected_dce_phase_count": expected_phase_count,
        "grouping_methods": json.dumps(sorted({str(row.get("temporal_method", "")) for row in original_rows if row.get("temporal_method")}), ensure_ascii=False),
        "series_descriptions": json.dumps(_unique(row.get("series_description", "") for row in rows), ensure_ascii=False),
        "needs_review": needs_review,
        "review_reason": reason,
        "example_source_path": first.get("source_path", ""),
    }


def _summary_rows(study_rows: list[dict[str, Any]], sequence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    series_by_study: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sequence_rows:
        series_by_study[_study_key(row)].append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in study_rows:
        grouped[(str(row.get("dataset_id", "")), str(row.get("organization_type", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (dataset_id, organization), rows in sorted(grouped.items()):
        study_keys = [_study_key(row) for row in rows]
        series = [series_row for key in study_keys for series_row in series_by_study.get(key, [])]
        original_series = [row for row in series if row.get("analysis_series_role") in DCE_ORIGINAL_ROLES]
        derived_series = [row for row in series if row.get("analysis_series_role") not in DCE_ORIGINAL_ROLES]
        example = rows[0] if rows else {}
        output.append(
            {
                "dataset_id": dataset_id,
                "organization_type": organization,
                "study_count": len(rows),
                "series_count": len(series),
                "original_series_count": len(original_series),
                "derived_map_count": len(derived_series),
                "example_subject_id": example.get("subject_id", ""),
                "example_study_uid": example.get("study_uid", ""),
            }
        )
    return output


def _reference_slice_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(
        _as_int(row.get("image_count"))
        for row in rows
        if _is_dce_reference_map(row) and _truthy(row.get("local_exists")) and (_as_int(row.get("image_count")) or 0) >= 8
    )
    counts.pop(None, None)
    if not counts:
        return {}

    output: dict[str, int] = {}
    for row in rows:
        if row.get("analysis_series_role") not in DCE_ORIGINAL_ROLES:
            continue
        image_count = _as_int(row.get("image_count")) or 0
        candidates = [
            count
            for count in counts
            if count
            and count < image_count
            and image_count % count == 0
            and 1 < image_count // count <= 20
        ]
        if not candidates:
            continue
        selected = sorted(candidates, key=lambda count: (-counts[count], -count))[0]
        output[row.get("series_uid") or row.get("source_path", "")] = selected
    return output


def _derived_dce_role(row: dict[str, str]) -> str:
    text = _series_text(row)
    if re.search(r"\bpe\s*[1-9]\b", text):
        return "DCE_PE"
    if re.search(r"\bser\b", text):
        return "DCE_SER"
    if _looks_like_dce_derived(row):
        return "derived"
    return ""


def _looks_like_dce_derived(row: dict[str, str]) -> bool:
    text = _series_text(row)
    return any(token in text for token in ("sub", "subtract", "subtraction", "mip", "cad", "tram", "reformat", "projection"))


def _series_text(row: dict[str, str]) -> str:
    return " ".join([row.get("series_description", ""), row.get("source_path", "")]).lower()


def _study_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset_id", "")),
        str(row.get("subject_id", "")),
        str(row.get("timepoint", "")),
        str(row.get("study_uid", "")),
    )


def _row_sort_key(row: dict[str, str]) -> tuple[str, str, str, float, str]:
    return (
        row.get("dataset_id", ""),
        row.get("subject_id", ""),
        row.get("timepoint", ""),
        _path_number(row.get("source_path", "")),
        row.get("series_uid", ""),
    )


def _path_number(value: str) -> float:
    name = Path(value).name
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


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _unique(values: Any) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in output:
            output.append(text)
    return output


def _join_reasons(rows: list[dict[str, Any]]) -> str:
    return "; ".join(_unique(row.get("review_reason", "") for row in rows if row.get("review_reason")))


def _render_markdown(summary_rows: list[dict[str, Any]], study_rows: list[dict[str, Any]]) -> str:
    lines = ["# DCE Sequence Pattern Analysis", ""]
    lines.append("This is a read-only DICOM-header analysis. It does not preprocess image arrays.")
    lines.extend(["", "## Pattern Summary", ""])
    if summary_rows:
        for row in summary_rows:
            lines.append(
                f"- `{row['dataset_id']}` `{row['organization_type']}`: "
                f"{row['study_count']} studies, {row['series_count']} series"
            )
    else:
        lines.append("- No DCE-related local series were found.")
    needs_review = [row for row in study_rows if _truthy(row.get("needs_review"))]
    lines.extend(["", "## Review", ""])
    lines.append(f"- Studies needing review: `{len(needs_review)}`")
    for row in needs_review[:60]:
        lines.append(
            f"- `{row['dataset_id']}` `{row['subject_id']}` `{row['organization_type']}`: "
            f"{row.get('review_reason', '')}"
        )
    if len(needs_review) > 60:
        lines.append(f"- ... {len(needs_review) - 60} more rows in `dce_study_patterns.csv`")
    lines.append("")
    return "\n".join(lines)


def add_dce_pattern_analysis_parser(preprocess_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = preprocess_subparsers.add_parser(
        "analyze-dce-patterns",
        help="Full read-only analysis of DCE series organization patterns.",
    )
    parser.add_argument("--output-dir", default="outputs/local/dce_pattern_analysis/local_audit")
    parser.add_argument("--plan-dir")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--max-subjects-per-dataset", type=int, default=80)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(func=run_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = analyze_dce_sequence_patterns(
        output_dir=args.output_dir,
        plan_dir=args.plan_dir,
        max_patients=args.max_patients,
        max_subjects_per_dataset=args.max_subjects_per_dataset,
        limit=args.limit,
        workers=max(args.workers, 1),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
