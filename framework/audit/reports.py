from __future__ import annotations

from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy


SERIES_FIELDS = [
    "patient_uid",
    "study_uid",
    "series_uid",
    "image_count_sampled",
    "series_description",
    "protocol_name",
    "sequence_name",
    "modality_dicom",
    "manufacturer",
    "magnetic_field_strength",
    "dimensions",
    "pixel_spacing_values",
    "slice_thickness",
    "spacing_between_slices",
    "orientation_count",
    "temporal_position_count",
    "instance_min",
    "instance_max",
    "duplicate_instance_count",
    "missing_instance_count",
    "inferred_modality",
    "modality_status",
    "modality_confidence",
    "modality_evidence",
    "modality_flags",
    "example_relative_paths",
]

PATIENT_FIELDS = [
    "patient_uid",
    "study_count",
    "series_count",
    "image_count_sampled",
    "modalities",
    "needs_review_series_count",
]

ISSUE_FIELDS = [
    "severity",
    "category",
    "message",
    "patient_uid",
    "study_uid",
    "series_uid",
    "relative_path",
    "status",
]

SKIPPED_ARCHIVE_FIELDS = ["relative_path", "size_bytes", "reason"]

LABEL_FIELD_FIELDS = [
    "label_file",
    "sheet",
    "name",
    "status",
    "missing_count",
    "missing_fraction",
    "unique_values_sampled",
    "top_values",
]


def write_dataset_reports(result: dict[str, Any], output_dir: Path, policy: PathPolicy) -> None:
    policy.ensure_output_dir(output_dir)
    atomic_write_json(output_dir / "audit.json", result, policy=policy)
    atomic_write_text(output_dir / "report.md", render_dataset_markdown(result), policy=policy)
    atomic_write_csv(
        output_dir / "series_statistics.csv",
        result["dicom"]["series_statistics"],
        SERIES_FIELDS,
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "patient_statistics.csv",
        result["dicom"]["patient_statistics"],
        PATIENT_FIELDS,
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "modality_inference.csv",
        result["dicom"]["modality_inference"],
        ["patient_uid", "study_uid", "series_uid", "inferred_modality", "status", "confidence", "evidence", "flags"],
        policy=policy,
    )
    atomic_write_csv(output_dir / "quality_issues.csv", result["quality_issues"], ISSUE_FIELDS, policy=policy)
    atomic_write_csv(
        output_dir / "unrecognized_items.csv",
        result["unrecognized_items"],
        ["category", "relative_path", "series_uid", "status", "message"],
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "skipped_archives.csv",
        result.get("skipped_archives", []),
        SKIPPED_ARCHIVE_FIELDS,
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "file_type_counts.csv",
        [{"file_type": key, "count": value} for key, value in sorted(result["inventory"]["type_counts"].items())],
        ["file_type", "count"],
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "extension_counts.csv",
        [
            {"extension": key, "count": value}
            for key, value in sorted(result["inventory"]["extension_counts"].items())
        ],
        ["extension", "count"],
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "directory_tree.csv",
        result["directory_tree"],
        ["relative_path", "depth", "child_count", "children_sample", "truncated"],
        policy=policy,
    )
    atomic_write_csv(
        output_dir / "label_fields.csv",
        flatten_label_fields(result["labels"]),
        LABEL_FIELD_FIELDS,
        policy=policy,
    )
    if "patient_mapping_secure" in result:
        atomic_write_csv(
            output_dir / "patient_mapping_secure.csv",
            result["patient_mapping_secure"],
            ["patient_uid", "raw_patient_id"],
            policy=policy,
        )


def write_run_index(run_result: dict[str, Any], run_dir: Path, policy: PathPolicy) -> None:
    atomic_write_json(run_dir / "audit_run.json", run_result, policy=policy)
    atomic_write_text(run_dir / "README.md", render_run_markdown(run_result), policy=policy)
    atomic_write_csv(
        run_dir / "datasets.csv",
        run_result["datasets"],
        ["dataset_id", "name", "status", "report_dir", "file_count", "dicom_header_count", "quality_issue_count"],
        policy=policy,
    )


def render_run_markdown(run_result: dict[str, Any]) -> str:
    lines = [
        f"# Audit Run `{run_result['run_id']}`",
        "",
        f"- Created at: `{run_result['created_at']}`",
        f"- Dataset count: `{run_result['dataset_count']}`",
        f"- Output directory: `{run_result['output_dir']}`",
        "",
        "## Raw Roots",
        "",
    ]
    lines.extend(f"- `{root}`" for root in run_result["raw_roots"])
    lines.extend(["", "## Datasets", ""])
    for dataset in run_result["datasets"]:
        lines.append(
            "- `{dataset_id}`: status `{status}`, files `{file_count}`, DICOM headers `{dicom_header_count}`, issues `{quality_issue_count}`".format(
                **dataset
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_dataset_markdown(result: dict[str, Any]) -> str:
    inventory = result["inventory"]
    dicom = result["dicom"]
    lines = [
        f"# Dataset Audit: `{result['dataset_id']}`",
        "",
        f"- Name: `{result['name']}`",
        f"- Status: `{result['status']}`",
        f"- Path: `{result['path']}`",
        f"- Files: `{inventory['file_count']}`",
        f"- Directories: `{inventory['directory_count']}`",
        f"- Total size: `{_format_bytes(inventory['total_bytes'])}`",
        f"- Skipped archives: `{inventory.get('skipped_archive_count', 0)}`",
        f"- DICOM headers read: `{dicom['headers_read']}`",
        f"- Series sampled: `{dicom['series_count']}`",
        f"- Patients sampled: `{dicom['patient_count']}`",
        f"- Quality issues: `{len(result['quality_issues'])}`",
        "",
        "## File Types",
        "",
    ]
    for key, value in sorted(inventory["type_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Modality Summary", ""])
    modality_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in dicom["series_statistics"]:
        modality_counts[row["inferred_modality"]] = modality_counts.get(row["inferred_modality"], 0) + 1
        status_counts[row["modality_status"]] = status_counts.get(row["modality_status"], 0) + 1
    if modality_counts:
        for key, value in sorted(modality_counts.items()):
            lines.append(f"- `{key}`: `{value}` series")
    else:
        lines.append("- No readable DICOM series were sampled.")
    lines.extend(["", "## Review Status", ""])
    for key, value in sorted(status_counts.items()):
        lines.append(f"- `{key}`: `{value}` series")
    if not status_counts:
        lines.append("- No modality status available.")
    lines.extend(["", "## Label Files", ""])
    if result["labels"]:
        for label in result["labels"]:
            lines.append(f"- `{label.get('relative_path', label.get('path'))}`: `{label.get('status')}`")
    else:
        lines.append("- No label-like files were found in the scanned subset.")
    lines.extend(["", "## Skipped Archives", ""])
    skipped_archives = result.get("skipped_archives", [])
    if skipped_archives:
        for archive in skipped_archives[:50]:
            lines.append(f"- `{archive['relative_path']}`: `{archive['reason']}`")
        if len(skipped_archives) > 50:
            lines.append(f"- ... {len(skipped_archives) - 50} more archives in `skipped_archives.csv`")
    else:
        lines.append("- No archive files were skipped in the scanned subset.")
    lines.extend(["", "## Preprocessing Recommendations", ""])
    lines.extend(f"- {item}" for item in result["preprocessing_recommendations"])
    lines.extend(["", "## Open Quality Issues", ""])
    for issue in result["quality_issues"][:50]:
        lines.append(f"- `{issue['severity']}` `{issue['category']}`: {issue['message']}")
    if len(result["quality_issues"]) > 50:
        lines.append(f"- ... {len(result['quality_issues']) - 50} more issues in `quality_issues.csv`")
    lines.append("")
    return "\n".join(lines)


def flatten_label_fields(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_file = label.get("relative_path", label.get("path", ""))
        if label.get("format") == "excel":
            for sheet in label.get("sheets", []):
                for field in sheet.get("fields", []):
                    rows.append({"label_file": label_file, "sheet": sheet.get("sheet", ""), **field})
        else:
            for field in label.get("fields", []):
                rows.append({"label_file": label_file, "sheet": "", **field})
    return rows


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"
