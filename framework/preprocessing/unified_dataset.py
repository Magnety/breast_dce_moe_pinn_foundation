from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.data.manifest import MANIFEST_FIELDS
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.convert import read_dicom_directory, read_dicom_series
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, silence_simpleitk_warnings, suppress_warnings


PREPROCESSING_VERSION = "realdata_unified_v1"
READY_STATUS = "ready_for_conversion"

FAILURE_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "primary_dce_series_uid",
    "primary_dce_path",
    "stage",
    "error",
]

MASK_MANIFEST_FIELDS = [
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
    "conversion_status",
]

DATASET_QC_FIELDS = [
    "dataset_id",
    "inference_rows",
    "training_rows",
    "unique_subjects",
    "pcr_positive",
    "pcr_negative",
    "pcr_unknown",
    "with_mask_metadata",
    "split_train",
    "split_val",
    "split_test",
    "split_inference",
]

PATIENT_QC_FIELDS = [
    "dataset_id",
    "subject_id",
    "sample_count",
    "timepoints",
    "has_training_label",
    "pcr_status",
    "splits",
]

LABEL_SUMMARY_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "sample_dir",
    "pCR",
    "molecular_subtype",
    "split",
    "labels_json",
]


def build_unified_dataset(
    output_dir: str | Path,
    plan_dir: str | Path | None = None,
    max_patients: int | None = None,
    limit: int | None = None,
    target_shape: tuple[int, int, int] = (160, 160, 96),
    overwrite: bool = False,
    write_nifti: bool = True,
) -> dict[str, Any]:
    """Build a fixed-shape dataset for training/inference from the real-data plan.

    The generated `training_manifest.csv` points `standardized_data_path` at `.npy`
    tensors with shape `(1, z, y, x)`. `inference_manifest.csv` contains every
    successfully converted image, including rows without a pCR label.
    """

    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _ensure_plan(plan_dir, out_dir, max_patients=max_patients)
    tasks_csv = plan_path / "preprocessing_tasks.csv"
    labels_csv = plan_path / "labels.csv"
    masks_csv = plan_path / "masks.csv"

    label_map = _load_label_map(labels_csv)
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempted = 0
    converted = 0
    skipped_existing = 0

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SimpleITK is required to build the unified dataset.") from exc
    silence_simpleitk_warnings(sitk)

    with tasks_csv.open("r", encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
        for task in progress_iter(task_rows, total=len(task_rows), desc="Build unified", unit="task"):
            if task.get("task_status") != READY_STATUS:
                failures.append(_failure(task, "plan", task.get("reason") or task.get("task_status", "")))
                continue
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            source = Path(task.get("primary_dce_path", ""))
            if not path_exists(source):
                failures.append(_failure(task, "source", "primary DCE source path does not exist"))
                continue

            sample_dir = _sample_dir(out_dir, task)
            array_path = sample_dir / "image.npy"
            nifti_path = sample_dir / "DCE_primary.nii.gz"
            metadata_path = sample_dir / "metadata.json"
            if array_path.exists() and metadata_path.exists() and (not write_nifti or nifti_path.exists()) and not overwrite:
                skipped_existing += 1
                manifest_row = _manifest_row(
                    task=task,
                    array_path=array_path,
                    nifti_path=nifti_path if write_nifti else None,
                    label_map=label_map,
                    image_shape="unknown",
                    spacing="unknown",
                    split_seed=2026,
                )
                rows.append(manifest_row)
                if manifest_row["pcr_status"] in {"0", "1"}:
                    training_rows.append(manifest_row)
                continue

            try:
                if task.get("dataset_id") == "fujian_pCR":
                    image = read_dicom_series(sitk, source, task.get("primary_dce_series_uid", ""))
                else:
                    image = read_dicom_directory(sitk, source)
                standardized, metadata = standardize_image(sitk, image, target_shape=target_shape)
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(array_path, standardized)
                if write_nifti:
                    sitk.WriteImage(image, str(nifti_path))
                metadata.update(task)
                metadata["standardized_array_path"] = str(array_path)
                metadata["nifti_path"] = str(nifti_path) if write_nifti else ""
                atomic_write_json(metadata_path, metadata)
            except Exception as exc:
                failures.append(_failure(task, "convert", str(exc)))
                continue

            converted += 1
            manifest_row = _manifest_row(
                task=task,
                array_path=array_path,
                nifti_path=nifti_path if write_nifti else None,
                label_map=label_map,
                image_shape=str(tuple(int(value) for value in standardized.shape)),
                spacing=json.dumps(metadata.get("resampled_spacing", []), ensure_ascii=False),
                split_seed=2026,
            )
            rows.append(manifest_row)
            if manifest_row["pcr_status"] in {"0", "1"}:
                training_rows.append(manifest_row)

    mask_count = _write_mask_manifest(masks_csv, out_dir / "mask_manifest.csv")
    atomic_write_csv(out_dir / "inference_manifest.csv", rows, MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "training_manifest.csv", training_rows, MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "failures.csv", failures, FAILURE_FIELDS)
    finalization = finalize_unified_dataset(out_dir)

    summary = {
        "output_dir": str(out_dir),
        "plan_dir": str(plan_path),
        "target_shape_xyz": list(target_shape),
        "attempted": attempted,
        "converted": converted,
        "skipped_existing": skipped_existing,
        "inference_rows": len(rows),
        "training_rows": len(training_rows),
        "mask_rows": mask_count,
        "failures": len(failures),
        "label_sidecars": finalization["label_sidecars"],
        "patient_qc_rows": finalization["patient_count"],
        "training_rows_by_dataset": dict(Counter(row["dataset_id"] for row in training_rows)),
        "inference_rows_by_dataset": dict(Counter(row["dataset_id"] for row in rows)),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def finalize_unified_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Write labels.json sidecars and QC summaries for an existing unified dataset."""

    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    inference_rows = _read_csv_rows(out_dir / "inference_manifest.csv")
    training_rows = _read_csv_rows(out_dir / "training_manifest.csv")
    mask_rows = _read_csv_rows(out_dir / "mask_manifest.csv")

    training_keys = {
        (row.get("dataset_id", ""), row.get("patient_uid", ""), row.get("timepoint", ""), row.get("study_uid", ""))
        for row in training_rows
    }
    mask_keys = {(row.get("dataset_id", ""), row.get("subject_id", "")) for row in mask_rows}

    label_summary_rows: list[dict[str, Any]] = []
    sidecar_count = 0
    for row in progress_iter(
        inference_rows,
        total=len(inference_rows),
        desc="Finalize unified",
        unit="sample",
    ):
        sample_dir = Path(row["standardized_data_path"]).parent
        labels_payload = _labels_payload(row)
        _write_json_if_changed(sample_dir / "labels.json", labels_payload)
        sidecar_count += 1
        label_summary_rows.append(
            {
                "dataset_id": row.get("dataset_id", ""),
                "subject_id": row.get("patient_uid", ""),
                "timepoint": row.get("timepoint", ""),
                "study_uid": row.get("study_uid", ""),
                "sample_dir": str(sample_dir),
                "pCR": labels_payload.get("pCR", ""),
                "molecular_subtype": labels_payload.get("molecular_subtype", "unknown"),
                "split": row.get("split", ""),
                "labels_json": str(sample_dir / "labels.json"),
            }
        )

    dataset_rows = _dataset_qc_rows(inference_rows, training_keys, mask_keys)
    patient_rows = _patient_qc_rows(inference_rows, training_keys)

    atomic_write_csv(out_dir / "labels_summary.csv", label_summary_rows, LABEL_SUMMARY_FIELDS)
    atomic_write_csv(out_dir / "dataset_qc.csv", dataset_rows, DATASET_QC_FIELDS)
    atomic_write_csv(out_dir / "patient_qc.csv", patient_rows, PATIENT_QC_FIELDS)

    qc_json = {
        "output_dir": str(out_dir),
        "inference_rows": len(inference_rows),
        "training_rows": len(training_rows),
        "label_sidecars": sidecar_count,
        "dataset_qc": dataset_rows,
        "patient_count": len(patient_rows),
    }
    atomic_write_json(out_dir / "dataset_qc.json", qc_json)
    _merge_summary(out_dir / "summary.json", {"label_sidecars": sidecar_count, "patient_qc_rows": len(patient_rows)})
    return qc_json


def standardize_image(
    sitk: Any,
    image: Any,
    target_shape: tuple[int, int, int] = (160, 160, 96),
) -> tuple[np.ndarray, dict[str, Any]]:
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got dimension {image.GetDimension()}")

    image = sitk.Cast(image, sitk.sitkFloat32)
    original_size = tuple(int(value) for value in image.GetSize())
    original_spacing = tuple(float(value) for value in image.GetSpacing())
    target_size = tuple(int(value) for value in target_shape)
    target_spacing = tuple(
        (original_size[index] * original_spacing[index]) / max(target_size[index], 1)
        for index in range(3)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)

    array = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)
    array = _normalize_intensity(array)
    array = array[np.newaxis, ...].astype(np.float32, copy=False)
    metadata = {
        "original_size_xyz": original_size,
        "original_spacing_xyz": original_spacing,
        "target_size_xyz": target_size,
        "resampled_spacing": target_spacing,
        "array_shape_czyx": tuple(int(value) for value in array.shape),
        "normalization": "clip_0.5_99.5_then_zscore_nonzero",
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    return array, metadata


def _normalize_intensity(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    foreground = finite[np.abs(finite) > 1e-6]
    if foreground.size < 32:
        foreground = finite
    low, high = np.percentile(foreground, [0.5, 99.5])
    if high > low:
        array = np.clip(array, low, high, out=array)
    foreground = array[np.isfinite(array) & (np.abs(array) > 1e-6)]
    if foreground.size < 32:
        foreground = array[np.isfinite(array)]
    mean = float(np.mean(foreground)) if foreground.size else 0.0
    std = float(np.std(foreground)) if foreground.size else 1.0
    if std < 1e-6:
        std = 1.0
    return ((array - mean) / std).astype(np.float32, copy=False)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _labels_payload(row: dict[str, str]) -> dict[str, Any]:
    raw_labels = _json_object(row.get("clinical_labels", ""))
    pcr_text = str(row.get("pcr_status", "")).strip()
    pcr_value: int | None
    if pcr_text in {"0", "1"}:
        pcr_value = int(pcr_text)
    else:
        pcr_value = None
    return {
        "dataset_id": row.get("dataset_id", ""),
        "patient_id": row.get("patient_uid", ""),
        "timepoint": row.get("timepoint", ""),
        "study_uid": row.get("study_uid", ""),
        "pCR": pcr_value,
        "pcr_status": pcr_text or "unknown",
        "HER2": _first_nonempty_label(raw_labels, ("HER2", "HER2结果", "HER2 status", "HER2 Status")),
        "molecular_subtype": row.get("molecular_subtype") or _extract_molecular_subtype(row.get("clinical_labels", "")),
        "split": row.get("split", ""),
        "raw_labels": raw_labels,
    }


def _dataset_qc_rows(
    inference_rows: list[dict[str, str]],
    training_keys: set[tuple[str, str, str, str]],
    mask_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in inference_rows:
        grouped.setdefault(row.get("dataset_id", ""), []).append(row)

    output: list[dict[str, Any]] = []
    for dataset_id, rows in sorted(grouped.items()):
        split_counts = Counter(row.get("split", "") for row in rows)
        subjects = {row.get("patient_uid", "") for row in rows}
        training_count = sum(
            (
                row.get("dataset_id", ""),
                row.get("patient_uid", ""),
                row.get("timepoint", ""),
                row.get("study_uid", ""),
            )
            in training_keys
            for row in rows
        )
        output.append(
            {
                "dataset_id": dataset_id,
                "inference_rows": len(rows),
                "training_rows": training_count,
                "unique_subjects": len(subjects),
                "pcr_positive": sum(row.get("pcr_status") == "1" for row in rows),
                "pcr_negative": sum(row.get("pcr_status") == "0" for row in rows),
                "pcr_unknown": sum(row.get("pcr_status") not in {"0", "1"} for row in rows),
                "with_mask_metadata": sum((dataset_id, row.get("patient_uid", "")) in mask_keys for row in rows),
                "split_train": split_counts.get("train", 0),
                "split_val": split_counts.get("val", 0),
                "split_test": split_counts.get("test", 0),
                "split_inference": split_counts.get("inference", 0),
            }
        )
    return output


def _patient_qc_rows(
    inference_rows: list[dict[str, str]],
    training_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in inference_rows:
        grouped.setdefault((row.get("dataset_id", ""), row.get("patient_uid", "")), []).append(row)

    output: list[dict[str, Any]] = []
    for (dataset_id, subject_id), rows in sorted(grouped.items()):
        pcr_values = sorted({row.get("pcr_status", "unknown") for row in rows if row.get("pcr_status") in {"0", "1"}})
        splits = sorted({row.get("split", "") for row in rows if row.get("split", "")})
        has_training_label = any(
            (
                row.get("dataset_id", ""),
                row.get("patient_uid", ""),
                row.get("timepoint", ""),
                row.get("study_uid", ""),
            )
            in training_keys
            for row in rows
        )
        output.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "sample_count": len(rows),
                "timepoints": sorted({row.get("timepoint", "") for row in rows}),
                "has_training_label": has_training_label,
                "pcr_status": pcr_values[0] if len(pcr_values) == 1 else ("mixed" if pcr_values else "unknown"),
                "splits": splits,
            }
        )
    return output


def _json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_nonempty_label(labels: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = labels.get(key)
        if value not in (None, ""):
            return value
    return "unknown"


def _merge_summary(path: Path, updates: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(updates)
    atomic_write_json(path, payload)


def _write_json_if_changed(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_json(path, data)


def _ensure_plan(plan_dir: str | Path | None, out_dir: Path, max_patients: int | None) -> Path:
    if plan_dir is not None:
        plan_path = Path(plan_dir).expanduser().resolve(strict=False)
        if not (plan_path / "preprocessing_tasks.csv").exists():
            raise FileNotFoundError(f"Missing preprocessing_tasks.csv under plan directory: {plan_path}")
        return plan_path
    plan_path = out_dir / "realdata_plan"
    build_realdata_preprocessing_plan(plan_path, max_patients=max_patients)
    return plan_path


def _load_label_map(labels_csv: Path) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    if not labels_csv.exists():
        return labels
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("dataset_id", ""), row.get("subject_id", ""))
            labels.setdefault(key, row.get("labels", ""))
    return labels


def _sample_dir(out_dir: Path, task: dict[str, str]) -> Path:
    sample_key = hashlib.sha256(
        "|".join(
            [
                task.get("study_uid", ""),
                task.get("primary_dce_series_uid", ""),
                task.get("primary_dce_path", ""),
            ]
        ).encode("utf-8", errors="replace")
    ).hexdigest()[:10]
    return (
        out_dir
        / "samples"
        / _safe_name(task.get("dataset_id", "unknown"))
        / _safe_name(task.get("subject_id", "unknown"))
        / _safe_name(task.get("timepoint", "unknown"))
        / sample_key
    )


def _manifest_row(
    task: dict[str, str],
    array_path: Path,
    nifti_path: Path | None,
    label_map: dict[tuple[str, str], str],
    image_shape: str,
    spacing: str,
    split_seed: int,
) -> dict[str, Any]:
    dataset_id = task.get("dataset_id", "")
    subject_id = task.get("subject_id", "")
    pcr_status = task.get("pcr_status", "unknown")
    labels = label_map.get((dataset_id, subject_id), "")
    split = _split_for_subject(dataset_id, subject_id, split_seed) if pcr_status in {"0", "1"} else "inference"
    return {
        "dataset_id": dataset_id,
        "patient_uid": subject_id,
        "study_uid": task.get("study_uid", "unknown"),
        "original_patient_mapping": "not_exported",
        "original_path": task.get("primary_dce_path", ""),
        "standardized_data_path": str(array_path),
        "timepoint": task.get("timepoint", "unknown"),
        "laterality": "unknown",
        "lesion_id": "unknown",
        "available_modalities": task.get("roles_available", ""),
        "dce_phase_count": 1,
        "image_shape": image_shape,
        "spacing": spacing,
        "clinical_labels": labels,
        "pathology_labels": "",
        "molecular_subtype": _extract_molecular_subtype(labels),
        "response_label": pcr_status,
        "pcr_status": pcr_status,
        "survival_followup": "",
        "roi_status": "metadata_only" if task.get("mask_series_uids", "[]") not in {"", "[]"} else "not_available",
        "missing_modalities": "",
        "qc_status": "ready",
        "exclusion_reason": "",
        "preprocessing_version": PREPROCESSING_VERSION,
        "split": split,
        "source_audit_json": "",
    }


def _extract_molecular_subtype(labels_json: str) -> str:
    if not labels_json:
        return "unknown"
    try:
        labels = json.loads(labels_json)
    except json.JSONDecodeError:
        return "unknown"
    for key in ("分子分型", "molecular_subtype", "Molecular subtype", "Molecular Subtype"):
        value = labels.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _split_for_subject(dataset_id: str, subject_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{dataset_id}|{subject_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def _write_mask_manifest(masks_csv: Path, target_csv: Path) -> int:
    if not masks_csv.exists():
        atomic_write_csv(target_csv, [], MASK_MANIFEST_FIELDS)
        return 0
    rows: list[dict[str, Any]] = []
    with masks_csv.open("r", encoding="utf-8", newline="") as handle:
        mask_rows = list(csv.DictReader(handle))
        for row in progress_iter(mask_rows, total=len(mask_rows), desc="Write masks", unit="mask"):
            item = dict(row)
            item["conversion_status"] = "metadata_only"
            rows.append(item)
    atomic_write_csv(target_csv, rows, MASK_MANIFEST_FIELDS)
    return len(rows)


def _failure(task: dict[str, str], stage: str, error: str) -> dict[str, str]:
    return {
        "dataset_id": task.get("dataset_id", ""),
        "subject_id": task.get("subject_id", ""),
        "timepoint": task.get("timepoint", ""),
        "study_uid": task.get("study_uid", ""),
        "primary_dce_series_uid": task.get("primary_dce_series_uid", ""),
        "primary_dce_path": task.get("primary_dce_path", ""),
        "stage": stage,
        "error": error,
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return cleaned.strip("._") or "unknown"


def add_unified_parser(preprocess_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = preprocess_subparsers.add_parser(
        "build-unified",
        help="Build fixed-shape NPY/NIfTI outputs and train/inference manifests from real datasets.",
    )
    parser.add_argument("--output-dir", default="outputs/local/unified_dataset/full")
    parser.add_argument("--plan-dir")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(160, 160, 96), metavar=("X", "Y", "Z"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-nifti", action="store_true")
    parser.set_defaults(func=run_from_args)

    finalize = preprocess_subparsers.add_parser(
        "finalize-unified",
        help="Write labels.json sidecars and QC tables for an existing unified dataset.",
    )
    finalize.add_argument("--output-dir", default="outputs/local/unified_dataset/full")
    finalize.set_defaults(func=run_finalize_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = build_unified_dataset(
        output_dir=args.output_dir,
        plan_dir=args.plan_dir,
        max_patients=args.max_patients,
        limit=args.limit,
        target_shape=tuple(args.target_shape),
        overwrite=args.overwrite,
        write_nifti=not args.no_nifti,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_finalize_from_args(args: argparse.Namespace) -> int:
    summary = finalize_unified_dataset(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
