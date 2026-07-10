from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.data.manifest import MANIFEST_FIELDS
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.acrin_labels import build_acrin_core_labels
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.acrin_sequences import (
    effective_acrin_series_role,
    filter_acrin_dce_sources,
    is_acrin_derived_series,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.convert import (
    dicom_temporal_grouping,
    read_dicom_directory,
    read_dicom_series,
    read_dicom_series_by_temporal_position,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.dce_metadata import analyze_series_temporal_metadata, refine_group_dce_phases
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.fujian_sequences import (
    FUJIAN_DATASET_ID,
    effective_fujian_pcr_series_role,
    fujian_phase_sort_number,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.operations import should_skip_preprocessing_path
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.public_breast_sequences import (
    PUBLIC_BREAST_DATASET_IDS,
    effective_public_breast_series_role,
    filter_public_breast_dce_sources,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.realdata_plan import build_realdata_preprocessing_plan
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.unified_dataset import standardize_image
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, silence_simpleitk_warnings, suppress_warnings


PREPROCESSING_VERSION = "realdata_multimodal_v3"
DEFAULT_AUTO_WORKER_CAP = 4

DCE_ORIGINAL_ROLES = {"DCE_PRE", "DCE", "DCE_POST"}
DCE_MAP_ROLES = {"DCE_PE", "DCE_SER"}
MRI_IMAGE_ROLES = {
    "T1",
    "T1_PRE",
    "T1_POST",
    "T2",
    "DWI",
    "ADC",
    "derived",
    *DCE_ORIGINAL_ROLES,
    *DCE_MAP_ROLES,
}
SKIP_SERIES_ROLES = {"mask", "mask_seg", "localizer", "secondary_capture", "unknown"}

TIMEPOINT_MANIFEST_FIELDS = [
    *MANIFEST_FIELDS,
    "subject_id",
    "sample_dir",
    "series_manifest_path",
    "labels_json",
    "masks_manifest_path",
    "image_file_count",
    "modalities",
    "series_roles",
    "t1_count",
    "t2_count",
    "dwi_count",
    "adc_count",
    "derived_count",
    "mask_metadata_count",
    "HER2",
    "ER",
    "PR",
]

IMAGE_MANIFEST_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "sample_dir",
    "file_name",
    "file_path",
    "relative_path",
    "slice_png",
    "modality",
    "series_role",
    "source_series_role",
    "phase_index",
    "dce_contrast_phase",
    "dce_phase_evidence",
    "dce_acquisition_time",
    "contrast_bolus_start_time",
    "temporal_position_identifier",
    "temporal_position_count",
    "number_of_temporal_positions",
    "dce_temporal_grouping_method",
    "dce_temporal_grouping_evidence",
    "dce_temporal_group_sizes",
    "dce_reference_slice_count",
    "series_uid",
    "series_description",
    "source_path",
    "image_count",
    "array_shape_czyx",
    "original_size_xyz",
    "original_spacing_xyz",
    "resampled_spacing",
    "conversion_status",
]

FAILURE_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "series_uid",
    "series_role",
    "source_path",
    "stage",
    "error",
]

LABEL_SUMMARY_FIELDS = [
    "dataset_id",
    "subject_id",
    "timepoint",
    "study_uid",
    "labels_json",
    "pCR",
    "pcr_status",
    "HER2",
    "ER",
    "PR",
    "molecular_subtype",
    "laterality",
    "age",
    "BMI",
    "source_label_count",
    "split",
]


def build_multimodal_dataset(
    output_dir: str | Path,
    plan_dir: str | Path | None = None,
    max_patients: int | None = None,
    limit: int | None = None,
    target_shape: tuple[int, int, int] = (160, 160, 96),
    overwrite: bool = False,
    workers: int = 1,
    write_previews: bool = True,
) -> dict[str, Any]:
    """Build a dataset as dataset/subject/timepoint/(multi-modal images, labels, masks).

    Unlike the legacy single-image builder, this function reads the full `series.csv`
    table and converts every locally available MRI series with a recognized role. DCE
    source series are kept as separate phase files named `original_00.npy`,
    `original_01.npy`, ... so downstream code can use temporal DCE information.
    """

    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _ensure_plan(plan_dir, out_dir, max_patients=max_patients)
    workers = resolve_preprocessing_workers(workers)

    labels_by_subject = _load_labels(plan_path / "labels.csv")
    masks_by_group, masks_by_subject = _load_masks(plan_path / "masks.csv")
    groups = _load_series_groups(plan_path / "series.csv")
    conflict_keys = _conflicting_timepoint_keys(groups)

    if workers > 1:
        return _build_multimodal_dataset_parallel(
            out_dir=out_dir,
            plan_path=plan_path,
            labels_by_subject=labels_by_subject,
            masks_by_group=masks_by_group,
            masks_by_subject=masks_by_subject,
            groups=groups,
            conflict_keys=conflict_keys,
            limit=limit,
            target_shape=target_shape,
            overwrite=overwrite,
            workers=workers,
            write_previews=write_previews,
        )

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SimpleITK is required to build the multimodal dataset.") from exc
    silence_simpleitk_warnings(sitk)

    timepoint_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    label_summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    attempted_groups = 0
    converted_images = 0
    skipped_existing_groups = 0
    patient_label_dirs: set[Path] = set()

    group_items = sorted(groups.items())
    for group_key, series_records in progress_iter(
        group_items,
        total=len(group_items),
        desc="Build multimodal",
        unit="study",
    ):
        if limit is not None and attempted_groups >= limit:
            break
        dataset_id, subject_id, timepoint, study_uid = group_key
        sample_dir = _sample_dir(out_dir, group_key, conflict_keys)
        series_manifest_path = sample_dir / "series_manifest.json"
        labels_path = sample_dir / "labels.json"
        masks_manifest_path = _masks_manifest_path(sample_dir)
        labels_payload = _standardize_labels(
            dataset_id=dataset_id,
            subject_id=subject_id,
            timepoint=timepoint,
            study_uid=study_uid,
            label_records=labels_by_subject.get((dataset_id, subject_id), []),
        )
        group_masks = _mask_records_for_group(group_key, masks_by_group, masks_by_subject)

        stale_existing_sample = False
        if series_manifest_path.exists() and not overwrite:
            manifest = _read_json_object(series_manifest_path)
            existing_images = [image for image in manifest.get("images", []) if Path(image.get("file_path", "")).exists()]
            if _manifest_is_current(manifest) and existing_images:
                skipped_existing_groups += 1
                _write_json_if_changed(labels_path, labels_payload)
                _write_patient_labels_if_needed(sample_dir.parent, labels_payload, patient_label_dirs)
                _write_json_if_changed(masks_manifest_path, _masks_payload(group_key, group_masks))
                row = _timepoint_row(group_key, sample_dir, labels_payload, existing_images, group_masks)
                timepoint_rows.append(row)
                label_summary_rows.append(_label_summary_row(group_key, sample_dir, labels_payload, row["split"]))
                image_rows.extend(_image_rows_from_manifest(group_key, sample_dir, existing_images))
                if row["pcr_status"] in {"0", "1"}:
                    training_rows.append(row)
                continue
            stale_existing_sample = True

        attempted_groups += 1
        if overwrite or stale_existing_sample:
            _clear_sample_outputs(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        rebuild_sample_outputs = overwrite or stale_existing_sample or not series_manifest_path.exists()
        converted_for_group: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        dce_phase_index = 0

        annotated_records = _annotated_group_series(series_records)
        for record in _ordered_series(annotated_records):
            source = Path(record["source_path"])
            if should_skip_preprocessing_path(source):
                failures.append(_failure(record, "source", "archive files are treated as already extracted and skipped"))
                continue
            if not path_exists(source):
                failures.append(_failure(record, "source", "series source path does not exist"))
                continue
            try:
                components = _read_series_components(sitk, record)
            except Exception as exc:
                failures.append(_failure(record, "read", str(exc)))
                continue

            for component_index, component_info in enumerate(components):
                component = component_info["image"]
                try:
                    standardized, metadata = standardize_image(sitk, component, target_shape=target_shape)
                    if record["series_role"] in DCE_ORIGINAL_ROLES:
                        file_name = f"original_{dce_phase_index:02d}.npy"
                        phase_index = dce_phase_index
                        modality = "DCE"
                        component_record = _with_dce_component_metadata(record, component_info, dce_phase_index)
                        dce_phase_index += 1
                    else:
                        file_name = _series_file_name(record, counters, component_index, len(components))
                        phase_index = component_info.get("component_index", component_index) if len(components) > 1 else ""
                        modality = _modality_name(record["series_role"])
                        component_record = record
                    array_path = _image_array_path(sample_dir, file_name, component_record["series_role"], modality)
                    if rebuild_sample_outputs or not array_path.exists():
                        _save_npy_atomic(array_path, standardized)
                        converted_images += 1
                    slice_path = (
                        _write_slice_preview(
                            standardized,
                            sample_dir,
                            array_path.name,
                            overwrite=rebuild_sample_outputs,
                        )
                        if write_previews
                        else ""
                    )
                    entry = _image_entry(
                        group_key=group_key,
                        sample_dir=sample_dir,
                        array_path=array_path,
                        slice_path=slice_path,
                        record=component_record,
                        metadata=metadata,
                        modality=modality,
                        phase_index=phase_index,
                    )
                    converted_for_group.append(entry)
                except Exception as exc:
                    failures.append(_failure(record, "standardize", str(exc)))

        if not converted_for_group:
            continue

        manifest_payload = {
            "schema_version": PREPROCESSING_VERSION,
            "dataset_id": dataset_id,
            "subject_id": subject_id,
            "timepoint": timepoint,
            "study_uid": study_uid,
            "target_shape_xyz": list(target_shape),
            "images": converted_for_group,
            "counts_by_role": dict(Counter(image["series_role"] for image in converted_for_group)),
            "counts_by_modality": dict(Counter(image["modality"] for image in converted_for_group)),
        }
        _write_json_if_changed(series_manifest_path, manifest_payload)
        _write_json_if_changed(labels_path, labels_payload)
        _write_patient_labels_if_needed(sample_dir.parent, labels_payload, patient_label_dirs)
        _write_json_if_changed(masks_manifest_path, _masks_payload(group_key, group_masks))

        row = _timepoint_row(group_key, sample_dir, labels_payload, converted_for_group, group_masks)
        timepoint_rows.append(row)
        label_summary_rows.append(_label_summary_row(group_key, sample_dir, labels_payload, row["split"]))
        image_rows.extend(_image_rows_from_manifest(group_key, sample_dir, converted_for_group))
        if row["pcr_status"] in {"0", "1"}:
            training_rows.append(row)

    mask_rows = _flatten_mask_rows(masks_by_group, masks_by_subject)
    atomic_write_csv(out_dir / "inference_manifest.csv", timepoint_rows, TIMEPOINT_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "training_manifest.csv", training_rows, TIMEPOINT_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "image_manifest.csv", image_rows, IMAGE_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "labels_summary.csv", label_summary_rows, LABEL_SUMMARY_FIELDS)
    atomic_write_csv(out_dir / "failures.csv", failures, FAILURE_FIELDS)
    atomic_write_csv(out_dir / "mask_manifest.csv", mask_rows, _mask_csv_fields(mask_rows))
    dataset_qc = _dataset_qc_rows(timepoint_rows, image_rows)
    atomic_write_csv(out_dir / "dataset_qc.csv", dataset_qc, _dataset_qc_fields(dataset_qc))

    summary = {
        "output_dir": str(out_dir),
        "plan_dir": str(plan_path),
        "target_shape_xyz": list(target_shape),
        "workers": workers,
        "attempted_timepoints": attempted_groups,
        "skipped_existing_timepoints": skipped_existing_groups,
        "timepoint_rows": len(timepoint_rows),
        "training_rows": len(training_rows),
        "image_rows": len(image_rows),
        "label_summary_rows": len(label_summary_rows),
        "converted_images": converted_images,
        "write_previews": write_previews,
        "failure_rows": len(failures),
        "mask_metadata_rows": len(mask_rows),
        "timepoints_by_dataset": dict(Counter(row["dataset_id"] for row in timepoint_rows)),
        "images_by_role": dict(Counter(row["series_role"] for row in image_rows)),
        "images_by_modality": dict(Counter(row["modality"] for row in image_rows)),
        "dce_phase_files": sum(1 for row in image_rows if row["file_name"].startswith("original_")),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def resolve_preprocessing_workers(workers: int | None, *, auto_cap: int = DEFAULT_AUTO_WORKER_CAP) -> int:
    """Resolve worker count for IDE/PyCharm runs.

    `0` or negative values mean "auto". The cap avoids launching too many
    SimpleITK processes on large DICOM folders, where disk I/O usually becomes
    the bottleneck before CPU does.
    """

    if workers is not None and workers > 0:
        return workers
    cpu_count = os.cpu_count() or 4
    return max(1, min(max(cpu_count - 1, 1), auto_cap))


def _ensure_plan(plan_dir: str | Path | None, out_dir: Path, max_patients: int | None) -> Path:
    if plan_dir is not None:
        plan_path = Path(plan_dir).expanduser().resolve(strict=False)
        if not (plan_path / "series.csv").exists():
            raise FileNotFoundError(f"Missing series.csv under plan directory: {plan_path}")
        return plan_path
    plan_path = out_dir / "realdata_plan"
    build_realdata_preprocessing_plan(plan_path, max_patients=max_patients)
    return plan_path


def _build_multimodal_dataset_parallel(
    out_dir: Path,
    plan_path: Path,
    labels_by_subject: dict[tuple[str, str], list[dict[str, Any]]],
    masks_by_group: dict[tuple[str, str, str], list[dict[str, Any]]],
    masks_by_subject: dict[tuple[str, str], list[dict[str, Any]]],
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]],
    conflict_keys: set[tuple[str, str, str]],
    limit: int | None,
    target_shape: tuple[int, int, int],
    overwrite: bool,
    workers: int,
    write_previews: bool,
) -> dict[str, Any]:
    items = sorted(groups.items())
    if limit is not None:
        items = items[:limit]

    jobs = []
    for group_key, series_records in items:
        dataset_id, subject_id, timepoint, study_uid = group_key
        sample_dir = _sample_dir(out_dir, group_key, conflict_keys)
        labels_payload = _standardize_labels(
            dataset_id=dataset_id,
            subject_id=subject_id,
            timepoint=timepoint,
            study_uid=study_uid,
            label_records=labels_by_subject.get((dataset_id, subject_id), []),
        )
        group_masks = _mask_records_for_group(group_key, masks_by_group, masks_by_subject)
        jobs.append(
            {
                "group_key": group_key,
                "series_records": series_records,
                "sample_dir": str(sample_dir),
                "labels_payload": labels_payload,
                "group_masks": group_masks,
                "target_shape": target_shape,
                "overwrite": overwrite,
                "write_previews": write_previews,
            }
        )

    timepoint_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    label_summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    attempted_groups = 0
    converted_images = 0
    skipped_existing_groups = 0
    patient_labels: dict[Path, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_process_group_worker, job) for job in jobs]
        total = len(futures)
        for future in progress_iter(
            as_completed(futures),
            total=total,
            desc="Build multimodal",
            unit="study",
        ):
            result = future.result()
            attempted_groups += int(result.get("attempted_group", 0))
            converted_images += int(result.get("converted_images", 0))
            skipped_existing_groups += int(result.get("skipped_existing_group", 0))
            failures.extend(result.get("failures", []))
            row = result.get("timepoint_row")
            if row:
                timepoint_rows.append(row)
                label_summary_rows.append(result["label_summary_row"])
                image_rows.extend(result.get("image_rows", []))
                if row.get("pcr_status") in {"0", "1"}:
                    training_rows.append(row)
                patient_labels[Path(row["sample_dir"]).parent] = result["patient_labels_payload"]

    for patient_dir, labels_payload in patient_labels.items():
        _write_patient_labels_if_needed(patient_dir, labels_payload, set())

    timepoint_rows.sort(key=lambda row: (row["dataset_id"], row["patient_uid"], row["timepoint"], row["study_uid"]))
    training_rows.sort(key=lambda row: (row["dataset_id"], row["patient_uid"], row["timepoint"], row["study_uid"]))
    image_rows.sort(key=lambda row: (row["dataset_id"], row["subject_id"], row["timepoint"], row["file_name"]))
    label_summary_rows.sort(key=lambda row: (row["dataset_id"], row["subject_id"], row["timepoint"], row["study_uid"]))

    mask_rows = _flatten_mask_rows(masks_by_group, masks_by_subject)
    atomic_write_csv(out_dir / "inference_manifest.csv", timepoint_rows, TIMEPOINT_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "training_manifest.csv", training_rows, TIMEPOINT_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "image_manifest.csv", image_rows, IMAGE_MANIFEST_FIELDS)
    atomic_write_csv(out_dir / "labels_summary.csv", label_summary_rows, LABEL_SUMMARY_FIELDS)
    atomic_write_csv(out_dir / "failures.csv", failures, FAILURE_FIELDS)
    atomic_write_csv(out_dir / "mask_manifest.csv", mask_rows, _mask_csv_fields(mask_rows))
    dataset_qc = _dataset_qc_rows(timepoint_rows, image_rows)
    atomic_write_csv(out_dir / "dataset_qc.csv", dataset_qc, _dataset_qc_fields(dataset_qc))

    summary = {
        "output_dir": str(out_dir),
        "plan_dir": str(plan_path),
        "target_shape_xyz": list(target_shape),
        "workers": workers,
        "attempted_timepoints": attempted_groups,
        "skipped_existing_timepoints": skipped_existing_groups,
        "timepoint_rows": len(timepoint_rows),
        "training_rows": len(training_rows),
        "image_rows": len(image_rows),
        "label_summary_rows": len(label_summary_rows),
        "converted_images": converted_images,
        "write_previews": write_previews,
        "failure_rows": len(failures),
        "mask_metadata_rows": len(mask_rows),
        "timepoints_by_dataset": dict(Counter(row["dataset_id"] for row in timepoint_rows)),
        "images_by_role": dict(Counter(row["series_role"] for row in image_rows)),
        "images_by_modality": dict(Counter(row["modality"] for row in image_rows)),
        "dce_phase_files": sum(1 for row in image_rows if row["file_name"].startswith("original_")),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def _process_group_worker(job: dict[str, Any]) -> dict[str, Any]:
    suppress_warnings()
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SimpleITK is required to build the multimodal dataset.") from exc
    silence_simpleitk_warnings(sitk)

    group_key = tuple(job["group_key"])
    sample_dir = Path(job["sample_dir"])
    series_records = job["series_records"]
    labels_payload = job["labels_payload"]
    group_masks = job["group_masks"]
    target_shape = tuple(job["target_shape"])
    overwrite = bool(job["overwrite"])
    write_previews = bool(job.get("write_previews", True))

    series_manifest_path = sample_dir / "series_manifest.json"
    labels_path = sample_dir / "labels.json"
    masks_manifest_path = _masks_manifest_path(sample_dir)

    stale_existing_sample = False
    if series_manifest_path.exists() and not overwrite:
        manifest = _read_json_object(series_manifest_path)
        existing_images = [image for image in manifest.get("images", []) if Path(image.get("file_path", "")).exists()]
        if _manifest_is_current(manifest) and existing_images:
            _write_json_if_changed(labels_path, labels_payload)
            _write_json_if_changed(masks_manifest_path, _masks_payload(group_key, group_masks))
            row = _timepoint_row(group_key, sample_dir, labels_payload, existing_images, group_masks)
            return {
                "attempted_group": 0,
                "skipped_existing_group": 1,
                "converted_images": 0,
                "failures": [],
                "timepoint_row": row,
                "label_summary_row": _label_summary_row(group_key, sample_dir, labels_payload, row["split"]),
                "image_rows": _image_rows_from_manifest(group_key, sample_dir, existing_images),
                "patient_labels_payload": _patient_labels_payload(labels_payload),
            }
        stale_existing_sample = True

    if overwrite or stale_existing_sample:
        _clear_sample_outputs(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rebuild_sample_outputs = overwrite or stale_existing_sample or not series_manifest_path.exists()
    converted_for_group: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    converted_images = 0
    counters: Counter[str] = Counter()
    dce_phase_index = 0

    annotated_records = _annotated_group_series(series_records)
    for record in _ordered_series(annotated_records):
        source = Path(record["source_path"])
        if should_skip_preprocessing_path(source):
            failures.append(_failure(record, "source", "archive files are treated as already extracted and skipped"))
            continue
        if not path_exists(source):
            failures.append(_failure(record, "source", "series source path does not exist"))
            continue
        try:
            components = _read_series_components(sitk, record)
        except Exception as exc:
            failures.append(_failure(record, "read", str(exc)))
            continue

        for component_index, component_info in enumerate(components):
            component = component_info["image"]
            try:
                standardized, metadata = standardize_image(sitk, component, target_shape=target_shape)
                if record["series_role"] in DCE_ORIGINAL_ROLES:
                    file_name = f"original_{dce_phase_index:02d}.npy"
                    phase_index = dce_phase_index
                    modality = "DCE"
                    component_record = _with_dce_component_metadata(record, component_info, dce_phase_index)
                    dce_phase_index += 1
                else:
                    file_name = _series_file_name(record, counters, component_index, len(components))
                    phase_index = component_info.get("component_index", component_index) if len(components) > 1 else ""
                    modality = _modality_name(record["series_role"])
                    component_record = record
                array_path = _image_array_path(sample_dir, file_name, component_record["series_role"], modality)
                if rebuild_sample_outputs or not array_path.exists():
                    _save_npy_atomic(array_path, standardized)
                    converted_images += 1
                slice_path = (
                    _write_slice_preview(
                        standardized,
                        sample_dir,
                        array_path.name,
                        overwrite=rebuild_sample_outputs,
                    )
                    if write_previews
                    else ""
                )
                converted_for_group.append(
                    _image_entry(
                        group_key=group_key,
                        sample_dir=sample_dir,
                        array_path=array_path,
                        slice_path=slice_path,
                        record=component_record,
                        metadata=metadata,
                        modality=modality,
                        phase_index=phase_index,
                    )
                )
            except Exception as exc:
                failures.append(_failure(record, "standardize", str(exc)))

    if not converted_for_group:
        return {
            "attempted_group": 1,
            "skipped_existing_group": 0,
            "converted_images": converted_images,
            "failures": failures,
            "timepoint_row": None,
        }

    manifest_payload = {
        "schema_version": PREPROCESSING_VERSION,
        "dataset_id": group_key[0],
        "subject_id": group_key[1],
        "timepoint": group_key[2],
        "study_uid": group_key[3],
        "target_shape_xyz": list(target_shape),
        "images": converted_for_group,
        "counts_by_role": dict(Counter(image["series_role"] for image in converted_for_group)),
        "counts_by_modality": dict(Counter(image["modality"] for image in converted_for_group)),
    }
    _write_json_if_changed(series_manifest_path, manifest_payload)
    _write_json_if_changed(labels_path, labels_payload)
    _write_json_if_changed(masks_manifest_path, _masks_payload(group_key, group_masks))
    row = _timepoint_row(group_key, sample_dir, labels_payload, converted_for_group, group_masks)
    return {
        "attempted_group": 1,
        "skipped_existing_group": 0,
        "converted_images": converted_images,
        "failures": failures,
        "timepoint_row": row,
        "label_summary_row": _label_summary_row(group_key, sample_dir, labels_payload, row["split"]),
        "image_rows": _image_rows_from_manifest(group_key, sample_dir, converted_for_group),
        "patient_labels_payload": _patient_labels_payload(labels_payload),
    }


def _load_series_groups(series_csv: Path) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    with series_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["source_series_role"] = row.get("series_role", "")
            row["series_role"] = _effective_series_role(row)
            role = row.get("series_role", "")
            if role in SKIP_SERIES_ROLES or role not in MRI_IMAGE_ROLES:
                continue
            if row.get("local_exists", "") != "True":
                continue
            key = (
                row.get("dataset_id", ""),
                row.get("subject_id", ""),
                row.get("timepoint", "unknown") or "unknown",
                row.get("study_uid", "unknown") or "unknown",
            )
            groups[key].append(row)
    return {key: _dedupe_series(rows) for key, rows in groups.items()}


def _annotated_group_series(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    annotated = [dict(row) for row in rows]
    temporal_summaries: dict[str, dict[str, Any]] = {}
    reference_slice_counts = _dce_reference_slice_counts(annotated)
    for row in annotated:
        if row.get("series_role") not in DCE_ORIGINAL_ROLES:
            continue
        reference_slice_count = reference_slice_counts.get(row.get("series_uid") or row.get("source_path", ""), 0)
        if reference_slice_count:
            row["dce_reference_slice_count"] = str(reference_slice_count)
        try:
            summary = analyze_series_temporal_metadata(
                row.get("source_path", ""),
                series_uid=row.get("series_uid", ""),
                series_description=row.get("series_description", ""),
                max_files=48,
            ).to_dict()
        except Exception as exc:
            summary = {
                "contrast_phase": "unknown",
                "phase_evidence": f"temporal metadata read failed: {exc}",
            }
        key = row.get("series_uid") or row.get("source_path", "")
        temporal_summaries[key] = summary
        for field in (
            "contrast_phase",
            "phase_evidence",
            "acquisition_time_min",
            "contrast_bolus_start_time",
            "temporal_position_count",
            "number_of_temporal_positions",
        ):
            row[f"dce_{field}" if field in {"contrast_phase", "phase_evidence"} else field] = str(summary.get(field, ""))
        row.update(_dce_temporal_grouping_summary(row))
    annotated = filter_acrin_dce_sources(annotated)
    annotated = filter_public_breast_dce_sources(annotated)
    annotated = _select_primary_dce_sources(annotated)
    refine_group_dce_phases(annotated, temporal_summaries)
    return annotated


def _dedupe_series(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("source_path", ""), row.get("series_uid", ""), row.get("series_role", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _estimate_dce_temporal_group_count(row: dict[str, str]) -> int:
    summary = _dce_temporal_grouping_summary(row)
    return _to_int(summary.get("dce_temporal_group_count", "")) or 1


def _dce_temporal_grouping_summary(row: dict[str, str]) -> dict[str, str]:
    try:
        grouping = dicom_temporal_grouping(
            Path(row.get("source_path", "")),
            row.get("series_uid", ""),
            reference_slice_count=_to_int(row.get("dce_reference_slice_count", "")) or None,
        )
    except Exception as exc:
        return {
            "dce_temporal_group_count": "1",
            "dce_temporal_grouping_method": "read_failed",
            "dce_temporal_grouping_evidence": str(exc),
            "dce_temporal_group_sizes": "",
        }
    return {
        "dce_temporal_group_count": str(max(grouping.group_count, 1)),
        "dce_temporal_grouping_method": grouping.method,
        "dce_temporal_grouping_evidence": grouping.evidence,
        "dce_temporal_group_sizes": json.dumps(grouping.group_sizes, ensure_ascii=False),
        "dce_reference_slice_count": str(grouping.reference_slice_count or row.get("dce_reference_slice_count", "")),
        # Keep the file groups in memory for this worker so the conversion step does not
        # need to inspect every DICOM header a second time.
        "dce_temporal_groups_json": json.dumps(grouping.groups, ensure_ascii=False),
    }


def _dce_reference_slice_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    reference_counts = [
        _to_int(row.get("image_count", ""))
        for row in rows
        if row.get("series_role") in DCE_MAP_ROLES
        and _to_int(row.get("image_count", "")) >= 8
        and row.get("local_exists", "") == "True"
    ]
    if not reference_counts:
        return {}
    counts = Counter(reference_counts)
    output: dict[str, int] = {}
    for row in rows:
        if row.get("series_role") not in DCE_ORIGINAL_ROLES:
            continue
        image_count = _to_int(row.get("image_count", ""))
        candidates = [
            count
            for count, frequency in counts.items()
            if count < image_count and image_count % count == 0 and 1 < image_count // count <= 20
        ]
        if not candidates:
            continue
        selected = sorted(candidates, key=lambda count: (-counts[count], -count))[0]
        output[row.get("series_uid") or row.get("source_path", "")] = selected
    return output


def _select_primary_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(row.get("dataset_id", "") in PUBLIC_BREAST_DATASET_IDS for row in rows):
        return rows

    dce_rows = [row for row in rows if row.get("series_role") in DCE_ORIGINAL_ROLES]
    if len(dce_rows) <= 1:
        return rows

    multiphase_rows = [row for row in dce_rows if _to_int(row.get("dce_temporal_group_count", "")) > 1]
    if not multiphase_rows:
        return rows

    selected = sorted(multiphase_rows, key=_primary_dce_source_key)[0]
    selected_uid = selected.get("series_uid", "")
    selected_path = selected.get("source_path", "")
    output: list[dict[str, str]] = []
    for row in rows:
        if row.get("series_role") not in DCE_ORIGINAL_ROLES:
            output.append(row)
            continue
        if row.get("series_uid", "") == selected_uid and row.get("source_path", "") == selected_path:
            output.append(row)
    return output


def _primary_dce_source_key(row: dict[str, str]) -> tuple[int, int, int, float, str, str]:
    phase_count = _to_int(row.get("dce_temporal_group_count", ""))
    image_count = _to_int(row.get("image_count", ""))
    role_rank = {"DCE": 0, "DCE_PRE": 1, "DCE_POST": 2}.get(row.get("source_series_role", row.get("series_role", "")), 9)
    return (
        -phase_count,
        -image_count,
        role_rank,
        _path_number(row.get("source_path", "")),
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _to_int(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _conflicting_timepoint_keys(groups: dict[tuple[str, str, str, str], list[dict[str, str]]]) -> set[tuple[str, str, str]]:
    by_timepoint: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for dataset_id, subject_id, timepoint, study_uid in groups:
        by_timepoint[(dataset_id, subject_id, timepoint)].add(study_uid)
    return {key for key, studies in by_timepoint.items() if len(studies) > 1}


def _sample_dir(out_dir: Path, group_key: tuple[str, str, str, str], conflicts: set[tuple[str, str, str]]) -> Path:
    dataset_id, subject_id, timepoint, study_uid = group_key
    timepoint_name = _safe_name(timepoint)
    if (dataset_id, subject_id, timepoint) in conflicts:
        timepoint_name = f"{timepoint_name}__{_short_hash(study_uid)}"
    return out_dir / _safe_name(dataset_id) / _safe_name(subject_id) / timepoint_name


def _ordered_series(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=_series_order_key)


def _series_order_key(row: dict[str, str]) -> tuple[int, float, float, str, str]:
    if row.get("dataset_id", "") == "ispy1" and row.get("series_role", "") in DCE_ORIGINAL_ROLES:
        return (
            _role_sort_key(row.get("series_role", "")),
            _ispy1_acquisition_sort_number(row),
            _path_number(row.get("source_path", "")),
            row.get("series_description", ""),
            row.get("series_uid", ""),
        )
    return (
        _role_sort_key(row.get("series_role", "")),
        _path_number(row.get("source_path", "")),
        _phase_sort_number(row),
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _role_sort_key(role: str) -> int:
    order = {
        "DCE_PRE": 0,
        "DCE": 1,
        "DCE_POST": 2,
        "DCE_PE": 3,
        "DCE_SER": 4,
        "T1_PRE": 5,
        "T1": 6,
        "T1_POST": 7,
        "T2": 8,
        "DWI": 9,
        "ADC": 10,
        "derived": 11,
    }
    return order.get(role, 99)


def _effective_series_role(row: dict[str, str]) -> str:
    if row.get("dataset_id", "") == "acrin_contralateral":
        return effective_acrin_series_role(row)
    if row.get("dataset_id", "") == FUJIAN_DATASET_ID:
        return effective_fujian_pcr_series_role(row)
    if row.get("dataset_id", "") in PUBLIC_BREAST_DATASET_IDS:
        return effective_public_breast_series_role(row)
    role = row.get("series_role", "")
    if role in DCE_ORIGINAL_ROLES and _looks_like_derived_dce(row):
        return "derived"
    return role


def _looks_like_derived_dce(row: dict[str, str]) -> bool:
    if row.get("dataset_id", "") == "acrin_contralateral":
        return is_acrin_derived_series(row)
    text = " ".join([row.get("series_description", ""), row.get("source_path", "")]).lower()
    return any(token in text for token in ("sdyn", "sub", "mip", "cad", "tram", "reformat", "projection"))


def _phase_sort_number(row: dict[str, str]) -> int:
    if row.get("series_role", "") not in DCE_ORIGINAL_ROLES:
        return 10**9
    return fujian_phase_sort_number(" ".join([row.get("series_description", ""), row.get("source_path", "")]))


def _ispy1_acquisition_sort_number(row: dict[str, str]) -> float:
    for key in ("dce_acquisition_time", "acquisition_time_min"):
        parsed = _time_to_sort_number(row.get(key, ""))
        if parsed is not None:
            return parsed
    match = re.search(r"acquisition_time=([^;]+)", row.get("notes", ""))
    if match:
        parsed = _time_to_sort_number(match.group(1))
        if parsed is not None:
            return parsed
    return float("inf")


def _time_to_sort_number(value: Any) -> float | None:
    text = str(value or "").strip().replace(":", "")
    if not text:
        return None
    try:
        if "." in text:
            whole, frac = text.split(".", 1)
        else:
            whole, frac = text, "0"
        whole = whole.rjust(6, "0")[:6]
        return int(whole[0:2]) * 3600 + int(whole[2:4]) * 60 + int(whole[4:6]) + float(f"0.{frac}")
    except (TypeError, ValueError):
        return None


def _path_number(value: str) -> float:
    name = Path(value).name
    match = re.match(r"(\d+(?:\.\d+)?)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 1e9
    return 1e9


def _read_series_image(sitk: Any, record: dict[str, str]) -> Any:
    source = Path(record.get("source_path", ""))
    series_uid = record.get("series_uid", "")
    try:
        return read_dicom_series(sitk, source, series_uid)
    except Exception:
        if record.get("dataset_id", "") == FUJIAN_DATASET_ID and series_uid:
            raise
        return read_dicom_directory(sitk, source)


def _read_series_components(sitk: Any, record: dict[str, str]) -> list[dict[str, Any]]:
    if record.get("series_role") in DCE_ORIGINAL_ROLES:
        try:
            temporal_images = _read_cached_dce_temporal_images(sitk, record)
            if not temporal_images:
                temporal_images = read_dicom_series_by_temporal_position(
                    sitk,
                    Path(record.get("source_path", "")),
                    record.get("series_uid", ""),
                    reference_slice_count=_to_int(record.get("dce_reference_slice_count", "")) or None,
                )
        except Exception:
            temporal_images = [("", _read_series_image(sitk, record))]
        components: list[dict[str, Any]] = []
        for temporal_index, (temporal_position, image) in enumerate(temporal_images):
            for component_index, component in enumerate(_split_components(sitk, image)):
                components.append(
                    {
                        "image": component,
                        "temporal_position": temporal_position,
                        "temporal_index": temporal_index,
                        "component_index": component_index,
                    }
                )
        return components

    image = _read_series_image(sitk, record)
    return [
        {"image": component, "temporal_position": "", "temporal_index": "", "component_index": index}
        for index, component in enumerate(_split_components(sitk, image))
    ]


def _read_cached_dce_temporal_images(sitk: Any, record: dict[str, str]) -> list[tuple[str, Any]]:
    groups = _cached_dce_temporal_groups(record)
    if not groups:
        return []
    silence_simpleitk_warnings(sitk)
    reader = sitk.ImageSeriesReader()
    output: list[tuple[str, Any]] = []
    for temporal_position, names in groups:
        if not names:
            continue
        reader.SetFileNames(tuple(names))
        output.append((temporal_position, reader.Execute()))
    return output


def _cached_dce_temporal_groups(record: dict[str, str]) -> list[tuple[str, list[str]]]:
    text = record.get("dce_temporal_groups_json", "")
    if not text:
        return []
    try:
        raw_groups = json.loads(text)
    except json.JSONDecodeError:
        return []
    groups: list[tuple[str, list[str]]] = []
    if not isinstance(raw_groups, list):
        return groups
    for item in raw_groups:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        temporal_position, names = item
        if not isinstance(names, list):
            continue
        clean_names = [str(name) for name in names if str(name)]
        if clean_names:
            groups.append((str(temporal_position or ""), clean_names))
    return groups


def _with_dce_component_metadata(
    record: dict[str, str],
    component_info: dict[str, Any],
    phase_index: int,
) -> dict[str, str]:
    contrast_phase = record.get("dce_contrast_phase", "")
    evidence = record.get("dce_phase_evidence", "")
    temporal_position = str(component_info.get("temporal_position", "") or "")
    if temporal_position:
        if phase_index == 0 and contrast_phase in {"", "unknown"}:
            contrast_phase = "pre"
            evidence = evidence or "first temporal position in DCE series; injection time unavailable"
        elif phase_index > 0 and contrast_phase in {"", "unknown", "pre"}:
            contrast_phase = "post"
            evidence = evidence or "subsequent temporal position in DCE series; injection time unavailable"
    output = dict(record)
    output.update(
        {
            "dce_contrast_phase": contrast_phase,
            "dce_phase_evidence": evidence,
            "temporal_position_count": record.get("temporal_position_count", ""),
            "number_of_temporal_positions": record.get("number_of_temporal_positions", ""),
            "dce_temporal_grouping_method": record.get("dce_temporal_grouping_method", ""),
            "dce_temporal_grouping_evidence": record.get("dce_temporal_grouping_evidence", ""),
            "dce_temporal_group_sizes": record.get("dce_temporal_group_sizes", ""),
            "dce_reference_slice_count": record.get("dce_reference_slice_count", ""),
            "temporal_position_identifier": temporal_position,
        }
    )
    if contrast_phase == "pre":
        output["series_role"] = "DCE_PRE"
    elif contrast_phase == "post":
        output["series_role"] = "DCE_POST"
    else:
        output["series_role"] = "DCE"
    return output


def _split_components(sitk: Any, image: Any) -> list[Any]:
    dimension = int(image.GetDimension())
    if dimension == 3:
        return [image]
    if dimension != 4:
        raise ValueError(f"Expected 3D or 4D image, got dimension {dimension}")
    size = list(image.GetSize())
    components: list[Any] = []
    for index in range(size[3]):
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, index])
        components.append(extractor.Execute(image))
    return components


def _write_slice_preview(array_czyx: np.ndarray, sample_dir: Path, image_file_name: str, overwrite: bool = False) -> Path:
    slice_dir = sample_dir / "slice"
    slice_dir.mkdir(parents=True, exist_ok=True)
    target = slice_dir / f"{Path(image_file_name).stem}.png"
    if target.exists() and not overwrite:
        return target
    volume = np.asarray(array_czyx)
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim == 2:
        plane = volume
    elif volume.ndim == 3:
        plane = volume[volume.shape[0] // 2]
    else:
        plane = np.squeeze(volume)
        if plane.ndim > 2:
            plane = plane[plane.shape[0] // 2]
    plane = np.asarray(plane, dtype=np.float32)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        scaled = np.zeros(plane.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((plane - low) / (high - low), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(scaled).save(target)
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.imsave(target, scaled, cmap="gray", vmin=0, vmax=255)
    return target


def _series_file_name(
    record: dict[str, str],
    counters: Counter[str],
    component_index: int,
    component_count: int,
) -> str:
    role = record.get("series_role", "unknown")
    if role == "DCE_PE":
        base = _pe_name(record) or "PE"
    elif role == "DCE_SER":
        base = "SER"
    elif role == "derived":
        base = "derived"
    else:
        base = role
    base = _safe_name(base)
    count = counters[base]
    counters[base] += 1
    if component_count > 1:
        return f"{base}_{count:02d}_{component_index:02d}.npy"
    if count == 0:
        return f"{base}.npy"
    return f"{base}_{count:02d}.npy"


def _pe_name(record: dict[str, str]) -> str:
    text = " ".join([record.get("series_description", ""), record.get("source_path", "")]).lower()
    match = re.search(r"\bpe\s*([1-9])\b", text)
    return f"PE{match.group(1)}" if match else ""


def _modality_name(role: str) -> str:
    if role in {"DCE_PE", "DCE_SER"}:
        return "DCE"
    if role.startswith("T1"):
        return "T1"
    if role in {"T2", "DWI", "ADC"}:
        return role
    return role


def _image_array_path(sample_dir: Path, file_name: str, series_role: str, modality: str) -> Path:
    return sample_dir / _modality_dir_name(series_role, modality) / file_name


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _modality_dir_name(series_role: str, modality: str) -> str:
    if series_role in {"DCE_PE", "DCE_SER", "derived"}:
        return "parametric_maps"
    if modality in {"DCE", "T1", "T2", "DWI", "ADC"}:
        return modality
    return _safe_name(modality or series_role or "images")


def _masks_manifest_path(sample_dir: Path) -> Path:
    return sample_dir / "masks" / "manifest.json"


def _image_entry(
    group_key: tuple[str, str, str, str],
    sample_dir: Path,
    array_path: Path,
    slice_path: Path | str,
    record: dict[str, str],
    metadata: dict[str, Any],
    modality: str,
    phase_index: int | str,
) -> dict[str, Any]:
    dataset_id, subject_id, timepoint, study_uid = group_key
    return {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "timepoint": timepoint,
        "study_uid": study_uid,
        "sample_dir": str(sample_dir),
        "file_name": array_path.name,
        "file_path": str(array_path),
        "relative_path": str(array_path.relative_to(sample_dir.parent.parent.parent)).replace("\\", "/"),
        "slice_png": str(slice_path) if slice_path else "",
        "modality": modality,
        "series_role": record.get("series_role", ""),
        "source_series_role": record.get("source_series_role", record.get("series_role", "")),
        "phase_index": phase_index,
        "dce_contrast_phase": record.get("dce_contrast_phase", ""),
        "dce_phase_evidence": record.get("dce_phase_evidence", ""),
        "dce_acquisition_time": record.get("dce_acquisition_time", record.get("acquisition_time_min", "")),
        "contrast_bolus_start_time": record.get("contrast_bolus_start_time", ""),
        "temporal_position_identifier": record.get("temporal_position_identifier", ""),
        "temporal_position_count": record.get("temporal_position_count", ""),
        "number_of_temporal_positions": record.get("number_of_temporal_positions", ""),
        "dce_temporal_grouping_method": record.get("dce_temporal_grouping_method", ""),
        "dce_temporal_grouping_evidence": record.get("dce_temporal_grouping_evidence", ""),
        "dce_temporal_group_sizes": record.get("dce_temporal_group_sizes", ""),
        "dce_reference_slice_count": record.get("dce_reference_slice_count", ""),
        "series_uid": record.get("series_uid", ""),
        "series_description": record.get("series_description", ""),
        "source_path": record.get("source_path", ""),
        "image_count": record.get("image_count", ""),
        "array_shape_czyx": metadata.get("array_shape_czyx", ""),
        "original_size_xyz": metadata.get("original_size_xyz", ""),
        "original_spacing_xyz": metadata.get("original_spacing_xyz", ""),
        "resampled_spacing": metadata.get("resampled_spacing", ""),
        "conversion_status": "converted",
    }


def _image_rows_from_manifest(
    group_key: tuple[str, str, str, str],
    sample_dir: Path,
    images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in images:
        row = dict(image)
        row.setdefault("dataset_id", group_key[0])
        row.setdefault("subject_id", group_key[1])
        row.setdefault("timepoint", group_key[2])
        row.setdefault("study_uid", group_key[3])
        row.setdefault("sample_dir", str(sample_dir))
        rows.append(row)
    return rows


def _timepoint_row(
    group_key: tuple[str, str, str, str],
    sample_dir: Path,
    labels: dict[str, Any],
    images: list[dict[str, Any]],
    masks: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_id, subject_id, timepoint, study_uid = group_key
    roles = sorted({str(image.get("series_role", "")) for image in images if image.get("series_role", "")})
    modalities = sorted({str(image.get("modality", "")) for image in images if image.get("modality", "")})
    role_counts = Counter(str(image.get("series_role", "")) for image in images)
    primary = _primary_image_path(images)
    pcr_status = str(labels.get("pcr_status", "unknown"))
    split = _split_for_subject(dataset_id, subject_id) if pcr_status in {"0", "1"} else "inference"
    clinical_labels = json.dumps(_public_label_fields(labels), ensure_ascii=False, sort_keys=True)
    row = {field: "" for field in TIMEPOINT_MANIFEST_FIELDS}
    row.update(
        {
            "dataset_id": dataset_id,
            "patient_uid": subject_id,
            "subject_id": subject_id,
            "study_uid": study_uid,
            "original_patient_mapping": "not_exported",
            "original_path": ";".join(sorted({str(image.get("source_path", "")) for image in images})),
            "standardized_data_path": primary,
            "timepoint": timepoint,
            "laterality": labels.get("laterality", "unknown") or "unknown",
            "lesion_id": "unknown",
            "available_modalities": json.dumps(modalities, ensure_ascii=False),
            "dce_phase_count": sum(1 for image in images if str(image.get("file_name", "")).startswith("original_")),
            "image_shape": "(1, 96, 160, 160)",
            "spacing": "see_series_manifest",
            "clinical_labels": clinical_labels,
            "pathology_labels": json.dumps(labels.get("pathology", {}), ensure_ascii=False, sort_keys=True),
            "molecular_subtype": labels.get("molecular_subtype", "unknown") or "unknown",
            "response_label": pcr_status,
            "pcr_status": pcr_status,
            "survival_followup": json.dumps(labels.get("survival", {}), ensure_ascii=False, sort_keys=True),
            "roi_status": "metadata_only" if masks else "not_available",
            "missing_modalities": json.dumps(_missing_modalities(modalities), ensure_ascii=False),
            "qc_status": "ready",
            "exclusion_reason": "",
            "preprocessing_version": PREPROCESSING_VERSION,
            "split": split,
            "source_audit_json": "",
            "sample_dir": str(sample_dir),
            "series_manifest_path": str(sample_dir / "series_manifest.json"),
            "labels_json": str(sample_dir / "labels.json"),
            "masks_manifest_path": str(_masks_manifest_path(sample_dir)),
            "image_file_count": len(images),
            "modalities": json.dumps(modalities, ensure_ascii=False),
            "series_roles": json.dumps(roles, ensure_ascii=False),
            "t1_count": role_counts.get("T1", 0) + role_counts.get("T1_PRE", 0) + role_counts.get("T1_POST", 0),
            "t2_count": role_counts.get("T2", 0),
            "dwi_count": role_counts.get("DWI", 0),
            "adc_count": role_counts.get("ADC", 0),
            "derived_count": role_counts.get("derived", 0) + role_counts.get("DCE_PE", 0) + role_counts.get("DCE_SER", 0),
            "mask_metadata_count": len(masks),
            "HER2": labels.get("HER2", "unknown"),
            "ER": labels.get("ER", "unknown"),
            "PR": labels.get("PR", "unknown"),
        }
    )
    return row


def _label_summary_row(
    group_key: tuple[str, str, str, str],
    sample_dir: Path,
    labels: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    dataset_id, subject_id, timepoint, study_uid = group_key
    return {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "timepoint": timepoint,
        "study_uid": study_uid,
        "labels_json": str(sample_dir / "labels.json"),
        "pCR": labels.get("pCR"),
        "pcr_status": labels.get("pcr_status", "unknown"),
        "HER2": labels.get("HER2", "unknown"),
        "ER": labels.get("ER", "unknown"),
        "PR": labels.get("PR", "unknown"),
        "molecular_subtype": labels.get("molecular_subtype", "unknown"),
        "laterality": labels.get("laterality", "unknown"),
        "age": labels.get("age"),
        "BMI": labels.get("BMI"),
        "source_label_count": labels.get("source_label_count", 0),
        "split": split,
    }


def _primary_image_path(images: list[dict[str, Any]]) -> str:
    for image in images:
        if str(image.get("file_name", "")).startswith("original_"):
            return str(image.get("file_path", ""))
    return str(images[0].get("file_path", "")) if images else ""


def _missing_modalities(modalities: list[str]) -> list[str]:
    expected = ["DCE", "T2", "DWI", "ADC"]
    present = set(modalities)
    return [item for item in expected if item not in present]


def _load_labels(labels_csv: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not labels_csv.exists():
        return by_subject
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels = _read_json_object_text(row.get("labels", ""))
            by_subject[(row.get("dataset_id", ""), row.get("subject_id", ""))].append(
                {
                    "source_file": row.get("source_file", ""),
                    "key_column": row.get("key_column", ""),
                    "labels": labels,
                    "notes": row.get("notes", ""),
                }
            )
    return by_subject


def _standardize_labels(
    dataset_id: str,
    subject_id: str,
    timepoint: str,
    study_uid: str,
    label_records: list[dict[str, Any]],
) -> dict[str, Any]:
    is_acrin = dataset_id == "acrin_contralateral"
    pcr_value, pcr_source = _normalized_pcr(label_records)
    acrin_summary = build_acrin_core_labels(label_records) if is_acrin else {}
    tcga_summary_value = _first_label(label_records, ("tcga_core_labels",))
    tcga_summary = tcga_summary_value if isinstance(tcga_summary_value, dict) else {}
    tcga_pathology = _first_present(tcga_summary, ("pathology",))
    tcga_pathology = tcga_pathology if isinstance(tcga_pathology, dict) else {}
    tcga_survival = _first_present(tcga_summary, ("survival",))
    tcga_survival = tcga_survival if isinstance(tcga_survival, dict) else {}
    payload = {
        "schema_version": "unified_labels_v1",
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "patient_id": subject_id,
        "timepoint": timepoint,
        "study_uid": study_uid,
        "pCR": pcr_value,
        "pcr_status": str(pcr_value) if pcr_value in {0, 1} else "unknown",
        "HER2": _normalized_receptor(_first_label(label_records, ("HER2结果", "HER2 Positive", "HER2", "HER2 status", "HER2 Status"))),
        "ER": _normalized_receptor(_first_label(label_records, ("ER", "ER positive", "ER Positive", "ER阳性比率"))),
        "PR": _normalized_receptor(_first_label(label_records, ("PR", "PR Positive", "PR positive", "PR阳性比率"))),
        "molecular_subtype": _value_or_unknown(
            _first_label(label_records, ("分子分型", "HR_HER2_STATUS", "HR_HER2_CATEGORY", "Molecular subtype"))
        ),
        "age": _first_present(acrin_summary, ("age",))
        or _first_label(label_records, ("年龄", "AGE at MRI1 (yrs)", "Age", "Age at mammo (days)", "Patient age at Registration")),
        "BMI": _first_label(label_records, ("BMI",)),
        "sex": _value_or_unknown(_first_present(acrin_summary, ("sex",)) or _first_label(label_records, ("性别", "Sex", "Gender"))),
        "laterality": _normalized_laterality(
            _first_present(acrin_summary, ("laterality",))
            or _first_label(label_records, ("患侧", "breast laterality", "Laterality", "site of lesion biopsied", "SITE OF LESION BIOPSIED"))
        ),
        "clinical_stage": _value_or_unknown(_first_label(label_records, ("临床分期", "治疗前分期（cTNM）", "Clinical stage"))),
        "pathologic_stage": _value_or_unknown(
            _first_present(acrin_summary, ("pathologic_stage",))
            or _first_label(label_records, ("Pathologic stage", "新辅后RCB分级", "新辅后Miller-Payne分级"))
        ),
        "tumor_size_cm": _first_present(acrin_summary, ("tumor_size_cm",))
        or _first_label(label_records, ("治疗前肿瘤最大径（cm）", "Clinical size pre", "LD 1 (cm) ", "LD_T0")),
        "treatment": _value_or_unknown(_first_label(label_records, ("治疗方案", "chemo", "AC only=0, taxol=1"))),
        "response": _value_or_unknown(_first_label(label_records, RESPONSE_ALIASES)),
        "pathology": {
            "histologic_type": _first_present(acrin_summary, ("histologic_type",)) or _first_label(label_records, ("Histologic type", "Cancer type")),
            "grade": _first_present(acrin_summary, ("grade",)) or _first_label(label_records, ("Grade", "Tumor Grade")),
            **({"pathologic_t": acrin_summary.get("pathologic_t")} if acrin_summary.get("pathologic_t") else {}),
            **({"pathologic_n": acrin_summary.get("pathologic_n")} if acrin_summary.get("pathologic_n") else {}),
            **({"pathologic_m": acrin_summary.get("pathologic_m")} if acrin_summary.get("pathologic_m") else {}),
        },
        "imaging_features": _collect_prefixed_labels(label_records, ("VOLUME_", "SPHERICITY_", "LD_", "BPE_", "FTV_", "SER Volume")),
        "survival": {
            "dfs_time_weeks": _first_label(label_records, ("DFS time (weeks)",)),
            "recurrence_type": _first_label(label_records, ("recur type",)),
            "censor": _first_label(label_records, ("censor",)),
            **({"participant_status": acrin_summary["participant_status"]} if acrin_summary.get("participant_status") else {}),
            **(
                {"death_relationship_to_breast_cancer": acrin_summary["death_relationship_to_breast_cancer"]}
                if acrin_summary.get("death_relationship_to_breast_cancer")
                else {}
            ),
            **({"non_study_breast_cancer": acrin_summary["non_study_breast_cancer"]} if acrin_summary.get("non_study_breast_cancer") else {}),
        },
        "standardized_sources": {
            "pCR": pcr_source,
        },
        "source_label_count": _source_label_count(label_records, default=len(label_records)),
        "source_files": sorted({record.get("source_file", "") for record in label_records if record.get("source_file", "")}),
    }
    if tcga_summary:
        molecular_subtype = _first_present(tcga_summary, ("molecular_subtype",))
        age = _first_present(tcga_summary, ("age",))
        sex = _first_present(tcga_summary, ("sex",))
        pathologic_stage = _first_present(tcga_summary, ("pathologic_stage",))
        treatment = _first_present(tcga_summary, ("treatment",))
        if _known_label_value(molecular_subtype):
            payload["molecular_subtype"] = molecular_subtype
        if _known_label_value(age):
            payload["age"] = age
        if _known_label_value(sex):
            payload["sex"] = sex
        if _known_label_value(pathologic_stage):
            payload["pathologic_stage"] = pathologic_stage
        if _known_label_value(treatment):
            payload["treatment"] = treatment
        if tcga_pathology:
            pathology = dict(payload.get("pathology", {}))
            pathology.update({key: value for key, value in tcga_pathology.items() if _known_label_value(value)})
            payload["pathology"] = pathology
        if tcga_survival:
            survival = dict(payload.get("survival", {}))
            survival.update({key: value for key, value in tcga_survival.items() if _known_label_value(value)})
            payload["survival"] = survival
        tcga_sample = _first_present(tcga_summary, ("sample",))
        if isinstance(tcga_sample, dict) and _known_label_value(tcga_sample):
            payload["sample"] = tcga_sample
        payload["tcga_core_labels"] = tcga_summary
    if is_acrin:
        payload["acrin_core_labels"] = _compact_acrin_core_labels(payload)
    else:
        payload["raw_label_records"] = label_records
    return _json_safe(payload)


PCR_ALIASES = (
    "是否PCR",
    "pCR",
    "PCR",
    "Pathologic Response to Neoadjuvant Therapy",
    "pathologic_complete_response",
)

RESPONSE_ALIASES = (
    "新辅后病理评估",
    "Clinical response",
    "Pathologic Response to Neoadjuvant Therapy",
    "response",
    "rec",
)


def _public_label_fields(labels: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pCR",
        "pcr_status",
        "HER2",
        "ER",
        "PR",
        "molecular_subtype",
        "age",
        "BMI",
        "sex",
        "laterality",
        "clinical_stage",
        "pathologic_stage",
        "tumor_size_cm",
        "treatment",
        "response",
    ]
    return {key: labels.get(key) for key in keys}


def _source_label_count(label_records: list[dict[str, Any]], default: int) -> int:
    for record in label_records:
        labels = record.get("labels", {})
        try:
            return int(float(labels.get("source_label_count")))
        except (TypeError, ValueError):
            continue
    return default


def _normalized_pcr(label_records: list[dict[str, Any]]) -> tuple[int | None, list[str]]:
    value, source = _first_label_with_source(label_records, PCR_ALIASES)
    if value in (None, ""):
        return None, []
    text = str(value).strip().lower()
    if text in {"是", "1", "1.0", "true", "yes", "pcr", "pcr yes", "complete response", "pathologic complete response"}:
        return 1, source
    if text in {"否", "0", "0.0", "false", "no", "non-pcr", "non pcr", "not pcr"}:
        return 0, source
    return None, source


def _normalized_receptor(value: Any) -> Any:
    if value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"阳性", "positive", "pos", "1", "1.0", "+", "3+"}:
        return "positive"
    if text in {"阴性", "negative", "neg", "0", "0.0", "-"}:
        return "negative"
    if any(token in text for token in ("不确定", "equivocal", "2+")):
        return "equivocal"
    return value


def _normalized_laterality(value: Any) -> Any:
    if value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "bilateral" in text:
        return "bilateral"
    return value


def _collect_prefixed_labels(label_records: list[dict[str, Any]], prefixes: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    lowered_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for record in label_records:
        for key, value in record.get("labels", {}).items():
            key_text = str(key)
            if key_text.lower().startswith(lowered_prefixes) and value not in (None, ""):
                output[key_text] = value
    return output


def _compact_acrin_core_labels(labels: dict[str, Any]) -> dict[str, Any]:
    core_keys = (
        "pCR",
        "pcr_status",
        "HER2",
        "ER",
        "PR",
        "molecular_subtype",
        "age",
        "sex",
        "laterality",
        "clinical_stage",
        "pathologic_stage",
        "tumor_size_cm",
        "response",
    )
    output = {key: labels.get(key) for key in core_keys if _known_label_value(labels.get(key))}
    pathology = {
        key: value
        for key, value in labels.get("pathology", {}).items()
        if _known_label_value(value)
    }
    if pathology:
        output["pathology"] = pathology
    survival = {
        key: value
        for key, value in labels.get("survival", {}).items()
        if _known_label_value(value)
    }
    if survival:
        output["survival"] = survival
    imaging_features = labels.get("imaging_features", {})
    if _known_label_value(imaging_features):
        output["imaging_features"] = imaging_features
    return output


def _known_label_value(value: Any) -> bool:
    if value in (None, "", "unknown"):
        return False
    if isinstance(value, dict):
        return any(_known_label_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_known_label_value(item) for item in value)
    return True


def _first_present(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> Any:
    value, _ = _first_label_with_source(label_records, aliases)
    return value


def _first_label_with_source(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> tuple[Any, list[str]]:
    for record in label_records:
        labels = record.get("labels", {})
        lowered = {str(key).lower(): key for key in labels}
        for alias in aliases:
            key = alias if alias in labels else lowered.get(alias.lower())
            if key is None:
                continue
            value = labels.get(key)
            if value not in (None, ""):
                return value, [str(key)]
    return None, []


def _value_or_unknown(value: Any) -> Any:
    return "unknown" if value in (None, "") else value


def _load_masks(mask_csv: Path) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not mask_csv.exists():
        return by_group, by_subject
    with mask_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mask_type") == "wsi_mask":
                continue
            item = dict(row)
            item["conversion_status"] = "metadata_only"
            by_group[(row.get("dataset_id", ""), row.get("subject_id", ""), row.get("study_uid", ""))].append(item)
            by_subject[(row.get("dataset_id", ""), row.get("subject_id", ""))].append(item)
    return by_group, by_subject


def _mask_records_for_group(
    group_key: tuple[str, str, str, str],
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]],
    by_subject: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dataset_id, subject_id, _, study_uid = group_key
    exact = by_group.get((dataset_id, subject_id, study_uid), [])
    if exact:
        return exact
    return by_subject.get((dataset_id, subject_id), [])


def _masks_payload(group_key: tuple[str, str, str, str], masks: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_id, subject_id, timepoint, study_uid = group_key
    return {
        "schema_version": PREPROCESSING_VERSION,
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "timepoint": timepoint,
        "study_uid": study_uid,
        "conversion_note": "WSI masks are excluded; MRI masks/SEG are retained as metadata unless conversion is verified.",
        "masks": masks,
    }


def _flatten_mask_rows(
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]],
    by_subject: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for group_rows in list(by_group.values()) + list(by_subject.values()):
        for row in group_rows:
            key = (
                row.get("dataset_id", ""),
                row.get("subject_id", ""),
                row.get("study_uid", ""),
                row.get("series_uid", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _mask_csv_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = [
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
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _dataset_qc_rows(timepoint_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        images_by_dataset[row.get("dataset_id", "")].append(row)
    timepoints_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timepoint_rows:
        timepoints_by_dataset[row.get("dataset_id", "")].append(row)
    output: list[dict[str, Any]] = []
    for dataset_id, rows in sorted(timepoints_by_dataset.items()):
        images = images_by_dataset.get(dataset_id, [])
        role_counts = Counter(row.get("series_role", "") for row in images)
        output.append(
            {
                "dataset_id": dataset_id,
                "timepoint_rows": len(rows),
                "training_rows": sum(row.get("pcr_status") in {"0", "1"} for row in rows),
                "unique_subjects": len({row.get("patient_uid", "") for row in rows}),
                "image_files": len(images),
                "dce_original_files": sum(str(row.get("file_name", "")).startswith("original_") for row in images),
                "t2_files": role_counts.get("T2", 0),
                "dwi_files": role_counts.get("DWI", 0),
                "adc_files": role_counts.get("ADC", 0),
                "derived_files": role_counts.get("derived", 0) + role_counts.get("DCE_PE", 0) + role_counts.get("DCE_SER", 0),
                "with_mask_metadata": sum(int(row.get("mask_metadata_count", 0) or 0) > 0 for row in rows),
                "pcr_positive": sum(row.get("pcr_status") == "1" for row in rows),
                "pcr_negative": sum(row.get("pcr_status") == "0" for row in rows),
                "pcr_unknown": sum(row.get("pcr_status") not in {"0", "1"} for row in rows),
            }
        )
    return output


def _dataset_qc_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields or ["dataset_id"]


def _write_patient_labels_if_needed(patient_dir: Path, labels_payload: dict[str, Any], written: set[Path]) -> None:
    if patient_dir in written:
        return
    patient_payload = dict(labels_payload)
    patient_payload["timepoint"] = "patient_level"
    patient_payload["study_uid"] = "patient_level"
    _write_json_if_changed(patient_dir / "labels.json", patient_payload)
    written.add(patient_dir)


def _patient_labels_payload(labels_payload: dict[str, Any]) -> dict[str, Any]:
    patient_payload = dict(labels_payload)
    patient_payload["timepoint"] = "patient_level"
    patient_payload["study_uid"] = "patient_level"
    return patient_payload


def _clear_sample_outputs(sample_dir: Path) -> None:
    if not sample_dir.exists():
        return
    for pattern in ("*.npy", "series_manifest.json", "labels.json", "masks_manifest.json"):
        for path in sample_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    for dirname in ("DCE", "T1", "T2", "DWI", "ADC", "parametric_maps", "masks", "slice"):
        target = sample_dir / dirname
        if target.exists() and target.is_dir():
            shutil.rmtree(target)


def _failure(record: dict[str, str], stage: str, error: str) -> dict[str, str]:
    return {
        "dataset_id": record.get("dataset_id", ""),
        "subject_id": record.get("subject_id", ""),
        "timepoint": record.get("timepoint", ""),
        "study_uid": record.get("study_uid", ""),
        "series_uid": record.get("series_uid", ""),
        "series_role": record.get("series_role", ""),
        "source_path": record.get("source_path", ""),
        "stage": stage,
        "error": error,
    }


def _split_for_subject(dataset_id: str, subject_id: str, seed: int = 2026) -> str:
    digest = hashlib.sha256(f"{seed}|{dataset_id}|{subject_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_is_current(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("schema_version", "")) == PREPROCESSING_VERSION


def _read_json_object_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_if_changed(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_json(path, data)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:8]


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return cleaned.strip("._") or "unknown"


def add_multimodal_parser(preprocess_subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = preprocess_subparsers.add_parser(
        "build-multimodal",
        help="Build dataset/subject/timepoint folders with multi-modal MRI arrays, labels and mask metadata.",
    )
    parser.add_argument("--output-dir", default="/home/ubuntu/liuyiyao1/multimodal_dataset/full")
    parser.add_argument("--plan-dir")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(160, 160, 96), metavar=("X", "Y", "Z"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="2 is disk-friendly; 0 means auto; use 1 for serial debugging.")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--write-previews", dest="write_previews", action="store_true")
    preview_group.add_argument("--no-previews", dest="write_previews", action="store_false")
    parser.set_defaults(write_previews=False)
    parser.set_defaults(func=run_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    summary = build_multimodal_dataset(
        output_dir=args.output_dir,
        plan_dir=args.plan_dir,
        max_patients=args.max_patients,
        limit=args.limit,
        target_shape=tuple(args.target_shape),
        overwrite=args.overwrite,
        workers=resolve_preprocessing_workers(args.workers),
        write_previews=args.write_previews,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
