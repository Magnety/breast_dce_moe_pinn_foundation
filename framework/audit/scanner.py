from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.config import AuditRunConfig
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.dicom_metadata import (
    DicomReadError,
    hash_identifier,
    pydicom_available,
    read_dicom_header,
    sanitize_header,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.file_types import classify_file, file_extension, is_archive_file, is_dicom_candidate
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.labels import analyze_label_file
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.modality import infer_modality
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.reports import write_dataset_reports, write_run_index
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.logging import get_logger
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_counter, progress_iter, suppress_warnings
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy, is_relative_to

LOGGER = get_logger(__name__)


class AuditScanner:
    """Read-only scanner for raw breast MRI dataset roots."""

    def __init__(self, settings: AuditRunConfig, policy: PathPolicy):
        self.settings = settings
        self.policy = policy

    def discover_datasets(self) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for raw_root in self.settings.raw_roots:
            if not raw_root.exists():
                datasets.append(
                    {
                        "dataset_id": _dataset_id(raw_root, raw_root),
                        "name": raw_root.name,
                        "raw_root": str(raw_root),
                        "path": str(raw_root),
                        "status": "missing_raw_root",
                        "is_dataset_dir": False,
                    }
                )
                continue
            self.policy.assert_read_allowed(raw_root)
            child_dirs: list[Path] = []
            root_files: list[Path] = []
            try:
                with os.scandir(raw_root) as iterator:
                    for entry in iterator:
                        entry_path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            child_dirs.append(entry_path)
                        elif entry.is_file(follow_symlinks=False):
                            root_files.append(entry_path)
            except OSError as exc:
                datasets.append(
                    {
                        "dataset_id": _dataset_id(raw_root, raw_root),
                        "name": raw_root.name,
                        "raw_root": str(raw_root),
                        "path": str(raw_root),
                        "status": "needs_review_read_error",
                        "error": str(exc),
                        "is_dataset_dir": False,
                    }
                )
                continue

            if child_dirs:
                for child in sorted(child_dirs, key=lambda item: item.name.lower()):
                    datasets.append(
                        {
                            "dataset_id": _dataset_id(raw_root, child),
                            "name": child.name,
                            "raw_root": str(raw_root),
                            "path": str(child),
                            "status": "ready",
                            "is_dataset_dir": True,
                        }
                    )
            elif root_files:
                datasets.append(
                    {
                        "dataset_id": _dataset_id(raw_root, raw_root),
                        "name": raw_root.name,
                        "raw_root": str(raw_root),
                        "path": str(raw_root),
                        "status": "ready",
                        "is_dataset_dir": True,
                    }
                )
            for root_file in sorted(root_files, key=lambda item: item.name.lower()):
                if classify_file(root_file) == "archive":
                    if (
                        self.settings.skip_archive_files
                        and not self.settings.include_top_level_archives_as_datasets
                    ):
                        LOGGER.info(
                            "Skipping top-level archive by policy because archives are already extracted: %s",
                            root_file,
                        )
                        continue
                    if self.settings.skip_archives_with_extracted_dirs and _has_matching_extracted_dir(
                        root_file, child_dirs
                    ):
                        LOGGER.info(
                            "Skipping archive because matching extracted folder exists: %s",
                            root_file,
                        )
                        continue
                    datasets.append(
                        {
                            "dataset_id": _dataset_id(raw_root, root_file),
                            "name": root_file.name,
                            "raw_root": str(raw_root),
                            "path": str(root_file),
                            "status": "archive_not_expanded",
                            "is_dataset_dir": False,
                        }
                    )
        return datasets

    def run(self, run_id: str | None = None, dataset_names: set[str] | None = None) -> dict[str, Any]:
        suppress_warnings()
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.policy.ensure_output_dir(self.settings.output_root / "audit" / run_id)
        datasets = self.discover_datasets()

        selected_results: list[dict[str, Any]] = []
        selected_datasets = [
            dataset
            for dataset in datasets
            if not dataset_names or dataset["name"] in dataset_names or dataset["dataset_id"] in dataset_names
        ]
        for dataset in progress_iter(
            selected_datasets,
            total=len(selected_datasets),
            desc="Audit datasets",
            unit="dataset",
        ):
            LOGGER.info("Auditing dataset %s", dataset["dataset_id"])
            result = self.scan_dataset(dataset)
            selected_results.append(result)
            write_dataset_reports(result, run_dir / "datasets" / result["dataset_id"], self.policy)

        run_result = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(run_dir),
            "raw_roots": [str(path) for path in self.settings.raw_roots],
            "dataset_count": len(selected_results),
            "datasets": [
                {
                    "dataset_id": result["dataset_id"],
                    "name": result["name"],
                    "status": result["status"],
                    "report_dir": str(run_dir / "datasets" / result["dataset_id"]),
                    "file_count": result["inventory"]["file_count"],
                    "dicom_header_count": result["dicom"]["headers_read"],
                    "quality_issue_count": len(result["quality_issues"]),
                }
                for result in selected_results
            ],
        }
        write_run_index(run_result, run_dir, self.policy)
        return run_result

    def scan_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        path = Path(dataset["path"])
        if path.is_file():
            return self._archive_result(dataset, path)
        if dataset["status"] != "ready":
            return self._non_scanned_result(dataset, path)
        self.policy.assert_read_allowed(path)

        inventory = {
            "file_count": 0,
            "directory_count": 0,
            "total_bytes": 0,
            "type_counts": Counter(),
            "extension_counts": Counter(),
            "truncated": False,
            "scan_limit": self.settings.max_files_per_dataset,
            "skipped_archive_count": 0,
        }
        tree_entries: list[dict[str, Any]] = []
        quality_issues: list[dict[str, Any]] = []
        unrecognized_items: list[dict[str, Any]] = []
        label_candidates: list[Path] = []
        skipped_archives: list[dict[str, Any]] = []
        series_map: dict[str, dict[str, Any]] = {}
        patient_mapping: dict[str, str] = {}
        dicom_headers_read = 0
        dicom_read_errors = 0
        pydicom_ready = pydicom_available()
        if not pydicom_ready:
            quality_issues.append(_issue("error", "dependency", "pydicom is not installed; DICOM headers were not read"))

        stack: list[tuple[Path, int]] = [(path, 0)]
        with progress_counter(desc=f"Scan {dataset['dataset_id']}", unit="file") as scan_progress:
            while stack:
                current, depth = stack.pop()
                try:
                    entries = sorted(list(os.scandir(current)), key=lambda item: item.name.lower())
                except OSError as exc:
                    quality_issues.append(
                        _issue("warning", "read_error", f"Could not scan directory: {exc}", relative_path=_rel(current, path))
                    )
                    continue
                inventory["directory_count"] += 1
                if depth <= self.settings.tree_max_depth:
                    tree_entries.append(
                        {
                            "relative_path": _rel(current, path),
                            "depth": depth,
                            "child_count": len(entries),
                            "children_sample": [entry.name for entry in entries[: self.settings.tree_max_entries_per_dir]],
                            "truncated": len(entries) > self.settings.tree_max_entries_per_dir,
                        }
                    )
                for entry in reversed(entries):
                    entry_path = Path(entry.path)
                    if entry.is_symlink() and not self.settings.follow_symlinks:
                        quality_issues.append(
                            _issue("info", "skipped_symlink", "Symlink skipped", relative_path=_rel(entry_path, path))
                        )
                        continue
                    if entry.is_dir(follow_symlinks=self.settings.follow_symlinks):
                        stack.append((entry_path, depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=self.settings.follow_symlinks):
                        continue
                    inventory["file_count"] += 1
                    scan_progress.update()
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        size = 0
                    inventory["total_bytes"] += size
                    kind = classify_file(entry_path)
                    extension = file_extension(entry_path)
                    inventory["type_counts"][kind] += 1
                    inventory["extension_counts"][extension] += 1
                    if self.settings.skip_archive_files and is_archive_file(
                        entry_path, self.settings.archive_extensions
                    ):
                        inventory["skipped_archive_count"] += 1
                        if len(skipped_archives) < 1000:
                            skipped_archives.append(
                                {
                                    "relative_path": _rel(entry_path, path),
                                    "size_bytes": size,
                                    "reason": "archive_already_extracted",
                                }
                            )
                        continue
                    if extension in self.settings.label_extensions:
                        label_candidates.append(entry_path)
                    if pydicom_ready and is_dicom_candidate(entry_path, self.settings.dicom_candidate_extensions):
                        if (
                            self.settings.max_dicom_headers_per_dataset is None
                            or dicom_headers_read < self.settings.max_dicom_headers_per_dataset
                        ):
                            try:
                                header = read_dicom_header(entry_path)
                            except DicomReadError:
                                dicom_read_errors += 1
                            else:
                                dicom_headers_read += 1
                                inventory["type_counts"]["dicom"] += 1 if kind != "dicom" else 0
                                self._record_dicom_header(
                                    series_map=series_map,
                                    header=header,
                                    file_path=entry_path,
                                    dataset_root=path,
                                    patient_mapping=patient_mapping,
                                )
                    if (
                        self.settings.max_files_per_dataset is not None
                        and inventory["file_count"] >= self.settings.max_files_per_dataset
                    ):
                        inventory["truncated"] = True
                        stack.clear()
                        break

        if dicom_read_errors:
            quality_issues.append(
                _issue(
                    "info",
                    "dicom_candidate_read_errors",
                    f"{dicom_read_errors} candidate files were not readable as DICOM headers",
                )
            )

        series_rows, patient_rows, modality_rows = self._finalize_dicom_tables(series_map, quality_issues)
        label_reports = []
        selected_label_candidates = label_candidates[:200]
        for label_path in progress_iter(
            selected_label_candidates,
            total=len(selected_label_candidates),
            desc=f"Analyze labels {dataset['dataset_id']}",
            unit="file",
        ):
            report = analyze_label_file(label_path, max_size_mb=self.settings.max_label_file_size_mb)
            report["relative_path"] = _rel(label_path, path)
            label_reports.append(report)
            if report.get("status") != "ok":
                unrecognized_items.append(
                    {
                        "category": "label_file",
                        "relative_path": _rel(label_path, path),
                        "status": report.get("status", "needs_review"),
                    }
                )
        if len(label_candidates) > 200:
            quality_issues.append(
                _issue("warning", "label_limit", "More than 200 label-like files found; only first 200 analyzed")
            )

        for series in series_rows:
            if series["modality_status"] != "identified":
                unrecognized_items.append(
                    {
                        "category": "modality",
                        "series_uid": series["series_uid"],
                        "status": series["modality_status"],
                        "message": "Modality requires review",
                    }
                )

        result = {
            "dataset_id": dataset["dataset_id"],
            "name": dataset["name"],
            "raw_root": dataset["raw_root"],
            "path": str(path),
            "status": "ok" if not inventory["truncated"] else "partial_scan",
            "audit_limits": {
                "max_files_per_dataset": self.settings.max_files_per_dataset,
                "max_dicom_headers_per_dataset": self.settings.max_dicom_headers_per_dataset,
            },
            "inventory": _counter_to_plain(inventory),
            "directory_tree": tree_entries,
            "dicom": {
                "headers_read": dicom_headers_read,
                "series_count": len(series_rows),
                "patient_count": len(patient_rows),
                "series_statistics": series_rows,
                "patient_statistics": patient_rows,
                "modality_inference": modality_rows,
            },
            "labels": label_reports,
            "skipped_archives": skipped_archives,
            "quality_issues": quality_issues,
            "unrecognized_items": unrecognized_items,
            "preprocessing_recommendations": _preprocessing_recommendations(series_rows, quality_issues),
        }
        if self.settings.save_patient_mapping:
            result["patient_mapping_secure"] = [
                {"patient_uid": uid, "raw_patient_id": raw_id} for raw_id, uid in sorted(patient_mapping.items())
            ]
        return result

    def _record_dicom_header(
        self,
        series_map: dict[str, dict[str, Any]],
        header: dict[str, Any],
        file_path: Path,
        dataset_root: Path,
        patient_mapping: dict[str, str],
    ) -> None:
        series_uid = str(header.get("SeriesInstanceUID") or f"path:{hash_identifier(file_path.parent, self.settings.anonymization_salt)}")
        study_uid = str(header.get("StudyInstanceUID") or "unknown")
        raw_patient_id = str(header.get("PatientID") or f"missing_patient_id:{study_uid}:{series_uid}")
        patient_uid = hash_identifier(raw_patient_id, self.settings.anonymization_salt)
        patient_mapping.setdefault(raw_patient_id, patient_uid)

        item = series_map.setdefault(
            series_uid,
            {
                "series_uid": series_uid,
                "study_uid": study_uid,
                "patient_uid": patient_uid,
                "headers_sampled": 0,
                "relative_paths_sampled": [],
                "values": defaultdict(Counter),
                "instance_numbers": [],
                "temporal_positions": [],
                "safe_header": {},
            },
        )
        item["headers_sampled"] += 1
        if len(item["relative_paths_sampled"]) < 5:
            item["relative_paths_sampled"].append(_rel(file_path, dataset_root))
        item["safe_header"] = sanitize_header(header)
        for key, value in sanitize_header(header).items():
            item["values"][key][_hashable_value(value)] += 1
        instance_number = _as_int(header.get("InstanceNumber"))
        if instance_number is not None:
            item["instance_numbers"].append(instance_number)
        temporal_position = _as_int(header.get("TemporalPositionIdentifier"))
        if temporal_position is not None:
            item["temporal_positions"].append(temporal_position)

    def _finalize_dicom_tables(
        self,
        series_map: dict[str, dict[str, Any]],
        quality_issues: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        series_rows: list[dict[str, Any]] = []
        modality_rows: list[dict[str, Any]] = []
        patient_accumulator: dict[str, dict[str, Any]] = {}
        series_items = list(series_map.values())
        for item in progress_iter(
            series_items,
            total=len(series_items),
            desc="Finalize DICOM",
            unit="series",
        ):
            values: dict[str, Counter[str]] = item["values"]
            header = {key: _best(counter) for key, counter in values.items()}
            image_count = int(item["headers_sampled"])
            modality = infer_modality(header, image_count=image_count)
            dimensions = sorted(
                {
                    f"{_best(values.get('Rows', Counter()))}x{_best(values.get('Columns', Counter()))}"
                    for _ in [0]
                    if values.get("Rows") and values.get("Columns")
                }
            )
            spacings = sorted(set(values.get("PixelSpacing", Counter()).keys()))
            orientations = sorted(set(values.get("ImageOrientationPatient", Counter()).keys()))
            instance_numbers = item["instance_numbers"]
            duplicate_instances = len(instance_numbers) - len(set(instance_numbers))
            missing_instances = _missing_instance_count(instance_numbers)
            if len(spacings) > 1:
                quality_issues.append(
                    _issue(
                        "warning",
                        "inconsistent_spacing",
                        "Multiple PixelSpacing values in one series",
                        patient_uid=item["patient_uid"],
                        study_uid=item["study_uid"],
                        series_uid=item["series_uid"],
                    )
                )
            if len(orientations) > 1:
                quality_issues.append(
                    _issue(
                        "warning",
                        "inconsistent_orientation",
                        "Multiple ImageOrientationPatient values in one series",
                        patient_uid=item["patient_uid"],
                        study_uid=item["study_uid"],
                        series_uid=item["series_uid"],
                    )
                )
            if duplicate_instances > 0:
                quality_issues.append(
                    _issue(
                        "warning",
                        "duplicate_instance_numbers",
                        f"{duplicate_instances} duplicate InstanceNumber values",
                        patient_uid=item["patient_uid"],
                        study_uid=item["study_uid"],
                        series_uid=item["series_uid"],
                    )
                )
            if missing_instances > 0:
                quality_issues.append(
                    _issue(
                        "warning",
                        "missing_instance_numbers",
                        f"{missing_instances} missing InstanceNumber values in sampled range",
                        patient_uid=item["patient_uid"],
                        study_uid=item["study_uid"],
                        series_uid=item["series_uid"],
                    )
                )

            row = {
                "patient_uid": item["patient_uid"],
                "study_uid": item["study_uid"],
                "series_uid": item["series_uid"],
                "image_count_sampled": image_count,
                "series_description": header.get("SeriesDescription", ""),
                "protocol_name": header.get("ProtocolName", ""),
                "sequence_name": header.get("SequenceName", ""),
                "modality_dicom": header.get("Modality", ""),
                "manufacturer": header.get("Manufacturer", ""),
                "magnetic_field_strength": header.get("MagneticFieldStrength", ""),
                "dimensions": dimensions,
                "pixel_spacing_values": spacings,
                "slice_thickness": header.get("SliceThickness", ""),
                "spacing_between_slices": header.get("SpacingBetweenSlices", ""),
                "orientation_count": len(orientations),
                "temporal_position_count": len(set(item["temporal_positions"])),
                "instance_min": min(instance_numbers) if instance_numbers else "",
                "instance_max": max(instance_numbers) if instance_numbers else "",
                "duplicate_instance_count": duplicate_instances,
                "missing_instance_count": missing_instances,
                "example_relative_paths": item["relative_paths_sampled"],
                "inferred_modality": modality["primary"],
                "modality_status": modality["status"],
                "modality_confidence": modality["confidence"],
                "modality_evidence": modality["evidence"],
                "modality_flags": modality["flags"],
            }
            series_rows.append(row)
            modality_rows.append(
                {
                    "patient_uid": row["patient_uid"],
                    "study_uid": row["study_uid"],
                    "series_uid": row["series_uid"],
                    "inferred_modality": row["inferred_modality"],
                    "status": row["modality_status"],
                    "confidence": row["modality_confidence"],
                    "evidence": row["modality_evidence"],
                    "flags": row["modality_flags"],
                }
            )
            patient = patient_accumulator.setdefault(
                item["patient_uid"],
                {
                    "patient_uid": item["patient_uid"],
                    "study_uids": set(),
                    "series_count": 0,
                    "image_count_sampled": 0,
                    "modalities": set(),
                    "needs_review_series_count": 0,
                },
            )
            patient["study_uids"].add(item["study_uid"])
            patient["series_count"] += 1
            patient["image_count_sampled"] += image_count
            patient["modalities"].add(modality["primary"])
            if modality["status"] != "identified":
                patient["needs_review_series_count"] += 1

        patient_rows = []
        for patient in patient_accumulator.values():
            patient_rows.append(
                {
                    "patient_uid": patient["patient_uid"],
                    "study_count": len(patient["study_uids"]),
                    "series_count": patient["series_count"],
                    "image_count_sampled": patient["image_count_sampled"],
                    "modalities": sorted(patient["modalities"]),
                    "needs_review_series_count": patient["needs_review_series_count"],
                }
            )
        return (
            sorted(series_rows, key=lambda row: (row["patient_uid"], row["study_uid"], row["series_uid"])),
            sorted(patient_rows, key=lambda row: row["patient_uid"]),
            sorted(modality_rows, key=lambda row: (row["patient_uid"], row["series_uid"])),
        )

    def _non_scanned_result(self, dataset: dict[str, Any], path: Path) -> dict[str, Any]:
        return {
            "dataset_id": dataset["dataset_id"],
            "name": dataset["name"],
            "raw_root": dataset["raw_root"],
            "path": str(path),
            "status": dataset["status"],
            "inventory": {
                "file_count": 0,
                "directory_count": 0,
                "total_bytes": 0,
                "type_counts": {},
                "extension_counts": {},
                "truncated": False,
                "skipped_archive_count": 0,
            },
            "directory_tree": [],
            "dicom": {
                "headers_read": 0,
                "series_count": 0,
                "patient_count": 0,
                "series_statistics": [],
                "patient_statistics": [],
                "modality_inference": [],
            },
            "labels": [],
            "skipped_archives": [],
            "quality_issues": [_issue("info", dataset["status"], dataset.get("error", dataset["status"]))],
            "unrecognized_items": [],
            "preprocessing_recommendations": [],
        }

    def _archive_result(self, dataset: dict[str, Any], path: Path) -> dict[str, Any]:
        size = path.stat().st_size if path.exists() else 0
        return {
            "dataset_id": dataset["dataset_id"],
            "name": dataset["name"],
            "raw_root": dataset["raw_root"],
            "path": str(path),
            "status": "archive_not_scanned",
            "inventory": {
                "file_count": 1,
                "directory_count": 0,
                "total_bytes": size,
                "type_counts": {"archive": 1},
                "extension_counts": {file_extension(path): 1},
                "truncated": False,
                "skipped_archive_count": 1,
            },
            "directory_tree": [],
            "dicom": {
                "headers_read": 0,
                "series_count": 0,
                "patient_count": 0,
                "series_statistics": [],
                "patient_statistics": [],
                "modality_inference": [],
            },
            "labels": [],
            "skipped_archives": [
                {
                    "relative_path": path.name,
                    "size_bytes": size,
                    "reason": "archive_not_scanned",
                }
            ],
            "quality_issues": [
                _issue(
                    "info",
                    "archive_not_scanned",
                    "Archive file was inventoried but not opened; use the extracted directory for audit",
                    relative_path=path.name,
                )
            ],
            "unrecognized_items": [],
            "preprocessing_recommendations": ["Use the already extracted matching directory when available."],
        }


def _dataset_id(raw_root: Path, dataset_path: Path) -> str:
    drive = raw_root.drive.replace(":", "") or "root"
    try:
        rel = dataset_path.relative_to(raw_root)
    except ValueError:
        rel = Path(dataset_path.name)
    text = "__".join((drive, raw_root.name, *rel.parts))
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:120] or "dataset"


def _has_matching_extracted_dir(archive_path: Path, child_dirs: list[Path]) -> bool:
    archive_stems = _archive_stem_candidates(archive_path)
    child_names = {child.name.lower() for child in child_dirs}
    return any(stem.lower() in child_names for stem in archive_stems)


def _archive_stem_candidates(path: Path) -> set[str]:
    lower_name = path.name.lower()
    stems = {path.stem}
    for extension in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"):
        if lower_name.endswith(extension):
            stems.add(path.name[: -len(extension)])
    return {stem for stem in stems if stem}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _hashable_value(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return "" if value is None else str(value)


def _best(counter: Counter[str] | None) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _missing_instance_count(values: list[int]) -> int:
    if not values:
        return 0
    unique = sorted(set(values))
    expected = unique[-1] - unique[0] + 1
    return max(expected - len(unique), 0)


def _issue(
    severity: str,
    category: str,
    message: str,
    patient_uid: str = "",
    study_uid: str = "",
    series_uid: str = "",
    relative_path: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "message": message,
        "patient_uid": patient_uid,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "relative_path": relative_path,
        "status": "open",
    }


def _counter_to_plain(inventory: dict[str, Any]) -> dict[str, Any]:
    plain = dict(inventory)
    plain["type_counts"] = dict(inventory["type_counts"])
    plain["extension_counts"] = dict(inventory["extension_counts"])
    return plain


def _preprocessing_recommendations(series_rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    modalities = {row["inferred_modality"] for row in series_rows}
    if not series_rows:
        recommendations.append("No readable DICOM series were sampled; verify extraction and pydicom support.")
    if "DCE" in modalities:
        recommendations.append("Review DCE phase ordering and injection metadata before deriving enhancement maps.")
    if any(issue["category"] in {"inconsistent_spacing", "inconsistent_orientation"} for issue in issues):
        recommendations.append("Run orientation and spacing harmonization checks before conversion.")
    if any(row["modality_status"] != "identified" for row in series_rows):
        recommendations.append("Resolve needs_review/unknown modality mappings before preprocessing.")
    if not recommendations:
        recommendations.append("Proceed with dataset-specific adapter design after reviewing labels and QC tables.")
    return recommendations
