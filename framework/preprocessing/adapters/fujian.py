from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.dicom_metadata import DicomReadError, read_dicom_header
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import (
    AdapterResult,
    CaseRecord,
    DatasetAdapter,
    LabelRecord,
    MaskRecord,
    SeriesRecord,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.classification import classify_series_role, infer_timepoint, is_mask_role
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings


class FujianPCRAdapter(DatasetAdapter):
    dataset_id = "fujian_pCR"

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        self.mri_root = self.dataset_root / "pCR-MRI" / "001-493"
        self.label_file = _find_first_excel(self.dataset_root)

    def scan(self, max_patients: int | None = None) -> AdapterResult:
        suppress_warnings()
        result = AdapterResult(dataset_id=self.dataset_id, source_root=self.dataset_root)
        labels = self._load_labels(result)
        patient_dirs = sorted(
            [path for path in self.mri_root.iterdir() if path.is_dir() and path.name.isdigit()],
            key=lambda p: p.name,
        )
        if max_patients is not None:
            patient_dirs = patient_dirs[:max_patients]

        for patient_dir in progress_iter(
            patient_dirs,
            total=len(patient_dirs),
            desc=f"Scan {self.dataset_id}",
            unit="patient",
        ):
            subject_id = patient_dir.name
            patient_series = self._scan_patient_series(subject_id, patient_dir, result)
            roles = {series.series_role for series in patient_series}
            mask_count = sum(1 for series in patient_series if is_mask_role(series.series_role))
            study_uids = {series.study_uid for series in patient_series}
            result.cases.append(
                CaseRecord(
                    dataset_id=self.dataset_id,
                    subject_id=subject_id,
                    study_count=len(study_uids),
                    series_count=len(patient_series),
                    mask_count=mask_count,
                    available_roles=tuple(sorted(roles)),
                    has_label=subject_id in labels,
                    source_root=str(self.dataset_root),
                    notes="contains_WSI_masks" if _find_named_dir(patient_dir, "HE切片").exists() else "",
                )
            )
        return result

    def _scan_patient_series(
        self,
        subject_id: str,
        patient_dir: Path,
        result: AdapterResult,
    ) -> list[SeriesRecord]:
        records: list[SeriesRecord] = []
        mri_dir = _find_named_dir(patient_dir, "MRI")
        if not mri_dir.exists():
            result.issues.append(_issue("warning", "missing_mri_dir", str(patient_dir)))
            return records
        timepoint_dirs = sorted([path for path in mri_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
        for timepoint_dir in progress_iter(
            timepoint_dirs,
            total=len(timepoint_dirs),
            desc=f"Scan {subject_id}",
            unit="timepoint",
        ):
            grouped = self._group_timepoint_series(timepoint_dir, result)
            for series_uid, item in grouped.items():
                description = item["description"]
                role = classify_series_role(
                    description,
                    modality=item["modality"],
                    sop_class_name=item["sop"],
                    dataset_id=self.dataset_id,
                )
                record = SeriesRecord(
                    dataset_id=self.dataset_id,
                    subject_id=subject_id,
                    study_uid=item["study_uid"] or f"{subject_id}:{timepoint_dir.name}",
                    study_date=item["study_date"],
                    series_uid=series_uid,
                    series_description=description,
                    modality=item["modality"],
                    series_role=role,
                    timepoint=infer_timepoint(subject_id, timepoint_dir.name),
                    image_count=int(item["count"]),
                    source_path=str(timepoint_dir),
                    relative_path=str(timepoint_dir.relative_to(self.dataset_root)),
                    sop_class_name=item["sop"],
                    local_exists=True,
                )
                records.append(record)
                result.series.append(record)
                if is_mask_role(role):
                    result.masks.append(
                        MaskRecord(
                            dataset_id=self.dataset_id,
                            subject_id=subject_id,
                            study_uid=record.study_uid,
                            study_date=record.study_date,
                            series_uid=record.series_uid,
                            mask_type=role,
                            source_path=record.source_path,
                            relative_path=record.relative_path,
                            image_count=record.image_count,
                        )
                    )
        self._record_wsi_masks(subject_id, patient_dir, result)
        return records

    def _group_timepoint_series(
        self,
        timepoint_dir: Path,
        result: AdapterResult,
    ) -> dict[str, dict[str, Any]]:
        files = _dicom_candidate_files(timepoint_dir)
        grouped: dict[str, dict[str, Any]] = {}
        for dicom_path in files:
            try:
                header = _read_fast_series_header(dicom_path)
            except DicomReadError as exc:
                result.issues.append(_issue("warning", "dicom_header_read_error", f"{dicom_path}: {exc}"))
                continue
            description = _series_display_description(header)
            series_uid = _series_group_key(header, timepoint_dir, description)
            item = grouped.setdefault(
                series_uid,
                {
                    "count": 0,
                    "description": description,
                    "study_uid": str(header.get("StudyInstanceUID") or ""),
                    "study_date": str(header.get("StudyDate") or ""),
                    "modality": str(header.get("Modality") or ""),
                    "sop": str(header.get("SOPClassUID") or ""),
                    "series_number": str(header.get("SeriesNumber") or ""),
                },
            )
            item["count"] += 1
            _merge_series_header(item, header, description)
        if files and not any(
            classify_series_role(
                item["description"],
                modality=item["modality"],
                sop_class_name=item["sop"],
                dataset_id=self.dataset_id,
            )
            in {"DCE", "DCE_PRE", "DCE_POST", "DCE_PE", "DCE_SER"}
            for item in grouped.values()
        ):
            result.issues.append(_issue("warning", "fujian_no_dce_series", str(timepoint_dir)))
        return grouped

    def _record_wsi_masks(self, subject_id: str, patient_dir: Path, result: AdapterResult) -> None:
        he_dir = _find_named_dir(patient_dir, "HE切片")
        if not he_dir.exists():
            return
        for mask_file in sorted(he_dir.rglob("*.mask")):
            result.masks.append(
                MaskRecord(
                    dataset_id=self.dataset_id,
                    subject_id=subject_id,
                    study_uid=f"{subject_id}:WSI",
                    study_date="",
                    series_uid=str(mask_file.relative_to(self.dataset_root)),
                    mask_type="wsi_mask",
                    source_path=str(mask_file),
                    relative_path=str(mask_file.relative_to(self.dataset_root)),
                    image_count=1,
                    notes="Pathology WSI mask, not MRI ROI.",
                )
            )

    def _load_labels(self, result: AdapterResult) -> dict[str, LabelRecord]:
        labels: dict[str, LabelRecord] = {}
        if not self.label_file:
            result.issues.append(_issue("warning", "missing_label_file", str(self.dataset_root)))
            return labels
        frame = pd.read_excel(self.label_file, sheet_name=0)
        label_rows = list(frame.dropna(how="all").iterrows())
        for _, row in progress_iter(
            label_rows,
            total=len(label_rows),
            desc=f"Load labels {self.dataset_id}",
            unit="row",
        ):
            raw_id = row.get("编号")
            if pd.isna(raw_id):
                continue
            subject_id = f"{int(raw_id):03d}" if isinstance(raw_id, (int, float)) else str(raw_id).strip()
            labels_dict = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
            record = LabelRecord(
                dataset_id=self.dataset_id,
                subject_id=subject_id,
                source_file=str(self.label_file),
                key_column="编号",
                labels=labels_dict,
                notes="pCR label in 是否PCR; molecular subtype in 分子分型",
            )
            labels[subject_id] = record
            result.labels.append(record)
        return labels


def _find_first_excel(root: Path) -> Path | None:
    candidates = sorted(root.glob("*.xlsx"))
    return candidates[0] if candidates else None


def _find_named_dir(root: Path, expected: str) -> Path:
    direct = root / expected
    if direct.exists():
        return direct
    expected_clean = expected.strip().lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.strip().lower() == expected_clean:
            return child
    return direct


def _dicom_candidate_files(root: Path) -> list[Path]:
    try:
        files = [path for path in root.iterdir() if path.is_file()]
    except OSError:
        return []
    preferred = [
        path
        for path in files
        if path.suffix.lower() in {".dcm", ".dicom", ".ima"} or path.suffix == ""
    ]
    return sorted(preferred or files, key=lambda path: path.name.lower())


def _series_group_key(header: dict[str, Any], timepoint_dir: Path, description: str) -> str:
    series_uid = str(header.get("SeriesInstanceUID") or "").strip()
    if series_uid:
        return series_uid
    series_number = str(header.get("SeriesNumber") or "").strip()
    acquisition_number = str(header.get("AcquisitionNumber") or "").strip()
    fallback = ":".join(part for part in (timepoint_dir.name, series_number, acquisition_number, description) if part)
    return f"path:{fallback or timepoint_dir.name}"


def _merge_series_header(item: dict[str, Any], header: dict[str, Any], description: str) -> None:
    if description and len(description) > len(str(item.get("description") or "")):
        item["description"] = description
    for item_key, header_key in (
        ("study_uid", "StudyInstanceUID"),
        ("study_date", "StudyDate"),
        ("modality", "Modality"),
        ("sop", "SOPClassUID"),
        ("series_number", "SeriesNumber"),
    ):
        if not item.get(item_key) and header.get(header_key):
            item[item_key] = str(header.get(header_key) or "")


def _series_display_description(header: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("SeriesDescription", "ProtocolName", "SequenceName"):
        value = str(header.get(key) or "").strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            if not any(value.lower() == part.lower() for part in parts):
                parts.append(value)
    return " // ".join(parts)


def _read_fast_series_header(path: Path) -> dict[str, Any]:
    try:
        import pydicom
    except ModuleNotFoundError:
        return read_dicom_header(path)
    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "SeriesInstanceUID",
                "StudyInstanceUID",
                "StudyDate",
                "Modality",
                "SOPClassUID",
                "SeriesDescription",
                "ProtocolName",
                "SequenceName",
                "SeriesNumber",
                "AcquisitionNumber",
            ],
        )
    except Exception as exc:
        raise DicomReadError(str(exc)) from exc
    if not any(hasattr(dataset, key) for key in ("SeriesInstanceUID", "StudyInstanceUID", "SOPClassUID")):
        raise DicomReadError("file does not contain required DICOM identity tags")
    return {
        "SeriesInstanceUID": str(getattr(dataset, "SeriesInstanceUID", "")),
        "StudyInstanceUID": str(getattr(dataset, "StudyInstanceUID", "")),
        "StudyDate": str(getattr(dataset, "StudyDate", "")),
        "Modality": str(getattr(dataset, "Modality", "")),
        "SOPClassUID": str(getattr(dataset, "SOPClassUID", "")),
        "SeriesDescription": str(getattr(dataset, "SeriesDescription", "")),
        "ProtocolName": str(getattr(dataset, "ProtocolName", "")),
        "SequenceName": str(getattr(dataset, "SequenceName", "")),
        "SeriesNumber": str(getattr(dataset, "SeriesNumber", "")),
        "AcquisitionNumber": str(getattr(dataset, "AcquisitionNumber", "")),
    }


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _issue(severity: str, category: str, message: str) -> dict[str, Any]:
    return {"severity": severity, "category": category, "message": message, "status": "open"}
