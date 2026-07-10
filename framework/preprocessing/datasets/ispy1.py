from __future__ import annotations

import os
import re
from collections import defaultdict
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
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.classification import (
    classify_series_role,
    infer_timepoint,
    is_mask_role,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.common import PathLike, resolve_path
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists, safe_absolute_path, windows_extended_path
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings


DATASET_ID = "ispy1"
DEFAULT_ROOT = Path("H:/breast/ISPY1")
DEFAULT_METADATA = Path("metadata/metadata.csv")
DEFAULT_LABEL_FILE = "I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx"
CLINICAL_SHEET = "TCIA Patient Clinical Subset"
OUTCOME_SHEET = "TCIA Outcomes Subset"
SUBJECT_ID_COLUMN = "SUBJECTID"
REQUIRED_METADATA_COLUMNS = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "S5cmdManifestPath",
)
ISPY1_DCE_SOURCE_ROLES = {"DCE", "DCE_PRE", "DCE_POST"}


def build_adapter(
    dataset_root: PathLike | None = None,
    metadata_csv: PathLike | None = None,
    label_root: PathLike | None = None,
) -> DatasetAdapter:
    root = resolve_path(dataset_root, DEFAULT_ROOT)
    label_base = resolve_path(label_root, root)
    return ISPY1Adapter(
        dataset_root=root,
        metadata_csv=resolve_path(metadata_csv, root / DEFAULT_METADATA),
        label_file=label_base / DEFAULT_LABEL_FILE,
    )


class ISPY1Adapter(DatasetAdapter):
    """Adapter for the IDC-style local I-SPY 1 download manifest."""

    dataset_id = DATASET_ID

    def __init__(
        self,
        dataset_root: str | Path,
        metadata_csv: str | Path,
        label_file: str | Path,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        self.metadata_csv = Path(metadata_csv).expanduser().resolve(strict=False)
        self.label_file = Path(label_file).expanduser().resolve(strict=False)

    def scan(self, max_patients: int | None = None) -> AdapterResult:
        suppress_warnings()
        result = AdapterResult(dataset_id=self.dataset_id, source_root=self.dataset_root)
        labels_by_subject = self._load_labels(result)

        if not self.metadata_csv.exists():
            result.issues.append(_issue("error", "missing_metadata", str(self.metadata_csv)))
            return result

        frame = self._load_metadata(result)
        if frame.empty:
            return result
        subjects = self._ordered_subjects(frame)
        if max_patients is not None:
            subjects = subjects[:max_patients]
        frame = frame[frame["_subject_id"].isin(subjects)].copy()

        rows = list(frame.iterrows())
        series_infos: list[dict[str, Any]] = []
        for _, row in progress_iter(
            rows,
            total=len(rows),
            desc=f"Scan {self.dataset_id}",
            unit="series",
        ):
            series_infos.append(self._scan_series_row(row))

        timepoint_by_study = _ispy1_timepoints_by_study(series_infos)
        _annotate_ispy1_dce_layout(series_infos)

        subject_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"studies": set(), "series_count": 0, "mask_count": 0, "roles": set()}
        )
        for info in sorted(series_infos, key=lambda item: _series_info_output_sort_key(item, timepoint_by_study)):
            timepoint = timepoint_by_study.get(
                (info["subject_id"], info["study_uid"]),
                infer_timepoint(info["subject_id"], info["series_description"], info["study_date"]),
            )
            relative_path = _safe_relative(info["source_path"], self.dataset_root)
            notes = _series_notes(info)

            series_record = SeriesRecord(
                dataset_id=self.dataset_id,
                subject_id=info["subject_id"],
                study_uid=info["study_uid"],
                study_date=info["study_date"],
                series_uid=info["series_uid"],
                series_description=info["series_description"],
                modality=info["modality"],
                series_role=info["series_role"],
                timepoint=timepoint,
                image_count=info["image_count"],
                source_path=str(info["source_path"]),
                relative_path=relative_path,
                sop_class_name=info["sop_class_name"],
                manufacturer=info["manufacturer"],
                local_exists=info["local_exists"],
                notes=notes,
            )
            result.series.append(series_record)

            subject = subject_stats[info["subject_id"]]
            subject["studies"].add(info["study_uid"])
            subject["series_count"] += 1
            subject["roles"].add(info["series_role"])
            if is_mask_role(info["series_role"]):
                subject["mask_count"] += 1
                result.masks.append(
                    MaskRecord(
                        dataset_id=self.dataset_id,
                        subject_id=info["subject_id"],
                        study_uid=info["study_uid"],
                        study_date=info["study_date"],
                        series_uid=info["series_uid"],
                        mask_type=info["series_role"],
                        source_path=str(info["source_path"]),
                        relative_path=relative_path,
                        image_count=info["image_count"],
                        notes=info["series_description"],
                    )
                )
            if not info["local_exists"]:
                result.issues.append(
                    _issue("warning", "missing_series_path", f"{info['subject_id']}: {info['source_path']}")
                )
            elif info["header_error"]:
                result.issues.append(
                    _issue(
                        "warning",
                        "dicom_header_read_error",
                        f"{info['subject_id']}: {info['header_error']}",
                    )
                )

        for subject_id, stats in progress_iter(
            sorted(subject_stats.items()),
            total=len(subject_stats),
            desc=f"Cases {self.dataset_id}",
            unit="case",
        ):
            result.cases.append(
                CaseRecord(
                    dataset_id=self.dataset_id,
                    subject_id=subject_id,
                    study_count=len(stats["studies"]),
                    series_count=stats["series_count"],
                    mask_count=stats["mask_count"],
                    available_roles=tuple(sorted(stats["roles"])),
                    has_label=subject_id in labels_by_subject,
                    source_root=str(self.dataset_root),
                )
            )
        return result

    def _load_metadata(self, result: AdapterResult) -> pd.DataFrame:
        try:
            frame = pd.read_csv(self.metadata_csv, low_memory=False)
        except Exception as exc:
            result.issues.append(_issue("error", "metadata_read_error", f"{self.metadata_csv}: {exc}"))
            return pd.DataFrame()
        missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in frame.columns]
        if missing:
            result.issues.append(_issue("error", "metadata_missing_columns", ", ".join(missing)))
            return pd.DataFrame()
        frame = frame.dropna(subset=["PatientID"]).copy()
        frame["_subject_id"] = frame["PatientID"].map(normalize_ispy1_subject_id)
        frame = frame[frame["_subject_id"] != ""]
        frame["_sort_path"] = frame["S5cmdManifestPath"].astype(str).map(_path_sort_key)
        return frame.sort_values(["_subject_id", "_sort_path", "SeriesInstanceUID"])

    def _ordered_subjects(self, frame: pd.DataFrame) -> list[str]:
        subjects = sorted({_clean_value(value) for value in frame["_subject_id"] if _clean_value(value)})
        return sorted(subjects, key=_subject_sort_key)

    def _scan_series_row(self, row: pd.Series) -> dict[str, Any]:
        subject_id = normalize_ispy1_subject_id(row.get("PatientID", ""))
        study_uid = _clean_value(row.get("StudyInstanceUID"))
        series_uid = _clean_value(row.get("SeriesInstanceUID"))
        source_path = self._resolve_series_path(row.get("S5cmdManifestPath", ""))
        local_exists = path_exists(source_path)
        image_count = _count_files(source_path) if local_exists else 0
        header, header_error = (
            _read_first_header(source_path) if local_exists else ({}, "missing series path")
        )
        description = _clean_value(
            header.get("SeriesDescription") or header.get("ProtocolName") or Path(source_path).name
        )
        modality = _clean_value(header.get("Modality"))
        sop_class_uid = _clean_value(header.get("SOPClassUID"))
        study_date = _clean_value(
            header.get("StudyDate") or header.get("SeriesDate") or header.get("AcquisitionDate")
        )
        role = classify_series_role(
            description,
            modality=modality,
            sop_class_name=sop_class_uid,
            source_path=str(source_path),
            dataset_id=self.dataset_id,
        )
        return {
            "subject_id": subject_id,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "source_path": source_path,
            "local_exists": local_exists,
            "image_count": image_count,
            "header_error": header_error,
            "series_description": description,
            "modality": modality,
            "sop_class_name": sop_class_uid,
            "study_date": study_date,
            "study_time": _clean_value(header.get("StudyTime")),
            "acquisition_time": _clean_value(header.get("AcquisitionTime") or header.get("SeriesTime")),
            "manufacturer": _clean_value(header.get("Manufacturer")),
            "series_role": role,
            "completion_status": _clean_value(row.get("completion_status")),
            "study_folder": _study_folder(source_path, self.dataset_root),
            "series_folder": Path(source_path).name,
        }

    def _resolve_series_path(self, manifest_path: Any) -> Path:
        text = _clean_value(manifest_path).replace("\\", "/")
        if text:
            direct = Path(text).expanduser()
            if path_exists(direct):
                return safe_absolute_path(direct)
            marker = "/ispy1/"
            lowered = text.lower()
            if marker in lowered:
                tail = text[lowered.index(marker) + 1 :]
                candidate = self.dataset_root / Path(*Path(tail).parts)
                if path_exists(candidate):
                    return safe_absolute_path(candidate)
        return safe_absolute_path(self.dataset_root / text)

    def _load_labels(self, result: AdapterResult) -> dict[str, LabelRecord]:
        if not self.label_file.exists():
            result.issues.append(_issue("warning", "missing_label_file", str(self.label_file)))
            return {}
        try:
            clinical = pd.read_excel(self.label_file, sheet_name=CLINICAL_SHEET)
            outcomes = pd.read_excel(self.label_file, sheet_name=OUTCOME_SHEET)
        except Exception as exc:
            result.issues.append(_issue("warning", "label_read_error", f"{self.label_file}: {exc}"))
            return {}

        clinical_rows = _rows_by_subject(clinical)
        outcome_rows = _rows_by_subject(outcomes)
        subject_ids = sorted(set(clinical_rows) | set(outcome_rows))
        labels: dict[str, LabelRecord] = {}
        for subject_id in progress_iter(
            subject_ids,
            total=len(subject_ids),
            desc=f"Labels {self.dataset_id}",
            unit="case",
        ):
            clinical_row = clinical_rows.get(subject_id, {})
            outcome_row = outcome_rows.get(subject_id, {})
            payload = {
                **{key: _json_safe(value) for key, value in clinical_row.items()},
                **{key: _json_safe(value) for key, value in outcome_row.items()},
            }
            payload["pCR"] = _json_safe(outcome_row.get("PCR", ""))
            payload["ispy1_core_labels"] = _core_label_summary(payload)
            record = LabelRecord(
                dataset_id=self.dataset_id,
                subject_id=subject_id,
                source_file=str(self.label_file),
                key_column=SUBJECT_ID_COLUMN,
                labels=payload,
                notes=f"{CLINICAL_SHEET}; {OUTCOME_SHEET}",
            )
            labels[subject_id] = record
            result.labels.append(record)
        return labels


def normalize_ispy1_subject_id(value: Any) -> str:
    text = _clean_value(value)
    if not text:
        return text
    text = text.replace("-", "_").upper()
    if text.startswith("ISPY1_"):
        suffix = text.split("ISPY1_", 1)[1]
    else:
        suffix = text
    try:
        suffix = str(int(float(suffix)))
    except ValueError:
        suffix = "".join(ch for ch in suffix if ch.isdigit()) or suffix
    return f"ISPY1_{suffix}"


def _ispy1_timepoints_by_study(series_infos: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    studies: dict[tuple[str, str], dict[str, str]] = {}
    for info in series_infos:
        key = (info["subject_id"], info["study_uid"])
        current = studies.setdefault(
            key,
            {
                "study_date": "",
                "study_time": "",
                "study_folder": info.get("study_folder", ""),
                "study_uid": info["study_uid"],
            },
        )
        if info.get("study_date") and (not current["study_date"] or info["study_date"] < current["study_date"]):
            current["study_date"] = info["study_date"]
        if info.get("study_time") and (not current["study_time"] or info["study_time"] < current["study_time"]):
            current["study_time"] = info["study_time"]

    output: dict[tuple[str, str], str] = {}
    by_subject: dict[str, list[tuple[tuple[str, str], dict[str, str]]]] = defaultdict(list)
    for key, value in studies.items():
        by_subject[key[0]].append((key, value))
    for subject_id, subject_studies in by_subject.items():
        ordered = sorted(subject_studies, key=lambda item: _study_sort_key(item[1]))
        for index, (key, _) in enumerate(ordered):
            output[key] = f"MRI{index + 1}"
    return output


def _annotate_ispy1_dce_layout(series_infos: list[dict[str, Any]]) -> None:
    by_study: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for info in series_infos:
        by_study[(info["subject_id"], info["study_uid"])].append(info)

    for study_infos in by_study.values():
        originals = [info for info in study_infos if info.get("series_role") in ISPY1_DCE_SOURCE_ROLES]
        maps = [info for info in study_infos if info.get("series_role") in {"DCE_PE", "DCE_SER"}]
        map_counts_by_base: dict[str, list[int]] = defaultdict(list)
        for info in maps:
            if info.get("image_count", 0) >= 8:
                map_counts_by_base[_base_dce_description(info)].append(int(info["image_count"]))

        originals_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for info in originals:
            originals_by_base[_base_dce_description(info)].append(info)

        for base, rows in originals_by_base.items():
            ordered = sorted(rows, key=_ispy1_dce_acquisition_sort_key)
            map_counts = map_counts_by_base.get(base, [])
            for index, info in enumerate(ordered):
                reference_count = _best_reference_slice_count(info.get("image_count", 0), map_counts)
                inferred_phase_count = (
                    int(info["image_count"]) // reference_count
                    if reference_count and int(info["image_count"]) % reference_count == 0
                    else 1
                )
                if inferred_phase_count > 1:
                    layout = "single_series_multiphase"
                elif len(ordered) > 1:
                    layout = "split_series_phase"
                else:
                    layout = "single_series_single_phase"
                info["dce_layout"] = layout
                info["dce_base"] = base
                info["dce_phase_index"] = str(index)
                info["dce_phase_count_in_study"] = str(len(ordered))
                info["dce_reference_slice_count"] = str(reference_count or "")
                info["dce_inferred_phase_count"] = str(inferred_phase_count if inferred_phase_count > 1 else "")


def _series_notes(info: dict[str, Any]) -> str:
    parts = [
        ("status", info.get("completion_status", "")),
        ("study_folder", info.get("study_folder", "")),
        ("series_folder", info.get("series_folder", "")),
        ("acquisition_time", info.get("acquisition_time", "")),
        ("dce_layout", info.get("dce_layout", "")),
        ("dce_phase_index", info.get("dce_phase_index", "")),
        ("dce_phase_count_in_study", info.get("dce_phase_count_in_study", "")),
        ("dce_reference_slice_count", info.get("dce_reference_slice_count", "")),
        ("dce_inferred_phase_count", info.get("dce_inferred_phase_count", "")),
    ]
    return "; ".join(f"{key}={value}" for key, value in parts if not _is_empty(value))


def _series_info_output_sort_key(
    info: dict[str, Any],
    timepoint_by_study: dict[tuple[str, str], str],
) -> tuple[tuple[int, str], int, str, str, int, float, tuple[int, ...], str]:
    role = _clean_value(info.get("series_role"))
    acquisition_time = _time_to_seconds(info.get("acquisition_time")) if role in ISPY1_DCE_SOURCE_ROLES else None
    return (
        _subject_sort_key(_clean_value(info.get("subject_id"))),
        _timepoint_sort_number(
            timepoint_by_study.get(
                (_clean_value(info.get("subject_id")), _clean_value(info.get("study_uid"))),
                "",
            )
        ),
        _clean_value(info.get("study_date")),
        _clean_value(info.get("study_uid")),
        _role_sort_number(role),
        acquisition_time if acquisition_time is not None else float("inf"),
        _path_sort_key(info.get("source_path", "")),
        _clean_value(info.get("series_uid")),
    )


def _timepoint_sort_number(timepoint: str) -> int:
    match = re.search(r"(\d+)$", _clean_value(timepoint))
    if match:
        return int(match.group(1))
    return 10**9


def _role_sort_number(role: str) -> int:
    return {
        "DCE": 0,
        "DCE_PRE": 1,
        "DCE_POST": 2,
        "DCE_PE": 3,
        "DCE_SER": 4,
        "T1": 5,
        "T1_PRE": 6,
        "T1_POST": 7,
        "T2": 8,
        "DWI": 9,
        "ADC": 10,
        "derived": 11,
        "mask": 12,
        "mask_seg": 13,
        "localizer": 14,
    }.get(role, 99)


def _rows_by_subject(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if SUBJECT_ID_COLUMN not in frame.columns:
        return rows
    for row in frame.dropna(how="all").to_dict("records"):
        subject_id = normalize_ispy1_subject_id(row.get(SUBJECT_ID_COLUMN, ""))
        if subject_id:
            rows[subject_id] = row
    return rows


def _core_label_summary(labels: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "age",
        "race_id",
        "ERpos",
        "PgRpos",
        "HR Pos",
        "Her2MostPos",
        "HR_HER2_CATEGORY",
        "HR_HER2_STATUS",
        "BilateralCa",
        "Laterality",
        "MRI LD Baseline",
        "MRI LD 1-3dAC",
        "MRI LD InterReg",
        "MRI LD PreSurg",
        "PCR",
        "RCBClass",
        "RFS",
        "rfs_ind",
    )
    return {key: labels[key] for key in keys if key in labels and not _is_empty(labels[key])}


def _subject_sort_key(subject_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", subject_id)
    return (int(match.group(1)) if match else 10**12, subject_id)


def _study_sort_key(value: dict[str, str]) -> tuple[str, str, tuple[int, ...], str]:
    return (
        value.get("study_date", ""),
        value.get("study_time", ""),
        _path_sort_key(value.get("study_folder", "")),
        value.get("study_uid", ""),
    )


def _path_sort_key(value: Any) -> tuple[int, ...]:
    parts = re.findall(r"\d+", _clean_value(value))
    return tuple(int(part) for part in parts[-3:])


def _study_folder(series_path: Path, root: Path) -> str:
    try:
        relative = series_path.relative_to(root)
        parts = relative.parts
        if len(parts) >= 3 and parts[0].lower() == "ispy1":
            return parts[2]
        if len(parts) >= 2:
            return parts[-2]
    except ValueError:
        pass
    return series_path.parent.name


def _base_dce_description(info: dict[str, Any]) -> str:
    text = _clean_value(info.get("series_description")).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s*:\s*(?:ser|pe\s*[1-9])\s*$", "", text)
    text = re.sub(r"\b(?:phase\s*=\s*\d+)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _best_reference_slice_count(image_count: int, map_counts: list[int]) -> int:
    candidates = [
        count
        for count in map_counts
        if count >= 8 and count < image_count and image_count % count == 0 and 1 < image_count // count <= 20
    ]
    if not candidates:
        return 0
    counts = {count: map_counts.count(count) for count in set(candidates)}
    return sorted(candidates, key=lambda count: (-counts[count], -count))[0]


def _ispy1_dce_acquisition_sort_key(info: dict[str, Any]) -> tuple[float, tuple[int, ...], str]:
    time_value = _time_to_seconds(info.get("acquisition_time", ""))
    return (
        time_value if time_value is not None else float("inf"),
        _path_sort_key(info.get("source_path", "")),
        _clean_value(info.get("series_uid")),
    )


def _time_to_seconds(value: Any) -> float | None:
    text = _clean_value(value).replace(":", "")
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


def _read_first_header(series_dir: Path) -> tuple[dict[str, Any], str]:
    if not path_exists(series_dir):
        return {}, "missing series path"
    try:
        entries = sorted(
            Path(windows_extended_path(series_dir)).iterdir(),
            key=lambda path: path.name,
        )
    except OSError as exc:
        return {}, str(exc)
    last_error = ""
    for path in entries:
        if not path.is_file():
            continue
        try:
            return read_dicom_header(path), ""
        except (DicomReadError, OSError) as exc:
            last_error = str(exc)
    return {}, last_error or "no readable DICOM files"


def _count_files(path: Path) -> int:
    try:
        return sum(1 for entry in os.scandir(windows_extended_path(path)) if entry.is_file())
    except OSError:
        return 0


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _clean_value(value: Any) -> str:
    if _is_empty(value):
        return ""
    return str(value).strip()


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if _is_empty(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _issue(severity: str, category: str, message: str) -> dict[str, Any]:
    return {"severity": severity, "category": category, "message": message, "status": "open"}
