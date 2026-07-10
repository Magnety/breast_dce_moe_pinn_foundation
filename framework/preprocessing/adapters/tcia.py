from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.acrin_labels import build_acrin_core_labels
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
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists, safe_absolute_path, windows_extended_path
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_counter, progress_iter, suppress_warnings


TCGA_CORE_LABEL_FILES = {
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_timeline_status.txt",
    "data_timeline_treatment.txt",
}

TCGA_TREATMENT_DETAIL_FIELDS = (
    "START_DATE",
    "STOP_DATE",
    "TREATMENT_TYPE",
    "TREATMENT_SUBTYPE",
    "AGENT",
    "NUMBER_OF_CYCLES",
    "MEASURE_OF_RESPONSE",
    "REGIMEN_INDICATION",
    "ROUTE_OF_ADMINISTRATION",
    "THERAPY_ONGOING",
    "RADIATION_DOSAGE",
    "RADIATION_UNITS",
)


class TCIAMetadataAdapter(DatasetAdapter):
    """Adapter for TCIA datasets with a `metadata.csv` series table."""

    def __init__(
        self,
        dataset_id: str,
        dataset_root: str | Path,
        metadata_csv: str | Path,
        label_files: list[str | Path] | None = None,
        label_key_candidates: tuple[str, ...] = ("Patient ID", "Subject ID", "CLINICAL-TRIAL-SUBJECT-ID"),
    ):
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        self.metadata_csv = Path(metadata_csv).expanduser().resolve(strict=False)
        self.label_files = [Path(path).expanduser().resolve(strict=False) for path in (label_files or [])]
        self.label_key_candidates = label_key_candidates
        self._path_cache: dict[str, Path] = {}
        self._dir_index: dict[str, list[Path]] | None = None

    def scan(self, max_patients: int | None = None) -> AdapterResult:
        suppress_warnings()
        result = AdapterResult(dataset_id=self.dataset_id, source_root=self.dataset_root)
        if not self.metadata_csv.exists():
            result.issues.append(_issue("error", "missing_metadata", str(self.metadata_csv)))
            return result
        frame = pd.read_csv(self.metadata_csv, low_memory=False)
        if max_patients is not None:
            subjects = list(frame["Subject ID"].dropna().astype(str).drop_duplicates().head(max_patients))
            frame = frame[frame["Subject ID"].astype(str).isin(subjects)]
        labels_by_subject = self._load_labels(result)

        subject_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"studies": set(), "series_count": 0, "mask_count": 0, "roles": set()}
        )
        metadata_rows = list(frame.iterrows())
        for _, row in progress_iter(
            metadata_rows,
            total=len(metadata_rows),
            desc=f"Scan {self.dataset_id}",
            unit="series",
        ):
            subject_id = normalize_subject_id(str(row.get("Subject ID", "")).strip())
            study_uid = str(row.get("Study UID", "")).strip()
            study_date = str(row.get("Study Date", "")).strip()
            description = str(row.get("Series Description", "") or "")
            modality = str(row.get("Modality", "") or "")
            sop_class = str(row.get("SOP Class Name", "") or "")
            file_location = str(row.get("File Location", "") or "")
            role = classify_series_role(
                description,
                modality=modality,
                sop_class_name=sop_class,
                source_path=file_location,
                dataset_id=self.dataset_id,
            )
            source_path = self._resolve_file_location(file_location)
            local_exists = path_exists(source_path)
            rel_path = _safe_relative(source_path, self.dataset_root)
            image_count = _safe_int(row.get("Number of Images"))
            timepoint = infer_timepoint(subject_id, str(row.get("Study Description", "") or ""), study_date)
            series_record = SeriesRecord(
                dataset_id=self.dataset_id,
                subject_id=subject_id,
                study_uid=study_uid,
                study_date=study_date,
                series_uid=str(row.get("Series UID", "")).strip(),
                series_description=description,
                modality=modality,
                series_role=role,
                timepoint=timepoint,
                image_count=image_count,
                source_path=str(source_path),
                relative_path=rel_path,
                sop_class_name=sop_class,
                manufacturer=str(row.get("Manufacturer", "") or ""),
                local_exists=local_exists,
            )
            result.series.append(series_record)
            subject = subject_stats[subject_id]
            subject["studies"].add(study_uid)
            subject["series_count"] += 1
            subject["roles"].add(role)
            if is_mask_role(role):
                subject["mask_count"] += 1
                result.masks.append(
                    MaskRecord(
                        dataset_id=self.dataset_id,
                        subject_id=subject_id,
                        study_uid=study_uid,
                        study_date=study_date,
                        series_uid=series_record.series_uid,
                        mask_type=role,
                        source_path=str(source_path),
                        relative_path=rel_path,
                        image_count=image_count,
                        notes=sop_class,
                    )
                )
            if not local_exists:
                result.issues.append(
                    _issue(
                        "warning",
                        "missing_series_path",
                        f"{subject_id}: {file_location}",
                    )
                )

        subject_items = sorted(subject_stats.items())
        for subject_id, stats in progress_iter(
            subject_items,
            total=len(subject_items),
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

    def _resolve_file_location(self, file_location: str) -> Path:
        cleaned = file_location.replace("\\", "/").lstrip("./")
        if cleaned in self._path_cache:
            return self._path_cache[cleaned]
        candidates = [
            self.metadata_csv.parent / cleaned,
            self.dataset_root / cleaned,
            self.metadata_csv.parent / Path(cleaned).name,
        ]
        for candidate in candidates:
            if path_exists(candidate):
                resolved = safe_absolute_path(candidate)
                self._path_cache[cleaned] = resolved
                return resolved
        found = self._find_existing_tail(cleaned)
        self._path_cache[cleaned] = found
        return found

    def _find_existing_tail(self, cleaned: str) -> Path:
        parts = Path(cleaned).parts
        if not parts:
            return safe_absolute_path(self.metadata_csv.parent / cleaned)
        leaf = parts[-1]
        tail_lengths = [min(len(parts), value) for value in (4, 3, 2, 1)]
        roots = [self.metadata_csv.parent, self.dataset_root]
        seen_roots: set[str] = set()
        dir_index = self._get_dir_index()
        for root in roots:
            root_key = str(root).lower()
            if root_key in seen_roots or not path_exists(root):
                continue
            seen_roots.add(root_key)
            for candidate in dir_index.get(leaf.lower(), []):
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
                candidate_parts = candidate.parts
                for tail_length in tail_lengths:
                    if tuple(part.lower() for part in candidate_parts[-tail_length:]) == tuple(
                        part.lower() for part in parts[-tail_length:]
                    ):
                        return safe_absolute_path(candidate)
        return safe_absolute_path(self.metadata_csv.parent / cleaned)

    def _get_dir_index(self) -> dict[str, list[Path]]:
        if self._dir_index is not None:
            return self._dir_index
        index: dict[str, list[Path]] = defaultdict(list)
        roots = [self.metadata_csv.parent, self.dataset_root]
        seen_roots: set[str] = set()
        for root in roots:
            root_key = str(root).lower()
            if root_key in seen_roots or not path_exists(root):
                continue
            seen_roots.add(root_key)
            with progress_counter(desc=f"Index dirs {root.name}", unit="dir") as scan_progress:
                for current_dir, dir_names, _ in os.walk(windows_extended_path(root)):
                    current_path = Path(current_dir)
                    scan_progress.update()
                    for dir_name in dir_names:
                        index[dir_name.lower()].append(current_path / dir_name)
        self._dir_index = index
        return index

    def _load_labels(self, result: AdapterResult) -> dict[str, LabelRecord]:
        labels: dict[str, LabelRecord] = {}
        if self.dataset_id == "acrin_contralateral":
            acrin_form_descriptions = _load_acrin_form_descriptions(self.label_files)
            return self._load_acrin_core_labels(result, acrin_form_descriptions)
        if self.dataset_id == "tcga_brca":
            return self._load_tcga_core_labels(result)
        for path in progress_iter(
            self.label_files,
            total=len(self.label_files),
            desc=f"Load labels {self.dataset_id}",
            unit="file",
        ):
            if self.dataset_id == "acrin_contralateral" and _is_acrin_dictionary_file(path):
                continue
            if not path.exists():
                result.issues.append(_issue("warning", "missing_label_file", str(path)))
                continue
            try:
                rows = _read_label_rows(path)
            except Exception as exc:
                result.issues.append(_issue("warning", "label_read_error", f"{path}: {exc}"))
                continue
            for row in progress_iter(
                rows,
                total=len(rows),
                desc=f"Labels {path.name}",
                unit="row",
            ):
                key_column = _find_key_column(row, self.label_key_candidates)
                if not key_column:
                    continue
                raw_subject_id = row.get(key_column, "")
                if _is_empty_value(raw_subject_id):
                    continue
                subject_id = normalize_subject_id(str(raw_subject_id), dataset_id=self.dataset_id)
                if not subject_id:
                    continue
                row_labels = {key: _json_safe(value) for key, value in row.items()}
                notes = ""
                label_record = LabelRecord(
                    dataset_id=self.dataset_id,
                    subject_id=subject_id,
                    source_file=str(path),
                    key_column=key_column,
                    labels=row_labels,
                    notes=notes,
                )
                labels[subject_id] = label_record
                result.labels.append(label_record)
        return labels

    def _load_tcga_core_labels(self, result: AdapterResult) -> dict[str, LabelRecord]:
        labels: dict[str, LabelRecord] = {}
        rows_by_file: dict[str, list[dict[str, Any]]] = {}
        for path in progress_iter(
            self.label_files,
            total=len(self.label_files),
            desc=f"Load labels {self.dataset_id}",
            unit="file",
        ):
            if path.name not in TCGA_CORE_LABEL_FILES:
                continue
            if not path.exists():
                result.issues.append(_issue("warning", "missing_label_file", str(path)))
                continue
            try:
                rows_by_file[path.name] = _read_label_rows(path)
            except Exception as exc:
                result.issues.append(_issue("warning", "label_read_error", f"{path}: {exc}"))

        grouped = {
            name: _group_tcga_rows_by_subject(rows)
            for name, rows in rows_by_file.items()
        }
        subject_ids: set[str] = set()
        for by_subject in grouped.values():
            subject_ids.update(by_subject.keys())

        subject_items = sorted(subject_ids)
        for subject_id in progress_iter(
            subject_items,
            total=len(subject_items),
            desc=f"Core labels {self.dataset_id}",
            unit="case",
        ):
            patient_rows = grouped.get("data_clinical_patient.txt", {}).get(subject_id, [])
            sample_rows = grouped.get("data_clinical_sample.txt", {}).get(subject_id, [])
            status_rows = grouped.get("data_timeline_status.txt", {}).get(subject_id, [])
            treatment_rows = grouped.get("data_timeline_treatment.txt", {}).get(subject_id, [])
            core_labels = _build_tcga_core_labels(patient_rows, sample_rows, status_rows, treatment_rows)
            if not core_labels:
                continue
            source_files = [
                str(path)
                for path in self.label_files
                if path.name in rows_by_file
                and grouped.get(path.name, {}).get(subject_id)
            ]
            label_record = LabelRecord(
                dataset_id=self.dataset_id,
                subject_id=subject_id,
                source_file=";".join(source_files),
                key_column="PATIENT_ID",
                labels=core_labels,
                notes="TCGA clinical labels reduced to MRI preprocessing core fields.",
            )
            labels[subject_id] = label_record
            result.labels.append(label_record)
        return labels

    def _load_acrin_core_labels(
        self,
        result: AdapterResult,
        acrin_form_descriptions: dict[str, str],
    ) -> dict[str, LabelRecord]:
        labels: dict[str, LabelRecord] = {}
        raw_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        source_files_by_subject: dict[str, set[str]] = defaultdict(set)
        key_columns_by_subject: dict[str, set[str]] = defaultdict(set)
        for path in progress_iter(
            self.label_files,
            total=len(self.label_files),
            desc=f"Load labels {self.dataset_id}",
            unit="file",
        ):
            if _is_acrin_dictionary_file(path):
                continue
            if not path.exists():
                result.issues.append(_issue("warning", "missing_label_file", str(path)))
                continue
            try:
                rows = _read_label_rows(path)
            except Exception as exc:
                result.issues.append(_issue("warning", "label_read_error", f"{path}: {exc}"))
                continue
            acrin_dictionary = _load_acrin_field_dictionary(path)
            for row in progress_iter(
                rows,
                total=len(rows),
                desc=f"Labels {path.name}",
                unit="row",
            ):
                key_column = _find_key_column(row, self.label_key_candidates)
                if not key_column:
                    continue
                raw_subject_id = row.get(key_column, "")
                if _is_empty_value(raw_subject_id):
                    continue
                subject_id = normalize_subject_id(str(raw_subject_id), dataset_id=self.dataset_id)
                if not subject_id:
                    continue
                row_labels = _decode_acrin_label_row(
                    row,
                    path,
                    acrin_form_descriptions,
                    dictionary=acrin_dictionary,
                )
                raw_by_subject[subject_id].append(
                    {
                        "source_file": str(path),
                        "key_column": key_column,
                        "labels": row_labels,
                    }
                )
                source_files_by_subject[subject_id].add(str(path))
                key_columns_by_subject[subject_id].add(key_column)

        subject_items = sorted(raw_by_subject.items())
        for subject_id, raw_records in progress_iter(
            subject_items,
            total=len(subject_items),
            desc=f"Core labels {self.dataset_id}",
            unit="case",
        ):
            core_labels = build_acrin_core_labels(raw_records)
            label_record = LabelRecord(
                dataset_id=self.dataset_id,
                subject_id=subject_id,
                source_file=";".join(sorted(source_files_by_subject[subject_id])),
                key_column=";".join(sorted(key_columns_by_subject[subject_id])),
                labels=core_labels,
                notes="ACRIN clinical labels reduced to ISPY2-like core fields.",
            )
            labels[subject_id] = label_record
            result.labels.append(label_record)
        return labels


def normalize_subject_id(value: str, dataset_id: str = "") -> str:
    text = value.strip()
    if text == "":
        return text
    text = text.replace("_", "-")
    if dataset_id == "acrin_contralateral" and text.isdigit():
        return f"ACRIN-Contralateral-Breast-MR-{int(text):03d}"
    if dataset_id == "ispy2" and text.isdigit():
        return f"ISPY2-{int(text)}"
    if text.isdigit():
        return text
    if text.upper().startswith("UCSF-BR-"):
        return text.upper()
    if text.upper().startswith("ISPY2-"):
        return text.upper()
    return text


def _read_label_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False).to_dict("records")
    if suffix == ".txt":
        frame = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        return frame.dropna(how="all").to_dict("records")
    if suffix in {".xls", ".xlsx"}:
        if "SharedClinicalAndRFS" in path.name:
            frame = pd.read_excel(path, sheet_name="Clinical and DFS Data")
        elif path.name == "Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx":
            frame = _read_advanced_mri_breast_lesions_labels(path)
        elif path.name == "Clinical_and_Other_Features.xlsx":
            frame = pd.read_excel(path, sheet_name="Data", header=1)
        elif path.name == "Annotation_Boxes.xlsx":
            frame = pd.read_excel(path)
        elif path.name == "Multi-feature-MRI-NACT-Data.xlsx":
            frame = pd.read_excel(path, sheet_name=0)
            if "CLINICAL-TRIAL-SUBJECT-ID" in frame.columns:
                frame["CLINICAL-TRIAL-SUBJECT-ID"] = frame["CLINICAL-TRIAL-SUBJECT-ID"].apply(
                    lambda value: f"ISPY2-{int(value)}" if pd.notna(value) else ""
                )
        else:
            frame = pd.read_excel(path, sheet_name=0)
        return frame.dropna(how="all").to_dict("records")
    return []


def _read_advanced_mri_breast_lesions_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="MRI BREASTS with Delay", header=1)
    unnamed_columns = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:") and frame[column].isna().all()
    ]
    if unnamed_columns:
        frame = frame.drop(columns=unnamed_columns)
    frame = frame.dropna(how="all")
    frame = frame.replace({-1: pd.NA, "-1": pd.NA})
    return frame


def _group_tcga_rows_by_subject(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject_id = normalize_subject_id(str(_tcga_value(row, "PATIENT_ID")), dataset_id="tcga_brca")
        if subject_id:
            grouped[subject_id].append(row)
    return grouped


def _build_tcga_core_labels(
    patient_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    patient = patient_rows[0] if patient_rows else {}
    sample = _pick_tcga_sample_row(sample_rows)
    latest_status = _latest_tcga_status_row(status_rows)
    source_label_count = len(patient_rows) + len(sample_rows) + len(status_rows) + len(treatment_rows)
    if source_label_count == 0:
        return {}

    labels: dict[str, Any] = {"source_label_count": source_label_count}
    age = _tcga_value(patient, "AGE")
    sex = _tcga_value(patient, "SEX")
    subtype_raw = _tcga_value(patient, "SUBTYPE")
    subtype = _normalize_tcga_subtype(subtype_raw)
    stage = _tcga_value(patient, "AJCC_PATHOLOGIC_TUMOR_STAGE") or _tcga_first_value(
        status_rows,
        "PATHOLOGIC_STAGE",
    )
    path_t = _tcga_value(patient, "PATH_T_STAGE") or _tcga_first_value(status_rows, "PATHOLOGIC_T")
    path_n = _tcga_value(patient, "PATH_N_STAGE") or _tcga_first_value(status_rows, "PATHOLOGIC_N")
    path_m = _tcga_value(patient, "PATH_M_STAGE") or _tcga_first_value(status_rows, "PATHOLOGIC_M")
    histologic_type = _tcga_value(sample, "CANCER_TYPE_DETAILED") or _tcga_value(sample, "TUMOR_TYPE")
    grade = _tcga_value(sample, "GRADE")
    treatment_summary = _tcga_treatment_summary(patient, treatment_rows)

    _put_known(labels, "Age", age)
    _put_known(labels, "Sex", sex)
    _put_known(labels, "SUBTYPE", subtype_raw)
    _put_known(labels, "Molecular subtype", subtype)
    _put_known(labels, "Pathologic stage", stage)
    _put_known(labels, "AJCC_PATHOLOGIC_TUMOR_STAGE", _tcga_value(patient, "AJCC_PATHOLOGIC_TUMOR_STAGE"))
    _put_known(labels, "PATH_T_STAGE", path_t)
    _put_known(labels, "PATH_N_STAGE", path_n)
    _put_known(labels, "PATH_M_STAGE", path_m)
    _put_known(labels, "HISTORY_NEOADJUVANT_TRTYN", _tcga_value(patient, "HISTORY_NEOADJUVANT_TRTYN"))
    _put_known(labels, "RADIATION_THERAPY", _tcga_value(patient, "RADIATION_THERAPY"))
    _put_known(labels, "Treatment summary", treatment_summary)
    _put_known(labels, "Histologic type", histologic_type)
    _put_known(labels, "Cancer type", _tcga_value(sample, "CANCER_TYPE_DETAILED") or _tcga_value(sample, "CANCER_TYPE"))
    _put_known(labels, "Grade", grade)
    _put_known(labels, "ONCOTREE_CODE", _tcga_value(sample, "ONCOTREE_CODE"))
    _put_known(labels, "SAMPLE_TYPE", _tcga_value(sample, "SAMPLE_TYPE"))

    pathology = _compact_known_dict(
        {
            "pathologic_stage": stage,
            "pathologic_t": path_t,
            "pathologic_n": path_n,
            "pathologic_m": path_m,
            "histologic_type": histologic_type,
            "grade": grade,
            "oncotree_code": _tcga_value(sample, "ONCOTREE_CODE"),
            "tumor_type": _tcga_value(sample, "TUMOR_TYPE"),
            "tumor_tissue_site": _tcga_value(sample, "TUMOR_TISSUE_SITE"),
            "icd_o_3_histology": _tcga_value(patient, "ICD_O_3_HISTOLOGY"),
            "icd_o_3_site": _tcga_value(patient, "ICD_O_3_SITE"),
        }
    )
    sample_summary = _compact_known_dict(
        {
            "sample_id": _tcga_value(sample, "SAMPLE_ID"),
            "sample_type": _tcga_value(sample, "SAMPLE_TYPE"),
            "oncotree_code": _tcga_value(sample, "ONCOTREE_CODE"),
            "cancer_type": _tcga_value(sample, "CANCER_TYPE"),
            "cancer_type_detailed": _tcga_value(sample, "CANCER_TYPE_DETAILED"),
            "tmb_nonsynonymous": _tcga_value(sample, "TMB_NONSYNONYMOUS"),
            "tissue_source_site": _tcga_value(sample, "TISSUE_SOURCE_SITE"),
        }
    )
    survival = _tcga_survival_summary(patient, latest_status)
    treatment_details = _tcga_treatment_details(treatment_rows)

    _put_known(labels, "pathology", pathology)
    _put_known(labels, "sample", sample_summary)
    _put_known(labels, "survival", survival)
    _put_known(labels, "treatment_details", treatment_details)
    labels["tcga_core_labels"] = _compact_known_dict(
        {
            "age": age,
            "sex": sex,
            "molecular_subtype": subtype,
            "pathologic_stage": stage,
            "pathology": pathology,
            "treatment": treatment_summary,
            "survival": survival,
            "sample": sample_summary,
        }
    )
    return _tcga_json_safe(labels)


def _pick_tcga_sample_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            0 if str(_tcga_value(row, "SAMPLE_TYPE")).lower() == "primary" else 1,
            str(_tcga_value(row, "SAMPLE_ID")),
        ),
    )[0]


def _latest_tcga_status_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if _tcga_known(_tcga_value(row, "STATUS")) or _tcga_known(_tcga_value(row, "VITAL_STATUS"))]
    if not candidates:
        candidates = rows
    if not candidates:
        return {}
    return sorted(candidates, key=_tcga_start_date, reverse=True)[0]


def _tcga_first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in sorted(rows, key=_tcga_start_date, reverse=True):
        value = _tcga_value(row, key)
        if _tcga_known(value):
            return value
    return ""


def _tcga_survival_summary(patient: dict[str, Any], latest_status: dict[str, Any]) -> dict[str, Any]:
    return _compact_known_dict(
        {
            "os_status": _tcga_value(patient, "OS_STATUS"),
            "os_months": _tcga_value(patient, "OS_MONTHS"),
            "dss_status": _tcga_value(patient, "DSS_STATUS"),
            "dss_months": _tcga_value(patient, "DSS_MONTHS"),
            "dfs_status": _tcga_value(patient, "DFS_STATUS"),
            "dfs_months": _tcga_value(patient, "DFS_MONTHS"),
            "pfs_status": _tcga_value(patient, "PFS_STATUS"),
            "pfs_months": _tcga_value(patient, "PFS_MONTHS"),
            "person_neoplasm_cancer_status": _tcga_value(patient, "PERSON_NEOPLASM_CANCER_STATUS"),
            "new_tumor_event_after_initial_treatment": _tcga_value(
                patient,
                "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
            ),
            "latest_timeline_status": _tcga_value(latest_status, "STATUS"),
            "latest_tumor_status": _tcga_value(latest_status, "TUMOR_STATUS"),
            "latest_vital_status": _tcga_value(latest_status, "VITAL_STATUS"),
        }
    )


def _tcga_treatment_summary(patient: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    neoadjuvant = _tcga_value(patient, "HISTORY_NEOADJUVANT_TRTYN")
    radiation = _tcga_value(patient, "RADIATION_THERAPY")
    treatment_types = _tcga_unique_values(rows, "TREATMENT_TYPE")
    agents = _tcga_unique_values(rows, "AGENT")
    responses = _tcga_unique_values(rows, "MEASURE_OF_RESPONSE")
    if _tcga_known(neoadjuvant):
        parts.append(f"neoadjuvant={neoadjuvant}")
    if _tcga_known(radiation):
        parts.append(f"radiation={radiation}")
    if treatment_types:
        parts.append("types=" + "|".join(treatment_types))
    if agents:
        parts.append("agents=" + "|".join(agents[:20]))
    if len(agents) > 20:
        parts.append(f"agents_truncated={len(agents) - 20}")
    if responses:
        parts.append("response=" + "|".join(responses))
    return "; ".join(parts)


def _tcga_treatment_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in sorted(rows, key=_tcga_start_date):
        detail = _compact_known_dict({field: _tcga_value(row, field) for field in TCGA_TREATMENT_DETAIL_FIELDS})
        if detail:
            details.append(detail)
    return details


def _tcga_unique_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = _tcga_value(row, key)
        if not _tcga_known(value):
            continue
        text = str(value)
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(text)
    return sorted(values)


def _tcga_start_date(row: dict[str, Any]) -> float:
    value = _tcga_value(row, "START_DATE")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _normalize_tcga_subtype(value: Any) -> Any:
    if not _tcga_known(value):
        return ""
    text = str(value).strip()
    mapping = {
        "brca_luma": "Luminal A",
        "brca_lumb": "Luminal B",
        "brca_her2": "HER2-enriched",
        "brca_basal": "Basal-like",
        "brca_normal": "Normal-like",
    }
    return mapping.get(text.lower(), text.replace("BRCA_", ""))


def _compact_known_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _tcga_json_safe(value) for key, value in values.items() if _tcga_known(value)}


def _put_known(output: dict[str, Any], key: str, value: Any) -> None:
    if _tcga_known(value):
        output[key] = _tcga_json_safe(value)


def _tcga_value(row: dict[str, Any], key: str) -> Any:
    if not row:
        return ""
    value = row.get(key, "")
    if _is_empty_value(value):
        return ""
    if isinstance(value, str):
        text = " ".join(value.replace("\xa0", " ").split())
        if not text:
            return ""
        if text.upper() in {"NA", "N/A", "NAN", "NONE", "NULL", "[NOT AVAILABLE]", "[NOT APPLICABLE]"}:
            return ""
        return text
    return _json_safe(value)


def _tcga_known(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, dict):
        return any(_tcga_known(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_tcga_known(item) for item in value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "not reported", "not available"}
    return True


def _tcga_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _tcga_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tcga_json_safe(item) for item in value]
    return _json_safe(value)


def _is_acrin_dictionary_file(path: Path) -> bool:
    name = path.name.lower()
    return "dictionary reviewed" in name or "formdictionary reviewed" in name


def _load_acrin_form_descriptions(paths: list[Path]) -> dict[str, str]:
    if not paths:
        return {}
    parent = paths[0].parent
    form_path = parent / "6667_FormDictionary reviewed.csv"
    if not form_path.exists():
        return {}
    frame = pd.read_csv(form_path, low_memory=False)
    output: dict[str, str] = {}
    form_rows = list(frame.iterrows())
    for _, row in progress_iter(
        form_rows,
        total=len(form_rows),
        desc="Load ACRIN forms",
        unit="row",
    ):
        form = _clean_text(row.get("form", "")).upper()
        description = _clean_text(row.get("form_desc", ""))
        if form and description:
            output[form] = description
    return output


def _decode_acrin_label_row(
    row: dict[str, Any],
    source_path: Path,
    form_descriptions: dict[str, str] | None = None,
    dictionary: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    form = _acrin_form_name(source_path)
    dictionary = dictionary if dictionary is not None else _load_acrin_field_dictionary(source_path)
    labels: dict[str, Any] = {
        "acrin_form": form,
        "acrin_form_description": (form_descriptions or {}).get(form, ""),
    }
    variable_codes: dict[str, Any] = {}

    for key, value in row.items():
        if _is_empty_value(value):
            continue
        key_text = str(key).strip()
        key_lower = key_text.lower()
        if key_lower == "cn":
            labels["clinical_trial_subject_id"] = _json_safe(value)
            labels["ACRIN subject id"] = normalize_subject_id(str(value), dataset_id="acrin_contralateral")
            continue

        meta = dictionary.get(key_lower, {})
        field_name = str(meta.get("field_name") or key_text)
        label = _clean_label(str(meta.get("label") or key_text))
        decoded_value = _decode_acrin_value(value, meta.get("responses", {}))
        _set_unique_label(labels, label, decoded_value)
        variable_codes[field_name] = {
            "label": label,
            "raw_value": _json_safe(value),
            "decoded_value": decoded_value,
        }

    if variable_codes:
        labels["acrin_variable_codes"] = variable_codes
    return labels


def _acrin_form_name(path: Path) -> str:
    name = path.name
    if not name.startswith("6667_"):
        return ""
    return name.split("6667_", 1)[1].split(" reviewed.csv", 1)[0].upper()


def _load_acrin_field_dictionary(data_path: Path) -> dict[str, dict[str, Any]]:
    form = _acrin_form_name(data_path)
    if not form:
        return {}
    dictionary_path = data_path.with_name(f"6667_{form}_Dictionary reviewed.csv")
    if not dictionary_path.exists():
        return {}
    frame = pd.read_csv(dictionary_path, low_memory=False)
    output: dict[str, dict[str, Any]] = {}
    dictionary_rows = list(frame.iterrows())
    for _, row in progress_iter(
        dictionary_rows,
        total=len(dictionary_rows),
        desc=f"Load ACRIN {form}",
        unit="row",
    ):
        field_name = _clean_text(row.get("field_name", ""))
        if not field_name:
            continue
        key = field_name.lower()
        label = _acrin_fallback_label(field_name, _clean_label(_clean_text(row.get("ques_desc", ""))))
        meta = output.setdefault(key, {"field_name": field_name, "label": label, "responses": {}})
        if meta.get("label") == field_name and label != field_name:
            meta["label"] = label
        response_code = _response_key(row.get("resp_seq_no"))
        response_text = _clean_text(row.get("resp_desc", ""))
        if response_code and response_text and not response_text.startswith(("$F", "best", "BEST")):
            meta["responses"][response_code] = response_text
    return output


def _acrin_fallback_label(field_name: str, label: str) -> str:
    if label:
        return label
    fallback_labels = {
        "f1e50": "Cause of death relationship to breast cancer",
        "pae13": "Types of atypia",
    }
    return fallback_labels.get(field_name.lower(), field_name)


def _decode_acrin_value(value: Any, responses: dict[str, str]) -> Any:
    if _is_empty_value(value):
        return ""
    response = responses.get(_response_key(value))
    if response is not None:
        return response
    return _json_safe(value)


def _response_key(value: Any) -> str:
    if _is_empty_value(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).strip()


def _set_unique_label(labels: dict[str, Any], key: str, value: Any) -> None:
    if key not in labels or labels[key] == value:
        labels[key] = value
        return
    index = 2
    while f"{key} ({index})" in labels and labels[f"{key} ({index})"] != value:
        index += 1
    labels[f"{key} ({index})"] = value


def _clean_label(value: str) -> str:
    text = " ".join(str(value).replace("\xa0", " ").split())
    return text.strip(" :")


def _clean_text(value: Any) -> str:
    if _is_empty_value(value):
        return ""
    return str(value).strip()


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _find_key_column(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): str(key) for key in row.keys()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for key in row:
        key_text = str(key).lower()
        if "patient" in key_text and "id" in key_text:
            return str(key)
    return ""


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _issue(severity: str, category: str, message: str) -> dict[str, Any]:
    return {"severity": severity, "category": category, "message": message, "status": "open"}
