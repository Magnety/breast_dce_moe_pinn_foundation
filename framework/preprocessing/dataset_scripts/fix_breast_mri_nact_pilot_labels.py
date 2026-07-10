from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DATASET_ID = "breast_mri_nact_pilot"
DEFAULT_NACT_ROOT = Path("F:/open_data/breast/Breast-MRI-NACT-Pilot")
DEFAULT_DATASET_ROOT = Path(os.environ.get("BREAST_MRI_DATASET_ROOT", "/home/ubuntu/liuyiyao1/multimodal_dataset"))
DEFAULT_PROCESSED_ROOT = DEFAULT_DATASET_ROOT / "full_v4_breast_mri_nact_pilot"
DEFAULT_LABEL_FILE = "SharedClinicalAndRFS.xls"
DEFAULT_SHEET_NAME = "Clinical and DFS Data"

PATIENT_ID = "Patient ID"
PCR_COLUMN = "Clinical response"

MISSING_TOKENS = {"", "NA", "N/A", "NC", "NP", "NONE", "NULL", "NAN"}
RECEPTOR_MAP = {0: "negative", 1: "positive"}
YES_NO_MAP = {0: "no", 1: "yes"}

CLINICAL_RESPONSE_MAP = {
    1: "ned_no_evidence_of_disease",
    2: "partial_response_gt_one_third_ld_decrease",
    3: "minor_or_no_response_lt_one_third_ld_decrease",
    4: "stable_or_progressive_disease",
}
HR_HER2_STATUS_MAP = {
    "hrposher2neg": "HR+/HER2-",
    "her2pos": "HER2-positive",
    "tripleneg": "Triple-negative",
}
SURGERY_TYPE_MAP = {
    "M": "mastectomy",
    "L": "lumpectomy",
    "L (Q)": "quadrantectomy",
    "L (Q)- NO RD TX": "quadrantectomy_no_radiation",
}
RECURRENCE_TYPE_MAP = {
    "met": "distant_metastasis",
    "local": "local_recurrence",
    "unknown": "unknown",
}

NACT_LABEL_SUMMARY_FIELDS = [
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
    "race",
    "tumor_size_cm",
    "dfs_event",
    "dfs_time_weeks",
    "source_label_count",
    "split",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair Breast-MRI-NACT-Pilot labels in an already-built multimodal dataset. "
            "The script rebuilds labels from SharedClinicalAndRFS.xls and overwrites "
            "Breast-MRI-NACT-Pilot labels.json files only when executed."
        )
    )
    parser.add_argument("--nact-root", type=Path, default=DEFAULT_NACT_ROOT)
    parser.add_argument("--label-xls", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME)
    parser.add_argument("--dry-run", action="store_true", help="Analyze and print counts without writing files.")
    parser.add_argument("--backup", action="store_true", help="Create labels.json.nact_label_fix.bak before overwriting.")
    args = parser.parse_args(argv)

    label_xls = args.label_xls or args.nact_root / DEFAULT_LABEL_FILE
    labels_by_subject = load_nact_clinical_labels(
        label_xls=label_xls,
        sheet_name=args.sheet_name,
        dataset_id=args.dataset_id,
    )
    summary = repair_processed_nact_labels(
        processed_root=args.processed_root,
        labels_by_subject=labels_by_subject,
        label_xls=label_xls,
        dataset_id=args.dataset_id,
        dry_run=args.dry_run,
        backup=args.backup,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_nact_clinical_labels(
    label_xls: Path,
    sheet_name: str = DEFAULT_SHEET_NAME,
    dataset_id: str = DATASET_ID,
) -> dict[str, dict[str, Any]]:
    if not label_xls.exists():
        raise FileNotFoundError(f"Missing Breast-MRI-NACT-Pilot label file: {label_xls}")

    rows = read_xls_rows(label_xls, sheet_name=sheet_name)
    required = {PATIENT_ID, PCR_COLUMN, "ER positive", "PR Positive", "HER2 Positive"}
    headers = set(rows[0].keys()) if rows else set()
    missing = sorted(required.difference(headers))
    if missing:
        raise ValueError(f"Missing required Breast-MRI-NACT-Pilot columns in {label_xls}: {missing}")

    labels_by_subject: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject_id = normalize_nact_subject_id(row.get(PATIENT_ID, ""))
        if not subject_id.startswith("UCSF-BR-"):
            continue
        labels_by_subject[subject_id] = build_nact_label_payload(
            subject_id=subject_id,
            row=row,
            label_xls=label_xls,
            dataset_id=dataset_id,
        )
    return labels_by_subject


def build_nact_label_payload(
    subject_id: str,
    row: dict[str, Any],
    label_xls: Path,
    dataset_id: str = DATASET_ID,
) -> dict[str, Any]:
    pcr_value = normalize_nact_pcr(row.get(PCR_COLUMN))
    er = decode_receptor(row.get("ER positive"))
    pr = decode_receptor(row.get("PR Positive"))
    her2 = decode_receptor(row.get("HER2 Positive"))
    molecular_subtype = normalize_molecular_subtype(row.get("HR_HER2_STATUS"), er=er, pr=pr, her2=her2)
    survival = build_survival_payload(row)
    response = {
        "clinical_response_code": numeric_or_none(row.get(PCR_COLUMN)),
        "clinical_response": decode_map(row.get(PCR_COLUMN), CLINICAL_RESPONSE_MAP),
        "pathologic_complete_response": pcr_value,
    }
    tumor_measurements = build_tumor_measurements(row)
    treatment = build_treatment_payload(row)
    payload = {
        "schema_version": "unified_labels_v1",
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "patient_id": subject_id,
        "timepoint": "unknown",
        "study_uid": "unknown",
        "pCR": pcr_value,
        "pcr_status": str(pcr_value) if pcr_value in {0, 1} else "unknown",
        "HER2": her2,
        "ER": er,
        "PR": pr,
        "molecular_subtype": molecular_subtype,
        "age": numeric_or_none(row.get("AGE at MRI1 (yrs)")),
        "BMI": None,
        "sex": "female",
        "race": clean_text(row.get("Race")),
        "race_ethnicity": clean_text(row.get("Race")),
        "caucasian_race": clean_text(row.get("Caucasian race")),
        "laterality": normalize_laterality(row.get("breast laterality")),
        "clinical_stage": "unknown",
        "pathologic_stage": "unknown",
        "tumor_size_cm": first_non_missing(row.get("Clinical size pre"), row.get("LD 1 (cm)")),
        "clinical_size_pre_cm": numeric_or_none(row.get("Clinical size pre")),
        "clinical_size_post_cm": numeric_or_none(row.get("Clinical size post")),
        "pathologic_size_cm": numeric_or_none(row.get("path size (cm)")),
        "lymph_node_positive_after_surgery": decode_yes_no(row.get("LN")),
        "pathology": {
            "histologic_type": clean_text(row.get("Cancer type")),
            "grade": None,
            "pathologic_size_cm": numeric_or_none(row.get("path size (cm)")),
            "lymph_node_positive_after_surgery": decode_yes_no(row.get("LN")),
        },
        "treatment": treatment,
        "response": response,
        "survival": survival,
        "tumor_measurements": tumor_measurements,
        "standardized_sources": {
            "pCR": [PCR_COLUMN],
            "HER2": ["HER2 Positive"],
            "ER": ["ER positive"],
            "PR": ["PR Positive"],
            "molecular_subtype": ["HR_HER2_STATUS", "HR_HER2_CATEGORY"],
            "survival": ["censor", "DFS time (weeks)", "recur type"],
        },
        "source_label_count": 1,
        "source_files": [str(label_xls)],
        "nact_core_labels": {},
        "raw_label_records": [
            {
                "source_file": str(label_xls),
                "key_column": PATIENT_ID,
                "labels": json_safe(row),
                "notes": "Breast-MRI-NACT-Pilot clinical labels rebuilt with dataset-specific pCR rule.",
            }
        ],
    }
    payload["nact_core_labels"] = compact_core_labels(payload)
    return json_safe(payload)


def normalize_nact_pcr(value: Any) -> int | None:
    """Breast-MRI-NACT-Pilot-specific pCR rule requested by the user.

    Clinical response:
    1 -> pCR, any other non-missing code -> non-pCR, blank -> missing.
    """

    if is_missing(value):
        return None
    number = int_or_none(value)
    if number is None:
        return 0
    return 1 if number == 1 else 0


def build_treatment_payload(row: dict[str, Any]) -> dict[str, Any]:
    taxol_code = numeric_or_none(row.get("AC only=0, taxol=1"))
    return {
        "chemo": clean_text(row.get("chemo")),
        "ac_only_or_taxol_code": taxol_code,
        "taxol_added": decode_yes_no(taxol_code),
        "surgery_type": normalize_surgery_type(row.get("Surgery type")),
        "surgery_type_raw": clean_text(row.get("Surgery type")),
        "mri_1_available": normalize_yes_blank(row.get("MRI 1")),
        "mri_2_available": normalize_yes_blank(row.get("MRI 2")),
        "mri_3_available": normalize_yes_blank(row.get("MRI 3")),
        "mri_4_available": normalize_yes_blank(row.get("MRI 4")),
    }


def build_survival_payload(row: dict[str, Any]) -> dict[str, Any]:
    censor = int_or_none(row.get("censor"))
    dfs_time_weeks = numeric_or_none(row.get("DFS time (weeks)"))
    dfs_event = 1 if censor == 0 else 0 if censor == 1 else None
    recurrence_type = normalize_recurrence_type(row.get("recur type"))
    return {
        "available": dfs_time_weeks is not None or dfs_event is not None,
        "censor": censor,
        "censor_label": "recurrence" if censor == 0 else "no_recurrence" if censor == 1 else "unknown",
        "recurrence_event": dfs_event,
        "recurrence_type": recurrence_type,
        "dfs_event": dfs_event,
        "dfs_time_weeks": dfs_time_weeks,
        "dfs_time_days": round(float(dfs_time_weeks) * 7.0, 3) if dfs_time_weeks is not None else None,
        "rfs_event": dfs_event,
        "rfs_time_weeks": dfs_time_weeks,
        "os_event": None,
        "os_time_days": None,
    }


def build_tumor_measurements(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ld_cm": {
            "mri_1": numeric_or_none(row.get("LD 1 (cm)")),
            "mri_2": numeric_or_none(row.get("LD 2 (cm)")),
            "mri_3": numeric_or_none(row.get("LD 3 (cm)")),
            "mri_4": numeric_or_none(row.get("LD 4 (cm)")),
        },
        "ser_volume_cc": {
            "mri_1": numeric_or_none(row.get("SER Volume 1 (cc)")),
            "mri_2": numeric_or_none(row.get("SER Volume 2 (cc)")),
            "mri_3": numeric_or_none(row.get("SER Volume 3 (cc)")),
            "mri_4": numeric_or_none(row.get("SER Volume 4 (cc)")),
        },
        "volume": {
            "mri_1": numeric_or_none(row.get("volume_1")),
            "mri_2": numeric_or_none(row.get("volume_2")),
            "mri_3": numeric_or_none(row.get("volume_3")),
            "mri_4": numeric_or_none(row.get("volume_4")),
        },
    }


def repair_processed_nact_labels(
    processed_root: Path,
    labels_by_subject: dict[str, dict[str, Any]],
    label_xls: Path,
    dataset_id: str = DATASET_ID,
    dry_run: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    processed_root = processed_root.expanduser().resolve(strict=False)
    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed Breast-MRI-NACT-Pilot dataset root: {processed_root}")

    label_paths = sorted(path for path in processed_root.rglob("labels.json") if is_nact_label_path(path, dataset_id))
    updated_labels = 0
    missing_subjects: list[str] = []
    labels_summary_rows: list[dict[str, Any]] = []

    for labels_path in label_paths:
        existing = read_json_object(labels_path)
        subject_id = normalize_nact_subject_id(
            existing.get("subject_id")
            or existing.get("patient_id")
            or infer_subject_from_label_path(labels_path, dataset_id)
        )
        if not subject_id or subject_id not in labels_by_subject:
            if subject_id:
                missing_subjects.append(subject_id)
            continue
        repaired = merge_existing_label(
            existing=existing,
            repaired=labels_by_subject[subject_id],
            label_xls=label_xls,
        )
        if not dry_run:
            if backup:
                backup_path = labels_path.with_name(labels_path.name + ".nact_label_fix.bak")
                if not backup_path.exists():
                    backup_path.write_text(
                        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            write_json_atomic(labels_path, repaired)
        updated_labels += 1
        labels_summary_rows.append(label_summary_row(labels_path, repaired))

    updated_csvs = update_processed_csvs(
        processed_root=processed_root,
        labels_by_subject=labels_by_subject,
        labels_summary_rows=labels_summary_rows,
        dataset_id=dataset_id,
        dry_run=dry_run,
    )
    return {
        "label_xls": str(label_xls),
        "processed_root": str(processed_root),
        "subjects_in_label_file": len(labels_by_subject),
        "nact_label_json_files_found": len(label_paths),
        "nact_label_json_files_updated": updated_labels,
        "missing_subjects_in_label_file": sorted(set(missing_subjects))[:50],
        "missing_subject_count": len(set(missing_subjects)),
        "csv_files_updated": updated_csvs,
        "dry_run": dry_run,
    }


def merge_existing_label(
    existing: dict[str, Any],
    repaired: dict[str, Any],
    label_xls: Path,
) -> dict[str, Any]:
    merged = dict(existing)
    preserved_keys = {
        "timepoint": existing.get("timepoint", repaired.get("timepoint", "unknown")),
        "study_uid": existing.get("study_uid", repaired.get("study_uid", "unknown")),
        "sample_dir": existing.get("sample_dir"),
        "split": existing.get("split"),
        "imaging_features": existing.get("imaging_features", {}),
    }
    merged.update(repaired)
    for key, value in preserved_keys.items():
        if value not in (None, ""):
            merged[key] = value

    old_records = existing.get("raw_label_records", [])
    if isinstance(old_records, list):
        clinical_name = label_xls.name.lower()
        non_clinical_records = [
            record
            for record in old_records
            if clinical_name not in str(record.get("source_file", "")).lower()
        ]
        merged["raw_label_records"] = [*non_clinical_records, *repaired.get("raw_label_records", [])]
        merged["source_label_count"] = len(merged["raw_label_records"])

    source_files = set(repaired.get("source_files", []))
    source_files.update(str(record.get("source_file", "")) for record in merged.get("raw_label_records", []) if record.get("source_file"))
    merged["source_files"] = sorted(source_files)
    merged["nact_core_labels"] = compact_core_labels(merged)
    return json_safe(merged)


def update_processed_csvs(
    processed_root: Path,
    labels_by_subject: dict[str, dict[str, Any]],
    labels_summary_rows: list[dict[str, Any]],
    dataset_id: str,
    dry_run: bool,
) -> list[str]:
    updated: list[str] = []
    for name in ("inference_manifest.csv", "training_manifest.csv", "labels_summary.csv"):
        for csv_path in sorted(processed_root.rglob(name)):
            if name == "labels_summary.csv" and labels_summary_rows:
                changed = update_labels_summary_csv(csv_path, labels_summary_rows, dataset_id, dry_run=dry_run)
                if changed:
                    updated.append(str(csv_path))
                continue
            changed = update_manifest_like_csv(csv_path, labels_by_subject, dataset_id, dry_run=dry_run)
            if changed:
                updated.append(str(csv_path))

    for inference_path in sorted(processed_root.rglob("inference_manifest.csv")):
        training_path = inference_path.with_name("training_manifest.csv")
        rows, fieldnames = read_csv_rows(inference_path)
        if not rows:
            continue
        training_rows = [row for row in rows if row.get("pcr_status") in {"0", "1"}]
        if not dry_run:
            write_csv_atomic(training_path, training_rows, fieldnames)
        updated.append(str(training_path))
    return sorted(set(updated))


def update_labels_summary_csv(
    csv_path: Path,
    repaired_rows: list[dict[str, Any]],
    dataset_id: str,
    dry_run: bool,
) -> bool:
    existing_rows, existing_fields = read_csv_rows(csv_path)
    fields = list(existing_fields or NACT_LABEL_SUMMARY_FIELDS)
    for field in NACT_LABEL_SUMMARY_FIELDS:
        if field not in fields:
            fields.append(field)
    repaired_by_key = {
        (
            row.get("dataset_id", dataset_id),
            row.get("subject_id", ""),
            row.get("timepoint") or "unknown",
            row.get("study_uid") or "unknown",
        ): row
        for row in repaired_rows
    }

    changed = False
    output_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for row in existing_rows:
        key = (
            row.get("dataset_id", ""),
            row.get("subject_id", ""),
            row.get("timepoint") or "unknown",
            row.get("study_uid") or "unknown",
        )
        if row.get("dataset_id") == dataset_id and key in repaired_by_key:
            output_rows.append(repaired_by_key[key])
            seen_keys.add(key)
            changed = True
        else:
            output_rows.append(row)

    for key, row in repaired_by_key.items():
        if key in seen_keys:
            continue
        output_rows.append(row)
        changed = True

    if changed and not dry_run:
        write_csv_atomic(csv_path, output_rows, fields)
    return changed


def update_manifest_like_csv(
    csv_path: Path,
    labels_by_subject: dict[str, dict[str, Any]],
    dataset_id: str,
    dry_run: bool,
) -> bool:
    rows, fieldnames = read_csv_rows(csv_path)
    if not rows:
        return False
    changed = False
    for row in rows:
        if row.get("dataset_id", "") not in {"", dataset_id}:
            continue
        subject_id = normalize_nact_subject_id(row.get("subject_id") or row.get("patient_uid") or row.get("patient_id"))
        labels = labels_by_subject.get(subject_id)
        if not labels:
            continue
        update_row_with_public_labels(row, labels)
        changed = True
    if changed and not dry_run:
        write_csv_atomic(csv_path, rows, fieldnames)
    return changed


def read_xls_rows(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    try:
        return read_xls_rows_with_xlrd(path, sheet_name)
    except ModuleNotFoundError:
        return read_xls_rows_with_pandas(path, sheet_name)


def read_xls_rows_with_xlrd(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    import xlrd

    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name(sheet_name) if sheet_name in workbook.sheet_names() else workbook.sheet_by_index(0)
    headers = _unique_headers([clean_header(sheet.cell_value(0, col)) for col in range(sheet.ncols)])
    rows: list[dict[str, Any]] = []
    for row_index in range(1, sheet.nrows):
        row = _row_dict(headers, [sheet.cell_value(row_index, col) for col in range(sheet.ncols)])
        if any(not is_missing(value) for value in row.values()):
            rows.append(row)
    return rows


def read_xls_rows_with_pandas(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_excel(path, sheet_name=sheet_name)
    frame = frame.dropna(how="all")
    frame.columns = [clean_header(column) for column in frame.columns]
    return [json_safe(record) for record in frame.to_dict("records")]


def clean_header(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def normalize_molecular_subtype(value: Any, er: str, pr: str, her2: str) -> str:
    if not is_missing(value):
        key = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
        if key in HR_HER2_STATUS_MAP:
            return HR_HER2_STATUS_MAP[key]
        return str(value).strip()
    if her2 == "positive":
        return "HER2-positive"
    if er == "positive" or pr == "positive":
        return "HR+/HER2-"
    if er == "negative" and pr == "negative" and her2 == "negative":
        return "Triple-negative"
    return "unknown"


def normalize_surgery_type(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    text = str(value).strip()
    key = text.upper()
    return SURGERY_TYPE_MAP.get(key, text)


def normalize_recurrence_type(value: Any) -> str:
    if is_missing(value):
        return "none_or_not_reported"
    text = str(value).strip().lower()
    return RECURRENCE_TYPE_MAP.get(text, str(value).strip())


def decode_receptor(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    number = int_or_none(value)
    if number is None:
        return str(value).strip()
    return RECEPTOR_MAP.get(number, str(number))


def decode_yes_no(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    number = int_or_none(value)
    if number is None:
        return str(value).strip()
    return YES_NO_MAP.get(number, str(number))


def normalize_yes_blank(value: Any) -> str:
    if is_missing(value):
        return "no_or_not_reported"
    text = str(value).strip().lower()
    if text == "yes":
        return "yes"
    if text == "no":
        return "no"
    return str(value).strip()


def decode_map(value: Any, mapping: dict[int, str]) -> str:
    if is_missing(value):
        return "unknown"
    number = int_or_none(value)
    if number is None:
        return str(value).strip()
    return mapping.get(number, str(number))


def normalize_laterality(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    text = str(value).strip().lower()
    if text.startswith("l"):
        return "left"
    if text.startswith("r"):
        return "right"
    if text.startswith("b"):
        return "bilateral"
    return str(value).strip()


def normalize_nact_subject_id(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().replace("_", "-").upper()
    match = re.search(r"UCSF-BR-?(\d+)", text)
    if match:
        return f"UCSF-BR-{int(match.group(1)):02d}"
    digits = "".join(char for char in text if char.isdigit())
    return f"UCSF-BR-{int(digits):02d}" if digits else text


def _unique_headers(values: Sequence[Any]) -> list[str]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for index, value in enumerate(values):
        base = str(value).strip() if value not in (None, "") else f"unnamed_{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        headers.append(base if count == 0 else f"{base}.{count}")
    return headers


def _row_dict(headers: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in zip(headers, values):
        if key.startswith("unnamed_"):
            continue
        output[key] = clean_value(value)
    return output


def is_nact_label_path(path: Path, dataset_id: str) -> bool:
    parts = {part.lower() for part in path.parts}
    if dataset_id.lower() in parts:
        return True
    existing = read_json_object(path)
    return existing.get("dataset_id") == dataset_id


def infer_subject_from_label_path(path: Path, dataset_id: str) -> str:
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if dataset_id.lower() in lowered:
        index = lowered.index(dataset_id.lower())
        if index + 1 < len(parts):
            return normalize_nact_subject_id(parts[index + 1])
    return ""


def label_summary_row(labels_path: Path, labels: dict[str, Any]) -> dict[str, Any]:
    survival = labels.get("survival", {}) if isinstance(labels.get("survival"), dict) else {}
    return {
        "dataset_id": labels.get("dataset_id", DATASET_ID),
        "subject_id": labels.get("subject_id", ""),
        "timepoint": labels.get("timepoint", "unknown"),
        "study_uid": labels.get("study_uid", "unknown"),
        "labels_json": json.dumps(public_label_fields(labels), ensure_ascii=False, sort_keys=True),
        "pCR": labels.get("pCR"),
        "pcr_status": labels.get("pcr_status", "unknown"),
        "HER2": labels.get("HER2", "unknown"),
        "ER": labels.get("ER", "unknown"),
        "PR": labels.get("PR", "unknown"),
        "molecular_subtype": labels.get("molecular_subtype", "unknown"),
        "laterality": labels.get("laterality", "unknown"),
        "age": labels.get("age"),
        "race": labels.get("race"),
        "tumor_size_cm": labels.get("tumor_size_cm"),
        "dfs_event": survival.get("dfs_event"),
        "dfs_time_weeks": survival.get("dfs_time_weeks"),
        "source_label_count": labels.get("source_label_count", 1),
        "split": labels.get("split", split_for_subject(labels.get("dataset_id", DATASET_ID), labels.get("subject_id", ""))),
    }


def update_row_with_public_labels(row: dict[str, str], labels: dict[str, Any]) -> None:
    public = public_label_fields(labels)
    row["labels_json"] = json.dumps(public, ensure_ascii=False, sort_keys=True)
    for key in ("pCR", "pcr_status", "HER2", "ER", "PR"):
        if key in row:
            row[key] = csv_value(labels.get(key))
    if "modalities" in row:
        row.setdefault("dataset_id", DATASET_ID)


def public_label_fields(labels: dict[str, Any]) -> dict[str, Any]:
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
        "race",
        "laterality",
        "clinical_stage",
        "pathologic_stage",
        "tumor_size_cm",
        "clinical_size_pre_cm",
        "clinical_size_post_cm",
        "pathologic_size_cm",
        "lymph_node_positive_after_surgery",
        "pathology",
        "treatment",
        "response",
        "survival",
        "tumor_measurements",
    ]
    return {key: labels.get(key) for key in keys}


def compact_core_labels(labels: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pCR",
        "pcr_status",
        "HER2",
        "ER",
        "PR",
        "molecular_subtype",
        "age",
        "sex",
        "race",
        "laterality",
        "tumor_size_cm",
        "clinical_size_pre_cm",
        "clinical_size_post_cm",
        "pathologic_size_cm",
        "lymph_node_positive_after_surgery",
        "pathology",
        "treatment",
        "response",
        "survival",
        "tumor_measurements",
    ]
    return {key: labels.get(key) for key in keys if known_value(labels.get(key))}


def known_value(value: Any) -> bool:
    if value in (None, "", "unknown"):
        return False
    if isinstance(value, dict):
        return any(known_value(item) for item in value.values())
    if isinstance(value, list):
        return any(known_value(item) for item in value)
    return True


def split_for_subject(dataset_id: str, subject_id: str, seed: int = 2026) -> str:
    import hashlib

    digest = hashlib.sha256(f"{seed}|{dataset_id}|{subject_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})
    tmp.replace(path)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def clean_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def clean_text(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    return str(value).strip()


def numeric_or_none(value: Any) -> float | int | None:
    if is_missing(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def int_or_none(value: Any) -> int | None:
    number = numeric_or_none(value)
    if number is None:
        return None
    return int(number)


def first_non_missing(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            return clean_value(value)
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    return text.upper() in MISSING_TOKENS


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
