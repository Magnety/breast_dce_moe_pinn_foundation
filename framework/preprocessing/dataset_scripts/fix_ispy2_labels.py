from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DATASET_ID = "ispy2"
DEFAULT_ISPY2_ROOT = Path("H:/breast/ISPY2")
DEFAULT_DATASET_ROOT = Path(os.environ.get("BREAST_MRI_DATASET_ROOT", "/home/ubuntu/liuyiyao1/multimodal_dataset"))
DEFAULT_PROCESSED_ROOT = DEFAULT_DATASET_ROOT / "full_v4_ispy2"
DEFAULT_CLINICAL_FILE = "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"

PATIENT_ID = "Patient_ID"
PCR_COLUMN = "pCR"
HER2_COLUMN = "HER2"
HR_COLUMN = "HR"
MP_COLUMN = "MP"

MISSING_TOKENS = {"", "NA", "N/A", "NC", "NP", "NONE", "NULL", "NAN"}
RECEPTOR_MAP = {0: "negative", 1: "positive"}

ISPY2_LABEL_SUMMARY_FIELDS = [
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
    "HR",
    "molecular_subtype",
    "age",
    "race",
    "ethnicity",
    "menopause",
    "treatment_arm",
    "source_label_count",
    "split",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair ISPY2 labels in an already-built multimodal dataset. "
            "The script rebuilds labels from ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx "
            "and overwrites ISPY2 labels.json files only when executed."
        )
    )
    parser.add_argument("--ispy2-root", type=Path, default=DEFAULT_ISPY2_ROOT)
    parser.add_argument("--clinical-xlsx", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dry-run", default=False, help="Analyze and print counts without writing files.")
    parser.add_argument("--backup", default=False, help="Create labels.json.ispy2_label_fix.bak before overwriting.")
    args = parser.parse_args(argv)
    print(f"Using ISPY2 root: {args.ispy2_root}")
    clinical_xlsx = args.clinical_xlsx or args.ispy2_root / DEFAULT_CLINICAL_FILE
    print(f"Loading ISPY2 clinical labels from {clinical_xlsx}...")
    labels_by_subject = load_ispy2_clinical_labels(clinical_xlsx, dataset_id=args.dataset_id)
    print(f"Loaded labels for {len(labels_by_subject)} subjects from ISPY2 clinical file.")
    summary = repair_processed_ispy2_labels(
        processed_root=args.processed_root,
        labels_by_subject=labels_by_subject,
        clinical_xlsx=clinical_xlsx,
        dataset_id=args.dataset_id,
        dry_run=args.dry_run,
        backup=args.backup,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_ispy2_clinical_labels(clinical_xlsx: Path, dataset_id: str = DATASET_ID) -> dict[str, dict[str, Any]]:
    if not clinical_xlsx.exists():
        raise FileNotFoundError(f"Missing ISPY2 clinical label file: {clinical_xlsx}")

    rows = read_xlsx_first_sheet(clinical_xlsx)
    if not rows:
        raise ValueError(f"No rows found in ISPY2 clinical label file: {clinical_xlsx}")
    headers = _unique_headers([str(value).strip() if value is not None else "" for value in rows[0]])
    required = {PATIENT_ID, PCR_COLUMN, HER2_COLUMN, HR_COLUMN}
    missing = sorted(required.difference(headers))
    if missing:
        raise ValueError(f"Missing required ISPY2 clinical columns in {clinical_xlsx}: {missing}")

    labels_by_subject: dict[str, dict[str, Any]] = {}
    for row_values in rows[1:]:
        row = _row_dict(headers, row_values)
        subject_id = normalize_ispy2_subject_id(row.get(PATIENT_ID, ""))
        if not subject_id:
            continue
        labels_by_subject[subject_id] = build_ispy2_label_payload(
            subject_id=subject_id,
            row=row,
            clinical_xlsx=clinical_xlsx,
            dataset_id=dataset_id,
        )
    return labels_by_subject


def build_ispy2_label_payload(
    subject_id: str,
    row: dict[str, Any],
    clinical_xlsx: Path,
    dataset_id: str = DATASET_ID,
) -> dict[str, Any]:
    pcr_value = normalize_binary_label(row.get(PCR_COLUMN))
    her2 = decode_receptor(row.get(HER2_COLUMN))
    hormone_receptor = decode_receptor(row.get(HR_COLUMN))
    er, pr = infer_er_pr_from_hr(row.get(HR_COLUMN))
    molecular_subtype = build_molecular_subtype(row.get(HR_COLUMN), row.get(HER2_COLUMN))
    mammoprint = normalize_binary_label(row.get(MP_COLUMN))
    survival = build_survival_payload(row)
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
        "HR": hormone_receptor,
        "hormone_receptor": hormone_receptor,
        "molecular_subtype": molecular_subtype,
        "age": numeric_or_none(row.get("Age_at_Screening")),
        "BMI": None,
        "sex": "female",
        "race": clean_text(row.get("Race")),
        "race_ethnicity": clean_text(row.get("Race")),
        "ethnicity": clean_text(row.get("ethnicity")),
        "menopause": normalize_menopause(row.get("menopausal_status")),
        "menopausal_status": clean_text(row.get("menopausal_status")),
        "treatment": {
            "arm": clean_text(row.get("Arm")),
            "neoadjuvant_therapy": clean_text(row.get("Arm")),
        },
        "response": {
            "pathologic_complete_response": pcr_value,
            "mammoprint": mammoprint,
            "mammoprint_status": decode_binary(row.get(MP_COLUMN)),
        },
        "survival": survival,
        "clinical_stage": "unknown",
        "pathologic_stage": "unknown",
        "tumor_size_cm": None,
        "pathology": {
            "grade": None,
            "histologic_type": None,
        },
        "standardized_sources": {
            "pCR": [PCR_COLUMN],
            "HER2": [HER2_COLUMN],
            "HR": [HR_COLUMN],
            "ER": ["inferred negative only when HR=0; otherwise unavailable in this clinical file"],
            "PR": ["inferred negative only when HR=0; otherwise unavailable in this clinical file"],
            "molecular_subtype": [HR_COLUMN, HER2_COLUMN],
            "survival": [],
        },
        "source_label_count": 1,
        "source_files": [str(clinical_xlsx)],
        "ispy2_core_labels": {},
        "raw_label_records": [
            {
                "source_file": str(clinical_xlsx),
                "key_column": PATIENT_ID,
                "labels": json_safe(row),
                "notes": (
                    "ISPY2 clinical labels rebuilt from Imaging Cohort 1 clinical file. "
                    "This file has HR but no independent ER/PR survival columns."
                ),
            }
        ],
    }
    payload["ispy2_core_labels"] = compact_core_labels(payload)
    return json_safe(payload)


def repair_processed_ispy2_labels(
    processed_root: Path,
    labels_by_subject: dict[str, dict[str, Any]],
    clinical_xlsx: Path,
    dataset_id: str = DATASET_ID,
    dry_run: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    processed_root = processed_root.expanduser().resolve(strict=False)
    if not processed_root.exists():
        raise FileNotFoundError(f"Missing processed ISPY2 dataset root: {processed_root}")

    label_paths = sorted(path for path in processed_root.rglob("labels.json") if is_ispy2_label_path(path, dataset_id))
    updated_labels = 0
    missing_subjects: list[str] = []
    labels_summary_rows: list[dict[str, Any]] = []

    for labels_path in label_paths:
        print(f"Processing {labels_path}...")
        existing = read_json_object(labels_path)
        subject_id = normalize_ispy2_subject_id(
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
            clinical_xlsx=clinical_xlsx,
        )
        if not dry_run:
            if backup:
                backup_path = labels_path.with_name(labels_path.name + ".ispy2_label_fix.bak")
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
        "clinical_xlsx": str(clinical_xlsx),
        "processed_root": str(processed_root),
        "subjects_in_clinical_file": len(labels_by_subject),
        "ispy2_label_json_files_found": len(label_paths),
        "ispy2_label_json_files_updated": updated_labels,
        "missing_subjects_in_clinical_file": sorted(set(missing_subjects))[:50],
        "missing_subject_count": len(set(missing_subjects)),
        "csv_files_updated": updated_csvs,
        "dry_run": dry_run,
    }


def merge_existing_label(
    existing: dict[str, Any],
    repaired: dict[str, Any],
    clinical_xlsx: Path,
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
        clinical_name = clinical_xlsx.name.lower()
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
    merged["ispy2_core_labels"] = compact_core_labels(merged)
    return json_safe(merged)


def build_survival_payload(row: dict[str, Any]) -> dict[str, Any]:
    survival_terms = (
        "survival",
        "os",
        "dfs",
        "rfs",
        "recurrence",
        "relapse",
        "death",
        "follow",
        "censor",
        "event",
    )
    found = {}
    for key, value in row.items():
        key_text = str(key).strip().lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", key_text) if token]
        if not any(term in tokens or term in key_text for term in survival_terms):
            continue
        if is_missing(value):
            continue
        found[key] = clean_value(value)
    return {
        "available": bool(found),
        "source_columns": list(found.keys()),
        "values": found,
        "note": "No explicit survival columns were found in ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx." if not found else "",
    }


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
    fields = list(existing_fields or ISPY2_LABEL_SUMMARY_FIELDS)
    for field in ISPY2_LABEL_SUMMARY_FIELDS:
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
        subject_id = normalize_ispy2_subject_id(row.get("subject_id") or row.get("patient_uid") or row.get("patient_id"))
        labels = labels_by_subject.get(subject_id)
        if not labels:
            continue
        update_row_with_public_labels(row, labels)
        changed = True
    if changed and not dry_run:
        write_csv_atomic(csv_path, rows, fieldnames)
    return changed


def read_xlsx_first_sheet(path: Path) -> list[list[Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive, ns)
        sheet_names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
        if not sheet_names:
            return []
        sheet_xml = archive.read(sorted(sheet_names)[0])

    root = ET.fromstring(sheet_xml)
    rows: list[list[Any]] = []
    for row_node in root.findall(".//a:sheetData/a:row", ns):
        values_by_index: dict[int, Any] = {}
        for cell in row_node.findall("a:c", ns):
            ref = cell.attrib.get("r", "")
            index = excel_column_index(ref)
            values_by_index[index] = read_xlsx_cell(cell, shared_strings, ns)
        if values_by_index:
            max_index = max(values_by_index)
            rows.append([values_by_index.get(index, "") for index in range(1, max_index + 1)])
    return rows


def read_shared_strings(archive: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("a:si", ns):
        texts = [node.text or "" for node in item.findall(".//a:t", ns)]
        values.append("".join(texts))
    return values


def read_xlsx_cell(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> Any:
    value_node = cell.find("a:v", ns)
    if value_node is None:
        inline_text = cell.find(".//a:t", ns)
        return inline_text.text if inline_text is not None else ""
    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    return numeric_or_text(raw)


def excel_column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - 64
    return index


def numeric_or_text(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def normalize_binary_label(value: Any) -> int | None:
    if is_missing(value):
        return None
    number = int_or_none(value)
    if number in {0, 1}:
        return number
    return None


def decode_binary(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    number = int_or_none(value)
    if number == 0:
        return "negative"
    if number == 1:
        return "positive"
    return str(value).strip()


def decode_receptor(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    number = int_or_none(value)
    if number is None:
        return str(value).strip()
    return RECEPTOR_MAP.get(number, str(number))


def infer_er_pr_from_hr(value: Any) -> tuple[str, str]:
    number = int_or_none(value)
    if number == 0:
        return "negative", "negative"
    return "unknown", "unknown"


def build_molecular_subtype(hr_value: Any, her2_value: Any) -> str:
    hr = int_or_none(hr_value)
    her2 = int_or_none(her2_value)
    if hr is None or her2 is None:
        return "unknown"
    if hr == 1 and her2 == 1:
        return "HR+/HER2+"
    if hr == 1 and her2 == 0:
        return "HR+/HER2-"
    if hr == 0 and her2 == 1:
        return "HER2-enriched"
    if hr == 0 and her2 == 0:
        return "Triple-negative"
    return "unknown"


def normalize_menopause(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    text = str(value).strip().lower()
    if "pre" in text:
        return "pre"
    if "peri" in text:
        return "peri"
    if "post" in text:
        return "post"
    if "age > 50" in text:
        return "post_like_age_gt_50"
    if "age < 50" in text:
        return "pre_like_age_lt_50"
    return str(value).strip()


def normalize_ispy2_subject_id(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).strip().replace("_", "-")
    if text.upper().startswith("ISPY2-"):
        suffix = text.split("-", 1)[1]
        return f"ISPY2-{normalize_numeric_id(suffix)}"
    digits = "".join(char for char in text if char.isdigit())
    return f"ISPY2-{normalize_numeric_id(digits)}" if digits else text


def normalize_numeric_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


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


def is_ispy2_label_path(path: Path, dataset_id: str) -> bool:
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
            return normalize_ispy2_subject_id(parts[index + 1])
    return ""


def label_summary_row(labels_path: Path, labels: dict[str, Any]) -> dict[str, Any]:
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
        "HR": labels.get("HR", "unknown"),
        "molecular_subtype": labels.get("molecular_subtype", "unknown"),
        "age": labels.get("age"),
        "race": labels.get("race"),
        "ethnicity": labels.get("ethnicity"),
        "menopause": labels.get("menopause"),
        "treatment_arm": labels.get("treatment", {}).get("arm") if isinstance(labels.get("treatment"), dict) else "",
        "source_label_count": labels.get("source_label_count", 1),
        "split": labels.get("split", split_for_subject(labels.get("dataset_id", DATASET_ID), labels.get("subject_id", ""))),
    }


def update_row_with_public_labels(row: dict[str, str], labels: dict[str, Any]) -> None:
    public = public_label_fields(labels)
    row["labels_json"] = json.dumps(public, ensure_ascii=False, sort_keys=True)
    for key in ("pCR", "pcr_status", "HER2", "ER", "PR", "HR"):
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
        "HR",
        "hormone_receptor",
        "molecular_subtype",
        "age",
        "BMI",
        "sex",
        "race",
        "ethnicity",
        "menopause",
        "treatment",
        "response",
        "survival",
    ]
    return {key: labels.get(key) for key in keys}


def compact_core_labels(labels: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "pCR",
        "pcr_status",
        "HER2",
        "ER",
        "PR",
        "HR",
        "hormone_receptor",
        "molecular_subtype",
        "age",
        "sex",
        "race",
        "ethnicity",
        "menopause",
        "treatment",
        "response",
        "survival",
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
