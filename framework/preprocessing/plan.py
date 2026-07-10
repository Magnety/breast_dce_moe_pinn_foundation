from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings

PLAN_FIELDS = [
    "dataset_id",
    "patient_uid",
    "study_uid",
    "available_modalities",
    "missing_modalities",
    "qc_status",
    "preprocessing_status",
    "reason",
    "standardized_data_path",
]


def build_preprocessing_plan(manifest_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    suppress_warnings()
    manifest_path = Path(manifest_csv).expanduser().resolve(strict=False)
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(manifest_path)
    plan_rows = [
        _plan_row(row)
        for row in progress_iter(rows, total=len(rows), desc="Build preprocess plan", unit="record")
    ]
    summary = {
        "manifest_csv": str(manifest_path),
        "output_dir": str(out_dir),
        "row_count": len(plan_rows),
        "status_counts": dict(Counter(row["preprocessing_status"] for row in plan_rows)),
    }
    atomic_write_csv(out_dir / "preprocessing_plan.csv", plan_rows, PLAN_FIELDS)
    atomic_write_json(out_dir / "preprocessing_plan_summary.json", summary)
    return summary


def _plan_row(row: dict[str, str]) -> dict[str, str]:
    qc_status = row.get("qc_status", "needs_review")
    standardized_path = row.get("standardized_data_path", "not_available")
    if qc_status == "needs_review":
        status = "blocked_needs_review"
        reason = "Audit marked this patient/study as needing review."
    elif standardized_path not in ("", "not_available", "unknown"):
        status = "already_standardized"
        reason = "Manifest already points to standardized data."
    else:
        status = "pending_dataset_adapter"
        reason = "Dataset-specific DICOM conversion adapter must be selected after audit review."
    return {
        "dataset_id": row.get("dataset_id", ""),
        "patient_uid": row.get("patient_uid", ""),
        "study_uid": row.get("study_uid", ""),
        "available_modalities": row.get("available_modalities", ""),
        "missing_modalities": row.get("missing_modalities", ""),
        "qc_status": qc_status,
        "preprocessing_status": status,
        "reason": reason,
        "standardized_data_path": standardized_path,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
