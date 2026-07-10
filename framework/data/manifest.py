from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.data.schemas import ManifestRecord
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy, resolve_path

MANIFEST_FIELDS = [
    "dataset_id",
    "patient_uid",
    "study_uid",
    "original_patient_mapping",
    "original_path",
    "standardized_data_path",
    "timepoint",
    "laterality",
    "lesion_id",
    "available_modalities",
    "dce_phase_count",
    "image_shape",
    "spacing",
    "clinical_labels",
    "pathology_labels",
    "molecular_subtype",
    "response_label",
    "pcr_status",
    "survival_followup",
    "roi_status",
    "missing_modalities",
    "qc_status",
    "exclusion_reason",
    "preprocessing_version",
    "split",
    "source_audit_json",
]


def build_manifest_from_audit(audit_run_dir: str | Path) -> list[ManifestRecord]:
    suppress_warnings()
    run_dir = Path(audit_run_dir).expanduser().resolve(strict=False)
    dataset_files = sorted(run_dir.glob("datasets/*/audit.json"))
    records: list[ManifestRecord] = []
    for audit_json in progress_iter(dataset_files, total=len(dataset_files), desc="Build manifest", unit="dataset"):
        with audit_json.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        records.extend(_records_for_dataset(result, source_audit_json=str(audit_json)))
    return sorted(records, key=lambda row: (row.dataset_id, row.patient_uid, row.study_uid))


def write_manifest(
    records: list[ManifestRecord],
    output_dir: str | Path,
    policy: PathPolicy | None = None,
) -> dict[str, str]:
    suppress_warnings()
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    if policy is not None:
        policy.ensure_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        record.to_dict()
        for record in progress_iter(records, total=len(records), desc="Write manifest", unit="record")
    ]
    paths = {
        "csv": str(atomic_write_csv(out_dir / "master_manifest.csv", rows, MANIFEST_FIELDS, policy=policy)),
        "json": str(atomic_write_json(out_dir / "master_manifest.json", rows, policy=policy)),
    }
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return paths
    frame = pd.DataFrame(rows)
    xlsx_path = out_dir / "master_manifest.xlsx"
    parquet_path = out_dir / "master_manifest.parquet"
    if policy is not None:
        policy.assert_write_allowed(xlsx_path)
        policy.assert_write_allowed(parquet_path)
    try:
        frame.to_excel(xlsx_path, index=False)
        paths["excel"] = str(xlsx_path)
    except Exception:
        pass
    try:
        frame.to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception:
        pass
    return paths


def _records_for_dataset(result: dict[str, Any], source_audit_json: str) -> list[ManifestRecord]:
    dataset_id = result["dataset_id"]
    expected_modalities = ("T1", "T2", "DWI", "ADC", "DCE")
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "modalities": set(),
            "dce_phase_count": 0,
            "shapes": set(),
            "spacings": set(),
            "needs_review": False,
        }
    )
    series_rows = result.get("dicom", {}).get("series_statistics", [])
    for series in progress_iter(
        series_rows,
        total=len(series_rows),
        desc=f"Manifest {dataset_id}",
        unit="series",
    ):
        key = (series.get("patient_uid", "unknown"), series.get("study_uid", "unknown"))
        item = grouped[key]
        modality = series.get("inferred_modality", "unknown")
        if modality and modality != "unknown":
            item["modalities"].add(modality)
        if modality == "DCE":
            item["dce_phase_count"] = max(
                int(item["dce_phase_count"]),
                _safe_int(series.get("temporal_position_count")),
            )
        for shape in series.get("dimensions", []) or []:
            if shape:
                item["shapes"].add(str(shape))
        for spacing in series.get("pixel_spacing_values", []) or []:
            if spacing:
                item["spacings"].add(str(spacing))
        if series.get("modality_status") != "identified":
            item["needs_review"] = True

    records = []
    for (patient_uid, study_uid), item in grouped.items():
        modalities = tuple(sorted(item["modalities"]))
        missing = tuple(modality for modality in expected_modalities if modality not in modalities)
        records.append(
            ManifestRecord(
                dataset_id=dataset_id,
                patient_uid=patient_uid,
                study_uid=study_uid,
                available_modalities=modalities,
                missing_modalities=missing,
                dce_phase_count=int(item["dce_phase_count"]),
                image_shape=";".join(sorted(item["shapes"])) or "unknown",
                spacing=";".join(sorted(item["spacings"])) or "unknown",
                qc_status="needs_review" if item["needs_review"] else "audit_only",
                source_audit_json=source_audit_json,
            )
        )
    return records


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def add_manifest_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("manifest", help="Build a preliminary master manifest from audit output.")
    parser.add_argument("--audit-run", required=True, help="Audit run directory, e.g. outputs/audit/<run_id>.")
    parser.add_argument("--output-dir", required=True, help="Manifest output directory.")
    parser.set_defaults(func=run_from_args)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    records = build_manifest_from_audit(args.audit_run)
    output_dir = resolve_path(args.output_dir)
    paths = write_manifest(records, output_dir)
    print(f"Wrote {len(records)} manifest records")
    for kind, path in sorted(paths.items()):
        print(f"{kind}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a master manifest from audit output.")
    parser.add_argument("--audit-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return run_from_args(parser.parse_args(argv))
