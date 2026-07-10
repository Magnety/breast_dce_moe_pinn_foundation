"""Build a unified per-volume manifest for the breast_dce_moe_pinn_foundation pipeline.

The ``MultimodalManifestDataset`` expects a per-volume manifest with one row per
3D ``.npy`` file. Eight datasets sit under
``/home/ubuntu/liuyiyao1/multimodal_dataset/full_v4_*`` with their own
``training_manifest.csv`` (one row per study) and per-study ``series_manifest.json``
(volumes with a Linux-friendly ``relative_path``). Their stored ``sample_dir``
columns still point at Windows paths from preprocessing, so we resolve everything
through ``relative_path`` instead.

This script walks every dataset, expands each study into its individual volume
records, harmonizes labels across the heterogeneous JSON shapes, and writes a
single CSV that the foundation pipeline can consume directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ROOT = Path("/home/ubuntu/liuyiyao1/multimodal_dataset")
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "foundation_manifest_full_v4.csv"
)

DATASET_DIRS: tuple[str, ...] = (
    "full_v4_acrin_contralateral",
    "full_v4_advanced_mri_breast_lesions",
    "full_v4_breast_mri_nact_pilot",
    "full_v4_duke",
    "full_v4_fujian_pcr",
    "full_v4_ispy1",
    "full_v4_ispy2",
    "full_v4_tcga_brca",
    "full_v4_acrin_6698"

)

OUTPUT_FIELDS: tuple[str, ...] = (
    "patient_id",
    "sample_id",
    "dataset_id",
    "visit_timepoint",
    "split",
    "modality",
    "phase_index",
    "relative_time",
    "path",
    "pCR",
    "HER2",
    "ER",
    "PR",
    "HR",
    "molecular_subtype",
    "survival_time",
    "survival_event",
)

POSITIVE_TOKENS = {"1", "true", "yes", "positive", "pos", "pcr"}
NEGATIVE_TOKENS = {"0", "false", "no", "negative", "neg", "non-pcr", "non_pcr"}
MISSING_TOKENS = {
    "",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "missing",
    "not_available",
    "-1",
}

SUBTYPE_TO_ID = {
    "luminal_a": 0,
    "luminal-a": 0,
    "hrposher2neg": 0,  # I-SPY2 / NACT pilot label
    "hr_pos_her2_neg": 0,
    "luminal_b": 1,
    "luminal-b": 1,
    "her2pos": 2,
    "her2_pos": 2,
    "her2-enriched": 2,
    "her2_enriched": 2,
    "her2+": 2,
    "tripleneg": 3,
    "triple_negative": 3,
    "triple-negative": 3,
    "tnbc": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Multimodal dataset root containing full_v4_* directories.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Destination CSV path.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_DIRS),
        help="Optional subset of full_v4_* directory names to scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.expanduser().resolve(strict=False)
    out: Path = args.out.expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for dir_name in args.datasets:
        dataset_dir = root / dir_name
        if not dataset_dir.exists():
            print(f"[skip] {dir_name}: directory missing")
            continue
        manifest_rows, manifest_kind = _read_dataset_manifest(dataset_dir)
        if not manifest_rows:
            print(f"[skip] {dir_name}: no manifest rows in training_manifest.csv or inference_manifest.csv")
            continue

        per_dataset_rows: list[dict[str, Any]] = []
        missing_volumes = 0
        missing_studies = 0
        for study in manifest_rows:
            study_records, missed = _expand_study(study, dataset_dir, manifest_kind)
            missing_volumes += missed
            if not study_records:
                missing_studies += 1
                continue
            per_dataset_rows.extend(study_records)
        rows.extend(per_dataset_rows)

        split_counts = Counter(row["split"] for row in per_dataset_rows)
        modality_counts = Counter(row["modality"] for row in per_dataset_rows)
        summary.append(
            {
                "dataset_dir": dir_name,
                "studies_in": len(manifest_rows),
                "studies_missing_volumes": missing_studies,
                "volumes_missing_files": missing_volumes,
                "volumes_out": len(per_dataset_rows),
                "splits": dict(split_counts),
                "modalities": dict(modality_counts),
                "manifest_kind": manifest_kind,
            }
        )

    _write_manifest(out, rows)
    _print_summary(summary, out, len(rows))
    return 0


def _read_dataset_manifest(dataset_dir: Path) -> tuple[list[dict[str, str]], str]:
    """Return manifest rows plus a tag describing where they came from.

    Falls back to ``inference_manifest.csv`` when ``training_manifest.csv`` is empty.
    Datasets in the inference-only fallback get every row's ``split`` rewritten to
    ``train`` so they can still feed self-supervised pretraining.
    """

    training_path = dataset_dir / "training_manifest.csv"
    if training_path.exists():
        rows = _read_csv(training_path)
        if rows:
            return rows, "training"
    inference_path = dataset_dir / "inference_manifest.csv"
    if inference_path.exists():
        rows = _read_csv(inference_path)
        for row in rows:
            row["split"] = "train"
        return rows, "inference"
    return [], "missing"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _expand_study(
    study: dict[str, str],
    dataset_dir: Path,
    manifest_kind: str,
) -> tuple[list[dict[str, Any]], int]:
    dataset_id = (study.get("dataset_id") or "").strip()
    patient_id = (study.get("patient_uid") or study.get("subject_id") or "").strip()
    timepoint = (study.get("timepoint") or "").strip()
    if not (dataset_id and patient_id and timepoint):
        return [], 0

    inner_root = _resolve_inner_root(dataset_dir, dataset_id)
    study_dir = inner_root / patient_id / timepoint
    series_manifest = study_dir / "series_manifest.json"
    if not series_manifest.exists():
        return [], 0

    try:
        payload = json.loads(series_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0

    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        return [], 0

    labels_payload = _load_labels(study, study_dir)
    base_labels = _resolve_labels(study, labels_payload)
    sample_id = f"{patient_id}__{timepoint}"
    split = (study.get("split") or "train").strip() or "train"

    out_rows: list[dict[str, Any]] = []
    missing = 0
    for image in images:
        if not isinstance(image, dict):
            continue
        rel = image.get("relative_path") or ""
        if not rel:
            missing += 1
            continue
        absolute = (dataset_dir / rel).resolve(strict=False)
        if not absolute.exists():
            missing += 1
            continue
        modality = str(image.get("modality") or image.get("series_role") or "unknown")
        phase_index = image.get("phase_index")
        relative_time = image.get("relative_time")
        out_rows.append(
            {
                "patient_id": patient_id,
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "visit_timepoint": timepoint,
                "split": split,
                "modality": modality,
                "phase_index": _stringify(phase_index),
                "relative_time": _stringify(relative_time),
                "path": str(absolute),
                **base_labels,
            }
        )
    return out_rows, missing


def _resolve_inner_root(dataset_dir: Path, dataset_id: str) -> Path:
    """Return the directory that contains per-patient folders.

    Most datasets use ``<dataset_dir>/<dataset_id>``. ``full_v4_fujian_pcr`` keeps
    its inner folder as ``fujian_pCR`` (mixed case), so we fall back to the first
    directory child when the lowercase guess is missing.
    """

    primary = dataset_dir / dataset_id
    if primary.exists():
        return primary
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name.lower() == dataset_id.lower():
            return child
    # Final fallback: the first non-hidden directory child.
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            return child
    return dataset_dir


def _load_labels(study: dict[str, str], study_dir: Path) -> dict[str, Any]:
    """Merge label sources into one dict, preferring labels_optimized.json on disk."""

    payload: dict[str, Any] = {}

    raw = (study.get("labels_json") or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                payload.update(data)
        except json.JSONDecodeError:
            pass

    # 先读原始 labels.json，再让 labels_optimized.json 覆盖同名字段，
    # 确保训练最终使用优化后的标签。
    for name in ("labels.json", "labels_optimized.json"):
        candidate = study_dir / name
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payload.update(data)
            except json.JSONDecodeError:
                continue
    return payload


def _resolve_labels(study: dict[str, str], labels: dict[str, Any]) -> dict[str, str]:
    pcr = _binary_label(study.get("pcr_status"))
    if pcr == "":
        pcr = _binary_label(labels.get("pCR"))
        if pcr == "":
            pcr = _binary_label(labels.get("pcr_status"))

    her2 = _binary_label(study.get("HER2") or labels.get("HER2"))
    er = _binary_label(study.get("ER") or labels.get("ER"))
    pr = _binary_label(study.get("PR") or labels.get("PR"))
    hr = _binary_label(labels.get("HR") or labels.get("hormone_receptor"))
    if hr == "" and er in {"0", "1"} and pr in {"0", "1"}:
        hr = "1" if int(er) or int(pr) else "0"

    subtype_value = (
        labels.get("molecular_subtype")
        or study.get("molecular_subtype")
        or ""
    )
    subtype = _subtype_label(subtype_value)

    survival_time, survival_event = _survival_labels(labels)

    return {
        "pCR": pcr,
        "HER2": her2,
        "ER": er,
        "PR": pr,
        "HR": hr,
        "molecular_subtype": subtype,
        "survival_time": survival_time,
        "survival_event": survival_event,
    }


def _binary_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if value != value:  # NaN guard
            return ""
        return "1" if value >= 0.5 else "0"
    text = str(value).strip().lower()
    if text in MISSING_TOKENS:
        return ""
    if text in POSITIVE_TOKENS:
        return "1"
    if text in NEGATIVE_TOKENS:
        return "0"
    try:
        numeric = float(text)
    except ValueError:
        return ""
    if numeric != numeric or numeric < 0:
        return ""
    return "1" if numeric >= 0.5 else "0"


def _subtype_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace(" ", "_")
    if not text or text in MISSING_TOKENS:
        return ""
    if text in SUBTYPE_TO_ID:
        return str(SUBTYPE_TO_ID[text])
    try:
        numeric = int(float(text))
    except ValueError:
        return ""
    return str(numeric) if 0 <= numeric <= 10 else ""


def _survival_labels(labels: dict[str, Any]) -> tuple[str, str]:
    survival = labels.get("survival") if isinstance(labels.get("survival"), dict) else {}
    if not survival.get("available", True):
        return "", ""
    time_value = (
        survival.get("dfs_time_days")
        or survival.get("os_time_days")
        or survival.get("last_contact_days")
    )
    event_value = survival.get("dfs_event")
    if event_value is None:
        event_value = survival.get("os_event")
    if event_value is None:
        event_value = survival.get("recurrence_event")
    if time_value is None or event_value is None:
        return "", ""
    try:
        numeric_time = float(time_value)
    except (TypeError, ValueError):
        return "", ""
    if numeric_time != numeric_time or numeric_time < 0:
        return "", ""
    if isinstance(event_value, str):
        text = event_value.strip().lower()
        if text in {"yes", "true", "1", "event"}:
            event = 1
        elif text in {"no", "false", "0", "censored", "censor", ""}:
            event = 0
        else:
            return "", ""
    else:
        try:
            event = 1 if int(event_value) else 0
        except (TypeError, ValueError):
            return "", ""
    return f"{numeric_time:g}", str(event)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No volume rows produced — refusing to overwrite manifest with an empty file.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_FIELDS})


def _print_summary(summary: list[dict[str, Any]], out: Path, total_rows: int) -> None:
    print(f"Manifest written: {out}")
    print(f"Total volume rows: {total_rows}")
    for entry in summary:
        print(
            "  {dir:42s} studies={studies_in:5d} missing_studies={missing_studies:4d} "
            "missing_files={missing_volumes:5d} volumes={volumes_out:6d} "
            "kind={kind} splits={splits} modalities={modalities}".format(
                dir=entry["dataset_dir"],
                studies_in=entry["studies_in"],
                missing_studies=entry["studies_missing_volumes"],
                missing_volumes=entry["volumes_missing_files"],
                volumes_out=entry["volumes_out"],
                kind=entry["manifest_kind"],
                splits=entry["splits"],
                modalities=entry["modalities"],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
