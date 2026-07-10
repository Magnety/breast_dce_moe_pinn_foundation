from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABEL_COPY_NAME = "label_copy.json"
LABEL_COPY_SCHEMA_VERSION = "label_copy_v1"

DEFAULT_IMPORTANT_LABEL_KEYS = (
    "schema_version",
    "dataset_id",
    "subject_id",
    "patient_id",
    "patient_uid",
    "case_id",
    "timepoint",
    "study_uid",
    "split",
    "pCR",
    "pcr_status",
    "response",
    "HER2",
    "ER",
    "PR",
    "molecular_subtype",
    "laterality",
    "age",
    "BMI",
    "sex",
    "race",
    "ethnicity",
    "clinical_stage",
    "pathologic_stage",
    "tumor_size_cm",
    "histologic_type",
    "grade",
    "pathologic_t",
    "pathologic_n",
    "pathologic_m",
    "treatment",
    "pathology",
)

COPY_METADATA_KEYS = {
    "schema_version",
    "source_label_schema_version",
    "source_labels_json",
    "dataset_id",
    "subject_id",
    "patient_id",
    "patient_uid",
    "timepoint",
    "study_uid",
    "secondary_key_count",
    "secondary_keys",
    "secondary_labels",
}


def optimize_label_payload(
    labels: dict[str, Any],
    *,
    existing_copy: dict[str, Any] | None = None,
    extra_important_keys: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a labels payload into compact primary labels plus a secondary copy."""

    important_keys = _important_keys(extra_important_keys)
    primary = {key: labels[key] for key in important_keys if key in labels}
    secondary = {key: value for key, value in labels.items() if key not in primary}
    if existing_copy:
        merged_secondary = {**_secondary_from_existing_copy(existing_copy), **secondary}
        secondary = {key: value for key, value in merged_secondary.items() if key not in primary}

    copy_payload = _label_copy_payload(labels, secondary)
    return primary, copy_payload


def optimize_label_file(
    labels_path: str | Path,
    *,
    dry_run: bool = True,
    backup: bool = False,
    extra_important_keys: Iterable[str] = (),
) -> dict[str, Any]:
    path = Path(labels_path).expanduser().resolve(strict=False)
    copy_path = path.with_name(LABEL_COPY_NAME)
    result = {
        "labels_path": str(path),
        "label_copy_path": str(copy_path),
        "dataset_id": "",
        "subject_id": "",
        "timepoint": "",
        "important_key_count": 0,
        "secondary_key_count": 0,
        "changed_labels": False,
        "changed_label_copy": False,
        "written": False,
        "error": "",
    }

    try:
        labels = _read_json_object(path)
        existing_copy = _read_json_object(copy_path) if copy_path.exists() else {}
        optimized, copy_payload = optimize_label_payload(
            labels,
            existing_copy=existing_copy,
            extra_important_keys=extra_important_keys,
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    result.update(
        {
            "dataset_id": str(optimized.get("dataset_id") or labels.get("dataset_id") or ""),
            "subject_id": str(
                optimized.get("subject_id")
                or optimized.get("patient_id")
                or optimized.get("patient_uid")
                or labels.get("subject_id")
                or labels.get("patient_id")
                or labels.get("patient_uid")
                or ""
            ),
            "timepoint": str(optimized.get("timepoint") or labels.get("timepoint") or ""),
            "important_key_count": len(optimized),
            "secondary_key_count": len(copy_payload.get("secondary_labels", {})),
        }
    )

    labels_text = _json_text(optimized)
    copy_text = _json_text(copy_payload)
    result["changed_labels"] = _text_changed(path, labels_text)
    result["changed_label_copy"] = _text_changed(copy_path, copy_text)

    if not dry_run and (result["changed_labels"] or result["changed_label_copy"]):
        if result["changed_labels"]:
            _backup_if_requested(path, backup)
            _write_json_text(path, labels_text)
        if result["changed_label_copy"]:
            _backup_if_requested(copy_path, backup)
            _write_json_text(copy_path, copy_text)
        result["written"] = True
    return result


def optimize_label_tree(
    roots: Iterable[str | Path],
    *,
    dry_run: bool = True,
    backup: bool = False,
    dataset_ids: Iterable[str] = (),
    extra_important_keys: Iterable[str] = (),
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_filter = {str(item) for item in dataset_ids if str(item).strip()}
    roots_list = [Path(root).expanduser().resolve(strict=False) for root in roots]
    results: list[dict[str, Any]] = []

    for path in _iter_label_paths(roots_list):
        if limit is not None and len(results) >= limit:
            break
        if dataset_filter:
            payload = _read_json_object(path)
            if str(payload.get("dataset_id", "")) not in dataset_filter:
                continue
        results.append(
            optimize_label_file(
                path,
                dry_run=dry_run,
                backup=backup,
                extra_important_keys=extra_important_keys,
            )
        )

    errors = [row for row in results if row.get("error")]
    datasets = Counter(row.get("dataset_id", "") for row in results if row.get("dataset_id"))
    summary = {
        "dry_run": dry_run,
        "roots": [str(root) for root in roots_list],
        "label_files": len(results),
        "changed_labels": sum(bool(row.get("changed_labels")) for row in results),
        "changed_label_copy": sum(bool(row.get("changed_label_copy")) for row in results),
        "written_files": sum(bool(row.get("written")) for row in results),
        "secondary_keys_total": sum(int(row.get("secondary_key_count", 0)) for row in results),
        "datasets": dict(sorted(datasets.items())),
        "errors": errors,
    }
    return summary


def add_label_optimization_parser(
    preprocess_subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    parser = preprocess_subparsers.add_parser(
        "optimize-labels",
        help="Move secondary labels/features from labels.json into sibling label_copy.json files.",
    )
    parser.add_argument("--root", action="append", default=None, help="Root to scan. Can be repeated.")
    parser.add_argument("--dataset-id", action="append", default=[], help="Only process this dataset id. Can be repeated.")
    parser.add_argument("--important-key", action="append", default=[], help="Keep an extra key in labels.json.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true", help="Write changes. Omit for dry-run.")
    parser.add_argument("--backup", action="store_true", help="Create .label_optimize.bak files before overwriting.")
    parser.set_defaults(func=run_label_optimization_from_args)
    return parser


def run_label_optimization_from_args(args: argparse.Namespace) -> int:
    roots = args.root or _default_output_roots()
    summary = optimize_label_tree(
        roots,
        dry_run=not args.apply,
        backup=args.backup,
        dataset_ids=args.dataset_id,
        extra_important_keys=args.important_key,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["errors"] else 1


def _default_output_roots() -> list[Path]:
    dataset_root = os.environ.get("BREAST_MRI_DATASET_ROOT")
    if dataset_root:
        return [Path(dataset_root).expanduser()]
    return [
        Path("/home/ubuntu/liuyiyao1/multimodal_dataset"),
    ]


def _important_keys(extra_important_keys: Iterable[str]) -> tuple[str, ...]:
    keys: list[str] = []
    for key in [*DEFAULT_IMPORTANT_LABEL_KEYS, *extra_important_keys]:
        text = str(key).strip()
        if text and text not in keys:
            keys.append(text)
    return tuple(keys)


def _iter_label_paths(roots: list[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "labels.json":
            candidates = [root]
        elif root.exists():
            candidates = root.rglob("labels.json")
        else:
            candidates = []
        for path in candidates:
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _label_copy_payload(labels: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    subject_id = labels.get("subject_id") or labels.get("patient_id") or labels.get("patient_uid") or labels.get("case_id") or ""
    payload: dict[str, Any] = {
        "schema_version": LABEL_COPY_SCHEMA_VERSION,
        "source_label_schema_version": labels.get("schema_version", ""),
        "source_labels_json": "labels.json",
        "dataset_id": labels.get("dataset_id", ""),
        "subject_id": subject_id,
        "patient_id": labels.get("patient_id", subject_id),
        "patient_uid": labels.get("patient_uid", ""),
        "timepoint": labels.get("timepoint", ""),
        "study_uid": labels.get("study_uid", ""),
        "secondary_key_count": len(secondary),
        "secondary_keys": list(secondary.keys()),
        "secondary_labels": secondary,
    }
    return {key: value for key, value in payload.items() if value not in ("", None) or key in {"secondary_labels", "secondary_keys", "secondary_key_count"}}


def _secondary_from_existing_copy(payload: dict[str, Any]) -> dict[str, Any]:
    secondary = payload.get("secondary_labels")
    if isinstance(secondary, dict):
        return dict(secondary)
    return {key: value for key, value in payload.items() if key not in COPY_METADATA_KEYS}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _text_changed(path: Path, desired: str) -> bool:
    return not path.exists() or path.read_text(encoding="utf-8") != desired


def _backup_if_requested(path: Path, backup: bool) -> None:
    if not backup or not path.exists():
        return
    backup_path = path.with_name(path.name + ".label_optimize.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def _write_json_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise
