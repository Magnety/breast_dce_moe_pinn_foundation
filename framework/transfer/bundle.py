from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.config import load_config
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_csv, atomic_write_json, atomic_write_text
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy, resolve_path


@dataclass(frozen=True)
class BundleResult:
    bundle_dir: Path
    file_count: int
    total_bytes: int
    checksum_file: Path
    files_csv: Path
    bundle_json: Path
    zip_path: Path | None = None


def create_processed_data_bundle(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    make_zip: bool | None = None,
) -> BundleResult:
    suppress_warnings()
    config = load_config(config_path)
    policy = PathPolicy.from_config(config)
    project_root = policy.project_root
    transfer_cfg = config.get("transfer", {})
    preprocessing_cfg = config.get("preprocessing", {})

    resolved_processed_root = resolve_path(
        processed_root or preprocessing_cfg.get("processed_root") or config.get("paths", {}).get("processed_root"),
        base=project_root,
    )
    resolved_manifest_path = (
        resolve_path(manifest_path, base=project_root) if manifest_path else _default_manifest_path(project_root)
    )
    bundle_root = resolve_path(
        output_dir or transfer_cfg.get("bundle_root", "outputs/upload_packages"),
        base=project_root,
    )
    name = bundle_name or datetime.now().strftime("processed_%Y%m%d_%H%M%S")
    bundle_dir = policy.ensure_output_dir(bundle_root / name)

    include_patterns = tuple(transfer_cfg.get("include_patterns", ["*"]))
    exclude_dir_names = set(transfer_cfg.get("exclude_dir_names", ["cache", "tmp", "__pycache__"]))
    rows = collect_file_rows(resolved_processed_root, include_patterns, exclude_dir_names)
    total_bytes = int(sum(row["size_bytes"] for row in rows))

    manifest_copy = None
    if resolved_manifest_path.exists():
        manifest_copy = bundle_dir / resolved_manifest_path.name
        policy.assert_write_allowed(manifest_copy)
        shutil.copy2(resolved_manifest_path, manifest_copy)

    checksum_file = bundle_dir / "checksums.sha256"
    checksums_text = "\n".join(f"{row['sha256']}  {row['relative_path']}" for row in rows)
    if checksums_text:
        checksums_text += "\n"
    atomic_write_text(checksum_file, checksums_text, policy=policy)

    files_csv = bundle_dir / "files.csv"
    atomic_write_csv(
        files_csv,
        rows,
        ["relative_path", "size_bytes", "sha256", "source_path"],
        policy=policy,
    )

    bundle_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(resolve_path(config_path)),
        "processed_root": str(resolved_processed_root),
        "manifest_path": str(resolved_manifest_path),
        "manifest_copied_to": str(manifest_copy) if manifest_copy else "",
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "server_processed_root": transfer_cfg.get("server_processed_root", "needs_configuration"),
        "server_manifest_root": transfer_cfg.get("server_manifest_root", "needs_configuration"),
        "status": "ok" if rows else "no_processed_files_found",
    }
    bundle_json = bundle_dir / "bundle_manifest.json"
    atomic_write_json(bundle_json, bundle_payload, policy=policy)
    atomic_write_text(bundle_dir / "README_UPLOAD.md", render_upload_readme(bundle_payload), policy=policy)

    zip_path = None
    should_zip = bool(transfer_cfg.get("make_zip", False) if make_zip is None else make_zip)
    if should_zip:
        zip_path = bundle_root / f"{name}.zip"
        policy.assert_write_allowed(zip_path)
        archive_items = [item for item in bundle_dir.rglob("*") if item.is_file()]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for item in progress_iter(
                archive_items,
                total=len(archive_items),
                desc="Zip bundle",
                unit="file",
            ):
                archive.write(item, item.relative_to(bundle_dir))

    return BundleResult(
        bundle_dir=bundle_dir,
        file_count=len(rows),
        total_bytes=total_bytes,
        checksum_file=checksum_file,
        files_csv=files_csv,
        bundle_json=bundle_json,
        zip_path=zip_path,
    )


def collect_file_rows(
    processed_root: Path,
    include_patterns: tuple[str, ...],
    exclude_dir_names: set[str],
) -> list[dict[str, Any]]:
    suppress_warnings()
    if not processed_root.exists():
        return []
    files: dict[Path, None] = {}
    for pattern in progress_iter(
        include_patterns,
        total=len(include_patterns),
        desc="Find bundle files",
        unit="pattern",
    ):
        for path in progress_iter(
            processed_root.rglob(pattern),
            desc=f"Scan {pattern}",
            unit="file",
        ):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(processed_root).parts
            if any(part in exclude_dir_names for part in relative_parts):
                continue
            files[path] = None
    rows = []
    sorted_files = sorted(files, key=lambda item: str(item.relative_to(processed_root)).lower())
    for path in progress_iter(sorted_files, total=len(sorted_files), desc="Hash bundle files", unit="file"):
        rows.append(
            {
                "relative_path": str(path.relative_to(processed_root)).replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "source_path": str(path),
            }
        )
    return rows


def verify_bundle(bundle_dir: str | Path, processed_root: str | Path) -> dict[str, Any]:
    suppress_warnings()
    bundle = Path(bundle_dir).expanduser().resolve(strict=False)
    root = Path(processed_root).expanduser().resolve(strict=False)
    files_csv = bundle / "files.csv"
    if not files_csv.exists():
        raise FileNotFoundError(f"Bundle file list does not exist: {files_csv}")
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    with files_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        for row in progress_iter(rows, total=len(rows), desc="Verify bundle", unit="file"):
            target = root / row["relative_path"]
            if not target.exists():
                missing.append(row["relative_path"])
                continue
            checked += 1
            if sha256_file(target) != row["sha256"]:
                mismatched.append(row["relative_path"])
    return {
        "bundle_dir": str(bundle),
        "processed_root": str(root),
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "ok": not missing and not mismatched,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def render_upload_readme(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Processed Data Upload Package",
            "",
            f"- Created at: `{payload['created_at']}`",
            f"- Local processed root: `{payload['processed_root']}`",
            f"- File count: `{payload['file_count']}`",
            f"- Total bytes: `{payload['total_bytes']}`",
            f"- Server processed root: `{payload['server_processed_root']}`",
            f"- Server manifest root: `{payload['server_manifest_root']}`",
            f"- Status: `{payload['status']}`",
            "",
            "Upload the processed-data directory contents to the configured server processed root.",
            "Then run the server verification IDE configuration before starting training.",
            "",
        ]
    )


def _default_manifest_path(project_root: Path) -> Path:
    manifest_root = project_root / "outputs" / "manifests"
    candidates = sorted(manifest_root.glob("*/master_manifest.csv"), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]
    return manifest_root / "master_manifest.csv"
