from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.file_types import ARCHIVE_EXTENSIONS
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.config import load_config
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import resolve_path


@dataclass(frozen=True)
class AuditRunConfig:
    project_root: Path
    raw_roots: tuple[Path, ...]
    output_root: Path
    workers: int = 1
    max_files_per_dataset: int | None = None
    max_dicom_headers_per_dataset: int | None = 5000
    max_label_file_size_mb: int = 64
    tree_max_depth: int = 4
    tree_max_entries_per_dir: int = 30
    follow_symlinks: bool = False
    dicom_candidate_extensions: tuple[str, ...] = (".dcm", ".dicom", ".ima")
    label_extensions: tuple[str, ...] = (".csv", ".tsv", ".xlsx", ".xls", ".json")
    archive_extensions: tuple[str, ...] = ARCHIVE_EXTENSIONS
    skip_archive_files: bool = True
    include_top_level_archives_as_datasets: bool = False
    skip_archives_with_extracted_dirs: bool = True
    anonymization_salt: str = "change-me"
    save_patient_mapping: bool = False

    @classmethod
    def from_file(
        cls,
        config_path: str | Path,
        raw_roots: list[str] | None = None,
        output_root: str | None = None,
        max_files: int | None = None,
        max_dicom_headers: int | None = None,
        workers: int | None = None,
    ) -> tuple["AuditRunConfig", dict[str, Any]]:
        mapping = load_config(config_path)
        return (
            cls.from_mapping(
                mapping,
                raw_roots=raw_roots,
                output_root=output_root,
                max_files=max_files,
                max_dicom_headers=max_dicom_headers,
                workers=workers,
            ),
            mapping,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        raw_roots: list[str] | None = None,
        output_root: str | None = None,
        max_files: int | None = None,
        max_dicom_headers: int | None = None,
        workers: int | None = None,
    ) -> "AuditRunConfig":
        config_path = mapping.get("_config_path")
        config_base = Path(config_path).parent if config_path else None
        project_root = resolve_path(
            mapping.get("project", {}).get("project_root", "."),
            base=config_base,
        )
        paths = mapping.get("paths", {})
        audit = mapping.get("audit", {})
        safety = mapping.get("safety", {})
        anonymization = mapping.get("anonymization", {})

        configured_raw_roots = raw_roots if raw_roots else paths.get("raw_roots", [])
        configured_output_root = output_root if output_root else paths.get("output_root", "outputs")

        return cls(
            project_root=project_root,
            raw_roots=tuple(resolve_path(path) for path in configured_raw_roots),
            output_root=resolve_path(configured_output_root, base=project_root),
            workers=int(workers if workers is not None else audit.get("workers", 1)),
            max_files_per_dataset=_none_or_int(
                max_files if max_files is not None else audit.get("max_files_per_dataset")
            ),
            max_dicom_headers_per_dataset=_none_or_int(
                max_dicom_headers
                if max_dicom_headers is not None
                else audit.get("max_dicom_headers_per_dataset", 5000)
            ),
            max_label_file_size_mb=int(audit.get("max_label_file_size_mb", 64)),
            tree_max_depth=int(audit.get("tree_max_depth", 4)),
            tree_max_entries_per_dir=int(audit.get("tree_max_entries_per_dir", 30)),
            follow_symlinks=bool(safety.get("follow_symlinks", False)),
            dicom_candidate_extensions=tuple(
                item.lower() for item in audit.get("dicom_candidate_extensions", [".dcm", ".dicom", ".ima"])
            ),
            label_extensions=tuple(
                item.lower()
                for item in audit.get("label_extensions", [".csv", ".tsv", ".xlsx", ".xls", ".json"])
            ),
            archive_extensions=tuple(
                item.lower() for item in audit.get("archive_extensions", list(ARCHIVE_EXTENSIONS))
            ),
            skip_archive_files=bool(audit.get("skip_archive_files", True)),
            include_top_level_archives_as_datasets=bool(
                audit.get("include_top_level_archives_as_datasets", False)
            ),
            skip_archives_with_extracted_dirs=bool(
                audit.get("skip_archives_with_extracted_dirs", True)
            ),
            anonymization_salt=str(anonymization.get("salt", "change-me")),
            save_patient_mapping=bool(anonymization.get("save_patient_mapping", False)),
        )


def _none_or_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
