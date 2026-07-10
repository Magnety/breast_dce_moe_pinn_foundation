from __future__ import annotations

from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.file_types import ARCHIVE_EXTENSIONS, is_archive_file


def should_skip_preprocessing_path(
    path: str | Path,
    skip_archive_files: bool = True,
    archive_extensions: tuple[str, ...] = ARCHIVE_EXTENSIONS,
) -> bool:
    """Return True for source files that must never enter preprocessing.

    Archive packages are assumed to be already extracted in the dataset folders.
    NIfTI files such as `.nii.gz` are not treated as archives.
    """

    return bool(skip_archive_files and is_archive_file(Path(path), archive_extensions))


def sort_dicom_paths_by_instance(headers: list[dict[str, Any]], paths: list[str | Path]) -> list[Path]:
    paired = []
    for header, path in zip(headers, paths):
        instance = _as_int(header.get("InstanceNumber"))
        acquisition_time = str(header.get("AcquisitionTime") or header.get("SeriesTime") or "")
        paired.append((instance if instance is not None else 10**12, acquisition_time, Path(path)))
    return [path for _, _, path in sorted(paired)]


def spacing_is_consistent(headers: list[dict[str, Any]]) -> bool:
    values = {
        (
            str(header.get("PixelSpacing", "")),
            str(header.get("SliceThickness", "")),
            str(header.get("SpacingBetweenSlices", "")),
        )
        for header in headers
    }
    return len(values) <= 1


def orientation_is_consistent(headers: list[dict[str, Any]]) -> bool:
    values = {str(header.get("ImageOrientationPatient", "")) for header in headers}
    values.discard("")
    return len(values) <= 1


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
