from __future__ import annotations

from pathlib import Path


ARCHIVE_EXTENSIONS = (
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".gz",
    ".bz2",
    ".xz",
)

EXTENSION_TO_TYPE = {
    ".dcm": "dicom",
    ".dicom": "dicom",
    ".ima": "dicom",
    ".nii": "nifti",
    ".gz": "compressed_or_nifti",
    ".nrrd": "nrrd",
    ".nhdr": "nrrd",
    ".mha": "itk_image",
    ".mhd": "itk_image",
    ".csv": "table",
    ".tsv": "table",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".json": "json",
    ".zip": "archive",
    ".7z": "archive",
    ".rar": "archive",
    ".tar": "archive",
    ".bz2": "archive",
    ".xz": "archive",
    ".txt": "text",
    ".md": "text",
    ".xml": "metadata",
    ".xml.gz": "metadata",
}


def file_extension(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".xml.gz"):
        return ".xml.gz"
    for extension in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz"):
        if lower_name.endswith(extension):
            return extension
    return path.suffix.lower() or "<no_ext>"


def classify_file(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return "nifti"
    if lower_name.endswith(".xml.gz"):
        return "metadata"
    if is_archive_file(path):
        return "archive"
    return EXTENSION_TO_TYPE.get(path.suffix.lower(), "unknown")


def is_archive_file(path: Path, archive_extensions: tuple[str, ...] = ARCHIVE_EXTENSIONS) -> bool:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz") or lower_name.endswith(".xml.gz"):
        return False
    return any(lower_name.endswith(extension.lower()) for extension in archive_extensions)


def is_dicom_candidate(path: Path, candidate_extensions: tuple[str, ...]) -> bool:
    suffix = path.suffix.lower()
    if suffix in candidate_extensions:
        return True
    if suffix == "":
        return True
    return False
