from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import suppress_warnings


class DicomReadError(RuntimeError):
    """Raised when a candidate file is not readable as a DICOM header."""


DICOM_KEYWORDS = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "BitsStored",
    "InstanceNumber",
    "TemporalPositionIdentifier",
    "NumberOfTemporalPositions",
    "ContrastBolusAgent",
    "ContrastBolusStartTime",
    "ContrastBolusTotalDose",
    "BodyPartExamined",
    "Laterality",
    "ImageType",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "InversionTime",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "SOPClassUID",
)

EXTRA_TAGS = {
    "DiffusionBValue": (0x0018, 0x9087),
    "DiffusionGradientOrientation": (0x0018, 0x9089),
    "MRDiffusionSequence": (0x0018, 0x9117),
}


def pydicom_available() -> bool:
    try:
        import pydicom  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def read_dicom_header(path: Path) -> dict[str, Any]:
    suppress_warnings()
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise DicomReadError("pydicom is not installed") from exc

    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            defer_size="1 KB",
            specific_tags=None,
        )
    except Exception as exc:  # pydicom raises several format-specific exceptions.
        raise DicomReadError(str(exc)) from exc

    if not any(hasattr(dataset, key) for key in ("SOPClassUID", "SeriesInstanceUID", "StudyInstanceUID")):
        raise DicomReadError("file does not contain required DICOM identity tags")

    header: dict[str, Any] = {}
    for keyword in DICOM_KEYWORDS:
        if hasattr(dataset, keyword):
            header[keyword] = to_jsonable(getattr(dataset, keyword))
    for keyword, tag in EXTRA_TAGS.items():
        if tag in dataset:
            header[keyword] = to_jsonable(dataset[tag].value)
    return header


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__iter__") and value.__class__.__name__ == "MultiValue":
        return [to_jsonable(item) for item in value]
    return str(value)


def hash_identifier(identifier: object, salt: str) -> str:
    text = "" if identifier is None else str(identifier)
    payload = f"{salt}|{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


def sanitize_header(header: dict[str, Any]) -> dict[str, Any]:
    safe = dict(header)
    for key in (
        "PatientID",
        "PatientName",
        "PatientBirthDate",
        "PatientAddress",
        "OtherPatientIDs",
        "OtherPatientNames",
    ):
        safe.pop(key, None)
    return safe
