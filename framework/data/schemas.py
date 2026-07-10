from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImageSeriesRecord:
    dataset_id: str
    patient_uid: str
    study_uid: str
    series_uid: str
    modality: str
    modality_status: str
    relative_paths: tuple[str, ...] = ()
    spacing: tuple[float, ...] | None = None
    shape: tuple[int, ...] | None = None
    dce_phase_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestRecord:
    dataset_id: str
    patient_uid: str
    study_uid: str = "unknown"
    original_patient_mapping: str = "not_exported"
    original_path: str = "not_exported"
    standardized_data_path: str = "not_available"
    timepoint: str = "unknown"
    laterality: str = "unknown"
    lesion_id: str = "unknown"
    available_modalities: tuple[str, ...] = ()
    dce_phase_count: int = 0
    image_shape: str = "unknown"
    spacing: str = "unknown"
    clinical_labels: dict[str, Any] = field(default_factory=dict)
    pathology_labels: dict[str, Any] = field(default_factory=dict)
    molecular_subtype: str = "unknown"
    response_label: str = "unknown"
    pcr_status: str = "unknown"
    survival_followup: dict[str, Any] = field(default_factory=dict)
    roi_status: str = "unknown"
    missing_modalities: tuple[str, ...] = ()
    qc_status: str = "needs_review"
    exclusion_reason: str = ""
    preprocessing_version: str = "not_preprocessed"
    split: str = "unassigned"
    source_audit_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    category: str
    message: str
    patient_uid: str = ""
    study_uid: str = ""
    series_uid: str = ""
    relative_path: str = ""
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

