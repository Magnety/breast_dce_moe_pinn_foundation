from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeriesRecord:
    dataset_id: str
    subject_id: str
    study_uid: str
    study_date: str
    series_uid: str
    series_description: str
    modality: str
    series_role: str
    timepoint: str
    image_count: int
    source_path: str
    relative_path: str
    sop_class_name: str = ""
    manufacturer: str = ""
    local_exists: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaskRecord:
    dataset_id: str
    subject_id: str
    study_uid: str
    study_date: str
    series_uid: str
    mask_type: str
    source_path: str
    relative_path: str
    image_count: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelRecord:
    dataset_id: str
    subject_id: str
    source_file: str
    key_column: str
    labels: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        return row


@dataclass(frozen=True)
class CaseRecord:
    dataset_id: str
    subject_id: str
    study_count: int
    series_count: int
    mask_count: int
    available_roles: tuple[str, ...]
    has_label: bool
    source_root: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterResult:
    dataset_id: str
    source_root: Path
    cases: list[CaseRecord] = field(default_factory=list)
    series: list[SeriesRecord] = field(default_factory=list)
    masks: list[MaskRecord] = field(default_factory=list)
    labels: list[LabelRecord] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


class DatasetAdapter:
    dataset_id: str

    def scan(self, max_patients: int | None = None) -> AdapterResult:
        raise NotImplementedError

