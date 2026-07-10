from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.audit.dicom_metadata import DicomReadError, read_dicom_header
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.classification import classify_dce_phase
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.fujian_sequences import fujian_phase_sort_number
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists, windows_extended_path


@dataclass(frozen=True)
class SeriesTemporalSummary:
    source_path: str
    series_uid: str
    files_sampled: int
    acquisition_time_min: str
    acquisition_time_max: str
    contrast_bolus_start_time: str
    temporal_position_count: int
    number_of_temporal_positions: str
    unique_slice_positions_sampled: int
    repeated_slice_positions_sampled: bool
    contrast_phase: str
    phase_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "series_uid": self.series_uid,
            "files_sampled": self.files_sampled,
            "acquisition_time_min": self.acquisition_time_min,
            "acquisition_time_max": self.acquisition_time_max,
            "contrast_bolus_start_time": self.contrast_bolus_start_time,
            "temporal_position_count": self.temporal_position_count,
            "number_of_temporal_positions": self.number_of_temporal_positions,
            "unique_slice_positions_sampled": self.unique_slice_positions_sampled,
            "repeated_slice_positions_sampled": self.repeated_slice_positions_sampled,
            "contrast_phase": self.contrast_phase,
            "phase_evidence": self.phase_evidence,
        }


def analyze_series_temporal_metadata(
    source_path: str | Path,
    series_uid: str = "",
    series_description: str = "",
    max_files: int = 64,
) -> SeriesTemporalSummary:
    """Read sampled DICOM headers and infer DCE temporal/pre-post metadata.

    The function is deliberately lightweight: it reads headers only and never
    opens pixel data. If injection-time metadata is absent, it falls back to text
    and series order evidence, clearly recording that reduced confidence.
    """

    source = Path(source_path).expanduser().resolve(strict=False)
    files = _candidate_files(source)
    if series_uid:
        files = _filter_by_series_uid(files, series_uid, search_limit=max(max_files * 8, max_files))
    sampled = _sample_paths(files, max_files=max_files)
    headers: list[dict[str, Any]] = []
    for file_path in sampled:
        try:
            headers.append(read_dicom_header(file_path))
        except DicomReadError:
            continue

    acquisition_times = sorted({_clean_time(header.get("AcquisitionTime")) for header in headers if _clean_time(header.get("AcquisitionTime"))})
    bolus_times = sorted(
        {_clean_time(header.get("ContrastBolusStartTime")) for header in headers if _clean_time(header.get("ContrastBolusStartTime"))}
    )
    temporal_positions = {_as_int(header.get("TemporalPositionIdentifier")) for header in headers}
    temporal_positions.discard(None)
    temporal_count_values = [str(header.get("NumberOfTemporalPositions") or "") for header in headers]
    temporal_count_values = [value for value in temporal_count_values if value]
    slice_positions = {_slice_key(header.get("ImagePositionPatient")) for header in headers}
    slice_positions.discard("")

    text_phase, text_evidence = classify_dce_phase(series_description, str(source))
    phase, evidence = _phase_from_times(acquisition_times, bolus_times)
    if phase == "unknown" and temporal_positions:
        if temporal_positions == {1} and not bolus_times:
            phase = "unknown"
            evidence = "temporal position starts at 1 but injection time is missing"
        else:
            phase = "post" if min(temporal_positions) > 1 else "unknown"
            evidence = "inferred from TemporalPositionIdentifier without injection time"
    if phase == "unknown" and text_phase != "unknown":
        phase = text_phase
        evidence = text_evidence
    if phase == "unknown":
        evidence = evidence or "no injection time, acquisition time, or explicit text evidence"

    return SeriesTemporalSummary(
        source_path=str(source),
        series_uid=series_uid,
        files_sampled=len(headers),
        acquisition_time_min=acquisition_times[0] if acquisition_times else "",
        acquisition_time_max=acquisition_times[-1] if acquisition_times else "",
        contrast_bolus_start_time=bolus_times[0] if bolus_times else "",
        temporal_position_count=len(temporal_positions),
        number_of_temporal_positions=_most_common(temporal_count_values),
        unique_slice_positions_sampled=len(slice_positions),
        repeated_slice_positions_sampled=bool(headers and len(slice_positions) < len(headers)),
        contrast_phase=phase,
        phase_evidence=evidence,
    )


def refine_group_dce_phases(records: list[dict[str, str]], temporal_summaries: dict[str, dict[str, Any]]) -> None:
    """Mutate DCE source records with phase order and pre/post metadata.

    Records can represent either separate DCE phases or a single multi-temporal
    DICOM series. This function only annotates metadata; conversion still keeps
    all available source phases.
    """

    dce_records = [row for row in records if row.get("series_role") in {"DCE", "DCE_PRE", "DCE_POST"}]
    if not dce_records:
        return
    ordered = sorted(dce_records, key=_dce_order_key)
    for index, record in enumerate(ordered):
        key = record.get("series_uid") or record.get("source_path", "")
        summary = temporal_summaries.get(key, {})
        phase = summary.get("contrast_phase") or "unknown"
        evidence = summary.get("phase_evidence") or ""
        if phase == "unknown":
            if record.get("series_role") == "DCE_PRE" or index == 0:
                phase = "pre"
                evidence = evidence or "first DCE source in sorted group; verify manually if injection time is missing"
            elif record.get("series_role") == "DCE_POST" or index > 0:
                phase = "post"
                evidence = evidence or "later DCE source in sorted group; verify manually if injection time is missing"
        record["dce_phase_order"] = str(index)
        record["dce_contrast_phase"] = phase
        record["dce_phase_evidence"] = evidence
        record["dce_acquisition_time"] = str(summary.get("acquisition_time_min", ""))
        record["contrast_bolus_start_time"] = str(summary.get("contrast_bolus_start_time", ""))
        # Keep the source series role unchanged here. Multi-temporal DCE series can
        # contain both pre- and post-contrast components in one DICOM series; the
        # component-level writer assigns DCE_PRE/DCE_POST per output image.


def _candidate_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not path_exists(source):
        return []
    try:
        root = Path(windows_extended_path(source))
        return sorted([path for path in root.rglob("*") if path.is_file()], key=lambda path: str(path).lower())
    except OSError:
        return []


def _filter_by_series_uid(files: list[Path], series_uid: str, search_limit: int) -> list[Path]:
    matched: list[Path] = []
    for file_path in _sample_paths(files, max_files=max(search_limit, 1)):
        try:
            header = read_dicom_header(file_path)
        except DicomReadError:
            continue
        if str(header.get("SeriesInstanceUID") or "") == str(series_uid):
            matched.append(file_path)
    return matched or files


def _sample_paths(paths: list[Path], max_files: int) -> list[Path]:
    if len(paths) <= max_files:
        return paths
    indices = {0, len(paths) - 1}
    step = max(1, len(paths) // max_files)
    indices.update(range(0, len(paths), step))
    return [paths[index] for index in sorted(index for index in indices if 0 <= index < len(paths))[:max_files]]


def _phase_from_times(acquisition_times: list[str], bolus_times: list[str]) -> tuple[str, str]:
    if not acquisition_times or not bolus_times:
        return "unknown", "injection or acquisition time missing"
    acq = _time_to_seconds(acquisition_times[0])
    bolus = _time_to_seconds(bolus_times[0])
    if acq is None or bolus is None:
        return "unknown", "injection or acquisition time is not parseable"
    if acq < bolus:
        return "pre", "AcquisitionTime is before ContrastBolusStartTime"
    return "post", "AcquisitionTime is at or after ContrastBolusStartTime"


def _dce_order_key(record: dict[str, str]) -> tuple[int, float, int, str, str]:
    if record.get("dataset_id", "") == "ispy1":
        acquisition_time = record.get("dce_acquisition_time", "") or record.get("acquisition_time_min", "")
        if not acquisition_time:
            acquisition_time = _ispy1_note_value(record.get("notes", ""), "acquisition_time")
        parsed_time = _time_to_seconds(acquisition_time)
        return (
            {"DCE_PRE": 0, "DCE": 1, "DCE_POST": 2}.get(record.get("series_role", ""), 9),
            parsed_time if parsed_time is not None else 1e12,
            fujian_phase_sort_number(" ".join([record.get("series_description", ""), record.get("source_path", "")])),
            record.get("series_description", ""),
            record.get("series_uid", ""),
        )
    phase_rank = {"DCE_PRE": 0, "DCE": 1, "DCE_POST": 2}.get(record.get("series_role", ""), 9)
    phase_number = fujian_phase_sort_number(
        " ".join([record.get("series_description", ""), record.get("source_path", "")])
    )
    return (
        phase_rank,
        _path_number(record.get("source_path", "")),
        phase_number,
        record.get("series_description", ""),
        record.get("series_uid", ""),
    )


def _path_number(value: str) -> float:
    name = Path(value).name
    match = __import__("re").match(r"(\d+(?:\.\d+)?)", name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 1e12
    return 1e12


def _ispy1_note_value(notes: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(notes or "").split(";"):
        text = part.strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""


def _clean_time(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    return text.replace(":", "")


def _time_to_seconds(value: str) -> float | None:
    text = _clean_time(value)
    if not text:
        return None
    try:
        if "." in text:
            whole, frac = text.split(".", 1)
        else:
            whole, frac = text, "0"
        whole = whole.rjust(6, "0")[:6]
        hours = int(whole[0:2])
        minutes = int(whole[2:4])
        seconds = int(whole[4:6])
        return hours * 3600 + minutes * 60 + seconds + float(f"0.{frac}")
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slice_key(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return ",".join(f"{float(item):.3f}" for item in value[:3])
    if value in (None, ""):
        return ""
    return str(value)


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
