from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.io import atomic_write_json
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.paths import path_exists, path_is_dir, windows_extended_path
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, silence_simpleitk_warnings, suppress_warnings


SKIP_ROLES = {"mask_seg", "mask", "localizer", "derived", "secondary_capture", "unknown"}


@dataclass(frozen=True)
class DicomTemporalGrouping:
    method: str
    evidence: str
    groups: list[tuple[str, list[str]]]
    total_files: int
    unique_slice_positions: int
    repeated_slice_positions: int
    reference_slice_count: int

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def group_sizes(self) -> list[int]:
        return [len(names) for _, names in self.groups]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "evidence": self.evidence,
            "group_count": self.group_count,
            "group_sizes": self.group_sizes,
            "total_files": self.total_files,
            "unique_slice_positions": self.unique_slice_positions,
            "repeated_slice_positions": self.repeated_slice_positions,
            "reference_slice_count": self.reference_slice_count,
        }


def convert_tasks_to_nifti(
    tasks_csv: str | Path,
    output_root: str | Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    suppress_warnings()
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SimpleITK is required for DICOM-to-NIfTI conversion.") from exc
    silence_simpleitk_warnings(sitk)

    tasks_path = Path(tasks_csv).expanduser().resolve(strict=False)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    attempted = 0
    converted = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    with tasks_path.open("r", encoding="utf-8", newline="") as handle:
        task_rows = list(csv.DictReader(handle))
        for row in progress_iter(task_rows, total=len(task_rows), desc="Convert NIfTI", unit="task"):
            if row.get("task_status") != "ready_for_conversion":
                skipped += 1
                continue
            source = Path(row["primary_dce_path"])
            if not path_exists(source):
                failures.append({"source": str(source), "error": "missing_source"})
                continue
            attempted += 1
            target_dir = output_root / row["dataset_id"] / row["subject_id"] / row["timepoint"]
            target_dir.mkdir(parents=True, exist_ok=True)
            target_image = target_dir / "DCE_primary.nii.gz"
            if target_image.exists() and not overwrite:
                skipped += 1
                if limit is not None and attempted >= limit:
                    break
                continue
            try:
                image = read_dicom_series(sitk, source, row.get("primary_dce_series_uid", ""))
                sitk.WriteImage(image, str(target_image))
                atomic_write_json(target_dir / "metadata.json", row)
                converted += 1
            except Exception as exc:
                failures.append({"source": str(source), "error": str(exc)})
            if limit is not None and attempted >= limit:
                break
    summary = {"attempted": attempted, "converted": converted, "skipped": skipped, "failures": failures}
    atomic_write_json(output_root / "conversion_summary.json", summary)
    return summary


def read_dicom_series(sitk, series_dir: Path, series_uid: str = ""):
    silence_simpleitk_warnings(sitk)
    reader = sitk.ImageSeriesReader()
    dicom_names = _gdcm_series_files(reader, series_dir, series_uid)
    if not dicom_names:
        dicom_names = _enumerated_series_files(series_dir, series_uid)
    if not dicom_names:
        raise RuntimeError(f"No DICOM files found by GDCM in {series_dir}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def read_dicom_series_by_temporal_position(
    sitk,
    series_dir: Path,
    series_uid: str = "",
    reference_slice_count: int | None = None,
) -> list[tuple[str, Any]]:
    """Read one DICOM series as separate temporal volumes when possible."""

    silence_simpleitk_warnings(sitk)
    grouped = dicom_temporal_groups(series_dir, series_uid, reference_slice_count=reference_slice_count)
    if len(grouped) <= 1:
        return [("", read_dicom_series(sitk, series_dir, series_uid))]

    output: list[tuple[str, Any]] = []
    reader = sitk.ImageSeriesReader()
    for temporal_position, names in grouped:
        reader.SetFileNames(tuple(names))
        output.append((temporal_position, reader.Execute()))
    return output


def dicom_temporal_groups(
    series_dir: Path,
    series_uid: str = "",
    reference_slice_count: int | None = None,
) -> list[tuple[str, list[str]]]:
    """Return DICOM file groups that represent temporal DCE volumes.

    Some public breast MRI releases store all DCE phases in one SeriesInstanceUID
    without TemporalPositionIdentifier. In those cases the only reliable signal is
    that each slice position appears once per phase.
    """

    return dicom_temporal_grouping(series_dir, series_uid, reference_slice_count=reference_slice_count).groups


def dicom_temporal_grouping(
    series_dir: Path,
    series_uid: str = "",
    reference_slice_count: int | None = None,
) -> DicomTemporalGrouping:
    """Return temporal DCE grouping plus the evidence used to choose it."""

    dicom_names = _series_file_names_for_uid(series_dir, series_uid)
    return _group_files_by_temporal_position(dicom_names, reference_slice_count=reference_slice_count)


def read_dicom_directory(sitk, series_dir: Path):
    silence_simpleitk_warnings(sitk)
    reader = sitk.ImageSeriesReader()
    dicom_names = _dicom_files(series_dir)
    if not dicom_names:
        raise RuntimeError(f"No DICOM files found in {series_dir}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def _gdcm_series_files(reader: Any, series_dir: Path, series_uid: str) -> tuple[str, ...]:
    directory = windows_extended_path(series_dir)
    if series_uid:
        names = tuple(reader.GetGDCMSeriesFileNames(directory, series_uid))
        if names:
            return names
    return tuple(reader.GetGDCMSeriesFileNames(directory))


def _series_file_names_for_uid(series_dir: Path, series_uid: str) -> tuple[str, ...]:
    enumerated = _enumerated_series_files(series_dir, series_uid)
    if enumerated:
        return enumerated
    sitk = __import__("SimpleITK")
    silence_simpleitk_warnings(sitk)
    reader = sitk.ImageSeriesReader()
    dicom_names = _gdcm_series_files(reader, series_dir, series_uid)
    if not dicom_names:
        dicom_names = enumerated
    return tuple(dicom_names)


def _group_files_by_temporal_position(
    dicom_names: tuple[str, ...],
    reference_slice_count: int | None = None,
) -> DicomTemporalGrouping:
    if not dicom_names:
        return DicomTemporalGrouping(
            method="no_files",
            evidence="no DICOM files were found",
            groups=[],
            total_files=0,
            unique_slice_positions=0,
            repeated_slice_positions=0,
            reference_slice_count=reference_slice_count or 0,
        )
    try:
        import pydicom
    except ModuleNotFoundError:
        return _grouping_result(
            "single_group_no_pydicom",
            "pydicom is not installed; temporal grouping could not inspect headers",
            [("", list(dicom_names))],
            [],
            reference_slice_count,
        )

    records: list[dict[str, Any]] = []
    for name in dicom_names:
        try:
            header = pydicom.dcmread(
                name,
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "TemporalPositionIdentifier",
                    "NumberOfTemporalPositions",
                    "AcquisitionNumber",
                    "AcquisitionTime",
                    "AcquisitionDateTime",
                    "ContentTime",
                    "TriggerTime",
                    "InstanceNumber",
                    "ImagePositionPatient",
                    "SliceLocation",
                    "InStackPositionNumber",
                ],
            )
        except Exception:
            records.append(
                {
                    "name": name,
                    "temporal": "",
                    "acquisition_number": "",
                    "acquisition": "",
                    "acquisition_datetime": "",
                    "content_time": "",
                    "trigger_time": "",
                    "instance": 10**12,
                    "position": "",
                }
            )
            continue
        temporal = str(getattr(header, "TemporalPositionIdentifier", "") or "").strip()
        instance = _as_int(getattr(header, "InstanceNumber", None))
        records.append(
            {
                "name": name,
                "temporal": temporal,
                "acquisition_number": str(getattr(header, "AcquisitionNumber", "") or "").strip(),
                "acquisition": str(getattr(header, "AcquisitionTime", "") or "").strip(),
                "acquisition_datetime": str(getattr(header, "AcquisitionDateTime", "") or "").strip(),
                "content_time": str(getattr(header, "ContentTime", "") or "").strip(),
                "trigger_time": str(getattr(header, "TriggerTime", "") or "").strip(),
                "instance": instance if instance is not None else 10**12,
                "position": _slice_position_key(header),
            }
        )
    temporal_groups = _group_by_complete_metadata_key(records, "temporal")
    if temporal_groups:
        return _grouping_result(
            "temporal_position",
            "TemporalPositionIdentifier forms complete 3D volumes",
            temporal_groups,
            records,
            reference_slice_count,
        )

    for key_name in (
        "acquisition_number",
        "acquisition_datetime",
        "acquisition",
        "content_time",
        "trigger_time",
    ):
        metadata_groups = _group_by_complete_metadata_key(records, key_name)
        if metadata_groups:
            return _grouping_result(
                key_name,
                f"{key_name} forms complete 3D volumes",
                metadata_groups,
                records,
                reference_slice_count,
            )

    repeated_position_groups = _group_by_repeated_slice_positions(records)
    if repeated_position_groups:
        return _grouping_result(
            "repeated_slice_positions",
            "each slice position repeats the same number of times",
            repeated_position_groups,
            records,
            reference_slice_count,
        )

    near_repeated_position_groups = _group_by_near_complete_repeated_slice_positions(records)
    if near_repeated_position_groups:
        return _grouping_result(
            "near_complete_repeated_slice_positions",
            "slice positions repeat nearly the same number of times; incomplete temporal volumes are preserved",
            near_repeated_position_groups,
            records,
            reference_slice_count,
        )

    slice_block_groups = _group_by_reference_slice_blocks(records, reference_slice_count)
    if slice_block_groups:
        return _grouping_result(
            "reference_slice_blocks",
            "file count is divisible by a trusted sibling-series slice count",
            slice_block_groups,
            records,
            reference_slice_count,
        )

    ordered = sorted(records, key=_record_sort_key)
    return _grouping_result(
        "single_volume",
        "no reliable temporal grouping metadata or repeated slice positions were found",
        [("", [record["name"] for record in ordered])],
        records,
        reference_slice_count,
    )


def _group_records(records: list[dict[str, Any]], key_name: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get(key_name, "") or ""), []).append(record)
    return grouped


def _grouping_result(
    method: str,
    evidence: str,
    groups: list[tuple[str, list[str]]],
    records: list[dict[str, Any]],
    reference_slice_count: int | None,
) -> DicomTemporalGrouping:
    position_values = [record["position"] for record in records if record.get("position")]
    position_counts = Counter(position_values)
    repeated_positions = sum(1 for count in position_counts.values() if count > 1)
    return DicomTemporalGrouping(
        method=method,
        evidence=evidence,
        groups=groups,
        total_files=len(records) if records else sum(len(names) for _, names in groups),
        unique_slice_positions=len(position_counts),
        repeated_slice_positions=repeated_positions,
        reference_slice_count=reference_slice_count or 0,
    )


def _sorted_grouped_records(grouped: dict[str, list[dict[str, Any]]]) -> list[tuple[str, list[str]]]:
    output: list[tuple[str, list[str]]] = []
    for key, values in sorted(grouped.items(), key=lambda item: _temporal_sort_key(item[0])):
        ordered = sorted(values, key=_record_sort_key)
        output.append((key, [record["name"] for record in ordered]))
    return output


def _group_by_repeated_slice_positions(records: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    position_values = [record["position"] for record in records if record["position"]]
    if len(position_values) != len(records):
        return []
    position_counts = Counter(position_values)
    if len(position_counts) < 8:
        return []
    repeat_counts = set(position_counts.values())
    if len(repeat_counts) != 1:
        return []
    repeat_count = next(iter(repeat_counts))
    if repeat_count <= 1:
        return []

    ordered = sorted(records, key=_temporal_occurrence_sort_key)
    seen_by_position: dict[str, int] = defaultdict(int)
    grouped: dict[str, list[dict[str, Any]]] = {str(index + 1): [] for index in range(repeat_count)}
    for record in ordered:
        position = record["position"]
        occurrence = seen_by_position[position]
        if occurrence >= repeat_count:
            return []
        seen_by_position[position] += 1
        grouped[str(occurrence + 1)].append(record)

    expected_slices = len(position_counts)
    if any(len(values) != expected_slices for values in grouped.values()):
        return []
    return _sorted_grouped_records(grouped)


def _group_by_near_complete_repeated_slice_positions(records: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    position_values = [record["position"] for record in records if record["position"]]
    if len(position_values) != len(records):
        return []
    position_counts = Counter(position_values)
    if len(position_counts) < 8:
        return []
    repeat_count, complete_position_count = sorted(
        Counter(position_counts.values()).items(),
        key=lambda item: (-item[0], -item[1]),
    )[0]
    if repeat_count <= 1:
        return []
    if any(count > repeat_count for count in position_counts.values()):
        return []
    if complete_position_count / len(position_counts) < 0.8:
        return []

    total_expected = repeat_count * len(position_counts)
    missing_slices = total_expected - len(records)
    max_missing_slices = max(2, int(total_expected * 0.05))
    if missing_slices <= 0 or missing_slices > max_missing_slices:
        return []

    return _group_by_instance_slice_blocks(
        records,
        expected_slices=len(position_counts),
        phase_count=repeat_count,
    )


def _group_by_instance_slice_blocks(
    records: list[dict[str, Any]],
    expected_slices: int,
    phase_count: int,
) -> list[tuple[str, list[str]]]:
    instances = [_as_int(record.get("instance")) for record in records]
    if any(instance is None or instance >= 10**12 for instance in instances):
        return []
    numeric_instances = [int(instance) for instance in instances if instance is not None]
    if len(set(numeric_instances)) != len(numeric_instances):
        return []

    expected_total = expected_slices * phase_count
    min_instance = min(numeric_instances)
    instance_span = max(numeric_instances) - min_instance + 1
    if instance_span > expected_total or expected_total - instance_span > max(2, int(expected_total * 0.05)):
        return []

    grouped: dict[str, list[dict[str, Any]]] = {str(index + 1): [] for index in range(phase_count)}
    for record in records:
        instance = _as_int(record.get("instance"))
        if instance is None:
            return []
        block_index = (instance - min_instance) // expected_slices
        if block_index < 0 or block_index >= phase_count:
            return []
        grouped[str(block_index + 1)].append(record)

    min_group_slices = max(8, int(expected_slices * 0.8))
    for values in grouped.values():
        if not (min_group_slices <= len(values) <= expected_slices):
            return []
        positions = [record["position"] for record in values if record.get("position")]
        if len(positions) != len(values) or len(set(positions)) != len(positions):
            return []
    return _sorted_grouped_records(grouped)


def _group_by_complete_acquisition_time(records: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    return _group_by_complete_metadata_key(records, "acquisition")


def _group_by_complete_metadata_key(records: list[dict[str, Any]], key_name: str) -> list[tuple[str, list[str]]]:
    key_values = {str(record.get(key_name, "") or "") for record in records if str(record.get(key_name, "") or "")}
    if len(key_values) <= 1:
        return []
    grouped = _group_records(records, key_name=key_name)
    grouped = {key: values for key, values in grouped.items() if key}
    if len(grouped) <= 1:
        return []

    position_sets: list[set[str]] = []
    group_lengths = {len(values) for values in grouped.values()}
    if all(not record.get("position") for record in records):
        if len(group_lengths) == 1 and next(iter(group_lengths)) >= 8:
            return _sorted_grouped_records(grouped)
        return []
    for values in grouped.values():
        positions = {record["position"] for record in values if record["position"]}
        if len(positions) < 8 or len(positions) != len(values):
            return []
        position_sets.append(positions)
    first_positions = position_sets[0]
    if any(positions != first_positions for positions in position_sets[1:]):
        return []
    return _sorted_grouped_records(grouped)


def _group_by_reference_slice_blocks(
    records: list[dict[str, Any]],
    reference_slice_count: int | None,
) -> list[tuple[str, list[str]]]:
    if reference_slice_count is None or reference_slice_count < 8:
        return []
    if len(records) <= reference_slice_count or len(records) % reference_slice_count != 0:
        return []
    phase_count = len(records) // reference_slice_count
    if phase_count <= 1 or phase_count > 20:
        return []

    ordered = sorted(records, key=_record_sort_key)
    grouped: dict[str, list[dict[str, Any]]] = {}
    position_sets: list[set[str]] = []
    for index in range(phase_count):
        block = ordered[index * reference_slice_count : (index + 1) * reference_slice_count]
        if len(block) != reference_slice_count:
            return []
        positions = {record["position"] for record in block if record.get("position")}
        if positions and len(positions) != reference_slice_count:
            return []
        if positions:
            position_sets.append(positions)
        grouped[str(index + 1)] = block

    if position_sets:
        first_positions = position_sets[0]
        if any(positions != first_positions for positions in position_sets[1:]):
            return []
    return _sorted_grouped_records(grouped)


def _enumerated_series_files(series_dir: Path, series_uid: str) -> tuple[str, ...]:
    if not path_is_dir(series_dir):
        return ()
    try:
        import pydicom
    except ModuleNotFoundError:
        return ()

    files = [path for path in Path(windows_extended_path(series_dir)).iterdir() if path.is_file()]
    if not files:
        return ()
    matched: list[Path] = []
    for file_path in files:
        if series_uid == "":
            matched.append(file_path)
            continue
        try:
            header = pydicom.dcmread(
                str(file_path),
                stop_before_pixels=True,
                force=True,
                specific_tags=["SeriesInstanceUID", "InstanceNumber"],
            )
        except Exception:
            continue
        if str(getattr(header, "SeriesInstanceUID", "")) == series_uid:
            matched.append(file_path)
    if not matched and series_uid == "":
        matched = files
    return tuple(windows_extended_path(path) for path in sorted(matched, key=lambda path: path.name))


def _dicom_files(series_dir: Path) -> tuple[str, ...]:
    if not path_is_dir(series_dir):
        return ()
    files = [
        path
        for path in Path(windows_extended_path(series_dir)).iterdir()
        if path.is_file() and path.name.lower().endswith((".dcm", ".dicom", ".ima"))
    ]
    return tuple(windows_extended_path(path) for path in sorted(files, key=lambda path: path.name))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _temporal_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, value)


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _as_int(record.get("instance")) if _as_int(record.get("instance")) is not None else 10**12,
        _position_sort_key(str(record.get("position", "") or "")),
        str(record.get("name", "") or ""),
    )


def _temporal_occurrence_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _as_int(record.get("instance")) if _as_int(record.get("instance")) is not None else 10**12,
        _temporal_sort_key(str(record.get("acquisition_number", "") or "")),
        _temporal_sort_key(str(record.get("acquisition_datetime", "") or "")),
        _temporal_sort_key(str(record.get("acquisition", "") or "")),
        _temporal_sort_key(str(record.get("content_time", "") or "")),
        _temporal_sort_key(str(record.get("trigger_time", "") or "")),
        str(record.get("name", "") or ""),
    )


def _slice_position_key(header: Any) -> str:
    position = _position_key(getattr(header, "ImagePositionPatient", ""))
    if position:
        return position
    slice_location = getattr(header, "SliceLocation", "")
    if slice_location not in (None, ""):
        try:
            return f"slice:{float(slice_location):.3f}"
        except (TypeError, ValueError):
            return f"slice:{slice_location}"
    in_stack = _as_int(getattr(header, "InStackPositionNumber", None))
    if in_stack is not None:
        return f"stack:{in_stack}"
    return ""


def _position_key(value: Any) -> str:
    if value not in (None, "") and not isinstance(value, (str, bytes)):
        try:
            items = list(value)
        except TypeError:
            return str(value)
        if len(items) >= 3:
            try:
                return ",".join(f"{float(item):.3f}" for item in items[:3])
            except (TypeError, ValueError):
                return str(value)
    return "" if value in (None, "") else str(value)


def _position_sort_key(value: str) -> tuple[Any, ...]:
    if value.startswith("slice:"):
        try:
            return (1, float(value.split(":", 1)[1]))
        except ValueError:
            return (8, value)
    if value.startswith("stack:"):
        try:
            return (2, int(value.split(":", 1)[1]))
        except ValueError:
            return (8, value)
    try:
        return (0, *[float(item) for item in value.split(",")[:3]])
    except ValueError:
        return (8, value)
