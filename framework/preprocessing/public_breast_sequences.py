from __future__ import annotations

import re
from collections import Counter
from typing import Any


PUBLIC_BREAST_DATASET_IDS = {"duke", "ispy1", "ispy2", "breast_mri_nact_pilot", "advanced_mri_breast_lesions"}
PUBLIC_BREAST_DCE_ROLES = {"DCE_PRE", "DCE", "DCE_POST"}
PUBLIC_BREAST_DCE_MAP_ROLES = {"DCE_PE", "DCE_SER"}


def classify_public_breast_series_role(
    description: str,
    modality: str = "",
    sop_class_name: str = "",
    source_path: str = "",
    dataset_id: str = "",
) -> str:
    """Classify public breast MRI datasets by observed release names.

    These public datasets have DCE phases organized in several different ways:
    a single original series with sibling SER/PE maps, multiple per-phase series,
    or a multi-phase folder whose name alone is the strongest clue.  The rules
    below mirror the explicit string matching style used in the reference
    project while keeping this project's common output roles unchanged.
    """

    dataset_key = (dataset_id or "").lower()
    if dataset_key not in PUBLIC_BREAST_DATASET_IDS:
        return "unknown"

    series_text = _series_text(description, source_path)
    text = _normal_text(series_text, modality, sop_class_name)
    if dataset_key == "ispy1":
        return _classify_ispy1_role(text)
    if dataset_key == "ispy2":
        return _classify_ispy2_role(text)
    if dataset_key == "duke":
        return _classify_duke_role(text)
    if dataset_key == "breast_mri_nact_pilot":
        return _classify_nact_pilot_role(text, _normal_text(series_text))
    if dataset_key == "advanced_mri_breast_lesions":
        return _classify_advanced_mri_breast_lesions_role(text)
    return "unknown"


def effective_public_breast_series_role(row: dict[str, str]) -> str:
    """Return the dataset-specific role, falling back to the existing plan role."""

    role = row.get("series_role", "")
    dataset_id = row.get("dataset_id", "")
    if dataset_id not in PUBLIC_BREAST_DATASET_IDS:
        return role
    public_role = classify_public_breast_series_role(
        row.get("series_description", ""),
        modality=row.get("modality", ""),
        sop_class_name=row.get("sop_class_name", ""),
        source_path=row.get("source_path", "") or row.get("relative_path", ""),
        dataset_id=dataset_id,
    )
    return public_role if public_role != "unknown" else role


def filter_public_breast_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply dataset-level DCE source selection before the generic selector runs."""

    if not rows:
        return rows
    dataset_ids = {row.get("dataset_id", "") for row in rows}
    public_ids = dataset_ids & PUBLIC_BREAST_DATASET_IDS
    if not public_ids:
        return rows
    dataset_id = sorted(public_ids)[0]
    if dataset_id == "ispy1":
        return _filter_ispy1_dce_sources(rows)
    if dataset_id == "ispy2":
        return _filter_ispy2_dce_sources(rows)
    if dataset_id == "duke":
        return _filter_duke_dce_sources(rows)
    if dataset_id == "breast_mri_nact_pilot":
        return _filter_nact_pilot_dce_sources(rows)
    if dataset_id == "advanced_mri_breast_lesions":
        return _filter_advanced_mri_breast_lesions_dce_sources(rows)
    return rows


def is_public_breast_dataset_row(row: dict[str, str]) -> bool:
    return row.get("dataset_id", "") in PUBLIC_BREAST_DATASET_IDS


def _classify_ispy2_role(text: str) -> str:
    if _looks_like_mask(text):
        return "mask_seg" if "analysis mask" in text or "segmentation" in text else "mask"
    if _has_pe_map_token(text):
        return "DCE_PE"
    if _has_ser_map_token(text):
        return "DCE_SER"
    if "original dce" in text or ("volser" in text and "original" in text):
        return "DCE"
    if re.search(r"\bph\s*[1-9]", text):
        return "DCE_POST"
    if _looks_like_ispy2_dce_source(text):
        return _phase_role_from_text(text)
    return "unknown"


def _classify_ispy1_role(text: str) -> str:
    if _looks_like_mask(text):
        return "mask_seg" if "segmentation" in text or "voi" in text else "mask"
    if _has_pe_map_token(text):
        return "DCE_PE"
    if _has_ser_map_token(text):
        return "DCE_SER"
    if _looks_like_ispy1_dce_source(text):
        return _phase_role_from_text(text)
    return "unknown"


def _classify_duke_role(text: str) -> str:
    if "ideal" in text:
        return "unknown"
    if _looks_like_mask(text):
        return "mask_seg" if "segmentation" in text else "mask"
    if _looks_like_derived_map(text):
        return "derived"
    if _has_duke_pre_token(text):
        return "DCE_PRE"
    if _has_duke_post_token(text):
        return "DCE_POST"
    if _looks_like_duke_dce_source(text):
        return _phase_role_from_text(text)
    return "unknown"


def _classify_nact_pilot_role(text: str, series_text: str) -> str:
    compact = _compact(text)
    series_compact = _compact(series_text)
    if _is_nact_known_non_dce(series_text, series_compact):
        return "unknown"
    if _looks_like_mask(text):
        return "mask_seg" if "segmentation" in text else "mask"
    if _has_pe_map_token(text):
        return "DCE_PE"
    if _has_ser_map_token(text):
        return "DCE_SER"
    if _looks_like_nact_dce_source(text, compact) or _looks_like_nact_dce_source(series_text, series_compact):
        return _phase_role_from_text(text)
    return "unknown"


def _classify_advanced_mri_breast_lesions_role(text: str) -> str:
    if _looks_like_mask(text):
        return "mask_seg" if "segmentation" in text or re.search(r"\broi\b", text) is not None else "mask"
    if _looks_like_derived_map(text):
        return "derived"
    if _looks_like_advanced_registered_multiphase(text):
        return "DCE"
    return "unknown"


def _filter_ispy2_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dce_sources = [row for row in rows if row.get("series_role") in PUBLIC_BREAST_DCE_ROLES]
    if len(dce_sources) <= 1:
        return rows

    volser_original = [
        row
        for row in dce_sources
        if "volser" in _row_text(row) and ("original dce" in _row_text(row) or "original" in _row_text(row))
    ]
    if not volser_original:
        return rows

    selected = sorted(volser_original, key=_ispy2_original_key)[0]
    return _preserve_original_order(rows, [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected])


def _filter_ispy1_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dce_sources = [row for row in rows if row.get("series_role") in PUBLIC_BREAST_DCE_ROLES]
    if len(dce_sources) <= 1:
        return rows

    split_series_phase_sources = [
        row
        for row in dce_sources
        if _looks_like_ispy1_split_phase_source(row)
        and _to_int(row.get("dce_temporal_group_count", "")) <= 1
    ]
    if len(split_series_phase_sources) > 1:
        return rows

    multiphase = [row for row in dce_sources if _to_int(row.get("dce_temporal_group_count", "")) > 1]
    if multiphase:
        selected = sorted(multiphase, key=_ispy1_original_key)[0]
        keep = [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected]
        return _preserve_original_order(rows, keep)

    originals = [row for row in dce_sources if _looks_like_ispy1_dce_source(_row_text(row))]
    if not originals:
        return rows
    selected = sorted(originals, key=_ispy1_original_key)[0]
    keep = [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected]
    return _preserve_original_order(rows, keep)


def _filter_duke_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # Duke frequently stores DCE as explicit cross-series phases such as
    # ax dyn pre + ax dyn 1st/2nd/3rd pass or Ph1/Ph2/Ph3 branches.  Keeping
    # all source phases is safer than collapsing to one apparently multi-phase row.
    return rows


def _filter_nact_pilot_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dce_sources = [row for row in rows if row.get("series_role") in PUBLIC_BREAST_DCE_ROLES]
    if len(dce_sources) <= 1:
        return rows

    map_counts_by_base = _map_slice_counts_by_base(rows)
    map_backed = [row for row in dce_sources if _best_reference_count(row, map_counts_by_base) > 0]
    if map_backed:
        selected = sorted(map_backed, key=lambda row: _nact_original_key(row, map_counts_by_base))[0]
        keep = [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected]
        return _preserve_original_order(rows, keep)

    multiphase = [row for row in dce_sources if _to_int(row.get("dce_temporal_group_count", "")) > 1]
    if multiphase:
        selected = sorted(multiphase, key=_complete_multiphase_key)[0]
        keep = [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected]
        return _preserve_original_order(rows, keep)
    return rows


def _filter_advanced_mri_breast_lesions_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dce_sources = [row for row in rows if row.get("series_role") in PUBLIC_BREAST_DCE_ROLES]
    if len(dce_sources) <= 1:
        return rows

    registered_multiphase = [row for row in dce_sources if _looks_like_advanced_registered_multiphase(_row_text(row))]
    if not registered_multiphase:
        return rows

    selected = sorted(registered_multiphase, key=_advanced_registered_multiphase_key)[0]
    keep = [row for row in rows if row.get("series_role") not in PUBLIC_BREAST_DCE_ROLES] + [selected]
    return _preserve_original_order(rows, keep)


def _looks_like_ispy2_dce_source(text: str) -> bool:
    return any(
        token in text
        for token in (
            "dce",
            "dynaviews",
            "dyna views",
            "ax dyn",
            "dynamic",
            "vibe",
            "volser",
        )
    )


def _looks_like_ispy1_dce_source(text: str) -> bool:
    return any(
        token in text
        for token in (
            "dynamic 3dfgre",
            "dynamic3dfgre",
            "3dfgre",
            "dyn 3dfgre",
            "fl3d",
            "dce",
            "dynamic",
            "volser",
        )
    )


def _looks_like_ispy1_split_phase_source(row: dict[str, str]) -> bool:
    text = _row_text(row)
    notes = _normal_text(row.get("notes", ""))
    phase_count = _to_int(row.get("dce_phase_count_in_study", "")) or _to_int(
        _note_value(row.get("notes", ""), "dce_phase_count_in_study")
    )
    return (
        "split series phase" in notes
        or ("fl3d" in text and _to_int(row.get("image_count", "")) >= 8)
        or (
            _looks_like_ispy1_dce_source(text)
            and _to_int(row.get("image_count", "")) >= 8
            and _to_int(row.get("dce_temporal_group_count", "")) <= 1
            and phase_count > 1
        )
    )


def _looks_like_duke_dce_source(text: str) -> bool:
    return any(
        token in text
        for token in (
            "ax dyn",
            "ax 3d dyn",
            "ax dynamic",
            "dynamic",
            "multiphase",
            "multi phase",
            "vibrant",
            "fl3d",
            "3d dyn",
        )
    )


def _looks_like_nact_dce_source(text: str, compact: str) -> bool:
    exact_compact_names = {
        "ad",
        "lbreast",
        "dynamic3dfgre",
        "sagittalir3dfgre",
        "sagittal3dfgre",
        "sagittal3dwithfatsupp",
        "rbreastsag3dgradientecho",
        "breastpasag3dgradientecho",
        "breastpasag3dspgrsatnp",
    }
    if compact in exact_compact_names:
        return True
    return any(
        token in compact
        for token in (
            "dynamic3dfgre",
            "ir3dfgre",
            "sagittal3dfgre",
            "sagittal3dwithfatsupp",
            "breastpasag3dspgr",
            "breastpasag3dgradient",
            "breastsag3dgradient",
        )
    ) or any(token in text for token in ("dynamic 3dfgre", "sagittal 3dfgre", "sagittal ir3dfgre"))


def _is_nact_known_non_dce(text: str, compact: str) -> bool:
    if compact.isdigit():
        return True
    return compact in {"pjn"} or re.fullmatch(r"pjn\d*", compact or "") is not None


def _looks_like_advanced_registered_multiphase(text: str) -> bool:
    return "registered" in text and ("multiphase" in text or "multi phase" in text)


def _has_duke_pre_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bax\s+dyn\s+pre\b",
            r"\bax\s+3d\s+pre\b",
            r"\bax\s+3d\s+dyn\s+pre\b",
            r"\bdyn\s+pre\b",
            r"\bt1\s*fl3d\b.*\bpre\b",
            r"\bpre[\s_-]*contrast\b",
        )
    )


def _has_duke_post_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bph\s*[1-9]",
            r"\b[1-9](?:st|nd|rd|th)\s*pass\b",
            r"\bpass\s*[1-9]\b",
            r"\bpost\b",
            r"\bpost[\s_-]*gad\b",
            r"\bpost[\s_-]*contrast\b",
        )
    )


def _phase_role_from_text(text: str) -> str:
    has_pre = _has_pre_token(text)
    has_post = _has_post_token(text)
    if has_pre and has_post:
        return "DCE"
    if has_pre:
        return "DCE_PRE"
    if has_post:
        return "DCE_POST"
    return "DCE"


def _has_pre_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpre\b",
            r"\bpregad\b",
            r"\bpre[\s_-]*gad\b",
            r"\bpre[\s_-]*contrast\b",
            r"\bnon[\s_-]*contrast\b",
        )
    )


def _has_post_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpost\b",
            r"\bposts\b",
            r"\bpostgad\b",
            r"\bpost[\s_-]*gad\b",
            r"\bpost[\s_-]*contrast\b",
            r"\bgad\b",
            r"\bpass\b",
            r"\b[1-9](?:st|nd|rd|th)\b",
            r"\bph\s*[1-9]",
        )
    )


def _looks_like_mask(text: str) -> bool:
    return "segmentation" in text or re.search(r"\b(mask|roi|voi|contour|annotation)\b", text) is not None


def _looks_like_derived_map(text: str) -> bool:
    return any(
        token in text
        for token in (
            "sub",
            "subtract",
            "subtraction",
            "mip",
            "cad",
            "tram",
            "reformat",
            "projection",
        )
    )


def _has_ser_map_token(text: str) -> bool:
    return re.search(r"(?<![a-z0-9])ser(?![a-z0-9])", text) is not None


def _has_pe_map_token(text: str) -> bool:
    return re.search(r"(?<![a-z0-9])pe\s*[1-9](?![0-9])", text) is not None


def _map_slice_counts_by_base(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    counts_by_base: dict[str, list[int]] = {}
    for row in rows:
        if row.get("series_role") not in PUBLIC_BREAST_DCE_MAP_ROLES:
            continue
        image_count = _to_int(row.get("image_count", ""))
        if image_count < 8:
            continue
        counts_by_base.setdefault(_base_without_map_suffix(row), []).append(image_count)
    return counts_by_base


def _best_reference_count(row: dict[str, str], map_counts_by_base: dict[str, list[int]]) -> int:
    image_count = _to_int(row.get("image_count", ""))
    base = _base_without_map_suffix(row)
    candidates = [
        count
        for count in map_counts_by_base.get(base, [])
        if count < image_count and image_count % count == 0 and 1 < image_count // count <= 20
    ]
    if candidates:
        return sorted(candidates, key=lambda count: (-Counter(map_counts_by_base[base])[count], -count))[0]
    return 0


def _ispy2_original_key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
    text = _row_text(row)
    preferred = 0 if "cropped" in text and ("uni lateral" in text or "unilateral" in text) else 1
    return (
        -_to_int(row.get("dce_temporal_group_count", "")),
        -_to_int(row.get("image_count", "")),
        preferred,
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _ispy1_original_key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
    text = _row_text(row)
    plain_dynamic = 0 if "dynamic 3dfgre" in text and not (_has_ser_map_token(text) or _has_pe_map_token(text)) else 1
    return (
        -_to_int(row.get("dce_temporal_group_count", "")),
        -_to_int(row.get("image_count", "")),
        plain_dynamic,
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _nact_original_key(row: dict[str, str], map_counts_by_base: dict[str, list[int]]) -> tuple[int, int, int, str, str]:
    reference_count = _best_reference_count(row, map_counts_by_base)
    image_count = _to_int(row.get("image_count", ""))
    inferred_phases = image_count // reference_count if reference_count else 1
    return (
        0 if reference_count else 1,
        -max(_to_int(row.get("dce_temporal_group_count", "")), inferred_phases),
        -image_count,
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _complete_multiphase_key(row: dict[str, str]) -> tuple[int, int, str, str]:
    return (
        -_to_int(row.get("dce_temporal_group_count", "")),
        -_to_int(row.get("image_count", "")),
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _advanced_registered_multiphase_key(row: dict[str, str]) -> tuple[int, int, float, str, str]:
    return (
        -_to_int(row.get("dce_temporal_group_count", "")),
        -_to_int(row.get("image_count", "")),
        _path_number(row.get("source_path", "")),
        row.get("series_description", ""),
        row.get("series_uid", ""),
    )


def _base_without_map_suffix(row: dict[str, str]) -> str:
    text = _row_text(row)
    text = re.sub(r"\b(?:ser|pe\s*[1-9])\b", " ", text)
    text = re.sub(r"\banalysis\s+mask\b|\bmask\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _row_text(row: dict[str, str]) -> str:
    return _normal_text(_series_text(row.get("series_description", ""), row.get("source_path", "") or row.get("relative_path", "")))


def _series_text(description: str, source_path: str = "") -> str:
    description = str(description or "").strip()
    if description and description.lower() not in {"nan", "none", "null"}:
        return _strip_series_number_prefix(_normal_text(description))
    leaf = _leaf_name(source_path)
    return _strip_series_number_prefix(_normal_text(leaf))


def _leaf_name(path: str) -> str:
    text = str(path or "").strip().rstrip("\\/")
    if not text:
        return ""
    parts = [part for part in re.split(r"[\\/]+", text) if part]
    return parts[-1] if parts else text


def _strip_series_number_prefix(text: str) -> str:
    return re.sub(r"^\d+(?:\s+\d+)?\s+", "", text).strip()


def _normal_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[\[\]{}(),;:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _to_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _note_value(notes: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(notes or "").split(";"):
        text = part.strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""


def _path_number(value: str) -> float:
    leaf = _leaf_name(value)
    match = re.match(r"(\d+(?:\.\d+)?)", leaf)
    if not match:
        return 1e9
    try:
        return float(match.group(1))
    except ValueError:
        return 1e9


def _preserve_original_order(rows: list[dict[str, str]], keep_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep_ids = {id(row) for row in keep_rows}
    return [row for row in rows if id(row) in keep_ids]
