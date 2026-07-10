from __future__ import annotations

import re
from collections import Counter
from typing import Any


ACRIN_DATASET_ID = "acrin_contralateral"
ACRIN_DCE_ROLES = {"DCE_PRE", "DCE", "DCE_POST"}


def classify_acrin_series_role(
    description: str,
    modality: str = "",
    sop_class_name: str = "",
    source_path: str = "",
) -> str:
    """Classify messy ACRIN 6667 series names using observed public-release patterns."""

    text = _normal_text(description, modality, sop_class_name, source_path)
    sdyn_text = _sdyn_boundary_text(description, modality, sop_class_name, source_path)
    modality_upper = (modality or "").upper()
    if "segmentation" in text or modality_upper == "SEG":
        return "mask_seg"
    if re.search(r"\b(mask|roi|voi|contour|annotation)\b", text):
        return "mask"
    if _looks_like_acrin_derived(text, sdyn_text):
        return "derived"
    if any(token in text for token in ("localizer", "locator", "scout", "calibration")):
        return "localizer"
    if re.search(r"\bloc\b", text):
        return "localizer"
    if "secondary capture" in text or modality_upper in {"SC", "OT"}:
        return "secondary_capture"
    if "adc" in text:
        return "ADC"
    if any(token in text for token in ("diffusion", "dwi", "dwssfse", " b=", "b800")):
        return "DWI"
    if "t2" in text or "stir" in text:
        return "T2"
    if not _looks_like_acrin_dce_source(text):
        return "unknown"

    has_pre = _has_acrin_pre_token(text)
    has_post = _has_acrin_post_token(text)
    if has_pre and has_post:
        return "DCE"
    if has_pre:
        return "DCE_PRE"
    if has_post:
        return "DCE_POST"
    return "DCE"


def effective_acrin_series_role(row: dict[str, str]) -> str:
    role = row.get("series_role", "")
    if role in ACRIN_DCE_ROLES and is_acrin_derived_series(row):
        return "derived"
    if row.get("dataset_id", "") != ACRIN_DATASET_ID:
        return role
    acrin_role = classify_acrin_series_role(
        row.get("series_description", ""),
        modality=row.get("modality", ""),
        sop_class_name=row.get("sop_class_name", ""),
        source_path=row.get("source_path", "") or row.get("relative_path", ""),
    )
    return acrin_role if acrin_role != "unknown" else role


def is_acrin_derived_series(row: dict[str, str]) -> bool:
    text = _normal_text(row.get("series_description", ""), row.get("source_path", ""), row.get("relative_path", ""))
    sdyn_text = _sdyn_boundary_text(
        row.get("series_description", ""),
        row.get("source_path", ""),
        row.get("relative_path", ""),
    )
    return _looks_like_acrin_derived(text, sdyn_text)


def filter_acrin_dce_sources(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep robust ACRIN DCE candidates without changing the common output format.

    Rules are intentionally conservative:
    - SDYN/subtraction/CAD/MIP/TRAM-like outputs are left as derived maps.
    - If one or more source folders already split into multiple temporal volumes,
      keep the most complete one, usually the largest Optional Sag/3D SPGR folder.
    - If DCE is stored as several single-phase folders, drop odd slice-count
      outliers only when a majority slice count exists.
    """

    if not rows or not any(row.get("dataset_id") == ACRIN_DATASET_ID for row in rows):
        return rows
    non_dce = [row for row in rows if row.get("series_role") not in ACRIN_DCE_ROLES]
    dce = [row for row in rows if row.get("series_role") in ACRIN_DCE_ROLES]
    if len(dce) <= 1:
        return rows

    valid_counts = [_to_int(row.get("image_count", "")) for row in dce if _to_int(row.get("image_count", "")) >= 8]
    if len(valid_counts) >= 2:
        count_frequency = Counter(valid_counts)
        keep_count, keep_frequency = count_frequency.most_common(1)[0]
        if keep_frequency >= 2 and not _has_acrin_combined_multiphase_candidate(dce):
            filtered_dce = [
                row
                for row in dce
                if _to_int(row.get("image_count", "")) == keep_count or _to_int(row.get("image_count", "")) < 8
            ]
            if _looks_like_acrin_split_phase_set(filtered_dce):
                evidence = (
                    "ACRIN split DCE folders selected by majority slice count "
                    f"({keep_count} slices in {keep_frequency} folders)"
                )
                for row in filtered_dce:
                    if _to_int(row.get("image_count", "")) == keep_count:
                        _mark_as_acrin_split_phase_folder(row, evidence)
                return _preserve_original_order(rows, non_dce + filtered_dce)

    multiphase = [_row for _row in dce if _to_int(_row.get("dce_temporal_group_count", "")) > 1]
    if multiphase:
        selected = sorted(multiphase, key=_acrin_multiphase_key)[0]
        return _preserve_original_order(rows, non_dce + [selected])

    if len(valid_counts) < 3:
        return rows
    count_frequency = Counter(valid_counts)
    keep_count, keep_frequency = count_frequency.most_common(1)[0]
    if keep_frequency < 2 or len(count_frequency) <= 1:
        return rows
    filtered_dce = [
        row
        for row in dce
        if _to_int(row.get("image_count", "")) == keep_count or _to_int(row.get("image_count", "")) < 8
    ]
    return _preserve_original_order(rows, non_dce + filtered_dce)


def _acrin_multiphase_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    text = _normal_text(row.get("series_description", ""), row.get("source_path", ""))
    preferred = 0 if any(token in text for token in ("optional sag 3d", "pre 3 post", "3 posts", "prepost")) else 1
    return (
        -_to_int(row.get("dce_temporal_group_count", "")),
        -_to_int(row.get("image_count", "")),
        preferred,
        row.get("series_description", ""),
    )


def _preserve_original_order(rows: list[dict[str, str]], keep_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keep_ids = {id(row) for row in keep_rows}
    return [row for row in rows if id(row) in keep_ids]


def _looks_like_acrin_split_phase_set(rows: list[dict[str, str]]) -> bool:
    phase_rows = [row for row in rows if row.get("series_role") in ACRIN_DCE_ROLES and _to_int(row.get("image_count", "")) >= 8]
    if len(phase_rows) < 2:
        return False
    texts = [_normal_text(row.get("series_description", ""), row.get("source_path", "")) for row in phase_rows]
    if any(_looks_like_acrin_combined_multiphase_folder(text) for text in texts):
        return False
    return any(_looks_like_acrin_split_phase_folder(text) for text in texts)


def _has_acrin_combined_multiphase_candidate(rows: list[dict[str, str]]) -> bool:
    return any(
        row.get("series_role") in ACRIN_DCE_ROLES
        and _looks_like_acrin_combined_multiphase_folder(
            _normal_text(row.get("series_description", ""), row.get("source_path", ""))
        )
        for row in rows
    )


def _looks_like_acrin_split_phase_folder(text: str) -> bool:
    return any(
        token in text
        for token in (
            "3dtransdynprecon",
            "transdynprecon",
            "breast dyn",
            "nci dyn",
            "nci-dyn",
            "grass pregad",
            "grass rodeo",
            "spgr post",
            "spgr pre",
            "pre gad",
            "post gad",
            "post cont",
            "3dsagpre",
            "3dsagpost",
            "fspgr pre",
            "fspgr post",
            "bilat ax pre",
            "bilat ax post",
            "ax 3dgre",
            "sag t1 fs",
        )
    )


def _looks_like_acrin_combined_multiphase_folder(text: str) -> bool:
    return any(
        token in text
        for token in (
            "optional sag 3d",
            "pre 3 posts",
            "pre 3 post",
            "1 pre 2 post",
            "prepost",
            "multi phase",
            "multiphase",
        )
    )


def _mark_as_acrin_split_phase_folder(row: dict[str, str], evidence: str) -> None:
    image_count = _to_int(row.get("image_count", ""))
    row["dce_temporal_group_count"] = "1"
    row["dce_temporal_grouping_method"] = "acrin_split_phase_folder"
    row["dce_temporal_grouping_evidence"] = evidence
    row["dce_temporal_group_sizes"] = f"[{image_count}]" if image_count else ""
    row["dce_reference_slice_count"] = ""
    row["dce_temporal_groups_json"] = ""


def _looks_like_acrin_derived(text: str, sdyn_text: str = "") -> bool:
    return _has_acrin_sdyn_derived_token(sdyn_text or text) or any(
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
    ) or re.search(r"\bser\b|\bpe\s*[1-9]\b", text) is not None


def _has_acrin_sdyn_derived_token(text: str) -> bool:
    return re.search(r"(^|[ _])sdyn($|[ _])", text) is not None


def _looks_like_acrin_dce_source(text: str) -> bool:
    if _has_acrin_pre_token(text) or _has_acrin_post_token(text):
        return True
    return any(
        token in text
        for token in (
            "nci-dyn",
            " dyn",
            "dynamic",
            "3dtransdynprecon",
            "grass",
            "rodeo",
            "spgr",
            "fspgr",
            "3dspgr",
            "3d spgr",
            "vibe",
            "lufgre",
            "3dgre",
            "3d gre",
            "sag 3d",
            "ax 3d",
            "bilat ax",
            "hi-res",
            "sns prepost",
            "optional sag 3d",
        )
    )


def _has_acrin_pre_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpre\b",
            r"pregad",
            r"pre[\s_-]*gad",
            r"pre[\s_-]*contrast",
            r"prepost",
            r"pre\s*\d+\s*posts?",
            r"\d+d?sagpre",
            r"\d+d?fspgr\s*pre",
            r"\d+d?grass\s*pregad",
            r"\bfat\s*sat\s*off\b",
            r"\bnon\b",
        )
    )


def _has_acrin_post_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpost\b",
            r"\bposts\b",
            r"postgad",
            r"post[\s_-]*gad",
            r"post[\s_-]*contrast",
            r"post\s*cont",
            r"prepost",
            r"\d+\s*posts?",
            r"\d+d?sagpost",
            r"\d+d?fspgr\s*post",
            r"\bgad\b",
            r"\bcont\b",
            r"\buse\s*this\b",
        )
    )


def _normal_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _sdyn_boundary_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    text = text.replace("/", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def _to_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0
