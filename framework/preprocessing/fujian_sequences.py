from __future__ import annotations

import re
from typing import Any


FUJIAN_DATASET_ID = "fujian_pCR"


def classify_fujian_pcr_series_role(
    description: str,
    modality: str = "",
    sop_class_name: str = "",
    source_path: str = "",
) -> str:
    """Classify Fujian pCR flat-folder DICOM series.

    Fujian cases often store all DICOM files from several RadiAnt-visible series
    in one directory.  The useful DCE sources appear either as one VIBRANT DCE
    series with multiple temporal volumes or as cross-series Ph1/Ph2/... Dyn AX
    VIBRANT phases.  T2 series can coexist in the same directory and must stay
    visible as T2 rather than being swallowed by the DCE scan.
    """

    text = _normal_text(description, source_path, modality, sop_class_name)
    modality_upper = (modality or "").upper()
    if "segmentation" in text or modality_upper == "SEG":
        return "mask_seg"
    if re.search(r"\b(mask|roi|voi|contour|annotation)\b", text):
        return "mask"
    if any(token in text for token in ("localizer", "locator", "scout", "calibration")):
        return "localizer"
    if re.search(r"\bloc\b", text):
        return "localizer"
    if "secondary capture" in text or modality_upper in {"SC", "OT"}:
        return "secondary_capture"
    if _looks_like_derived_map(text):
        return "derived"
    if "adc" in text or "apparent diffusion" in text:
        return "ADC"
    if any(token in text for token in ("diffusion", "dwi", "dwssfse", "ep2d", "epi", " b=", "b800")):
        return "DWI"
    if _looks_like_t2(text):
        return "T2"
    if _looks_like_fujian_dce_source(text):
        return _phase_role_from_text(text)
    if "t1" in text or "spgr" in text or "vibe" in text or "gre" in text:
        if _has_post_token(text):
            return "T1_POST"
        if _has_pre_token(text):
            return "T1_PRE"
        return "T1"
    return "unknown"


def effective_fujian_pcr_series_role(row: dict[str, str]) -> str:
    role = row.get("series_role", "")
    if row.get("dataset_id", "") != FUJIAN_DATASET_ID:
        return role
    fujian_role = classify_fujian_pcr_series_role(
        row.get("series_description", ""),
        modality=row.get("modality", ""),
        sop_class_name=row.get("sop_class_name", ""),
        source_path=row.get("source_path", "") or row.get("relative_path", ""),
    )
    return fujian_role if fujian_role != "unknown" else role


def fujian_phase_sort_number(text: str) -> int:
    normalized = _normal_text(text)
    match = re.search(r"\bph\s*([0-9]+)\b", normalized)
    if match:
        return int(match.group(1))
    match = re.search(r"\bphase\s*([0-9]+)\b", normalized)
    if match:
        return int(match.group(1))
    return 10**9


def _looks_like_fujian_dce_source(text: str) -> bool:
    if re.search(r"\bph\s*[0-9]+\b", text) and any(token in text for token in ("dyn", "dce", "vibrant")):
        return True
    return any(
        token in text
        for token in (
            "vibrant dce",
            "dce vibrant",
            "dyn ax vibrant",
            "dynamic ax vibrant",
            "ax vibrant",
            "vibrant",
            "dynamic",
            " dce ",
            " dyn ",
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
            r"\bpre\s*gad\b",
            r"\bpre\s*contrast\b",
            r"\bnon\s*contrast\b",
            r"\bph\s*0\b",
        )
    )


def _has_post_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpost\b",
            r"\bposts\b",
            r"\bpostgad\b",
            r"\bpost\s*gad\b",
            r"\bpost\s*contrast\b",
            r"\bgad\b",
            r"\bpass\b",
            r"\b[1-9](?:st|nd|rd|th)\b",
            r"\bph\s*[1-9][0-9]*\b",
        )
    )


def _looks_like_t2(text: str) -> bool:
    return any(token in text for token in ("t2", "stir", "fse", "tse")) and not _looks_like_fujian_dce_source(text)


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
    ) or re.search(r"\b(pe\s*[1-9]|ser)\b", text) is not None


def _normal_text(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
