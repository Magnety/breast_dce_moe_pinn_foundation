from __future__ import annotations

import re


def classify_series_role(
    description: str,
    modality: str = "",
    sop_class_name: str = "",
    source_path: str = "",
    dataset_id: str = "",
) -> str:
    """Classify a series into a conservative preprocessing role.

    Public TCIA manifests describe series at directory level, so this function
    uses robust text and modality cues. DCE phase order is refined later from
    DICOM headers during sequence analysis and conversion.
    """

    if dataset_id == "acrin_contralateral":
        from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.acrin_sequences import classify_acrin_series_role

        acrin_role = classify_acrin_series_role(
            description,
            modality=modality,
            sop_class_name=sop_class_name,
            source_path=source_path,
        )
        if acrin_role != "unknown":
            return acrin_role

    if dataset_id in {"duke", "ispy1", "ispy2", "breast_mri_nact_pilot", "advanced_mri_breast_lesions"}:
        from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.public_breast_sequences import classify_public_breast_series_role

        public_role = classify_public_breast_series_role(
            description,
            modality=modality,
            sop_class_name=sop_class_name,
            source_path=source_path,
            dataset_id=dataset_id,
        )
        if public_role != "unknown":
            return public_role

    if dataset_id == "fujian_pCR":
        from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.fujian_sequences import classify_fujian_pcr_series_role

        fujian_role = classify_fujian_pcr_series_role(
            description,
            modality=modality,
            sop_class_name=sop_class_name,
            source_path=source_path,
        )
        if fujian_role != "unknown":
            return fujian_role

    text = " ".join([description or "", modality or "", sop_class_name or "", source_path or ""]).lower()
    modality_upper = (modality or "").upper()

    if "segmentation" in text or modality_upper == "SEG":
        return "mask_seg"
    if re.search(r"\b(mask|roi|voi|contour|annotation)\b", text):
        return "mask"
    if "tram" in text:
        return "derived"
    if any(token in text for token in ("localizer", "locator", "scout", "calibration", "asset calibration")):
        return "localizer"
    if re.search(r"\bloc\b", text) or re.search(r"\bloc(?:alizer|ator)?$", text):
        return "localizer"
    if "secondary capture" in text or modality_upper in {"SC", "OT"}:
        return "secondary_capture"

    if _looks_like_parametric_map(text):
        if re.search(r"\bpe\s*([1-9])\b", text):
            return "DCE_PE"
        if re.search(r"\bser\b", text):
            return "DCE_SER"
        return "derived"

    if "adc" in text or "apparent diffusion" in text:
        return "ADC"
    if any(token in text for token in ("diffusion", "dwi", "dwssfse", "ep2d", "epi", " b=", "b800")):
        return "DWI"

    if _looks_like_dce_source(text):
        has_pre = _has_pre_contrast_token(text)
        has_post = _has_post_contrast_token(text)
        if has_pre and has_post:
            return "DCE"
        if has_pre:
            return "DCE_PRE"
        if has_post:
            return "DCE_POST"
        return "DCE"

    if "t2" in text or "stir" in text or "fse" in text or "tse" in text:
        return "T2"
    if "t1" in text or "spgr" in text or "vibe" in text or "gre" in text:
        if _has_post_contrast_token(text):
            return "T1_POST"
        if _has_pre_contrast_token(text):
            return "T1_PRE"
        return "T1"
    return "unknown"


def classify_dce_phase(description: str, source_path: str = "") -> tuple[str, str]:
    """Return a best-effort DCE phase label from text alone.

    The final pre/post assignment is improved with DICOM temporal metadata and
    acquisition times when available. This text rule keeps ambiguous cases marked
    as `unknown` instead of pretending certainty.
    """

    text = " ".join([description or "", source_path or ""]).lower()
    if _has_pre_contrast_token(text) or re.search(r"\bph\s*0\b", text):
        return "pre", "series text contains pre-contrast keyword"
    if _has_post_contrast_token(text):
        return "post", "series text contains post/pass/phase keyword"
    return "unknown", "no explicit pre/post keyword"


def infer_timepoint(subject_id: str, study_description: str, study_date: str = "") -> str:
    text = " ".join([subject_id or "", study_description or ""]).lower()
    match = re.search(r"\bt([0-4])\b", text)
    if match:
        return f"T{match.group(1)}"
    match = re.search(r"mri\s*([1-4])", text)
    if match:
        return f"MRI{match.group(1)}"
    if any(token in text for token in ("新辅前", "baseline", "pre")):
        return "pre_nact"
    if any(token in text for token in ("术前", "post", "follow")):
        return "pre_surgery"
    return study_date or "unknown"


def is_mask_role(series_role: str) -> bool:
    return series_role in {"mask", "mask_seg"}


def _looks_like_parametric_map(text: str) -> bool:
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


def _looks_like_dce_source(text: str) -> bool:
    if re.search(r"\bpre\b.*\b\d+\s*posts?\b", text) or re.search(r"\b\d+\s*posts?\b.*\bpre\b", text):
        return True
    if _has_pre_contrast_token(text) and _has_post_contrast_token(text) and re.search(
        r"\b(3d|ax|axial|sag|sagittal|t1|gre|spgr|fspgr|fgre|flash|vibe|thrive)\b",
        text,
    ):
        return True
    return any(
        token in text
        for token in (
            "dce",
            "dynamic",
            "dyn",
            "multiphase",
            "vibrant",
            "fl3d",
            "3dfgre",
            "ir3dfgre",
            "fspgr",
            "spgr",
            "thrive",
            "vibe",
            "fgre",
            "flash",
        )
    )


def _has_pre_contrast_token(text: str) -> bool:
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


def _has_post_contrast_token(text: str) -> bool:
    return any(
        re.search(pattern, text) is not None
        for pattern in (
            r"\bpost\b",
            r"\bposts\b",
            r"\bpostgad\b",
            r"\bpost[\s_-]*gad\b",
            r"\bpost[\s_-]*contrast\b",
            r"\+c\b",
            r"\bgad\b",
            r"\bpass\b",
            r"\b[1-9](?:st|nd|rd|th)\b",
            r"\bph\s*[1-9](?=[^0-9]|$)",
        )
    )
