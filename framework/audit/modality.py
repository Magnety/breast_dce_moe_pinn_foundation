from __future__ import annotations

from typing import Any


def infer_modality(header: dict[str, Any], image_count: int = 0) -> dict[str, Any]:
    """Infer MRI modality conservatively from DICOM metadata.

    The result includes evidence and a status. Ambiguous cases are explicitly marked
    `needs_review` or `unknown` instead of being forced into a modality.
    """

    text_fields = [
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ImageType",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
    ]
    text = " ".join(_flatten(header.get(field)) for field in text_fields).lower()
    evidence: list[str] = []
    candidates: list[str] = []
    flags: dict[str, bool] = {
        "fat_suppressed": False,
        "contrast_enhanced": False,
        "pre_contrast": False,
        "post_contrast": False,
        "dce_candidate": False,
    }

    def add(candidate: str, why: str) -> None:
        if candidate not in candidates:
            candidates.append(candidate)
        evidence.append(why)

    if "adc" in text or "apparent diffusion" in text:
        add("ADC", "series text contains ADC")
    diffusion_b = header.get("DiffusionBValue")
    if any(token in text for token in ("dwi", "diff", "diffusion", "tracew", "trace")) or diffusion_b is not None:
        add("DWI", "diffusion keyword or b-value present")
    if any(token in text for token in ("t2", "tse", "frfse", "stir")):
        add("T2", "T2/STIR-like sequence keyword present")
    if any(token in text for token in ("t1", "spgr", "vibe", "thrive", "fl3d", "flash")):
        add("T1", "T1-like sequence keyword present")

    if any(token in text for token in ("fat", "fs", "fatsat", "fat sat", "spair", "stir")):
        flags["fat_suppressed"] = True
        evidence.append("fat suppression keyword present")
    if header.get("ContrastBolusAgent") or any(
        token in text for token in ("post", "gd", "gadolinium", "contrast", "+c")
    ):
        flags["contrast_enhanced"] = True
        evidence.append("contrast metadata or keyword present")
    if any(token in text for token in ("pre", "plain", "noncontrast", "non contrast")):
        flags["pre_contrast"] = True
        evidence.append("pre-contrast keyword present")
    if any(token in text for token in ("post", "phase", "dyn", "dynamic", "sub", "subtract")):
        flags["post_contrast"] = True
        evidence.append("post/dynamic/subtraction keyword present")

    temporal_position = header.get("TemporalPositionIdentifier")
    temporal_count = header.get("NumberOfTemporalPositions")
    if temporal_position is not None or _as_int(temporal_count) > 1:
        flags["dce_candidate"] = True
        add("DCE", "temporal DICOM metadata present")
    elif any(token in text for token in ("dce", "dyn", "dynamic", "phase", "subtraction", "subtract")):
        flags["dce_candidate"] = True
        add("DCE", "dynamic/phase/subtraction keyword present")

    if len(candidates) == 1:
        status = "identified"
        primary = candidates[0]
        confidence = 0.85 if evidence else 0.5
    elif len(candidates) > 1:
        if "DCE" in candidates and "T1" in candidates:
            status = "identified"
            primary = "DCE"
            confidence = 0.8
        elif "ADC" in candidates and "DWI" in candidates:
            status = "needs_review"
            primary = "ADC"
            confidence = 0.6
            evidence.append("ADC and DWI evidence both present")
        else:
            status = "needs_review"
            primary = candidates[0]
            confidence = 0.5
    else:
        status = "unknown"
        primary = "unknown"
        confidence = 0.0

    return {
        "primary": primary,
        "candidates": candidates,
        "status": status,
        "confidence": confidence,
        "flags": flags,
        "evidence": evidence,
        "image_count": image_count,
    }


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

