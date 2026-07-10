from __future__ import annotations

from typing import Any


def build_acrin_core_labels(label_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce decoded ACRIN clinical forms to model-useful clinical labels."""

    output: dict[str, Any] = {}

    pcr_status = _normalized_pcr(_first_label(label_records, ("pCR", "PCR", "pathologic_complete_response")))
    if pcr_status in {"0", "1"}:
        output["pCR"] = int(pcr_status)
        output["pcr_status"] = pcr_status

    for key, aliases in {
        "HER2": ("HER2", "HER2 status", "HER2 Status", "HER2 Positive"),
        "ER": ("ER", "ER positive", "ER Positive"),
        "PR": ("PR", "PR positive", "PR Positive"),
    }.items():
        value = _first_label(label_records, aliases)
        normalized = _normalized_receptor(value)
        if _known_label_value(normalized):
            output[key] = normalized

    molecular_subtype = _first_label(label_records, ("molecular_subtype", "Molecular subtype", "HR_HER2_STATUS"))
    if _known_label_value(molecular_subtype):
        output["molecular_subtype"] = molecular_subtype

    age = _first_label(label_records, ("age", "Age", "Patient age at Registration"))
    if _known_label_value(age):
        output["age"] = age
    sex = _first_label(label_records, ("sex", "Sex", "Gender"))
    if _known_label_value(sex):
        output["sex"] = sex
    laterality = _normalized_laterality(
        _first_label(
            label_records,
            ("laterality", "Laterality", "site of breast malignancy", "site of lesion biopsied"),
        )
    )
    if _known_label_value(laterality):
        output["laterality"] = laterality

    pathologic_t = _first_nested_pathology(label_records, "pathologic_t") or _first_label(
        label_records,
        ("pathologic_t", "t classification / pathologic", "T CLASSIFICATION / PATHOLOGIC"),
    )
    pathologic_n = _first_nested_pathology(label_records, "pathologic_n") or _first_label(
        label_records,
        ("pathologic_n", "n stage / pathologic", "N STAGE / PATHOLOGIC"),
    )
    pathologic_m = _first_nested_pathology(label_records, "pathologic_m") or _first_label(
        label_records,
        ("pathologic_m", "m stage / pathologic", "M STAGE / PATHOLOGIC"),
    )
    pathologic_stage = _first_label(label_records, ("pathologic_stage", "Pathologic stage"))
    if not _known_label_value(pathologic_stage) and any(_known_label_value(value) for value in (pathologic_t, pathologic_n, pathologic_m)):
        pathologic_stage = " ".join(str(value) for value in (pathologic_t, pathologic_n, pathologic_m) if _known_label_value(value))
    if _known_label_value(pathologic_stage):
        output["pathologic_stage"] = pathologic_stage
    for key, value in {
        "pathologic_t": pathologic_t,
        "pathologic_n": pathologic_n,
        "pathologic_m": pathologic_m,
    }.items():
        if _known_label_value(value):
            output[key] = value

    tumor_size_cm = _first_float_label(label_records, ("tumor_size_cm", "Clinical size pre"))
    if tumor_size_cm is None:
        tumor_size_mm = _largest_numeric_label(
            label_records,
            (
                "tumor size (mm): dimension x (nn)",
                "tumor size (mm): dimension y (nn)",
                "tumor size (mm): dimension z (nn)",
                "TUMOR SIZE (MM): DIMENSION X (NN)",
                "TUMOR SIZE (MM): DIMENSION Y (NN)",
                "TUMOR SIZE (MM): DIMENSION Z (NN)",
            ),
        )
        tumor_size_cm = round(tumor_size_mm / 10.0, 4) if tumor_size_mm is not None else None
    if tumor_size_cm is not None:
        output["tumor_size_cm"] = tumor_size_cm

    pathology = {
        "histologic_type": _first_nested_pathology(label_records, "histologic_type")
        or _first_label(
            label_records,
            (
                "histologic_type",
                "Histologic type",
                "Cancer type",
                "types of invasive carcinoma",
                "TYPES OF INVASIVE CARCINOMA",
                "histology of index lesion",
                "HISTOLOGY OF INDEX LESION",
                "HISTOLOGY OF ADDITIONAL SPECIMEN",
            ),
        )
        or _first_yes_label(
            label_records,
            (
                "infiltrating ductal carcinoma",
                "infiltrating lobular carcinoma",
                "infiltrating carcinoma with ductal and lobular features",
                "tubular carcinoma",
                "mucinous carcinoma",
                "medullary carcinoma",
                "ductal carcinoma in situ",
                "lobular carcinoma in situ",
            ),
        ),
        "grade": _first_nested_pathology(label_records, "grade")
        or _first_label(label_records, ("grade", "Grade", "Tumor Grade", "grade of invasive ca", "GRADE OF INVASIVE CA", "grade of dcis", "GRADE OF DCIS")),
        "pathologic_t": pathologic_t,
        "pathologic_n": pathologic_n,
        "pathologic_m": pathologic_m,
    }
    pathology = {key: value for key, value in pathology.items() if _known_label_value(value)}
    if pathology:
        for key in ("histologic_type", "grade"):
            if _known_label_value(pathology.get(key)):
                output[key] = pathology[key]
        output["pathology"] = pathology

    survival = {
        "participant_status": _first_nested_survival(label_records, "participant_status")
        or _first_label(label_records, ("participant_status", "Participant's status")),
        "death_relationship_to_breast_cancer": _first_nested_survival(
            label_records, "death_relationship_to_breast_cancer"
        )
        or _first_label(label_records, ("death_relationship_to_breast_cancer", "Cause of death relationship to breast cancer")),
        "non_study_breast_cancer": _first_nested_survival(label_records, "non_study_breast_cancer")
        or _first_label(
            label_records,
            ("non_study_breast_cancer", "diagnosis of dcis or invasive cancer in the non-study breast"),
        ),
    }
    survival = {key: value for key, value in survival.items() if _known_label_value(value)}
    if survival:
        for key, value in survival.items():
            output[key] = value
        output["survival"] = survival

    source_count = _first_int_label(label_records, ("source_label_count",))
    output["source_label_count"] = source_count if source_count is not None else len(label_records)
    source_files = _source_files(label_records)
    if source_files:
        output["source_files"] = source_files
    return output


def _first_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> Any:
    alias_map = {alias.lower(): alias for alias in aliases}
    for record in label_records:
        labels = record.get("labels", {})
        lowered = {str(key).lower(): key for key in labels}
        for alias_lower in alias_map:
            key = lowered.get(alias_lower)
            if key is None:
                continue
            value = labels.get(key)
            if _known_label_value(value):
                return value
    return None


def _first_nested_pathology(label_records: list[dict[str, Any]], key: str) -> Any:
    return _first_nested(label_records, "pathology", key)


def _first_nested_survival(label_records: list[dict[str, Any]], key: str) -> Any:
    return _first_nested(label_records, "survival", key)


def _first_nested(label_records: list[dict[str, Any]], section: str, key: str) -> Any:
    for record in label_records:
        value = record.get("labels", {}).get(section, {})
        if isinstance(value, dict) and _known_label_value(value.get(key)):
            return value.get(key)
    return None


def _first_float_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> float | None:
    value = _first_label(label_records, aliases)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_int_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> int | None:
    value = _first_label(label_records, aliases)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _largest_numeric_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> float | None:
    values: list[float] = []
    alias_set = {alias.lower() for alias in aliases}
    for record in label_records:
        labels = record.get("labels", {})
        lowered = {str(key).lower(): key for key in labels}
        for alias in alias_set:
            key = lowered.get(alias)
            if key is None:
                continue
            try:
                values.append(float(labels.get(key)))
            except (TypeError, ValueError):
                continue
    return max(values) if values else None


def _first_yes_label(label_records: list[dict[str, Any]], aliases: tuple[str, ...]) -> Any:
    alias_set = {alias.lower() for alias in aliases}
    for record in label_records:
        labels = record.get("labels", {})
        lowered = {str(key).lower(): key for key in labels}
        for alias in alias_set:
            key = lowered.get(alias)
            if key is not None and str(labels.get(key)).strip().lower() == "yes":
                return key
    return None


def _normalized_pcr(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "pcr", "complete response", "pathologic complete response"}:
        return "1"
    if text in {"0", "0.0", "false", "no", "non-pcr", "non pcr", "not pcr"}:
        return "0"
    return "unknown"


def _normalized_receptor(value: Any) -> Any:
    if value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"positive", "pos", "1", "1.0", "+", "3+"}:
        return "positive"
    if text in {"negative", "neg", "0", "0.0", "-"}:
        return "negative"
    if any(token in text for token in ("equivocal", "2+")):
        return "equivocal"
    return value


def _normalized_laterality(value: Any) -> Any:
    if value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "bilateral" in text:
        return "bilateral"
    return value


def _source_files(label_records: list[dict[str, Any]]) -> list[str]:
    files: set[str] = set()
    for record in label_records:
        source_file = record.get("source_file", "")
        if source_file:
            files.add(str(source_file))
        labels = record.get("labels", {})
        label_files = labels.get("source_files", [])
        if isinstance(label_files, list):
            files.update(str(item) for item in label_files if item)
    return sorted(files)


def _known_label_value(value: Any) -> bool:
    if value in (None, "", "unknown"):
        return False
    if isinstance(value, dict):
        return any(_known_label_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_known_label_value(item) for item in value)
    return True
