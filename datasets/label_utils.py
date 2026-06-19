from __future__ import annotations

import math
from typing import Any

import numpy as np

BINARY_POSITIVE = {"1", "true", "yes", "y", "positive", "pos", "pcr", "response", "responder", "+"}
BINARY_NEGATIVE = {
    "0",
    "false",
    "no",
    "n",
    "negative",
    "neg",
    "non-pcr",
    "non_pcr",
    "nonresponse",
    "non-responder",
    "-",
}
MISSING_VALUES = {"", "na", "nan", "none", "null", "unknown", "missing", "not_available", "-1"}

DEFAULT_TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival_time", "survival_event")
BINARY_TASKS = ("pCR", "HER2", "ER", "PR", "HR")
SUBTYPE_TO_ID = {
    "luminal_a": 0,
    "luminal-a": 0,
    "luminal_b": 1,
    "luminal-b": 1,
    "her2_positive": 2,
    "her2+": 2,
    "triple_negative": 3,
    "tnbc": 3,
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in MISSING_VALUES


def parse_binary_label(value: Any) -> tuple[float, float]:
    """Return ``(label, mask)`` without treating missing labels as negative."""

    if is_missing(value):
        return -1.0, 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric < 0:
            return -1.0, 0.0
        return float(numeric >= 0.5), 1.0
    text = str(value).strip().lower()
    if text in BINARY_POSITIVE:
        return 1.0, 1.0
    if text in BINARY_NEGATIVE:
        return 0.0, 1.0
    try:
        return float(float(text) >= 0.5), 1.0
    except ValueError:
        return -1.0, 0.0


def parse_continuous_label(value: Any) -> tuple[float, float]:
    if is_missing(value):
        return -1.0, 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -1.0, 0.0
    if numeric < 0:
        return -1.0, 0.0
    return numeric, 1.0


def parse_subtype_label(value: Any) -> tuple[float, float]:
    if is_missing(value):
        return -1.0, 0.0
    text = str(value).strip().lower().replace(" ", "_")
    if text in SUBTYPE_TO_ID:
        return float(SUBTYPE_TO_ID[text]), 1.0
    try:
        numeric = int(float(text))
    except ValueError:
        return -1.0, 0.0
    return float(numeric), 1.0 if numeric >= 0 else 0.0


def parse_labels(row: dict[str, Any], label_columns: dict[str, str] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    columns = label_columns or {task: task for task in DEFAULT_TASKS}
    labels: dict[str, float] = {}
    masks: dict[str, float] = {}
    for task in DEFAULT_TASKS:
        column = columns.get(task, task)
        value = row.get(column)
        if task in BINARY_TASKS:
            label, mask = parse_binary_label(value)
        elif task == "molecular_subtype":
            label, mask = parse_subtype_label(value)
        else:
            label, mask = parse_continuous_label(value)
        labels[task] = label
        masks[task] = mask

    if masks.get("HR", 0.0) == 0.0 and masks.get("ER", 0.0) > 0.0 and masks.get("PR", 0.0) > 0.0:
        labels["HR"] = float(max(labels["ER"], labels["PR"]) >= 0.5)
        masks["HR"] = 1.0
    masks["survival"] = float(masks.get("survival_time", 0.0) > 0.0 and masks.get("survival_event", 0.0) > 0.0)
    return labels, masks
