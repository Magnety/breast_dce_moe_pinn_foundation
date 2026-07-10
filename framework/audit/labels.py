from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


IDENTIFIER_HINTS = ("patient", "case", "subject", "study", "series", "mrn", "id", "uid", "accession")


def analyze_label_file(path: Path, max_size_mb: int = 64) -> dict[str, Any]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return {
            "path": str(path),
            "status": "skipped_large_file",
            "size_mb": round(size_mb, 3),
            "columns": [],
            "fields": [],
        }
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return _analyze_delimited(path, delimiter="\t" if suffix == ".tsv" else None)
    if suffix in (".xlsx", ".xls"):
        return _analyze_excel(path)
    if suffix == ".json":
        return _analyze_json(path)
    return {
        "path": str(path),
        "status": "unsupported_label_extension",
        "columns": [],
        "fields": [],
    }


def _analyze_delimited(path: Path, delimiter: str | None = None) -> dict[str, Any]:
    rows_seen = 0
    missing: Counter[str] = Counter()
    values: dict[str, Counter[str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            if delimiter is None:
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","
            if not isinstance(delimiter, str) or len(delimiter) != 1:
                delimiter = ","
            try:
                reader = csv.DictReader(handle, delimiter=delimiter)
                columns = list(reader.fieldnames or [])
                for column in columns:
                    values[column] = Counter()
                for row in reader:
                    rows_seen += 1
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if value == "":
                            missing[column] += 1
                        elif not _looks_like_identifier(column):
                            values[column][value] += 1
                    if rows_seen >= 10000:
                        break
            except (csv.Error, ValueError) as exc:
                return {
                    "path": str(path),
                    "status": "needs_review_delimited_parse_error",
                    "error": str(exc),
                    "columns": [],
                    "fields": [],
                }
    except UnicodeDecodeError:
        return {
            "path": str(path),
            "status": "needs_review_encoding",
            "columns": [],
            "fields": [],
        }

    fields = [_field_summary(column, rows_seen, missing[column], values[column]) for column in columns]
    return {
        "path": str(path),
        "status": "ok",
        "format": "delimited",
        "rows_sampled": rows_seen,
        "columns": columns,
        "candidate_key_columns": [column for column in columns if _looks_like_identifier(column)],
        "fields": fields,
    }


def _analyze_excel(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {
            "path": str(path),
            "status": "needs_optional_dependency",
            "dependency": "pandas/openpyxl",
            "columns": [],
            "fields": [],
        }
    try:
        workbook = pd.ExcelFile(path)
        sheets: list[dict[str, Any]] = []
        for sheet_name in workbook.sheet_names[:10]:
            frame = workbook.parse(sheet_name=sheet_name, nrows=10000)
            fields = []
            for column in frame.columns:
                series = frame[column]
                counter = Counter(
                    str(value)
                    for value in series.dropna().astype(str).head(10000).tolist()
                    if not _looks_like_identifier(str(column))
                )
                fields.append(
                    _field_summary(str(column), int(len(frame)), int(series.isna().sum()), counter)
                )
            sheets.append(
                {
                    "sheet": str(sheet_name),
                    "rows_sampled": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                    "candidate_key_columns": [
                        str(column) for column in frame.columns if _looks_like_identifier(str(column))
                    ],
                    "fields": fields,
                }
            )
    except Exception as exc:
        return {
            "path": str(path),
            "status": "needs_review_read_error",
            "error": str(exc),
            "columns": [],
            "fields": [],
        }
    return {"path": str(path), "status": "ok", "format": "excel", "sheets": sheets}


def _analyze_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return {
            "path": str(path),
            "status": "needs_review_read_error",
            "error": str(exc),
            "columns": [],
            "fields": [],
        }
    keys = sorted(_json_keys(data))[:500]
    return {
        "path": str(path),
        "status": "ok",
        "format": "json",
        "columns": keys,
        "candidate_key_columns": [key for key in keys if _looks_like_identifier(key)],
        "fields": [{"name": key, "status": "needs_review"} for key in keys],
    }


def _json_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full_key)
            keys.update(_json_keys(item, full_key))
    elif isinstance(value, list):
        for item in value[:20]:
            keys.update(_json_keys(item, prefix))
    return keys


def _field_summary(name: str, rows_seen: int, missing_count: int, counter: Counter[str]) -> dict[str, Any]:
    non_missing = max(rows_seen - missing_count, 0)
    unique_count = len(counter)
    status = "needs_review" if _looks_like_identifier(name) else "ok"
    return {
        "name": name,
        "status": status,
        "missing_count": int(missing_count),
        "missing_fraction": round(missing_count / rows_seen, 6) if rows_seen else None,
        "unique_values_sampled": unique_count,
        "top_values": counter.most_common(20) if status == "ok" and non_missing else [],
    }


def _looks_like_identifier(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in IDENTIFIER_HINTS)
