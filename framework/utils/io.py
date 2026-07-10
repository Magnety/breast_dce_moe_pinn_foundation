from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .security import PathPolicy


def _prepare_target(path: str | os.PathLike[str], policy: PathPolicy | None) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    if policy is not None:
        target = policy.assert_write_allowed(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str, policy: PathPolicy | None = None) -> Path:
    target = _prepare_target(path, policy)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise
    return target


def atomic_write_json(
    path: str | os.PathLike[str],
    data: Any,
    policy: PathPolicy | None = None,
    indent: int = 2,
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True),
        policy=policy,
    )


def atomic_write_csv(
    path: str | os.PathLike[str],
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
    policy: PathPolicy | None = None,
) -> Path:
    target = _prepare_target(path, policy)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise
    return target


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value

