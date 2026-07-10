from __future__ import annotations

from pathlib import Path


PathLike = str | Path


def resolve_path(value: PathLike | None, default: PathLike) -> Path:
    return Path(value if value is not None else default).expanduser().resolve(strict=False)


def label_file(label_root: PathLike | None, default_root: PathLike, name: str) -> Path:
    return resolve_path(label_root, default_root) / name

