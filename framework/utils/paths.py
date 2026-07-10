from __future__ import annotations

import os
from pathlib import Path


def safe_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path without resolving symlinks or touching slow drives."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def windows_extended_path(path: str | os.PathLike[str]) -> str:
    """Return a Windows extended-length path when needed.

    Some TCIA extraction trees exceed the classic 260-character Windows path
    limit. pathlib can enumerate those directories but then report false from
    exists()/is_dir(). The extended prefix keeps local preprocessing honest.
    """

    text = str(safe_absolute_path(path))
    if os.name != "nt":
        return text
    if text.startswith("\\\\?\\"):
        return text
    if len(text) < 248:
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def path_exists(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path).expanduser()
    try:
        if candidate.exists():
            return True
    except OSError:
        pass
    if os.name != "nt":
        return False
    try:
        return Path(windows_extended_path(candidate)).exists()
    except OSError:
        return False


def path_is_dir(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path).expanduser()
    try:
        if candidate.is_dir():
            return True
    except OSError:
        pass
    if os.name != "nt":
        return False
    try:
        return Path(windows_extended_path(candidate)).is_dir()
    except OSError:
        return False
