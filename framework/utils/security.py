from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class PathSecurityError(RuntimeError):
    """Raised when a path violates the configured read/write policy."""


def resolve_path(path: str | os.PathLike[str], base: str | os.PathLike[str] | None = None) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() and base is not None:
        raw = Path(base).expanduser() / raw
    return raw.resolve(strict=False)


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def is_relative_to(path: Path, parent: Path) -> bool:
    child_norm = _norm(path)
    parent_norm = _norm(parent)
    try:
        return os.path.commonpath([child_norm, parent_norm]) == parent_norm
    except ValueError:
        return False


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = _norm(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


@dataclass(frozen=True)
class PathPolicy:
    """Restricts reads to known roots and writes to configured output locations."""

    raw_roots: tuple[Path, ...]
    output_root: Path
    project_root: Path
    extra_read_roots: tuple[Path, ...] = field(default_factory=tuple)
    extra_write_roots: tuple[Path, ...] = field(default_factory=tuple)
    allow_writes_under_raw_roots: bool = False

    @classmethod
    def from_config(cls, config: dict, project_root: str | Path | None = None) -> "PathPolicy":
        config_path = config.get("_config_path")
        config_base = Path(config_path).parent if config_path else None
        root_base = resolve_path(
            project_root or config.get("project", {}).get("project_root", "."),
            base=config_base,
        )
        paths = config.get("paths", {})
        raw_roots = tuple(resolve_path(path) for path in paths.get("raw_roots", []))
        output_root = resolve_path(paths.get("output_root", "outputs"), base=root_base)
        write_roots = []
        for raw_path in (
            paths.get("processed_root"),
            paths.get("upload_bundle_root"),
            config.get("preprocessing", {}).get("processed_root"),
            config.get("training", {}).get("experiments_root"),
        ):
            if raw_path:
                write_roots.append(resolve_path(raw_path, base=root_base))
        return cls(
            raw_roots=raw_roots,
            output_root=output_root,
            project_root=root_base,
            extra_read_roots=tuple(write_roots),
            extra_write_roots=tuple(write_roots),
            allow_writes_under_raw_roots=bool(
                config.get("safety", {}).get("allow_writes_under_raw_roots", False)
            ),
        )

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return tuple(_dedupe((*self.raw_roots, self.output_root, self.project_root, *self.extra_read_roots)))

    def assert_read_allowed(self, path: str | os.PathLike[str]) -> Path:
        resolved = resolve_path(path)
        if any(is_relative_to(resolved, root) for root in self.read_roots):
            return resolved
        roots = ", ".join(str(root) for root in self.read_roots)
        raise PathSecurityError(f"Read path is outside configured roots: {resolved}; roots: {roots}")

    def assert_write_allowed(self, path: str | os.PathLike[str]) -> Path:
        resolved = resolve_path(path)
        if not self.allow_writes_under_raw_roots and any(
            is_relative_to(resolved, raw_root) for raw_root in self.raw_roots
        ):
            raise PathSecurityError(f"Refusing to write under raw data root: {resolved}")
        write_roots = (self.output_root, self.project_root, *self.extra_write_roots)
        if not any(is_relative_to(resolved, root) for root in write_roots):
            raise PathSecurityError(
                f"Write path must be under an allowed output root: {resolved}"
            )
        return resolved

    def ensure_output_dir(self, path: str | os.PathLike[str]) -> Path:
        resolved = self.assert_write_allowed(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
