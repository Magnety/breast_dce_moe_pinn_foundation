from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
import warnings
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")

_TRUTHY = {"1", "true", "yes", "on"}
_WARNINGS_SUPPRESSED = False


def suppress_warnings() -> None:
    """Hide non-fatal warning noise from IDE and terminal preprocessing runs."""

    global _WARNINGS_SUPPRESSED
    if _WARNINGS_SUPPRESSED:
        return
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    logging.getLogger("py.warnings").setLevel(logging.ERROR)
    logging.getLogger("pydicom").setLevel(logging.ERROR)
    try:
        import SimpleITK as sitk
    except Exception:
        _WARNINGS_SUPPRESSED = True
        return
    silence_simpleitk_warnings(sitk)
    _WARNINGS_SUPPRESSED = True


def silence_simpleitk_warnings(sitk: Any) -> None:
    try:
        sitk.ProcessObject_GlobalWarningDisplayOff()
    except Exception:
        pass


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool | None = None,
) -> Iterator[T]:
    """Iterate with tqdm when available, otherwise with a small built-in progress line."""

    disabled = _progress_disabled() if disable is None else disable
    if disabled:
        yield from iterable
        return

    try:
        from tqdm.auto import tqdm
    except Exception:
        yield from _basic_progress_iter(iterable, total=total, desc=desc, unit=unit)
        return

    yield from tqdm(
        iterable,
        total=total,
        desc=desc or None,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


@contextlib.contextmanager
def progress_counter(
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool | None = None,
) -> Iterator[Any]:
    disabled = _progress_disabled() if disable is None else disable
    if disabled:
        yield _NullProgressCounter()
        return

    try:
        from tqdm.auto import tqdm
    except Exception:
        counter = _BasicProgressCounter(total=total, desc=desc, unit=unit)
    else:
        counter = tqdm(total=total, desc=desc or None, unit=unit, dynamic_ncols=True, leave=True)
    try:
        yield counter
    finally:
        counter.close()


def _progress_disabled() -> bool:
    if os.environ.get("BREAST_MRI_DISABLE_PROGRESS", "").strip().lower() in _TRUTHY:
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return False


def _basic_progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterator[T]:
    counter = _BasicProgressCounter(total=total, desc=desc, unit=unit)
    try:
        for item in iterable:
            yield item
            counter.update()
    finally:
        counter.close()


class _NullProgressCounter:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


class _BasicProgressCounter:
    def __init__(self, *, total: int | None, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc or "Progress"
        self.unit = unit
        self.count = 0
        self.start = time.monotonic()
        self.last_write = 0.0
        self._closed = False
        self._write(force=True)

    def update(self, n: int = 1) -> None:
        self.count += n
        self._write(force=self.total is not None and self.count >= self.total)

    def close(self) -> None:
        if self._closed:
            return
        self._write(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._closed = True

    def _write(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_write < 0.5:
            return
        self.last_write = now
        if self.total:
            ratio = min(max(self.count / self.total, 0.0), 1.0)
            width = 24
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            text = f"\r{self.desc}: |{bar}| {self.count}/{self.total} {self.unit}"
        else:
            elapsed = max(now - self.start, 1e-6)
            rate = self.count / elapsed
            text = f"\r{self.desc}: {self.count} {self.unit} [{rate:.1f} {self.unit}/s]"
        sys.stderr.write(text)
        sys.stderr.flush()
