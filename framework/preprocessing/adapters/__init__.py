"""Dataset-specific preprocessing adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter


def build_default_adapters() -> list["DatasetAdapter"]:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.registry import build_default_adapters as _build_default_adapters

    return _build_default_adapters()

__all__ = ["build_default_adapters"]
