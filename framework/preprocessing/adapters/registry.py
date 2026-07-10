from __future__ import annotations

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets import build_dataset_adapters


def build_default_adapters() -> list[DatasetAdapter]:
    return build_dataset_adapters()

