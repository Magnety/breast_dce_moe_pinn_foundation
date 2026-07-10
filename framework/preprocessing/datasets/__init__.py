from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets import (
    acrin_contralateral,
    advanced_mri_breast_lesions,
    breast_mri_nact_pilot,
    duke,
    fujian_pcr,
    ispy2,
    tcga_brca,
)


DATASET_BUILDERS = {
    duke.DATASET_ID: duke.build_adapter,
    ispy2.DATASET_ID: ispy2.build_adapter,
    breast_mri_nact_pilot.DATASET_ID: breast_mri_nact_pilot.build_adapter,
    advanced_mri_breast_lesions.DATASET_ID: advanced_mri_breast_lesions.build_adapter,
    tcga_brca.DATASET_ID: tcga_brca.build_adapter,
    acrin_contralateral.DATASET_ID: acrin_contralateral.build_adapter,
    fujian_pcr.DATASET_ID: fujian_pcr.build_adapter,
}


def available_dataset_ids() -> tuple[str, ...]:
    return tuple(DATASET_BUILDERS.keys())


def build_dataset_adapter(
    dataset_id: str,
    dataset_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    label_root: str | Path | None = None,
) -> DatasetAdapter:
    try:
        builder = DATASET_BUILDERS[dataset_id]
    except KeyError as exc:
        options = ", ".join(available_dataset_ids())
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Available: {options}") from exc
    return builder(dataset_root=dataset_root, metadata_csv=metadata_csv, label_root=label_root)


def build_dataset_adapters(dataset_ids: Iterable[str] | None = None) -> list[DatasetAdapter]:
    selected = list(dataset_ids) if dataset_ids is not None else list(available_dataset_ids())
    return [build_dataset_adapter(dataset_id) for dataset_id in selected]

