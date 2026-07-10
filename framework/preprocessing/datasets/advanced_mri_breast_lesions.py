from __future__ import annotations

from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.tcia import TCIAMetadataAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.common import PathLike, resolve_path


DATASET_ID = "advanced_mri_breast_lesions"
DEFAULT_ROOT = Path("G:/breast/Advanced-MRI-Breast-Lesions-DA-RAD-APR2024-00")


def build_adapter(
    dataset_root: PathLike | None = None,
    metadata_csv: PathLike | None = None,
    label_root: PathLike | None = None,
) -> DatasetAdapter:
    root = resolve_path(dataset_root, DEFAULT_ROOT)
    label_base = resolve_path(label_root, root / "Advanced-MRI-Breast-Lesions")
    return TCIAMetadataAdapter(
        dataset_id=DATASET_ID,
        dataset_root=root,
        metadata_csv=resolve_path(metadata_csv, root / "metadata.csv"),
        label_files=[label_base / "Advanced-MRI-Breast-Lesions-DA-Clinical-Sep2024.xlsx"],
        label_key_candidates=("Patient ID",),
    )

