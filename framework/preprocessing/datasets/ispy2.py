from __future__ import annotations

from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.tcia import TCIAMetadataAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.common import PathLike, resolve_path


DATASET_ID = "ispy2"
DEFAULT_ROOT = Path("H:/breast/ISPY2")


def build_adapter(
    dataset_root: PathLike | None = None,
    metadata_csv: PathLike | None = None,
    label_root: PathLike | None = None,
) -> DatasetAdapter:
    root = resolve_path(dataset_root, DEFAULT_ROOT)
    label_base = resolve_path(label_root, root)
    return TCIAMetadataAdapter(
        dataset_id=DATASET_ID,
        dataset_root=root,
        metadata_csv=resolve_path(metadata_csv, root / "manifest-1641168072464" / "metadata.csv"),
        label_files=[
            label_base / "Multi-feature-MRI-NACT-Data.xlsx",
            label_base / "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx",
        ],
        label_key_candidates=("CLINICAL-TRIAL-SUBJECT-ID", "Patient_ID"),
    )

