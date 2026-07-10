from __future__ import annotations

from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.tcia import TCIAMetadataAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.common import PathLike, resolve_path


DATASET_ID = "acrin_contralateral"
DEFAULT_ROOT = Path("F:/open_data/breast/ACRIN-Contralateral-Breast-MR")
MANIFEST_DIR = (
    "ACRIN-Contralateral-Breast-MR-Feb-2021-manifest/"
    "ACRIN-Contralateral-Breast-MR-Feb-2021-manifest"
)
LABEL_DIR = (
    "ACRIN-6667-Contralateral-Breast-MRI-Clinical-Data-Anonymized/"
    "ACRIN 6667 Contralateral Breast MRI Clinical Data Anonymized"
)


def build_adapter(
    dataset_root: PathLike | None = None,
    metadata_csv: PathLike | None = None,
    label_root: PathLike | None = None,
) -> DatasetAdapter:
    root = resolve_path(dataset_root, DEFAULT_ROOT)
    label_base = resolve_path(label_root, root / LABEL_DIR)
    label_files = sorted(label_base.glob("* reviewed.csv")) if label_base.exists() else []
    return TCIAMetadataAdapter(
        dataset_id=DATASET_ID,
        dataset_root=root,
        metadata_csv=resolve_path(metadata_csv, root / MANIFEST_DIR / "metadata.csv"),
        label_files=label_files,
        label_key_candidates=("cn", "m3e4deid", "m4e6deid", "qae11deid"),
    )

