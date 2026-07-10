from __future__ import annotations

from pathlib import Path

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.base import DatasetAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.adapters.tcia import TCIAMetadataAdapter
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.preprocessing.datasets.common import PathLike, resolve_path


DATASET_ID = "tcga_brca"
DEFAULT_ROOT = Path("G:/breast/TCIA_TCGA-BRCA_09-16-2015")
INNER_DIR = "TCIA_TCGA-BRCA_09-16-2015"
LABEL_DIR = "brca_tcga_pan_can_atlas_2018"


def build_adapter(
    dataset_root: PathLike | None = None,
    metadata_csv: PathLike | None = None,
    label_root: PathLike | None = None,
) -> DatasetAdapter:
    root = resolve_path(dataset_root, DEFAULT_ROOT)
    inner = root / INNER_DIR
    label_base = resolve_path(label_root, inner / LABEL_DIR)
    return TCIAMetadataAdapter(
        dataset_id=DATASET_ID,
        dataset_root=root,
        metadata_csv=resolve_path(metadata_csv, inner / "metadata.csv"),
        label_files=[
            label_base / "data_clinical_patient.txt",
            label_base / "data_clinical_sample.txt",
            label_base / "data_timeline_status.txt",
            label_base / "data_timeline_treatment.txt",
        ],
        label_key_candidates=("PATIENT_ID",),
    )

