# Breast-DCE-MoE-PINN Foundation Model

This folder is the standalone implementation of our DCE-MRI-first multimodal breast cancer foundation model. It follows the organization style of the referenced `breast_tumor_foundation_model` project, but it does not share the generic SOTA comparison solver or model registry because this method has different assumptions: variable modalities, variable DCE phases, variable treatment visits, masked labels, ROI-free lesion attention and PINN hemodynamic modeling.

The self-contained training stack is:

- `main.py`: unified command line entry point.
- `solver.py`: dedicated Breast-DCE-MoE-PINN solver for pretrain, finetune and infer modes.
- `datasets/`: manifest loading, variable-modality collate, label parsing, transforms and 3D augmentation.
- `models/`: patch embedding, metadata embeddings, DCE temporal encoder, lesion-query MIL, PINN, MoE and heads.
- `losses/`: masked multi-task, survival, PINN, temporal and consistency losses.
- `analysis/`: Grad-CAM hooks, attention export, hemodynamic map export, metrics and visualization helpers.
- `configs/`: method-local configs and an example manifest.

## What Is Implemented

- Manifest-based variable modality dataset.
- Batch collate for different numbers of volumes, DCE phases, visits and labels.
- Shared 3D patch embedding with modality, phase, visit and dataset embeddings.
- DCE temporal encoder with phase masks and temporal attention output.
- ROI-free lesion query MIL with patch attention export tensors.
- Simplified PINN hemodynamic module using relative enhancement and extended-Tofts-style constraints.
- Task-aware MoE fusion and multi-task heads for pCR, HER2, ER, PR, HR, molecular subtype and survival risk.
- Masked multi-task loss so missing labels never become negative labels.
- Pretrain, finetune, inference and hemodynamic map export entry points.
- Unified inference artifacts for dashboarding: probabilities, final-layer
  features, t-SNE figures and modality contribution SHAP-style plots.

## Manifest Format

One row can represent one volume:

```csv
patient_id,sample_id,dataset_id,visit_timepoint,split,modality,phase_index,relative_time,path,pCR,HER2,ER,PR,HR,molecular_subtype,survival_time,survival_event
demo_patient_001,demo_patient_001_MRI1,ispy2,MRI1,train,DCE,0,0.0,/path/dce0.npy,1,0,1,1,1,0,1200,0
demo_patient_001,demo_patient_001_MRI1,ispy2,MRI1,train,DCE,1,1.0,/path/dce1.npy,1,0,1,1,1,0,1200,0
demo_patient_001,demo_patient_001_MRI1,ispy2,MRI1,train,T2,,0.0,/path/t2.npy,1,0,1,1,1,0,1200,0
```

Alternatively, a row may include `series_manifest_path` pointing to a JSON file with an `images` list. Supported image keys include `path`, `file_path`, `npy_path`, `modality`, `phase_index` and `relative_time`.

Internal tensor convention is `[C, D, H, W]` for each volume and `[B, V, C, D, H, W]` after collate.

## Commands

Pretrain:

```bash
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation pretrain
```

Finetune:

```bash
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation finetune
```

Infer:

```bash
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation infer \
  --output-csv outputs/breast_dce_moe_pinn/infer/predictions.csv
```

Export hemodynamic maps from a `[T,D,H,W]` DCE sequence:

```bash
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation export-maps \
  --dce-npy /path/to/dce_sequence.npy \
  --output-dir outputs/breast_dce_moe_pinn/maps
```

External project-level configs under `configs/breast_dce_moe_pinn_foundation/` are kept as convenience copies, but the default commands above use the method-local configs in this folder.

## Data Splitting

Pretraining and finetuning use different default policies:

- **Pretraining** consumes every sample in the manifest by default. The
  `train_split` field is set to `null`, so DCE/T1/T2/etc. rows from every
  dataset and split column flow into the pretraining loader. Use
  `include_datasets` / `exclude_datasets` to restrict it (e.g. drop a noisy
  cohort) without losing the rest of the manifest.
- **Finetuning** picks samples through `data.split_strategy`:
  - `mode: all` — train on every sample.
  - `mode: manifest` — honour the existing `split` column with
    `train_split` / `val_split` / `test_split`.
  - `mode: by_dataset` — pick `train_datasets: [ispy2, duke]` and treat the
    rest (or whatever is listed under `test_datasets`) as held-out inference.
  - `mode: by_ratio` — deterministic hash split with `train_ratio: 0.7`. The
    remaining 30% becomes the inference / test subset.
- **Inference** reads the same `split_strategy` block so finetune and infer
  configs stay in sync. The inferencer takes the `test` subset (falling back
  to `val`/`train` if it is empty).

The same overrides exist on the CLI:

```bash
# Train on ispy2 + duke, infer the rest
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation \
  --mode finetune \
  --split-mode by_dataset \
  --train-datasets ispy2,duke

# 70/30 random split
PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation \
  --mode finetune \
  --split-mode by_ratio \
  --train-ratio 0.7
```

## MVP Notes

This is the first runnable implementation. Advanced items such as full voxel-level learned PINN fitting, time-dependent survival AUC, richer Grad-CAM overlays and cross-modality reconstruction are exposed as module boundaries and can be expanded without changing the dataset or forward contract.
