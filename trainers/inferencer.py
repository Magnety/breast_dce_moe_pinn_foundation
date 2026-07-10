from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import (
    MultimodalManifestDataset,
    parse_split_config,
    split_dataset,
    variable_modalities_collate,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.collate_variable_modalities import MODALITY_VOCAB
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import default_device, move_batch_to_device
from src.breast_mri_ai.experiment_tracking import (
    export_advanced_visualization_artifacts,
    export_binary_classification_analysis,
    export_prediction_artifacts,
    logits_to_probability_array,
)
from src.breast_mri_ai.sota_baselines.framework.explainability.grad_cam import GradCAM3D


class Inferencer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = default_device()
        self.model = BreastDCEMoEPINNModel(**config.get("model", {})).to(self.device)
        checkpoint = config.get("inference", {}).get("checkpoint") or config.get("training", {}).get("checkpoint")
        if checkpoint:
            load_checkpoint(checkpoint, self.model, strict=False)

    def build_loader(self) -> DataLoader:
        data_cfg = self.config["data"]
        loader_kwargs = self._loader_kwargs()
        dataset_kwargs = self._dataset_kwargs(data_cfg)
        split_payload = data_cfg.get("split_strategy")
        if split_payload:
            split_cfg = parse_split_config(split_payload, default_mode="manifest")
            dataset = MultimodalManifestDataset(
                split=None,
                **dataset_kwargs,
            )
            subsets = split_dataset(dataset, split_cfg)
            target = subsets.get("test") or subsets.get("val") or subsets.get("train")
            return DataLoader(target, **loader_kwargs)

        dataset = MultimodalManifestDataset(
            split=data_cfg.get("split"),
            **dataset_kwargs,
        )
        return DataLoader(dataset, **loader_kwargs)

    def _dataset_kwargs(self, data_cfg: dict[str, Any]) -> dict[str, Any]:
        cache_cfg = data_cfg.get("cache", {}) or {}
        return {
            "manifest_path": data_cfg["manifest_path"],
            "dataset_root": data_cfg.get("dataset_root"),
            "label_columns": data_cfg.get("label_columns"),
            "target_shape": tuple(
                data_cfg.get("target_shape", self.config.get("model", {}).get("image_size", (64, 128, 128)))
            ),
            "modalities": data_cfg.get("modalities"),
            "normalize": bool(data_cfg.get("normalize", True)),
            "allow_empty_labels": True,
            "max_volumes": data_cfg.get("max_volumes"),
            "include_datasets": data_cfg.get("include_datasets"),
            "exclude_datasets": data_cfg.get("exclude_datasets"),
            "cache_processed": bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False))),
            "cache_dir": data_cfg.get("cache_dir", cache_cfg.get("dir")),
            "cache_after_normalize": bool(cache_cfg.get("after_normalize", True)),
            "npy_mmap_mode": cache_cfg.get("npy_mmap_mode", data_cfg.get("npy_mmap_mode")),
        }

    def _loader_kwargs(self) -> dict[str, Any]:
        inference_cfg = self.config.get("inference", {}) or {}
        training_cfg = self.config.get("training", {}) or {}
        num_workers = int(inference_cfg.get("num_workers", training_cfg.get("num_workers", 0)))
        loader_kwargs: dict[str, Any] = {
            "batch_size": int(inference_cfg.get("batch_size", training_cfg.get("batch_size", 1))),
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": variable_modalities_collate,
            "pin_memory": bool(inference_cfg.get("pin_memory", training_cfg.get("pin_memory", self.device.type == "cuda")))
            and self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                inference_cfg.get("persistent_workers", training_cfg.get("persistent_workers", True))
            )
            prefetch_factor = inference_cfg.get("prefetch_factor", training_cfg.get("prefetch_factor"))
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            multiprocessing_context = inference_cfg.get(
                "multiprocessing_context", training_cfg.get("multiprocessing_context")
            )
            if multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = mp.get_context(str(multiprocessing_context))
        return loader_kwargs

    @torch.no_grad()
    def run(self, output_csv: str | Path) -> None:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        feature_blocks: list[np.ndarray] = []
        contribution_blocks: list[np.ndarray] = []
        gradcam_examples: list[dict[str, Any]] = []
        binary_tasks = self._binary_prediction_tasks()
        tsne_task = self._pick_analysis_task("tsne_task")
        shap_task = self._pick_analysis_task("shap_task")
        shap_feature_names = self._analysis_modalities()
        threshold = self._evaluation_threshold()
        for batch in self.build_loader():
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch, mode="infer")
            preds = outputs["predictions"]
            features = outputs.get("features", {})
            aux = outputs.get("aux", {})
            foundation_feature = features.get("foundation_feature")
            if torch.is_tensor(foundation_feature):
                feature_blocks.append(foundation_feature.detach().cpu().numpy().astype(np.float32, copy=False))
            if shap_task and shap_feature_names and shap_task in preds:
                contribution_blocks.append(
                    self._compute_modality_contributions(
                        batch=batch,
                        base_outputs=outputs,
                        task=shap_task,
                        modality_names=shap_feature_names,
                    )
                )
            if (
                binary_tasks
                and len(gradcam_examples) < self._max_gradcam_samples()
            ):
                gradcam_example = self._collect_gradcam_example(
                    batch=batch,
                    task=binary_tasks[0],
                )
                if gradcam_example is not None:
                    gradcam_examples.append(gradcam_example)
            label_map = self._label_maps(batch)
            for idx, patient_id in enumerate(batch["patient_id"]):
                row: dict[str, Any] = {
                    "patient_id": patient_id,
                    "sample_id": batch["sample_id"][idx],
                    "dataset_id": batch["dataset_name"][idx],
                    "visit_timepoint": batch["visit_timepoint"][idx],
                    "num_available_volumes": int(batch["modality_available_mask"][idx].sum().item()),
                    "num_dce_phases": int(batch["temporal_dce_mask"][idx].sum().item()),
                    "relative_time_span": self._relative_time_span(batch, idx),
                }
                for task, values in label_map.items():
                    row[f"{task}_label"] = values[idx]
                if "survival_time" in label_map:
                    row["survival_time"] = label_map["survival_time"][idx]
                if "survival_event" in label_map:
                    row["survival_event"] = label_map["survival_event"][idx]
                for task in ("pCR", "HER2", "ER", "PR", "HR"):
                    if task in preds:
                        row[f"{task}_probability"] = float(torch.sigmoid(preds[task][idx]).detach().cpu())
                if "molecular_subtype" in preds:
                    probs = torch.softmax(preds["molecular_subtype"][idx], dim=-1).detach().cpu().tolist()
                    for cls_idx, value in enumerate(probs):
                        row[f"molecular_subtype_prob_{cls_idx}"] = float(value)
                if "survival_risk" in preds:
                    row["survival_risk_score"] = float(preds["survival_risk"][idx].detach().cpu())
                temporal_attention = aux.get("temporal_attention")
                if torch.is_tensor(temporal_attention):
                    for phase_index, value in enumerate(temporal_attention[idx].detach().cpu().tolist()):
                        row[f"temporal_attention_{phase_index}"] = float(value)
                hemodynamic = aux.get("hemodynamic", {})
                if isinstance(hemodynamic, dict):
                    for key, value in hemodynamic.items():
                        if torch.is_tensor(value) and value.shape[0] > idx:
                            row[f"{key}_summary"] = float(value[idx].detach().cpu())
                expert_weights = aux.get("expert_weights", {})
                if isinstance(expert_weights, dict):
                    for task, weight_tensor in expert_weights.items():
                        if not torch.is_tensor(weight_tensor) or weight_tensor.shape[0] <= idx:
                            continue
                        for expert_index, value in enumerate(weight_tensor[idx].detach().cpu().tolist()):
                            row[f"{task}_expert_{expert_index}"] = float(value)
                task_context_weights = aux.get("task_context_weights", {})
                if isinstance(task_context_weights, dict):
                    pcr_context = task_context_weights.get("pCR")
                    if torch.is_tensor(pcr_context) and pcr_context.shape[0] > idx:
                        aux_tasks = [task for task in ("HER2", "ER", "PR", "HR", "molecular_subtype", "survival")]
                        for task_name, value in zip(aux_tasks, pcr_context[idx].detach().cpu().tolist()):
                            row[f"pCR_context_{task_name}"] = float(value)
                rows.append(row)
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        feature_matrix = np.concatenate(feature_blocks, axis=0) if feature_blocks else None
        contribution_matrix = np.concatenate(contribution_blocks, axis=0) if contribution_blocks else None
        root_dir = self.config.get("inference", {}).get("output_dir") or path.parent
        export_prediction_artifacts(
            root_dir=root_dir,
            family="foundation",
            mode="infer",
            model_name="breast_dce_moe_pinn",
            prediction_rows=rows,
            binary_tasks=binary_tasks,
            threshold=threshold,
            top_level_prediction_csv=path,
            final_features=feature_matrix,
            final_feature_names=self._feature_names(feature_matrix),
            tsne_task=tsne_task,
            shap_values=contribution_matrix,
            shap_feature_names=shap_feature_names if contribution_matrix is not None else None,
            shap_task=shap_task,
            extra_summary={
                "checkpoint": self.config.get("inference", {}).get("checkpoint")
                or self.config.get("training", {}).get("checkpoint"),
                "threshold": threshold,
            },
        )
        export_binary_classification_analysis(
            root_dir=root_dir,
            prediction_rows=rows,
            binary_tasks=binary_tasks,
            threshold=threshold,
            subgroup_columns=self._subgroup_columns(),
            figure_formats=self._figure_formats(),
            figure_dpi=self._figure_dpi(),
            stage="infer",
            epoch=0,
        )
        export_advanced_visualization_artifacts(
            root_dir=root_dir,
            family="foundation",
            prediction_rows=rows,
            binary_tasks=binary_tasks,
            threshold=threshold,
            final_features=feature_matrix,
            embedding_task=tsne_task,
            gradcam_examples=gradcam_examples,
            bootstrap_samples=self._bootstrap_samples(),
        )

    def _binary_prediction_tasks(self) -> list[str]:
        return [task for task in ("pCR", "HER2", "ER", "PR", "HR") if task in self.config.get("data", {}).get("label_columns", {})]

    def _pick_analysis_task(self, key: str) -> str | None:
        analysis_cfg = self.config.get("analysis", {}) or {}
        inference_cfg = self.config.get("inference", {}) or {}
        preferred = analysis_cfg.get(key) or inference_cfg.get(key)
        tasks = self._binary_prediction_tasks()
        if preferred in tasks:
            return str(preferred)
        return "pCR" if "pCR" in tasks else (tasks[0] if tasks else None)

    def _analysis_modalities(self) -> list[str]:
        data_cfg = self.config.get("data", {}) or {}
        configured = [str(item) for item in data_cfg.get("modalities", []) if str(item).strip()]
        if configured:
            return [item for item in configured if item != "unknown"]
        return [name for name in MODALITY_VOCAB if name != "unknown"]

    def _feature_names(self, matrix: np.ndarray | None) -> list[str] | None:
        if matrix is None:
            return None
        return [f"foundation_feature_{index:04d}" for index in range(matrix.shape[1])]

    def _evaluation_threshold(self) -> float:
        evaluation_cfg = self.config.get("evaluation", {}) or {}
        inference_cfg = self.config.get("inference", {}) or {}
        analysis_cfg = self.config.get("analysis", {}) or {}
        threshold = (
            inference_cfg.get("threshold")
            or analysis_cfg.get("threshold")
            or evaluation_cfg.get("threshold")
            or 0.5
        )
        return float(threshold)

    def _figure_formats(self) -> list[str] | None:
        analysis_cfg = self.config.get("analysis", {}) or {}
        inference_cfg = self.config.get("inference", {}) or {}
        formats = analysis_cfg.get("figure_formats") or inference_cfg.get("figure_formats")
        if formats is None:
            return None
        if isinstance(formats, str):
            return [item.strip() for item in formats.split(",") if item.strip()]
        if isinstance(formats, (list, tuple)):
            return [str(item) for item in formats if str(item).strip()]
        return None

    def _figure_dpi(self) -> int:
        analysis_cfg = self.config.get("analysis", {}) or {}
        inference_cfg = self.config.get("inference", {}) or {}
        return int(analysis_cfg.get("figure_dpi", inference_cfg.get("figure_dpi", 300)))

    def _subgroup_columns(self) -> list[str]:
        analysis_cfg = self.config.get("analysis", {}) or {}
        columns = analysis_cfg.get("subgroup_columns", ["dataset_id", "visit_timepoint"])
        if isinstance(columns, str):
            return [item.strip() for item in columns.split(",") if item.strip()]
        if isinstance(columns, (list, tuple)):
            return [str(item) for item in columns if str(item).strip()]
        return ["dataset_id", "visit_timepoint"]

    def _bootstrap_samples(self) -> int:
        analysis_cfg = self.config.get("analysis", {}) or {}
        return int(analysis_cfg.get("bootstrap_samples", 160))

    def _max_gradcam_samples(self) -> int:
        analysis_cfg = self.config.get("analysis", {}) or {}
        return int(analysis_cfg.get("max_gradcam_samples", 4))

    def _label_maps(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        tasks = list(batch.get("label_tasks", []))
        label_values = batch.get("label_values")
        label_mask_values = batch.get("label_mask_values")
        if not torch.is_tensor(label_values) or not torch.is_tensor(label_mask_values):
            return {}
        output: dict[str, list[Any]] = {}
        for column, task in enumerate(tasks):
            values: list[Any] = []
            for row_index in range(label_values.shape[0]):
                valid = bool(label_mask_values[row_index, column].item() > 0.5)
                if not valid:
                    values.append("")
                    continue
                value = float(label_values[row_index, column].item())
                values.append(int(value) if float(value).is_integer() else value)
            output[str(task)] = values
        return output

    def _compute_modality_contributions(
        self,
        *,
        batch: dict[str, Any],
        base_outputs: dict[str, Any],
        task: str,
        modality_names: list[str],
    ) -> np.ndarray:
        base_prob = logits_to_probability_array(base_outputs["predictions"][task])
        contributions = np.zeros((base_prob.shape[0], len(modality_names)), dtype=np.float32)
        base_available = batch["modality_available_mask"].bool()
        modality_ids = batch["modality_id"]
        for feature_index, modality_name in enumerate(modality_names):
            modality_code = MODALITY_VOCAB.get(modality_name)
            if modality_code is None:
                continue
            ablation_mask = base_available & (modality_ids == modality_code)
            if not bool(ablation_mask.any().item()):
                continue
            ablated_batch = dict(batch)
            ablated_batch["volumes"] = batch["volumes"].clone()
            ablated_batch["modality_available_mask"] = batch["modality_available_mask"].clone()
            ablated_batch["phase_available_mask"] = batch["phase_available_mask"].clone()
            ablated_batch["temporal_dce_mask"] = batch["temporal_dce_mask"].clone()
            ablated_batch["volumes"][ablation_mask] = 0
            ablated_batch["modality_available_mask"][ablation_mask] = False
            ablated_batch["phase_available_mask"][ablation_mask] = False
            ablated_batch["temporal_dce_mask"][ablation_mask] = False
            ablated_outputs = self.model(ablated_batch, mode="infer")
            ablated_prob = logits_to_probability_array(ablated_outputs["predictions"][task])
            contributions[:, feature_index] = base_prob - ablated_prob
        return contributions

    def _relative_time_span(self, batch: dict[str, Any], index: int) -> float:
        relative_time = batch.get("relative_time")
        dce_mask = batch.get("temporal_dce_mask")
        if not torch.is_tensor(relative_time) or not torch.is_tensor(dce_mask):
            return 0.0
        values = relative_time[index][dce_mask[index].bool()]
        if values.numel() <= 1:
            return 0.0
        return float((values.max() - values.min()).detach().cpu())

    def _collect_gradcam_example(
        self,
        *,
        batch: dict[str, Any],
        task: str,
    ) -> dict[str, Any] | None:
        if not hasattr(self.model, "patch_embed") or not hasattr(self.model.patch_embed, "proj"):
            return None
        single_batch = self._slice_batch(batch, 0)
        try:
            with torch.enable_grad():
                gradcam = GradCAM3D(self.model, self.model.patch_embed.proj)
                cam = gradcam(single_batch, task_name=task, mode="infer")
        except Exception:  # noqa: BLE001
            return None
        finally:
            if "gradcam" in locals():
                gradcam.remove_hooks()
        if not torch.is_tensor(cam) or cam.numel() == 0:
            return None
        volumes = single_batch.get("volumes")
        available = single_batch.get("modality_available_mask")
        temporal_mask = single_batch.get("temporal_dce_mask")
        if not torch.is_tensor(volumes):
            return None
        sequence_index = self._preferred_volume_index(single_batch)
        if cam.shape[0] == volumes.shape[1]:
            cam_index = sequence_index
        else:
            if not torch.is_tensor(available):
                return None
            valid_positions = torch.nonzero(available[0].bool(), as_tuple=False).flatten()
            matches = torch.nonzero(valid_positions == sequence_index, as_tuple=False).flatten()
            if matches.numel() == 0:
                return None
            cam_index = int(matches[0].item())
        cam_volume = cam[cam_index : cam_index + 1]
        image_volume = volumes[0, sequence_index : sequence_index + 1, :1].reshape(1, 1, *volumes.shape[-3:])
        cam_up = torch.nn.functional.interpolate(
            cam_volume,
            size=tuple(image_volume.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        )[0, 0].detach().cpu().numpy()
        image_np = image_volume[0, 0].detach().cpu().numpy()
        depth_index = image_np.shape[0] // 2
        phase_label = ""
        if torch.is_tensor(temporal_mask) and temporal_mask.shape[1] > sequence_index:
            phase_label = "DCE" if bool(temporal_mask[0, sequence_index].item()) else "MRI"
        return {
            "image": image_np[depth_index],
            "attention": cam_up[depth_index],
            "title": f"{task} | {phase_label} | {single_batch['sample_id'][0]}",
            "patient_id": single_batch["patient_id"][0],
            "sample_id": single_batch["sample_id"][0],
        }

    def _preferred_volume_index(self, batch: dict[str, Any]) -> int:
        available = batch.get("modality_available_mask")
        temporal_mask = batch.get("temporal_dce_mask")
        if torch.is_tensor(available) and torch.is_tensor(temporal_mask):
            preferred = torch.nonzero(available[0].bool() & temporal_mask[0].bool(), as_tuple=False).flatten()
            if preferred.numel() > 0:
                return int(preferred[0].item())
        if torch.is_tensor(available):
            fallback = torch.nonzero(available[0].bool(), as_tuple=False).flatten()
            if fallback.numel() > 0:
                return int(fallback[0].item())
        return 0

    def _slice_batch(self, value: Any, index: int) -> Any:
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value
            return value[index : index + 1].clone()
        if isinstance(value, dict):
            return {key: self._slice_batch(item, index) for key, item in value.items()}
        if isinstance(value, list):
            return [value[index]] if index < len(value) else []
        return value
