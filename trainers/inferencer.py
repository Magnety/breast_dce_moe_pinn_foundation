from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import (
    MultimodalManifestDataset,
    parse_split_config,
    split_dataset,
    variable_modalities_collate,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import default_device, move_batch_to_device


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
        for batch in self.build_loader():
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch, mode="infer")
            preds = outputs["predictions"]
            for idx, patient_id in enumerate(batch["patient_id"]):
                row: dict[str, Any] = {
                    "patient_id": patient_id,
                    "sample_id": batch["sample_id"][idx],
                    "dataset_id": batch["dataset_name"][idx],
                    "visit_timepoint": batch["visit_timepoint"][idx],
                }
                for task in ("pCR", "HER2", "ER", "PR", "HR"):
                    if task in preds:
                        row[f"{task}_probability"] = float(torch.sigmoid(preds[task][idx]).detach().cpu())
                if "molecular_subtype" in preds:
                    probs = torch.softmax(preds["molecular_subtype"][idx], dim=-1).detach().cpu().tolist()
                    for cls_idx, value in enumerate(probs):
                        row[f"molecular_subtype_prob_{cls_idx}"] = float(value)
                if "survival_risk" in preds:
                    row["survival_risk_score"] = float(preds["survival_risk"][idx].detach().cpu())
                rows.append(row)
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
