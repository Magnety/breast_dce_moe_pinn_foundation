from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import MultimodalManifestDataset, variable_modalities_collate
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint, save_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import default_device, move_batch_to_device
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.logger import build_logger


class PretrainTrainer:
    """MVP pretraining loop for DCE-MAE and PINN physical pretraining."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = default_device()
        self.output_dir = Path(config.get("output", {}).get("root", "outputs/dce_moe_pinn_pretrain"))
        self.logger = build_logger(self.output_dir / "logs", "pretrain")
        self.model = BreastDCEMoEPINNModel(**config.get("model", {})).to(self.device)
        train_cfg = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(train_cfg.get("epochs", 100)))
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        if train_cfg.get("resume_path"):
            load_checkpoint(train_cfg["resume_path"], self.model, self.optimizer, self.scheduler, strict=False)

    def build_loader(self) -> DataLoader:
        data_cfg = self.config["data"]
        dataset = MultimodalManifestDataset(
            manifest_path=data_cfg["manifest_path"],
            dataset_root=data_cfg.get("dataset_root"),
            split=data_cfg.get("train_split"),
            target_shape=tuple(data_cfg.get("target_shape", self.config.get("model", {}).get("image_size", (64, 128, 128)))),
            modalities=data_cfg.get("modalities"),
            allow_empty_labels=True,
            max_volumes=data_cfg.get("max_volumes"),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("training", {}).get("batch_size", 1)),
            shuffle=True,
            num_workers=int(self.config.get("training", {}).get("num_workers", 2)),
            collate_fn=variable_modalities_collate,
            pin_memory=self.device.type == "cuda",
        )

    def run(self) -> None:
        loader = self.build_loader()
        epochs = int(self.config.get("training", {}).get("epochs", 100))
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(loader, epoch)
            self.logger.info("epoch=%s %s", epoch, metrics)
            save_checkpoint(self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth", self.model, self.optimizer, self.scheduler, epoch, metrics["loss"], self.config)

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"pretrain {epoch}", ncols=110):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(batch, mode="pretrain")
                loss = outputs["losses"]["total"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().item())
            steps += 1
        self.scheduler.step()
        return {"loss": total / max(steps, 1), "lr": self.optimizer.param_groups[0]["lr"]}
