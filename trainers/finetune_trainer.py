from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import MultimodalManifestDataset, variable_modalities_collate
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.losses import MaskedMultitaskLoss
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint, save_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import default_device, move_batch_to_device
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.logger import build_logger


class FinetuneTrainer:
    """Masked multi-task finetuning loop."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = default_device()
        self.output_dir = Path(config.get("output", {}).get("root", "outputs/dce_moe_pinn_finetune"))
        self.logger = build_logger(self.output_dir / "logs", "finetune")
        self.model = BreastDCEMoEPINNModel(**config.get("model", {})).to(self.device)
        train_cfg = config.get("training", {})
        if train_cfg.get("pretrained_checkpoint"):
            load_checkpoint(train_cfg["pretrained_checkpoint"], self.model, strict=False)
        self.criterion = MaskedMultitaskLoss(config.get("loss", {}).get("task_weights"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 5e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(train_cfg.get("epochs", 50)))
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def build_loader(self, split: str | None, shuffle: bool) -> DataLoader:
        data_cfg = self.config["data"]
        dataset = MultimodalManifestDataset(
            manifest_path=data_cfg["manifest_path"],
            dataset_root=data_cfg.get("dataset_root"),
            split=split,
            label_columns=data_cfg.get("label_columns"),
            target_shape=tuple(data_cfg.get("target_shape", self.config.get("model", {}).get("image_size", (64, 128, 128)))),
            modalities=data_cfg.get("modalities"),
            allow_empty_labels=False,
            max_volumes=data_cfg.get("max_volumes"),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("training", {}).get("batch_size", 1)),
            shuffle=shuffle,
            num_workers=int(self.config.get("training", {}).get("num_workers", 2)),
            collate_fn=variable_modalities_collate,
            pin_memory=self.device.type == "cuda",
        )

    def run(self) -> None:
        data_cfg = self.config["data"]
        train_loader = self.build_loader(data_cfg.get("train_split", "train"), shuffle=True)
        val_loader = self.build_loader(data_cfg.get("val_split", "val"), shuffle=False) if data_cfg.get("val_split") else None
        best = -1.0
        epochs = int(self.config.get("training", {}).get("epochs", 50))
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, epoch) if val_loader is not None else {}
            score = -val_metrics.get("loss", train_metrics["loss"])
            best = max(best, score)
            self.logger.info("epoch=%s train=%s val=%s", epoch, train_metrics, val_metrics)
            save_checkpoint(self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth", self.model, self.optimizer, self.scheduler, epoch, best, self.config)
            if score >= best:
                save_checkpoint(self.output_dir / "checkpoints" / "best.pth", self.model, self.optimizer, self.scheduler, epoch, best, self.config)

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"finetune {epoch}", ncols=110):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(batch, mode="finetune")
                supervised, supervised_losses = self.criterion(outputs["predictions"], batch["labels"], batch["label_mask"])
                aux_loss = 0.01 * outputs["losses"].get("moe_balance", supervised * 0.0)
                loss = supervised + aux_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().item())
            steps += 1
        self.scheduler.step()
        return {"loss": total / max(steps, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, epoch: int) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        total = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"val {epoch}", ncols=110):
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch, mode="finetune")
            loss, _ = self.criterion(outputs["predictions"], batch["labels"], batch["label_mask"])
            total += float(loss.detach().item())
            steps += 1
        return {"loss": total / max(steps, 1)}
