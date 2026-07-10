from __future__ import annotations

from pathlib import Path

import numpy as np

from src.breast_mri_ai.sota_baselines.framework.training.checkpoint import load_checkpoint


class Predictor:
    def __init__(self, model, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required for inference") from exc
        self.torch = torch
        self.model = model.to(device)
        self.device = device
        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.model, map_location=device)
        self.model.eval()

    def predict_array(self, image: np.ndarray) -> float:
        tensor = self.torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logit = self.model(tensor).view(-1)[0]
            return float(self.torch.sigmoid(logit).cpu())

