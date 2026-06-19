from __future__ import annotations

import torch


class TaskGradCAM:
    """Minimal task-specific Grad-CAM hook for future visualization expansion."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
