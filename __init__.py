"""Breast-DCE-MoE-PINN foundation model package.

Heavy PyTorch modules are imported lazily by the training/inference entry
points so config inspection still works in non-torch environments.
"""

__all__ = ["BreastDCEMoEPINNModel"]


def __getattr__(name: str):
    if name == "BreastDCEMoEPINNModel":
        from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models.breast_dce_moe_pinn_model import (
            BreastDCEMoEPINNModel,
        )

        return BreastDCEMoEPINNModel
    raise AttributeError(name)
