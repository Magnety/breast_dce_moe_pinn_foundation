from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PINNHemodynamicModule(nn.Module):
    """Simplified extended-Tofts PINN module.

    Training mode operates on patch/global DCE enhancement curves for speed.
    ``voxel_level_export`` computes high-resolution relative maps directly from
    a DCE sequence for inference-time export.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, aif_mode: str = "population", eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.aif_mode = aif_mode
        self.param_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        self.feature_proj = nn.Linear(8, embed_dim)

    def forward(
        self,
        dce_feature: torch.Tensor,
        dce_curve: torch.Tensor,
        relative_time: torch.Tensor,
        dce_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.param_head(dce_feature)
        ktrans = F.softplus(raw[:, 0])
        ve = torch.sigmoid(raw[:, 1]).clamp_min(self.eps)
        vp = 0.2 * torch.sigmoid(raw[:, 2])
        kep = ktrans / (ve + self.eps)
        bat = F.softplus(raw[:, 3])
        cp = self.population_aif(relative_time, bat.unsqueeze(1))
        pred = self.extended_tofts(cp, relative_time, ktrans, ve, vp, kep)
        valid = dce_mask.float()
        signal_loss = (((pred - dce_curve) * valid) ** 2).sum() / valid.sum().clamp_min(1.0)
        ode_loss = self._ode_loss(pred, cp, relative_time, ktrans, ve, dce_mask)
        param_range = (F.relu(ve - 1.0).mean() + F.relu(vp - 0.2).mean())
        has_dce = dce_mask.any(dim=1)
        peak = dce_curve.masked_fill(~dce_mask, -1e4).max(dim=1).values
        peak = torch.where(has_dce, peak, peak.new_zeros(peak.shape))
        maps = {
            "Ktrans": ktrans,
            "ve": ve,
            "vp": vp,
            "kep": kep,
            "BAT": bat,
            "IAUC": torch.trapz(torch.clamp(dce_curve, min=0.0) * valid, relative_time.clamp_min(0.0), dim=1),
            "washin_slope": _slope_stat(dce_curve, valid, first=True),
            "washout_slope": _slope_stat(dce_curve, valid, first=False),
            "peak_enhancement": peak,
        }
        feature = self.feature_proj(torch.stack([maps[k] for k in ("Ktrans", "ve", "vp", "kep", "BAT", "IAUC", "washin_slope", "peak_enhancement")], dim=-1))
        return {
            "pred_curve": pred,
            "parameter_maps": maps,
            "feature": feature,
            "losses": {
                "pinn_signal": signal_loss,
                "pinn_ode": ode_loss,
                "pinn_param_range": param_range,
                "pinn_aif": cp[:, 1:].sub(cp[:, :-1]).abs().mean() if cp.shape[1] > 1 else cp.mean() * 0.0,
            },
        }

    def population_aif(self, t: torch.Tensor, bat: torch.Tensor) -> torch.Tensor:
        shifted = (t - bat).clamp_min(0.0)
        return 5.0 * shifted * torch.exp(-shifted / 1.5)

    def extended_tofts(
        self,
        cp: torch.Tensor,
        t: torch.Tensor,
        ktrans: torch.Tensor,
        ve: torch.Tensor,
        vp: torch.Tensor,
        kep: torch.Tensor,
    ) -> torch.Tensor:
        b, steps = cp.shape
        pred = []
        for idx in range(steps):
            ti = t[:, idx : idx + 1]
            tau = t[:, : idx + 1]
            dt = torch.diff(torch.cat([torch.zeros_like(tau[:, :1]), tau], dim=1), dim=1).clamp_min(0.0)
            kernel = torch.exp(-kep.unsqueeze(1) * (ti - tau).clamp_min(0.0))
            integral = (cp[:, : idx + 1] * kernel * dt).sum(dim=1)
            pred.append(vp * cp[:, idx] + ktrans * integral)
        return torch.stack(pred, dim=1)

    def _ode_loss(self, ct: torch.Tensor, cp: torch.Tensor, t: torch.Tensor, ktrans: torch.Tensor, ve: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ct.shape[1] < 2:
            return ct.mean() * 0.0
        dt = (t[:, 1:] - t[:, :-1]).abs().clamp_min(self.eps)
        dct = (ct[:, 1:] - ct[:, :-1]) / dt
        residual = dct - (ktrans / (ve + self.eps)).unsqueeze(1) * (cp[:, :-1] - ct[:, :-1])
        valid = (mask[:, 1:] & mask[:, :-1]).float()
        return ((residual * valid) ** 2).sum() / valid.sum().clamp_min(1.0)

    @torch.no_grad()
    def voxel_level_export(self, dce_sequence: torch.Tensor, relative_time: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Export approximate voxel-level hemodynamic maps from ``[T, D, H, W]``."""

        if dce_sequence.ndim != 4:
            raise ValueError(f"Expected [T, D, H, W] DCE tensor, got {tuple(dce_sequence.shape)}")
        baseline = dce_sequence[:1].mean(dim=0)
        enhancement = (dce_sequence - baseline) / (baseline.abs() + self.eps)
        peak = enhancement.max(dim=0).values
        iauc = enhancement.clamp_min(0).sum(dim=0)
        washin = enhancement[min(1, enhancement.shape[0] - 1)] - enhancement[0]
        washout = enhancement[-1] - peak
        ktrans = F.softplus(washin)
        ve = torch.sigmoid(peak)
        vp = 0.2 * torch.sigmoid(iauc / max(enhancement.shape[0], 1))
        kep = ktrans / (ve + self.eps)
        return {
            "Ktrans": ktrans,
            "ve": ve,
            "vp": vp,
            "kep": kep,
            "IAUC": iauc,
            "washin_slope": washin,
            "washout_slope": washout,
            "peak_enhancement": peak,
        }


def _slope_stat(curve: torch.Tensor, valid: torch.Tensor, first: bool) -> torch.Tensor:
    if curve.shape[1] < 2:
        return curve.mean(dim=1) * 0.0
    diff = curve[:, 1:] - curve[:, :-1]
    mask = (valid[:, 1:] * valid[:, :-1]).bool()
    diff = diff.masked_fill(~mask, 0.0)
    return diff[:, : max(1, diff.shape[1] // 2)].mean(dim=1) if first else diff[:, max(0, diff.shape[1] // 2) :].mean(dim=1)
