from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class InsufficientDCEPrerequisites(RuntimeError):
    """Raised when pharmacokinetic parameters would be unsupported by metadata."""


@dataclass(frozen=True)
class DCEPrerequisiteReport:
    ok: bool
    missing: tuple[str, ...]
    reliability: str


def semi_quantitative_features(
    signal: np.ndarray,
    times: np.ndarray,
    baseline_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, float]:
    """Compute semi-quantitative DCE time-signal features for one curve.

    Parameters are derived only from supplied signal and time arrays. No pharmacokinetic
    assumptions are made here.
    """

    signal = np.asarray(signal, dtype=float)
    times = np.asarray(times, dtype=float)
    if signal.ndim != 1 or times.ndim != 1 or signal.shape[0] != times.shape[0]:
        raise ValueError("signal and times must be one-dimensional arrays with matching length")
    if signal.shape[0] < 2:
        raise ValueError("at least two DCE time points are required")
    order = np.argsort(times)
    signal = signal[order]
    times = times[order]
    baseline_indices = tuple(baseline_indices or [0])
    baseline = float(np.mean(signal[list(baseline_indices)]))
    eps = 1e-8
    relative = (signal - baseline) / (abs(baseline) + eps)
    peak_index = int(np.argmax(relative))
    peak_enhancement = float(relative[peak_index])
    time_to_peak = float(times[peak_index] - times[0])
    early_index = 1 if len(times) > 1 else 0
    early_enhancement_rate = float(relative[early_index])
    wash_in_slope = float((signal[peak_index] - baseline) / max(times[peak_index] - times[0], eps))
    delayed_enhancement = float(relative[-1])
    wash_out_slope = float(
        (signal[-1] - signal[peak_index]) / max(times[-1] - times[peak_index], eps)
    )
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(integrator(relative, times))
    return {
        "baseline_signal": baseline,
        "early_enhancement_rate": early_enhancement_rate,
        "peak_enhancement": peak_enhancement,
        "time_to_peak": time_to_peak,
        "wash_in_slope": wash_in_slope,
        "wash_out_slope": wash_out_slope,
        "delayed_enhancement": delayed_enhancement,
        "relative_auc": auc,
    }


def check_pharmacokinetic_prerequisites(metadata: dict[str, Any]) -> DCEPrerequisiteReport:
    required = (
        "injection_time",
        "flip_angle",
        "repetition_time",
        "baseline_t1",
        "arterial_input_function",
        "temporal_resolution_seconds",
    )
    missing = tuple(key for key in required if metadata.get(key) in (None, "", "unknown"))
    if missing:
        return DCEPrerequisiteReport(ok=False, missing=missing, reliability="not_supported")
    temporal_resolution = float(metadata["temporal_resolution_seconds"])
    reliability = "high" if temporal_resolution <= 20 else "limited_temporal_resolution"
    return DCEPrerequisiteReport(ok=True, missing=(), reliability=reliability)


def require_pharmacokinetic_prerequisites(metadata: dict[str, Any]) -> None:
    report = check_pharmacokinetic_prerequisites(metadata)
    if not report.ok:
        missing = ", ".join(report.missing)
        raise InsufficientDCEPrerequisites(
            f"Cannot compute pharmacokinetic DCE parameters; missing prerequisites: {missing}"
        )
