from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_heatmap_png(image: np.ndarray, heatmap: np.ndarray, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid = image.shape[0] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(image[mid], cmap="gray")
    plt.imshow(heatmap[mid], cmap="magma", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_pretrain_visualizations(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    epoch: int,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    mae_path = output_dir / "mae_reconstruction.png"
    _save_mae_panel(payload, mae_path, plt=plt)
    saved["mae_reconstruction"] = str(mae_path)

    phase_path = output_dir / "phase_order.png"
    _save_phase_order_panel(payload, phase_path, plt=plt)
    saved["phase_order"] = str(phase_path)

    pinn_path = output_dir / "pinn_curve.png"
    _save_pinn_panel(payload, pinn_path, plt=plt)
    saved["pinn_curve"] = str(pinn_path)

    summary = {
        "epoch": int(epoch),
        "files": saved,
        "identifiers": payload.get("identifiers", {}),
        "volume_labels": payload.get("volume_labels", []),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    saved["summary"] = str(summary_path)
    return saved


def _save_mae_panel(payload: dict[str, Any], output_path: Path, *, plt) -> None:
    original = _to_numpy(payload.get("original"))
    masked_input = _to_numpy(payload.get("masked_input"))
    reconstruction = _to_numpy(payload.get("reconstruction"))
    absolute_error = _to_numpy(payload.get("absolute_error"))
    mask = _to_numpy(payload.get("mask"))
    labels = payload.get("volume_labels", [])

    if original is None or masked_input is None or reconstruction is None or absolute_error is None or mask is None:
        _save_text_panel("No MAE visualization payload was available.", output_path, plt=plt)
        return

    rows = max(1, int(original.shape[0]))
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows), squeeze=False)
    titles = ("Original", "Masked Input", "Reconstruction", "Absolute Error", "Mask")
    for col, title in enumerate(titles):
        axes[0][col].set_title(title)
    for row in range(rows):
        row_label = labels[row]["label"] if row < len(labels) and isinstance(labels[row], dict) else f"volume {row}"
        panels = (
            _mid_slice(original[row]),
            _mid_slice(masked_input[row]),
            _mid_slice(reconstruction[row]),
            _mid_slice(absolute_error[row]),
            _mid_slice(mask[row]),
        )
        cmaps = ("gray", "gray", "gray", "magma", "gray")
        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            axis = axes[row][col]
            image = _normalize_image(panel) if col != 3 else panel
            axis.imshow(image, cmap=cmap)
            axis.axis("off")
            if col == 0:
                axis.set_ylabel(row_label, fontsize=9)
    _apply_figure_header(fig, payload, "MAE reconstruction preview")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_phase_order_panel(payload: dict[str, Any], output_path: Path, *, plt) -> None:
    phase = payload.get("phase_order") or {}
    dce_volumes = _to_numpy(phase.get("dce_volumes"))
    target = _to_numpy(phase.get("target_phase_id"))
    predicted = _to_numpy(phase.get("predicted_phase_id"))
    confidence = _to_numpy(phase.get("confidence"))
    attention = _to_numpy(phase.get("temporal_attention"))
    relative_time = _to_numpy(phase.get("relative_time"))
    dce_mask = _to_numpy(phase.get("dce_mask"))

    if (
        dce_volumes is None
        or target is None
        or predicted is None
        or attention is None
        or relative_time is None
        or dce_mask is None
    ):
        _save_text_panel("No phase-order visualization payload was available.", output_path, plt=plt)
        return

    valid = dce_mask[0].astype(bool)
    if not np.any(valid):
        _save_text_panel("The preview sample has no valid DCE phases.", output_path, plt=plt)
        return

    volumes = dce_volumes[0][valid]
    target_y = target[0][valid].astype(int)
    pred_y = predicted[0][valid].astype(int)
    x = relative_time[0][valid]
    attn_y = attention[0][valid]
    conf_y = confidence[0][valid] if confidence is not None else np.ones_like(target_y, dtype=np.float32)

    shuffle_order = np.random.permutation(len(volumes))
    shuffled_target = target_y[shuffle_order]
    shuffled_pred = pred_y[shuffle_order]
    shuffled_x = x[shuffle_order]
    shuffled_attn = attn_y[shuffle_order]
    shuffled_conf = conf_y[shuffle_order]
    restored_order = np.lexsort((shuffled_x, shuffled_pred))

    cols = max(1, len(volumes))
    fig, axes = plt.subplots(2, cols, figsize=(max(3.0 * cols, 8.0), 6.4), squeeze=False)
    for row_axes in axes:
        for axis in row_axes:
            axis.axis("off")

    for col, index in enumerate(shuffle_order):
        axis = axes[0][col]
        axis.imshow(_normalize_image(_mid_slice(volumes[index])), cmap="gray")
        axis.set_title(
            _phase_tile_title(
                phase_id=int(target_y[index]),
                predicted=int(pred_y[index]),
                relative_time=float(x[index]),
                attention=float(attn_y[index]),
                confidence=float(conf_y[index]),
            ),
            fontsize=8,
        )
        axis.axis("off")

    for col, shuffled_index in enumerate(restored_order):
        axis = axes[1][col]
        axis.imshow(_normalize_image(_mid_slice(volumes[shuffle_order[shuffled_index]])), cmap="gray")
        axis.set_title(
            _phase_tile_title(
                phase_id=int(shuffled_target[shuffled_index]),
                predicted=int(shuffled_pred[shuffled_index]),
                relative_time=float(shuffled_x[shuffled_index]),
                attention=float(shuffled_attn[shuffled_index]),
                confidence=float(shuffled_conf[shuffled_index]),
            ),
            fontsize=8,
        )
        axis.axis("off")

    _apply_figure_header(
        fig,
        payload,
        "Temporal order preview: randomized DCE phases and order restored by predicted phase ids",
    )
    fig.text(0.015, 0.69, "Shuffled", rotation=90, va="center", ha="center", fontsize=11)
    fig.text(0.015, 0.27, "Restored", rotation=90, va="center", ha="center", fontsize=11)
    fig.text(
        0.5,
        0.015,
        (
            f"shuffle target phases={shuffled_target.tolist()} | "
            f"restored predicted phases={shuffled_pred[restored_order].tolist()}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.05, 1.0, 0.90))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pinn_panel(payload: dict[str, Any], output_path: Path, *, plt) -> None:
    pinn = payload.get("pinn") or {}
    relative_time = _to_numpy(pinn.get("relative_time"))
    phase_id = _to_numpy(pinn.get("phase_id"))
    target_curve = _to_numpy(pinn.get("target_curve"))
    pred_curve = _to_numpy(pinn.get("pred_curve"))
    aif_curve = _to_numpy(pinn.get("aif_curve"))
    dce_mask = _to_numpy(pinn.get("dce_mask"))
    parameters = pinn.get("parameters") or {}

    if relative_time is None or target_curve is None or pred_curve is None or dce_mask is None:
        _save_text_panel("No PINN visualization payload was available.", output_path, plt=plt)
        return

    valid = dce_mask[0].astype(bool)
    if not np.any(valid):
        _save_text_panel("The preview sample has no valid DCE curve for PINN plotting.", output_path, plt=plt)
        return

    x = relative_time[0][valid]
    target_y = target_curve[0][valid]
    pred_y = pred_curve[0][valid]
    order = np.argsort(x)
    x = x[order]
    target_y = target_y[order]
    pred_y = pred_y[order]
    residual = pred_y - target_y
    phase_values = phase_id[0][valid][order].astype(int) if phase_id is not None else None
    aif_y = aif_curve[0][valid][order] if aif_curve is not None else None
    param_names = []
    param_values = []
    preferred_order = ("Ktrans", "ve", "vp", "kep", "BAT", "IAUC", "washin_slope", "washout_slope", "peak_enhancement")
    for key in preferred_order:
        if key not in parameters:
            continue
        value_np = _to_numpy(parameters.get(key))
        if value_np is None or value_np.size == 0:
            continue
        param_names.append(str(key))
        param_values.append(float(np.ravel(value_np)[0]))
    for key, value in parameters.items():
        if key in preferred_order:
            continue
        value_np = _to_numpy(value)
        if value_np is None or value_np.size == 0:
            continue
        param_names.append(str(key))
        param_values.append(float(np.ravel(value_np)[0]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0][0].plot(x, target_y, marker="o", linewidth=2.2, label="target")
    axes[0][0].plot(x, pred_y, marker="s", linewidth=2.2, label="prediction")
    if phase_values is not None:
        for idx, (x_value, y_value, phase_value) in enumerate(zip(x, target_y, phase_values)):
            axes[0][0].annotate(f"p{phase_value}", (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[0][0].set_title("DCE enhancement curve")
    axes[0][0].set_xlabel("relative time")
    axes[0][0].set_ylabel("relative enhancement")
    axes[0][0].legend(loc="best")
    axes[0][0].grid(alpha=0.25)

    axes[0][1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[0][1].plot(x, residual, marker="d", linewidth=2.0, color="#b04a3a")
    axes[0][1].fill_between(x, 0.0, residual, color="#e8b8ab", alpha=0.45)
    axes[0][1].set_title(f"Residuals (RMSE={float(np.sqrt(np.mean(np.square(residual)))):.4f})")
    axes[0][1].set_xlabel("relative time")
    axes[0][1].set_ylabel("prediction - target")
    axes[0][1].grid(alpha=0.25)

    if aif_y is not None:
        axes[1][0].plot(x, aif_y, marker="^", linewidth=2.0, color="#2f6c8f")
        axes[1][0].fill_between(x, 0.0, aif_y, color="#b8d4e5", alpha=0.35)
        axes[1][0].set_title("Population AIF used by PINN")
        axes[1][0].set_xlabel("relative time")
        axes[1][0].set_ylabel("AIF")
        axes[1][0].grid(alpha=0.25)
    else:
        axes[1][0].text(0.5, 0.5, "No AIF curve", ha="center", va="center")
        axes[1][0].axis("off")

    if param_names:
        positions = np.arange(len(param_names))
        axes[1][1].barh(positions, param_values, color="#4d8f6d")
        axes[1][1].set_yticks(positions)
        axes[1][1].set_yticklabels(param_names)
        axes[1][1].set_xlabel("value")
        axes[1][1].set_title("Hemodynamic parameters")
        axes[1][1].grid(axis="x", alpha=0.25)
    else:
        axes[1][1].text(0.5, 0.5, "No PINN parameters", ha="center", va="center")
        axes[1][1].axis("off")
    _apply_figure_header(fig, payload, "PINN hemodynamic fitting preview")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_text_panel(message: str, output_path: Path, *, plt) -> None:
    fig = plt.figure(figsize=(8, 2.5))
    axis = fig.add_subplot(111)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _mid_slice(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected [C, D, H, W] or [D, H, W] volume, got {array.shape}")
    return array[array.shape[0] // 2]


def _normalize_image(image: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(image, dtype=np.float32), copy=False)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high <= low:
        low = float(array.min()) if array.size else 0.0
        high = float(array.max()) if array.size else 1.0
    return np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _apply_figure_header(fig, payload: dict[str, Any], title: str) -> None:
    fig.suptitle(title, fontsize=13, y=0.985)
    sample_caption = _sample_caption(payload)
    if sample_caption:
        fig.text(0.5, 0.955, sample_caption, ha="center", va="top", fontsize=10)


def _sample_caption(payload: dict[str, Any]) -> str:
    identifiers = payload.get("identifiers") or {}
    if not isinstance(identifiers, dict):
        return ""
    sample_name = identifiers.get("sample_id") or identifiers.get("patient_id") or "unknown"
    extras = []
    if identifiers.get("dataset_name"):
        extras.append(f"dataset={identifiers['dataset_name']}")
    if identifiers.get("visit_timepoint"):
        extras.append(f"visit={identifiers['visit_timepoint']}")
    extra_text = f" | {' | '.join(extras)}" if extras else ""
    return f"sample: {sample_name}{extra_text}"


def _phase_tile_title(
    *,
    phase_id: int,
    predicted: int,
    relative_time: float,
    attention: float,
    confidence: float,
) -> str:
    return (
        f"gt p{phase_id} | pred p{predicted}\n"
        f"t={relative_time:.2f}  attn={attention:.2f}  conf={confidence:.2f}"
    )
