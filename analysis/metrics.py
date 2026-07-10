from __future__ import annotations

import numpy as np


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return {}
    y_true = y_true[mask].astype(int)
    y_pred = (y_prob[mask] >= threshold).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "sensitivity": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
        "ppv": tp / max(tp + fp, 1.0),
        "npv": tn / max(tn + fn, 1.0),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1.0),
    }
