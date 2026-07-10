from __future__ import annotations

import numpy as np


def concordance_index_safe(time, risk, event) -> float:
    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(np.asarray(time), -np.asarray(risk), np.asarray(event)))
    except Exception:
        return float("nan")
