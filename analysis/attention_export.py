from __future__ import annotations

from pathlib import Path

import numpy as np


def save_attention_array(attention, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(attention))
    return path
