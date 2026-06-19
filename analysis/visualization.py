from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heatmap_png(image: np.ndarray, heatmap: np.ndarray, output_path: str | Path) -> None:
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
