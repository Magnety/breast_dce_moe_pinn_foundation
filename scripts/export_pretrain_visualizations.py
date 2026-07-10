from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.analysis.visualization import (
    save_pretrain_visualizations,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import (
    MultimodalManifestDataset,
    make_collate,
    parse_split_config,
    split_dataset,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CPU-only pretrain visualizations from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path saved by training.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number for naming the export.")
    parser.add_argument("--output-dir", required=True, help="Directory where visualization files will be written.")
    parser.add_argument(
        "--preview-indices",
        default="",
        help="Comma-separated dataset indices inside the resolved pretrain train subset. Default: random sample.",
    )
    parser.add_argument("--preview-batch-size", type=int, default=0, help="Override preview batch size.")
    parser.add_argument(
        "--max-visualized-volumes",
        type=int,
        default=0,
        help="Override how many valid volumes to include in MAE panels.",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_existing_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = dict(payload.get("config") or {})
    if not config:
        raise ValueError(f"Checkpoint {checkpoint_path} did not contain a config payload.")

    _limit_cpu_threads()
    set_seed(int(config.get("seed", 2026)) + int(args.epoch))

    visualization_cfg = dict(config.get("visualization", {}) or {})
    random_preview_sample = bool(visualization_cfg.get("random_preview_sample", True))
    preview_batch_size = max(1, int(args.preview_batch_size or visualization_cfg.get("preview_batch_size", 1)))
    max_visualized_volumes = max(
        1,
        int(args.max_visualized_volumes or visualization_cfg.get("max_visualized_volumes", 4)),
    )

    preview_dataset = _build_pretrain_preview_dataset(config)
    preview_indices = _parse_preview_indices(args.preview_indices)
    if not preview_indices and not random_preview_sample:
        preview_indices = _parse_config_indices(visualization_cfg.get("preview_indices"))
    selected = [int(index) for index in preview_indices if 0 <= int(index) < len(preview_dataset)]
    if not selected:
        selected = _choose_random_indices(len(preview_dataset), preview_batch_size)
    samples = [preview_dataset[index] for index in selected[:preview_batch_size]]
    data_cfg = config.get("data", {})
    collate_fn = make_collate(
        int(data_cfg.get("max_volumes")) if data_cfg.get("max_volumes") else None,
        defer_volume_stack=False,
        collect_diagnostics=False,
    )
    batch = collate_fn(samples)

    model_kwargs = dict(config.get("model", {}))
    model_kwargs["use_param_anchor"] = False
    model = BreastDCEMoEPINNModel(**model_kwargs).cpu()
    load_checkpoint(checkpoint_path, model, strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(
            batch,
            mode="pretrain",
            collect_visuals=True,
            max_visualized_volumes=max_visualized_volumes,
        )
    visuals = outputs.get("aux", {}).get("pretrain_visuals")
    if not isinstance(visuals, dict):
        visuals = {"identifiers": {"warning": "pretrain_visuals payload was empty"}}
    identifiers = dict(visuals.get("identifiers") or {})
    identifiers["preview_dataset_index"] = int(selected[0]) if selected else None
    identifiers["preview_dataset_indices"] = [int(index) for index in selected[:preview_batch_size]]
    visuals["identifiers"] = identifiers

    output_dir = _resolve_output_path(args.output_dir)
    files = save_pretrain_visualizations(visuals, output_dir, epoch=int(args.epoch))
    print(json.dumps({"epoch": int(args.epoch), "output_dir": str(output_dir), "files": files}, ensure_ascii=False))
    return 0


def _build_pretrain_preview_dataset(config: dict[str, Any]) -> Dataset | Subset:
    data_cfg = config.get("data", {})
    split_payload = data_cfg.get("split_strategy")
    train_split_value = data_cfg.get("train_split")
    if split_payload:
        base_dataset = _build_dataset(config, split=None)
        split_cfg = parse_split_config(split_payload, default_mode="all")
        subsets = split_dataset(base_dataset, split_cfg)
        train_subset = subsets.get("train")
        if train_subset is None or len(train_subset) == 0:
            raise ValueError("Resolved pretrain train subset is empty.")
        return train_subset
    dataset = _build_dataset(config, split=train_split_value)
    if len(dataset) == 0:
        raise ValueError("Resolved pretrain dataset is empty.")
    return dataset


def _build_dataset(config: dict[str, Any], split: str | None) -> MultimodalManifestDataset:
    data_cfg = config.get("data", {})
    cache_cfg = data_cfg.get("cache", {}) or {}
    restrict_to_dce_phases = bool(data_cfg.get("restrict_to_dce_phases", False))
    return MultimodalManifestDataset(
        manifest_path=data_cfg["manifest_path"],
        dataset_root=data_cfg.get("dataset_root"),
        split=split,
        label_columns=data_cfg.get("label_columns"),
        target_shape=tuple(
            data_cfg.get("target_shape", config.get("model", {}).get("image_size", (64, 128, 128)))
        ),
        modalities=data_cfg.get("modalities"),
        normalize=bool(data_cfg.get("normalize", True)),
        allow_empty_labels=True,
        max_volumes=data_cfg.get("max_volumes"),
        augmentation=data_cfg.get("augmentation"),
        augment=False,
        include_datasets=data_cfg.get("include_datasets"),
        exclude_datasets=data_cfg.get("exclude_datasets"),
        cache_processed=bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False))),
        cache_dir=data_cfg.get("cache_dir", cache_cfg.get("dir")),
        cache_after_normalize=bool(cache_cfg.get("after_normalize", True)),
        npy_mmap_mode=cache_cfg.get("npy_mmap_mode", data_cfg.get("npy_mmap_mode")),
        profile_io=False,
        compact_sample_tensors=False,
        include_identifiers=True,
        include_volume_paths=False,
        restrict_to_dce_phases=restrict_to_dce_phases,
    )


def _parse_preview_indices(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_config_indices(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    return _parse_preview_indices(str(raw))


def _choose_random_indices(dataset_size: int, count: int) -> list[int]:
    if dataset_size <= 0:
        return []
    chosen = torch.randperm(dataset_size)[: max(1, int(count))]
    return [int(index) for index in chosen.tolist()]


def _resolve_existing_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    candidates = [
        Path.cwd() / path,
        _foundation_dir() / path,
        _repo_root() / path,
    ]
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate.exists():
            return candidate.resolve(strict=True)
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find path '{raw_path}'. Checked: {checked}")


def _resolve_output_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path.cwd() / path).resolve(strict=False)


def _foundation_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _limit_cpu_threads() -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    set_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop):
        set_interop(1)


if __name__ == "__main__":
    raise SystemExit(main())
