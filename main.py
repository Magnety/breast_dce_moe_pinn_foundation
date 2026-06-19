from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

DEFAULT_PRETRAIN_CONFIG = CONFIG_DIR / "pretrain_dce_moe_pinn.yaml"
DEFAULT_FINETUNE_CONFIG = CONFIG_DIR / "finetune_multitask.yaml"
DEFAULT_INFER_CONFIG = CONFIG_DIR / "infer_export_maps.yaml"

DEFAULT_OUTPUT_CSV = Path("outputs/breast_dce_moe_pinn/infer/predictions.csv")
DEFAULT_MAP_OUTPUT_DIR = Path("outputs/breast_dce_moe_pinn/maps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Breast-DCE-MoE-PINN Foundation Model entry point."
    )

    # 对齐 pcr_project 的风格：用 --mode 控制运行阶段，而不是 sub-command
    parser.add_argument(
        "--mode",
        default="pretrain",
        choices=["pretrain", "finetune", "infer", "both", "export-maps"],
        help=(
            "Run mode. "
            "pretrain: self-supervised pretraining; "
            "finetune: masked multi-task finetuning; "
            "infer: inference only; "
            "both: finetune then infer; "
            "export-maps: export hemodynamic maps from a DCE .npy file."
        ),
    )

    # 通用配置。若指定 --config，则所有 mode 优先使用这个配置。
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to yaml config. If provided, it overrides mode-specific config paths.",
    )

    # 分阶段配置。没有传 --config 时，会根据 mode 自动选择对应配置。
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=DEFAULT_PRETRAIN_CONFIG,
        help="Path to pretrain yaml config.",
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=DEFAULT_FINETUNE_CONFIG,
        help="Path to finetune yaml config.",
    )
    parser.add_argument(
        "--infer-config",
        type=Path,
        default=DEFAULT_INFER_CONFIG,
        help="Path to infer yaml config.",
    )

    # 类比参考项目 main_distributed.py 的常用参数
    parser.add_argument(
        "--gpus",
        default='0,1,2',
        help="GPU ids, e.g. '0' or '0,1'. If not set, use current CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint if supported by solver.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed in config.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config and exit.",
    )

    # 类比 classic main 中常见的 fold / checkpoint 参数
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional fold index. If not set, use config setting.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for finetune/infer.",
    )

    # 推理参数
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for inference predictions.",
    )

    # 快速覆盖 batch_size / num_workers，方便 PyCharm 调试
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for current mode.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional num_workers override for current mode.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Optional DataLoader prefetch_factor override when num_workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=None,
        help="Keep DataLoader workers alive across epochs.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Enable DataLoader pinned host memory.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pinned host memory.",
    )
    parser.add_argument(
        "--cuda-prefetch",
        dest="cuda_prefetch",
        action="store_true",
        default=None,
        help="Move the next batch to CUDA on a side stream while the current batch computes.",
    )
    parser.add_argument(
        "--no-cuda-prefetch",
        dest="cuda_prefetch",
        action="store_false",
        help="Disable CUDA-side batch prefetching.",
    )
    parser.add_argument(
        "--drop-last",
        dest="drop_last",
        action="store_true",
        default=None,
        help="Drop the last incomplete training batch.",
    )
    parser.add_argument(
        "--no-drop-last",
        dest="drop_last",
        action="store_false",
        help="Keep the last incomplete training batch.",
    )
    parser.add_argument(
        "--loader-multiprocessing-context",
        choices=["spawn", "fork", "forkserver"],
        default=None,
        help="Optional DataLoader multiprocessing context. 'spawn' is safest for DDP + workers.",
    )
    parser.add_argument(
        "--cache-processed",
        dest="cache_processed",
        action="store_true",
        default=None,
        help="Cache cropped/normalized volumes on disk for faster later epochs.",
    )
    parser.add_argument(
        "--no-cache-processed",
        dest="cache_processed",
        action="store_false",
        help="Disable processed-volume disk cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for processed-volume cache.",
    )
    parser.add_argument(
        "--warmup-cache",
        action="store_true",
        help="Build the processed-volume cache before starting training/inference.",
    )
    parser.add_argument(
        "--warmup-cache-only",
        action="store_true",
        help="Build the processed-volume cache and exit without training/inference.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit torch/OMP/MKL CPU threads per process to reduce worker contention.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_true",
        default=None,
        help="Enable torch.backends.cudnn.benchmark.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        dest="cudnn_benchmark",
        action="store_false",
        help="Disable torch.backends.cudnn.benchmark.",
    )
    parser.add_argument(
        "--allow-tf32",
        dest="allow_tf32",
        action="store_true",
        default=None,
        help="Enable TF32 matmul/cudnn kernels on supported NVIDIA GPUs.",
    )
    parser.add_argument(
        "--no-allow-tf32",
        dest="allow_tf32",
        action="store_false",
        help="Disable TF32 matmul/cudnn kernels.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="How often to synchronize and print step loss when progress bar is enabled.",
    )
    parser.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action="store_true",
        default=None,
        help="Enable tqdm progress bars during train/eval.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="progress_bar",
        action="store_false",
        help="Disable tqdm progress bars to avoid per-step overhead.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Evaluate every N epochs. Use 0 to disable validation during training.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save periodic checkpoints every N epochs. Use 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--no-save-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        default=None,
        help="Disable checkpoint writing during training.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Override gradient clipping norm. Use 0 to skip grad clipping.",
    )
    parser.add_argument(
        "--optimizer-fused",
        dest="optimizer_fused",
        action="store_true",
        default=None,
        help="Try fused AdamW on CUDA, falling back automatically if unsupported.",
    )
    parser.add_argument(
        "--no-optimizer-fused",
        dest="optimizer_fused",
        action="store_false",
        help="Disable fused AdamW.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        dest="ddp_static_graph",
        action="store_true",
        default=None,
        help="Enable DDP static_graph fast path when the used/unused parameter set is stable.",
    )
    parser.add_argument(
        "--no-ddp-static-graph",
        dest="ddp_static_graph",
        action="store_false",
        help="Disable DDP static_graph.",
    )
    parser.add_argument(
        "--torch-compile",
        dest="torch_compile",
        action="store_true",
        default=None,
        help="Try torch.compile for the model when available.",
    )
    parser.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--channels-last-3d",
        dest="channels_last_3d",
        action="store_true",
        default=None,
        help="Use channels_last_3d memory format before Conv3d patch embedding on CUDA.",
    )
    parser.add_argument(
        "--no-channels-last-3d",
        dest="channels_last_3d",
        action="store_false",
        help="Disable channels_last_3d Conv3d input layout.",
    )

    # export-maps 参数
    parser.add_argument(
        "--dce-npy",
        type=Path,
        default=None,
        help="Path to DCE sequence .npy file. Required when --mode export-maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MAP_OUTPUT_DIR,
        help="Output directory for hemodynamic maps.",
    )
    parser.add_argument(
        "--no-nifti",
        action="store_true",
        help="Do not save NIfTI files when exporting hemodynamic maps.",
    )

    # ------------------------------------------------------------------
    # Data split overrides. These translate to data.split_strategy.* and let
    # users pick how training / inference samples are drawn without having to
    # edit yaml. They apply to finetune and infer modes; pretraining keeps
    # using every sample by default unless --split-mode is set explicitly.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--split-mode",
        choices=["all", "manifest", "by_dataset", "by_ratio"],
        default=None,
        help=(
            "Override data.split_strategy.mode. "
            "all: use every sample. "
            "manifest: honour the split column. "
            "by_dataset: pick datasets via --train-datasets / --test-datasets. "
            "by_ratio: split by --train-ratio."
        ),
    )
    parser.add_argument(
        "--train-datasets",
        default=None,
        help="Comma-separated dataset_id list for split-mode=by_dataset (e.g. 'ispy2,duke').",
    )
    parser.add_argument(
        "--test-datasets",
        default=None,
        help="Comma-separated dataset_id list for the held-out side of split-mode=by_dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train fraction used by split-mode=by_ratio (e.g. 0.7).",
    )
    parser.add_argument(
        "--include-datasets",
        default=None,
        help="Comma-separated dataset_id list to keep before splitting (applied to all modes).",
    )
    parser.add_argument(
        "--exclude-datasets",
        default=None,
        help="Comma-separated dataset_id list to drop before splitting (applied to all modes).",
    )

    # ------------------------------------------------------------------
    # Multi-GPU / DDP launcher options. By default we infer the world size
    # from --gpus, but users can pin it explicitly. The master port is also
    # exposed so concurrent runs on the same host do not collide.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of DDP workers. Defaults to the number of GPUs in --gpus / CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--master-addr",
        default=None,
        help="Override MASTER_ADDR for DDP rendezvous (default 127.0.0.1).",
    )
    parser.add_argument(
        "--master-port",
        default=None,
        help="Override MASTER_PORT for DDP rendezvous (default: random free port).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.mode == "export-maps":
        return _run_export_maps(args)

    if args.mode == "pretrain":
        config = _load_stage_config(args, stage="pretrain")
        return _run_stage(config=config, mode="pretrain", args=args)

    if args.mode == "finetune":
        config = _load_stage_config(args, stage="finetune")
        return _run_stage(config=config, mode="finetune", args=args)

    if args.mode == "infer":
        config = _load_stage_config(args, stage="infer")
        return _run_infer_stage(config=config, output_csv=args.output_csv, args=args)

    if args.mode == "both":
        # 根据你的需求，这里的 both 表示：先 finetune，再 infer
        finetune_config = _load_stage_config(args, stage="finetune")
        ret = _run_stage(config=finetune_config, mode="finetune", args=args)
        if ret != 0:
            return ret

        infer_config = _load_stage_config(args, stage="infer")
        return _run_infer_stage(config=infer_config, output_csv=args.output_csv, args=args)

    raise ValueError(f"Unsupported mode: {args.mode}")


def _load_stage_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.config import load_config

    if args.config is not None:
        config_path = args.config
    elif stage == "pretrain":
        config_path = args.pretrain_config
    elif stage == "finetune":
        config_path = args.finetune_config
    elif stage == "infer":
        config_path = args.infer_config
    else:
        raise ValueError(f"Unknown stage: {stage}")

    config = load_config(config_path)
    config = _apply_runtime_overrides(config, args=args, stage=stage)
    config["_runtime"] = {
        "stage": stage,
        "config_path": str(config_path),
        "mode": args.mode,
        "gpus": args.gpus,
        "resume": bool(args.resume),
    }
    return config


def _apply_runtime_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
) -> dict[str, Any]:
    """
    将命令行参数覆盖到 yaml config 中。
    这样 PyCharm 里只改 Parameters 就可以快速调试，不必频繁改 yaml。
    """

    if args.seed is not None:
        config["seed"] = int(args.seed)

    if args.resume:
        config["resume"] = True

    if args.fold is not None:
        # 兼容不同 config 命名方式
        config.setdefault("finetune", {})
        config["finetune"]["fold"] = int(args.fold)

        config.setdefault("infer", {})
        config["infer"]["fold"] = int(args.fold)

    if args.checkpoint is not None:
        ckpt = str(args.checkpoint)
        training_cfg = config.setdefault("training", {})

        if stage == "finetune":
            config.setdefault("finetune", {})
            config["finetune"]["pretrained_path"] = ckpt
            config["finetune"]["checkpoint"] = ckpt
            training_cfg["pretrained_checkpoint"] = ckpt

        if stage == "infer":
            config.setdefault("infer", {})
            config["infer"]["checkpoint"] = ckpt
            config["infer"]["checkpoint_path"] = ckpt
            config.setdefault("inference", {})
            config["inference"]["checkpoint"] = ckpt
            training_cfg["checkpoint"] = ckpt

    loader_targets = _loader_override_targets(config, stage)
    if args.batch_size is not None:
        _set_loader_override(loader_targets, "batch_size", int(args.batch_size))

    if args.num_workers is not None:
        _set_loader_override(loader_targets, "num_workers", int(args.num_workers))
    if args.prefetch_factor is not None:
        _set_loader_override(loader_targets, "prefetch_factor", int(args.prefetch_factor))
    if getattr(args, "persistent_workers", None) is not None:
        _set_loader_override(loader_targets, "persistent_workers", bool(args.persistent_workers))
    if getattr(args, "pin_memory", None) is not None:
        _set_loader_override(loader_targets, "pin_memory", bool(args.pin_memory))
    if getattr(args, "drop_last", None) is not None:
        _set_loader_override(loader_targets, "drop_last", bool(args.drop_last))
    loader_context = getattr(args, "loader_multiprocessing_context", None)
    if loader_context is not None:
        _set_loader_override(
            loader_targets,
            "multiprocessing_context",
            str(loader_context),
        )
    training_cfg = config.setdefault("training", {})
    if getattr(args, "cuda_prefetch", None) is not None:
        training_cfg["cuda_prefetch"] = bool(args.cuda_prefetch)
    if getattr(args, "log_interval", None) is not None:
        training_cfg["log_interval"] = int(args.log_interval)
    if getattr(args, "progress_bar", None) is not None:
        training_cfg["progress_bar"] = bool(args.progress_bar)
    if getattr(args, "eval_interval", None) is not None:
        training_cfg["eval_interval"] = int(args.eval_interval)
    if getattr(args, "checkpoint_interval", None) is not None:
        training_cfg["checkpoint_interval"] = int(args.checkpoint_interval)
    if getattr(args, "save_checkpoints", None) is not None:
        training_cfg["save_checkpoints"] = bool(args.save_checkpoints)
    if getattr(args, "grad_clip_norm", None) is not None:
        training_cfg["grad_clip_norm"] = float(args.grad_clip_norm)
    if getattr(args, "optimizer_fused", None) is not None:
        training_cfg["optimizer_fused"] = bool(args.optimizer_fused)

    data_cfg = config.setdefault("data", {})
    cache_cfg = data_cfg.setdefault("cache", {})
    if getattr(args, "cache_processed", None) is not None:
        cache_cfg["enabled"] = bool(args.cache_processed)
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir is not None:
        cache_cfg["dir"] = str(cache_dir)

    performance_cfg = config.setdefault("performance", {})
    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        performance_cfg["cpu_num_threads"] = int(cpu_threads)
    if getattr(args, "cudnn_benchmark", None) is not None:
        performance_cfg["cudnn_benchmark"] = bool(args.cudnn_benchmark)
    if getattr(args, "allow_tf32", None) is not None:
        performance_cfg["allow_tf32"] = bool(args.allow_tf32)
    if getattr(args, "ddp_static_graph", None) is not None:
        performance_cfg["ddp_static_graph"] = bool(args.ddp_static_graph)
    if getattr(args, "torch_compile", None) is not None:
        performance_cfg["torch_compile"] = bool(args.torch_compile)

    model_cfg = config.setdefault("model", {})
    if getattr(args, "channels_last_3d", None) is not None:
        model_cfg["channels_last_3d"] = bool(args.channels_last_3d)

    _apply_split_overrides(config, args)

    return config


def _loader_override_targets(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = [config.setdefault(stage, {})]
    if stage in {"pretrain", "finetune"}:
        targets.append(config.setdefault("training", {}))
    elif stage == "infer":
        targets.extend(
            [
                config.setdefault("infer", {}),
                config.setdefault("inference", {}),
                config.setdefault("training", {}),
            ]
        )

    unique_targets: list[dict[str, Any]] = []
    seen: set[int] = set()
    for target in targets:
        if id(target) in seen:
            continue
        unique_targets.append(target)
        seen.add(id(target))
    return unique_targets


def _set_loader_override(targets: list[dict[str, Any]], key: str, value: Any) -> None:
    for target in targets:
        target[key] = value


def _apply_split_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Translate CLI flags into config['data']['split_strategy'] / dataset filters.

    Only writes keys the user explicitly set on the command line so existing
    yaml values keep working untouched.
    """

    data_cfg = config.setdefault("data", {})

    if args.include_datasets is not None:
        data_cfg["include_datasets"] = _split_csv(args.include_datasets)
    if args.exclude_datasets is not None:
        data_cfg["exclude_datasets"] = _split_csv(args.exclude_datasets)

    needs_strategy = any(
        value is not None
        for value in (args.split_mode, args.train_datasets, args.test_datasets, args.train_ratio)
    )
    if not needs_strategy:
        return

    strategy = dict(data_cfg.get("split_strategy") or {})
    if args.split_mode is not None:
        strategy["mode"] = args.split_mode
    if args.train_datasets is not None:
        strategy["train_datasets"] = _split_csv(args.train_datasets)
    if args.test_datasets is not None:
        strategy["test_datasets"] = _split_csv(args.test_datasets)
    if args.train_ratio is not None:
        strategy["train_ratio"] = float(args.train_ratio)
    # Default mode when caller only passes train_datasets / train_ratio.
    if "mode" not in strategy:
        if "train_ratio" in strategy:
            strategy["mode"] = "by_ratio"
        elif "train_datasets" in strategy:
            strategy["mode"] = "by_dataset"
        else:
            strategy["mode"] = "manifest"
    data_cfg["split_strategy"] = strategy


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _configure_torch_runtime(config: dict[str, Any]) -> None:
    performance_cfg = config.get("performance", {}) or {}
    training_cfg = config.get("training", {}) or {}

    cpu_threads = performance_cfg.get("cpu_num_threads", training_cfg.get("cpu_num_threads"))
    if cpu_threads is not None:
        thread_count = max(1, int(cpu_threads))
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
    try:
        import torch
    except ModuleNotFoundError:
        return

    if cpu_threads is not None:
        torch.set_num_threads(max(1, int(cpu_threads)))

    if torch.cuda.is_available():
        cudnn_benchmark = bool(performance_cfg.get("cudnn_benchmark", training_cfg.get("cudnn_benchmark", True)))
        torch.backends.cudnn.benchmark = cudnn_benchmark
        allow_tf32 = bool(performance_cfg.get("allow_tf32", training_cfg.get("allow_tf32", True)))
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def _warmup_processed_cache(config: dict[str, Any], mode: str) -> None:
    data_cfg = config.get("data", {}) or {}
    cache_cfg = data_cfg.get("cache", {}) or {}
    cache_enabled = bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False)))
    if not cache_enabled:
        print("[cache] processed-volume cache is disabled; skipping warmup.")
        return

    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import (
        MultimodalManifestDataset,
    )

    try:
        from tqdm import tqdm
    except ModuleNotFoundError:  # pragma: no cover
        tqdm = None

    dataset = MultimodalManifestDataset(
        manifest_path=data_cfg["manifest_path"],
        dataset_root=data_cfg.get("dataset_root"),
        split=None,
        label_columns=data_cfg.get("label_columns"),
        target_shape=tuple(data_cfg.get("target_shape", config.get("model", {}).get("image_size", (64, 128, 128)))),
        modalities=data_cfg.get("modalities"),
        normalize=bool(data_cfg.get("normalize", True)),
        allow_empty_labels=True,
        max_volumes=data_cfg.get("max_volumes"),
        augmentation=data_cfg.get("augmentation"),
        augment=False,
        include_datasets=data_cfg.get("include_datasets"),
        exclude_datasets=data_cfg.get("exclude_datasets"),
        cache_processed=True,
        cache_dir=data_cfg.get("cache_dir", cache_cfg.get("dir")),
        cache_after_normalize=bool(cache_cfg.get("after_normalize", True)),
        npy_mmap_mode=cache_cfg.get("npy_mmap_mode", data_cfg.get("npy_mmap_mode")),
    )

    iterator = range(len(dataset))
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"warmup {mode} processed cache", ncols=100)
    for index in iterator:
        dataset[index]
    print(f"[cache] warmed {len(dataset)} samples in {data_cfg.get('cache_dir', cache_cfg.get('dir')) or dataset.cache_dir}")


def _run_stage(config: dict[str, Any], mode: str, args: argparse.Namespace) -> int:
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    _configure_torch_runtime(config)
    if args.warmup_cache or args.warmup_cache_only:
        _warmup_processed_cache(config=config, mode=mode)
        if args.warmup_cache_only:
            return 0

    world_size = _resolve_world_size(args, config)
    if world_size > 1:
        return _spawn_workers(config=config, mode=mode, args=args, world_size=world_size)
    return _run_stage_worker(rank=0, world_size=1, config=config, mode=mode, args=args)


def _resolve_world_size(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Decide how many DDP workers to spawn.

    Priority:
    1. ``--world-size`` if provided.
    2. The number of visible GPUs (``CUDA_VISIBLE_DEVICES`` count, or
       ``torch.cuda.device_count`` as a fallback).
    3. Single-process when CUDA is unavailable.
    """

    if getattr(args, "world_size", None) is not None and args.world_size > 0:
        return int(args.world_size)

    if args.gpus:
        gpu_ids = [g for g in str(args.gpus).split(",") if g.strip() != ""]
        if gpu_ids:
            return len(gpu_ids)

    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        pass
    return 1


def _spawn_workers(
    config: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    world_size: int,
) -> int:
    """Launch ``world_size`` DDP workers via ``torch.multiprocessing.spawn``.

    Using spawn (rather than ``torchrun``) keeps the PyCharm "Run" button
    workflow working: the IDE invokes ``main.py`` once and the script forks
    its own process group.
    """

    import torch.multiprocessing as mp

    print(f"[ddp] launching {world_size} workers for stage={mode}")
    master_port = str(getattr(args, "master_port", None) or os.environ.get("MASTER_PORT") or _pick_free_port())
    master_addr = str(getattr(args, "master_addr", None) or os.environ.get("MASTER_ADDR") or "127.0.0.1")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    ctx = mp.spawn(
        _ddp_worker_entry,
        args=(world_size, config, mode, args, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )
    # mp.spawn returns None on success and raises on failure.
    return 0


def _ddp_worker_entry(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    master_addr: str,
    master_port: str,
) -> None:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import (
        destroy_distributed,
        init_distributed,
    )

    init_distributed(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
    try:
        _run_stage_worker(rank=rank, world_size=world_size, config=config, mode=mode, args=args)
    finally:
        destroy_distributed()


def _run_stage_worker(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
) -> int:
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.solver import (
        BreastDCEMoEPINNSolver,
    )
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.seed import set_seed

    # Stagger the seed by rank so workers do not all see identical augmentation
    # streams; the same base seed keeps experiments reproducible.
    _configure_torch_runtime(config)
    set_seed(int(config.get("seed", 2026)) + rank)

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Running Breast-DCE-MoE-PINN stage: {mode} (world_size={world_size})")
        print("=" * 80)

    solver = BreastDCEMoEPINNSolver(config, mode=mode)

    # 如果你的 solver 内部已经根据 config["resume"] 自动 resume，这里不会冲突。
    # 如果 solver 有 resume_checkpoint 方法，则优先调用。
    if args.resume and hasattr(solver, "resume_checkpoint"):
        solver.resume_checkpoint(auto=True)

    solver.run()
    return 0


def _pick_free_port() -> int:
    """Pick an unused TCP port on localhost so concurrent runs don't clash."""

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_infer_stage(
    config: dict[str, Any],
    output_csv: Path,
    args: argparse.Namespace,
) -> int:
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    _configure_torch_runtime(config)
    if args.warmup_cache or args.warmup_cache_only:
        _warmup_processed_cache(config=config, mode="infer")
        if args.warmup_cache_only:
            return 0

    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.trainers.inferencer import (
        Inferencer,
    )

    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 80)
    print("Running Breast-DCE-MoE-PINN inference")
    print("=" * 80)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    Inferencer(config).run(output_csv)
    return 0


def _run_export_maps(args: argparse.Namespace) -> int:
    if args.dce_npy is None:
        raise ValueError(
            "--dce-npy is required when --mode export-maps. "
            "Example: python main.py --mode export-maps --dce-npy sample_dce.npy"
        )

    import numpy as np
    import torch

    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.analysis.hemodynamic_map_export import (
        export_hemodynamic_maps,
    )

    sequence = torch.from_numpy(np.load(args.dce_npy)).float()
    paths = export_hemodynamic_maps(
        sequence=sequence,
        output_dir=args.output_dir,
        save_nifti=not args.no_nifti,
    )

    for name, path in paths.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
