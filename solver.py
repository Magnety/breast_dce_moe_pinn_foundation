from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets import (
    MultimodalManifestDataset,
    build_gpu_augment,
    dataloader_worker_init,
    make_collate,
    parse_split_config,
    split_dataset,
    variable_modalities_collate,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.losses import MaskedMultitaskLoss
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.models import BreastDCEMoEPINNModel
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.checkpoint import load_checkpoint, save_checkpoint
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.distributed import (
    CudaBatchPrefetcher,
    barrier,
    default_device,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    move_batch_to_device,
    reduce_scalar,
)
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.logger import build_logger
from src.breast_mri_ai.experiment_tracking import append_history_metrics

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None

try:
    _DATALOADER_SUPPORTS_IN_ORDER = "in_order" in inspect.signature(DataLoader).parameters
except (TypeError, ValueError):  # pragma: no cover
    _DATALOADER_SUPPORTS_IN_ORDER = False


def _identity_batch(batch):
    return batch


def _supports_batch_fetch(dataset: Dataset | Subset) -> bool:
    base: Dataset | Subset = dataset
    while isinstance(base, Subset):
        base = base.dataset
    return callable(getattr(base, "get_batch", None))


def _fetch_dataset_batch(dataset: Dataset | Subset, indices: list[int] | tuple[int, ...]) -> dict[str, Any]:
    base: Dataset | Subset = dataset
    resolved = [int(index) for index in indices]
    while isinstance(base, Subset):
        resolved = [int(base.indices[index]) for index in resolved]
        base = base.dataset
    get_batch = getattr(base, "get_batch", None)
    if not callable(get_batch):
        raise TypeError(f"Dataset {type(base).__name__} does not implement get_batch().")
    return get_batch(resolved)


class _EpochAwareBatchSampler(BatchSampler):
    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class _DirectBatchDataset(Dataset):
    def __init__(self, dataset: Dataset | Subset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist() if index.ndim > 0 else int(index.item())
        if isinstance(index, tuple):
            index = list(index)
        if isinstance(index, list):
            return _fetch_dataset_batch(self.dataset, index)
        return self.dataset[int(index)]


class VolumeBalancedDistributedSampler(Sampler[int]):
    """DDP sampler that balances real volume counts within each step.

    Fixed padded shapes make GPU compute uniform, but worker IO still scales with
    the number of real volumes in a sample. This sampler builds each global
    batch, then greedily assigns samples to ranks so every rank gets a similar
    real-volume load for that optimizer step.
    """

    def __init__(
        self,
        dataset: Dataset | Subset,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        max_volumes: int | None,
        modalities: list[str] | tuple[str, ...] | None,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        dataset_len = len(dataset)
        if self.drop_last:
            self.global_batches = dataset_len // self.global_batch_size
        else:
            self.global_batches = (dataset_len + self.global_batch_size - 1) // self.global_batch_size
        self.total_size = self.global_batches * self.global_batch_size
        self.num_samples = self.global_batches * self.batch_size
        self.volume_counts = [
            _sample_volume_count(dataset, index, max_volumes=max_volumes, modalities=modalities)
            for index in range(dataset_len)
        ]

    def __iter__(self):
        dataset_len = len(self.dataset)
        if dataset_len == 0 or self.num_samples == 0:
            return iter(())
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(dataset_len, generator=generator).tolist()
        else:
            indices = list(range(dataset_len))

        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            repeat = (self.total_size - len(indices) + len(indices) - 1) // len(indices)
            indices.extend((indices * repeat)[: self.total_size - len(indices)])

        rank_indices: list[int] = []
        for start in range(0, len(indices), self.global_batch_size):
            window = indices[start : start + self.global_batch_size]
            bins: list[list[int]] = [[] for _ in range(self.num_replicas)]
            loads = [0 for _ in range(self.num_replicas)]
            for index in sorted(window, key=lambda item: self.volume_counts[item], reverse=True):
                candidates = [rank for rank in range(self.num_replicas) if len(bins[rank]) < self.batch_size]
                target_rank = min(candidates, key=lambda rank: loads[rank])
                bins[target_rank].append(index)
                loads[target_rank] += self.volume_counts[index]
            rank_indices.extend(bins[self.rank])
        return iter(rank_indices[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _sample_volume_count(
    dataset: Dataset | Subset,
    index: int,
    *,
    max_volumes: int | None,
    modalities: list[str] | tuple[str, ...] | None,
) -> int:
    base: Dataset | Subset = dataset
    actual_index = int(index)
    while isinstance(base, Subset):
        actual_index = int(base.indices[actual_index])
        base = base.dataset
    samples = getattr(base, "samples", None)
    if not samples:
        return 1
    records = samples[actual_index].get("volume_records", [])
    modality_filter = {str(item) for item in (modalities or []) if str(item).strip()}
    count = 0
    for record in records:
        if modality_filter and record.get("modality") not in modality_filter:
            continue
        count += 1
        if max_volumes is not None and count >= int(max_volumes):
            break
    return max(1, count)


class BreastDCEMoEPINNSolver:
    """Standalone solver for our multimodal, multiphase DCE-MoE-PINN model.

    This solver intentionally lives inside ``breast_dce_moe_pinn_foundation`` so
    it is decoupled from the generic SOTA comparison framework. It owns dataset
    construction, augmentation switches, optimization, masked multi-task loss,
    PINN/MAE pretraining losses, logging and checkpointing for this method.
    """

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        if mode not in {"pretrain", "finetune", "infer"}:
            raise ValueError(f"Unsupported Breast-DCE-MoE-PINN mode: {mode}")
        # 关键：限制主进程 CPU 线程数。32-core 机器上不限制时，BLAS / numpy 默认
        # 开 cpu_count 个线程，多 worker dataloader + DDP all_reduce + 主进程
        # GPU prefetch H2D 会互相抢 CPU，wait 时间抖到秒级。参考 pcr_project
        # main_distributed.py 的做法，主进程 + 每个 worker 都锁到 1 thread。
        cpu_threads = int(config.get("performance", {}).get("cpu_num_threads", 1))
        if cpu_threads > 0:
            os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
            torch.set_num_threads(cpu_threads)
        self.config = config
        self.mode = mode
        self.device = default_device()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.is_main = is_main_process()
        self.output_dir = Path(
            config.get("output", {}).get("root", f"outputs/breast_dce_moe_pinn/{mode}")
        ).expanduser().resolve(strict=False)
        self.logger = build_logger(self.output_dir / "logs", f"{mode}-rank{self.rank}")
        self.training_cfg = config.get("training", {})
        self.performance_cfg = config.get("performance", {})
        self.visualization_cfg = config.get("visualization", {}) or {}
        self._visualization_process: subprocess.Popen | None = None
        self._device_skip_keys = {"labels", "label_mask"}
        # 按 world_size + ddp_static_graph 决定是否启用 ``use_param_anchor``。
        # 多卡 DDP static_graph 下需要它锁定 used-set 一致；单卡或没启 static_graph
        # 时它只是无收益的 graph 开销。``use_param_anchor`` 不是 yaml 用户字段，
        # 仅在这里注入，避免污染用户 model 配置。
        model_kwargs: dict[str, Any] = dict(config.get("model", {}))
        model_kwargs["use_param_anchor"] = bool(
            is_distributed()
            and mode != "infer"
            and bool(self.performance_cfg.get("ddp_static_graph", False))
        )
        self.model = BreastDCEMoEPINNModel(**model_kwargs).to(self.device)
        self._maybe_compile_model()
        self.criterion = MaskedMultitaskLoss(config.get("loss", {}).get("task_weights"))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._log_optimizer_schedule()
        self.amp_enabled = bool(self.training_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        # GPU 端 batch augmentation：把 dataset/worker 上的 numpy aug 搬到 GPU。
        # 同样一组 flip/scale/shift/noise，在 batch tensor 上一次性做完只要几十微秒，
        # 而 numpy 版每个 sample 要花 100+ ms（randn 主导）。
        gpu_aug_cfg = self.config.get("data", {}).get("augmentation") if self.mode != "infer" else None
        self.gpu_augment = build_gpu_augment(gpu_aug_cfg) if self.device.type == "cuda" else None
        self.best_metric: float | None = None
        self.start_epoch = 1
        self._load_initial_checkpoint()
        # Wrap with DDP after the (potentially) initial checkpoint load so the
        # state-dict matches the unwrapped model. ``find_unused_parameters`` is
        # on because the MoE / mask paths can leave some experts inactive on a
        # given batch.
        if is_distributed() and self.mode != "infer":
            self._wrap_ddp_model()

    def _maybe_compile_model(self) -> None:
        compile_enabled = bool(self.performance_cfg.get("torch_compile", False))
        if not compile_enabled:
            return
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            if self.is_main:
                self.logger.warning("torch_compile requested but torch.compile is unavailable in this PyTorch build.")
            return
        default_mode = "max-autotune-no-cudagraphs" if self.device.type == "cuda" else "default"
        mode = str(self.performance_cfg.get("torch_compile_mode", default_mode))
        try:
            self.model = compile_fn(self.model, mode=mode)
            if self.is_main:
                self.logger.info("enabled torch.compile(mode=%s)", mode)
        except Exception as exc:  # noqa: BLE001
            if self.is_main:
                self.logger.warning("torch.compile failed, continuing without it: %s", exc)

    def _wrap_ddp_model(self) -> None:
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        static_graph = bool(self.performance_cfg.get("ddp_static_graph", False))
        find_unused = bool(self.performance_cfg.get("ddp_find_unused_parameters", True))
        kwargs: dict[str, Any] = {
            "device_ids": device_ids,
            "output_device": self.device.index if self.device.type == "cuda" else None,
            "find_unused_parameters": False if static_graph else find_unused,
        }
        if static_graph:
            kwargs["static_graph"] = True
        if bool(self.performance_cfg.get("ddp_gradient_as_bucket_view", True)):
            kwargs["gradient_as_bucket_view"] = True
        try:
            self.model = DDP(self.model, **kwargs)
        except TypeError:
            kwargs.pop("static_graph", None)
            kwargs.pop("gradient_as_bucket_view", None)
            if static_graph:
                kwargs["find_unused_parameters"] = find_unused
            self.model = DDP(self.model, **kwargs)

    def run(self) -> dict[str, float]:
        if self.mode == "infer":
            loader = self._build_inference_loader()
            return self.evaluate(loader, epoch=0, stage="infer")

        train_loader, val_loader = self._build_training_loaders()
        final_metrics: dict[str, float] = {}
        epochs = int(self.training_cfg.get("epochs", 100))
        eval_interval = int(self.training_cfg.get("eval_interval", 1))
        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            final_metrics = train_metrics
            should_eval = (
                val_loader is not None
                and eval_interval > 0
                and (epoch % eval_interval == 0 or epoch == epochs)
            )
            if should_eval:
                final_metrics = self.evaluate(val_loader, epoch=epoch, stage="val")
            self._after_epoch(epoch, train_metrics, final_metrics, evaluated=should_eval or val_loader is None)
        return final_metrics

    # ------------------------------------------------------------------
    # Loader construction. Pretraining ignores split filters by default and
    # uses every manifest sample. Finetuning and inference go through the
    # SplitConfig pipeline so users can pick by_dataset / by_ratio / manifest
    # without touching code.
    # ------------------------------------------------------------------

    def _build_training_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        data_cfg = self.config.get("data", {})

        if self.mode == "pretrain":
            # Pretraining: by default consume every sample, regardless of the
            # ``split`` column, unless the user explicitly opts into a split
            # strategy or sets ``train_split`` themselves.
            split_payload = data_cfg.get("split_strategy")
            train_split_value = data_cfg.get("train_split")
            if split_payload:
                split_cfg = parse_split_config(split_payload, default_mode="all")
                base_dataset = self._build_dataset(split=None, augment=True)
                subsets = split_dataset(base_dataset, split_cfg)
                train_subset = subsets.get("train")
                self._log_split_summary("pretrain", split_cfg, subsets)
                return self._make_loader(train_subset, shuffle=True), None
            base_dataset = self._build_dataset(split=train_split_value, augment=True)
            return self._make_loader(base_dataset, shuffle=True), None

        # finetune
        split_payload = data_cfg.get("split_strategy")
        if split_payload:
            split_cfg = parse_split_config(split_payload, default_mode="manifest")
            base_dataset = self._build_dataset(split=None, augment=True)
            subsets = split_dataset(base_dataset, split_cfg)
            self._log_split_summary("finetune", split_cfg, subsets)
            train_subset = subsets.get("train")
            val_subset = subsets.get("val") or subsets.get("test")
            train_loader = self._make_loader(train_subset, shuffle=True)
            val_loader = self._make_loader(val_subset, shuffle=False) if val_subset and len(val_subset) > 0 else None
            return train_loader, val_loader

        # Backward compatible path: legacy configs only specify train_split / val_split columns.
        train_dataset = self._build_dataset(split=data_cfg.get("train_split", "train"), augment=True)
        val_split = data_cfg.get("val_split")
        val_dataset = self._build_dataset(split=val_split, augment=False) if val_split else None
        return (
            self._make_loader(train_dataset, shuffle=True),
            self._make_loader(val_dataset, shuffle=False) if val_dataset is not None else None,
        )

    def _build_inference_loader(self) -> DataLoader:
        data_cfg = self.config.get("data", {})
        split_payload = data_cfg.get("split_strategy")
        if split_payload:
            split_cfg = parse_split_config(split_payload, default_mode="manifest")
            base_dataset = self._build_dataset(split=None, augment=False)
            subsets = split_dataset(base_dataset, split_cfg)
            self._log_split_summary("infer", split_cfg, subsets)
            target = subsets.get("test") or subsets.get("val") or subsets.get("train")
            return self._make_loader(target, shuffle=False)
        return self._make_loader(self._build_dataset(split=data_cfg.get("split"), augment=False), shuffle=False)

    def build_loader(self, split: str | None, shuffle: bool, augment: bool = False) -> DataLoader:
        # Kept for backwards compatibility with callers that drove the solver
        # by hand. New code should rely on _build_training_loaders /
        # _build_inference_loader.
        dataset = self._build_dataset(split=split, augment=augment)
        return self._make_loader(dataset, shuffle=shuffle)

    def _build_dataset(self, split: str | None, augment: bool) -> MultimodalManifestDataset:
        data_cfg = self.config["data"]
        cache_cfg = data_cfg.get("cache", {}) or {}
        restrict_to_dce_phases = bool(data_cfg.get("restrict_to_dce_phases", False))
        sample_pack_ready = bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False))) and bool(
            cache_cfg.get("after_normalize", True)
        )
        default_compact_sample_tensors = self.device.type == "cuda" and sample_pack_ready
        compact_sample_tensors = self.device.type == "cuda" and bool(
            self.training_cfg.get("defer_cpu_batch_stack", default_compact_sample_tensors)
        )
        # CPU 端 augmentation 强制关闭：在 dataloader worker 里跑 numpy aug 会
        # 把 GPU 饿死（randn 单次 ~30ms × N volumes/sample）。改在 GPU 上对整个
        # batch 做，速度快两个数量级。GPU augment 在 train_one_epoch 里调用。
        cpu_augment = False
        include_identifiers = self.mode == "infer"
        include_volume_paths = bool(self.training_cfg.get("timing_probe", False) or data_cfg.get("profile_io", False))
        return MultimodalManifestDataset(
            manifest_path=data_cfg["manifest_path"],
            dataset_root=data_cfg.get("dataset_root"),
            split=split,
            label_columns=data_cfg.get("label_columns"),
            target_shape=tuple(
                data_cfg.get("target_shape", self.config.get("model", {}).get("image_size", (64, 128, 128)))
            ),
            modalities=data_cfg.get("modalities"),
            normalize=bool(data_cfg.get("normalize", True)),
            allow_empty_labels=self.mode in {"pretrain", "infer"},
            max_volumes=data_cfg.get("max_volumes"),
            augmentation=data_cfg.get("augmentation"),
            augment=cpu_augment,
            include_datasets=data_cfg.get("include_datasets"),
            exclude_datasets=data_cfg.get("exclude_datasets"),
            cache_processed=bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False))),
            cache_dir=data_cfg.get("cache_dir", cache_cfg.get("dir")),
            cache_after_normalize=bool(cache_cfg.get("after_normalize", True)),
            npy_mmap_mode=cache_cfg.get("npy_mmap_mode", data_cfg.get("npy_mmap_mode")),
            profile_io=bool(self.training_cfg.get("timing_probe", False) or data_cfg.get("profile_io", False)),
            compact_sample_tensors=compact_sample_tensors,
            include_identifiers=include_identifiers,
            include_volume_paths=include_volume_paths,
            restrict_to_dce_phases=restrict_to_dce_phases,
        )

    def _make_loader(self, dataset: Dataset | Subset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise ValueError("Cannot build DataLoader from None dataset.")
        data_cfg = self.config.get("data", {})
        cache_cfg = data_cfg.get("cache", {}) or {}
        sample_pack_ready = bool(data_cfg.get("cache_processed", cache_cfg.get("enabled", False))) and bool(
            cache_cfg.get("after_normalize", True)
        )
        # 多卡 steady-state 里更怕的是 rank 间 sample IO 不均衡：某个 rank 恰好
        # 拿到更多真实 volume，它就会在 worker/collate/H2D 上拖尾，其他 GPU 只能
        # 等它做完 backward/all_reduce。sample-pack 能把单 sample 的 syscall 降下
        # 来，但不会自动均衡 rank 间的真实 volume 总量，所以 DDP + shuffle 时把
        # “按真实 volume 数平衡 rank 负载”作为默认策略。
        default_volume_balanced = bool(is_distributed() and shuffle and self.device.type == "cuda" and sample_pack_ready)
        balance_by_volume = bool(self.training_cfg.get("volume_balanced_sampling", default_volume_balanced))
        sampler = None
        drop_last = bool(shuffle and self.training_cfg.get("drop_last", False))
        batch_size = int(
            self.training_cfg.get("batch_size", self.config.get("inference", {}).get("batch_size", 1))
        )
        pad_to_max_volumes = data_cfg.get("max_volumes")
        if is_distributed():
            # DistributedSampler shards the dataset across ranks. For shuffled
            # training, balance each global step by real volume count so one
            # slow IO-heavy rank does not stall the others.
            if shuffle and balance_by_volume:
                sampler = VolumeBalancedDistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                    seed=int(self.config.get("seed", 2026)),
                    max_volumes=int(pad_to_max_volumes) if pad_to_max_volumes else None,
                    modalities=data_cfg.get("modalities"),
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
        num_workers = int(self.training_cfg.get("num_workers", 2))
        timing_probe = bool(self.training_cfg.get("timing_probe", False))
        default_defer_volume_stack = self.device.type == "cuda" and sample_pack_ready
        defer_volume_stack = self.device.type == "cuda" and bool(
            self.training_cfg.get("defer_cpu_batch_stack", default_defer_volume_stack)
        )
        default_batch_fetch = not default_defer_volume_stack
        use_batch_fetch = (
            bool(self.training_cfg.get("dataset_batch_fetch", default_batch_fetch))
            and not defer_volume_stack
            and _supports_batch_fetch(dataset)
        )
        in_order = None
        if num_workers > 0 and _DATALOADER_SUPPORTS_IN_ORDER:
            # volume-balanced sampler 预先按“全局 step”配平了每个 rank 的负载；
            # 再允许 worker 乱序返回就会把这层对齐打散，所以这里强制 FIFO。
            if balance_by_volume:
                in_order = True
            else:
                configured_in_order = self.training_cfg.get("in_order")
                if configured_in_order is None:
                    in_order = not shuffle
                else:
                    in_order = bool(configured_in_order)
        if self.is_main:
            self.logger.info(
                "[loader] shuffle=%s sample_pack_ready=%s volume_balanced_sampling=%s "
                "dataset_batch_fetch=%s defer_cpu_batch_stack=%s in_order=%s num_workers=%d",
                shuffle,
                sample_pack_ready,
                balance_by_volume,
                use_batch_fetch,
                defer_volume_stack,
                in_order,
                num_workers,
            )
        loader_dataset: Dataset | Subset = _DirectBatchDataset(dataset) if use_batch_fetch else dataset
        collate_fn = _identity_batch if use_batch_fetch else make_collate(
            int(pad_to_max_volumes) if pad_to_max_volumes else None,
            defer_volume_stack=defer_volume_stack,
            collect_diagnostics=timing_probe,
        )
        loader_kwargs: dict[str, Any] = {
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": bool(self.training_cfg.get("pin_memory", self.device.type == "cuda"))
            and self.device.type == "cuda",
            # 关键：每个 worker 启动时锁线程数为 1，避免 BLAS/numpy 多线程
            # 在多 worker 场景下相互抢 CPU 导致 IO 抖到几百 ms。
            "worker_init_fn": dataloader_worker_init,
        }
        if use_batch_fetch:
            index_sampler = sampler
            if index_sampler is None:
                index_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            loader_kwargs.update(
                {
                    "batch_size": None,
                    "shuffle": False,
                    "sampler": _EpochAwareBatchSampler(index_sampler, batch_size=batch_size, drop_last=drop_last),
                }
            )
        else:
            loader_kwargs.update(
                {
                    "batch_size": batch_size,
                    "shuffle": False if sampler is not None else shuffle,
                    "sampler": sampler,
                    "drop_last": drop_last,
                }
            )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(self.training_cfg.get("persistent_workers", True))
            prefetch_factor = self.training_cfg.get("prefetch_factor")
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            multiprocessing_context = self.training_cfg.get("multiprocessing_context")
            if multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = mp.get_context(str(multiprocessing_context))
            if _DATALOADER_SUPPORTS_IN_ORDER and in_order is not None:
                loader_kwargs["in_order"] = bool(in_order)
        return DataLoader(loader_dataset, **loader_kwargs)

    def _iter_device_batches(self, loader):
        if self.device.type == "cuda" and bool(self.training_cfg.get("cuda_prefetch", True)):
            return CudaBatchPrefetcher(loader, self.device, skip_keys=self._device_skip_keys)
        return (move_batch_to_device(batch, self.device, self._device_skip_keys) for batch in loader)

    def _log_split_summary(self, stage: str, split_cfg, subsets: dict[str, Subset]) -> None:
        sizes = {name: len(subset) for name, subset in subsets.items()}
        self.logger.info(
            "[%s] split_strategy mode=%s sizes=%s payload=%s",
            stage,
            split_cfg.mode,
            sizes,
            split_cfg.raw,
        )

    def _make_progress_bar(self, loader, *, desc: str):
        if not (bool(self.training_cfg.get("progress_bar", False)) and tqdm is not None and self.is_main):
            return loader
        return tqdm(
            loader,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
            smoothing=0.05,
        )

    @staticmethod
    def _format_metrics_inline(metrics: dict[str, float], preferred: tuple[str, ...] = ("loss", "lr")) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for key in preferred:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={BreastDCEMoEPINNSolver._format_metric_value(key, float(value))}")
                seen.add(key)
        for key, value in metrics.items():
            if key in seen or not isinstance(value, (int, float)):
                continue
            parts.append(f"{key}={BreastDCEMoEPINNSolver._format_metric_value(key, float(value))}")
        return " ".join(parts) if parts else "-"

    @staticmethod
    def _format_metric_value(key: str, value: float) -> str:
        abs_value = abs(float(value))
        key_lower = str(key).lower()
        if key_lower == "lr" or key_lower.endswith("/lr") or (0.0 < abs_value < 1e-3) or abs_value >= 1e4:
            return f"{float(value):.6e}"
        return f"{float(value):.4f}"

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        total_loss = torch.zeros((), device=self.device)
        steps = 0
        last_loss_dict: dict[str, torch.Tensor] = {}
        log_interval = max(1, int(self.training_cfg.get("log_interval", 20)))
        progress_bar = bool(self.training_cfg.get("progress_bar", False))
        timing_enabled = bool(self.training_cfg.get("timing_probe", False)) and self.is_main_or_distributed()
        cuda_timing = timing_enabled and self.device.type == "cuda"
        wait_ms_acc = 0.0
        fwd_ms_acc = 0.0
        bwd_ms_acc = 0.0
        opt_ms_acc = 0.0
        timing_steps = 0
        progress_update_interval = max(1, min(log_interval, 10))
        running_loss = 0.0

        host_iterator = self._make_progress_bar(loader, desc=f"{self.mode} epoch {epoch}")
        iterator = self._iter_device_batches(host_iterator)
        wait_start = time.perf_counter() if timing_enabled else 0.0
        for step, batch in enumerate(iterator, start=1):
            if cuda_timing:
                torch.cuda.current_stream(self.device).synchronize()
            wait_ms = (time.perf_counter() - wait_start) * 1000.0 if timing_enabled else 0.0

            fwd_start = time.perf_counter() if timing_enabled else 0.0
            self.optimizer.zero_grad(set_to_none=True)
            if self.gpu_augment is not None and self.model.training:
                batch = self.gpu_augment(batch)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch, mode=self.mode)
                loss, loss_dict = self._compute_loss(outputs, batch)
            if cuda_timing:
                torch.cuda.current_stream(self.device).synchronize()
            fwd_ms = (time.perf_counter() - fwd_start) * 1000.0 if timing_enabled else 0.0

            bwd_start = time.perf_counter() if timing_enabled else 0.0
            self.scaler.scale(loss).backward()
            if cuda_timing:
                torch.cuda.current_stream(self.device).synchronize()
            bwd_ms = (time.perf_counter() - bwd_start) * 1000.0 if timing_enabled else 0.0

            opt_start = time.perf_counter() if timing_enabled else 0.0
            grad_clip = self.training_cfg.get("grad_clip_norm", 1.0)
            if grad_clip is not None and float(grad_clip) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._unwrapped_model().parameters(), float(grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if cuda_timing:
                torch.cuda.current_stream(self.device).synchronize()
            opt_ms = (time.perf_counter() - opt_start) * 1000.0 if timing_enabled else 0.0

            loss_detached = loss.detach()
            current_loss = float(loss_detached.item())
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            running_loss += current_loss
            total_loss = total_loss + loss_detached.float()
            steps += 1
            last_loss_dict = {key: value.detach() for key, value in loss_dict.items() if torch.is_tensor(value)}

            if timing_enabled:
                wait_ms_acc += wait_ms
                fwd_ms_acc += fwd_ms
                bwd_ms_acc += bwd_ms
                opt_ms_acc += opt_ms
                timing_steps += 1
                if step % log_interval == 0:
                    self._emit_step_timing(
                        epoch=epoch,
                        step=step,
                        loss=current_loss,
                        wait_ms=wait_ms_acc / timing_steps,
                        fwd_ms=fwd_ms_acc / timing_steps,
                        bwd_ms=bwd_ms_acc / timing_steps,
                        opt_ms=opt_ms_acc / timing_steps,
                    )
                    dbg = batch.get("_dbg_worker") if isinstance(batch, dict) else None
                    if dbg and self.is_main:
                        self.logger.info(
                            "[worker] n_vol=%d load=%.1fms torch=%.1fms getitem_total=%.1fms (max_sample=%.1fms) collate=%.1fms (alloc=%.1f meta=%.1f)",
                            dbg.get("n_vol", 0),
                            dbg.get("load_ms", 0.0),
                            dbg.get("torch_ms", 0.0),
                            dbg.get("total_ms", 0.0),
                            dbg.get("max_ms", 0.0),
                            dbg.get("collate_ms", 0.0),
                            dbg.get("alloc_ms", 0.0),
                            dbg.get("meta_ms", 0.0),
                        )
                    wait_ms_acc = fwd_ms_acc = bwd_ms_acc = opt_ms_acc = 0.0
                    timing_steps = 0
                wait_start = time.perf_counter()

            if progress_bar and tqdm is not None and self.is_main and hasattr(host_iterator, "set_postfix") and (step == 1 or step % progress_update_interval == 0):
                postfix = {"loss": f"{current_loss:.4f}", "avg": f"{running_loss / max(step, 1):.4f}", "lr": f"{current_lr:.2e}"}
                if timing_enabled:
                    postfix["wait"] = f"{wait_ms:.0f}ms"
                host_iterator.set_postfix(postfix, refresh=False)

        if self.scheduler is not None:
            self.scheduler.step()
        local_avg = total_loss / max(steps, 1)
        global_avg = reduce_scalar(local_avg, op="mean")
        metrics = {"loss": global_avg, "lr": self.optimizer.param_groups[0]["lr"]}
        metrics.update({f"loss/{key}": float(value.detach().item()) for key, value in last_loss_dict.items() if torch.is_tensor(value)})
        return metrics

    def is_main_or_distributed(self) -> bool:
        return True

    def _emit_step_timing(
        self,
        *,
        epoch: int,
        step: int,
        loss: float,
        wait_ms: float,
        fwd_ms: float,
        bwd_ms: float,
        opt_ms: float,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            local = torch.tensor([wait_ms, fwd_ms, bwd_ms, opt_ms], device=self.device, dtype=torch.float32)
            max_t = local.clone()
            min_t = local.clone()
            sum_t = local.clone()
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            avg_t = sum_t / float(self.world_size)
            if not self.is_main:
                return
            avg = avg_t.tolist()
            spread = (max_t - min_t).tolist()
            self.logger.info(
                "[timing] epoch=%d step=%d loss=%.4f | wait=%.1fms (Δ%.1f) fwd=%.1fms (Δ%.1f) bwd=%.1fms (Δ%.1f) opt=%.1fms (Δ%.1f) | step≈%.1fms",
                epoch, step, loss,
                avg[0], spread[0],
                avg[1], spread[1],
                avg[2], spread[2],
                avg[3], spread[3],
                avg[0] + avg[1] + avg[2] + avg[3],
            )
        else:
            self.logger.info(
                "[timing] epoch=%d step=%d loss=%.4f | wait=%.1fms fwd=%.1fms bwd=%.1fms opt=%.1fms | step≈%.1fms",
                epoch, step, loss,
                wait_ms, fwd_ms, bwd_ms, opt_ms,
                wait_ms + fwd_ms + bwd_ms + opt_ms,
            )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage: str) -> dict[str, float]:
        self.model.eval()
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        total_loss = torch.zeros((), device=self.device)
        steps = 0
        host_iterator = self._make_progress_bar(loader, desc=f"{stage} epoch {epoch}")
        iterator = self._iter_device_batches(host_iterator)
        for batch in iterator:
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                outputs = self.model(batch, mode="infer" if self.mode == "infer" else self.mode)
                loss, _ = self._compute_loss(outputs, batch)
            total_loss = total_loss + loss.detach().float()
            steps += 1
        local_avg = total_loss / max(steps, 1)
        global_avg = reduce_scalar(local_avg, op="mean")
        return {"loss": global_avg}

    def _compute_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.mode == "pretrain":
            loss = outputs["losses"].get("total")
            if loss is None:
                loss = sum(value for value in outputs["losses"].values() if torch.is_tensor(value))
            return loss, outputs["losses"]
        labels = batch["label_values"] if "label_values" in batch else batch["labels"]
        label_mask = batch["label_mask_values"] if "label_mask_values" in batch else batch["label_mask"]
        supervised, supervised_losses = self.criterion(outputs["predictions"], labels, label_mask)
        aux = (
            0.01 * outputs["losses"].get("moe_balance", supervised * 0.0)
            + 0.01 * outputs["losses"].get("attention_entropy", supervised * 0.0)
            + 0.1 * outputs["losses"].get("pinn_signal", supervised * 0.0)
        )
        param_anchor = outputs["losses"].get("_param_anchor")
        if torch.is_tensor(param_anchor):
            aux = aux + param_anchor
        losses = dict(supervised_losses)
        losses["auxiliary"] = aux
        losses.update(outputs["losses"])
        return supervised + aux, losses

    def _build_optimizer(self):
        optimizer_kwargs: dict[str, Any] = {
            "lr": 0.0 if self.mode == "infer" else float(self.training_cfg.get("lr", 1e-4 if self.mode == "pretrain" else 5e-5)),
            "weight_decay": float(self.training_cfg.get("weight_decay", 1e-4)),
        }
        if self.device.type == "cuda" and bool(self.training_cfg.get("optimizer_fused", False)):
            optimizer_kwargs["fused"] = True
        elif bool(self.training_cfg.get("optimizer_foreach", True)):
            optimizer_kwargs["foreach"] = True

        params = list(self._unwrapped_model().parameters())
        if self.mode == "infer":
            optimizer_kwargs["weight_decay"] = 0.0
        try:
            return torch.optim.AdamW(params, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer_kwargs.pop("foreach", None)
            return torch.optim.AdamW(params, **optimizer_kwargs)

    def _build_scheduler(self):
        if self.mode == "infer":
            return None
        scheduler_cfg = dict(self.training_cfg.get("scheduler", {}) or {})
        scheduler_name = str(
            scheduler_cfg.get("name", self.training_cfg.get("scheduler_name", "cosine"))
        ).strip().lower()

        if scheduler_name in {"", "none", "disabled"}:
            return None

        if scheduler_name == "step":
            step_size = max(
                1,
                int(scheduler_cfg.get("step_size", self.training_cfg.get("scheduler_step_size", 50))),
            )
            gamma = float(scheduler_cfg.get("gamma", self.training_cfg.get("scheduler_gamma", 0.995)))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(scheduler_cfg.get("t_max", self.training_cfg.get("epochs", 100)))),
            )

        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. Expected one of: step, cosine, none."
        )

    def _log_optimizer_schedule(self) -> None:
        if not self.is_main or self.mode == "infer":
            return
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        if self.scheduler is None:
            self.logger.info(
                "[optimizer] initial_lr=%s scheduler=none",
                self._format_metric_value("lr", current_lr),
            )
            return

        details: list[str] = []
        if hasattr(self.scheduler, "step_size"):
            details.append(f"step_size={int(self.scheduler.step_size)}")
        if hasattr(self.scheduler, "gamma"):
            details.append(f"gamma={float(self.scheduler.gamma):.6f}")
        if hasattr(self.scheduler, "T_max"):
            details.append(f"T_max={int(self.scheduler.T_max)}")
        if hasattr(self.scheduler, "eta_min"):
            details.append(f"eta_min={self._format_metric_value('lr', float(self.scheduler.eta_min))}")
        detail_text = ", ".join(details) if details else "default"
        self.logger.info(
            "[optimizer] initial_lr=%s scheduler=%s(%s)",
            self._format_metric_value("lr", current_lr),
            self.scheduler.__class__.__name__,
            detail_text,
        )

    def _load_initial_checkpoint(self) -> None:
        checkpoint = (
            self.training_cfg.get("resume_path")
            or self.training_cfg.get("pretrained_checkpoint")
            or self.training_cfg.get("checkpoint")
            or self.config.get("inference", {}).get("checkpoint")
        )
        if checkpoint:
            payload = load_checkpoint(
                checkpoint,
                self._unwrapped_model(),
                self.optimizer if self.training_cfg.get("resume_path") else None,
                self.scheduler if self.training_cfg.get("resume_path") else None,
                strict=False,
            )
            if self.training_cfg.get("resume_path"):
                self.start_epoch = int(payload.get("epoch", 0)) + 1
                self.best_metric = payload.get("best_metric")

    def _unwrapped_model(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _pretrain_visualization_enabled(self) -> bool:
        if self.mode != "pretrain":
            return False
        return bool(self.visualization_cfg.get("enabled", False))

    def _should_export_pretrain_visualizations(self, epoch: int) -> bool:
        if not self._pretrain_visualization_enabled():
            return False
        interval = int(self.visualization_cfg.get("interval", 0))
        if interval <= 0:
            return False
        return epoch % interval == 0

    def _poll_visualization_process(self) -> None:
        if self._visualization_process is None:
            return
        return_code = self._visualization_process.poll()
        if return_code is None:
            return
        if return_code == 0:
            self.logger.info("[visualization] background export pid=%d completed", self._visualization_process.pid)
        else:
            self.logger.warning(
                "[visualization] background export pid=%d exited with code %d",
                self._visualization_process.pid,
                return_code,
            )
        self._visualization_process = None

    def _launch_pretrain_visualization(self, checkpoint_path: Path, epoch: int) -> None:
        self._poll_visualization_process()
        if self._visualization_process is not None:
            self.logger.warning(
                "[visualization] epoch=%d skipped because previous export pid=%d is still running",
                epoch,
                self._visualization_process.pid,
            )
            return

        checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)
        output_dir = (self.output_dir / "visualizations" / f"epoch_{epoch:04d}").resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "export.log"
        cmd = [
            sys.executable,
            "-m",
            "src.breast_mri_ai.breast_dce_moe_pinn_foundation.scripts.export_pretrain_visualizations",
            "--checkpoint",
            str(checkpoint_path),
            "--epoch",
            str(epoch),
            "--output-dir",
            str(output_dir),
            "--preview-batch-size",
            str(int(self.visualization_cfg.get("preview_batch_size", 1))),
            "--max-visualized-volumes",
            str(int(self.visualization_cfg.get("max_visualized_volumes", 4))),
        ]
        preview_indices = self.visualization_cfg.get("preview_indices")
        if isinstance(preview_indices, (list, tuple)) and preview_indices:
            cmd.extend(["--preview-indices", ",".join(str(int(index)) for index in preview_indices)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MPLBACKEND"] = "Agg"
        repo_root = Path(__file__).resolve().parents[3]
        handle = log_path.open("a", encoding="utf-8")
        try:
            self._visualization_process = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            handle.close()
        self.logger.info(
            "[visualization] launched async CPU export for epoch=%d pid=%d output_dir=%s",
            epoch,
            self._visualization_process.pid,
            output_dir,
        )

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
        evaluated: bool,
    ) -> None:
        barrier()
        if not self.is_main:
            return
        score = -float(eval_metrics.get("loss", train_metrics.get("loss", 0.0)))
        is_best = evaluated and (self.best_metric is None or score > self.best_metric)
        if is_best:
            self.best_metric = score
        best_loss = -float(self.best_metric) if self.best_metric is not None else float("nan")
        self.logger.info(
            "epoch=%d | train[%s] | eval[%s] | evaluated=%s | best_loss=%.4f",
            epoch,
            self._format_metrics_inline(train_metrics, preferred=("loss", "lr")),
            self._format_metrics_inline(eval_metrics, preferred=("loss",)),
            evaluated,
            best_loss,
        )
        append_history_metrics(self.output_dir / "logs", epoch, "train", train_metrics)
        if eval_metrics:
            append_history_metrics(self.output_dir / "logs", epoch, "eval", eval_metrics)
        self._poll_visualization_process()

        if not bool(self.training_cfg.get("save_checkpoints", True)):
            return

        checkpoint_interval = int(self.training_cfg.get("checkpoint_interval", 1))
        should_save_periodic = checkpoint_interval > 0 and (
            epoch % checkpoint_interval == 0
            or epoch >= int(self.training_cfg.get("epochs", epoch))
        )
        checkpoint_dir = self.output_dir / "checkpoints"
        periodic_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        if bool(self.training_cfg.get("save_last_checkpoint", True)):
            save_checkpoint(
                checkpoint_dir / "last.pth",
                self._unwrapped_model(),
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config,
            )
        if should_save_periodic:
            save_checkpoint(
                periodic_checkpoint_path,
                self._unwrapped_model(),
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config,
            )
        if is_best and bool(self.training_cfg.get("save_best_checkpoint", True)):
            save_checkpoint(
                checkpoint_dir / "best.pth",
                self._unwrapped_model(),
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config,
            )
        if self._should_export_pretrain_visualizations(epoch):
            checkpoint_path = periodic_checkpoint_path
            if not should_save_periodic:
                checkpoint_path = checkpoint_dir / f"visualization_epoch_{epoch}.pth"
                save_checkpoint(
                    checkpoint_path,
                    self._unwrapped_model(),
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_metric,
                    self.config,
                )
            self._launch_pretrain_visualization(checkpoint_path, epoch)
