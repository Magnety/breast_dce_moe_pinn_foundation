from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
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

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


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
        self.output_dir = Path(config.get("output", {}).get("root", f"outputs/breast_dce_moe_pinn/{mode}"))
        self.logger = build_logger(self.output_dir / "logs", f"{mode}-rank{self.rank}")
        self.training_cfg = config.get("training", {})
        self.performance_cfg = config.get("performance", {})
        self._device_skip_keys = {"labels", "label_mask"}
        self.model = BreastDCEMoEPINNModel(**config.get("model", {})).to(self.device)
        self._maybe_compile_model()
        self.criterion = MaskedMultitaskLoss(config.get("loss", {}).get("task_weights"))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
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
        mode = str(self.performance_cfg.get("torch_compile_mode", "default"))
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
        # CPU 端 augmentation 强制关闭：在 dataloader worker 里跑 numpy aug 会
        # 把 GPU 饿死（randn 单次 ~30ms × N volumes/sample）。改在 GPU 上对整个
        # batch 做，速度快两个数量级。GPU augment 在 train_one_epoch 里调用。
        cpu_augment = False
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
        )

    def _make_loader(self, dataset: Dataset | Subset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise ValueError("Cannot build DataLoader from None dataset.")
        sampler = None
        drop_last = bool(shuffle and self.training_cfg.get("drop_last", False))
        if is_distributed():
            # DistributedSampler shards the dataset across ranks. We disable the
            # builtin DataLoader shuffle and let the sampler control it via
            # ``set_epoch`` (called from ``train_one_epoch``).
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        num_workers = int(self.training_cfg.get("num_workers", 2))
        # 固定 V 维度让所有 rank 跑等量 patch_embed/encoder，避免 straggler。
        pad_to_max_volumes = self.config.get("data", {}).get("max_volumes")
        collate_fn = make_collate(int(pad_to_max_volumes) if pad_to_max_volumes else None)
        loader_kwargs: dict[str, Any] = {
            "batch_size": int(
                self.training_cfg.get("batch_size", self.config.get("inference", {}).get("batch_size", 1))
            ),
            "shuffle": False if sampler is not None else shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": bool(self.training_cfg.get("pin_memory", self.device.type == "cuda"))
            and self.device.type == "cuda",
            "drop_last": drop_last,
            # 关键：每个 worker 启动时锁线程数为 1，避免 BLAS/numpy 多线程
            # 在多 worker 场景下相互抢 CPU 导致 IO 抖到几百 ms。
            "worker_init_fn": dataloader_worker_init,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(self.training_cfg.get("persistent_workers", True))
            prefetch_factor = self.training_cfg.get("prefetch_factor")
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            multiprocessing_context = self.training_cfg.get("multiprocessing_context")
            if multiprocessing_context:
                loader_kwargs["multiprocessing_context"] = mp.get_context(str(multiprocessing_context))
        return DataLoader(dataset, **loader_kwargs)

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

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        if isinstance(getattr(loader, "sampler", None), DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss = torch.zeros((), device=self.device)
        steps = 0
        last_loss_dict: dict[str, torch.Tensor] = {}
        log_interval = max(1, int(self.training_cfg.get("log_interval", 20)))
        progress_bar = bool(self.training_cfg.get("progress_bar", False))
        # Step-time 探针：每 ``log_interval`` 步打印一次 batch-wait / forward /
        # backward / optim 的平均耗时（毫秒），并在 DDP 下顺手报告 rank 间最大值
        # 与最小值，diff 大说明存在 straggler。开 ``cuda_prefetch`` 后 batch-wait
        # 是侧 stream 的等待时间，正常应接近 0；若持续 > 50 ms，说明 dataloader
        # 跟不上，需要加 num_workers 或检查磁盘/aug。
        timing_enabled = bool(self.training_cfg.get("timing_probe", True)) and self.is_main_or_distributed()
        cuda_timing = timing_enabled and self.device.type == "cuda"
        wait_ms_acc = 0.0
        fwd_ms_acc = 0.0
        bwd_ms_acc = 0.0
        opt_ms_acc = 0.0
        timing_steps = 0

        # Only show tqdm on the main rank to avoid garbled output.
        host_iterator = (
            tqdm(loader, desc=f"{self.mode} epoch {epoch}", ncols=110)
            if (progress_bar and tqdm is not None and self.is_main)
            else loader
        )
        iterator = self._iter_device_batches(host_iterator)
        wait_start = time.perf_counter() if timing_enabled else 0.0
        for step, batch in enumerate(iterator, start=1):
            if cuda_timing:
                # 把 prefetcher 的 side-stream copy 同步到默认 stream，确保下面
                # 测到的 forward 时间不再包含 H2D copy。
                torch.cuda.current_stream(self.device).synchronize()
            wait_ms = (time.perf_counter() - wait_start) * 1000.0 if timing_enabled else 0.0

            fwd_start = time.perf_counter() if timing_enabled else 0.0
            self.optimizer.zero_grad(set_to_none=True)
            # GPU 端 batch augmentation：必须只在 train 阶段做，且放在 forward
            # 之前。eval/infer 路径调用同一个 model.forward，所以这里 self.model.training
            # 已经为 True，再判一次 self.gpu_augment 是否就绪。
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
                torch.nn.utils.clip_grad_norm_(
                    self._unwrapped_model().parameters(),
                    float(grad_clip),
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if cuda_timing:
                torch.cuda.current_stream(self.device).synchronize()
            opt_ms = (time.perf_counter() - opt_start) * 1000.0 if timing_enabled else 0.0

            loss_detached = loss.detach()
            total_loss = total_loss + loss_detached.float()
            steps += 1
            last_loss_dict = {
                key: value.detach()
                for key, value in loss_dict.items()
                if torch.is_tensor(value)
            }
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
                        loss=float(loss_detached.item()),
                        wait_ms=wait_ms_acc / timing_steps,
                        fwd_ms=fwd_ms_acc / timing_steps,
                        bwd_ms=bwd_ms_acc / timing_steps,
                        opt_ms=opt_ms_acc / timing_steps,
                    )
                    # 诊断：打印 worker 端 __getitem__ + collate 的细分时间。
                    # 仅 main rank 打印当前 batch 的快照即可。
                    dbg = batch.get("_dbg_worker") if isinstance(batch, dict) else None
                    if dbg and self.is_main:
                        self.logger.info(
                            "[worker] n_vol=%d load=%.1fms torch=%.1fms "
                            "getitem_total=%.1fms (max_sample=%.1fms) "
                            "collate=%.1fms (alloc=%.1f meta=%.1f)",
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
            if (
                progress_bar
                and tqdm is not None
                and self.is_main
                and hasattr(host_iterator, "set_postfix")
                and step % log_interval == 0
            ):
                host_iterator.set_postfix({"loss": f"{float(loss_detached.item()):.4f}"})
        if self.scheduler is not None:
            self.scheduler.step()
        # Average loss across ranks so logs reflect the global epoch loss
        # rather than just rank 0's slice.
        local_avg = total_loss / max(steps, 1)
        global_avg = reduce_scalar(local_avg, op="mean")
        metrics = {"loss": global_avg, "lr": self.optimizer.param_groups[0]["lr"]}
        metrics.update(
            {
                f"loss/{key}": float(value.detach().item())
                for key, value in last_loss_dict.items()
                if torch.is_tensor(value)
            }
        )
        return metrics

    def is_main_or_distributed(self) -> bool:
        # 探针在所有 rank 上都测 —— 因为我们正是要看 rank 间差值；
        # 但只有 main rank 负责打印汇总。
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
        """单 rank：直接打印；多 rank：先 all_reduce 出 max/min 再由 rank0 打印。"""
        if dist.is_available() and dist.is_initialized():
            local = torch.tensor(
                [wait_ms, fwd_ms, bwd_ms, opt_ms],
                device=self.device,
                dtype=torch.float32,
            )
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
                "[timing] epoch=%d step=%d loss=%.4f | "
                "wait=%.1fms (Δ%.1f) fwd=%.1fms (Δ%.1f) bwd=%.1fms (Δ%.1f) opt=%.1fms (Δ%.1f) | "
                "step≈%.1fms",
                epoch, step, loss,
                avg[0], spread[0],
                avg[1], spread[1],
                avg[2], spread[2],
                avg[3], spread[3],
                avg[0] + avg[1] + avg[2] + avg[3],
            )
        else:
            self.logger.info(
                "[timing] epoch=%d step=%d loss=%.4f | "
                "wait=%.1fms fwd=%.1fms bwd=%.1fms opt=%.1fms | step≈%.1fms",
                epoch, step, loss,
                wait_ms, fwd_ms, bwd_ms, opt_ms,
                wait_ms + fwd_ms + bwd_ms + opt_ms,
            )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, stage: str) -> dict[str, float]:
        self.model.eval()
        if isinstance(getattr(loader, "sampler", None), DistributedSampler):
            loader.sampler.set_epoch(epoch)
        total_loss = torch.zeros((), device=self.device)
        steps = 0
        progress_bar = bool(self.training_cfg.get("progress_bar", False))
        host_iterator = (
            tqdm(loader, desc=f"{stage} epoch {epoch}", ncols=110)
            if (progress_bar and tqdm is not None and self.is_main)
            else loader
        )
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
        # 把模型 forward 里挂出的 0 系数全参数 anchor 加进总 loss，
        # 保证 finetune 阶段任意 mask/缺模态下 DDP 看到的参数集都是全集。
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
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(self.training_cfg.get("epochs", 100))),
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

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
        evaluated: bool,
    ) -> None:
        # Make sure all ranks have finished the epoch before rank 0 writes
        # checkpoints — otherwise other ranks could already be running into
        # the next epoch and disturbing the saved tensors.
        barrier()
        if not self.is_main:
            return
        score = -float(eval_metrics.get("loss", train_metrics.get("loss", 0.0)))
        is_best = evaluated and (self.best_metric is None or score > self.best_metric)
        if is_best:
            self.best_metric = score
        self.logger.info(
            "epoch=%s train=%s eval=%s evaluated=%s best=%s",
            epoch,
            train_metrics,
            eval_metrics,
            evaluated,
            self.best_metric,
        )
        if not bool(self.training_cfg.get("save_checkpoints", True)):
            return

        checkpoint_interval = int(self.training_cfg.get("checkpoint_interval", 1))
        should_save_periodic = checkpoint_interval > 0 and (
            epoch % checkpoint_interval == 0
            or epoch >= int(self.training_cfg.get("epochs", epoch))
        )
        checkpoint_dir = self.output_dir / "checkpoints"
        if should_save_periodic:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                self._unwrapped_model(),
                self.optimizer,
                self.scheduler,
                epoch,
                self.best_metric,
                self.config,
            )
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
