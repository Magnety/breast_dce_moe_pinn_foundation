"""Pretrain step-level benchmark.

跑少量 pretrain step 并打印 samples/sec / volumes/sec / step time，专门用来
判断"GPU 功耗低"是 dataloader 慢、模型 workload 太小、还是别的同步问题。
不要只看 GPU 功耗 —— batch_size=2 + 5090 不可能跑到 500W，必须看吞吐。

用法（单卡）：

    PYTHONPATH=src python -m breast_mri_ai.breast_dce_moe_pinn_foundation.scripts.benchmark_pretrain_step \\
        --config src/breast_mri_ai/breast_dce_moe_pinn_foundation/configs/pretrain_dce_moe_pinn.yaml \\
        --gpus 0 --batch-size 8 --num-workers 8 --steps 60 --warmup-steps 10

加 ``--fake-io`` 跑伪 IO 对照（dataset 直接返回 zeros），用来分离磁盘 IO 占比。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _add_repo_root_to_path() -> None:
    here = Path(__file__).resolve()
    # benchmark 脚本所在目录的上 4 层是项目 root（src 父目录）。
    repo_root = here.parents[5]
    candidate = repo_root / "src"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain step benchmark.")
    parser.add_argument("--config", type=Path, default=None, help="path to pretrain yaml")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids; only the first is used in benchmark")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--steps", type=int, default=60, help="how many steps to time")
    parser.add_argument("--warmup-steps", type=int, default=10, help="steps to skip before timing")
    parser.add_argument("--fake-io", action="store_true", help="set BREAST_FAKE_IO=1 to skip disk IO")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--timing-probe", action="store_true", help="enable solver-side cuda sync timing probe")
    parser.add_argument("--defer-cpu-batch-stack", action="store_true", help="enable experimental CUDA-side batch materialization")
    parser.add_argument("--volume-balanced-sampling", action="store_true", help="enable experimental DDP volume-balanced sampling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[0]
    if args.fake_io:
        os.environ["BREAST_FAKE_IO"] = "1"
    # 主进程 + 所有 worker 都锁单线程，避免 BLAS oversubscribe。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    _add_repo_root_to_path()

    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.solver import BreastDCEMoEPINNSolver
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.config import load_config
    from src.breast_mri_ai.breast_dce_moe_pinn_foundation.utils.seed import set_seed

    config_path = args.config or (
        Path(__file__).resolve().parents[1] / "configs" / "pretrain_dce_moe_pinn.yaml"
    )
    config = load_config(config_path)

    # CLI overrides。
    training_cfg = config.setdefault("training", {})
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        training_cfg["prefetch_factor"] = int(args.prefetch_factor)
    training_cfg["timing_probe"] = bool(args.timing_probe)
    if args.defer_cpu_batch_stack:
        training_cfg["defer_cpu_batch_stack"] = True
    if args.volume_balanced_sampling:
        training_cfg["volume_balanced_sampling"] = True
    training_cfg["progress_bar"] = False
    # benchmark 不要写 checkpoint 也不要做 eval。
    training_cfg["save_checkpoints"] = False
    training_cfg["eval_interval"] = 0
    training_cfg["log_interval"] = max(10, args.steps // 4)

    if args.max_volumes is not None:
        config.setdefault("data", {})["max_volumes"] = int(args.max_volumes)

    set_seed(int(config.get("seed", 2026)))
    solver = BreastDCEMoEPINNSolver(config, mode="pretrain")

    # 借 solver 的 _build_training_loaders 拿到一致的 loader 配置。
    train_loader, _ = solver._build_training_loaders()

    # 关键参数报告。
    data_cfg = config.get("data", {})
    bs = int(training_cfg.get("batch_size", 1))
    nw = int(training_cfg.get("num_workers", 0))
    mv = int(data_cfg.get("max_volumes", 0))
    target_shape = tuple(data_cfg.get("target_shape", (64, 128, 128)))
    print("=" * 70)
    print(f"benchmark config: batch_size={bs} num_workers={nw} max_volumes={mv}")
    print(f"  target_shape={target_shape}  fake_io={bool(args.fake_io)}  timing_probe={bool(args.timing_probe)}")
    print(f"  defer_cpu_batch_stack={bool(training_cfg.get('defer_cpu_batch_stack', False))}  volume_balanced_sampling={bool(training_cfg.get('volume_balanced_sampling', False))}")
    print(f"  steps={args.steps} warmup_steps={args.warmup_steps}")
    print("=" * 70, flush=True)

    # 不走 solver.run()（它会跑整 epoch 并写 checkpoint）；自己短跑训练循环。
    solver.model.train()
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    iterator = solver._iter_device_batches(train_loader)

    step_times: list[float] = []
    n_seen_samples = 0
    n_seen_volumes = 0
    last_t = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize(solver.device)
    last_t = time.perf_counter()

    for step_idx, batch in enumerate(iterator, start=1):
        solver.optimizer.zero_grad(set_to_none=True)
        if solver.gpu_augment is not None and solver.model.training:
            batch = solver.gpu_augment(batch)
        with torch.amp.autocast(device_type=solver.device.type, enabled=solver.amp_enabled):
            outputs = solver.model(batch, mode="pretrain")
            loss, _ = solver._compute_loss(outputs, batch)
        solver.scaler.scale(loss).backward()
        grad_clip = solver.training_cfg.get("grad_clip_norm", 1.0)
        if grad_clip is not None and float(grad_clip) > 0:
            solver.scaler.unscale_(solver.optimizer)
            torch.nn.utils.clip_grad_norm_(solver._unwrapped_model().parameters(), float(grad_clip))
        solver.scaler.step(solver.optimizer)
        solver.scaler.update()

        if torch.cuda.is_available():
            torch.cuda.synchronize(solver.device)
        now = time.perf_counter()
        elapsed = now - last_t
        last_t = now

        # 实际 batch 体积来自 batch tensor 形状，避免和 config drop_last 不一致。
        volumes_tensor = batch.get("volumes")
        if torch.is_tensor(volumes_tensor):
            cur_b, cur_v = volumes_tensor.shape[0], volumes_tensor.shape[1]
        else:
            cur_b, cur_v = bs, mv

        if step_idx > args.warmup_steps:
            step_times.append(elapsed * 1000.0)
            n_seen_samples += cur_b
            n_seen_volumes += cur_b * cur_v

        if step_idx % max(1, args.steps // 10) == 0:
            tag = "warmup" if step_idx <= args.warmup_steps else "timed"
            print(
                f"  [{tag}] step={step_idx} dt={elapsed * 1000.0:.1f}ms "
                f"loss={float(loss.detach().item()):.4f}",
                flush=True,
            )
        if step_idx >= args.warmup_steps + args.steps:
            break

    if not step_times:
        print("benchmark produced no timed steps; increase --steps or check the loader.", flush=True)
        return 1

    avg_ms = sum(step_times) / len(step_times)
    med_ms = statistics.median(step_times)
    timed_seconds = sum(step_times) / 1000.0
    samples_per_sec = n_seen_samples / max(timed_seconds, 1e-6)
    volumes_per_sec = n_seen_volumes / max(timed_seconds, 1e-6)

    print("=" * 70)
    print(f"timed_steps={len(step_times)}")
    print(f"step_time_ms: avg={avg_ms:.1f}  median={med_ms:.1f}  "
          f"min={min(step_times):.1f}  max={max(step_times):.1f}")
    print(f"throughput: samples/sec={samples_per_sec:.2f}  volumes/sec={volumes_per_sec:.2f}")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
