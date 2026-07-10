from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def default_device() -> torch.device:
    """Return the device for the *current* process.

    In a DDP run, ``LOCAL_RANK`` is set per worker (we do this in
    :func:`init_distributed`) so each rank picks its own GPU. In single-process
    mode we just take cuda:0 if available.
    """

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def move_batch_to_device(batch, device: torch.device, skip_keys: set[str] | None = None):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        result = {}
        deferred_volumes = batch.get("_volume_tensors") if batch.get("_defer_volume_stack") else None
        defer_target_volumes = batch.get("_defer_target_volumes") if batch.get("_defer_volume_stack") else None
        for key, value in batch.items():
            if key in {"_volume_tensors", "_defer_target_volumes", "_defer_volume_stack"}:
                continue
            result[key] = value if skip_keys and key in skip_keys else move_batch_to_device(value, device, skip_keys)
        if deferred_volumes is not None:
            result["volumes"] = _stack_deferred_volumes(
                deferred_volumes,
                device,
                target_volumes=defer_target_volumes,
            )
        return result
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, skip_keys) for value in batch]
    return batch


def _stack_deferred_volumes(
    volume_tensors: list[torch.Tensor],
    device: torch.device,
    *,
    target_volumes: int | None = None,
) -> torch.Tensor:
    """Materialize deferred sample tensors directly on the target device.

    The high-throughput DataLoader path intentionally avoids a giant CPU-side
    ``torch.stack`` in worker processes. When CUDA prefetching is enabled this
    function runs on the side stream, allocates the final GPU batch once, pads
    compact ``[n_vol, ...]`` samples to a shared ``V`` on-device, and copies
    the pinned sample tensors into it asynchronously.
    """

    if not volume_tensors:
        return torch.empty((0,), device=device)
    first = volume_tensors[0]
    tail_shape = tuple(first.shape[1:])
    if target_volumes is None:
        target_volumes = max(int(tensor.shape[0]) for tensor in volume_tensors)
    else:
        target_volumes = max(1, int(target_volumes))
    if device.type == "cuda":
        out = torch.zeros(
            (len(volume_tensors), target_volumes, *tail_shape),
            dtype=first.dtype,
            device=device,
        )
        for row, tensor in enumerate(volume_tensors):
            limit = min(int(tensor.shape[0]), target_volumes)
            if limit > 0:
                out[row, :limit].copy_(tensor[:limit], non_blocking=True)
        return out
    out = torch.zeros(
        (len(volume_tensors), target_volumes, *tail_shape),
        dtype=first.dtype,
    )
    for row, tensor in enumerate(volume_tensors):
        limit = min(int(tensor.shape[0]), target_volumes)
        if limit > 0:
            out[row, :limit].copy_(tensor[:limit], non_blocking=False)
    return out.to(device, non_blocking=True)


def _record_stream(batch, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(batch):
        if batch.device.type == "cuda":
            batch.record_stream(stream)
        return
    if isinstance(batch, dict):
        for value in batch.values():
            _record_stream(value, stream)
        return
    if isinstance(batch, list):
        for value in batch:
            _record_stream(value, stream)


class CudaBatchPrefetcher:
    """Move the next DataLoader batch to CUDA on a side stream."""

    def __init__(self, loader, device: torch.device, skip_keys: set[str] | None = None) -> None:
        self.loader = loader
        self.device = device
        self.skip_keys = skip_keys or set()

    def __iter__(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            for batch in self.loader:
                yield move_batch_to_device(batch, self.device, self.skip_keys)
            return

        with torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device)
            iterator = iter(self.loader)

            def preload():
                try:
                    raw_batch = next(iterator)
                except StopIteration:
                    return None
                with torch.cuda.stream(stream):
                    return move_batch_to_device(raw_batch, self.device, self.skip_keys)

            next_batch = preload()
            while next_batch is not None:
                current_stream = torch.cuda.current_stream(self.device)
                current_stream.wait_stream(stream)
                batch = next_batch
                _record_stream(batch, current_stream)
                next_batch = preload()
                yield batch


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def init_distributed(
    rank: int,
    world_size: int,
    *,
    backend: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """Initialise ``torch.distributed`` for one DDP worker.

    The caller (``torch.multiprocessing.spawn`` target) must call this before
    creating models / loaders. Sets ``LOCAL_RANK`` so :func:`default_device`
    can pick the right GPU for this process.
    """

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_scalar(value: float | torch.Tensor, op: str = "mean") -> float:
    """All-reduce a python scalar across ranks. Used for averaging losses."""

    if torch.is_tensor(value):
        tensor = value.detach().float()
        if tensor.ndim != 0:
            tensor = tensor.mean()
        if is_distributed() and tensor.device.type == "cpu":
            tensor = tensor.to(default_device())
    else:
        tensor = torch.tensor(float(value), device=default_device() if is_distributed() else "cpu")

    if not is_distributed():
        return float(tensor.item())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= float(get_world_size())
    return float(tensor.item())


def barrier() -> None:
    if is_distributed():
        dist.barrier()
