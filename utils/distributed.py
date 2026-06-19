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
        return {
            key: value if skip_keys and key in skip_keys else move_batch_to_device(value, device, skip_keys)
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, skip_keys) for value in batch]
    return batch


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
