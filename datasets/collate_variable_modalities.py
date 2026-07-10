from __future__ import annotations

from typing import Any

import torch

MODALITY_VOCAB = {"unknown": 0, "DCE": 1, "T1": 2, "T2": 3, "DWI": 4, "ADC": 5, "derived": 6}
VISIT_VOCAB = {"unknown": 0, "pre_nact": 1, "pre": 1, "MRI1": 1, "mid_nact": 2, "MRI2": 2, "MRI3": 3, "pre_surgery": 4, "MRI4": 4, "followup": 5}
DATASET_VOCAB = {
    "unknown": 0,
    "ispy1": 1,
    "ispy2": 2,
    "duke": 3,
    "fujian_pCR": 4,
    "fujian_pcr": 4,
    "breast_mri_nact_pilot": 5,
    "tcga_brca": 6,
    "acrin_contralateral": 7,
    "advanced_mri_breast_lesions": 8,
}
TASKS = ("pCR", "HER2", "ER", "PR", "HR", "molecular_subtype", "survival_time", "survival_event", "survival")
TASK_TO_INDEX = {task: idx for idx, task in enumerate(TASKS)}


def variable_modalities_collate(
    samples: list[dict[str, Any]],
    pad_to_max_volumes: int | None = None,
    *,
    defer_volume_stack: bool = False,
    collect_diagnostics: bool = False,
) -> dict[str, Any]:
    """Pad variable volume lists into a trainable batch.

    Output image tensor shape is ``[B, V, C, D, H, W]``. Masks keep the true
    modality/phase availability so padded volumes never participate in loss.

    DDP 注意：当 ``pad_to_max_volumes`` 设为固定值时，所有 rank 的
    ``[B, V, ...]`` 形状一致，``valid_volume_indices`` 也固定为
    ``arange(B*V)``，让 patch_embed/encoder 在所有 rank 上做完全等量的 work，
    避免 straggler。padded slot 用 ``new_zeros`` 一次性建好，无需逐 sample
    copy zero tensor，host 端开销与不 pad 时基本一致。
    """
    if collect_diagnostics:
        import time as _time

        _c_start = _time.perf_counter()
    else:
        _time = None
        _c_start = 0.0

    batch_size = len(samples)
    volumes: torch.Tensor | None = None
    deferred_volume_tensors: list[torch.Tensor] | None = None
    # 新协议：sample_tensor 可以是紧凑版 ``[n_vol, 1, D, H, W]``（命中
    # sample-pack 时不再在 worker 里 pad/copy），也可以是旧的固定 ``V_buf``。
    if "sample_tensor" in samples[0]:
        sample_tensors = [sample["sample_tensor"] for sample in samples]
        first_shape = tuple(sample_tensors[0].shape)
        same_shape = all(tuple(tensor.shape) == first_shape for tensor in sample_tensors)
        same_tail_shape = all(tuple(tensor.shape[1:]) == first_shape[1:] for tensor in sample_tensors)
        real_max = max(int(tensor.shape[0]) for tensor in sample_tensors)
        if pad_to_max_volumes is not None and pad_to_max_volumes > 0:
            max_volumes = max(int(pad_to_max_volumes), real_max)
        else:
            max_volumes = real_max
        if defer_volume_stack and same_tail_shape:
            # CUDA 训练路径：不要在 worker 里构造巨大的 CPU batch tensor。
            # 即使不同 sample 的真实 volume 数不同，也把 padding 延迟到
            # CudaBatchPrefetcher 的目标设备侧一次性完成。
            deferred_volume_tensors = sample_tensors
            _t_alloc_ms = 0.0
        else:
            _t_alloc_start = _time.perf_counter() if collect_diagnostics else 0.0
            if same_shape and first_shape[0] == max_volumes:
                volumes = torch.stack(sample_tensors, dim=0)
            else:
                tail_shape = tuple(sample_tensors[0].shape[1:])
                padded_sample_tensors: list[torch.Tensor] = []
                for tensor in sample_tensors:
                    current_volumes = int(tensor.shape[0])
                    if current_volumes >= max_volumes:
                        padded_sample_tensors.append(tensor[:max_volumes])
                    else:
                        pad = tensor.new_zeros((max_volumes - current_volumes, *tail_shape))
                        padded_sample_tensors.append(torch.cat([tensor, pad], dim=0))
                volumes = torch.stack(padded_sample_tensors, dim=0)
            _t_alloc_ms = (
                (_time.perf_counter() - _t_alloc_start) * 1000.0
                if collect_diagnostics
                else 0.0
            )
        sample_volume_records = [sample.get("volume_records", []) for sample in samples]
    else:
        # 旧协议（向后兼容）：samples[i]["volumes"] 是 list-of-dict 含 tensor。
        real_max = max(len(sample["volumes"]) for sample in samples)
        if pad_to_max_volumes is not None and pad_to_max_volumes > 0:
            max_volumes = max(int(pad_to_max_volumes), real_max)
        else:
            max_volumes = real_max
        first_tensor = samples[0]["volumes"][0]["tensor"]
        volume_shape = tuple(first_tensor.shape)

        _t_alloc_start = _time.perf_counter() if collect_diagnostics else 0.0
        sample_tensors: list[torch.Tensor] = []
        for sample in samples:
            real_list = [record["tensor"] for record in sample["volumes"]][:max_volumes]
            if len(real_list) == max_volumes:
                sample_tensors.append(torch.stack(real_list, dim=0))
            else:
                stacked_real = torch.stack(real_list, dim=0)
                pad = stacked_real.new_zeros((max_volumes - len(real_list), *volume_shape))
                sample_tensors.append(torch.cat([stacked_real, pad], dim=0))
        volumes = torch.stack(sample_tensors, dim=0)
        _t_alloc_ms = (
            (_time.perf_counter() - _t_alloc_start) * 1000.0
            if collect_diagnostics
            else 0.0
        )
        sample_volume_records = [sample["volumes"] for sample in samples]

    _t_meta_start = _time.perf_counter() if collect_diagnostics else 0.0
    if "modality_id_tensor" in samples[0]:
        meta_same_shape = all(int(sample["modality_id_tensor"].shape[0]) == max_volumes for sample in samples)
        if meta_same_shape:
            modality_id = torch.stack([sample["modality_id_tensor"] for sample in samples], dim=0)
            phase_id = torch.stack([sample["phase_id_tensor"] for sample in samples], dim=0)
            relative_time = torch.stack([sample["relative_time_tensor"] for sample in samples], dim=0)
            modality_available_mask = torch.stack([sample["modality_available_mask_tensor"] for sample in samples], dim=0)
            phase_available_mask = torch.stack([sample["phase_available_mask_tensor"] for sample in samples], dim=0)
            temporal_dce_mask = torch.stack([sample["temporal_dce_mask_tensor"] for sample in samples], dim=0)
        else:
            modality_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
            phase_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
            relative_time = torch.zeros((batch_size, max_volumes), dtype=torch.float32)
            modality_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
            phase_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
            temporal_dce_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
            for row, sample in enumerate(samples):
                limit = min(int(sample["modality_id_tensor"].shape[0]), max_volumes)
                if limit > 0:
                    modality_id[row, :limit] = sample["modality_id_tensor"][:limit]
                    phase_id[row, :limit] = sample["phase_id_tensor"][:limit]
                    relative_time[row, :limit] = sample["relative_time_tensor"][:limit]
                    modality_available_mask[row, :limit] = sample["modality_available_mask_tensor"][:limit]
                    phase_available_mask[row, :limit] = sample["phase_available_mask_tensor"][:limit]
                    temporal_dce_mask[row, :limit] = sample["temporal_dce_mask_tensor"][:limit]
        label_values = torch.stack([sample["label_values_tensor"] for sample in samples], dim=0)
        label_mask_values = torch.stack([sample["label_mask_values_tensor"] for sample in samples], dim=0)
        visit_id = torch.tensor([int(sample["visit_id_value"]) for sample in samples], dtype=torch.long)
        dataset_id = torch.tensor([int(sample["dataset_id_value"]) for sample in samples], dtype=torch.long)
    else:
        modality_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
        phase_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
        relative_time = torch.zeros((batch_size, max_volumes), dtype=torch.float32)
        modality_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
        phase_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
        temporal_dce_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)

        for row, records in enumerate(sample_volume_records):
            for col, record in enumerate(records[:max_volumes]):
                modality_id[row, col] = MODALITY_VOCAB.get(str(record.get("modality", "unknown")), 0)
                phase = record.get("phase_index")
                phase_id[row, col] = 0 if phase is None else int(phase) + 1
                rel_time = record.get("relative_time")
                relative_time[row, col] = 0.0 if rel_time is None else float(rel_time)
                available = bool(record.get("available", 1))
                modality_available_mask[row, col] = available
                phase_available_mask[row, col] = available and phase is not None
                temporal_dce_mask[row, col] = available and _is_dce_temporal_record(record)
        label_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
        label_mask_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
        for row, sample in enumerate(samples):
            for task, col in TASK_TO_INDEX.items():
                label_values[row, col] = float(sample["labels"].get(task, -1.0))
                label_mask_values[row, col] = float(sample["label_mask"].get(task, 0.0))
        visit_id = torch.tensor([_lookup(VISIT_VOCAB, sample["visit_timepoint"]) for sample in samples], dtype=torch.long)
        dataset_id = torch.tensor([_lookup(DATASET_VOCAB, sample["dataset_id"]) for sample in samples], dtype=torch.long)
    _t_meta_ms = (
        (_time.perf_counter() - _t_meta_start) * 1000.0
        if collect_diagnostics
        else 0.0
    )
    valid_volume_indices = torch.arange(batch_size * max_volumes, dtype=torch.long)

    batch = {
        "modality_id": modality_id,
        "phase_id": phase_id,
        "relative_time": relative_time,
        "visit_id": visit_id,
        "dataset_id": dataset_id,
        "modality_available_mask": modality_available_mask,
        "phase_available_mask": phase_available_mask,
        "temporal_dce_mask": temporal_dce_mask,
        "valid_volume_indices": valid_volume_indices,
        "valid_volume_indices_is_dense": True,
        "label_values": label_values,
        "label_mask_values": label_mask_values,
        "label_tasks": TASKS,
    }
    if "patient_id" in samples[0]:
        batch["patient_id"] = [sample["patient_id"] for sample in samples]
        batch["sample_id"] = [sample["sample_id"] for sample in samples]
        batch["dataset_name"] = [sample["dataset_id"] for sample in samples]
        batch["visit_timepoint"] = [sample["visit_timepoint"] for sample in samples]
    if "volume_paths" in samples[0]:
        batch["volume_paths"] = [list(sample["volume_paths"]) for sample in samples]
    if volumes is not None:
        batch["volumes"] = volumes
    else:
        batch["_volume_tensors"] = deferred_volume_tensors or []
        batch["_defer_target_volumes"] = max_volumes
        batch["_defer_volume_stack"] = True

    if collect_diagnostics:
        _dbg_load_ms = sum(s.get("_dbg_load_ms", 0.0) for s in samples)
        _dbg_torch_ms = sum(s.get("_dbg_torch_ms", 0.0) for s in samples)
        _dbg_total_ms = sum(s.get("_dbg_total_ms", 0.0) for s in samples)
        _dbg_max_ms = max((s.get("_dbg_total_ms", 0.0) for s in samples), default=0.0)
        _dbg_n_vol = sum(s.get("_dbg_n_vol", 0) for s in samples)
        _dbg_collate_ms = (_time.perf_counter() - _c_start) * 1000.0
        batch["_dbg_worker"] = {
            "load_ms": _dbg_load_ms,
            "torch_ms": _dbg_torch_ms,
            "total_ms": _dbg_total_ms,
            "max_ms": _dbg_max_ms,
            "n_vol": _dbg_n_vol,
            "collate_ms": _dbg_collate_ms,
            "alloc_ms": _t_alloc_ms,
            "meta_ms": _t_meta_ms,
        }
    return batch


def _lookup(vocab: dict[str, int], value: Any) -> int:
    text = str(value)
    return vocab.get(text, vocab.get(text.lower(), 0))


def _is_dce_temporal_record(record: dict[str, Any]) -> bool:
    if str(record.get("modality", "unknown")) != "DCE":
        return False
    return record.get("phase_index") is not None or record.get("relative_time") is not None


def make_collate(
    pad_to_max_volumes: int | None,
    *,
    defer_volume_stack: bool = False,
    collect_diagnostics: bool = False,
):
    """Return a top-level-importable collate that pads V to a fixed value.

    多进程 DataLoader 在 spawn 模式下需要 collate 可被 pickle，所以这里走的是
    一个普通函数而不是闭包/lambda。
    """
    if (
        (pad_to_max_volumes is None or int(pad_to_max_volumes) <= 0)
        and not defer_volume_stack
        and not collect_diagnostics
    ):
        return variable_modalities_collate
    return _FixedVolumesCollate(
        int(pad_to_max_volumes) if pad_to_max_volumes else None,
        defer_volume_stack=defer_volume_stack,
        collect_diagnostics=collect_diagnostics,
    )


class _FixedVolumesCollate:
    __slots__ = ("collect_diagnostics", "defer_volume_stack", "pad_to_max_volumes")

    def __init__(
        self,
        pad_to_max_volumes: int | None,
        *,
        defer_volume_stack: bool,
        collect_diagnostics: bool,
    ) -> None:
        self.pad_to_max_volumes = int(pad_to_max_volumes) if pad_to_max_volumes else None
        self.defer_volume_stack = bool(defer_volume_stack)
        self.collect_diagnostics = bool(collect_diagnostics)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return variable_modalities_collate(
            samples,
            pad_to_max_volumes=self.pad_to_max_volumes,
            defer_volume_stack=self.defer_volume_stack,
            collect_diagnostics=self.collect_diagnostics,
        )
