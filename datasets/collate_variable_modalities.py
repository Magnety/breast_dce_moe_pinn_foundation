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
    import time as _time
    _c_start = _time.perf_counter()

    batch_size = len(samples)
    # 新协议：每个 sample 已经是 (V_buf, 1, D, H, W) 的 sample_tensor，
    # V_buf 在 dataset 端就 pad 到 ``data.max_volumes``，所以这里 stack 一次即可。
    if "sample_tensor" in samples[0]:
        _t_alloc_start = _time.perf_counter()
        # 简单 stack：每个 sample 已经是 (V_buf, 1, D, H, W) torch.Tensor，
        # torch.stack 是连续 memcpy 的 C++ 实现，比 SHM 池循环更快也更安全。
        volumes = torch.stack([sample["sample_tensor"] for sample in samples], dim=0)
        _t_alloc_ms = (_time.perf_counter() - _t_alloc_start) * 1000.0
        max_volumes = volumes.shape[1]
        sample_volume_records = [sample["volume_records"] for sample in samples]
    else:
        # 旧协议（向后兼容）：samples[i]["volumes"] 是 list-of-dict 含 tensor。
        real_max = max(len(sample["volumes"]) for sample in samples)
        if pad_to_max_volumes is not None and pad_to_max_volumes > 0:
            max_volumes = max(int(pad_to_max_volumes), real_max)
        else:
            max_volumes = real_max
        first_tensor = samples[0]["volumes"][0]["tensor"]
        volume_shape = tuple(first_tensor.shape)

        _t_alloc_start = _time.perf_counter()
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
        _t_alloc_ms = (_time.perf_counter() - _t_alloc_start) * 1000.0
        sample_volume_records = [sample["volumes"] for sample in samples]

    _t_meta_start = _time.perf_counter()
    modality_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
    phase_id = torch.zeros((batch_size, max_volumes), dtype=torch.long)
    relative_time = torch.zeros((batch_size, max_volumes), dtype=torch.float32)
    modality_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
    phase_available_mask = torch.zeros((batch_size, max_volumes), dtype=torch.bool)
    volume_paths: list[list[str]] = []

    for row, records in enumerate(sample_volume_records):
        paths = []
        for col, record in enumerate(records[:max_volumes]):
            modality_id[row, col] = MODALITY_VOCAB.get(str(record.get("modality", "unknown")), 0)
            phase = record.get("phase_index")
            phase_id[row, col] = 0 if phase is None else int(phase) + 1
            rel_time = record.get("relative_time")
            relative_time[row, col] = 0.0 if rel_time is None else float(rel_time)
            available = bool(record.get("available", 1))
            modality_available_mask[row, col] = available
            phase_available_mask[row, col] = available and phase is not None
            paths.append(str(record.get("path", "")))
        for _ in range(len(records[:max_volumes]), max_volumes):
            paths.append("")
        volume_paths.append(paths)
    _t_meta_ms = (_time.perf_counter() - _t_meta_start) * 1000.0
    valid_volume_indices = torch.arange(batch_size * max_volumes, dtype=torch.long)

    label_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
    label_mask_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
    for row, sample in enumerate(samples):
        for task, col in TASK_TO_INDEX.items():
            label_values[row, col] = float(sample["labels"].get(task, -1.0))
            label_mask_values[row, col] = float(sample["label_mask"].get(task, 0.0))
    labels = {task: label_values[:, col] for task, col in TASK_TO_INDEX.items()}
    label_mask = {task: label_mask_values[:, col] for task, col in TASK_TO_INDEX.items()}
    visit_id = torch.tensor([_lookup(VISIT_VOCAB, sample["visit_timepoint"]) for sample in samples], dtype=torch.long)
    dataset_id = torch.tensor([_lookup(DATASET_VOCAB, sample["dataset_id"]) for sample in samples], dtype=torch.long)

    # 诊断：聚合 worker 端 __getitem__ 各步骤耗时；找出 batch 内最慢 sample。
    _dbg_load_ms = sum(s.get("_dbg_load_ms", 0.0) for s in samples)
    _dbg_torch_ms = sum(s.get("_dbg_torch_ms", 0.0) for s in samples)
    _dbg_total_ms = sum(s.get("_dbg_total_ms", 0.0) for s in samples)
    _dbg_max_ms = max((s.get("_dbg_total_ms", 0.0) for s in samples), default=0.0)
    _dbg_n_vol = sum(s.get("_dbg_n_vol", 0) for s in samples)
    _dbg_collate_ms = (_time.perf_counter() - _c_start) * 1000.0

    return {
        "volumes": volumes,
        "modality_id": modality_id,
        "phase_id": phase_id,
        "relative_time": relative_time,
        "visit_id": visit_id,
        "dataset_id": dataset_id,
        "modality_available_mask": modality_available_mask,
        "phase_available_mask": phase_available_mask,
        "valid_volume_indices": valid_volume_indices,
        "labels": labels,
        "label_mask": label_mask,
        "label_values": label_values,
        "label_mask_values": label_mask_values,
        "label_tasks": TASKS,
        "patient_id": [sample["patient_id"] for sample in samples],
        "sample_id": [sample["sample_id"] for sample in samples],
        "dataset_name": [sample["dataset_id"] for sample in samples],
        "visit_timepoint": [sample["visit_timepoint"] for sample in samples],
        "volume_paths": volume_paths,
        # 诊断字段：在 train_loop 里聚合打印，跑完诊断后可去掉。
        "_dbg_worker": {
            "load_ms": _dbg_load_ms,
            "torch_ms": _dbg_torch_ms,
            "total_ms": _dbg_total_ms,
            "max_ms": _dbg_max_ms,
            "n_vol": _dbg_n_vol,
            "collate_ms": _dbg_collate_ms,
            "alloc_ms": _t_alloc_ms,
            "meta_ms": _t_meta_ms,
        },
    }


def _lookup(vocab: dict[str, int], value: Any) -> int:
    text = str(value)
    return vocab.get(text, vocab.get(text.lower(), 0))


def make_collate(pad_to_max_volumes: int | None):
    """Return a top-level-importable collate that pads V to a fixed value.

    多进程 DataLoader 在 spawn 模式下需要 collate 可被 pickle，所以这里走的是
    一个普通函数而不是闭包/lambda。
    """
    if pad_to_max_volumes is None or int(pad_to_max_volumes) <= 0:
        return variable_modalities_collate
    return _FixedVolumesCollate(int(pad_to_max_volumes))


class _FixedVolumesCollate:
    __slots__ = ("pad_to_max_volumes",)

    def __init__(self, pad_to_max_volumes: int) -> None:
        self.pad_to_max_volumes = int(pad_to_max_volumes)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return variable_modalities_collate(samples, pad_to_max_volumes=self.pad_to_max_volumes)
