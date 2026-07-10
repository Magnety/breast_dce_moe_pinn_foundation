from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.augmentations_3d import build_augmentation
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.label_utils import parse_labels
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.transforms_3d import center_crop_or_pad, robust_normalize
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.datasets.collate_variable_modalities import (
    DATASET_VOCAB,
    MODALITY_VOCAB,
    TASKS,
    TASK_TO_INDEX,
    VISIT_VOCAB,
)

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object


def dataloader_worker_init(worker_id: int) -> None:
    """限制每个 DataLoader worker 只用一个 CPU 线程。

    32+ core 机器上，numpy/torch 默认每个进程开 ``cpu_count`` 个线程做
    BLAS / vectorize；8 worker × 32 thread 会在 OS 调度上互相抢，让本来
    几毫秒的 ``np.copyto`` 抖到几百毫秒。把每 worker 锁到 1 thread 后，
    各 worker 独立稳定地完成 IO，wait 时间会显著下降。
    """
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if torch is not None:
        torch.set_num_threads(1)


class MultimodalManifestDataset(Dataset):
    """Manifest-driven variable modality dataset.

    The dataset accepts either one row per sample with a ``series_manifest_path``
    JSON file, or one row per volume with a ``path``/``file_path`` column. It
    returns a list of volume records so missing modalities, missing DCE phases
    and missing treatment visits remain explicit instead of being forced into
    fixed channels.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path | None = None,
        split: str | None = None,
        label_columns: dict[str, str] | None = None,
        target_shape: tuple[int, int, int] = (64, 128, 128),
        modalities: list[str] | tuple[str, ...] | None = None,
        normalize: bool = True,
        allow_empty_labels: bool = True,
        max_volumes: int | None = None,
        augmentation: dict[str, Any] | None = None,
        augment: bool = False,
        include_datasets: list[str] | tuple[str, ...] | None = None,
        exclude_datasets: list[str] | tuple[str, ...] | None = None,
        cache_processed: bool = False,
        cache_dir: str | Path | None = None,
        cache_after_normalize: bool = True,
        npy_mmap_mode: str | None = None,
        profile_io: bool = False,
        compact_sample_tensors: bool = False,
        include_identifiers: bool = False,
        include_volume_paths: bool = False,
        restrict_to_dce_phases: bool = False,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError("torch is required for MultimodalManifestDataset")
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False) if dataset_root else self.manifest_path.parent
        # ``split`` 现在支持三种语义：
        #   * None / ""        -> 不过滤 split 列，使用 manifest 全部样本
        #   * "<single split>" -> 仅保留该 split
        #   * list/tuple/set   -> 保留若干 split（例如 ["train", "val"]）
        self.split = split
        self.label_columns = label_columns or {}
        self.target_shape = tuple(target_shape)
        self.modalities = {str(item) for item in (modalities or []) if str(item).strip()}
        self.normalize = normalize
        self.allow_empty_labels = allow_empty_labels
        self.max_volumes = max_volumes
        self.augmentation = build_augmentation(augmentation, enabled=augment)
        self.include_datasets = {str(item).strip() for item in (include_datasets or []) if str(item).strip()}
        self.exclude_datasets = {str(item).strip() for item in (exclude_datasets or []) if str(item).strip()}
        self.cache_processed = bool(cache_processed)
        self.cache_after_normalize = bool(cache_after_normalize)
        self.npy_mmap_mode = npy_mmap_mode
        self.cache_mmap_mode = (
            npy_mmap_mode
            if npy_mmap_mode is not None
            else ("r" if self.cache_processed and self.cache_after_normalize else None)
        )
        self.profile_io = bool(profile_io)
        self.include_identifiers = bool(include_identifiers)
        self.include_volume_paths = bool(include_volume_paths)
        self.restrict_to_dce_phases = bool(restrict_to_dce_phases)
        # sample-pack 命中后，直接把 memmap 视图包装成 sample tensor，
        # 交给 pin_memory + CUDA prefetch 去做唯一一次主机侧拷贝。这样能
        # 去掉 worker 内部那次 ``np.copyto -> np.empty(32MB)``。
        self.compact_sample_tensors = bool(compact_sample_tensors)
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve(strict=False)
            if cache_dir
            else (self.dataset_root / ".breast_dce_moe_pinn_cache").resolve(strict=False)
        )
        # ``processed-v1`` 是按 volume 一文件的旧布局；``processed-sample-v1`` 是
        # 按 sample 打包的新布局（一个 sample 的所有 selected volumes 写到同一
        # 个 ``[n_vol, D, H, W]`` 的 npy 里），开启 cache 时只需一次 open 而不是
        # max_volumes 次。两层 cache 各自维护版本号：fallback 路径仍可以命中
        # 已经存在的 volume 级 cache，不会因为新增 sample-pack 而被作废。
        self.cache_version = "processed-v1"
        self.sample_pack_version = "processed-sample-v1"
        # sample 级 packed cache 放在子目录里，避免和旧的 volume 级 cache 文件
        # 在同一个目录下混在一起。
        self.sample_cache_dir = self.cache_dir / "samples"
        if self.cache_processed:
            self.sample_cache_dir.mkdir(parents=True, exist_ok=True)
        # 调试用：跳过磁盘 IO，返回随机/零张量，纯粹测 GPU 端是否能跑满。
        # 用环境变量开关，避免污染 yaml；跑完关掉。
        self.fake_io = os.environ.get("BREAST_FAKE_IO") == "1"
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        selected_records = sample["selected_records"]
        v_buf = int(sample["selected_v_buf"])
        if self.profile_io:
            import time as _time

            t_start = _time.perf_counter()
        else:
            _time = None
            t_start = 0.0

        sample_tensor: torch.Tensor | None = None
        load_ms = 0.0
        n_vol = 0
        t_torch_ms = 0.0

        # 新快路径：sample-pack 命中后不再先分配 ``(V_buf, 1, D, H, W)`` 的
        # numpy buffer 再 ``np.copyto``。直接把 pack 的 memmap 视图包装成
        # ``[n_vol, 1, D, H, W]`` tensor，后续让 pin_memory / CUDA prefetch
        # 完成唯一一次主机侧 copy。这样 steady-state 能明显减少 worker 等待。
        if (
            self.compact_sample_tensors
            and not self.fake_io
            and self.cache_processed
            and self.cache_after_normalize
            and self.augmentation is None
            and selected_records
        ):
            try:
                t_pack_start = _time.perf_counter() if self.profile_io else 0.0
                packed = self._load_or_create_sample_pack(selected_records)
                n_vol = min(int(packed.shape[0]), len(selected_records))
                if self.profile_io:
                    load_ms = (_time.perf_counter() - t_pack_start) * 1000.0
                t_torch_start = _time.perf_counter() if self.profile_io else 0.0
                sample_tensor = torch.from_numpy(packed[:n_vol, None, ...])
                t_torch_ms = ((_time.perf_counter() - t_torch_start) * 1000.0) if self.profile_io else 0.0
            except Exception:
                sample_tensor = None
                load_ms = 0.0
                n_vol = 0

        if sample_tensor is None:
            phases = np.empty((v_buf, 1, *self.target_shape), dtype=np.float32)
            load_ms, _sample_total_ms, n_vol = self._fill_volume_tensor(
                phases,
                selected_records,
                time_module=_time,
            )
            if n_vol < v_buf:
                phases[n_vol:].fill(0.0)

            t_torch_start = _time.perf_counter() if self.profile_io else 0.0
            sample_tensor = torch.from_numpy(phases)
            t_torch_ms = ((_time.perf_counter() - t_torch_start) * 1000.0) if self.profile_io else 0.0

        result = {
            "sample_tensor": sample_tensor,
            "modality_id_tensor": sample["modality_id_tensor"],
            "phase_id_tensor": sample["phase_id_tensor"],
            "relative_time_tensor": sample["relative_time_tensor"],
            "modality_available_mask_tensor": sample["modality_available_mask_tensor"],
            "phase_available_mask_tensor": sample["phase_available_mask_tensor"],
            "temporal_dce_mask_tensor": sample["temporal_dce_mask_tensor"],
            "label_values_tensor": sample["label_values_tensor"],
            "label_mask_values_tensor": sample["label_mask_values_tensor"],
            "visit_id_value": sample["visit_id_value"],
            "dataset_id_value": sample["dataset_id_value"],
            "metadata": sample["metadata"],
        }
        if self.include_identifiers:
            result.update(
                {
                    "patient_id": sample["patient_id"],
                    "sample_id": sample["sample_id"],
                    "dataset_id": sample["dataset_id"],
                    "visit_timepoint": sample["visit_timepoint"],
                }
            )
        if self.include_volume_paths:
            result["volume_paths"] = list(sample["volume_paths"])
        if self.profile_io:
            total_ms = (_time.perf_counter() - t_start) * 1000.0
            result.update(
                {
                    "_dbg_load_ms": load_ms,
                    "_dbg_torch_ms": t_torch_ms,
                    "_dbg_total_ms": total_ms,
                    "_dbg_n_vol": n_vol,
                }
            )
        return result

    def get_batch(self, indices: list[int] | tuple[int, ...]) -> dict[str, Any]:
        index_list = [int(index) for index in indices]
        if not index_list:
            raise ValueError("Cannot build a batch from an empty index list.")

        batch_samples = [self.samples[index] for index in index_list]
        if self.profile_io:
            import time as _time

            t_start = _time.perf_counter()
        else:
            _time = None
            t_start = 0.0

        selected_by_sample = [self._selected_records(sample)[0] for sample in batch_samples]
        if self.max_volumes is not None:
            v_buf = max(1, int(self.max_volumes))
        else:
            v_buf = max((len(records) for records in selected_by_sample), default=1)

        batch_size = len(batch_samples)
        phases = np.empty((batch_size, v_buf, 1, *self.target_shape), dtype=np.float32)
        modality_id = torch.zeros((batch_size, v_buf), dtype=torch.long)
        phase_id = torch.zeros((batch_size, v_buf), dtype=torch.long)
        relative_time = torch.zeros((batch_size, v_buf), dtype=torch.float32)
        modality_available_mask = torch.zeros((batch_size, v_buf), dtype=torch.bool)
        phase_available_mask = torch.zeros((batch_size, v_buf), dtype=torch.bool)
        temporal_dce_mask = torch.zeros((batch_size, v_buf), dtype=torch.bool)
        label_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
        label_mask_values = torch.empty((batch_size, len(TASKS)), dtype=torch.float32)
        volume_paths: list[list[str]] = []

        total_load_ms = 0.0
        max_sample_ms = 0.0
        total_n_vol = 0
        for row, (sample, selected_records) in enumerate(zip(batch_samples, selected_by_sample)):
            sample_start = _time.perf_counter() if self.profile_io else 0.0
            load_ms, _sample_total_ms, n_vol = self._fill_volume_tensor(
                phases[row],
                selected_records,
                time_module=_time,
            )
            total_load_ms += load_ms
            total_n_vol += n_vol
            if n_vol < v_buf:
                phases[row, n_vol:].fill(0.0)

            paths: list[str] = []
            for col, record in enumerate(selected_records):
                modality_id[row, col] = _lookup_vocab(MODALITY_VOCAB, record.get("modality", "unknown"))
                phase = record.get("phase_index")
                phase_id[row, col] = 0 if phase is None else int(phase) + 1
                rel_time = record.get("relative_time")
                relative_time[row, col] = 0.0 if rel_time is None else float(rel_time)
                available = bool(record.get("available", 1))
                modality_available_mask[row, col] = available
                phase_available_mask[row, col] = available and phase is not None
                temporal_dce_mask[row, col] = available and _is_dce_temporal_record(record)
                paths.append(str(record.get("path", "")))
            for _ in range(len(selected_records), v_buf):
                paths.append("")
            volume_paths.append(paths)

            for task, col in TASK_TO_INDEX.items():
                label_values[row, col] = float(sample["labels"].get(task, -1.0))
                label_mask_values[row, col] = float(sample["label_mask"].get(task, 0.0))
            if self.profile_io:
                max_sample_ms = max(max_sample_ms, (_time.perf_counter() - sample_start) * 1000.0)

        torch_start = _time.perf_counter() if self.profile_io else 0.0
        volumes = torch.from_numpy(phases)
        torch_ms = ((_time.perf_counter() - torch_start) * 1000.0) if self.profile_io else 0.0

        batch = {
            "volumes": volumes,
            "modality_id": modality_id,
            "phase_id": phase_id,
            "relative_time": relative_time,
            "visit_id": torch.tensor(
                [_lookup_vocab(VISIT_VOCAB, sample["visit_timepoint"]) for sample in batch_samples],
                dtype=torch.long,
            ),
            "dataset_id": torch.tensor(
                [_lookup_vocab(DATASET_VOCAB, sample["dataset_id"]) for sample in batch_samples],
                dtype=torch.long,
            ),
            "modality_available_mask": modality_available_mask,
            "phase_available_mask": phase_available_mask,
            "temporal_dce_mask": temporal_dce_mask,
            "valid_volume_indices": torch.arange(batch_size * v_buf, dtype=torch.long),
            "valid_volume_indices_is_dense": True,
            "labels": {task: label_values[:, col] for task, col in TASK_TO_INDEX.items()},
            "label_mask": {task: label_mask_values[:, col] for task, col in TASK_TO_INDEX.items()},
            "label_values": label_values,
            "label_mask_values": label_mask_values,
            "label_tasks": TASKS,
            "patient_id": [sample["patient_id"] for sample in batch_samples],
            "sample_id": [sample["sample_id"] for sample in batch_samples],
            "dataset_name": [sample["dataset_id"] for sample in batch_samples],
            "visit_timepoint": [sample["visit_timepoint"] for sample in batch_samples],
            "volume_paths": volume_paths,
        }
        if self.profile_io:
            total_ms = (_time.perf_counter() - t_start) * 1000.0
            batch["_dbg_worker"] = {
                "load_ms": total_load_ms,
                "torch_ms": torch_ms,
                "total_ms": total_ms,
                "max_ms": max_sample_ms,
                "n_vol": total_n_vol,
                "collate_ms": 0.0,
                "alloc_ms": 0.0,
                "meta_ms": max(total_ms - total_load_ms - torch_ms, 0.0),
            }
        return batch

    def _selected_records(self, sample: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        selected_records = []
        for record in sample["volume_records"]:
            if self.restrict_to_dce_phases and not _is_dce_temporal_record(record):
                continue
            if self.modalities and record["modality"] not in self.modalities:
                continue
            selected_records.append(record)
            if self.max_volumes is not None and len(selected_records) >= int(self.max_volumes):
                break
        if self.max_volumes is not None:
            v_buf = max(1, int(self.max_volumes))
        else:
            v_buf = max(len(selected_records), 1)
        return selected_records, v_buf

    def _meta_records(self, selected_records: list[dict[str, Any]], v_buf: int) -> list[dict[str, Any]]:
        meta_records = [
            {
                "modality": record.get("modality", "unknown"),
                "phase_index": record.get("phase_index"),
                "relative_time": record.get("relative_time"),
                "path": str(record.get("path", "")),
                "available": 1,
            }
            for record in selected_records
        ]
        for _ in range(len(meta_records), v_buf):
            meta_records.append(
                {
                    "modality": "unknown",
                    "phase_index": None,
                    "relative_time": None,
                    "path": "",
                    "available": 0,
                }
            )
        return meta_records

    def _prepare_sample_runtime_metadata(self, sample: dict[str, Any]) -> None:
        selected_records, v_buf = self._selected_records(sample)
        meta_records = self._meta_records(selected_records, v_buf)

        modality_id = torch.zeros((v_buf,), dtype=torch.long)
        phase_id = torch.zeros((v_buf,), dtype=torch.long)
        relative_time = torch.zeros((v_buf,), dtype=torch.float32)
        modality_available_mask = torch.zeros((v_buf,), dtype=torch.bool)
        phase_available_mask = torch.zeros((v_buf,), dtype=torch.bool)
        temporal_dce_mask = torch.zeros((v_buf,), dtype=torch.bool)
        volume_paths: list[str] = []
        for col, record in enumerate(meta_records[:v_buf]):
            modality_id[col] = _lookup_vocab(MODALITY_VOCAB, record.get("modality", "unknown"))
            phase = record.get("phase_index")
            phase_id[col] = 0 if phase is None else int(phase) + 1
            rel_time = record.get("relative_time")
            relative_time[col] = 0.0 if rel_time is None else float(rel_time)
            available = bool(record.get("available", 1))
            modality_available_mask[col] = available
            phase_available_mask[col] = available and phase is not None
            temporal_dce_mask[col] = available and _is_dce_temporal_record(record)
            volume_paths.append(str(record.get("path", "")))

        sample["selected_records"] = selected_records
        sample["selected_v_buf"] = v_buf
        sample["selected_n_vol"] = len(selected_records)
        sample["modality_id_tensor"] = modality_id
        sample["phase_id_tensor"] = phase_id
        sample["relative_time_tensor"] = relative_time
        sample["modality_available_mask_tensor"] = modality_available_mask
        sample["phase_available_mask_tensor"] = phase_available_mask
        sample["temporal_dce_mask_tensor"] = temporal_dce_mask
        sample["volume_paths"] = volume_paths
        sample["visit_id_value"] = _lookup_vocab(VISIT_VOCAB, sample["visit_timepoint"])
        sample["dataset_id_value"] = _lookup_vocab(DATASET_VOCAB, sample["dataset_id"])
        sample["label_values_tensor"] = torch.tensor(
            [float(sample["labels"].get(task, -1.0)) for task in TASKS],
            dtype=torch.float32,
        )
        sample["label_mask_values_tensor"] = torch.tensor(
            [float(sample["label_mask"].get(task, 0.0)) for task in TASKS],
            dtype=torch.float32,
        )

    def _fill_volume_tensor(
        self,
        phases: np.ndarray,
        selected_records: list[dict[str, Any]],
        *,
        time_module,
    ) -> tuple[float, float, int]:
        load_ms = 0.0
        sample_start = time_module.perf_counter() if time_module is not None else 0.0
        n_vol = 0

        # Sample-packed cache 路径：一次 open + memmap 整个 ``[n_vol, D, H, W]``，
        # 然后用一次 ``np.copyto`` 把 slice 拷到 ``phases[:n_vol, 0]``。这样每个
        # sample 的 syscall 数从 ``max_volumes`` 次降到 1 次，page-cache 也按 sample
        # 连续命中。Augment 是 in-place 修改输入数组的，需要在 cache 命中后逐 slot
        # 调用，保持"cache 存 normalized 未 augment 数据"的旧语义。
        if (
            not self.fake_io
            and self.cache_processed
            and self.cache_after_normalize
            and selected_records
        ):
            try:
                t0 = time_module.perf_counter() if time_module is not None else 0.0
                packed = self._load_or_create_sample_pack(selected_records)
                pack_n = int(packed.shape[0])
                if pack_n > 0:
                    n_vol = min(pack_n, phases.shape[0])
                    np.copyto(phases[:n_vol, 0], packed[:n_vol], casting="same_kind")
                    if self.augmentation is not None:
                        for slot in range(n_vol):
                            phases[slot, 0] = self.augmentation(phases[slot, 0])
                    if time_module is not None:
                        load_ms += (time_module.perf_counter() - t0) * 1000.0
                    total_ms = (
                        (time_module.perf_counter() - sample_start) * 1000.0
                        if time_module is not None
                        else 0.0
                    )
                    return load_ms, total_ms, n_vol
            except Exception:
                # Pack 读取失败（例如缓存被破坏）就回退到逐 volume 路径，
                # 让 ``_load_volume`` 的常规逻辑继续工作并按需重建 cache。
                n_vol = 0

        for slot, record in enumerate(selected_records):
            t0 = time_module.perf_counter() if time_module is not None else 0.0
            arr = self._load_volume(record["path"])
            np.copyto(phases[slot, 0], arr, casting="same_kind")
            if time_module is not None:
                load_ms += (time_module.perf_counter() - t0) * 1000.0
            n_vol += 1
        total_ms = ((time_module.perf_counter() - sample_start) * 1000.0) if time_module is not None else 0.0
        return load_ms, total_ms, n_vol

    def _build_samples(self) -> list[dict[str, Any]]:
        rows = self._read_manifest_rows()
        # ``split`` 可能是 None / 字符串 / 可迭代集合，统一成 set 便于过滤。
        if self.split is None or self.split == "":
            split_filter: set[str] | None = None
        elif isinstance(self.split, str):
            split_filter = {self.split}
        else:
            split_filter = {str(item) for item in self.split if str(item).strip()}
            if not split_filter:
                split_filter = None
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in rows:
            row_split = row.get("split")
            if split_filter is not None and row_split and row_split not in split_filter:
                continue
            patient_id = _first(row, "patient_id", "patient_uid", "subject_id", default="unknown")
            sample_id = _first(row, "sample_id", "study_uid", default=f"{patient_id}:{row.get('timepoint', '')}")
            dataset_id = _first(row, "dataset_id", default="unknown")
            if self.include_datasets and dataset_id not in self.include_datasets:
                continue
            if self.exclude_datasets and dataset_id in self.exclude_datasets:
                continue
            visit = _first(row, "visit_timepoint", "timepoint", "visit", default="unknown")
            key = (dataset_id, patient_id, sample_id, visit)
            if key not in grouped:
                labels, label_mask = parse_labels(row, self.label_columns)
                grouped[key] = {
                    "patient_id": patient_id,
                    "sample_id": sample_id,
                    "dataset_id": dataset_id,
                    "visit_timepoint": visit,
                    "volume_records": [],
                    "labels": labels,
                    "label_mask": label_mask,
                    "metadata": {"manifest_row": dict(row)},
                }
            grouped[key]["volume_records"].extend(self._volume_records_from_row(row))
        samples = []
        for sample in grouped.values():
            if not self.allow_empty_labels and not any(float(v) > 0 for v in sample["label_mask"].values()):
                continue
            sample["volume_records"] = sorted(sample["volume_records"], key=_volume_sort_key)
            if self.restrict_to_dce_phases:
                selected_records, _ = self._selected_records(sample)
                if not selected_records:
                    continue
            self._prepare_sample_runtime_metadata(sample)
            samples.append(sample)
        return samples

    def _read_manifest_rows(self) -> list[dict[str, str]]:
        if self.manifest_path.suffix.lower() in {".json", ".jsonl"}:
            text = self.manifest_path.read_text(encoding="utf-8")
            if self.manifest_path.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            payload = json.loads(text)
            return payload if isinstance(payload, list) else payload.get("samples", [])
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _volume_records_from_row(self, row: dict[str, str]) -> list[dict[str, Any]]:
        manifest_value = _first(row, "series_manifest_path", "series_manifest", default="")
        if manifest_value:
            manifest_path = self._resolve_path(manifest_value, row)
            if manifest_path.exists():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                images = payload.get("images", payload if isinstance(payload, list) else [])
                records = []
                for item in images:
                    if not isinstance(item, dict):
                        continue
                    path_value = _first(item, "path", "file_path", "npy_path", "image_path", default="")
                    if not path_value:
                        continue
                    records.append(self._normalize_record(row, item, path_value))
                return records
        path_value = _first(row, "path", "file_path", "npy_path", "volume_path", default="")
        return [self._normalize_record(row, row, path_value)] if path_value else []

    def _normalize_record(self, row: dict[str, str], item: dict[str, Any], path_value: str) -> dict[str, Any]:
        modality = str(item.get("modality") or item.get("series_role") or row.get("modality") or "unknown")
        phase_raw = item.get("phase_index", item.get("phase", row.get("phase_index", row.get("phase"))))
        phase_index = _optional_int(phase_raw)
        relative_time = _optional_float(item.get("relative_time", row.get("relative_time")))
        # 真实采集时间缺失时，退回到 phase_index 作为单调伪时间轴，至少让
        # AIF / ODE 支路看到非零 dt，而不是整条 PINN 时间分支退化成 0。
        if relative_time is None and phase_index is not None:
            relative_time = float(phase_index)
        return {
            "modality": modality,
            "phase_index": phase_index,
            "relative_time": relative_time,
            "path": str(self._resolve_path(str(path_value), row)),
            "available": 1,
        }

    def _resolve_path(self, value: str, row: dict[str, str]) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        manifest_dir = Path(row.get("_manifest_dir", self.manifest_path.parent))
        for base in (manifest_dir, self.dataset_root, self.manifest_path.parent):
            candidate = (base / path).resolve(strict=False)
            if candidate.exists():
                return candidate
        return (self.dataset_root / path).resolve(strict=False)

    def _load_volume(self, path_value: str) -> np.ndarray:
        path = Path(path_value)
        if self.fake_io:
            # 跳过磁盘 IO，给一个零体积。诊断用：跑一次看 GPU 是否能满载，
            # 若能，瓶颈就是磁盘/解码；若仍卡，瓶颈在 collate/IPC/模型本身。
            return np.zeros(self.target_shape, dtype=np.float32)
        if self.cache_processed and self.cache_after_normalize:
            volume = self._load_or_create_processed_cache(path)
            if self.augmentation is not None:
                volume = self.augmentation(volume)
            return volume.astype(np.float32, copy=False)

        volume = self._load_source_volume(path)
        volume = self._prepare_volume(volume, path)
        if self.augmentation is not None:
            volume = self.augmentation(volume)
        return robust_normalize(volume) if self.normalize else volume.astype(np.float32)

    def _load_source_volume(self, path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            volume = np.load(path, mmap_mode=self.npy_mmap_mode)
        elif path.suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".nii":
            volume = _load_nifti(path)
        else:
            volume = np.load(path, mmap_mode=self.npy_mmap_mode)
        return np.asarray(volume, dtype=np.float32)

    def _prepare_volume(self, volume: np.ndarray, path: Path) -> np.ndarray:
        volume = np.squeeze(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got shape {volume.shape}")
        return center_crop_or_pad(volume.astype(np.float32, copy=False), self.target_shape)

    def _load_or_create_processed_cache(self, path: Path) -> np.ndarray:
        cache_path = self._processed_cache_path(path)
        if cache_path.exists():
            try:
                # 真正用 ``npy_mmap_mode`` —— ``__getitem__`` 后续的
                # ``np.copyto(buf, arr)`` 会按需从 page cache 读 page，
                # 不需要 ``np.asarray`` / ``np.ascontiguousarray`` 把整个
                # 4MB 强制物化一次。
                arr = np.load(cache_path, mmap_mode=self.cache_mmap_mode)
                if arr.dtype == np.float32 and tuple(arr.shape) == tuple(self.target_shape):
                    return arr
                # dtype 或 shape 不符合当前配置 —— 视为旧 cache，删除重建。
                _unlink_if_exists(cache_path)
            except Exception:
                _unlink_if_exists(cache_path)

        volume = self._load_source_volume(path)
        volume = self._prepare_volume(volume, path)
        volume = robust_normalize(volume) if self.normalize else volume.astype(np.float32, copy=False)
        self._write_cache_atomic(cache_path, volume)
        return volume

    def _processed_cache_path(self, path: Path) -> Path:
        stat = path.stat()
        payload = {
            "path": str(path.expanduser().resolve(strict=False)),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "target_shape": self.target_shape,
            "normalize": self.normalize,
            "version": self.cache_version,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    # ------------------------------------------------------------------
    # Sample-packed cache：把一个 sample 选中的若干 volume 拼成
    # ``[n_vol, D, H, W]`` 的一个 npy，避免按 volume 多文件读取带来的多次
    # ``open`` / ``mmap`` / page-cache 抖动。
    # ------------------------------------------------------------------

    def _sample_pack_mmap_mode(self) -> str | None:
        # ``torch.from_numpy`` 包一层 sample-pack memmap 时，希望底层数组对
        # 当前进程是 writeable 的；否则 PyTorch 会提示只读 ndarray。这里把
        # 只读 ``r`` 提升成 copy-on-write ``c``，仍然保持按 page lazy-load。
        if self.cache_mmap_mode == "r":
            return "c"
        return self.cache_mmap_mode

    def _load_or_create_sample_pack(self, selected_records: list[dict[str, Any]]) -> np.ndarray:
        cache_path = self._sample_pack_path(selected_records)
        n_vol = len(selected_records)
        mmap_mode = self._sample_pack_mmap_mode()
        if cache_path.exists():
            try:
                arr = np.load(cache_path, mmap_mode=mmap_mode)
                expected_shape = (n_vol, *self.target_shape)
                if arr.dtype == np.float32 and tuple(arr.shape) == expected_shape:
                    return arr
                _unlink_if_exists(cache_path)
            except Exception:
                _unlink_if_exists(cache_path)

        packed = np.empty((n_vol, *self.target_shape), dtype=np.float32)
        for slot, record in enumerate(selected_records):
            packed[slot] = self._load_processed_volume(Path(record["path"]))
        self._write_cache_atomic(cache_path, packed)
        # 写完之后用 mmap 重新打开，让后续 ``np.copyto`` 或 sample-tensor
        # 视图都走 page cache，避免常驻内存。
        if mmap_mode is not None:
            try:
                return np.load(cache_path, mmap_mode=mmap_mode)
            except Exception:
                pass
        return packed

    def _load_processed_volume(self, path: Path) -> np.ndarray:
        """读取并归一化单个 volume，但不写 volume 级 cache，也不做 augment。

        sample pack 只缓存"normalized 未 augment"的数据；augment 留给
        ``_fill_volume_tensor`` 在 cache 命中后逐 slot 应用。
        """

        volume = self._load_source_volume(path)
        volume = self._prepare_volume(volume, path)
        return robust_normalize(volume) if self.normalize else volume.astype(np.float32, copy=False)

    def _sample_pack_path(self, selected_records: list[dict[str, Any]]) -> Path:
        # Pack 的身份由 ``(目标 shape, normalize 设置, 各 volume 的 path+mtime+size)``
        # 决定。任何一个 volume 被替换都会重新算出新 digest，重建 pack。
        items: list[dict[str, Any]] = []
        for record in selected_records:
            path = Path(record["path"]).expanduser().resolve(strict=False)
            try:
                stat = path.stat()
                items.append(
                    {
                        "path": str(path),
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                    }
                )
            except FileNotFoundError:
                items.append({"path": str(path), "mtime_ns": 0, "size": 0})
        payload = {
            "items": items,
            "target_shape": self.target_shape,
            "normalize": self.normalize,
            "version": self.sample_pack_version,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        # 用前两位做一级目录，避免单目录下塞几万个文件让 ext4 吃不消。
        return self.sample_cache_dir / digest[:2] / f"{digest}.npy"

    def _write_cache_atomic(self, cache_path: Path, volume: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, volume.astype(np.float32, copy=False))
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                _unlink_if_exists(tmp_path)

    def _empty_volume_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "modality": "unknown",
            "phase_index": None,
            "relative_time": None,
            "path": "",
            "tensor": torch.zeros((1, *self.target_shape), dtype=torch.float32),
            "available": 0,
        }


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    except ModuleNotFoundError:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image).astype(np.float32)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _lookup_vocab(vocab: dict[str, int], value: Any) -> int:
    text = str(value)
    return vocab.get(text, vocab.get(text.lower(), 0))


def _first(mapping: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _optional_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _volume_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    modality_order = {"DCE": 0, "T1": 1, "T2": 2, "DWI": 3, "ADC": 4, "derived": 5, "unknown": 99}
    phase = record.get("phase_index")
    return (modality_order.get(str(record.get("modality", "unknown")), 90), 999 if phase is None else int(phase), str(record.get("path", "")))


def _is_dce_temporal_record(record: dict[str, Any]) -> bool:
    if str(record.get("modality", "unknown")) != "DCE":
        return False
    return record.get("phase_index") is not None or record.get("relative_time") is not None
