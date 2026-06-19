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
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve(strict=False)
            if cache_dir
            else (self.dataset_root / ".breast_dce_moe_pinn_cache").resolve(strict=False)
        )
        self.cache_version = "processed-v1"
        if self.cache_processed:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 调试用：跳过磁盘 IO，返回随机/零张量，纯粹测 GPU 端是否能跑满。
        # 用环境变量开关，避免污染 yaml；跑完关掉。
        self.fake_io = os.environ.get("BREAST_FAKE_IO") == "1"
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import time as _time
        _t_start = _time.perf_counter()
        sample = self.samples[index]
        # 先按 modality 过滤 + max_volumes 截断，再去 load —— 避免无谓的 IO/CPU。
        selected_records = []
        for record in sample["volume_records"]:
            if self.modalities and record["modality"] not in self.modalities:
                continue
            selected_records.append(record)
            if self.max_volumes is not None and len(selected_records) >= self.max_volumes:
                break

        # 直接 load 每个 volume 成 numpy；最后 stack + from_numpy 一次。
        # 不再用 SHM 池：池化在 fork-style worker 上反而触发 COW page fault，
        # 收益负向。参考 pcr_project 的实现（它也是简单 stack）。
        v_buf = int(self.max_volumes) if self.max_volumes else max(len(selected_records), 1)
        phases = np.empty((v_buf, 1, *self.target_shape), dtype=np.float32)
        # 用 np.empty + 写入 N slot + 显式 zero pad，避免整片 zeros 触发的 fault：
        # zeros 会立刻 commit 全部页面；empty 在 Linux 上是 lazy，等真正写入时再 commit。
        # 对于真正会被写入的 N slot，zeros / empty 等价；对于 padded slot，empty
        # 后我们用 ``[N:].fill(0.0)`` 显式清零（也只 commit 该范围）。
        _t_load = 0.0
        _n_vol = 0
        for slot, record in enumerate(selected_records):
            _t0 = _time.perf_counter()
            arr = self._load_volume(record["path"])
            np.copyto(phases[slot, 0], arr, casting="same_kind")
            _t_load += _time.perf_counter() - _t0
            _n_vol += 1
        # padded slot 清零
        if _n_vol < v_buf:
            phases[_n_vol:].fill(0.0)

        _t_torch_start = _time.perf_counter()
        sample_tensor = torch.from_numpy(phases)
        _t_torch = _time.perf_counter() - _t_torch_start

        # 每个 record 复制成轻量字典，不附 tensor —— collate 只看元数据。
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
            meta_records.append({
                "modality": "unknown",
                "phase_index": None,
                "relative_time": None,
                "path": "",
                "available": 0,
            })

        _t_total = _time.perf_counter() - _t_start
        return {
            "patient_id": sample["patient_id"],
            "sample_id": sample["sample_id"],
            "dataset_id": sample["dataset_id"],
            "visit_timepoint": sample["visit_timepoint"],
            "sample_tensor": sample_tensor,
            "volume_records": meta_records,
            "labels": dict(sample["labels"]),
            "label_mask": dict(sample["label_mask"]),
            "metadata": dict(sample["metadata"]),
            "_dbg_load_ms": _t_load * 1000.0,
            "_dbg_torch_ms": _t_torch * 1000.0,
            "_dbg_total_ms": _t_total * 1000.0,
            "_dbg_n_vol": _n_vol,
        }

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
        return {
            "modality": modality,
            "phase_index": _optional_int(phase_raw),
            "relative_time": _optional_float(item.get("relative_time", row.get("relative_time"))),
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
                # 不再用 mmap_mode：现在 __getitem__ 直接 ``np.copyto`` 进
                # sample buffer，需要源数据是 plain ndarray。``np.load`` 默认
                # 一次性读完返回 owned ndarray，page cache 命中时和 mmap 等价
                # 但避免了 ascontiguousarray 那一次额外的 4MB copy。
                arr = np.load(cache_path)
                if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
                    return arr
                return np.ascontiguousarray(arr, dtype=np.float32)
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
