"""Reusable train/inference split strategies for the foundation dataset.

The foundation manifest already carries a ``split`` column (e.g. ``train`` /
``val`` / ``test`` / ``inference``) and a ``dataset_id`` column. For pretraining
we typically want **all** samples regardless of those columns, while for
finetuning we often want one of the following layouts:

* ``all``         — use every sample (mirrors pretraining; rarely used for
  finetuning but supported for completeness).
* ``manifest``    — honour whatever ``train_split`` / ``val_split`` /
  ``test_split`` columns the manifest already contains.
* ``by_dataset``  — pick a subset of ``dataset_id`` values for training and
  treat the rest as the held-out inference / test set.
* ``by_ratio``    — take a deterministic proportion of every sample for
  training and use the remainder for inference / test.

The strategies operate on already-built ``MultimodalManifestDataset`` samples
so the heavy CSV parsing happens once. They return Python ``Subset`` objects
ready for ``DataLoader``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from torch.utils.data import Dataset, Subset


@dataclass
class SplitConfig:
    """Resolved data-split configuration.

    Only ``mode`` is required. The remaining fields apply to specific modes.
    """

    mode: str = "manifest"
    # by_dataset
    train_datasets: tuple[str, ...] = ()
    val_datasets: tuple[str, ...] = ()
    test_datasets: tuple[str, ...] = ()
    # by_ratio
    train_ratio: float = 0.8
    val_ratio: float = 0.0
    seed: int = 2026
    # manifest mode column names
    train_split: str | None = "train"
    val_split: str | None = "val"
    test_split: str | None = "test"
    # the raw user payload, kept for logging
    raw: dict[str, Any] = field(default_factory=dict)


def parse_split_config(payload: dict[str, Any] | None, *, default_mode: str = "manifest") -> SplitConfig:
    """Translate a ``data.split_strategy`` config block into a :class:`SplitConfig`.

    Accepts both the new structured form::

        split_strategy:
          mode: by_ratio
          train_ratio: 0.7
          seed: 2026

    and the legacy flat form where ``train_split`` / ``val_split`` columns are
    set directly on the ``data`` block. ``payload`` may be ``None`` to fall
    back to ``default_mode``.
    """

    payload = dict(payload or {})
    mode = str(payload.get("mode") or default_mode).strip().lower()
    if mode not in {"all", "manifest", "by_dataset", "by_ratio"}:
        raise ValueError(
            f"Unknown split mode '{mode}'. Expected one of: all, manifest, by_dataset, by_ratio."
        )

    return SplitConfig(
        mode=mode,
        train_datasets=tuple(_to_str_tuple(payload.get("train_datasets"))),
        val_datasets=tuple(_to_str_tuple(payload.get("val_datasets"))),
        test_datasets=tuple(_to_str_tuple(payload.get("test_datasets"))),
        train_ratio=float(payload.get("train_ratio", 0.8)),
        val_ratio=float(payload.get("val_ratio", 0.0)),
        seed=int(payload.get("seed", 2026)),
        train_split=_optional_str(payload.get("train_split", "train")),
        val_split=_optional_str(payload.get("val_split", "val")),
        test_split=_optional_str(payload.get("test_split", "test")),
        raw=payload,
    )


def split_dataset(dataset: Dataset, config: SplitConfig) -> dict[str, Subset]:
    """Partition ``dataset`` into a ``{train, val, test}`` dict of Subsets.

    The returned dict always contains ``train`` and ``test`` keys. ``val`` is
    present only when the strategy or manifest provides a validation split.
    """

    samples: Sequence[dict[str, Any]] = getattr(dataset, "samples", None)  # type: ignore[arg-type]
    if samples is None:
        raise ValueError(
            "split_dataset requires a dataset that exposes a `samples` list "
            "(e.g. MultimodalManifestDataset)."
        )

    indices = list(range(len(samples)))

    if config.mode == "all":
        return {"train": Subset(dataset, indices), "test": Subset(dataset, [])}

    if config.mode == "manifest":
        return _split_by_manifest_column(dataset, samples, config)

    if config.mode == "by_dataset":
        return _split_by_dataset_id(dataset, samples, config)

    if config.mode == "by_ratio":
        return _split_by_ratio(dataset, samples, config)

    raise ValueError(f"Unsupported split mode: {config.mode}")


def _split_by_manifest_column(
    dataset: Dataset, samples: Sequence[dict[str, Any]], config: SplitConfig
) -> dict[str, Subset]:
    train_idx, val_idx, test_idx = [], [], []
    train_split = (config.train_split or "").strip()
    val_split = (config.val_split or "").strip()
    test_split = (config.test_split or "").strip()
    for idx, sample in enumerate(samples):
        sample_split = str(
            sample.get("metadata", {}).get("manifest_row", {}).get("split", "")
        ).strip()
        if train_split and sample_split == train_split:
            train_idx.append(idx)
        elif val_split and sample_split == val_split:
            val_idx.append(idx)
        elif test_split and sample_split == test_split:
            test_idx.append(idx)
    out = {"train": Subset(dataset, train_idx), "test": Subset(dataset, test_idx)}
    if val_idx:
        out["val"] = Subset(dataset, val_idx)
    return out


def _split_by_dataset_id(
    dataset: Dataset, samples: Sequence[dict[str, Any]], config: SplitConfig
) -> dict[str, Subset]:
    if not config.train_datasets:
        raise ValueError(
            "split_strategy.mode='by_dataset' requires a non-empty `train_datasets` list."
        )
    train_set = set(config.train_datasets)
    val_set = set(config.val_datasets)
    test_set = set(config.test_datasets)
    train_pool_idx, val_idx, test_idx = [], [], []
    for idx, sample in enumerate(samples):
        dataset_id = str(sample.get("dataset_id", ""))
        if dataset_id in train_set:
            train_pool_idx.append(idx)
            continue
        if dataset_id in val_set:
            val_idx.append(idx)
            continue
        # When ``test_datasets`` is empty, every non-train dataset becomes the
        # held-out set. Otherwise we only keep the explicitly listed ones so
        # users can ignore noisy datasets entirely.
        if not test_set or dataset_id in test_set:
            test_idx.append(idx)
    if not val_idx and float(config.val_ratio) > 0.0:
        train_idx, sampled_val_idx = _split_indices_by_ratio(train_pool_idx, samples, val_ratio=float(config.val_ratio), seed=config.seed)
        val_idx.extend(sampled_val_idx)
    else:
        train_idx = train_pool_idx
    output = {"train": Subset(dataset, train_idx), "test": Subset(dataset, test_idx)}
    if val_idx:
        output["val"] = Subset(dataset, val_idx)
    return output


def _split_by_ratio(
    dataset: Dataset, samples: Sequence[dict[str, Any]], config: SplitConfig
) -> dict[str, Subset]:
    ratio = float(config.train_ratio)
    val_ratio = float(config.val_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"split_strategy.train_ratio must be in (0, 1), got {config.train_ratio}."
        )
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(
            f"split_strategy.val_ratio must be in [0, 1), got {config.val_ratio}."
        )
    if ratio + val_ratio >= 1.0:
        raise ValueError(
            "split_strategy.train_ratio + val_ratio must be < 1.0 so a held-out test split remains."
        )

    train_idx, val_idx, test_idx = [], [], []
    for idx, sample in enumerate(samples):
        bucket = _sample_bucket(sample, seed=config.seed)
        if bucket < ratio:
            train_idx.append(idx)
        elif bucket < ratio + val_ratio:
            val_idx.append(idx)
        else:
            test_idx.append(idx)
    output = {"train": Subset(dataset, train_idx), "test": Subset(dataset, test_idx)}
    if val_idx:
        output["val"] = Subset(dataset, val_idx)
    return output


def _split_indices_by_ratio(
    indices: Sequence[int],
    samples: Sequence[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    train_idx, val_idx = [], []
    for idx in indices:
        bucket = _sample_bucket(samples[idx], seed=seed)
        if bucket < val_ratio:
            val_idx.append(idx)
        else:
            train_idx.append(idx)
    return train_idx, val_idx


def _sample_bucket(sample: dict[str, Any], *, seed: int) -> float:
    patient_id = str(sample.get("patient_id", "") or sample.get("sample_id", "")).strip()
    token = f"{seed}|{sample.get('dataset_id', '')}|{patient_id}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 10_000.0


def _to_str_tuple(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item).strip())
    raise TypeError(f"Cannot interpret {value!r} as a list of dataset ids.")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
