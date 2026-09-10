"""Shared DataLoader construction for overlapped input processing."""
from collections.abc import Mapping

from torch.utils.data import DataLoader, Dataset


def build_data_loader(dataset: Dataset, train_cfg: Mapping, *, shuffle: bool) -> DataLoader:
    """Build a loader whose workers survive and prefetch across epochs.

    Worker-specific options must be omitted when ``num_workers`` is zero;
    PyTorch rejects ``persistent_workers`` and ``prefetch_factor`` otherwise.
    """
    num_workers = int(train_cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    kwargs = {
        "dataset": dataset,
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    if num_workers:
        prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        kwargs.update(
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(**kwargs)
