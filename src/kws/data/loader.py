"""Shared DataLoader construction for overlapped input processing."""

import sys
from collections.abc import Mapping

from torch.utils.data import DataLoader, Dataset

from kws.utils.logging import get_logger

logger = get_logger(__name__)


def build_data_loader(
    dataset: Dataset, train_cfg: Mapping, *, shuffle: bool
) -> DataLoader:
    """Build a loader whose workers survive and prefetch across epochs.

    Worker-specific options must be omitted when ``num_workers`` is zero;
    PyTorch rejects ``persistent_workers`` and ``prefetch_factor`` otherwise.
    """
    num_workers = int(train_cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    batch_size = int(train_cfg["batch_size"])
    prefetch_factor = None
    if num_workers:
        prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if sys.platform == "darwin":
            logger.warning(
                "macOS DataLoader workers are disabled because this PyTorch "
                "build cannot start torch_shm_manager reliably; using "
                "num_workers=0"
            )
            num_workers = 0

    if num_workers:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
