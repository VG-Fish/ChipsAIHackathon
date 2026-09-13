"""Shared DataLoader construction for overlapped input processing."""

from collections.abc import Iterator, Mapping
from typing import Callable, cast

import torch
from torch.utils.data import DataLoader, Dataset

from kws.utils.logging import get_logger

logger = get_logger(__name__)


def _worker_startup_error(error: RuntimeError) -> bool:
    """Return whether an error is safe to recover by disabling workers.

    PyTorch builds on macOS can fail while starting ``torch_shm_manager``
    (for example in a restricted shell or after an OS update).  This is a
    loader-infrastructure failure, not a dataset failure.  Do not mask other
    RuntimeErrors raised by ``Dataset.__getitem__``.
    """
    message = str(error).lower()
    # The generic DataLoader worker phrases also occur when __getitem__ or a
    # transform raises.  Treating those as infrastructure failures would hide
    # a real data/augmentation defect and silently rerun the epoch without
    # workers.  Keep this list limited to signatures emitted by PyTorch's
    # shared-memory path.
    return any(
        marker in message
        for marker in (
            "torch_shm_manager",
            "unexpected bus error encountered in worker",
            "insufficient shared memory (shm)",
            "unable to open shared memory object",
            "received 0 items of ancdata",
        )
    )


class _ResilientDataLoader:
    """DataLoader facade that retries worker *startup* in single-process mode.

    Workers are still the default on supported hosts.  If the first batch
    exposes a host-specific worker/shared-memory problem, restoring the exact
    generator state and retrying with ``num_workers=0`` keeps a run usable
    instead of failing after its model and datasets have already been built.
    """

    def __init__(self, dataset: Dataset, settings: dict):
        self._dataset = dataset
        self._settings = dict(settings)
        self._loader = self._make_loader(int(settings["num_workers"]))
        self._fell_back = False

    def _make_loader(self, num_workers: int) -> DataLoader:
        settings = dict(self._settings)
        settings["num_workers"] = num_workers
        if num_workers:
            return DataLoader(dataset=self._dataset, **settings)
        # These arguments are rejected by PyTorch when workers are disabled.
        settings.pop("persistent_workers", None)
        settings.pop("prefetch_factor", None)
        return DataLoader(dataset=self._dataset, **settings)

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator:
        if self._fell_back:
            yield from self._loader
            return

        generator = getattr(self._loader, "generator", None)
        generator_state = generator.get_state().clone() if generator is not None else None
        # DataLoader uses the process-global CPU generator when no explicit
        # generator is supplied.  Snapshot it too so a worker-startup retry
        # produces the same shuffle order for legacy callers that omit one.
        global_generator_state = (
            torch.random.get_rng_state() if generator is None else None
        )
        iterator = None
        try:
            iterator = iter(self._loader)
            first = next(iterator)
        except StopIteration:
            return
        except RuntimeError as error:
            if not _worker_startup_error(error):
                raise
            # Ensure persistent workers do not linger after a failed startup.
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                cast(Callable[[], None], shutdown)()
            if generator is not None and generator_state is not None:
                generator.set_state(generator_state)
            elif global_generator_state is not None:
                torch.random.set_rng_state(global_generator_state)
            self._loader = self._make_loader(0)
            self._fell_back = True
            logger.warning(
                "DataLoader workers failed to start (%s); retrying with "
                "num_workers=0",
                error,
            )
            yield from self._loader
            return

        yield first
        yield from iterator


def build_data_loader(
    dataset: Dataset,
    train_cfg: Mapping,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader | _ResilientDataLoader:
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

    settings = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
    }
    if num_workers:
        settings.update(
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=prefetch_factor,
        )
        return _ResilientDataLoader(dataset, settings)
    return DataLoader(dataset=dataset, **settings)
