import random
from collections.abc import Mapping

import numpy as np
import torch


def resolve_seed(config: Mapping, override: int | None = None) -> int:
    """Return a validated effective seed from a config and optional CLI override."""
    value = config.get("seed", 0) if override is None else override
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def with_seed(config: dict, override: int | None = None) -> dict:
    """Copy ``config`` with its seed replaced when a CLI override is supplied."""
    if override is None:
        return config
    updated = dict(config)
    updated["seed"] = resolve_seed(config, override)
    return updated


def set_seed(seed: int) -> None:
    seed = resolve_seed({}, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Capture process RNGs for a resumable epoch-boundary checkpoint."""
    from kws.utils.checkpointing import capture_rng_state as _capture

    return _capture()


def restore_rng_state(state: dict) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    from kws.utils.checkpointing import restore_rng_state as _restore

    _restore(state)
