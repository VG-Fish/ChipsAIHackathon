import random

import numpy as np
import pytest
import torch

from kws.utils.seed import set_seed, with_seed


def test_with_seed_overrides_without_mutating_the_config():
    config = {"seed": 3, "epochs": 2}

    updated = with_seed(config, 17)

    assert config["seed"] == 3
    assert updated["seed"] == 17
    assert updated["epochs"] == 2


def test_set_seed_repeats_python_numpy_and_torch_streams():
    set_seed(23)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(23)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        with_seed({"seed": 0}, -1)

    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)
