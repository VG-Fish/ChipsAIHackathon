import sys

import pytest
import torch
from torch.utils.data import TensorDataset

from kws.data.loader import build_data_loader


def test_loader_omits_worker_only_options_when_single_process():
    dataset = TensorDataset(torch.arange(8))
    loader = build_data_loader(
        dataset, {"batch_size": 2, "num_workers": 0}, shuffle=False
    )
    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_loader_configures_persistent_prefetch_workers():
    dataset = TensorDataset(torch.arange(8))
    loader = build_data_loader(
        dataset,
        {
            "batch_size": 2,
            "num_workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 3,
        },
        shuffle=True,
    )
    if sys.platform == "darwin":
        assert loader.num_workers == 0
        assert loader.persistent_workers is False
        assert loader.prefetch_factor is None
    else:
        assert loader.num_workers == 2
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": 1, "prefetch_factor": 0}, "prefetch_factor"),
    ],
)
def test_loader_rejects_invalid_worker_settings(overrides, message):
    dataset = TensorDataset(torch.arange(8))
    cfg = {"batch_size": 2, **overrides}
    with pytest.raises(ValueError, match=message):
        build_data_loader(dataset, cfg, shuffle=False)
