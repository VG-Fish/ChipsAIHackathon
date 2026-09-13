import pytest
import torch
from torch.utils.data import TensorDataset

from kws.data.loader import _worker_startup_error, build_data_loader


@pytest.mark.parametrize(
    "message",
    [
        "torch_shm_manager at '/tmp/torch_shm_manager': Operation not permitted",
        "Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).",
        "unable to open shared memory object </torch_123> in read-write mode",
        "received 0 items of ancdata",
    ],
)
def test_worker_fallback_only_matches_shared_memory_failures(message):
    assert _worker_startup_error(RuntimeError(message))


def test_worker_fallback_does_not_hide_dataset_error():
    error = RuntimeError(
        "Caught RuntimeError in DataLoader worker process 0. "
        "Original Traceback: ValueError: malformed augmented waveform"
    )
    assert not _worker_startup_error(error)


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
    # Worker startup is attempted on every host.  The loader may switch to a
    # single process only after the first batch if the host cannot start
    # torch_shm_manager (that fallback is exercised by the runtime path).
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
