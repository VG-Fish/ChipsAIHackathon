import random
from pathlib import Path

import pytest
import soundfile as sf
import torch
import yaml

from kws.data.silence import SilenceSampler
from kws.train import build_lr_scheduler, build_optimizer


ROOT = Path(__file__).parents[1]


@pytest.fixture
def noise_dir(tmp_path):
    """Two distinguishable `_background_noise_` recordings."""
    directory = tmp_path / "_background_noise_"
    directory.mkdir()
    generator = torch.Generator().manual_seed(0)
    for name in ("doing_the_dishes", "white_noise"):
        waveform = torch.randn(16_000 * 4, generator=generator) * 0.1
        sf.write(directory / f"{name}.wav", waveform.numpy(), 16_000)
    return directory


def _sampler(noise_dir):
    return SilenceSampler(
        noise_dir, sample_rate=16_000, clip_seconds=1.0, min_gain=0.0, max_gain=1.0,
    )


def test_silence_is_gain_scaled_background_noise_not_digital_zero(noise_dir):
    """NeMo builds _silence_ by scaling a one-second crop, so it is not zeros."""
    clip = _sampler(noise_dir).sample(random.Random(0))
    assert clip.shape == (1, 16_000)
    assert torch.count_nonzero(clip) > 0
    assert clip.abs().max() <= 0.1


def test_materialized_silence_is_fixed_and_split_disjoint(noise_dir):
    """A fixed on-disk silence set is what makes the reported accuracy reproducible."""
    sampler = _sampler(noise_dir)

    validation = sampler.materialize_clips(8, random.Random("0:silence:validation"))
    again = sampler.materialize_clips(8, random.Random("0:silence:validation"))
    testing = sampler.materialize_clips(8, random.Random("0:silence:testing"))

    assert len(validation) == 8
    assert all(clip.shape == (1, 16_000) for clip in validation)
    # Same seed -> same clips, so repeated evaluation sees identical silence.
    assert all(torch.equal(a, b) for a, b in zip(validation, again))
    # Different split -> different draw, mirroring NeMo's disjoint offsets.
    assert not all(torch.equal(a, b) for a, b in zip(validation, testing))


def test_unmaterialized_silence_redraws_on_every_access(noise_dir):
    sampler = _sampler(noise_dir)
    assert not torch.equal(sampler.sample(), sampler.sample())


def test_paper_recipe_configs_encode_the_reported_setup():
    data = yaml.safe_load((ROOT / "configs/data/speech_commands_v2_mfcc32_paper.yaml").read_text())
    model = yaml.safe_load((ROOT / "configs/model/sparknet_c16_paper.yaml").read_text())
    train = yaml.safe_load((ROOT / "configs/train/sparknet_c16_paper.yaml").read_text())

    # The balanced manifests were never released, so the recipe reproduces the
    # rules of the script that made them rather than copying its output.
    assert data["unknown"]["target_ratio_to_avg_keyword_count"] == 1.0
    assert data["unknown"]["rounding"] == "ceil"
    assert data["silence"]["target_ratio_to_avg_keyword_count"] == 1.0
    assert data["silence"]["rounding"] == "ceil"
    assert data["silence"]["background_noise_dir"] == "_background_noise_"
    assert data["silence"]["materialize"] is True
    assert (data["silence"]["min_gain"], data["silence"]["max_gain"]) == (0.0, 1.0)
    assert data["features"] == {
        "type": "mfcc", "n_mels": 32, "win_length_ms": 25,
        "hop_length_ms": 10, "log_mels": True,
    }
    assert model == {
        "family": "sparknet", "name": "sparknet_c16_paper", "channels": 16,
        "gate_channels": 32, "sparsity_weight": 1.0,
    }
    assert train["optimizer"] == "sgd"
    assert train["momentum"] == 0.9
    assert train["task_loss_scale"] == 100.0
    assert train["scheduler"] == "polynomial_hold"
    assert train["warmup_fraction"] == 0.05
    assert train["hold_fraction"] == 0.40
    assert train["polynomial_power"] == 2.0
    assert train["min_lr"] == 1e-6
    # NeMo's WhiteNoisePerturbation draws whole decibels, not a continuous level.
    assert train["augmentation"]["white_noise_integer_db"] is True
    assert train["augmentation"]["white_noise_db_range"] == [-90, -46]


def test_paper_optimizer_and_schedule_match_nemo_polynomial_hold():
    parameter = torch.nn.Parameter(torch.ones(()))
    config = {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.001, "momentum": 0.9}
    optimizer = build_optimizer([parameter], config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9

    scheduler = build_lr_scheduler(
        optimizer, total_steps=1_000, warmup_fraction=0.05,
        name="polynomial_hold", hold_fraction=0.40, min_lr=1e-6, power=2.0,
    )
    # NeMo starts its scheduler at step zero, so the first warmup LR is
    # nonzero: max_lr * (0 + 1) / (warmup_steps + 1).
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 / 51)

    for _ in range(450):
        optimizer.step()
        scheduler.step()
    # warmup=50, absolute hold boundary=450; decay begins at the peak.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    for _ in range(550):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
