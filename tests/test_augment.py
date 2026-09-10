from pathlib import Path
from unittest.mock import patch

import torch
import torchaudio

from kws.data.augment import (
    SpecAugmenter,
    WaveformAugmenter,
    crop_or_pad,
    mix_background_noise,
    time_shift,
)

REAL_NOISE_DIR = Path("data/raw/speech_commands_v0.02/_background_noise_")
HAS_REAL_DATA = REAL_NOISE_DIR.is_dir()


# ---- time_shift ----

def test_time_shift_preserves_shape():
    waveform = torch.randn(1, 16000)
    shifted = time_shift(waveform, max_shift_samples=1600)
    assert shifted.shape == waveform.shape


def test_time_shift_zero_max_is_noop():
    waveform = torch.randn(1, 16000)
    shifted = time_shift(waveform, max_shift_samples=0)
    assert torch.equal(shifted, waveform)


def test_time_shift_actually_shifts_content():
    waveform = torch.zeros(1, 100)
    waveform[0, 0] = 1.0
    with patch("kws.data.augment.random.randint", return_value=7):
        shifted = time_shift(waveform, max_shift_samples=10)
    assert shifted[0, 7].item() == 1.0
    assert shifted.sum().item() == 1.0  # spike moved, didn't duplicate or vanish


# ---- mix_background_noise ----

def test_mix_background_noise_preserves_shape_and_is_finite():
    waveform = torch.randn(1, 16000) * 0.1
    noise = [torch.randn(1, 16000)]
    mixed = mix_background_noise(waveform, noise, snr_db_range=(0, 15))
    assert mixed.shape == waveform.shape
    assert torch.isfinite(mixed).all()


def test_mix_background_noise_handles_silent_signal():
    """Edge case: an all-zero (or near-silent) waveform must not produce NaN/Inf."""
    waveform = torch.zeros(1, 16000)
    noise = [torch.randn(1, 16000)]
    mixed = mix_background_noise(waveform, noise, snr_db_range=(0, 15))
    assert torch.isfinite(mixed).all()


def test_mix_background_noise_handles_silent_noise_clip():
    """Edge case: an all-zero noise clip (noise_power ~0) must not produce NaN/Inf,
    thanks to the clamp_min guard on noise_power.
    """
    waveform = torch.randn(1, 16000) * 0.1
    noise = [torch.zeros(1, 16000)]
    mixed = mix_background_noise(waveform, noise, snr_db_range=(0, 15))
    assert torch.isfinite(mixed).all()


def test_mix_background_noise_pads_short_noise_clip():
    waveform = torch.randn(1, 16000) * 0.1
    short_noise = [torch.randn(1, 100)]
    mixed = mix_background_noise(waveform, short_noise, snr_db_range=(0, 15))
    assert mixed.shape == waveform.shape
    assert torch.isfinite(mixed).all()


def test_mix_background_noise_snr_roughly_respected():
    """At a fixed, generous SNR, the added noise power should be small relative to
    signal power -- a coarse sanity check on the SNR scaling math, not an exact one.
    """
    torch.manual_seed(0)
    waveform = torch.ones(1, 16000) * 0.5  # constant signal, power = 0.25
    noise = [torch.randn(1, 16000)]
    mixed = mix_background_noise(waveform, noise, snr_db_range=(20, 20))  # high SNR -> quiet noise
    residual = mixed - waveform
    residual_power = residual.pow(2).mean().item()
    signal_power = waveform.pow(2).mean().item()
    # 20dB SNR => noise power should be ~1% of signal power
    assert residual_power < signal_power * 0.05


# ---- crop_or_pad ----

def test_crop_or_pad_crops_longer_input():
    waveform = torch.randn(1, 200)
    out = crop_or_pad(waveform, 100)
    assert out.shape == (1, 100)
    assert torch.equal(out, waveform[:, :100])


def test_crop_or_pad_pads_shorter_input():
    waveform = torch.randn(1, 50)
    out = crop_or_pad(waveform, 100)
    assert out.shape == (1, 100)
    assert torch.equal(out[:, :50], waveform)
    assert torch.equal(out[:, 50:], torch.zeros(1, 50))


def test_crop_or_pad_noop_when_already_target_length():
    waveform = torch.randn(1, 100)
    out = crop_or_pad(waveform, 100)
    assert torch.equal(out, waveform)


# ---- WaveformAugmenter's speed perturbation (fast, cached-kernel path) ----

def test_speed_perturbation_is_fast():
    """Regression test for the ~10,000x-slower bug: a hand-rolled resample with
    an arbitrary continuous ratio measured ~830ms/call (torchaudio.functional
    .resample has to build a huge polyphase filter for a "non-nice" ratio,
    which made a single epoch of augmented training take days instead of
    minutes). torchaudio's own SpeedPerturbation samples from a small fixed
    set of factors and caches one Resample kernel per factor, which must stay
    well under a millisecond per call.
    """
    import time

    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
    speed_perturbation = torchaudio.transforms.SpeedPerturbation(16000, [0.9, 0.95, 1.0, 1.05, 1.1])

    start = time.time()
    for _ in range(50):
        speed_perturbation(waveform)
    elapsed_per_call = (time.time() - start) / 50
    assert elapsed_per_call < 0.01, f"speed perturbation took {elapsed_per_call*1000:.2f}ms/call, expected <10ms"


def test_speed_perturbation_actually_changes_the_signal():
    waveform = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 16000)).unsqueeze(0)
    speed_perturbation = torchaudio.transforms.SpeedPerturbation(16000, [1.2])
    out, _ = speed_perturbation(waveform)
    out = crop_or_pad(out, waveform.shape[-1])
    assert not torch.allclose(out, waveform, atol=1e-3)


# ---- SpecAugmenter ----

def test_spec_augmenter_preserves_shape():
    features = torch.randn(1, 40, 101)
    augmenter = SpecAugmenter(time_mask_param=20, freq_mask_param=8)
    out = augmenter(features)
    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_spec_augmenter_actually_masks_something():
    torch.manual_seed(0)
    features = torch.ones(1, 40, 101)  # constant nonzero, so masked regions are detectable
    augmenter = SpecAugmenter(time_mask_param=20, freq_mask_param=8, num_masks=1)
    out = augmenter(features)
    assert not torch.equal(out, features), "SpecAugment produced no change at all"


# ---- WaveformAugmenter integration (real background noise files) ----

def test_waveform_augmenter_end_to_end_on_real_noise_files():
    if not HAS_REAL_DATA:
        import pytest
        pytest.skip("Speech Commands dataset not downloaded")

    augmenter = WaveformAugmenter(REAL_NOISE_DIR, sample_rate=16000)
    assert len(augmenter.noise_waveforms) == 6  # 6 official background noise clips

    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
    torch.manual_seed(0)
    for _ in range(20):  # run several times to exercise the 50% noise-mix branch
        out = augmenter(waveform)
        assert out.shape == waveform.shape
        assert torch.isfinite(out).all()


def test_waveform_augmenter_is_fast_enough_for_full_training():
    """A full M2 run needs ~22k samples/epoch x 40 epochs x 4 models. At the old
    ~830ms/call speed_perturb bottleneck, a single epoch of the smallest model
    was projected to take ~4 hours (days for the full run). The full augmenter
    call (time-shift + speed-perturb + possible noise-mix) must stay well under
    10ms/call for that to be remotely tractable.
    """
    import time

    if not HAS_REAL_DATA:
        import pytest
        pytest.skip("Speech Commands dataset not downloaded")

    augmenter = WaveformAugmenter(REAL_NOISE_DIR, sample_rate=16000)
    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

    start = time.time()
    n = 100
    for _ in range(n):
        augmenter(waveform)
    elapsed_per_call = (time.time() - start) / n
    assert elapsed_per_call < 0.01, f"WaveformAugmenter took {elapsed_per_call*1000:.2f}ms/call, expected <10ms"
