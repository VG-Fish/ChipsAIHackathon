from pathlib import Path
from unittest.mock import patch

import torch

from kws.data.augment import (
    SpecAugmenter,
    WaveformAugmenter,
    _cached_resampler,
    _candidate_resample_rates,
    _quantized_resample_rate,
    mix_background_noise,
    speed_perturb,
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


# ---- speed_perturb ----

def test_speed_perturb_preserves_length_both_directions():
    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
    for factor_range in [(1.1, 1.1), (0.9, 0.9)]:
        out = speed_perturb(waveform, sample_rate=16000, factor_range=factor_range)
        assert out.shape == waveform.shape
        assert torch.isfinite(out).all()


def test_speed_perturb_actually_changes_the_signal():
    """Regression test: resampling down then immediately back up round-trips to
    ~the original signal (no real speed change, and 2x the compute for nothing).
    The fix resamples once and reinterprets at the original rate, so the output
    must differ meaningfully from a plain round-trip identity.
    """
    waveform = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 1, 16000)).unsqueeze(0)
    out = speed_perturb(waveform, sample_rate=16000, factor_range=(1.2, 1.2))
    # A same-rate round trip would match the original almost exactly; a real
    # speed change should not.
    assert not torch.allclose(out, waveform, atol=1e-3)


def test_speed_perturb_uses_small_rational_rate_grid_and_cached_kernels():
    assert _candidate_resample_rates(16000, (0.9, 1.1)) == list(range(14400, 17601, 400))
    assert _quantized_resample_rate(16000, 1.1) == 14400
    assert _quantized_resample_rate(16000, 0.9) == 17600

    _cached_resampler.cache_clear()
    waveform = torch.randn(1, 16000)
    with patch("kws.data.augment.random.uniform", return_value=1.1):
        speed_perturb(waveform, 16000, (0.9, 1.1))
        speed_perturb(waveform, 16000, (0.9, 1.1))
    cache_info = _cached_resampler.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


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
