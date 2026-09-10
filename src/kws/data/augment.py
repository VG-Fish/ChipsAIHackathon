"""Training-time augmentation: time-shift, background-noise mixing, SpecAugment,
speed/pitch perturbation. Applied only to the train split.
"""
import random
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from kws.data.audio_io import load_waveform


def time_shift(waveform: torch.Tensor, max_shift_samples: int) -> torch.Tensor:
    if max_shift_samples <= 0:
        return waveform
    shift = random.randint(-max_shift_samples, max_shift_samples)
    return torch.roll(waveform, shifts=shift, dims=-1)


def mix_background_noise(waveform: torch.Tensor, noise_waveforms: list[torch.Tensor],
                          snr_db_range: tuple[float, float]) -> torch.Tensor:
    noise = random.choice(noise_waveforms)
    clip_len = waveform.shape[-1]
    max_start = max(noise.shape[-1] - clip_len, 0)
    start = random.randint(0, max_start)
    noise_clip = noise[:, start:start + clip_len]
    if noise_clip.shape[-1] < clip_len:
        noise_clip = torch.nn.functional.pad(noise_clip, (0, clip_len - noise_clip.shape[-1]))

    snr_db = random.uniform(*snr_db_range)
    signal_power = waveform.pow(2).mean()
    noise_power = noise_clip.pow(2).mean().clamp_min(1e-10)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = torch.sqrt(target_noise_power / noise_power)
    return waveform + noise_clip * scale


def _resample_rate_step(sample_rate: int) -> int:
    """Use a 2.5% rate grid whose ratios reduce to small sinc kernels."""
    return max(round(sample_rate * 0.025), 1)


def _quantized_resample_rate(sample_rate: int, factor: float) -> int:
    """Approximate a speed factor with a nearby, computationally cheap rate."""
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    step = _resample_rate_step(sample_rate)
    return max(step, round((sample_rate / factor) / step) * step)


@lru_cache(maxsize=64)
def _cached_resampler(sample_rate: int, new_sample_rate: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(sample_rate, new_sample_rate)


def _candidate_resample_rates(sample_rate: int, factor_range: tuple[float, float]) -> list[int]:
    low_factor, high_factor = factor_range
    if low_factor <= 0 or high_factor < low_factor:
        raise ValueError("speed_factor_range must be positive and ordered")
    step = _resample_rate_step(sample_rate)
    lowest_rate = _quantized_resample_rate(sample_rate, high_factor)
    highest_rate = _quantized_resample_rate(sample_rate, low_factor)
    return list(range(lowest_rate, highest_rate + step, step))


def speed_perturb(waveform: torch.Tensor, sample_rate: int,
                  factor_range: tuple[float, float],
                  resamplers: dict[int, torchaudio.transforms.Resample] | None = None) -> torch.Tensor:
    """Resample to sample_rate/factor and reinterpret the result at the original
    sample_rate (i.e. don't resample back) -- this is what actually changes
    speed+pitch together. Resampling down and then immediately back up (the
    previous implementation) round-trips to ~the original signal, doing 2x the
    work for close to zero real augmentation effect.
    """
    factor = random.uniform(*factor_range)
    orig_len = waveform.shape[-1]
    new_sr = _quantized_resample_rate(sample_rate, factor)
    if new_sr == sample_rate:
        return waveform

    # Arbitrary integer sample-rate pairs can be nearly coprime, causing
    # torchaudio to build enormous sinc kernels for every example. Quantizing
    # to a small rational grid and reusing transforms avoids that pathological
    # setup cost while retaining several speed/pitch choices in the range.
    resampler = resamplers.get(new_sr) if resamplers is not None else None
    if resampler is None:
        resampler = _cached_resampler(sample_rate, new_sr)
    resampled = resampler(waveform)
    if resampled.shape[-1] > orig_len:
        resampled = resampled[:, :orig_len]
    else:
        resampled = torch.nn.functional.pad(resampled, (0, orig_len - resampled.shape[-1]))
    return resampled


class WaveformAugmenter:
    def __init__(self, background_noise_dir: Path, sample_rate: int,
                 max_shift_ms: float = 100, snr_db_range=(0, 15),
                 speed_factor_range=(0.9, 1.1)):
        self.sample_rate = sample_rate
        self.max_shift_samples = int(sample_rate * max_shift_ms / 1000)
        self.snr_db_range = snr_db_range
        self.speed_factor_range = speed_factor_range
        self.speed_resamplers = {
            rate: torchaudio.transforms.Resample(sample_rate, rate)
            for rate in _candidate_resample_rates(sample_rate, speed_factor_range)
            if rate != sample_rate
        }
        self.noise_waveforms = []
        for wav_path in sorted(Path(background_noise_dir).glob("*.wav")):
            self.noise_waveforms.append(load_waveform(wav_path, sample_rate))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = time_shift(waveform, self.max_shift_samples)
        waveform = speed_perturb(
            waveform, self.sample_rate, self.speed_factor_range, self.speed_resamplers,
        )
        if self.noise_waveforms and random.random() < 0.5:
            waveform = mix_background_noise(waveform, self.noise_waveforms, self.snr_db_range)
        return waveform


class SpecAugmenter:
    def __init__(self, time_mask_param: int = 20, freq_mask_param: int = 8, num_masks: int = 1):
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.num_masks = num_masks

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_masks):
            features = self.time_masking(features)
            features = self.freq_masking(features)
        return features
