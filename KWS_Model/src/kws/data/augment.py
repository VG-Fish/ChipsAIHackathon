"""Training-time augmentation: time-shift, background-noise mixing, SpecAugment,
speed/pitch perturbation, white noise. Applied only to the train split.

Without an ``augmentation`` block in the train config the recipe is the
original DS-CNN one: +/-150 ms circular shift, speed 0.85-1.15, background
noise with probability 0.75 at -5 to 15 dB SNR, and two SpecAugment mask
pairs. ``build_augmenters`` accepts a block that changes any of those, which
is how the SparkNet reference recipe (PLAN.md Finding 5) is expressed.
"""
import random
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

from kws.data.audio_io import load_waveform


TIME_SHIFT_MODES = ("roll", "zero_fill")


def time_shift(waveform: torch.Tensor, max_shift_samples: int, mode: str = "roll") -> torch.Tensor:
    """Shift by a uniform whole number of samples.

    ``roll`` wraps the displaced samples around. ``zero_fill`` drops them and
    pads the vacated end with zeros, as NeMo's ``ShiftPerturbation`` does.
    """
    if mode not in TIME_SHIFT_MODES:
        raise ValueError(f"unknown time shift mode {mode!r}")
    if max_shift_samples <= 0:
        return waveform
    shift = random.randint(-max_shift_samples, max_shift_samples)
    if mode == "roll":
        return torch.roll(waveform, shifts=shift, dims=-1)
    if shift == 0:
        return waveform
    shifted = torch.zeros_like(waveform)
    if shift > 0:
        shifted[..., shift:] = waveform[..., :-shift]
    else:
        shifted[..., :shift] = waveform[..., -shift:]
    return shifted


def add_white_noise(waveform: torch.Tensor, level_db_range: tuple[float, float]) -> torch.Tensor:
    """Add Gaussian noise whose standard deviation is ``10 ** (dB / 20)``.

    The level is relative to full scale, not to the signal, matching NeMo's
    ``WhiteNoisePerturbation``.
    """
    level_db = random.uniform(level_db_range[0], level_db_range[1])
    return waveform + torch.randn_like(waveform) * (10.0 ** (level_db / 20.0))


def mix_background_noise(waveform: torch.Tensor, noise_waveforms: list[torch.Tensor],
                          snr_db_range: tuple[float, float]) -> torch.Tensor:
    noise = random.choice(noise_waveforms)
    clip_len = waveform.shape[-1]
    max_start = max(noise.shape[-1] - clip_len, 0)
    start = random.randint(0, max_start)
    noise_clip = noise[:, start:start + clip_len]
    if noise_clip.shape[-1] < clip_len:
        noise_clip = torch.nn.functional.pad(noise_clip, (0, clip_len - noise_clip.shape[-1]))

    snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
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
    factor = random.uniform(factor_range[0], factor_range[1])
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


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


class WaveformAugmenter:
    """Shift, then speed perturbation, then background noise, then white noise.

    A probability of exactly 1 draws no random number, and white noise at
    probability 0 draws none either, so the default recipe consumes the same
    random stream as before these options existed.
    """

    def __init__(self, background_noise_dir: Path, sample_rate: int,
                 max_shift_ms: float = 150,
                 snr_db_range: tuple[float, float] = (-5.0, 15.0),
                 speed_factor_range: tuple[float, float] | None = (0.85, 1.15),
                 noise_probability: float = 0.75,
                 *, shift_mode: str = "roll", shift_probability: float = 1.0,
                 white_noise_probability: float = 0.0,
                 white_noise_db_range: tuple[float, float] = (-90.0, -46.0)):
        if shift_mode not in TIME_SHIFT_MODES:
            raise ValueError(f"unknown time shift mode {shift_mode!r}")
        _check_probability("noise_probability", noise_probability)
        _check_probability("shift_probability", shift_probability)
        _check_probability("white_noise_probability", white_noise_probability)
        self.sample_rate = sample_rate
        self.max_shift_samples = int(sample_rate * max_shift_ms / 1000)
        self.shift_mode = shift_mode
        self.shift_probability = shift_probability
        self.snr_db_range = snr_db_range
        self.speed_factor_range = speed_factor_range
        self.noise_probability = noise_probability
        self.white_noise_probability = white_noise_probability
        self.white_noise_db_range = tuple(white_noise_db_range)
        self.speed_resamplers = (
            {
                rate: torchaudio.transforms.Resample(sample_rate, rate)
                for rate in _candidate_resample_rates(sample_rate, speed_factor_range)
                if rate != sample_rate
            }
            if speed_factor_range is not None
            else {}
        )
        self.noise_waveforms = []
        if noise_probability > 0:
            for wav_path in sorted(Path(background_noise_dir).glob("*.wav")):
                self.noise_waveforms.append(load_waveform(wav_path, sample_rate))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.shift_probability >= 1.0 or random.random() < self.shift_probability:
            waveform = time_shift(waveform, self.max_shift_samples, self.shift_mode)
        if self.speed_factor_range is not None:
            waveform = speed_perturb(
                waveform, self.sample_rate, self.speed_factor_range, self.speed_resamplers,
            )
        if self.noise_waveforms and random.random() < self.noise_probability:
            waveform = mix_background_noise(waveform, self.noise_waveforms, self.snr_db_range)
        if self.white_noise_probability > 0 and random.random() < self.white_noise_probability:
            waveform = add_white_noise(waveform, self.white_noise_db_range)
        return waveform


class SpecAugmenter:
    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 10, num_masks: int = 2):
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.num_masks = num_masks

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_masks):
            features = self.time_masking(features)
            features = self.freq_masking(features)
        return features


AUGMENTATION_DEFAULTS = {
    "time_shift_ms": 150.0,
    "time_shift_mode": "roll",
    "time_shift_probability": 1.0,
    "speed_factor_range": [0.85, 1.15],
    "background_noise_probability": 0.75,
    "background_noise_snr_db_range": [-5.0, 15.0],
    "white_noise_probability": 0.0,
    "white_noise_db_range": [-90.0, -46.0],
    "spec_augment": True,
}


def build_augmenters(
    augmentation: Mapping | None,
    background_noise_dir: Path,
    sample_rate: int,
) -> tuple[WaveformAugmenter, SpecAugmenter | None]:
    """Build the training augmenters from a train config's ``augmentation`` block.

    Missing keys take ``AUGMENTATION_DEFAULTS``, the original recipe, so an
    absent block changes nothing. Unknown keys are rejected so a misspelled
    option cannot silently fall back to the default. ``speed_factor_range:
    null`` disables speed perturbation.
    """
    options = dict(augmentation or {})
    unknown = sorted(set(options).difference(AUGMENTATION_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown augmentation options: {unknown}")
    cfg = {**AUGMENTATION_DEFAULTS, **options}
    speed = cfg["speed_factor_range"]
    def float_pair(value: object, name: str) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must contain exactly two numbers")
        first, second = value
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        return float(first), float(second)

    waveform_augmenter = WaveformAugmenter(
        background_noise_dir,
        sample_rate,
        max_shift_ms=float(cfg["time_shift_ms"]),
        snr_db_range=float_pair(cfg["background_noise_snr_db_range"], "background_noise_snr_db_range"),
        speed_factor_range=float_pair(speed, "speed_factor_range") if speed is not None else None,
        noise_probability=float(cfg["background_noise_probability"]),
        shift_mode=cfg["time_shift_mode"],
        shift_probability=float(cfg["time_shift_probability"]),
        white_noise_probability=float(cfg["white_noise_probability"]),
        white_noise_db_range=float_pair(cfg["white_noise_db_range"], "white_noise_db_range"),
    )
    spec_augmenter = SpecAugmenter() if cfg["spec_augment"] else None
    return waveform_augmenter, spec_augmenter
