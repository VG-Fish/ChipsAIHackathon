"""Training-time augmentation: time-shift, background-noise mixing, SpecAugment,
speed/pitch perturbation. Applied only to the train split.
"""
import random
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


def crop_or_pad(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    if waveform.shape[-1] > target_len:
        return waveform[:, :target_len]
    if waveform.shape[-1] < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
    return waveform


class WaveformAugmenter:
    def __init__(self, background_noise_dir: Path, sample_rate: int,
                 max_shift_ms: float = 100, snr_db_range=(0, 15),
                 speed_factors=(0.9, 0.95, 1.0, 1.05, 1.1)):
        self.sample_rate = sample_rate
        self.max_shift_samples = int(sample_rate * max_shift_ms / 1000)
        self.snr_db_range = snr_db_range
        # torchaudio's own SpeedPerturbation samples from a small fixed set of
        # factors and caches one Resample kernel per factor at construction time.
        # A hand-rolled version calling torchaudio.functional.resample with an
        # arbitrary continuous ratio measured at ~830ms/call here -- resample
        # has to build a huge polyphase filter when the ratio isn't a "nice"
        # rational number, which a random continuous factor almost never is.
        # This cached version measured ~0.08ms/call: a ~10,000x difference,
        # which is the entire reason a single epoch of augmented training was
        # projected to take days instead of minutes.
        self.speed_perturbation = torchaudio.transforms.SpeedPerturbation(sample_rate, list(speed_factors))
        self.noise_waveforms = []
        for wav_path in sorted(Path(background_noise_dir).glob("*.wav")):
            self.noise_waveforms.append(load_waveform(wav_path, sample_rate))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        orig_len = waveform.shape[-1]
        waveform = time_shift(waveform, self.max_shift_samples)
        waveform, _ = self.speed_perturbation(waveform)
        waveform = crop_or_pad(waveform, orig_len)
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
