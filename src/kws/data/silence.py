"""Synthesized "silence" class: fresh random crops of background noise every epoch.

A fixed/cached silence set lets the model memorize a handful of clips and
collapse to a high false-accept rate in real deployment -- so this sampler
draws a new random crop + gain every time it's called, never caching results.
"""
import random
from pathlib import Path

import torch

from kws.data.audio_io import load_waveform


class SilenceSampler:
    def __init__(self, background_noise_dir: Path, sample_rate: int, clip_seconds: float,
                 min_gain: float, max_gain: float):
        self.clip_len = int(sample_rate * clip_seconds)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.noise_waveforms = []
        for wav_path in sorted(Path(background_noise_dir).glob("*.wav")):
            self.noise_waveforms.append(load_waveform(wav_path, sample_rate))
        if not self.noise_waveforms:
            raise ValueError(f"No background noise .wav files found in {background_noise_dir}")

    def sample(self) -> torch.Tensor:
        noise = random.choice(self.noise_waveforms)
        max_start = max(noise.shape[1] - self.clip_len, 0)
        start = random.randint(0, max_start)
        clip = noise[:, start:start + self.clip_len]
        if clip.shape[1] < self.clip_len:
            clip = torch.nn.functional.pad(clip, (0, self.clip_len - clip.shape[1]))
        gain = random.uniform(self.min_gain, self.max_gain)
        return clip * gain
