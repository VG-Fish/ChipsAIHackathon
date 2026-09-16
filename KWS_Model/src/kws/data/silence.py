"""Synthetic sources for the Speech Commands ``_silence_`` class.

Both Google's benchmark and NeMo's ``process_speech_commands_data.py`` (the
script that produced the balanced manifests the SparkNet reference config
consumes) build ``_silence_`` the same way: take a one-second crop of a
``_background_noise_`` recording and scale it by a gain drawn uniformly from
[0, 1).  Silence is therefore quiet *background noise*, not digital zero.

The two differ only in when the draw happens.  NeMo materializes the clips
once, writing them to disk, so every split gets a fixed set and the reported
test accuracy is reproducible.  This project's deployment-oriented default
instead redraws a crop and gain on every access, because a fixed silence set
lets the model memorize a handful of clips and collapse to a high false-accept
rate in the field.  ``materialize_clips`` provides the former when a recipe
needs to reproduce a published number.
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

    def sample(self, rng: random.Random | None = None) -> torch.Tensor:
        """Draw one silence clip.

        ``rng`` makes the draw reproducible; without it the shared module-level
        stream is used, so every access yields a different clip.
        """
        draw = rng if rng is not None else random
        noise = draw.choice(self.noise_waveforms)
        max_start = max(noise.shape[1] - self.clip_len, 0)
        start = draw.randint(0, max_start)
        clip = noise[:, start:start + self.clip_len]
        if clip.shape[1] < self.clip_len:
            clip = torch.nn.functional.pad(clip, (0, self.clip_len - clip.shape[1]))
        gain = draw.uniform(self.min_gain, self.max_gain)
        return clip * gain

    def materialize_clips(self, count: int, rng: random.Random) -> list[torch.Tensor]:
        """Draw ``count`` clips once, standing in for NeMo's on-disk silence set.

        The caller supplies a split-specific ``rng`` so training, validation,
        and test silence come from disjoint draws, exactly as NeMo's script
        slices disjoint offsets of its shuffled silence pool into the three
        manifests.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.sample(rng) for _ in range(count)]
