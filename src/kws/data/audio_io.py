"""WAV loading via `soundfile`, not `torchaudio.load`.

Recent torchaudio versions route load()/save() through TorchCodec, which needs a
system-installed FFmpeg (not just a pip package) -- an extra manual install step
that's a real barrier on Windows in particular. `soundfile` ships libsndfile as
a self-contained wheel on Windows/Mac/Linux, so it needs nothing beyond `pip
install`, and Speech Commands audio is plain 16-bit PCM WAV, well within what
libsndfile handles. torchaudio itself is still used for everything that's pure
tensor math (resampling, MelSpectrogram/MFCC, SpecAugment) since none of that
touches TorchCodec.
"""
import soundfile as sf
import torch
import torchaudio


def load_waveform(path, target_sample_rate: int) -> torch.Tensor:
    """Returns a (1, num_samples) float32 tensor, resampled to target_sample_rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
    return waveform
