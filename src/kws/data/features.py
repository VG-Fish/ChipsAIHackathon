"""Audio feature extraction: log-mel filterbank energies (default) or MFCC."""
import torch
import torchaudio


class FeatureExtractor:
    def __init__(self, sample_rate: int, n_mels: int, win_length_ms: float,
                 hop_length_ms: float, feature_type: str = "logmel"):
        self.feature_type = feature_type
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        if feature_type == "mfcc":
            self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mels,
                melkwargs=dict(n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels),
            )
        elif feature_type != "logmel":
            raise ValueError(f"Unknown feature_type: {feature_type}")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (1, num_samples) -> features: (1, n_mels, time)."""
        if self.feature_type == "mfcc":
            return self.mfcc(waveform)
        return self.amplitude_to_db(self.mel_spec(waveform))


def build_feature_extractor(data_cfg: dict) -> FeatureExtractor:
    feat_cfg = data_cfg["features"]
    return FeatureExtractor(
        sample_rate=data_cfg["dataset"]["sample_rate"],
        n_mels=feat_cfg["n_mels"],
        win_length_ms=feat_cfg["win_length_ms"],
        hop_length_ms=feat_cfg["hop_length_ms"],
        feature_type=feat_cfg["type"],
    )
