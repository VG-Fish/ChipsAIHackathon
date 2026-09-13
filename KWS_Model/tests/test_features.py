import torch
import torchaudio
from typing import Any, cast

from kws.data.features import FeatureExtractor


def test_logmel_output_shape():
    extractor = FeatureExtractor(sample_rate=16000, n_mels=40, win_length_ms=30, hop_length_ms=10)
    waveform = torch.randn(1, 16000)  # 1 second at 16kHz
    features = extractor(waveform)
    assert features.shape[0] == 1
    assert features.shape[1] == 40


def test_logmel_deterministic():
    extractor = FeatureExtractor(sample_rate=16000, n_mels=40, win_length_ms=30, hop_length_ms=10)
    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
    out1 = extractor(waveform)
    out2 = extractor(waveform)
    assert torch.allclose(out1, out2)


def test_mfcc_output_shape():
    extractor = FeatureExtractor(sample_rate=16000, n_mels=40, win_length_ms=30, hop_length_ms=10,
                                  feature_type="mfcc")
    waveform = torch.randn(1, 16000)
    features = extractor(waveform)
    assert features.shape[0] == 1
    assert features.shape[1] == 40


def test_mfcc_log_mels_matches_torchaudio_reference():
    # This is the SparkNet checkpoints' front end (PLAN.md Finding 5): 32 mel
    # bins/coefficients, a 25 ms window, and a 10 ms hop at 16 kHz.
    extractor = FeatureExtractor(sample_rate=16000, n_mels=32, win_length_ms=25, hop_length_ms=10,
                                  feature_type="mfcc", log_mels=True)
    mfcc = cast(Any, torchaudio.transforms.MFCC)
    reference = mfcc(
        sample_rate=16000, n_mfcc=32, log_mels=True,
        melkwargs=cast(Any, {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 32,
        }),
    )
    waveform = torch.randn(1, 16000)
    assert torch.allclose(extractor(waveform), reference(waveform))


def test_mfcc_log_mels_differs_from_default():
    log_mels_true = FeatureExtractor(sample_rate=16000, n_mels=32, win_length_ms=25, hop_length_ms=10,
                                      feature_type="mfcc", log_mels=True)
    log_mels_false = FeatureExtractor(sample_rate=16000, n_mels=32, win_length_ms=25, hop_length_ms=10,
                                       feature_type="mfcc", log_mels=False)
    waveform = torch.randn(1, 16000)
    assert not torch.allclose(log_mels_true(waveform), log_mels_false(waveform))
