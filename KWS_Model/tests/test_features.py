import torch

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
