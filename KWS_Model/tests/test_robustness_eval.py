import math
import random

import torch

from scripts import robustness_eval as rob


def test_mix_at_snr_hits_the_requested_snr():
    torch.manual_seed(0)
    wave = torch.randn(1, 16000) * 0.1
    crop = torch.randn(1, 16000) * 3.0
    for snr in (20.0, 0.0):
        added = rob.mix_at_snr(wave, crop, snr) - wave
        realized = 10 * math.log10(float(wave.pow(2).mean() / added.pow(2).mean()))
        assert abs(realized - snr) < 1e-3


def test_noise_crop_is_deterministic_per_clip():
    bank = [torch.arange(40000.0).view(1, -1), torch.ones(1, 20000)]
    a = rob.noise_crop(bank, 16000, random.Random("robustness-bgnoise:0:7"))
    b = rob.noise_crop(bank, 16000, random.Random("robustness-bgnoise:0:7"))
    assert a[:2] == b[:2] and torch.equal(a[2], b[2])


def test_per_channel_quantization_commutes_with_channel_scale():
    # Why BN folding is a no-op for per-output-channel quantization.
    torch.manual_seed(0)
    weight = torch.randn(6, 4, 1, 3)
    scale = torch.tensor([0.5, -2.0, 3.0, 1.0, -0.1, 7.0]).view(-1, 1, 1, 1)
    folded = rob.fake_quantize(weight * scale, 4, per_channel=True)
    assert torch.allclose(folded, rob.fake_quantize(weight, 4, per_channel=True) * scale, atol=1e-6)


def test_quantization_level_count():
    weight = torch.linspace(-1, 1, 1001).view(1, -1)
    for bits in (8, 6, 4):
        levels = rob.fake_quantize(weight, bits, per_channel=False).unique().numel()
        assert levels == 2 ** bits - 1


def test_ece_is_zero_when_confidence_equals_accuracy():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert rob.expected_calibration_error(probs, torch.tensor([0, 1])) == 0.0
    assert abs(rob.expected_calibration_error(probs, torch.tensor([1, 1])) - 0.5) < 1e-12
