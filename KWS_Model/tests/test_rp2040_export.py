"""Tests for kws.export.rp2040 and the C engine it specifies."""

from __future__ import annotations

import copy
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from kws.export import rp2040
from kws.export.rp2040_pipeline import bin_to_uf2, host_parity
from kws.models.registry import build_model
from kws.optimize.grow_clean_rebuild import RebuiltDendriteModule, group_conv_with_batchnorm

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 4


def _sparknet(dendrite_blocks: tuple[int, ...] = (), fc_dendrite: bool = False) -> nn.Module:
    """A C4 SparkNet with random BatchNorm statistics, optionally grown."""
    cfg = yaml.safe_load((ROOT / f"configs/model/sparknet_c{WIDTH}_paper.yaml").read_text())
    torch.manual_seed(0)
    model = build_model(cfg, (32, 101), 12)
    if dendrite_blocks:
        group_conv_with_batchnorm(model, [f".blocks.{index}.pointwise" for index in dendrite_blocks])
    with torch.no_grad():
        for index in dendrite_blocks:
            main = model.blocks[index].pointwise
            dendrite = copy.deepcopy(main)
            for parameter in dendrite.parameters():
                parameter.normal_(0.0, 0.5)
            module = RebuiltDendriteModule([dendrite, main], WIDTH, [1, -1, 1, 1])
            module.skip_weights[0].normal_()
            model.blocks[index].pointwise = module
        if fc_dendrite:
            dendrite = copy.deepcopy(model.fc)
            for parameter in dendrite.parameters():
                parameter.normal_(0.0, 2.0)
            model.fc = RebuiltDendriteModule([dendrite, model.fc], model.fc.out_features, [1, -1])
            model.fc.skip_weights[0].normal_()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.normal_(0.0, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0.0, 0.2)
    return model.eval()


def _features(count: int = 16, seed: int = 0) -> np.ndarray:
    return 10.0 * np.random.default_rng(seed).standard_normal((count, 32, 101))


def _quantized(model: nn.Module, act_bits: int) -> tuple[rp2040.FoldedSparkNet, rp2040.QuantizedSparkNet]:
    folded = rp2040.fold_sparknet(model)
    calibration = rp2040.calibrate(folded, [_features(32, seed=1)], "max")
    return folded, rp2040.quantize(folded, calibration, act_bits=act_bits)


# (pointwise dendrite blocks, fc dendrite)
GROWN = [((), False), ((2,), False), ((1, 2, 3), False), ((), True), ((2,), True)]


@pytest.mark.parametrize("grown", GROWN)
def test_folding_reproduces_the_torch_model(grown):
    model = _sparknet(*grown).double()
    features = _features()
    folded = rp2040.fold_sparknet(model)

    with torch.no_grad():
        expected = model(torch.from_numpy(features).unsqueeze(1)).numpy()
    assert np.abs(folded.forward(features) - expected).max() < 1e-9
    assert folded.parameter_count < sum(p.numel() for p in model.parameters())  # BatchNorms are gone


def test_folding_needs_eval_mode():
    with pytest.raises(ValueError, match="eval-mode"):
        rp2040.fold_sparknet(_sparknet().train())


@pytest.mark.parametrize("grown", GROWN)
def test_integer_logits_track_the_float_model(grown):
    folded, quantized = _quantized(_sparknet(*grown), act_bits=16)
    features = _features(seed=2)

    expected = folded.forward(features)
    logits = quantized.logits_float(quantized.forward_int(quantized.quantize_input(features)))
    assert np.abs(logits - expected).max() < 0.05 * np.abs(expected).max()


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
@pytest.mark.parametrize("act_bits", [8, 16])
@pytest.mark.parametrize("grown", GROWN)
def test_the_c_engine_matches_the_numpy_reference_bit_for_bit(tmp_path, act_bits, grown):
    _, quantized = _quantized(_sparknet(*grown), act_bits)
    header = rp2040.write_c_model(quantized, tmp_path / "model_under_test.h", "model_under_test")
    xq = quantized.quantize_input(_features(seed=3))
    # saturated clips exercise every clamp in the engine
    xq = np.concatenate([xq, np.full((1, 32, 101), quantized.act_max), np.full((1, 32, 101), -quantized.act_max)])

    parity = host_parity(quantized, header, "model_under_test", xq, quantized.forward_int(xq), "cc")

    assert parity["bit_exact"], parity


def test_requantization_rounds_to_nearest():
    multiplier, bias = 0.3721, 5.25
    rq = rp2040.make_requant([np.array([multiplier])], np.array([bias]))
    acc = np.arange(-5000, 5000, dtype=np.int64)[None, None, :]

    result = rp2040._requant([acc], rq)[0, 0]
    assert np.abs(result - (acc[0, 0] * multiplier + bias)).max() <= 0.5 + 1e-6


def test_the_fc_dendrite_is_folded_and_quantized():
    folded, quantized = _quantized(_sparknet(fc_dendrite=True), act_bits=16)

    assert folded.fc_den.shape == folded.fc_w.shape and folded.fc_den_skip.shape == (12,)
    assert quantized.fc_den_w.shape == quantized.fc_w.shape and len(quantized.fc_rq.multipliers) == 2
    assert "fc.den" in rp2040.check_integer_bounds(quantized)
    # the dendrite term reaches the logits: zeroing its skip weights changes them
    xq = quantized.quantize_input(_features(seed=4))
    quantized.fc_rq.multipliers[1][:] = 0
    assert not np.array_equal(quantized.forward_int(xq), _quantized(_sparknet(fc_dendrite=True), 16)[1].forward_int(xq))


def test_the_bounds_check_rejects_an_overflowing_model():
    _, quantized = _quantized(_sparknet(), act_bits=16)
    assert set(rp2040.check_integer_bounds(quantized)) >= {"b0.dw", "b3.out", "gate", "fc"}

    quantized.fc_rq.multipliers[0][:] = 2 ** 31
    with pytest.raises(OverflowError):
        rp2040.check_integer_bounds(quantized)


def test_uf2_blocks_cover_the_image_at_the_flash_base():
    image = bytes(range(256)) + b"\xaa" * 44
    uf2 = bin_to_uf2(image)

    assert len(uf2) == 2 * 512
    for index in range(2):
        block = uf2[index * 512:(index + 1) * 512]
        magic0, magic1, flags, address, size, number, count, family = struct.unpack("<8I", block[:32])
        assert (magic0, magic1, struct.unpack("<I", block[-4:])[0]) == (0x0A324655, 0x9E5D5157, 0x0AB16F30)
        assert (flags, address, size, number, count, family) == (0x2000, 0x10000000 + 256 * index, 256, index, 2, 0xE48BFF56)
    assert uf2[32:32 + 256] == image[:256]
    assert uf2[512 + 32:512 + 32 + 44] == image[256:]
