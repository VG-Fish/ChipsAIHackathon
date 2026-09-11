import pytest
import torch
import torch.nn as nn

from kws.models.ds_cnn import DSCNN
from kws.utils.profile import (
    count_macs,
    measure_latency,
    measure_peak_activation_bytes,
    profile_model,
    weight_memory_bytes,
)


def test_conv_macs_match_the_hand_computed_count():
    model = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
    # 'same' padding keeps 40x98, so 4 * 40 * 98 outputs, each a 1*3*3 reduction.
    assert count_macs(model, (40, 98)) == 4 * 40 * 98 * 9


def test_depthwise_macs_account_for_groups():
    # The profiler probes with the deployment input shape (one spectrogram
    # channel), so the grouped conv is measured behind a widening stem.
    stem = nn.Conv2d(1, 8, kernel_size=1, bias=False)
    stem_macs = count_macs(stem, (10, 10))
    grouped = nn.Sequential(stem, nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False))
    dense = nn.Sequential(stem, nn.Conv2d(8, 8, 3, padding=1, bias=False))

    # A depthwise conv reduces over one input channel, not all eight.
    assert (count_macs(dense, (10, 10)) - stem_macs) == 8 * (
        count_macs(grouped, (10, 10)) - stem_macs
    )


def test_dendrite_style_copies_are_counted_because_they_run_on_device():
    class TwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Conv2d(1, 4, 3, padding=1, bias=False)
            self.branch = nn.Conv2d(1, 4, 3, padding=1, bias=False)

        def forward(self, x):
            return self.main(x) + self.branch(x)

    single = count_macs(nn.Conv2d(1, 4, 3, padding=1, bias=False), (10, 10))
    assert count_macs(TwoBranch(), (10, 10)) == 2 * single


def test_weight_memory_scales_with_the_deployed_precision():
    model = nn.Linear(10, 10, bias=False)  # 100 parameters
    assert weight_memory_bytes(model, bits_per_weight=32) == 400
    assert weight_memory_bytes(model, bits_per_weight=8) == 100
    assert weight_memory_bytes(model, bits_per_weight=4) == 50


def test_peak_activation_is_the_largest_adjacent_pair_not_the_sum():
    model = nn.Sequential(nn.Flatten(), nn.Linear(40 * 98, 8), nn.Linear(8, 4))
    peak = measure_peak_activation_bytes(model, (40, 98), bytes_per_activation=1)

    # The first Linear dominates: 3920 in + 8 out. A running total of every
    # tensor would be far larger; an MCU arena only holds one pair at a time.
    assert peak == 40 * 98 + 8


def test_profile_reports_every_axis_the_frontier_compares():
    model = DSCNN(input_shape=(40, 98), num_classes=6, initial_channels=18,
                  initial_kernel=5, initial_stride=2, block_channels=[18, 18], dropout=0.2)
    cost = profile_model(model, (40, 98), latency_iterations=3, latency_warmup=1,
                         bits_per_weight=8)

    assert cost.params == 1716
    assert cost.weight_bytes == 1716  # 8-bit weights: one byte each
    assert cost.macs > 0 and cost.activation_peak_bytes > 0
    assert cost.latency_ms_p50 > 0
    assert set(cost.as_dict()) >= {"params", "macs", "weight_bytes", "latency_ms_p50"}
    assert cost.weight_memory_method == "projected_logical_precision"
    assert cost.activation_memory_method == "forward_hook_sequential_liveness_estimate"


def test_profiling_restores_training_mode():
    model = nn.Linear(4, 4).train()
    count_macs(model, (2, 4))
    measure_latency(model, (2, 4), iterations=2, warmup=0)
    assert model.training


def test_latency_rejects_empty_measurements():
    with pytest.raises(ValueError, match="positive"):
        measure_latency(nn.Linear(4, 4), (2, 4), iterations=0)


def test_quantized_graph_reports_packed_weights_and_compute():
    from kws.optimize.quantization_compat import (
        convert,
        get_default_qat_qconfig,
        prepare_qat,
    )
    from kws.optimize.quantize_qat import QATWrapper, _fuse_conv_bn_for_qat, _select_backend

    model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
    _fuse_conv_bn_for_qat(model)
    backend = _select_backend()
    torch.backends.quantized.engine = backend
    wrapped = QATWrapper(model)
    wrapped.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(wrapped, inplace=True)
    wrapped(torch.zeros(1, 1, 4, 4))  # calibrate observers before convert
    quantized = convert(wrapped.eval(), inplace=False)

    cost = profile_model(quantized, (4, 4), bits_per_weight=8,
                         latency_iterations=1, latency_warmup=0)
    assert cost.macs > 0
    assert cost.weight_bytes > 0


def test_realistic_dendritic_residual_survives_int8_conversion():
    from kws.optimize.quantize_qat import (
        QATWrapper,
        _select_backend,
        replace_clean_pai_modules,
    )
    from kws.optimize.quantization_compat import (
        convert,
        get_default_qat_qconfig,
        prepare_qat,
    )

    class FakeCleanPAI(nn.Module):
        def __init__(self):
            super().__init__()
            # PAI stores selected dendrites first and the base branch last.
            self.layer_array = nn.ModuleList(
                [nn.Conv2d(1, 2, 1), nn.Conv2d(1, 2, 1)]
            )
            self.skip_weights = nn.ParameterList(
                [nn.Parameter(torch.tensor([[0.25, -0.5]]))]
            )
            self.processor_array = [None, None]
            self.register_buffer("view_tuple", torch.tensor([1, -1, 1, 1]))

        def forward(self, x):
            dendrite = torch.tanh(self.layer_array[0](x))
            scale = self.skip_weights[0][0].view(1, -1, 1, 1)
            return self.layer_array[1](x) + scale * dendrite

    pai = FakeCleanPAI().eval()
    model = nn.Sequential(pai)
    sample = torch.randn(2, 1, 4, 4)
    with torch.no_grad():
        expected = model(sample)

    assert replace_clean_pai_modules(model) == ["0"]
    with torch.no_grad():
        actual = model(sample)
    assert torch.allclose(actual, expected, atol=1e-6)

    backend = _select_backend()
    torch.backends.quantized.engine = backend
    wrapped = QATWrapper(model)
    wrapped.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(wrapped.train(), inplace=True)
    wrapped(sample)  # calibrate observers before convert
    quantized = convert(wrapped.eval(), inplace=False)

    with torch.no_grad():
        output = quantized(sample[:1])
    assert output.shape == (1, 2, 4, 4)

    # Packed conv weights plus the FP32 per-channel skip coefficients.
    assert weight_memory_bytes(quantized, bits_per_weight=8) >= 2 * 2 + 2 * 4
