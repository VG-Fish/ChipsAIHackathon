import torch
import torch.nn.functional as F

from kws.evaluate import load_model_from_checkpoint
from kws.models.sparknet import SparkNet, build_sparknet
from kws.models.sparknet_port import map_state_dict
from kws.utils.profile import count_macs

KERNELS = (11, 15, 19, 29)


def _build_random_reference_state_dict(n_feat: int, channels: int, gate_channels: int,
                                        num_classes: int) -> dict:
    """A random reference-format state dict, shaped like the released checkpoints."""
    def randn(*shape):
        return torch.randn(*shape)

    def positive(*shape):
        # Running variances must stay positive for F.batch_norm.
        return torch.rand(*shape) + 0.1

    source: dict[str, torch.Tensor] = {}
    in_channels = n_feat
    for i, kernel_size in enumerate(KERNELS):
        source[f"fs.encoder.{i}.mconv.0.weight"] = randn(in_channels, 1, kernel_size)
        source[f"fs.encoder.{i}.mconv.1.weight"] = randn(channels, in_channels, 1)
        source[f"fs.encoder.{i}.mconv.2.weight"] = randn(channels)
        source[f"fs.encoder.{i}.mconv.2.bias"] = randn(channels)
        source[f"fs.encoder.{i}.mconv.2.running_mean"] = randn(channels)
        source[f"fs.encoder.{i}.mconv.2.running_var"] = positive(channels)
        if i > 0:
            source[f"fs.encoder.{i}.res.0.0.weight"] = randn(channels, in_channels, 1)
            source[f"fs.encoder.{i}.res.0.1.weight"] = randn(channels)
            source[f"fs.encoder.{i}.res.0.1.bias"] = randn(channels)
            source[f"fs.encoder.{i}.res.0.1.running_mean"] = randn(channels)
            source[f"fs.encoder.{i}.res.0.1.running_var"] = positive(channels)
        in_channels = channels

    source["output_layer.0.weight"] = randn(gate_channels, channels, 1)
    source["output_layer.0.bias"] = randn(gate_channels)
    source["output_layer.1.weight"] = randn(gate_channels)
    source["output_layer.1.bias"] = randn(gate_channels)
    source["output_layer.1.running_mean"] = randn(gate_channels)
    source["output_layer.1.running_var"] = positive(gate_channels)
    source["freq_linear_proj.weight"] = randn(num_classes, gate_channels)
    source["freq_linear_proj.bias"] = randn(num_classes)
    return source


def _reference_forward(x: torch.Tensor, source: dict, n_feat: int, channels: int) -> torch.Tensor:
    """Independent forward pass with F.conv1d/F.batch_norm, not going through SparkNet."""
    h = x.squeeze(1)  # (B, 1, F, T) -> (B, F, T)
    in_channels = n_feat
    for i, kernel_size in enumerate(KERNELS):
        main = F.conv1d(h, source[f"fs.encoder.{i}.mconv.0.weight"],
                         padding=kernel_size // 2, groups=in_channels)
        main = F.conv1d(main, source[f"fs.encoder.{i}.mconv.1.weight"])
        main = F.batch_norm(
            main, source[f"fs.encoder.{i}.mconv.2.running_mean"],
            source[f"fs.encoder.{i}.mconv.2.running_var"],
            source[f"fs.encoder.{i}.mconv.2.weight"], source[f"fs.encoder.{i}.mconv.2.bias"],
            training=False, eps=1e-3,
        )
        if i > 0:
            res = F.conv1d(h, source[f"fs.encoder.{i}.res.0.0.weight"])
            res = F.batch_norm(
                res, source[f"fs.encoder.{i}.res.0.1.running_mean"],
                source[f"fs.encoder.{i}.res.0.1.running_var"],
                source[f"fs.encoder.{i}.res.0.1.weight"], source[f"fs.encoder.{i}.res.0.1.bias"],
                training=False, eps=1e-3,
            )
            h = F.relu(main + res)
        else:
            h = F.relu(main)
        in_channels = channels

    gate = F.conv1d(h, source["output_layer.0.weight"], bias=source["output_layer.0.bias"])
    gate = F.batch_norm(
        gate, source["output_layer.1.running_mean"], source["output_layer.1.running_var"],
        source["output_layer.1.weight"], source["output_layer.1.bias"], training=False, eps=1e-5,
    )
    gate = torch.tanh(gate)
    z = torch.clamp(gate + 0.5, 0.0, 1.0)
    pooled = z.mean(dim=-1)
    return pooled @ source["freq_linear_proj.weight"].T + source["freq_linear_proj.bias"]


def test_ported_model_matches_an_independent_reference_forward():
    n_feat, channels, gate_channels, num_classes = 6, 8, 32, 12
    source = _build_random_reference_state_dict(n_feat, channels, gate_channels, num_classes)

    mapped, ported_channels, ported_gate_channels = map_state_dict(source)
    assert ported_channels == channels
    assert ported_gate_channels == gate_channels

    model = SparkNet(n_feat=n_feat, num_classes=num_classes, channels=channels,
                      gate_channels=gate_channels)
    model.load_state_dict(mapped, strict=True)
    model.eval()

    x = torch.randn(2, 1, n_feat, 9)
    with torch.no_grad():
        actual = model(x)
    expected = _reference_forward(x, source, n_feat, channels)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_costs_match_finding_2():
    # PLAN.md Finding 2, project MAC/parameter counts at 101 frames.
    logmel_40_c16 = SparkNet(n_feat=40, num_classes=12, channels=16, gate_channels=32)
    assert sum(p.numel() for p in logmel_40_c16.parameters()) == 4_852
    assert count_macs(logmel_40_c16, (40, 101)) == 418_120

    mfcc_32_c16 = SparkNet(n_feat=32, num_classes=12, channels=16, gate_channels=32)
    assert sum(p.numel() for p in mfcc_32_c16.parameters()) == 4_636
    assert count_macs(mfcc_32_c16, (32, 101)) == 396_304

    mfcc_32_c8 = SparkNet(n_feat=32, num_classes=12, channels=8, gate_channels=32)
    assert sum(p.numel() for p in mfcc_32_c8.parameters()) == 2_356
    assert count_macs(mfcc_32_c8, (32, 101)) == 177_336


def test_build_sparknet_reads_channels_from_model_cfg():
    model_cfg = {"name": "sparknet_c16", "channels": 16, "gate_channels": 32}
    model = build_sparknet(model_cfg, input_shape=(32, 101), num_classes=12)
    assert isinstance(model, SparkNet)
    assert sum(p.numel() for p in model.parameters()) == 4_636


def test_sparknet_checkpoint_loads_through_the_evaluate_dispatch(tmp_path):
    model_cfg = {"name": "sparknet_c8", "channels": 8, "gate_channels": 32}
    model = build_sparknet(model_cfg, input_shape=(32, 101), num_classes=12)
    checkpoint_path = tmp_path / "sparknet.pt"
    torch.save({
        "model_family": "sparknet",
        "model_cfg": model_cfg,
        "input_shape": [32, 101],
        "num_classes": 12,
        "num_keywords": 10,
        "label_map": {},
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)

    loaded_model, ckpt = load_model_from_checkpoint(str(checkpoint_path), torch.device("cpu"))
    assert isinstance(loaded_model, SparkNet)
    assert ckpt["model_family"] == "sparknet"

    x = torch.randn(1, 1, 32, 101)
    with torch.no_grad():
        assert torch.allclose(loaded_model(x), model.eval()(x))


def test_sparsity_term_is_the_reference_open_gate_probability():
    torch.manual_seed(0)
    model = SparkNet(n_feat=6, num_classes=12, channels=8, sparsity_weight=0.25)
    captured = {}
    model.gate_bn.register_forward_hook(
        lambda module, inputs, output: captured.update(gate_bn=output.detach())
    )
    model.train()
    model(torch.randn(3, 1, 6, 9))
    value, weight = model.auxiliary_losses()["gate_sparsity"]
    assert weight == 0.25

    mu = torch.tanh(captured["gate_bn"])  # the pre-noise gate
    # The reference writes it as mean(0.5 - 0.5 * erf((-0.5 - mu) / (sqrt(2) * 0.5))).
    expected = torch.mean(0.5 - 0.5 * torch.erf((-0.5 - mu) / (2 ** 0.5 * 0.5)))
    assert torch.allclose(value.detach(), expected, atol=1e-6)


def test_sparsity_term_is_popped_once_and_absent_in_eval():
    model = SparkNet(n_feat=6, num_classes=12, channels=8)
    x = torch.randn(2, 1, 6, 9)
    model.train()
    model(x)
    value, _ = model.auxiliary_losses()["gate_sparsity"]
    value.backward()
    assert model.gate_conv.weight.grad is not None
    assert model.auxiliary_losses() == {}

    model(x)
    model.eval()
    with torch.no_grad():
        model(x)
    assert model.auxiliary_losses() == {}


def test_build_sparknet_reads_sparsity_weight_and_records_input_shape():
    model = build_sparknet(
        {"channels": 8, "gate_channels": 32, "sparsity_weight": 0.0}, (40, 101), 12,
    )
    assert model.sparsity_weight == 0.0
    assert model.input_shape == (40, 101)
    default = build_sparknet({"channels": 8, "gate_channels": 32}, (40, 101), 12)
    assert default.sparsity_weight == 0.01


def test_phase_b_model_configs_have_the_planned_costs():
    import yaml
    from kws.models.registry import build_model

    expected = {8: (2_508, 192_688), 12: (3_584, 295_708), 16: (4_852, 418_120)}
    for channels, (params, macs) in expected.items():
        with open(f"configs/model/sparknet_c{channels}.yaml") as f:
            model_cfg = yaml.safe_load(f)
        model = build_model(model_cfg, (40, 101), 12)
        assert isinstance(model, SparkNet)
        assert sum(p.numel() for p in model.parameters()) == params
        assert count_macs(model, (40, 101)) == macs
