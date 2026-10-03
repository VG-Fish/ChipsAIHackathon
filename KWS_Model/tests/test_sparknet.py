import torch
import torch.nn.functional as F

from kws.evaluate import load_model_from_checkpoint
from kws.models.sparknet import DendriticPointwise, SparkNet, build_sparknet
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


def test_dendritic_pointwise_reads_only_its_local_window():
    layer = DendriticPointwise(channels=6, dendrites=2, fan_in=2)
    assert sum(p.numel() for p in layer.parameters()) == 6 * 2 * (2 + 2)
    x = torch.randn(3, 6, 1, 5, requires_grad=True)
    layer(x)[:, 0].sum().backward()
    # Neuron 0's dendrites read channels 0-1 and 2-3, never 4-5.
    touched = x.grad.abs().sum(dim=(0, 2, 3)) > 0
    assert not touched[4:].any()


def test_build_sparknet_swaps_only_the_c_to_c_pointwise_convs_for_dendrites():
    cfg = {"channels": 18, "gate_channels": 16, "dendrites": 2, "dendrite_fan_in": 4}
    model = build_sparknet(cfg, (32, 101), 12)
    kinds = [type(block.pointwise).__name__ for block in model.blocks]
    assert kinds == ["Conv2d", "DendriticPointwise", "DendriticPointwise", "DendriticPointwise"]
    assert sum(p.numel() for p in model.parameters()) == 4474
    assert model(torch.randn(2, 1, 32, 101)).shape == (2, 12)


def test_additive_dendrites_keep_the_pointwise_conv_and_add_a_branch():
    cfg = {"channels": 16, "gate_channels": 16, "dendrites": 1, "dendrite_fan_in": 16,
           "dendrite_mode": "add", "dendrite_blocks": [1]}
    model = build_sparknet(cfg, (32, 101), 12)
    assert all(isinstance(block.pointwise, torch.nn.Conv2d) for block in model.blocks)
    assert [block.dendrite_branch is not None for block in model.blocks] == [False, True, False, False]
    # A full-fan-in branch costs what a PAI blocks.1 pointwise dendrite does: C * (C + 2).
    assert sum(p.numel() for p in model.parameters()) == 4140 + 16 * 18
    assert model(torch.randn(2, 1, 32, 101)).shape == (2, 12)


def test_block0_dendrites_tile_local_windows_of_the_feature_bins():
    layer = DendriticPointwise(channels=4, dendrites=1, fan_in=8, in_channels=32)
    windows = layer.index.view(4, 8)
    assert windows[:, 0].tolist() == [0, 8, 16, 24]
    cfg = {"channels": 16, "gate_channels": 16, "dendrites": 2, "dendrite_fan_in": 8,
           "dendrite_mode": "add", "dendrite_blocks": [0]}
    model = build_sparknet(cfg, (32, 101), 12)
    assert sum(p.numel() for p in model.parameters()) == 4140 + 16 * 2 * (8 + 2)
    assert model(torch.randn(2, 1, 32, 101)).shape == (2, 12)


# --- Multi-scale dendritic (MSD) blocks -------------------------------------

MSD_CONFIGS = ("a_relu", "a_lin", "a_dense", "a_1scale", "a13_relu", "a13_lin", "a13_dense",
               "b_relu", "b_lin", "b_dense")


def _msd_block(**kwargs):
    from kws.models.sparknet import MSDBlock
    defaults = dict(in_channels=6, out_channels=4, kernel_size=5, residual=True,
                    dilations=(1, 2, 4), pointwise="dendritic", branches=6, fan_in=2)
    defaults.update(kwargs)
    return MSDBlock(**defaults)


def test_msd_block_shapes_and_param_counts():
    for in_channels, residual in ((32, False), (8, True)):
        C, S, K, f, k = 8, 3, 3, 7, 11
        block = _msd_block(in_channels=in_channels, out_channels=C, kernel_size=k,
                           residual=residual, branches=K, fan_in=f)
        x = torch.randn(2, in_channels, 1, 101)
        assert block(x).shape == (2, C, 1, 101)
        assert [conv.dilation for conv in block.depthwise] == [(1, 1), (1, 2), (1, 4)]
        expected = S * in_channels * k + C * K * (f + 2) + 2 * C
        if residual:
            expected += in_channels * C + 2 * C
        assert sum(p.numel() for p in block.parameters()) == expected

        dense = _msd_block(in_channels=in_channels, out_channels=C, kernel_size=k,
                           residual=residual, pointwise="dense")
        assert isinstance(dense.pointwise, torch.nn.Conv2d)
        assert dense(x).shape == (2, C, 1, 101)
        expected = S * in_channels * k + S * in_channels * C + 2 * C
        if residual:
            expected += in_channels * C + 2 * C
        assert sum(p.numel() for p in dense.parameters()) == expected


def test_msd_branch_j_reads_scale_j_mod_s():
    torch.manual_seed(0)
    block = _msd_block().eval()  # 4 neurons x 6 branches, 3 scales
    captured = {}
    block.pointwise.dendrite.register_forward_hook(
        lambda module, inputs, output: captured.update(h=output.detach().clone()))
    x = torch.randn(2, 6, 1, 13)
    with torch.no_grad():
        block(x)
        base = captured["h"]
        branch_scale = torch.arange(4 * 6) % 6 % 3  # unit n * K + j reads scale j % 3
        for s in range(3):
            saved = block.depthwise[s].weight.data.clone()
            block.depthwise[s].weight.data.zero_()
            block(x)
            block.depthwise[s].weight.data.copy_(saved)
            changed = (captured["h"] - base).abs().flatten(2).amax(dim=(0, 2)) > 0
            assert changed.tolist() == (branch_scale == s).tolist()


def test_msd_neuron_reads_only_its_local_windows():
    block = _msd_block(in_channels=12, out_channels=4, branches=3, fan_in=2, residual=False)
    x = torch.randn(3, 12, 1, 9, requires_grad=True)
    block.pointwise(torch.cat([conv(x) for conv in block.depthwise], dim=1))[:, 1].sum().backward()
    # Neuron 1 starts at channel 1 * 12 // 4 = 3 and has one window position (K / S = 1).
    touched = (x.grad.abs().sum(dim=(0, 2, 3)) > 0).nonzero().flatten().tolist()
    assert touched == [3, 4]
    index = block.pointwise.index.view(4, 3, 2)
    assert (index // 12).tolist()[1] == [[0, 0], [1, 1], [2, 2]]


def test_msd_identity_branches_equal_an_explicit_linear_map():
    torch.manual_seed(0)
    block = _msd_block(activation="identity")
    pw = block.pointwise
    C, K, f, width = 4, 6, 2, 6 * 3
    w_dendrite = pw.dendrite.weight.detach().view(C, K, f)
    w_soma = pw.soma.weight.detach().view(C, K)
    index = pw.index.view(C, K, f)
    weight = torch.zeros(C, width)
    for n in range(C):
        for j in range(K):
            for i in range(f):
                weight[n, index[n, j, i]] += w_soma[n, j] * w_dendrite[n, j, i]
    bias = (w_soma * pw.dendrite.bias.detach().view(C, K)).sum(dim=1)
    d = torch.randn(2, width, 1, 7)
    expected = F.conv2d(d, weight[:, :, None, None], bias)
    assert torch.allclose(pw(d), expected, atol=1e-5)
    block_relu = _msd_block(activation="relu")
    block_relu.load_state_dict(block.state_dict())
    assert not torch.allclose(block_relu.pointwise(d), expected, atol=1e-3)


def test_msd_options_validate():
    import pytest
    with pytest.raises(ValueError, match="multiple"):
        _msd_block(branches=4)
    with pytest.raises(ValueError, match="need block_type: msd"):
        build_sparknet({"channels": 8, "gate_channels": 16, "msd_branches": 3}, (32, 101), 12)
    with pytest.raises(ValueError, match="cannot be both"):
        build_sparknet({"channels": 8, "gate_channels": 16, "block_type": "msd",
                        "msd_branches": 3, "msd_fan_in": 2, "dendrites": 2,
                        "dendrite_fan_in": 4}, (32, 101), 12)


def test_default_configs_still_build_plain_tcs_blocks():
    from kws.models.sparknet import TCSBlock
    model = build_sparknet({"channels": 16, "gate_channels": 16}, (32, 101), 12)
    assert all(type(block) is TCSBlock for block in model.blocks)
    assert sum(p.numel() for p in model.parameters()) == 4140


def test_msd_blocks_selects_which_blocks_are_multi_scale():
    cfg = {"channels": 5, "gate_channels": 8, "block_type": "msd", "msd_branches": 3,
           "msd_fan_in": 3, "msd_blocks": [1, 3]}
    model = build_sparknet(cfg, (32, 101), 12)
    assert [type(block).__name__ for block in model.blocks] == [
        "TCSBlock", "MSDBlock", "TCSBlock", "MSDBlock"]
    assert model(torch.randn(2, 1, 32, 101)).shape == (2, 12)


def test_msd_qat_fusion_pairs_only_convs_that_feed_their_bn():
    import copy
    from kws.optimize.quantize_qat import _fuse_conv_bn_for_qat

    x = torch.randn(2, 1, 32, 101)
    for pointwise, expected in (("dense", {"pointwise+bn", "res_conv+res_bn"}),
                                ("dendritic", {"res_conv+res_bn"})):
        cfg = {"channels": 6, "gate_channels": 8, "block_type": "msd",
               "msd_pointwise": pointwise, "msd_branches": 3, "msd_fan_in": 3}
        model = build_sparknet(cfg, (32, 101), 12)
        model.train()
        model(x)  # populate BatchNorm running stats
        model.eval()
        with torch.no_grad():
            before = model(x)
        fused = _fuse_conv_bn_for_qat(copy.deepcopy(model))
        msd_pairs = {pair.split(":")[1] for pair in fused if pair.startswith("MSDBlock:")}
        assert msd_pairs == expected
        fused_model = copy.deepcopy(model)
        _fuse_conv_bn_for_qat(fused_model)
        fused_model.eval()
        with torch.no_grad():
            assert torch.allclose(fused_model(x), before, atol=1e-5)


def test_msd_model_configs_record_their_exact_params():
    import re

    import yaml
    from kws.models.registry import build_model

    for name in MSD_CONFIGS:
        path = f"configs/model/sparknet_msd_{name}.yaml"
        with open(path) as f:
            first_line = f.readline()
            f.seek(0)
            model_cfg = yaml.safe_load(f)
        recorded = int(re.match(r"# ([\d,]+) params", first_line).group(1).replace(",", ""))
        model = build_model(model_cfg, (32, 101), 12)
        assert isinstance(model, SparkNet)
        assert sum(p.numel() for p in model.parameters()) == recorded, path
        budget = 4140 if name.startswith("a") else 2023
        assert abs(recorded - budget) / budget <= 0.015, path
        assert count_macs(model, (32, 101)) > 0
        assert model(torch.randn(2, 1, 32, 101)).shape == (2, 12)


def test_msd_checkpoint_loads_through_the_evaluate_dispatch(tmp_path):
    import yaml
    with open("configs/model/sparknet_msd_a_relu.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model = build_sparknet(model_cfg, input_shape=(32, 101), num_classes=12)
    checkpoint_path = tmp_path / "msd.pt"
    torch.save({
        "model_family": "sparknet", "model_cfg": model_cfg, "input_shape": [32, 101],
        "num_classes": 12, "num_keywords": 10, "label_map": {},
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    loaded_model, _ = load_model_from_checkpoint(str(checkpoint_path), torch.device("cpu"))
    x = torch.randn(1, 1, 32, 101)
    with torch.no_grad():
        assert torch.allclose(loaded_model(x), model.eval()(x))
