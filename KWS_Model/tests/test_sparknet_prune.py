import pytest
import torch

from kws.models.sparknet import SparkNet, TCSBlock
from kws.optimize.prune import prune_sparknet


def _blocks(model: SparkNet) -> list[TCSBlock]:
    blocks = list(model.blocks)
    assert all(isinstance(block, TCSBlock) for block in blocks)
    return [block for block in blocks if isinstance(block, TCSBlock)]


@pytest.mark.parametrize("target", [10, 8, 6])
def test_prune_sparknet_threads_channels_and_preserves_heads(target):
    model = SparkNet(8, 4, channels=12, gate_channels=7, input_shape=(8, 20))
    original_fc = model.fc.weight.detach().clone()
    pruned = prune_sparknet(model, target)
    pruned_blocks = _blocks(pruned)

    assert [b.pointwise.out_channels for b in pruned_blocks] == [target] * 4
    assert all(b.depthwise.in_channels == target for b in pruned_blocks[1:])
    assert pruned.gate_conv.in_channels == target
    assert pruned.gate_conv.out_channels == 7
    assert torch.equal(pruned.fc.weight, original_fc)
    assert pruned(torch.randn(2, 1, 8, 20)).shape == (2, 4)
    assert pruned.gate_conv.bias is not None
    assert model.gate_conv.bias is not None
    assert torch.equal(pruned.gate_conv.bias, model.gate_conv.bias)


def test_prune_sparknet_rejects_unsupported_width():
    with pytest.raises(ValueError, match="narrower than the source width"):
        prune_sparknet(SparkNet(8, 4, channels=12), 12)


def test_prune_sparknet_preserves_batchnorm_counters_and_mode():
    model = SparkNet(8, 4, channels=12).eval()
    for block in _blocks(model):
        assert block.bn.num_batches_tracked is not None
        block.bn.num_batches_tracked.fill_(9)

    pruned = prune_sparknet(model, 8)

    assert not pruned.training
    for block in _blocks(pruned):
        assert block.bn.num_batches_tracked is not None
        assert block.bn.num_batches_tracked.item() == 9


def test_prune_sparknet_propagates_exact_l1_indices_through_residuals_and_gate():
    model = SparkNet(4, 3, channels=6, gate_channels=5, input_shape=(4, 20))
    model_blocks = _blocks(model)
    with torch.no_grad():
        for block in model_blocks:
            assert block.bn.running_mean is not None
            for output_channel in range(6):
                block.pointwise.weight[output_channel].fill_(output_channel + 1)
            block.bn.running_mean.copy_(torch.arange(6, dtype=torch.float32))
        model_blocks[1].depthwise.weight.copy_(
            torch.arange(model_blocks[1].depthwise.weight.numel()).reshape_as(
                model_blocks[1].depthwise.weight
            )
        )
        assert model_blocks[1].res_conv is not None
        model_blocks[1].res_conv.weight.copy_(
            torch.arange(model_blocks[1].res_conv.weight.numel()).reshape_as(
                model_blocks[1].res_conv.weight
            )
        )
        model.gate_conv.weight.copy_(
            torch.arange(model.gate_conv.weight.numel()).reshape_as(
                model.gate_conv.weight
            )
        )

    keep = torch.tensor([3, 4, 5])
    pruned = prune_sparknet(model, 3)
    pruned_blocks = _blocks(pruned)

    assert torch.equal(
        pruned_blocks[1].depthwise.weight,
        model_blocks[1].depthwise.weight[keep],
    )
    assert torch.equal(
        pruned_blocks[1].pointwise.weight,
        model_blocks[1].pointwise.weight[keep][:, keep],
    )
    assert pruned_blocks[1].res_conv is not None
    assert torch.equal(
        pruned_blocks[1].res_conv.weight,
        model_blocks[1].res_conv.weight[keep][:, keep],
    )
    assert pruned_blocks[1].bn.running_mean is not None
    assert torch.equal(pruned_blocks[1].bn.running_mean, keep.float())
    assert torch.equal(
        pruned.gate_conv.weight,
        model.gate_conv.weight[:, keep],
    )
