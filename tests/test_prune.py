import torch

from kws.models.ds_cnn import DSCNN
from kws.optimize.prune import prune_ds_cnn


def _build_model():
    return DSCNN(input_shape=(40, 98), num_classes=6, initial_channels=48, initial_kernel=10,
                 initial_stride=2, block_channels=[64, 64, 64, 64], dropout=0.2)


def test_prune_reduces_params_and_preserves_forward_shape():
    model = _build_model()
    model.eval()
    pruned = prune_ds_cnn(model, keep_ratio=0.5)
    pruned.eval()

    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned.parameters())
    assert pruned_params < orig_params

    x = torch.randn(2, 1, 40, 98)
    with torch.no_grad():
        out_orig = model(x)
        out_pruned = pruned(x)

    assert out_pruned.shape == out_orig.shape
    assert torch.isfinite(out_pruned).all()


def test_prune_channel_counts_match_keep_ratio():
    model = _build_model()
    pruned = prune_ds_cnn(model, keep_ratio=0.25)
    for block in pruned.blocks:
        assert block.pointwise.out_channels == 16  # round(64 * 0.25)
