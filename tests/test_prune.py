import pytest
import torch

from kws.models.ds_cnn import DSCNN
from kws.optimize.prune import (
    SparsitySpec,
    apply_nm_sparsity,
    apply_sparsity,
    nm_mask,
    prune_ds_cnn,
)


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


def test_nm_mask_keeps_n_largest_of_every_m():
    weight = torch.tensor([[4.0, -3.0, 2.0, -1.0, 0.5, -9.0, 0.1, 0.2]])
    mask = nm_mask(weight, n=2, m=4)

    assert mask.shape == weight.shape
    # Two survivors per block of four, and they are the largest by magnitude.
    assert mask[0, :4].sum() == 2 and mask[0, 4:].sum() == 2
    assert mask[0, 0] == 1 and mask[0, 1] == 1  # |4|, |-3| beat |2|, |-1|
    assert mask[0, 5] == 1  # |-9| is the largest in its block


def test_nm_mask_rejects_indivisible_and_invalid_patterns():
    with pytest.raises(ValueError):
        nm_mask(torch.zeros(2, 6), n=2, m=4)  # 6 is not a multiple of 4
    with pytest.raises(ValueError):
        nm_mask(torch.zeros(2, 8), n=4, m=4)  # n must be strictly less than m


def test_apply_nm_sparsity_skips_layers_that_cannot_hold_the_pattern():
    model = _build_model()
    masks = apply_nm_sparsity(model, n=2, m=4)

    # A depthwise conv reduces over 1*3*3 = 9 weights, which no 2:4 pattern
    # divides, so it must be left dense rather than forced into one.
    assert not any("depthwise" in name for name in masks.masks)
    assert masks.sparsity(model) == pytest.approx(0.5)


def test_masks_survive_an_optimizer_step_when_reapplied():
    model = _build_model()
    masks = apply_nm_sparsity(model, n=2, m=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    model(torch.randn(2, 1, 40, 98)).sum().backward()
    optimizer.step()
    assert masks.sparsity(model) < 0.5  # the step refilled the pruned weights

    masks.apply(model)
    assert masks.sparsity(model) == pytest.approx(0.5)


def test_sparsity_spec_validates_and_labels_itself():
    assert SparsitySpec(kind="structured", keep_ratio=0.45).label == "structured_keep0.45"
    assert SparsitySpec(kind="nm", n=2, m=4).label == "nm2of4"
    with pytest.raises(ValueError):
        SparsitySpec(kind="nm")
    with pytest.raises(ValueError):
        SparsitySpec(kind="nm", n=4, m=4)
    with pytest.raises(ValueError):
        SparsitySpec(kind="structured", keep_ratio=1.5)
    with pytest.raises(ValueError):
        SparsitySpec(kind="magnitude")


def test_apply_sparsity_dispatches_on_kind():
    structured, masks = apply_sparsity(_build_model(), SparsitySpec("structured", keep_ratio=0.5))
    assert masks is None
    assert structured.blocks[0].pointwise.out_channels == 32

    same, nm_masks = apply_sparsity(_build_model(), SparsitySpec("nm", n=1, m=4))
    assert nm_masks is not None
    assert same.blocks[0].pointwise.out_channels == 64  # shapes are untouched
