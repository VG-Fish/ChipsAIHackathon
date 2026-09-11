import pytest
import torch
import torch.nn as nn

from kws.models.ds_cnn import DSCNN
from kws.optimize.cluster import (
    ClusterSpec,
    apply_weight_clustering,
    bake_codebooks,
    cluster_weights,
    codebook_parameters,
    load_clustered_model,
    pack_indices,
    save_codebook_artifact,
    save_clustered_model,
    unpack_indices,
)


def _build_model():
    return DSCNN(input_shape=(40, 98), num_classes=6, initial_channels=18, initial_kernel=5,
                 initial_stride=2, block_channels=[40, 40], dropout=0.2)


def test_cluster_weights_returns_sorted_centroids_and_valid_indices():
    weight = torch.randn(8, 16)
    centroids, indices = cluster_weights(weight, clusters=8)

    assert centroids.numel() == 8
    assert torch.all(centroids[1:] >= centroids[:-1])  # sorted, for a stable codebook
    assert indices.shape == weight.shape
    assert int(indices.min()) >= 0 and int(indices.max()) < 8


def test_cluster_weights_handles_fewer_unique_values_than_clusters():
    weight = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    centroids, indices = cluster_weights(weight, clusters=16)

    assert centroids.numel() <= 2
    assert torch.allclose(centroids[indices], weight)


def test_clustering_limits_each_layer_to_the_codebook_size():
    model = _build_model()
    apply_weight_clustering(model, ClusterSpec(bits=3, min_weights=64))

    for name in ("stem.0", "blocks.0.pointwise", "fc"):
        weight = model.get_submodule(name).weight
        assert torch.unique(weight).numel() <= 8


def test_small_layers_stay_dense_because_the_table_would_cost_more():
    model = _build_model()
    report = apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=5000))

    assert report.clustered_layers == {}
    assert set(report.skipped_layers) >= {"fc", "blocks.0.depthwise"}
    # No sharing means no saving: the reported footprint stays dense.
    assert report.codebook_bytes == report.dense_bytes


def test_codebook_learning_moves_every_weight_that_shares_a_centroid():
    model = _build_model()
    apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))
    centroids = codebook_parameters(model)
    assert centroids

    model(torch.randn(2, 1, 40, 98)).sum().backward()
    assert all(centroid.grad is not None for centroid in centroids)

    before = model.blocks[0].pointwise.weight.detach().clone()
    with torch.no_grad():
        for centroid in centroids:
            centroid += 0.1
    after = model.blocks[0].pointwise.weight.detach()
    # One scalar per cluster moved, and every weight assigned to it followed.
    assert torch.allclose(after - before, torch.full_like(before, 0.1))


def test_baking_preserves_the_forward_pass_and_the_sharing_pattern():
    model = _build_model().eval()
    apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))
    sample = torch.randn(2, 1, 40, 98)
    with torch.no_grad():
        before = model(sample)

    projector = bake_codebooks(model)
    with torch.no_grad():
        after = model(sample)

    assert torch.allclose(before, after, atol=1e-6)
    assert len(projector) > 0
    assert torch.unique(model.blocks[0].pointwise.weight).numel() <= 16


def test_projector_reimposes_sharing_after_an_optimizer_step():
    model = _build_model()
    apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))
    projector = bake_codebooks(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(2, 1, 40, 98)).sum().backward()
    optimizer.step()
    assert torch.unique(model.blocks[0].pointwise.weight).numel() > 16

    projector(model)
    assert torch.unique(model.blocks[0].pointwise.weight).numel() <= 16


def test_cluster_spec_rejects_impossible_bit_widths():
    with pytest.raises(ValueError):
        ClusterSpec(bits=0)
    with pytest.raises(ValueError):
        ClusterSpec(bits=32)
    assert ClusterSpec(bits=4).clusters == 16


def test_reported_footprint_counts_indices_plus_the_centroid_table():
    model = nn.Sequential(nn.Linear(64, 64, bias=False))
    report = apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))

    # 4096 weights at 4 bits each, plus 16 FP32 centroids.
    assert report.codebook_bytes == 4096 // 2 + 16 * 4
    assert report.dense_bytes == 4096 * 4


def test_clustered_graph_round_trips_for_a_later_pipeline_stage(tmp_path):
    model = _build_model().eval()
    apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))
    projector = bake_codebooks(model)
    sample = torch.randn(1, 1, 40, 98)
    with torch.no_grad():
        expected = model(sample)

    path = tmp_path / "clustered.pt"
    save_clustered_model(
        model,
        projector,
        path,
        input_shape=(40, 98),
        candidate_id="candidate:abc",
        recipe_id="recipe:def",
    )
    restored = _build_model().eval()
    loaded_projector = load_clustered_model(
        restored,
        path,
        expected_candidate="candidate:abc",
        expected_recipe="recipe:def",
    )

    with torch.no_grad():
        actual = restored(sample)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert loaded_projector.describe() == projector.describe()

    with pytest.raises(ValueError, match="recipe"):
        load_clustered_model(restored, path, expected_recipe="recipe:changed")


def test_packed_codebook_indices_round_trip():
    indices = torch.tensor([0, 1, 15, 2, 7, 3, 9], dtype=torch.long)

    packed = pack_indices(indices, 4)
    unpacked = unpack_indices(packed, indices.numel(), 4)

    assert torch.equal(unpacked, indices)


def test_quantization_target_payload_retains_codebooks(tmp_path):
    model = nn.Sequential(nn.Linear(64, 64, bias=False)).eval()
    apply_weight_clustering(model, ClusterSpec(bits=4, min_weights=64))
    projector = bake_codebooks(model, bits=4)
    path = tmp_path / "deployed.codebook.pt"

    info = save_codebook_artifact(
        model,
        projector,
        path,
        graph_path=tmp_path / "deployed.pt",
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert info["format"] == "kws-codebook-v1"
    assert payload["bits"] == 4
    assert payload["layers"]["0"]["index_bits"] == 4
    assert info["storage_bytes"] < model[0].weight.numel() * 4
