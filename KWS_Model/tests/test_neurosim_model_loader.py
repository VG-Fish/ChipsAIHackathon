import torch

from kws.hardware.neurosim.model_loader import load_inference_model
from kws.hardware.neurosim.model_loader import load_clean_inference_model
from kws.models.registry import build_model


def test_loader_reuses_project_checkpoint_reconstruction(tmp_path):
    cfg = {
        "name": "tiny_ds_cnn",
        "initial_channels": 4,
        "initial_kernel": 3,
        "initial_stride": 2,
        "block_channels": [4],
    }
    model = build_model(cfg, (8, 12), 3)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_cfg": cfg,
            "input_shape": [8, 12],
            "num_classes": 3,
            "model_state_dict": model.state_dict(),
            "run_id": "run-123",
        },
        checkpoint,
    )

    loaded = load_inference_model(checkpoint)

    assert loaded.model.training is False
    assert loaded.run_id == "run-123"
    assert loaded.checkpoint_sha256
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())


def test_clean_pai_loader_requires_explicit_rebuild_dimensions(tmp_path):
    checkpoint = tmp_path / "final_clean_pai.safetensors"
    checkpoint.write_bytes(b"not loaded")
    config = tmp_path / "model.yaml"
    config.write_text("name: tiny\n", encoding="utf-8")

    try:
        load_clean_inference_model(checkpoint, model_config_path=config)
    except ValueError as exc:
        assert "explicit input_shape and num_classes" in str(exc)
    else:
        raise AssertionError("ambiguous clean artifact was accepted")
