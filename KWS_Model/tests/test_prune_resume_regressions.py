from contextlib import nullcontext
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

import kws.train as train_module
from kws.models.ds_cnn import DSCNN
from kws.optimize import prune as prune_module
from kws.optimize.prune import (
    SparsitySpec,
    _load_prune_resume_state,
    _resolve_prune_resume_from,
    _validate_prune_resume_state,
)
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import CHECKPOINT_FORMAT_VERSION, recipe_fingerprint


def _training_state(*, stage="sparsity", phase="prune_kd", recipe=None):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "kws_training_state",
        "stage": stage,
        "phase": phase,
        "recipe_fingerprint": recipe_fingerprint(recipe or {"recipe": "expected"}),
    }


def test_resume_resolves_candidate_phase_latest_under_output_root(tmp_path):
    spec = SparsitySpec("structured", keep_ratio=0.5)
    layout = ArtifactLayout(tmp_path)
    latest = layout.checkpoint_path(
        "sparsity", "prune_kd", "latest", candidate=spec.label
    )
    latest.parent.mkdir(parents=True)
    latest.touch()

    resolved = _resolve_prune_resume_from(
        resume=True,
        resume_from=None,
        output_dir=tmp_path,
        out_checkpoint=tmp_path / "ignored-best.pt",
        spec=spec,
    )

    assert resolved == latest


def test_explicit_resume_from_takes_precedence_over_standard_latest(tmp_path):
    spec = SparsitySpec("structured", keep_ratio=0.5)
    layout = ArtifactLayout(tmp_path / "run")
    latest = layout.checkpoint_path(
        "sparsity", "prune_kd", "latest", candidate=spec.label
    )
    latest.parent.mkdir(parents=True)
    latest.touch()
    explicit = tmp_path / "explicit-training-state.pt"

    resolved = _resolve_prune_resume_from(
        resume=True,
        resume_from=explicit,
        output_dir=layout.root,
        out_checkpoint=tmp_path / "ignored-best.pt",
        spec=spec,
    )

    assert resolved == explicit


def test_resume_refuses_epoch_one_restart_when_metrics_exist_without_latest(tmp_path):
    spec = SparsitySpec("nm", n=2, m=4)
    layout = ArtifactLayout(tmp_path)
    metrics = layout.metrics_path("sparsity", "prune_kd", candidate=spec.label)
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"epoch":1}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="refusing to restart at epoch 1"):
        _resolve_prune_resume_from(
            resume=True,
            resume_from=None,
            output_dir=tmp_path,
            out_checkpoint=tmp_path / "ignored-best.pt",
            spec=spec,
        )


def test_prune_resume_rejects_legacy_and_wrong_phase_states(tmp_path):
    legacy = tmp_path / "legacy-best.pt"
    torch.save({"model_state_dict": {}}, legacy)
    with pytest.raises(ValueError, match="not a resumable training-state"):
        _load_prune_resume_state(legacy, torch.device("cpu"))

    wrong_phase = tmp_path / "qat-latest.pt"
    torch.save(_training_state(stage="quantize", phase="qat"), wrong_phase)
    with pytest.raises(ValueError, match="quantize/qat, not sparsity/prune_kd"):
        _load_prune_resume_state(wrong_phase, torch.device("cpu"))


def test_prune_resume_rejects_recipe_mismatch():
    state = _training_state(recipe={"spec": "old"})

    with pytest.raises(ValueError, match="resume recipe mismatch"):
        _validate_prune_resume_state(
            state,
            source="latest.pt",
            recipe={"spec": "new"},
        )


def test_programmatic_resume_does_not_duplicate_committed_epoch_metrics(
    tmp_path, monkeypatch
):
    model_cfg = {
        "name": "tiny",
        "initial_channels": 4,
        "initial_kernel": 3,
        "initial_stride": 1,
        "block_channels": [4],
        "dropout": 0.0,
    }
    model = DSCNN(
        input_shape=(8, 8),
        num_classes=2,
        initial_channels=4,
        initial_kernel=3,
        initial_stride=1,
        block_channels=[4],
        dropout=0.0,
    )
    source = tmp_path / "source.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "input_shape": (8, 8),
            "num_classes": 2,
            "num_keywords": 1,
        },
        source,
    )
    features = torch.randn(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(features, labels)
    datasets = {"training": dataset, "validation": dataset}
    monkeypatch.setattr(
        prune_module,
        "build_datasets",
        lambda *args, **kwargs: (datasets, {"yes": 0, "_unknown_": 1}),
    )
    monkeypatch.setattr(prune_module, "get_device", lambda: torch.device("cpu"))

    data_cfg = {"target_keywords": ["yes"]}
    train_cfg = {
        "epochs": 2,
        "batch_size": 2,
        "label_smoothing": 0.0,
        "lr": 0.01,
        "weight_decay": 0.0,
        "warmup_fraction": 0.0,
        "augment": False,
        "seed": 7,
        "num_workers": 0,
    }
    output_dir = tmp_path / "run"
    spec = SparsitySpec("dense")
    destination = tmp_path / "unused-best.pt"

    run_finetune = train_module.run_finetune

    def interrupt_after_first_epoch(*args, **kwargs):
        def on_epoch_end(record):
            if record["epoch"] == 1:
                raise RuntimeError("simulated pruning interruption")

        return run_finetune(*args, **kwargs, on_epoch_end=on_epoch_end)

    monkeypatch.setattr(train_module, "run_finetune", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated pruning interruption"):
        prune_module.prune_and_fine_tune(
            str(source),
            data_cfg,
            train_cfg,
            spec,
            destination,
            output_dir=output_dir,
        )
    monkeypatch.setattr(train_module, "run_finetune", run_finetune)

    resumed = prune_module.prune_and_fine_tune(
        str(source),
        data_cfg,
        train_cfg,
        spec,
        destination,
        output_dir=output_dir,
        resume=True,
    )

    metrics = ArtifactLayout(output_dir).metrics_path(
        "sparsity", "prune_kd", candidate=spec.label
    )
    records = metrics.read_text(encoding="utf-8").splitlines()
    assert [record["epoch"] for record in resumed["history"]] == [1, 2]
    assert len(records) == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_resolves_resume_and_preserves_explicit_precedence(
    tmp_path, monkeypatch, explicit
):
    data_config = tmp_path / "data.yaml"
    train_config = tmp_path / "train.yaml"
    source = tmp_path / "source.pt"
    data_config.write_text("target_keywords: [yes]\n", encoding="utf-8")
    train_config.write_text("epochs: 2\n", encoding="utf-8")

    output_dir = tmp_path / "run"
    spec = SparsitySpec("structured", keep_ratio=0.5)
    standard_latest = ArtifactLayout(output_dir).checkpoint_path(
        "sparsity", "prune_kd", "latest", candidate=spec.label
    )
    standard_latest.parent.mkdir(parents=True)
    standard_latest.touch()
    explicit_path = tmp_path / "explicit.pt"

    captured = {}

    def fake_prune_and_fine_tune(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(prune_module, "prune_and_fine_tune", fake_prune_and_fine_tune)
    monkeypatch.setattr(prune_module, "run_session", lambda *args, **kwargs: nullcontext())
    argv = [
        "kws.optimize.prune",
        "--data-config",
        str(data_config),
        "--train-config",
        str(train_config),
        "--checkpoint",
        str(source),
        "--output-dir",
        str(output_dir),
        "--resume",
    ]
    if explicit:
        argv.extend(["--resume-from", str(explicit_path)])
    monkeypatch.setattr(sys, "argv", argv)

    prune_module.main()

    assert captured["resume"] is True
    assert captured["resume_from"] == (
        explicit_path if explicit else standard_latest
    )
