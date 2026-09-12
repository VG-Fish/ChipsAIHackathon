import pytest
import torch
import yaml

from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import build_training_state, require_manifest_run_id
from kws.utils.logging import run_session


def test_direct_layout_creates_and_reuses_manifest_run_id(tmp_path):
    root = tmp_path / "direct-run"

    first = ArtifactLayout(root)
    run_id = first.manifest_run_id
    assert run_id
    assert yaml.safe_load((root / "manifest.yaml").read_text())["run_id"] == run_id

    second = ArtifactLayout(root)
    assert second.manifest_run_id == run_id


@pytest.mark.parametrize("run_id", [None, "", "   ", 123])
def test_existing_manifest_requires_valid_run_id(tmp_path, run_id):
    root = tmp_path / "invalid-run"
    root.mkdir()
    (root / "manifest.yaml").write_text(
        yaml.safe_dump({"format_version": 1, "run_id": run_id}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_id"):
        ArtifactLayout(root)


def test_run_session_retains_layout_manifest_run_id(tmp_path):
    layout = ArtifactLayout(tmp_path / "session-run")
    expected = layout.manifest_run_id

    with run_session(layout=layout, command="test.run-id") as manifest:
        assert manifest["run_id"] == expected

    assert ArtifactLayout(layout.root).manifest_run_id == expected


def test_checkpoint_state_can_bind_to_manifest_run_id():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manifest = {"run_id": "manifest-run"}

    state = build_training_state(
        manifest=manifest,
        stage="train",
        phase="test",
        completed_epoch=0,
        target_epochs=1,
        global_step=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        best_metric_name="accuracy",
        best_metric_value=0.0,
        best_epoch=0,
        best_model_state_dict=model.state_dict(),
        history=[],
        metrics=None,
        recipe={},
    )
    assert state["run_id"] == "manifest-run"

    with pytest.raises(ValueError, match="does not match manifest"):
        build_training_state(
            run_id="other-run",
            manifest=manifest,
            stage="train",
            phase="test",
            completed_epoch=0,
            target_epochs=1,
            global_step=0,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            best_metric_name="accuracy",
            best_metric_value=0.0,
            best_epoch=0,
            best_model_state_dict=model.state_dict(),
            history=[],
            metrics=None,
            recipe={},
        )


def test_manifest_run_id_helper_requires_a_manifest_but_no_output_dir_is_compatible():
    assert require_manifest_run_id({"run_id": "run"}) == "run"
    with pytest.raises(ValueError, match="run_id"):
        require_manifest_run_id({})
    # build_training_state leaves run_id handling unchanged when no output
    # manifest is configured; this is the legacy in-memory path.
