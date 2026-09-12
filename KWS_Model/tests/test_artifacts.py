import json
import os
from pathlib import Path

import pytest
import yaml

from kws.utils.artifacts import ArtifactLayout, sha256_path
from kws.train import resolve_manifest_run_id
from kws.utils.checkpointing import MetricsRecorder, record_input
from kws.utils.logging import run_session


def test_layout_creates_documented_tree_and_handles_spaces(tmp_path):
    root = tmp_path / "run with spaces"
    layout = ArtifactLayout(root).ensure_tree()

    assert (root / "manifest.yaml").parent.exists()
    assert layout.metrics_path("teacher", "train").parent == root / "metrics" / "teacher"
    assert layout.checkpoint_path("teacher", "train", "latest") == root / "models" / "checkpoints" / "teacher" / "latest.pt"
    assert layout.pai_candidate_path("w18").parent == root / "pai" / "candidates"


def test_layout_rejects_escape_and_symlink_paths(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    with pytest.raises(ValueError):
        layout.output_path("../../outside.pt", category="models", default="x.pt")
    with pytest.raises(ValueError, match="must be below"):
        layout.output_path(
            layout.root / "reports" / "wrong-category.json",
            category="models",
            default="x.json",
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        layout._descendant(layout.root / "link" / "artifact.pt")
    with pytest.raises(ValueError, match="single logical path component"):
        layout.metrics_path("train/../escape", "phase")
    with pytest.raises(ValueError, match="single logical path component"):
        layout.checkpoint_path("stage", "phase", candidate="../escape")
    with pytest.raises(ValueError, match="invalid traversal"):
        layout.legacy_output_path(
            "../best.pt", category="models/checkpoints/teacher", default="best.pt"
        )


def test_run_lock_rejects_second_writer(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    first = layout.lock()
    first.__enter__()
    try:
        with pytest.raises(RuntimeError):
            with layout.lock():
                pass
    finally:
        first.__exit__(None, None, None)


def test_metrics_reconcile_removes_append_before_checkpoint_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path, stage="teacher", phase="train")
    recorder.append({"epoch": 1, "val_acc": 0.0})
    committed_digest = recorder.digest
    recorder.append({"epoch": 2, "val_acc": 0.5})

    recovered = MetricsRecorder(path, stage="teacher", phase="train")
    recovered.reconcile(1, committed_digest)
    assert [item["epoch"] for item in recovered.records] == [1]
    assert len(path.read_text().splitlines()) == 1


def test_metrics_recovers_trailing_fragment_then_reconciles_checkpoint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path, stage="teacher", phase="train")
    recorder.append({"epoch": 1, "val_acc": 0.1})
    committed_digest = recorder.digest
    committed_content = path.read_bytes()
    recorder.append({"epoch": 2, "val_acc": 0.2})
    with path.open("ab") as stream:
        stream.write(b'{"epoch":3,"val_acc":')
    torn_content = path.read_bytes()

    recovered = MetricsRecorder(path, stage="teacher", phase="train")

    assert [item["epoch"] for item in recovered.records] == [1, 2]
    assert path.read_bytes() == torn_content
    with pytest.raises(RuntimeError, match="reconcile with a checkpoint"):
        recovered.append({"epoch": 3, "val_acc": 0.3})
    recovered.reconcile(1, committed_digest)
    assert [item["epoch"] for item in recovered.records] == [1]
    assert recovered.digest == committed_digest
    assert path.read_bytes() == committed_content


def test_metrics_reconcile_digest_mismatch_preserves_original_bytes(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path)
    recorder.append({"epoch": 1})
    with path.open("ab") as stream:
        stream.write(b'{"epoch":2')
    original_content = path.read_bytes()

    recovered = MetricsRecorder(path)
    with pytest.raises(ValueError, match="cannot be reconciled"):
        recovered.reconcile(1, "not-the-committed-digest")

    assert path.read_bytes() == original_content


@pytest.mark.parametrize(
    "content",
    [
        b'{"epoch":1}\n{"epoch":\n{"epoch":3}\n',
        b'{"epoch":1}\n{"epoch":\n',
    ],
)
def test_metrics_rejects_non_fragment_corruption(tmp_path, content):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="invalid metrics JSONL"):
        MetricsRecorder(path)

    assert path.read_bytes() == content


def test_metrics_fresh_reset_archives_existing_history(tmp_path):
    path = tmp_path / "metrics.jsonl"
    recorder = MetricsRecorder(path, stage="teacher", phase="train")
    recorder.append({"epoch": 1, "val_acc": 0.1})
    with path.open("ab") as stream:
        stream.write(b'{"epoch":2')
    original_content = path.read_bytes()
    recorder = MetricsRecorder(path, stage="teacher", phase="train")

    archive = recorder.reset_for_fresh_run()

    assert archive is not None
    assert archive.read_bytes() == original_content
    assert not path.exists()
    assert recorder.records == []
    assert recorder.count == 0
    assert recorder.digest == MetricsRecorder(path).digest
    assert recorder.reset_for_fresh_run() is None
    assert not path.exists()
    recorder.append({"epoch": 1, "val_acc": 0.9})
    assert [item["val_acc"] for item in recorder.records] == [0.9]


def test_metrics_fresh_reset_rejects_symlink(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text('{"epoch":1}\n', encoding="utf-8")
    path = tmp_path / "metrics.jsonl"
    path.symlink_to(target)
    recorder = MetricsRecorder(path)

    with pytest.raises(ValueError, match="symlinked metrics path"):
        recorder.reset_for_fresh_run()

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == '{"epoch":1}\n'


def test_registering_overwritten_artifact_refreshes_integrity_metadata(tmp_path):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    manifest = layout.new_manifest(command="test")
    artifact_path = layout.report_path("result.json")
    artifact_path.write_text("first", encoding="utf-8")
    first = layout.register_file(manifest, artifact_path, role="stage_report")
    first_digest = first["sha256"]

    artifact_path.write_text("a longer replacement", encoding="utf-8")
    refreshed = layout.register_file(manifest, artifact_path, role="updated_report")

    assert refreshed is first
    assert len(manifest["artifacts"]) == 1
    assert refreshed["role"] == "updated_report"
    assert refreshed["sha256"] == sha256_path(artifact_path)
    assert refreshed["sha256"] != first_digest
    assert refreshed["size"] == artifact_path.stat().st_size
    layout.save_manifest(manifest)


def test_manifest_run_id_is_stable_and_is_the_only_output_run_identity(tmp_path):
    root = tmp_path / "run"
    first_layout = ArtifactLayout(root)
    first_run_id = first_layout.manifest_run_id

    second_layout = ArtifactLayout(root)
    assert second_layout.manifest_run_id == first_run_id
    assert resolve_manifest_run_id(second_layout, first_run_id) == first_run_id

    with pytest.raises(ValueError, match="does not match manifest run_id"):
        resolve_manifest_run_id(second_layout, "different-run")


def test_run_session_refreshes_artifact_without_downgrading_role(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    artifact_path = layout.report_path("result.json")

    with run_session(layout=layout, command="test.refresh") as manifest:
        artifact_path.write_text("first", encoding="utf-8")
        layout.register_file(manifest, artifact_path, role="stage_report")
        artifact_path.write_text("replacement", encoding="utf-8")

    saved = yaml.safe_load((layout.root / "manifest.yaml").read_text())
    artifact = next(
        item for item in saved["artifacts"] if item["path"] == "reports/result.json"
    )
    assert artifact["role"] == "stage_report"
    assert artifact["sha256"] == sha256_path(artifact_path)
    assert artifact["size"] == artifact_path.stat().st_size


def test_recording_replaced_input_refreshes_integrity_metadata(tmp_path):
    input_path = tmp_path / "config.yaml"
    input_path.write_text("version: 1\n", encoding="utf-8")
    manifest = {"inputs": []}
    first = record_input(manifest, input_path, role="train_config")
    first_digest = first["sha256"]

    input_path.write_text("version: 200\n", encoding="utf-8")
    refreshed = record_input(manifest, input_path, role="train_config")

    assert refreshed is first
    assert len(manifest["inputs"]) == 1
    assert refreshed["sha256"] == sha256_path(input_path)
    assert refreshed["sha256"] != first_digest
    assert refreshed["size"] == input_path.stat().st_size


def test_logging_captures_plain_streams_and_warnings(tmp_path, capsys):
    layout = ArtifactLayout(tmp_path / "run")
    with run_session(layout.root, command="test.logging"):
        print("plain stdout")
        import sys

        print("plain stderr", file=sys.stderr)
        import warnings

        warnings.warn("captured warning")
    log_files = list((layout.root / "logs").glob("*.log"))
    assert len(log_files) == 1
    log = log_files[0].read_text()
    assert "plain stdout" in log
    assert "plain stderr" in log
    assert "captured warning" in log


def test_logging_captures_exception_traceback(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    with pytest.raises(RuntimeError, match="boom"):
        with run_session(layout.root, command="test.failure"):
            raise RuntimeError("boom")
    log = next((layout.root / "logs").glob("*.log")).read_text()
    assert "RuntimeError: boom" in log


def test_logging_propagates_manifest_finalization_failure(tmp_path):
    layout = ArtifactLayout(tmp_path / "run")
    artifact_path = layout.report_path("vanishing.json")

    with pytest.raises(ValueError, match="manifest artifact is missing"):
        with run_session(layout=layout, command="test.finalization") as manifest:
            artifact_path.write_text("temporary", encoding="utf-8")
            layout.register_file(manifest, artifact_path, role="stage_report")
            artifact_path.unlink()

    saved = yaml.safe_load((layout.root / "manifest.yaml").read_text())
    assert saved["status"] == "running"


def test_logging_preserves_command_error_when_finalization_also_fails(
    tmp_path,
):
    layout = ArtifactLayout(tmp_path / "run")
    artifact_path = layout.report_path("vanishing.json")
    with pytest.raises(RuntimeError, match="command failed") as exc_info:
        with run_session(layout=layout, command="test.double_failure") as manifest:
            artifact_path.write_text("temporary", encoding="utf-8")
            layout.register_file(manifest, artifact_path, role="stage_report")
            artifact_path.unlink()
            raise RuntimeError("command failed")

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("manifest artifact is missing" in note for note in notes)
    log = next((layout.root / "logs").glob("*.log")).read_text()
    assert "manifest finalization also failed" in log
