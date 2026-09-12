import csv
import io
import json
import re
from pathlib import Path

import pytest

from kws.utils import graphs
from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import MetricsRecorder


@pytest.fixture(autouse=True)
def isolated_switch(monkeypatch):
    """Keep the process-wide switch from leaking between tests."""
    monkeypatch.setattr(graphs, "_enabled", None, raising=False)
    monkeypatch.delenv(graphs.GRAPHS_ENV_VAR, raising=False)
    graphs._index_signatures.clear()
    yield
    graphs._enabled = None


def _records(count=4):
    return [
        {
            "epoch": index + 1,
            "stage": "teacher",
            "phase": "ds_cnn_l",
            "global_step": (index + 1) * 10,
            "learning_rate": [0.001],
            "train_loss": 2.0 - index * 0.1,
            "val_loss": 2.1 - index * 0.12,
            "train_accuracy": 0.5 + index * 0.05,
            "val_acc": 0.4 + index * 0.06,
            "val_accuracy": 0.4 + index * 0.06,
        }
        for index in range(count)
    ]


def _payload(html_path: Path) -> dict:
    match = re.search(r"const DATA = (\{.*?\});\n", html_path.read_text(), re.S)
    assert match, "chart HTML must embed its own data"
    return json.loads(match.group(1))


def test_disabled_by_default_and_enabled_by_env(monkeypatch):
    assert graphs.enabled() is False
    monkeypatch.setenv(graphs.GRAPHS_ENV_VAR, "1")
    assert graphs.enabled() is True


def test_enable_propagates_to_child_processes():
    graphs.enable()
    assert graphs.enabled() is True
    # A stage launched as a subprocess must inherit the same switch.
    import os

    assert os.environ[graphs.GRAPHS_ENV_VAR] == "1"


def test_recorder_writes_chart_and_csv_only_when_enabled(tmp_path):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    metrics_path = layout.metrics_path("teacher", "ds_cnn_l")

    recorder = MetricsRecorder(metrics_path, layout=layout, stage="teacher", phase="ds_cnn_l")
    recorder.append(_records(1)[0])
    assert not (layout.root / "graphs").exists()

    graphs.enable()
    for record in _records()[1:]:
        recorder.append(record)

    csv_path = layout.root / "graphs" / "teacher" / "ds_cnn_l.csv"
    html_path = layout.root / "graphs" / "teacher" / "ds_cnn_l.html"
    assert csv_path.exists() and html_path.exists()
    assert (layout.root / "graphs" / "index.html").exists()

    rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    assert len(rows) == 4, "the CSV mirrors every JSONL record, not just new ones"
    assert rows[0]["epoch"] == "1" and rows[-1]["epoch"] == "4"
    # A single-element list is a scalar in disguise; keep it usable in a cell.
    assert rows[0]["learning_rate"] == "0.001"


def test_chart_buckets_axes_and_drops_aliased_series(tmp_path):
    root = tmp_path / "run"
    metrics_path = root / "metrics" / "teacher" / "ds_cnn_l.jsonl"
    html_path = graphs.update_phase(root, metrics_path, _records())

    payload = _payload(html_path)
    names = {item["name"]: item["axis"] for item in payload["series"]}
    assert names == {
        "train_loss": "loss",
        "val_loss": "loss",
        "train_accuracy": "accuracy",
        "val_accuracy": "accuracy",
    }, "val_acc duplicates val_accuracy and learning_rate is not a curve"
    assert payload["count"] == 4
    assert payload["best"].startswith("best val_accuracy")
    assert len({item["color"] for item in payload["series"]}) == 4


def test_chart_falls_back_for_an_unfamiliar_schema(tmp_path):
    root = tmp_path / "run"
    metrics_path = root / "metrics" / "sparsity" / "w12" / "pai.jsonl"
    records = [{"epoch": 1, "dendrites": 2.0}, {"epoch": 2, "dendrites": 5.0}]
    html_path = graphs.update_phase(root, metrics_path, records)

    assert html_path == root / "graphs" / "sparsity" / "w12" / "pai.html"
    payload = _payload(html_path)
    assert [item["name"] for item in payload["series"]] == ["dendrites"]
    assert payload["series"][0]["axis"] == "loss", "an unknown series still gets an axis"


def test_update_from_recorder_survives_a_broken_chart(tmp_path, monkeypatch):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    recorder = MetricsRecorder(
        layout.metrics_path("teacher", "ds_cnn_l"), layout=layout, stage="teacher", phase="ds_cnn_l"
    )
    graphs.enable()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(graphs, "update_phase", explode)
    # Losing a derived chart must not end an otherwise healthy training run.
    digest = recorder.append(_records(1)[0])
    assert digest and recorder.count == 1
    assert recorder.path.exists()


def test_rebuild_reads_existing_jsonl_and_skips_archives(tmp_path):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    metrics_path = layout.metrics_path("student", "distill")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()), encoding="utf-8"
    )
    archive = metrics_path.with_name("distill.deadbeef.archive.jsonl")
    archive.write_text(json.dumps(_records(1)[0]) + "\n", encoding="utf-8")

    written = graphs.rebuild(layout.root)

    assert written == [layout.root / "graphs" / "student" / "distill.html"]
    assert not (layout.root / "graphs" / "student" / "distill.deadbeef.archive.html").exists()


def test_rebuild_tolerates_a_partially_written_final_line(tmp_path):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    metrics_path = layout.metrics_path("teacher", "ds_cnn_l")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(record) + "\n" for record in _records())
    metrics_path.write_text(body + '{"epoch": 5, "train_los', encoding="utf-8")

    written = graphs.rebuild(layout.root)

    assert len(written) == 1
    assert _payload(written[0])["count"] == 4


def test_run_root_recovers_the_output_directory(tmp_path):
    root = tmp_path / "run"
    assert graphs.run_root_for(root / "metrics" / "teacher" / "a.jsonl") == root
    assert graphs.run_root_for(root / "metrics" / "sparsity" / "w12" / "pai.jsonl") == root
    assert graphs.run_root_for(tmp_path / "loose" / "a.jsonl") is None
