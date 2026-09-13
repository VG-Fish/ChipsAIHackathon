import sys
from contextlib import nullcontext
from typing import Any, cast

import pytest

import kws.pipeline as pipeline
import kws.train as train
from kws.utils.artifacts import resolve_resume_dir


def test_resume_dir_resolves_an_existing_output_root(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    assert resolve_resume_dir(None, output_dir) == str(output_dir)
    assert resolve_resume_dir(output_dir, output_dir) == str(output_dir)


def test_resume_dir_requires_an_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        resolve_resume_dir(None, tmp_path / "missing")

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("checkpoint")
    with pytest.raises(ValueError, match="existing directory"):
        resolve_resume_dir(None, file_path)


def test_resume_dir_rejects_a_different_output_root(tmp_path):
    resume_dir = tmp_path / "resume"
    output_dir = tmp_path / "other"
    resume_dir.mkdir()
    output_dir.mkdir()

    with pytest.raises(ValueError, match="same directory"):
        resolve_resume_dir(output_dir, resume_dir)


def test_no_resume_dir_preserves_the_existing_output_value(tmp_path):
    output_dir = tmp_path / "run"

    assert resolve_resume_dir(output_dir, None) == output_dir
    assert resolve_resume_dir(None, None) is None


def test_train_cli_resume_dir_implies_resume(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(train, "load_yaml", lambda _path: {})
    monkeypatch.setattr(
        train,
        "run_session",
        lambda *_args, **_kwargs: cast(Any, nullcontext()),
    )
    monkeypatch.setattr(
        train,
        "train",
        lambda *_args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kws.train",
            "--model-config",
            "model.yaml",
            "--train-config",
            "train.yaml",
            "--resume-dir",
            str(output_dir),
        ],
    )

    train.main()

    assert captured["output_dir"] == str(output_dir)
    assert captured["resume"] is True


def test_pipeline_cli_resume_dir_selects_the_existing_run_root(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(pipeline, "load_yaml", lambda _path: {"output_dir": None})
    monkeypatch.setattr(
        pipeline,
        "run_session",
        lambda *_args, **_kwargs: cast(Any, nullcontext()),
    )
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda *_args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["kws.pipeline", "--stages", "student", "--resume-dir", str(output_dir)],
    )

    pipeline.main()

    assert captured["output_dir"] == str(output_dir)
