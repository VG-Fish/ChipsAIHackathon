import subprocess

import pytest

from kws.hardware.neurosim.source import (
    SourceValidationError,
    validate_neurosim_source,
)


def test_source_validation_records_git_provenance_without_requiring_git(tmp_path, monkeypatch):
    (tmp_path / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    def fake_run(*args, **kwargs):
        command = args[0]
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        if command[-3:] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(command, 0, "https://example.invalid/neurosim\n", "")
        if command[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "main\n", "")
        if command[-2:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    identity = validate_neurosim_source(tmp_path)

    assert identity.git_commit == "abc123"
    assert identity.git_remote.endswith("neurosim")
    assert identity.dirty is False


def test_source_validation_fails_without_build_files(tmp_path):
    with pytest.raises(SourceValidationError, match="Makefile"):
        validate_neurosim_source(tmp_path)
