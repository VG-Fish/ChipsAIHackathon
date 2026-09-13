import pytest

from kws.optimize import pai_import


def test_pai_import_explains_headless_license_prompt(monkeypatch):
    def prompt_without_stdin(_module_name):
        raise EOFError

    monkeypatch.setattr(pai_import.importlib, "import_module", prompt_without_stdin)

    with pytest.raises(RuntimeError, match=r"uv run --env-file \.env") as error:
        pai_import.import_pai_module("perforatedai.globals_perforatedai")

    assert isinstance(error.value.__cause__, EOFError)
    assert ".venv/bin/python" in str(error.value)


def test_pai_import_returns_imported_module(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        pai_import.importlib,
        "import_module",
        lambda _module_name: expected,
    )

    assert pai_import.import_pai_module("perforatedai.utils_perforatedai") is expected


def test_pai_import_does_not_relabel_unrelated_import_failure(monkeypatch):
    expected = ModuleNotFoundError("compiled PAI module is unavailable")

    def fail_import(_module_name):
        raise expected

    monkeypatch.setattr(pai_import.importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError) as error:
        pai_import.import_pai_module("perforatedai.utils_perforatedai")

    assert error.value is expected
