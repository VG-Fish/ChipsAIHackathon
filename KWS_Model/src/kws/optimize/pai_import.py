"""Headless-safe import boundary for PerforatedAI's compiled modules."""

import importlib
from typing import Any


def import_pai_module(module_name: str) -> Any:
    """Import a PAI module and explain a non-interactive license prompt.

    ``perforatedbp`` validates its license while it is imported. When neither
    the run environment nor the working directory supplies that configuration,
    the package falls back to an interactive prompt. Pipeline processes have
    no interactive stdin, so the prompt raises ``EOFError`` before any KWS
    configuration or model code runs.
    """
    try:
        return importlib.import_module(module_name)
    except EOFError as exc:
        raise RuntimeError(
            "PerforatedAI attempted interactive license setup, but this command "
            "has no interactive stdin. Run PAI-dependent commands from the "
            "KWS_Model directory with `uv run --env-file .env python -m ...`; "
            "do not invoke `.venv/bin/python` from a temporary working directory."
        ) from exc
