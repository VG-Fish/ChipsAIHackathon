"""Validation and provenance for an external NeuroSim checkout."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import HardwareConfig


class SourceValidationError(ValueError):
    """Raised when an external NeuroSim source tree is not usable."""


@dataclass(frozen=True)
class SourceIdentity:
    root: Path
    source_sha256: str
    git_remote: str | None
    git_branch: str | None
    git_commit: str | None
    dirty: bool | None
    warnings: tuple[str, ...] = ()


def _git(root: Path, *arguments: str, allow_empty: bool = False) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value or allow_empty else None


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "build" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_neurosim_source(root: str | Path) -> SourceIdentity:
    path = Path(root).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise SourceValidationError(f"NeuroSim source root does not exist: {path}")
    root_makefile = next(
        (candidate for candidate in ("Makefile", "makefile") if (path / candidate).is_file()),
        None,
    )
    nested = path / "Training_pytorch" / "NeuroSIM"
    nested_makefile = next(
        (candidate for candidate in ("Makefile", "makefile") if (nested / candidate).is_file()),
        None,
    )
    if root_makefile is None and nested_makefile is None:
        raise SourceValidationError(f"NeuroSim source root is missing Makefile: {path}")

    warnings: list[str] = []
    remote = _git(path, "remote", "get-url", "origin")
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain", allow_empty=True)
    if commit is None:
        warnings.append("Git commit metadata is unavailable for the NeuroSim source")
    dirty = status is not None and bool(status)
    return SourceIdentity(
        root=path,
        source_sha256=_source_hash(path),
        git_remote=remote,
        git_branch=branch,
        git_commit=commit,
        dirty=dirty if status is not None else None,
        warnings=tuple(warnings),
    )


def source_build_root(root: str | Path) -> Path:
    """Locate the build directory in either official or flat source layouts."""

    path = Path(root).expanduser().resolve()
    nested = path / "Training_pytorch" / "NeuroSIM"
    for candidate in (nested, path):
        if any((candidate / name).is_file() for name in ("Makefile", "makefile")):
            return candidate
    raise SourceValidationError(f"NeuroSim source root is missing Makefile: {path}")


def resolve_source_root(config: HardwareConfig) -> Path:
    value = os.environ.get(config.source.root_env)
    if not value:
        raise SourceValidationError(
            f"environment variable {config.source.root_env} is not set; "
            "provide an external NeuroSim V2.1 checkout"
        )
    return Path(value).expanduser().resolve()
