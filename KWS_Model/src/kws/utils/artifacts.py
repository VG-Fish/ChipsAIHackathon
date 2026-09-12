"""Run-scoped artifact layout, manifests, locking, and atomic writers.

The KWS commands historically wrote to a mixture of paths supplied by CLI
flags and paths embedded in YAML.  This module is deliberately small and
dependency-light so every command can share one containment and persistence
contract without importing a training stage.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml


FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
_SENSITIVE_ARG_NAMES = {
    "token",
    "api-key",
    "api_key",
    "password",
    "passwd",
    "secret",
    "access-token",
    "access_token",
    "authorization",
    "private-key",
    "private_key",
}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _safe_component(value: str | Path, *, label: str) -> str:
    """Normalize one logical path component without accepting path syntax."""
    text = str(value)
    path = Path(text)
    if (
        not text
        or path.name != text
        or text in {".", ".."}
        or ".." in path.parts
    ):
        raise ValueError(f"{label} must be a single logical path component: {text!r}")
    return _slug(text)


def redact_argv(argv: list[str] | None) -> list[str]:
    """Keep invocation provenance useful without copying credential values."""
    values = list(argv or [])
    redacted: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        text = str(value)
        if text.startswith("--"):
            option = text[2:].split("=", 1)[0].lower()
            if option in _SENSITIVE_ARG_NAMES or any(
                marker in option for marker in ("token", "secret", "password", "credential")
            ):
                if "=" in text:
                    redacted.append(text.split("=", 1)[0] + "=<redacted>")
                else:
                    redacted.append(text)
                    redact_next = True
                continue
        redacted.append(text)
    return redacted


def sha256_path(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactLayout:
    """Paths and persistence helpers for one exact run root.

    ``root`` is never decorated with a timestamp.  Calling the same command
    with the same root is therefore the explicit resume/reuse operation.
    """

    def __init__(self, root: str | Path):
        if not str(root).strip():
            raise ValueError("artifact root must not be empty")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_run_id: str | None = None
        # The manifest is loaded lazily.  This lets ``run_session`` create a
        # new manifest with the actual command name while direct stage APIs can
        # still establish one through ``manifest_run_id`` before writing.
        if self.manifest_path.exists():
            self.load_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @staticmethod
    def _require_manifest_run_id(manifest: dict, *, source: str = "manifest") -> str:
        if not isinstance(manifest, dict):
            raise ValueError(f"{source} must be a mapping")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"{source} must contain a non-empty string run_id")
        return run_id

    def load_manifest(
        self,
        *,
        command: str = "kws",
        argv: list[str] | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> dict:
        """Load the run manifest, or create it for a direct layout user.

        Existing manifests must have a valid, stable ``run_id``.  A second
        manifest ID cannot be loaded through the same layout instance because
        that would make checkpoint provenance ambiguous.
        """
        manifest_path = self.manifest_path
        if manifest_path.exists():
            try:
                with manifest_path.open(encoding="utf-8") as stream:
                    manifest = yaml.safe_load(stream)
            except (OSError, yaml.YAMLError) as exc:
                raise ValueError(f"could not load artifact manifest {manifest_path}") from exc
            run_id = self._require_manifest_run_id(manifest, source=str(manifest_path))
            if self._manifest_run_id is not None and run_id != self._manifest_run_id:
                raise ValueError(
                    f"manifest run_id changed for {self.root}: "
                    f"was {self._manifest_run_id!r}, now {run_id!r}"
                )
            manifest.setdefault("invocations", [])
            manifest.setdefault("inputs", [])
            manifest.setdefault("artifacts", [])
            self._manifest_run_id = run_id
            return manifest
        return self.new_manifest(
            command=command,
            argv=argv,
            seed=seed,
            device=device,
        )

    @property
    def manifest_run_id(self) -> str:
        """Return the durable run ID shared by this output root."""
        manifest = self.load_manifest()
        return self._require_manifest_run_id(manifest, source=str(self.manifest_path))

    def ensure_tree(self) -> "ArtifactLayout":
        directories = (
            "logs",
            "metadata/configs",
            "metrics",
            "models/checkpoints/teacher",
            "models/checkpoints/student",
            "models/checkpoints/cluster",
            "models/checkpoints/quantize",
            "models/checkpoints/sparsity",
            "models/exported",
            "pai/candidates",
            "pai/reload",
            "reports",
        )
        for relative in directories:
            self._descendant(self.root / relative).mkdir(parents=True, exist_ok=True)
        return self

    def _descendant(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"artifact path escapes output root {self.root}: {path!s}"
            ) from exc
        return resolved

    def relative(self, path: str | Path) -> str:
        return self._descendant(path).relative_to(self.root).as_posix()

    def output_path(
        self,
        value: str | Path | None,
        *,
        category: str,
        default: str,
    ) -> Path:
        """Map an output-role name below a category and reject escapes.

        An absolute path is accepted only when it is already inside the active
        root.  Relative paths are category-local, which prevents a legacy
        ``reports/foo`` value from accidentally becoming a sibling of the run.
        """
        chosen = default if value is None else value
        raw = Path(chosen)
        category_root = self.root / category
        candidate = raw if raw.is_absolute() else category_root / raw
        resolved = self._descendant(candidate)
        if raw.is_absolute():
            try:
                resolved.relative_to(category_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"artifact path must be below {category_root}: {chosen!s}"
                ) from exc
        return resolved

    # Explicit aliases keep call sites readable and make the role distinction
    # obvious to callers migrating from legacy path flags.
    resolve_output = output_path

    def legacy_output_path(
        self,
        value: str | Path | None,
        *,
        category: str,
        default: str,
    ) -> Path:
        """Validate a legacy path, then map its filename to a fixed category."""
        chosen = default if value is None else value
        raw = Path(chosen).expanduser()
        if ".." in raw.parts or raw.name in {"", ".", ".."}:
            raise ValueError(f"output path contains invalid traversal: {chosen!s}")
        return self.output_path(
            raw if raw.is_absolute() else raw.name,
            category=category,
            default=default,
        )

    def input_path(self, value: str | Path) -> Path:
        """Resolve an input without rebasing it into the run root."""
        return Path(value).expanduser().resolve()

    def log_path(self, command: str, invocation_id: str | None = None) -> Path:
        stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        suffix = _slug(invocation_id or uuid.uuid4().hex[:8])
        return self._descendant(self.root / "logs" / f"{stamp}_{_slug(command)}_{suffix}.log")

    def metrics_path(self, stage: str, phase: str, *, candidate: str | None = None) -> Path:
        base = self.root / "metrics" / _safe_component(stage, label="stage")
        if candidate is not None:
            base = base / _safe_component(candidate, label="candidate")
        return self._descendant(
            base / f"{_safe_component(phase, label='phase')}.jsonl"
        )

    def checkpoint_path(
        self, stage: str, phase: str, kind: str = "latest", *, candidate: str | None = None
    ) -> Path:
        if kind not in {"best", "latest"}:
            raise ValueError("checkpoint kind must be 'best' or 'latest'")
        stage_name = _safe_component(stage, label="stage")
        base = self.root / "models" / "checkpoints" / stage_name
        if candidate is not None:
            base = base / _safe_component(candidate, label="candidate")
        base = base / _safe_component(phase, label="phase") if candidate is not None else base
        return self._descendant(base / f"{kind}.pt")

    def pai_candidate_path(self, candidate: str) -> Path:
        if Path(candidate).name != candidate or candidate in {"", ".", ".."}:
            raise ValueError("PAI candidate must be a single leaf name")
        return self._descendant(self.root / "pai" / "candidates" / _slug(candidate))

    def exported_path(self, name: str) -> Path:
        return self._descendant(self.root / "models" / "exported" / self._leaf(name))

    def report_path(self, name: str) -> Path:
        return self._descendant(self.root / "reports" / self._leaf(name))

    def metadata_config_path(self, name: str) -> Path:
        return self._descendant(self.root / "metadata" / "configs" / self._leaf(name))

    @staticmethod
    def _leaf(value: str | Path) -> str:
        path = Path(value)
        if path.name != str(value) or path.name in {"", ".", ".."}:
            raise ValueError(f"expected a single output filename, got {value!s}")
        return path.name

    @contextlib.contextmanager
    def lock(self) -> Iterator[Path]:
        """Acquire an advisory single-writer lock for this run."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".run.lock"
        stream = lock_path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError(f"run root is already locked: {self.root}") from exc
            else:
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError(f"run root is already locked: {self.root}") from exc
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
            yield lock_path
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()

    def _temporary(self, destination: Path) -> Path:
        destination = self._descendant(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"

    def atomic_text(self, destination: str | Path, value: str) -> Path:
        destination = self._descendant(destination)
        temporary = self._temporary(destination)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def atomic_json(self, destination: str | Path, value: Any) -> Path:
        return self.atomic_text(
            destination,
            json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        )

    def atomic_yaml(self, destination: str | Path, value: Any) -> Path:
        return self.atomic_text(
            destination,
            yaml.safe_dump(_json_safe(value), sort_keys=False, default_flow_style=False),
        )

    def atomic_torch_save(self, destination: str | Path, value: Any) -> Path:
        destination = self._descendant(destination)
        temporary = self._temporary(destination)
        try:
            with temporary.open("wb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    atomic_write_text = atomic_text
    atomic_write_json = atomic_json
    atomic_write_yaml = atomic_yaml
    atomic_save = atomic_torch_save

    def register_file(
        self,
        manifest: dict,
        path: str | Path,
        *,
        role: str,
        replace_role: bool = True,
    ) -> dict:
        resolved = self._descendant(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        relative = self.relative(resolved)
        digest = sha256_path(resolved)
        size = resolved.stat().st_size
        for existing in manifest.setdefault("artifacts", []):
            if existing.get("path") == relative:
                # A canonical path may be replaced many times during a run
                # (for example latest.pt or a stage report).  Refresh its
                # integrity metadata in place so callers holding this manifest
                # never retain a stale digest.  Generic discovery scans can
                # preserve a more specific role with replace_role=False.
                if replace_role or "role" not in existing:
                    existing["role"] = role
                existing["sha256"] = digest
                existing["size"] = size
                return existing
        artifact = {
            "role": role,
            "path": relative,
            "sha256": digest,
            "size": size,
        }
        manifest.setdefault("artifacts", []).append(artifact)
        return artifact

    def audit_manifest(self, manifest: dict) -> None:
        for artifact in manifest.get("artifacts", []):
            path = self._descendant(self.root / artifact["path"])
            if not path.is_file():
                raise ValueError(f"manifest artifact is missing: {path}")
            if path.is_symlink():
                raise ValueError(f"manifest artifact is a symlink: {path}")
            if path.stat().st_size != artifact.get("size"):
                raise ValueError(f"manifest artifact size changed: {path}")
            if sha256_path(path) != artifact.get("sha256"):
                raise ValueError(f"manifest artifact digest changed: {path}")

    def new_manifest(
        self,
        *,
        command: str,
        argv: list[str] | None = None,
        seed: int | None = None,
        device: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        self.ensure_tree()
        selected_run_id = run_id or uuid.uuid4().hex
        if not isinstance(selected_run_id, str) or not selected_run_id.strip():
            raise ValueError("manifest run_id must be a non-empty string")
        manifest = {
            "format_version": FORMAT_VERSION,
            "run_id": selected_run_id,
            "status": "running",
            "command": command,
            "argv": redact_argv(argv or sys.argv),
            "started_at": _utc_now(),
            "seed": seed,
            "device": device,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "packages": _package_versions(),
            "git": _git_metadata(),
            "invocations": [],
            "inputs": [],
            "artifacts": [],
        }
        self.atomic_yaml(self.manifest_path, manifest)
        self._manifest_run_id = selected_run_id
        return manifest

    def save_manifest(self, manifest: dict) -> Path:
        run_id = self._require_manifest_run_id(manifest, source="manifest")
        if self._manifest_run_id is not None and run_id != self._manifest_run_id:
            raise ValueError(
                f"manifest run_id {run_id!r} does not match layout run_id "
                f"{self._manifest_run_id!r}"
            )
        self.audit_manifest(manifest)
        self._manifest_run_id = run_id
        return self.atomic_yaml(self.manifest_path, manifest)


def _git_metadata() -> dict:
    try:
        root = Path.cwd()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip() != ""
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "numpy", "pyyaml", "perforatedai", "kws"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
