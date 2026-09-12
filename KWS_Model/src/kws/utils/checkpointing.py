"""Durable metrics and resumable training-state helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from kws.utils.artifacts import ArtifactLayout, sha256_path


CHECKPOINT_FORMAT_VERSION = 1


def recipe_fingerprint(payload: Any) -> str:
    encoded = yaml.safe_dump(_safe(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    # MPS has no public state API.  Keeping this explicit makes the limitation
    # visible in a checkpoint rather than pretending it can be restored.
    state["torch_mps_rng_state"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    if state.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state.get("torch_cuda_rng_states") or []
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def atomic_torch_save(path: str | Path, value: Any) -> Path:
    """Atomically save a torch payload beside its final destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
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


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    """Atomically replace a small durable state file with exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _digest_lines(lines: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line)
    return digest.hexdigest()


def is_training_state(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") == "kws_training_state"


def require_manifest_run_id(manifest: Mapping[str, Any]) -> str:
    """Return a manifest run ID, rejecting missing or malformed manifests.

    Callers that do not configure ``output_dir`` remain compatible by not
    calling this helper; checkpoint writers that do have a manifest must use
    its non-empty string ID.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest must contain a non-empty string run_id")
    return run_id


def require_training_state(value: Any, *, source: str | Path = "checkpoint") -> dict:
    if not is_training_state(value):
        raise ValueError(
            f"{source} is not a resumable training-state checkpoint; "
            "legacy best-only checkpoints are valid for inference or warm starts, "
            "but cannot be used with --resume-from"
        )
    if int(value.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{source} uses unsupported training checkpoint format "
            f"{value.get('format_version')!r}"
        )
    return value


def validate_resume_recipe(
    checkpoint: dict, expected_fingerprint: str | None, *, source: str | Path = "checkpoint"
) -> None:
    require_training_state(checkpoint, source=source)
    actual = checkpoint.get("recipe_fingerprint")
    if expected_fingerprint is not None and actual != expected_fingerprint:
        raise ValueError(
            f"resume recipe mismatch for {source}: checkpoint has {actual!r}, "
            f"requested {expected_fingerprint!r}; use a separately named warm-start/reset-optimizer path"
        )


class MetricsRecorder:
    """Append-only JSONL with a digest for checkpoint commit reconciliation."""

    schema_version = 1

    def __init__(
        self,
        path: str | Path,
        *,
        layout: ArtifactLayout | None = None,
        stage: str = "train",
        phase: str = "default",
    ):
        self.layout = layout
        self.path = Path(path)
        if layout is not None:
            self.path = layout._descendant(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.phase = phase
        (
            self.records,
            self._record_lines,
            self.digest,
            self._has_trailing_fragment,
        ) = self._read()

    def _read(self) -> tuple[list[dict], list[bytes], str, bool]:
        if not self.path.exists():
            return [], [], hashlib.sha256().hexdigest(), False
        content = self.path.read_bytes()
        records: list[dict] = []
        record_lines: list[bytes] = []
        offset = 0
        for line_number, raw in enumerate(content.splitlines(keepends=True), 1):
            next_offset = offset + len(raw)
            if not raw.strip():
                offset = next_offset
                continue
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                is_unterminated_final_line = (
                    next_offset == len(content)
                    and not raw.endswith((b"\n", b"\r"))
                )
                if is_unterminated_final_line:
                    # append() writes one newline-terminated record at a time.
                    # Only an unterminated malformed final physical line can be
                    # evidence of an interrupted append; all other corruption
                    # remains a hard error.  Keep the original bytes until a
                    # checkpoint commit validates which prefix is durable.
                    return records, record_lines, _digest_lines(record_lines), True
                raise ValueError(
                    f"invalid metrics JSONL at {self.path}:{line_number}"
                ) from exc
            records.append(record)
            record_lines.append(raw)
            offset = next_offset
        return records, record_lines, _digest_lines(record_lines), False

    def reconcile(self, committed_count: int, committed_digest: str | None) -> None:
        """Make the JSONL prefix agree with the latest checkpoint commit."""
        if committed_count < 0 or committed_count > len(self.records):
            raise ValueError(
                f"metrics/checkpoint commit mismatch: checkpoint count {committed_count}, "
                f"file count {len(self.records)}"
            )
        kept_records = self.records[:committed_count]
        kept_lines = self._record_lines[:committed_count]
        kept_digest = _digest_lines(kept_lines)
        if committed_digest is not None and kept_digest != committed_digest:
            raise ValueError("metrics prefix cannot be reconciled with the checkpoint")
        if committed_count == len(self.records) and not self._has_trailing_fragment:
            return
        _atomic_write_bytes(self.path, b"".join(kept_lines))
        self.records = kept_records
        self._record_lines = kept_lines
        self.digest = kept_digest
        self._has_trailing_fragment = False

    def reset_for_fresh_run(self) -> Path | None:
        """Archive conflicting canonical history and start with an empty JSONL.

        Fresh/forced phase callers must opt into this operation before epoch 1;
        resume callers should use :meth:`reconcile` instead.  Existing nonempty
        history is moved byte-for-byte to a unique sibling before the canonical
        path is reset.  Callers should hold the run lock and separately reset
        conflicting checkpoints or manifest records.
        """
        archive = None
        if self.path.is_symlink():
            raise ValueError(f"refusing to reset symlinked metrics path: {self.path}")
        if self.path.exists() and self.path.stat().st_size:
            archive = self.path.with_name(
                f"{self.path.stem}.{uuid.uuid4().hex}.archive{self.path.suffix}"
            )
            self.path.replace(archive)
            _fsync_directory(self.path.parent)
        elif self.path.exists():
            self.path.unlink()
            _fsync_directory(self.path.parent)
        self.records = []
        self._record_lines = []
        self.digest = hashlib.sha256().hexdigest()
        self._has_trailing_fragment = False
        return archive

    def append(self, record: dict) -> str:
        if self._has_trailing_fragment:
            raise RuntimeError(
                "metrics JSONL has an interrupted trailing fragment; "
                "reconcile with a checkpoint or reset for a fresh run before appending"
            )
        normalized = dict(record)
        normalized.setdefault("schema_version", self.schema_version)
        normalized.setdefault("stage", self.stage)
        normalized.setdefault("phase", self.phase)
        line = (json.dumps(_safe(normalized), sort_keys=True, separators=(",", ":")) + "\n").encode()
        is_new_file = not self.path.exists()
        with self.path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if is_new_file:
            _fsync_directory(self.path.parent)
        self.records.append(normalized)
        self._record_lines.append(line)
        self.digest = _digest_lines(self._record_lines)
        return self.digest

    @property
    def count(self) -> int:
        return len(self.records)


def write_phase_summary(
    layout: ArtifactLayout,
    *,
    stage: str,
    phase: str,
    result: Any,
    artifacts: dict[str, str | Path] | None = None,
    candidate: str | None = None,
) -> Path:
    """Atomically update the portable stage/phase summary."""
    path = layout.root / "metrics" / "summaries.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            summary = yaml.safe_load(stream) or {}
    else:
        summary = {"format_version": 1, "phases": {}}
    payload = result.as_dict() if hasattr(result, "as_dict") else _safe(result)
    if isinstance(payload, dict):
        payload = {
            key: value for key, value in payload.items() if key != "history"
        }
    payload["metrics"] = layout.relative(layout.metrics_path(stage, phase, candidate=candidate))
    for role, artifact in (artifacts or {}).items():
        payload[role] = layout.relative(artifact)
    summary.setdefault("phases", {})[f"{stage}/{phase}"] = payload
    layout.atomic_yaml(path, summary)
    return path


def build_training_state(
    *,
    run_id: str | None = None,
    stage: str,
    phase: str,
    completed_epoch: int,
    target_epochs: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
    best_model_state_dict: dict[str, Any],
    history: list[dict],
    metrics: MetricsRecorder | None,
    recipe: Any,
    upstream: dict[str, Any] | None = None,
    stage_specific_state: dict[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict:
    manifest_run_id = (
        require_manifest_run_id(manifest) if manifest is not None else None
    )
    if manifest_run_id is not None:
        if run_id is not None and run_id != manifest_run_id:
            raise ValueError(
                f"checkpoint run_id {run_id!r} does not match manifest run_id "
                f"{manifest_run_id!r}"
            )
        run_id = manifest_run_id
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            "checkpoint run_id is required when building a training state"
        )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "kws_training_state",
        "run_id": run_id,
        "stage": stage,
        "phase": phase,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "global_step": global_step,
        "target_epochs": target_epochs,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()) if scheduler else None,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "best_model_state_dict": copy.deepcopy(best_model_state_dict),
        "history_length": len(history) if metrics is None else metrics.count,
        "last_metric_digest": None if metrics is None else metrics.digest,
        **capture_rng_state(),
        "recipe": _safe(recipe),
        "recipe_fingerprint": recipe_fingerprint(recipe),
        "upstream": _safe(upstream or {}),
        "stage_specific_state": copy.deepcopy(stage_specific_state or {}),
    }


def record_input(manifest: dict, path: str | Path, *, role: str) -> dict:
    resolved = Path(path).expanduser().resolve()
    for existing in manifest.setdefault("inputs", []):
        if existing.get("role") == role and existing.get("path") == str(resolved):
            if resolved.is_file():
                existing["sha256"] = sha256_path(resolved)
                existing["size"] = resolved.stat().st_size
            else:
                existing.pop("sha256", None)
                existing.pop("size", None)
            return existing
    item = {"role": role, "path": str(resolved)}
    if resolved.is_file():
        item["sha256"] = sha256_path(resolved)
        item["size"] = resolved.stat().st_size
    manifest.setdefault("inputs", []).append(item)
    return item


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
