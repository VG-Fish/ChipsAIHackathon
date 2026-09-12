"""Project logging and per-invocation stdout/stderr capture."""

from __future__ import annotations

import contextlib
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from kws.utils.artifacts import ArtifactLayout, redact_argv


_HANDLER_MARKER = "_kws_handler"
_DEFAULT_CONFIGURED = False


class _Tee:
    def __init__(self, terminal, log_stream):
        self.terminal = terminal
        self.log_stream = log_stream

    def write(self, value):
        self.terminal.write(value)
        self.log_stream.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log_stream.flush()

    def isatty(self):
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")


def get_logger(name: str) -> logging.Logger:
    """Return a propagating logger; configuration belongs to ``run_session``."""
    global _DEFAULT_CONFIGURED
    if not _DEFAULT_CONFIGURED and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _DEFAULT_CONFIGURED = True
    return logging.getLogger(name)


@contextlib.contextmanager
def run_session(
    output_dir: str | Path | None = None,
    *,
    command: str = "kws",
    argv: list[str] | None = None,
    seed: int | None = None,
    device: str | None = None,
    layout: ArtifactLayout | None = None,
    inputs: list[tuple[str | Path, str]] | None = None,
) -> Iterator[dict | None]:
    """Configure one invocation and restore process-global streams on exit."""
    global _DEFAULT_CONFIGURED
    if output_dir is None and layout is None:
        yield None
        return

    layout = layout or ArtifactLayout(output_dir)
    layout.ensure_tree()
    invocation_id = uuid.uuid4().hex[:12]
    log_path = layout.log_path(command, invocation_id)
    manifest_path = layout.manifest_path

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_stream = log_path.open("a", encoding="utf-8")
    tee_stdout = _Tee(original_stdout, log_stream)
    tee_stderr = _Tee(original_stderr, log_stream)
    sys.stdout, sys.stderr = tee_stdout, tee_stderr
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    for old_handler in old_handlers:
        if getattr(old_handler, _HANDLER_MARKER, False):
            root.removeHandler(old_handler)
            with contextlib.suppress(Exception):
                old_handler.close()
    handler = logging.StreamHandler(tee_stdout)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    warnings_were_captured = getattr(logging, "_warnings_showwarning", None) is not None
    logging.captureWarnings(True)
    manifest = None
    invocation = None
    command_error: BaseException | None = None
    logger = logging.getLogger(command)
    try:
        with layout.lock():
            manifest = layout.load_manifest(
                command=command, argv=argv, seed=seed, device=device
            )
            if inputs:
                from kws.utils.checkpointing import record_input

                for path, role in inputs:
                    record_input(manifest, path, role=role)
            invocation = {
                "id": invocation_id,
                "command": command,
                "argv": redact_argv(argv or sys.argv),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "log": layout.relative(log_path),
                "status": "running",
            }
            manifest.setdefault("invocations", []).append(invocation)
            manifest["status"] = "running"
            layout.atomic_yaml(manifest_path, manifest)
            logger.info("KWS invocation %s started", invocation_id)
            try:
                yield manifest
            except BaseException as exc:
                command_error = exc
                logger.exception("KWS invocation failed")
                invocation["status"] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                invocation["error"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                invocation["ended_at"] = datetime.now(timezone.utc).isoformat()
                raise
            else:
                invocation["status"] = "completed"
                invocation["ended_at"] = datetime.now(timezone.utc).isoformat()
                manifest["status"] = "completed"
                manifest["ended_at"] = invocation["ended_at"]
            finally:
                logger.info("KWS invocation %s ended", invocation_id)
                handler.flush()
                log_stream.flush()
                try:
                    for artifact in layout.root.rglob("*"):
                        if (
                            not artifact.is_file()
                            or artifact.name in {"manifest.yaml", ".run.lock"}
                            or "logs" in artifact.relative_to(layout.root).parts
                            or artifact.name.endswith(".tmp")
                        ):
                            continue
                        layout.register_file(
                            manifest,
                            artifact,
                            role="runtime_artifact",
                            replace_role=False,
                        )
                    layout.register_file(manifest, log_path, role="invocation_log")
                    layout.save_manifest(manifest)
                except Exception as finalization_error:
                    if command_error is None:
                        raise
                    note = (
                        "manifest finalization also failed: "
                        f"{type(finalization_error).__name__}: {finalization_error}"
                    )
                    if hasattr(command_error, "add_note"):
                        command_error.add_note(note)
                    logger.exception(note)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        root.removeHandler(handler)
        handler.close()
        log_stream.close()
        root.setLevel(old_level)
        if not warnings_were_captured:
            logging.captureWarnings(False)
        _DEFAULT_CONFIGURED = False
