"""Explicit subprocess execution with per-sample evidence directories."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SimulationRun:
    sample_id: str
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    output_directory: Path
    timed_out: bool = False


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_simulator(
    command: Sequence[str],
    *,
    cwd: str | Path,
    sample_id: str,
    output_root: str | Path,
    timeout_seconds: float,
) -> SimulationRun:
    if not command:
        raise ValueError("simulator command must not be empty")
    directory = Path(output_root) / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=Path(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        timed_out = False
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = -1
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        stderr += "\nprocess timed out\n"
    (directory / "stdout.txt").write_text(stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(stderr, encoding="utf-8")
    (directory / "return_code.txt").write_text(f"{return_code}\n", encoding="utf-8")
    (directory / "command.txt").write_text("\n".join(command) + "\n", encoding="utf-8")
    return SimulationRun(
        sample_id=sample_id,
        command=tuple(command),
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        output_directory=directory,
        timed_out=timed_out,
    )
