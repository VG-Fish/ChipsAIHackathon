"""Tests for scripts/run_sparknet_grow_sweep.sh: arms, guards and skips.

Every test points OUTPUT_ROOT at a temporary directory and GROW_CMD at a stub
driver, so nothing trains.  Each runs under every bash on the machine
(/bin/bash is 3.2 on macOS).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_sparknet_grow_sweep.sh"
SWEEP_VARIABLES = (
    "WIDTHS", "SEEDS", "PLACEMENTS", "ARM_SUFFIX", "EXTRA_ARGS", "OUTPUT_ROOT",
    "DATA_CONFIG", "TRAIN_CONFIG", "SWITCH_EPOCH", "CANDIDATE_EPOCHS",
    "MAX_MINUTES", "DRY_RUN", "GROW_CMD",
)

_bashes = []
for candidate in ("/bin/bash", shutil.which("bash")):
    if candidate and os.path.exists(candidate):
        resolved = os.path.realpath(candidate)
        if resolved not in [os.path.realpath(b) for b in _bashes]:
            _bashes.append(candidate)
BASHES = _bashes

STUB = textwrap.dedent(
    """\
    #!{python}
    # Stub driver: records its argv and writes a complete summary.
    import json, os, sys
    from pathlib import Path

    argv = sys.argv[1:]
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(json.dumps(argv) + "\\n")

    def value(flag):
        return argv[argv.index(flag) + 1]

    reports = Path(value("--output-dir")) / "reports"
    reports.mkdir(parents=True)
    (reports / "grow_summary.yaml").write_text(
        "status: complete\\nplacement: {{}}\\narm: {{}}\\n".format(value("--placement"), value("--arm"))
    )
    """
)


@pytest.fixture(params=BASHES)
def bash(request):
    return request.param


@pytest.fixture
def sweep(tmp_path, bash):
    stub = tmp_path / "stub_driver.py"
    stub.write_text(STUB.format(python=sys.executable))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "stub_calls.jsonl"
    output_root = tmp_path / "out"

    def run(**overrides):
        env = {k: v for k, v in os.environ.items() if k not in SWEEP_VARIABLES}
        env.update(
            OUTPUT_ROOT=str(output_root), GROW_CMD=str(stub), STUB_LOG=str(log),
            WIDTHS="8", PLACEMENTS="fc",
        )
        for key, value in overrides.items():  # None: leave the script's default
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        result = subprocess.run(
            [bash, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    run.output_root = output_root
    return run


def write_summary(run_dir: Path, text: str) -> Path:
    (run_dir / "reports").mkdir(parents=True)
    summary = run_dir / "reports/grow_summary.yaml"
    summary.write_text(textwrap.dedent(text))
    return summary


def planned_runs(stdout: str) -> list[tuple[str, str]]:
    """(label, output_dir) of every `run` line of a dry-run plan."""
    return re.findall(r"^run   (.+?)  -> (\S+)  \(log:", stdout, flags=re.M)


def planned_commands(stdout: str) -> list[list[str]]:
    lines = stdout.splitlines()
    return [lines[i + 1].split() for i, line in enumerate(lines) if line.startswith("run   ")]


def table_rows(stdout: str) -> list[list[str]]:
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("arm ")) + 2
    return [line.split() for line in lines[start:] if line.strip()]


def test_default_seeds_are_zero_to_four_and_default_arm_dirs_are_unchanged(sweep):
    result, calls = sweep(DRY_RUN=1, PLACEMENTS=None)  # default placements
    assert result.returncode == 0, result.stderr
    assert calls == []
    runs = planned_runs(result.stdout)
    root = sweep.output_root
    expected = [
        (f"{placement} C8 seed{seed}", f"{root}/{placement}/c8-seed{seed}")
        for seed in range(5)
        for placement in ("fc", "pointwise")
    ]
    assert runs == expected
    for command, placement in zip(planned_commands(result.stdout), ["fc", "pointwise"] * 5):
        assert command[-2:] == ["--arm", placement]
    assert [row[0] for row in table_rows(result.stdout)] == ["fc", "pointwise"] * 5
    assert not root.exists()  # a dry run creates nothing


def test_arm_suffix_names_the_directory_and_passes_arm_then_extra_args(sweep):
    result, calls = sweep(
        DRY_RUN=1, SEEDS=0, ARM_SUFFIX="-wd1e-3", EXTRA_ARGS="--dendrite-weight-decay 0.001"
    )
    assert result.returncode == 0, result.stderr
    assert calls == []
    assert planned_runs(result.stdout) == [
        ("fc-wd1e-3 C8 seed0", f"{sweep.output_root}/fc-wd1e-3/c8-seed0")
    ]
    (command,) = planned_commands(result.stdout)
    assert command[-4:] == ["--arm", "fc-wd1e-3", "--dendrite-weight-decay", "0.001"]
    assert command[command.index("--placement") + 1] == "fc"
    assert command[command.index("--output-dir") + 1] == f"{sweep.output_root}/fc-wd1e-3/c8-seed0"
    assert table_rows(result.stdout) == [["fc-wd1e-3", "C8", "0", "-", "would-run"]]
    assert "arms: fc-wd1e-3" in result.stdout


def test_driver_receives_arm_and_extra_args_and_the_run_lands_in_the_arm_dir(sweep):
    result, calls = sweep(SEEDS=0, ARM_SUFFIX="-sham", EXTRA_ARGS="--sham")
    assert result.returncode == 0, result.stderr
    (argv,) = calls
    assert argv[-3:] == ["--arm", "fc-sham", "--sham"]
    run_dir = sweep.output_root / "fc-sham/c8-seed0"
    assert "arm: fc-sham" in (run_dir / "reports/grow_summary.yaml").read_text()
    assert (sweep.output_root / "fc-sham/c8-seed0.log").exists()
    assert not (sweep.output_root / "fc").exists()
    assert table_rows(result.stdout)[0][0] == "fc-sham"
    assert table_rows(result.stdout)[0][-1] == "ok"


def test_switch_epoch_goes_before_arm_and_extra_args(sweep):
    result, _ = sweep(DRY_RUN=1, SEEDS=0, ARM_SUFFIX="-switch170", SWITCH_EPOCH=170, EXTRA_ARGS="--sham")
    assert result.returncode == 0, result.stderr
    (command,) = planned_commands(result.stdout)
    assert command[-5:] == ["--switch-epoch", "170", "--arm", "fc-switch170", "--sham"]


@pytest.mark.parametrize("dry_run", [0, 1])
@pytest.mark.parametrize(
    "variant",
    [
        {"EXTRA_ARGS": "--sham"},
        {"EXTRA_ARGS": "--switch-epoch 170"},
        {"SWITCH_EPOCH": "170"},
        {"CANDIDATE_EPOCHS": "5"},
    ],
)
def test_variant_without_arm_suffix_is_refused(sweep, variant, dry_run):
    result, calls = sweep(SEEDS=0, DRY_RUN=dry_run, **variant)
    assert result.returncode == 1
    assert "ARM_SUFFIX is empty" in result.stderr
    assert calls == []
    assert not sweep.output_root.exists()
    assert "run   " not in result.stdout


def test_whitespace_only_extra_args_is_not_a_variant(sweep):
    result, _ = sweep(DRY_RUN=1, SEEDS=0, EXTRA_ARGS="   ")
    assert result.returncode == 0, result.stderr
    (command,) = planned_commands(result.stdout)
    assert command[-2:] == ["--arm", "fc"]


def test_invalid_arm_suffix_fails_preflight(sweep):
    result, calls = sweep(SEEDS=0, ARM_SUFFIX="/x")
    assert result.returncode == 1
    assert "invalid ARM_SUFFIX" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    "summary,arm_suffix,recorded",
    [
        ("status: complete\nplacement: fc\narm: fc\n", "-sham", "fc"),
        ("status: complete\nplacement: fc\n", "-sham", "fc"),  # no arm: counts as placement
        ("status: complete\nplacement: fc\narm: fc-sham\n", "", "fc-sham"),
        ("status: complete\nplacement: fc\narm: fc-wd1e-3\n", "-sham", "fc-wd1e-3"),
    ],
)
def test_complete_run_of_another_arm_is_refused_not_skipped_or_touched(
    sweep, summary, arm_suffix, recorded
):
    arm = f"fc{arm_suffix}"
    run_dir = sweep.output_root / arm / "c8-seed0"
    summary_path = write_summary(run_dir, summary)
    extra = {"ARM_SUFFIX": arm_suffix, "EXTRA_ARGS": "--sham"} if arm_suffix else {}

    dry, calls = sweep(DRY_RUN=1, SEEDS=0, **extra)
    assert dry.returncode == 0, dry.stderr
    assert f"complete run of arm '{recorded}', not '{arm}'" in dry.stderr
    assert table_rows(dry.stdout) == [[arm, "C8", "0", "-", f"would-refuse(arm={recorded})"]]

    real, calls = sweep(SEEDS=0, **extra)
    assert real.returncode == 1
    assert f"complete run of arm '{recorded}', not '{arm}'" in real.stderr
    assert table_rows(real.stdout) == [[arm, "C8", "0", "-", f"arm-mismatch({recorded})"]]
    assert calls == []
    assert summary_path.read_text() == summary
    assert sorted(p.name for p in (sweep.output_root / arm).iterdir()) == ["c8-seed0"]


@pytest.mark.parametrize(
    "summary,arm_suffix",
    [
        ("status: complete\nplacement: fc\narm: fc-sham\n", "-sham"),
        ("status: complete\nplacement: fc\narm: 'fc-sham'\n", "-sham"),
        ("status: complete\nplacement: fc\narm: fc\n", ""),
        ("status: complete\nplacement: fc\n", ""),  # written before arms existed
        ("status: complete\nplacement: fc\narm: null\n", ""),
    ],
)
def test_complete_run_of_the_same_arm_is_skipped(sweep, summary, arm_suffix):
    arm = f"fc{arm_suffix}"
    summary_path = write_summary(sweep.output_root / arm / "c8-seed0", summary)
    extra = {"ARM_SUFFIX": arm_suffix} if arm_suffix else {}
    result, calls = sweep(SEEDS=0, **extra)
    assert result.returncode == 0, result.stderr
    assert f"skip  {arm} C8 seed0" in result.stdout
    assert calls == []
    assert table_rows(result.stdout) == [[arm, "C8", "0", "-", "skipped"]]
    assert summary_path.read_text() == summary


def test_arm_mismatch_does_not_stop_the_rest_of_the_grid(sweep):
    write_summary(sweep.output_root / "fc-sham/c8-seed0", "status: complete\nplacement: fc\n")
    result, calls = sweep(SEEDS="0 1", ARM_SUFFIX="-sham", EXTRA_ARGS="--sham")
    assert result.returncode == 1
    assert [argv[argv.index("--seed") + 1] for argv in calls] == ["1"]
    statuses = [row[-1] for row in table_rows(result.stdout)]
    assert statuses == ["arm-mismatch(fc)", "ok"]


def test_incomplete_run_in_the_arm_dir_is_moved_aside_not_deleted(sweep):
    run_dir = sweep.output_root / "fc-sham/c8-seed0"
    write_summary(run_dir, "status: incomplete\nplacement: fc\narm: fc-sham\n")
    result, calls = sweep(SEEDS=0, ARM_SUFFIX="-sham", EXTRA_ARGS="--sham")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    asides = [p for p in (sweep.output_root / "fc-sham").iterdir() if ".partial-" in p.name]
    assert len(asides) == 1
    assert "status: incomplete" in (asides[0] / "reports/grow_summary.yaml").read_text()
    assert table_rows(result.stdout)[0][-1] == "moved-aside+ok"
