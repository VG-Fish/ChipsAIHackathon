import sys

from kws.hardware.neurosim.runner import run_simulator


def test_runner_isolates_sample_stdout_and_stderr(tmp_path):
    result = run_simulator(
        [sys.executable, "-c", "print('Forward Latency: 1 ns')"],
        cwd=tmp_path,
        sample_id="sample_000",
        output_root=tmp_path / "raw",
        timeout_seconds=10,
    )

    assert result.return_code == 0
    assert "Forward Latency" in result.stdout
    assert (tmp_path / "raw/sample_000/stdout.txt").exists()


def test_runner_timeout_preserves_text_diagnostics(tmp_path):
    result = run_simulator(
        [sys.executable, "-c", "import time; print('partial'); time.sleep(1)"],
        cwd=tmp_path,
        sample_id="sample_000",
        output_root=tmp_path / "raw",
        timeout_seconds=0.01,
    )

    assert result.timed_out is True
    assert "timed out" in (tmp_path / "raw/sample_000/stderr.txt").read_text()
