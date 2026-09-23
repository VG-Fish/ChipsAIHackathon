import json
import subprocess

import pytest

from kws.hardware.neurosim.build import prepare_isolated_build
from kws.hardware.neurosim.build import BuildError, compile_neurosim
from kws.hardware.neurosim.config import load_hardware_config


def _config(tmp_path):
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
schema_version: 1
backend: neurosim_v21_reram
source: {root_env: NEUROSIM_V21_ROOT}
simulation: {inference_only: true, mapping: novel, pipeline: false}
precision: {weight_bits: 8, activation_bits: 8, cell_bits: 2, adc_bits: 6}
array: {rows: 128, cols: 128, columns_per_adc: 8}
technology: {node_nm: 32, temperature_k: 300, clock_hz: 1000000000}
memory: {access_type: 1t1r, resistance_on_ohm: 240000, resistance_off_ohm: 24000000, read_voltage_v: 0.5, read_pulse_width_s: 1e-8, write_voltage_v: 4.0, write_pulse_width_s: 5e-8, access_resistance_ohm: 15000}
trace: {split: test, samples: 1, seed: 0}
graph: {grouped_conv_mode: patched, reject_unsupported_ops: true}
runtime: {timeout_seconds: 30}
""",
        encoding="utf-8",
    )
    return load_hardware_config(path)


def test_build_copies_source_and_writes_manifest_without_mutating_original(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (source / "Param.cpp").write_text("original\n", encoding="utf-8")
    artifact = prepare_isolated_build(source, _config(tmp_path), tmp_path / "cache")

    assert artifact.source_copy != source
    assert (artifact.source_copy / "Param.cpp").read_text(encoding="utf-8") == "original\n"
    assert (artifact.source_copy / "GeneratedUserConfig.inc").exists()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "neurosim_v21_reram"


def test_compile_timeout_preserves_diagnostics(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    artifact = prepare_isolated_build(source, _config(tmp_path), tmp_path / "cache")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output="partial", stderr="error")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(BuildError):
        compile_neurosim(artifact, timeout_seconds=1)
    assert (artifact.build_directory / "compile.stdout.txt").exists()
    assert (artifact.build_directory / "compile.stderr.txt").exists()
