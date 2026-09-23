import sys
import json
import pytest

from kws.hardware.neurosim.backend import run_exported_simulation
from kws.hardware.neurosim.config import load_hardware_config
from kws.hardware.neurosim.backend import export_inputs
import torch
from torch import nn


def test_run_exported_simulation_parses_each_sample_and_writes_report(tmp_path):
    config_path = tmp_path / "hardware.yaml"
    config_path.write_text(
        """
schema_version: 1
backend: neurosim_v21_reram
source: {root_env: NEUROSIM_V21_ROOT}
simulation: {inference_only: true, mapping: novel, pipeline: false}
precision: {weight_bits: 8, activation_bits: 8, cell_bits: 2, adc_bits: 6}
array: {rows: 128, cols: 128, columns_per_adc: 8}
technology: {node_nm: 32, temperature_k: 300, clock_hz: 1000000000}
memory: {access_type: 1t1r, resistance_on_ohm: 240000, resistance_off_ohm: 24000000, read_voltage_v: 0.5, read_pulse_width_s: 1e-8, write_voltage_v: 4.0, write_pulse_width_s: 5e-8, access_resistance_ohm: 15000}
trace: {split: test, samples: 2, seed: 0}
graph: {grouped_conv_mode: patched, reject_unsupported_ops: true}
runtime: {timeout_seconds: 30}
""",
        encoding="utf-8",
    )
    exported = export_inputs(
        nn.Sequential(nn.Linear(2, 2)).eval(),
        [("sample_000", 0, torch.ones(1, 2), 0), ("sample_001", 1, torch.ones(1, 2), 0)],
        load_hardware_config(config_path),
        tmp_path / "export",
    )

    result = run_exported_simulation(
        exported,
        [sys.executable, "-c", "print('Forward Latency: 2 ns'); print('Forward Dynamic Energy: 4 pJ')"],
        timeout_seconds=10,
    )

    assert result.forward_latency_s == 2e-9
    assert result.forward_dynamic_energy_j == 4e-12
    assert (tmp_path / "export/reports/hardware.json").exists()
    stats = json.loads((tmp_path / "export/reports/sample_statistics.json").read_text())
    assert stats["fps"]["mean"] == pytest.approx(500_000_000.0)
