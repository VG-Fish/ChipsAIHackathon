from pathlib import Path

import pytest
import torch
from torch import nn

from kws.hardware.neurosim.backend import export_inputs
from kws.hardware.neurosim.config import load_hardware_config


def _hardware_config(tmp_path: Path):
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
trace: {split: test, samples: 2, seed: 0}
graph: {grouped_conv_mode: patched, reject_unsupported_ops: true}
runtime: {timeout_seconds: 30}
""",
        encoding="utf-8",
    )
    return load_hardware_config(path)


def test_export_inputs_writes_ir_weights_traces_and_manifest(tmp_path):
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.Flatten(), nn.Linear(2 * 4 * 4, 3)).eval()
    samples = [
        ("sample_000", 10, torch.ones(1, 1, 4, 4), 2),
        ("sample_001", 11, torch.zeros(1, 1, 4, 4), 1),
    ]

    result = export_inputs(model, samples, _hardware_config(tmp_path), tmp_path / "export")

    assert result.ir.total_macs > 0
    assert (tmp_path / "export/ir/model_ir.json").exists()
    assert (tmp_path / "export/ir/network.csv").exists()
    assert len(list((tmp_path / "export/weights").glob("layer_*.csv"))) == 2
    assert (tmp_path / "export/traces/sample_000/activity.json").exists()
    assert result.manifest["trace"]["sample_ids"] == ["sample_000", "sample_001"]
    assert result.manifest["mac_cross_check"]["matches"] is True
    assert result.manifest["hardware_config_sha256"]


def test_export_inputs_supports_a_root_weighted_module(tmp_path):
    result = export_inputs(
        nn.Linear(2, 2).eval(),
        [("sample_000", 0, torch.ones(1, 2), 0)],
        _hardware_config(tmp_path),
        tmp_path / "export-root",
    )

    assert result.ir.layers[0].module_name == "__root__"
    assert (tmp_path / "export-root/weights/layer_000.csv").exists()


def test_export_rejects_nonfinite_later_real_sample(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        export_inputs(
            nn.Linear(2, 2).eval(),
            [
                ("sample_000", 0, torch.ones(1, 2), 0),
                ("sample_001", 1, torch.tensor([[float("nan"), 0.0]]), 0),
            ],
            _hardware_config(tmp_path),
            tmp_path / "export-nan",
        )
