from pathlib import Path

import pytest

from kws.hardware.neurosim.ir import HardwareLayer, HardwareModelIR
from kws.hardware.neurosim.parser import parse_neurosim_output
from kws.hardware.neurosim.v21 import build_v21_command, legacy_network_rows
from kws.hardware.neurosim.build import prepare_isolated_build
from kws.hardware.neurosim.config import load_hardware_config


def test_parser_accepts_official_v21_output_labels():
    result = parse_neurosim_output(
        """
        ChipArea : 1234.5um^2
        Chip readLatency of Forward (per epoch) is: 12.5ns
        Chip readDynamicEnergy of Forward (per epoch) is: 67.5pJ
        Chip leakage Power is: 0.25uW
        """
    )

    assert result.forward_latency_s == pytest.approx(12.5e-9)
    assert result.forward_dynamic_energy_j == pytest.approx(67.5e-12)
    assert result.leakage_power_w == pytest.approx(0.25e-6)
    assert result.chip_area_m2 == pytest.approx(1234.5e-12)


def test_v21_network_rows_preserve_grouped_layer_geometry():
    ir = HardwareModelIR(
        schema_version=1,
        model_name="test",
        input_shape=(1, 4, 1, 8),
        output_shape=(1, 4, 1, 8),
        layers=[
            HardwareLayer(
                execution_index=0,
                execution_name="depthwise#0",
                module_name="depthwise",
                module_type="Conv2d",
                op_type="conv2d",
                input_shape=(1, 4, 1, 8),
                output_shape=(1, 4, 1, 8),
                in_channels=4,
                out_channels=4,
                kernel_h=1,
                kernel_w=3,
                stride_h=1,
                stride_w=1,
                padding_h=0,
                padding_w=1,
                dilation_h=1,
                dilation_w=1,
                groups=4,
                weight_shape=(4, 1, 1, 3),
                has_bias=False,
                parameter_count=12,
                macs=96,
                weight_key="depthwise.weight",
            )
        ],
        non_cim_ops=(),
        total_parameters=12,
        total_macs=96,
    )

    rows = legacy_network_rows(ir)

    assert rows == [[1, 10, 1, 1, 3, 1, 0, 1]] * 4


def test_v21_command_uses_legacy_per_layer_argument_order(tmp_path: Path):
    command = build_v21_command(
        executable=tmp_path / "main",
        network_csv=tmp_path / "NetWork.csv",
        weight_bits=8,
        activation_bits=8,
        layer_files=[
            (
                tmp_path / "weight.csv",
                tmp_path / "weightOld.csv",
                tmp_path / "input.csv",
            )
        ],
    )

    assert command == [
        str(tmp_path / "main"),
        "0",
        str(tmp_path / "NetWork.csv"),
        "8",
        "8",
        str(tmp_path / "weight.csv"),
        str(tmp_path / "weightOld.csv"),
        str(tmp_path / "input.csv"),
        "1",
    ]


def test_prepare_build_recognizes_nested_official_layout_and_patches_copy(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    neuro_dir = source_root / "Training_pytorch" / "NeuroSIM"
    neuro_dir.mkdir(parents=True)
    (neuro_dir / "makefile").write_text("CXX := g++\n")
    (neuro_dir / "Param.cpp").write_text(
        "memcelltype = 3;\ntrainingEstimation = true;\n"
    )

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
trace: {split: test, samples: 1, seed: 0}
graph: {grouped_conv_mode: patched, reject_unsupported_ops: true}
runtime: {timeout_seconds: 30}
""",
        encoding="utf-8",
    )
    artifact = prepare_isolated_build(source_root, load_hardware_config(config_path), tmp_path / "cache")

    assert artifact.build_root == artifact.source_copy / "Training_pytorch" / "NeuroSIM"
    assert "memcelltype = 2;" in (artifact.build_root / "Param.cpp").read_text()
    assert "trainingEstimation = false;" in (artifact.build_root / "Param.cpp").read_text()
    assert "memcelltype = 3;" in (neuro_dir / "Param.cpp").read_text()
