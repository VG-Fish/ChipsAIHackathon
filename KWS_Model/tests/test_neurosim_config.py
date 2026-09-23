from pathlib import Path

import pytest

from kws.hardware.neurosim.config import (
    HardwareConfigError,
    load_hardware_config,
)


def _config(**overrides):
    config = {
        "schema_version": 1,
        "backend": "neurosim_v21_reram",
        "source": {"root_env": "NEUROSIM_V21_ROOT"},
        "simulation": {
            "inference_only": True,
            "mapping": "novel",
            "pipeline": False,
        },
        "precision": {
            "weight_bits": 8,
            "activation_bits": 8,
            "cell_bits": 2,
            "adc_bits": 6,
        },
        "array": {"rows": 128, "cols": 128, "columns_per_adc": 8},
        "technology": {"node_nm": 32, "temperature_k": 300, "clock_hz": 1_000_000_000},
        "memory": {
            "access_type": "1t1r",
            "resistance_on_ohm": 240_000,
            "resistance_off_ohm": 24_000_000,
            "read_voltage_v": 0.5,
            "read_pulse_width_s": 1e-8,
            "write_voltage_v": 4.0,
            "write_pulse_width_s": 5e-8,
            "access_resistance_ohm": 15_000,
        },
        "trace": {"split": "test", "samples": 16, "seed": 0},
        "graph": {"grouped_conv_mode": "patched", "reject_unsupported_ops": True},
        "runtime": {"timeout_seconds": 300},
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


def _write_config(tmp_path: Path, config: dict) -> Path:
    import yaml

    path = tmp_path / "hardware.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_valid_reram_config_is_loaded_into_immutable_sections(tmp_path):
    loaded = load_hardware_config(_write_config(tmp_path, _config()))

    assert loaded.backend == "neurosim_v21_reram"
    assert loaded.precision.weight_bits == 8
    assert loaded.array.columns_per_adc == 8
    with pytest.raises((AttributeError, TypeError)):
        loaded.precision.weight_bits = 4


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("root", "backend", "other", "backend"),
        ("simulation", "inference_only", False, "inference_only"),
        ("precision", "weight_bits", 0, "weight_bits"),
        ("precision", "activation_bits", 17, "activation_bits"),
        ("precision", "cell_bits", 9, "cell_bits"),
        ("precision", "adc_bits", 0, "adc_bits"),
        ("array", "rows", 100, "array.rows"),
        ("array", "cols", 100, "array.cols"),
        ("array", "columns_per_adc", 129, "columns_per_adc"),
        ("memory", "resistance_off_ohm", 100, "resistance_off_ohm"),
        ("memory", "read_pulse_width_s", 0, "read_pulse_width_s"),
        ("memory", "write_pulse_width_s", -1, "write_pulse_width_s"),
    ],
)
def test_invalid_hardware_setting_is_rejected(
    tmp_path, section, key, value, message
):
    config = _config()
    if section == "root":
        config[key] = value
    else:
        config[section][key] = value

    with pytest.raises(HardwareConfigError, match=message):
        load_hardware_config(_write_config(tmp_path, config))


def test_non_divisible_adc_columns_are_rejected(tmp_path):
    config = _config(array={"cols": 128, "columns_per_adc": 7})

    with pytest.raises(HardwareConfigError, match="divisible"):
        load_hardware_config(_write_config(tmp_path, config))


def test_missing_configuration_file_is_actionable(tmp_path):
    with pytest.raises(HardwareConfigError, match="does not exist"):
        load_hardware_config(tmp_path / "missing.yaml")
