from kws.hardware.neurosim.report import (
    HardwareResult,
    LayerHardwareResult,
    normalize_result,
    write_hardware_markdown,
)


def test_report_normalization_uses_si_units_and_two_ops_per_mac():
    result = normalize_result(
        backend="neurosim_v21_reram",
        model_macs=1_000_000,
        forward_latency_s=2e-3,
        forward_dynamic_energy_j=4e-6,
        leakage_power_w=1e-3,
        chip_area_m2=2e-6,
        per_layer=(
            LayerHardwareResult(
                execution_index=0,
                execution_name="conv#0",
                area_m2=2e-6,
                latency_s=2e-3,
                dynamic_energy_j=4e-6,
                mapped_rows=128,
                mapped_cols=128,
                num_subarrays=1,
            ),
        ),
    )

    assert isinstance(result, HardwareResult)
    assert result.fps == 500.0
    assert result.total_energy_per_inference_j == 6e-6
    assert result.tops == 1e-3
    assert result.validity == "EXACT_WITHIN_BACKEND_MODEL"


def test_markdown_report_contains_disclaimer_and_warnings(tmp_path):
    result = normalize_result(
        backend="neurosim_v21_reram",
        model_macs=1,
        forward_latency_s=1.0,
        forward_dynamic_energy_j=1.0,
        leakage_power_w=None,
        chip_area_m2=None,
        warnings=("example warning",),
    )
    path = tmp_path / "hardware.md"
    write_hardware_markdown(result, path)

    text = path.read_text(encoding="utf-8")
    assert "not a physical-device measurement" in text
    assert "example warning" in text
