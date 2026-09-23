from kws.hardware.neurosim.parser import parse_neurosim_output


def test_parser_identifies_forward_metrics_and_preserves_raw_fields():
    output = """
Forward Latency: 2.5 ns
Forward Dynamic Energy: 4.0 pJ
Leakage Power: 3.0 uW
Chip Area: 1.2 mm^2
"""

    result = parse_neurosim_output(output)

    assert result.forward_latency_s == 2.5e-9
    assert result.forward_dynamic_energy_j == 4.0e-12
    assert result.leakage_power_w == 3.0e-6
    assert result.chip_area_m2 == 1.2e-6
    assert result.raw_fields["Forward Latency"] == "2.5 ns"


def test_parser_rejects_missing_forward_latency():
    try:
        parse_neurosim_output("Total Energy: 4 pJ")
    except ValueError as exc:
        assert "forward latency" in str(exc).lower()
    else:
        raise AssertionError("missing forward latency was accepted")
