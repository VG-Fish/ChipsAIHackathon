from kws.hardware.neurosim.ir import HardwareLayer, HardwareModelIR
from kws.hardware.neurosim.network_csv import network_csv_text, write_network_csv


def _ir():
    return HardwareModelIR(
        schema_version=1,
        model_name="toy",
        input_shape=(1, 2, 4, 4),
        output_shape=(1, 3),
        layers=(
            HardwareLayer(
                execution_index=0,
                execution_name="conv#0",
                module_name="conv",
                module_type="Conv2d",
                op_type="conv2d",
                input_shape=(1, 2, 4, 4),
                output_shape=(1, 3, 2, 2),
                in_channels=2,
                out_channels=3,
                kernel_h=3,
                kernel_w=3,
                stride_h=2,
                stride_w=2,
                padding_h=0,
                padding_w=0,
                dilation_h=1,
                dilation_w=1,
                groups=1,
                weight_shape=(3, 2, 3, 3),
                has_bias=False,
                parameter_count=54,
                macs=108,
                weight_key="conv.weight",
            ),
        ),
        non_cim_ops=(),
        total_parameters=54,
        total_macs=108,
    )


def test_network_csv_contains_extended_geometry_and_groups(tmp_path):
    text = network_csv_text(_ir())

    assert "groups" in text.splitlines()[0]
    assert "3,2,2,2,2,1,1" in text
    path = tmp_path / "network.csv"
    write_network_csv(_ir(), path)
    assert path.read_text(encoding="utf-8") == text
