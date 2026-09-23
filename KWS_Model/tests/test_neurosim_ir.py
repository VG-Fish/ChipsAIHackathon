import numpy as np
import pytest

from kws.hardware.neurosim.ir import (
    HardwareLayer,
    HardwareModelIR,
    NonCIMOperation,
    conv_weight_matrix,
    linear_weight_matrix,
)


def test_standard_conv_matrix_is_transposed_and_preserves_values():
    weight = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2)

    matrix = conv_weight_matrix(weight, groups=1, in_channels=2)

    assert matrix.shape == (8, 2)
    np.testing.assert_array_equal(matrix, weight.reshape(2, -1).T)


def test_grouped_conv_matrix_keeps_disconnected_blocks():
    weight = np.arange(4 * 2, dtype=np.float32).reshape(4, 2, 1, 1)

    matrix = conv_weight_matrix(weight, groups=2, in_channels=4)

    assert matrix.shape == (4, 4)
    np.testing.assert_array_equal(
        matrix,
        np.array(
            [[0, 2, 0, 0], [1, 3, 0, 0], [0, 0, 4, 6], [0, 0, 5, 7]],
            dtype=np.float32,
        ),
    )


def test_invalid_grouped_conv_shape_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        conv_weight_matrix(np.zeros((3, 2, 1, 1)), groups=2, in_channels=4)


def test_linear_matrix_is_transposed():
    weight = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)

    matrix = linear_weight_matrix(weight)

    assert matrix.shape == (5, 3)
    np.testing.assert_array_equal(matrix, weight.T)


def test_ir_round_trip_json(tmp_path):
    layer = HardwareLayer(
        execution_index=0,
        execution_name="conv#0",
        module_name="conv",
        module_type="Conv2d",
        op_type="conv2d",
        input_shape=(1, 4, 8, 8),
        output_shape=(1, 4, 8, 8),
        in_channels=4,
        out_channels=4,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        padding_h=1,
        padding_w=1,
        dilation_h=1,
        dilation_w=1,
        groups=4,
        weight_shape=(4, 1, 3, 3),
        has_bias=True,
        parameter_count=40,
        macs=2304,
        weight_key="conv.weight",
    )
    ir = HardwareModelIR(
        schema_version=1,
        model_name="toy",
        input_shape=(1, 4, 8, 8),
        output_shape=(1, 4, 8, 8),
        layers=(layer,),
        non_cim_ops=(NonCIMOperation("relu", "ReLU", (1, 4, 8, 8), 1),),
        total_parameters=40,
        total_macs=2304,
    )

    path = tmp_path / "model_ir.json"
    ir.write_json(path)
    loaded = HardwareModelIR.read_json(path)

    assert loaded == ir
