import numpy as np

from kws.hardware.neurosim.quantization import symmetric_quantize


def test_symmetric_quantization_is_deterministic_and_non_mutating():
    values = np.array([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=np.float32)
    original = values.copy()

    first = symmetric_quantize(values, bits=3)
    second = symmetric_quantize(values, bits=3)

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.scale == second.scale
    assert first.values.min() >= -3
    assert first.values.max() <= 3


def test_zero_tensor_uses_unit_scale():
    result = symmetric_quantize(np.zeros((2, 2)), bits=8)

    assert result.scale == 1.0
    assert np.all(result.values == 0)
