import numpy as np
import torch
from torch import nn

from kws.hardware.neurosim.traces import (
    activity_factor,
    encode_fixed_point_bits,
)


def test_fixed_point_encoding_is_unsigned_binary_of_clipped_quantized_values():
    values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    bits = encode_fixed_point_bits(values, bits=3, scale=1.0)

    assert bits.shape == (3, 3)
    np.testing.assert_array_equal(bits, [[1, 1, 1], [0, 0, 0], [0, 0, 1]])


def test_activity_uses_generated_bits():
    bits = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)

    assert activity_factor(bits) == 0.5


def test_activation_capture_preserves_shared_execution_identity():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2)

        def forward(self, x):
            return self.projection(self.projection(x))

    arrays, provenance = __import__(
        "kws.hardware.neurosim.traces", fromlist=["collect_activation_arrays"]
    ).collect_activation_arrays(
        Shared().eval(),
        [("sample_000", 4, torch.ones(1, 2), 1)],
        ("projection#0", "projection#1"),
    )

    assert set(arrays) == {"projection#0", "projection#1"}
    assert provenance == [("sample_000", 4, 1)]
