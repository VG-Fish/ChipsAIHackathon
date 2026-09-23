"""Deterministic, non-mutating symmetric quantization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    values: np.ndarray
    bits: int
    scale: float
    original_min: float
    original_max: float
    quantized_min: int
    quantized_max: int

    @property
    def sha256(self) -> str:
        return sha256(self.values.tobytes()).hexdigest()


def _array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def symmetric_quantize(
    values: Any,
    *,
    bits: int,
    scale: float | None = None,
) -> QuantizedTensor:
    """Quantize using the guide's per-tensor symmetric integer convention."""

    if not 1 <= bits <= 16:
        raise ValueError("bits must be between 1 and 16")
    array = _array(values)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot quantize non-finite values")
    qmax = 2 ** (bits - 1) - 1
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if scale is None:
        scale = 1.0 if maximum == 0.0 else maximum / qmax
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    quantized = np.rint(array / scale).clip(-qmax, qmax).astype(np.int64)
    return QuantizedTensor(
        values=quantized,
        bits=bits,
        scale=float(scale),
        original_min=float(np.min(array)) if array.size else 0.0,
        original_max=float(np.max(array)) if array.size else 0.0,
        quantized_min=int(quantized.min()) if quantized.size else 0,
        quantized_max=int(quantized.max()) if quantized.size else 0,
    )
