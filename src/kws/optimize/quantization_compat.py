"""Compatibility boundary for PyTorch's eager int8 graph conversion.

PyTorch 2.10+ deprecates the public ``torch.ao.quantization`` eager entry
points and directs users to TorchAO.  TorchAO 0.18 does not yet provide the
static QNNPACK Conv2d conversion used by this project (its QAT transformer is
currently limited to Linear and Embedding), so the deployment graph still
needs PyTorch's eager implementation underneath this small boundary.

The deprecated decorators are deliberately bypassed when PyTorch exposes the
wrapped implementation.  This keeps the existing Conv2d/dendritic graph
working while avoiding the deprecation warning at every QAT step.  The narrow
packed-weight warning is also suppressed only around the legacy conversion;
it is emitted by PyTorch's implementation rather than by project code.
"""
from __future__ import annotations

import copy
import warnings

import torch.ao.quantization as _tq
from torch.ao.quantization.fuse_modules import fuse_modules_qat as _fuse_modules_qat
from torch.ao.quantization.quantize import (
    _add_observer_,
    _convert,
    _remove_qconfig,
    get_default_qat_module_mappings,
    propagate_qconfig_,
)


# These are modules/stubs, not deprecated eager conversion entry points.  Keep
# them behind this boundary so the rest of the optimizer has one quantization
# import surface.
DeQuantStub = _tq.DeQuantStub
QuantStub = _tq.QuantStub


def get_default_qat_qconfig(backend: str):
    """Return PyTorch's backend-specific eager QAT configuration."""
    return _tq.get_default_qat_qconfig(backend)


def fuse_modules_qat(parent, names, *, inplace: bool = True):
    """Fuse a QAT module pair without entering a deprecated API."""
    return _fuse_modules_qat(parent, names, inplace=inplace)


def prepare_qat(model, *, mapping=None, inplace: bool = False):
    """Prepare an eager QAT graph without emitting PyTorch's deprecation notice."""
    if not model.training:
        raise AssertionError("prepare_qat only works on models in training mode")
    if mapping is None:
        mapping = get_default_qat_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)

    # This is the implementation sequence used by PyTorch's deprecated
    # prepare_qat entry point, expressed in terms of its non-decorated helpers.
    # It keeps the exact eager QAT module mapping without invoking deprecated
    # public functions from inside the compatibility layer.
    propagate_qconfig_(model, qconfig_dict=None)
    _convert(model, mapping=mapping, inplace=True)
    _add_observer_(
        model,
        non_leaf_module_list=set(mapping.values()),
    )
    return model


def convert(
    model,
    *,
    mapping=None,
    inplace: bool = False,
    remove_qconfig: bool = True,
    is_reference: bool = False,
    convert_custom_config_dict=None,
    use_precomputed_fake_quant: bool = False,
):
    """Convert an eager QAT graph without emitting legacy API warnings.

    PyTorch 2.14 emits a ``UserWarning`` while constructing packed QNNPACK
    weights through the old eager backend.  It is scoped to this call because
    TorchAO has no equivalent Conv2d conversion yet; unrelated user warnings
    (for example, an uncalibrated observer) remain visible.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r"torch\.ao\.quantization is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=(
                r"torch\.quantize_per_tensor, torch\.quantize_per_channel.*"
                r"deprecated.*"
            ),
        )
        if not inplace:
            model = copy.deepcopy(model)
        _convert(
            model,
            mapping=mapping,
            inplace=True,
            is_reference=is_reference,
            convert_custom_config_dict=convert_custom_config_dict,
            use_precomputed_fake_quant=use_precomputed_fake_quant,
        )
        if remove_qconfig:
            _remove_qconfig(model)
        return model
