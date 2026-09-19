"""Rebuild a clean PerforatedAI export as a plain PyTorch model, without PAI.

``export_final_pai_model`` writes ``final_clean_pai.pt``: a safetensors state
of the model after PAI's inference cleanup.  Loading it through PAI needs the
licensed library; this module rebuilds the same function from the tensors
alone, so a grow run can be re-evaluated and diagnosed anywhere.

Every dendrite module in the file is a prefix with ``<prefix>.layer_array.<i>.*``
tensors.  ``layer_array`` holds deep copies of the original module: the
dendrites first and the original ("main") module last.  The prefix also holds
``skip_weights.<k>`` (shape ``(k + 1, C)``) and ``view_tuple`` (the broadcast
shape of one skip row, ``-1`` on the channel axis: ``[1, -1]`` for a Linear
on ``(B, C)``, ``[1, -1, 1, 1]`` for a Conv2d on ``(B, C, H, W)``).  The forward
is PAI's clean forward (``clean_perforatedai.PAIModulePyThread.forward``,
processors ``None`` as they are for Linear and Conv2d)::

    outs = [layer(x) for layer in layer_array]
    for o in range(len(outs)):
        cur = outs[o]
        for i in range(o):
            cur = cur + skip_weights[o - 1][i].view(view_tuple) * outs[i]
        if o < len(outs) - 1:
            cur = forward_function(cur)
        outs[o] = cur
    return cur

With one dendrite that is ``main(x) + skip[0] * f(dendrite(x))``.

Loading is strict: every tensor in the file must land in the rebuilt model,
except the PAI bookkeeping listed in :data:`PAI_BOOKKEEPING`, and every tensor
the rebuilt model needs must be in the file with the same shape.  PAI's
cleanup drops the skip weights of a one-dendrite module; the repo's exporter
restores them, and a file without them is refused rather than rebuilt as a
model whose dendrite silently does nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from kws.models.registry import build_model

LAYER_ARRAY = "layer_array"
SKIP_WEIGHTS = "skip_weights"
VIEW_TUPLE = "view_tuple"

# Tensors a clean PAI state may carry that play no part in the forward pass.
# From the PAI source (perforatedai 3.2.8):
#   node_index      buffer of clean_perforatedai.PAIModulePyThread: the output
#                   axis that holds channels (checked against view_tuple here).
#   num_cycles      buffer of PAIModulePyThread: how many n/p mode switches the
#                   module went through; PAI's own loader derives the dendrite
#                   count from it, the tensors here make it redundant.
#   tracker_string  top-level buffer utils_perforatedai registers on the
#                   network: the serialized PAI tracker (training state).
#   module_id       network_perforatedai.load_pai_model_from_dict drops every
#                   key with a ``module_id`` path component before loading.
# node_index and num_cycles are ignored only directly under a dendrite prefix;
# tracker_string and module_id anywhere.  A key the rebuilt model itself needs
# is always loaded, whatever its name.
PAI_MODULE_BOOKKEEPING = ("node_index", "num_cycles")
PAI_ANYWHERE_BOOKKEEPING = ("tracker_string", "module_id")
PAI_BOOKKEEPING = PAI_MODULE_BOOKKEEPING + PAI_ANYWHERE_BOOKKEEPING


class RebuiltDendriteModule(nn.Module):
    """A clean PAI dendrite module rebuilt from plain modules.

    ``layer_array[:-1]`` are the dendrites, ``layer_array[-1]`` the original
    module.  ``skip_weights[k]`` has shape ``(k + 1, C)``.  Same attributes as
    PAI's clean wrapper, so :mod:`kws.optimize.grow_diagnostics` treats both
    alike.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        channels: int,
        view_tuple: Sequence[int],
        forward_function: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("a dendrite module needs at least its original module")
        self.layer_array = nn.ModuleList(layers)
        self.processor_array = [None] * len(self.layer_array)
        self.skip_weights = nn.ParameterList(
            nn.Parameter(torch.zeros(k + 1, channels), requires_grad=False)
            for k in range(len(self.layer_array) - 1)
        )
        self.register_buffer(VIEW_TUPLE, torch.tensor([int(v) for v in view_tuple], dtype=torch.long))
        self.forward_function = forward_function

    @property
    def n_dendrites(self) -> int:
        return len(self.layer_array) - 1

    def forward(self, *args, **kwargs):
        # Dendrites first, the original module last, as PAI evaluates them.
        # Nothing below is in place: forward hooks may hold these tensors.
        outs = [layer(*args, **kwargs) for layer in self.layer_array]
        view = [int(v) for v in self.view_tuple.tolist()]
        current = outs[-1]
        for out_index in range(len(outs)):
            current = outs[out_index]
            if len(outs) > 1:
                if current.dim() != len(view):
                    raise RuntimeError(
                        f"view_tuple {view} does not match a {current.dim()}-d module output"
                    )
                for in_index in range(out_index):
                    skip = self.skip_weights[out_index - 1][in_index, :].view(view)
                    current = current + skip * outs[in_index]
                if out_index < len(outs) - 1:
                    current = self.forward_function(current)
            outs[out_index] = current
        return current

    def extra_repr(self) -> str:
        name = getattr(self.forward_function, "__name__", repr(self.forward_function))
        return f"n_dendrites={self.n_dendrites}, forward_function={name}"


def load_clean_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """``(tensors, metadata)`` of a ``final_clean_pai.pt`` safetensors file, on CPU."""
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def dendrite_prefixes(clean_state: Mapping[str, torch.Tensor]) -> list[str]:
    """Qualified names of the dendrite modules in a clean state, sorted."""
    marker = f".{LAYER_ARRAY}."
    prefixes = set()
    for key in clean_state:
        if key.startswith(f"{LAYER_ARRAY}."):
            raise ValueError(f"the model root cannot be a dendrite module: {key!r}")
        if marker in key:
            prefixes.add(key.split(marker, 1)[0])
    return sorted(prefixes)


def _view_tuple(prefix: str, clean_state: Mapping[str, torch.Tensor]) -> list[int]:
    key = f"{prefix}.{VIEW_TUPLE}"
    if key not in clean_state:
        raise ValueError(f"missing key {key!r}: every clean dendrite module has a view_tuple")
    tensor = clean_state[key]
    if tensor.dim() != 1 or tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        raise ValueError(f"{key} must be a 1-d integer tensor; got {tuple(tensor.shape)} {tensor.dtype}")
    view = [int(v) for v in tensor.tolist()]
    if view.count(-1) != 1 or any(v not in (-1, 1) for v in view):
        raise ValueError(f"{key} = {view}: expected one -1 (the channel axis) and 1 elsewhere")
    node_key = f"{prefix}.node_index"
    if node_key in clean_state:
        node_index = clean_state[node_key]
        if node_index.numel() != 1 or int(node_index.flatten()[0]) != view.index(-1):
            raise ValueError(
                f"{node_key} = {node_index.flatten().tolist()} disagrees with the channel "
                f"axis {view.index(-1)} of {VIEW_TUPLE} {view}"
            )
    return view


def _layer_count(prefix: str, clean_state: Mapping[str, torch.Tensor]) -> int:
    marker = f"{prefix}.{LAYER_ARRAY}."
    indices = set()
    for key in clean_state:
        if key.startswith(marker):
            index = key[len(marker):].split(".", 1)[0]
            if not index.isdigit():
                raise ValueError(f"unexpected key {key!r}: layer_array entries are numbered")
            indices.add(int(index))
    count = max(indices) + 1
    if indices != set(range(count)):
        raise ValueError(f"{prefix}.{LAYER_ARRAY} has gaps: entries {sorted(indices)}")
    return count


def _channels(prefix: str, module: nn.Module, clean_state: Mapping[str, torch.Tensor]) -> int:
    for attribute in ("out_channels", "out_features"):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
    first = clean_state.get(f"{prefix}.{SKIP_WEIGHTS}.0")
    if first is not None and first.dim() == 2:
        return int(first.shape[1])
    raise ValueError(
        f"cannot tell the channel count of {prefix} ({type(module).__name__}: no "
        "out_channels / out_features)"
    )


def _replace_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def _is_bookkeeping(key: str, prefixes: set[str]) -> bool:
    parts = key.split(".")
    if parts[-1] == "tracker_string" or "module_id" in parts:
        return True
    parent, _, leaf = key.rpartition(".")
    return leaf in PAI_MODULE_BOOKKEEPING and parent in prefixes


def _summarize(keys: Sequence[str], limit: int = 8) -> str:
    keys = sorted(keys)
    shown = ", ".join(keys[:limit])
    return shown + (f", ... ({len(keys)} total)" if len(keys) > limit else "")


def rebuild_clean_model(
    model_cfg: dict,
    input_shape: Sequence[int],
    num_classes: int,
    clean_state: Mapping[str, torch.Tensor],
    forward_function: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
) -> nn.Module:
    """The model a clean PAI state describes, as plain PyTorch, eval mode, CPU.

    Builds ``build_model(model_cfg, input_shape, num_classes)``, replaces every
    dendrite prefix of ``clean_state`` with a :class:`RebuiltDendriteModule`
    and loads all tensors strictly.  ``forward_function`` is PAI's
    ``pai_forward_function`` (``perforatedai.forward_function`` in the train
    config, tanh for the grow runs).

    Raises ``ValueError`` on an unexpected key, a missing key, a shape
    mismatch, or dendrite bookkeeping that contradicts itself.
    """
    model = build_model(model_cfg, (int(input_shape[0]), int(input_shape[1])), int(num_classes))
    state = {key: value.detach().cpu() for key, value in clean_state.items()}
    prefixes = dendrite_prefixes(state)
    for prefix in prefixes:
        try:
            original = model.get_submodule(prefix)
        except AttributeError as error:
            raise ValueError(f"clean state has a dendrite module {prefix!r} the model lacks") from error
        view = _view_tuple(prefix, state)
        count = _layer_count(prefix, state)
        channels = _channels(prefix, original, state)
        layers = [copy.deepcopy(original) for _ in range(count)]
        _replace_submodule(
            model, prefix, RebuiltDendriteModule(layers, channels, view, forward_function)
        )

    expected = model.state_dict()
    prefix_set = set(prefixes)
    unexpected = [
        key for key in state if key not in expected and not _is_bookkeeping(key, prefix_set)
    ]
    missing = [key for key in expected if key not in state]
    mismatched = [
        f"{key}: file {tuple(state[key].shape)} vs model {tuple(expected[key].shape)}"
        for key in expected
        if key in state and tuple(state[key].shape) != tuple(expected[key].shape)
    ]
    problems = []
    if unexpected:
        problems.append(f"unexpected keys: {_summarize(unexpected)}")
    if missing:
        hint = ""
        if any(f".{SKIP_WEIGHTS}." in key for key in missing):
            hint = (
                " (PAI's cleanup drops a one-dendrite module's skip weights; "
                "export_final_pai_model restores them, so this file was not written by it)"
            )
        problems.append(f"missing keys: {_summarize(missing)}{hint}")
    if mismatched:
        problems.append(f"shape mismatches: {_summarize(mismatched)}")
    if problems:
        raise ValueError("clean state does not match the rebuilt model: " + "; ".join(problems))

    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return model.cpu().eval()
