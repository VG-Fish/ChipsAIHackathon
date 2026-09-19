"""Does a grown dendrite add capacity, or stand in for capacity the base had?

Measured on a *clean* PerforatedAI model (the object ``export_final_pai_model``
returns, or an equivalent rebuilt from ``final_clean_pai.pt``).  A clean
dendrite module holds ``layer_array`` (dendrite copies first, the original
module last), ``skip_weights`` (entry k has shape ``(k + 1, C)``) and
``view_tuple`` (the broadcast shape of one skip row, with ``-1`` on the
channel axis).  Its forward is::

    out = main(x) + skip[0] * f(dendrite_0(x))        # one dendrite

The dendrite's contribution is measured exactly as ``out - main(x)``, so
nothing here depends on which forward function ``f`` PAI used.  ``main(x)`` and
``dendrite_0(x)`` are re-evaluated on the module's own input from a forward
hook on the dendrite module itself: PAI's clean wrapper calls the original
module through ``.forward()``, which never fires hooks registered on it.

Per module:

``linear_r2_vs_preactivation``
    R^2 of a per-channel affine fit of the contribution on the dendrite's own
    pre-activation ``dendrite_0(x)``.  The pre-activation is linear in the
    module input, so a value near 1 means the dendrite is (almost) a second
    copy of a linear layer, which the original module can already express.
    It is a lower bound on the fit a full linear map of the input would get.
``tanh_saturated_fraction`` / ``tanh_linear_fraction``
    Share of pre-activations with ``|z| > 2`` / ``|z| < 0.5``.
``corr_with_base_output``
    Pearson correlation between the contribution and the original module's
    output, pooled over every element.  Strongly positive means the dendrite
    mostly scales up what the module already outputs rather than correcting it.
``dendrite_to_base_std_ratio``
    ``std(contribution) / std(main output)``.

Model-level ``val_acc_dendrite_on`` / ``val_acc_dendrite_off`` evaluate the
same model with every skip weight as trained and then zeroed.  A large drop
with no net gain over the no-dendrite control means the base co-adapted and
handed work to the dendrite.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

SATURATED_ABS = 2.0
LINEAR_ABS = 0.5


def dendrite_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Clean PAI dendrite modules of ``model`` that hold at least one dendrite.

    A wrapper with only its original module (``len(layer_array) == 1``) has
    nothing to measure and no skip weights to report.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "layer_array")
        and hasattr(module, "skip_weights")
        and len(module.layer_array) > 1
    }


def _channel_axis(module: nn.Module) -> int:
    view = [int(v) for v in torch.as_tensor(module.view_tuple).flatten().tolist()]
    if view.count(-1) != 1:
        raise ValueError(f"cannot find the channel axis in view_tuple {view}")
    return view.index(-1)


def _as_rows(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """(..., C, ...) -> (N, C) in float64 on the CPU, channel axis last.

    On the CPU because MPS has no float64.  Two steps on purpose: a fused
    ``.to(device="cpu", dtype=torch.float64)`` of an MPS tensor returns zeros
    (torch 2.14).
    """
    rows = tensor.detach().movedim(axis, -1).reshape(-1, tensor.shape[axis])
    return rows.cpu().double()


class _ModuleStats:
    """Streaming sums for one dendrite module; no activations are kept."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.axis = _channel_axis(module)
        self.n_dendrites = len(module.layer_array) - 1
        processors = getattr(module, "processor_array", None) or []
        self.supported = self.n_dendrites == 1 and all(p is None for p in processors)
        self.pre: torch.Tensor | None = None
        self.main: torch.Tensor | None = None
        self.sums: dict[str, torch.Tensor] = {}
        self.count = 0
        self.saturated = 0
        self.linear = 0

    def _add(self, key: str, value: torch.Tensor) -> None:
        self.sums[key] = self.sums[key] + value if key in self.sums else value

    def observe(self, args: tuple, kwargs: dict, output: torch.Tensor) -> None:
        """Forward hook of the dendrite module: re-run both branches, then update."""
        layers = self.module.layer_array
        self.main = layers[-1](*args, **kwargs)
        if self.supported:
            self.pre = layers[0](*args, **kwargs)
        self.update(output)

    def update(self, output: torch.Tensor) -> None:
        if self.main is None:
            return
        main = _as_rows(self.main, self.axis)
        contribution = _as_rows(output, self.axis) - main
        # Pooled sums for the correlation and std ratio.
        self._add("y", contribution.sum())
        self._add("yy", (contribution * contribution).sum())
        self._add("m", main.sum())
        self._add("mm", (main * main).sum())
        self._add("ym", (contribution * main).sum())
        if self.supported and self.pre is not None:
            z = _as_rows(self.pre, self.axis)
            # Per-channel sums for the affine fit contribution ~ a*z + b.
            self._add("cn", torch.full_like(z[0], float(z.shape[0])))
            self._add("cz", z.sum(0))
            self._add("czz", (z * z).sum(0))
            self._add("cy", contribution.sum(0))
            self._add("cyy", (contribution * contribution).sum(0))
            self._add("czy", (z * contribution).sum(0))
            self.saturated += int((z.abs() > SATURATED_ABS).sum())
            self.linear += int((z.abs() < LINEAR_ABS).sum())
        self.count += main.numel()
        self.pre = self.main = None

    def result(self) -> dict:
        n = float(self.count)
        s = self.sums
        out: dict = {
            "n_dendrites": self.n_dendrites,
            "skip_weight_mean_abs": float(
                torch.cat([w.detach().flatten() for w in self.module.skip_weights]).abs().mean()
            ),
        }
        if n == 0:
            return out
        var_y = s["yy"] / n - (s["y"] / n) ** 2
        var_m = s["mm"] / n - (s["m"] / n) ** 2
        cov = s["ym"] / n - (s["y"] / n) * (s["m"] / n)
        out["corr_with_base_output"] = float(cov / (var_y * var_m).sqrt().clamp_min(1e-300))
        out["dendrite_to_base_std_ratio"] = float(var_y.clamp_min(0).sqrt() / var_m.clamp_min(1e-300).sqrt())
        if not self.supported:
            out["note"] = "linearity metrics need exactly one dendrite and no PAI processors"
            return out
        cn = s["cn"]
        szz = s["czz"] - s["cz"] ** 2 / cn
        syy = s["cyy"] - s["cy"] ** 2 / cn
        szy = s["czy"] - s["cz"] * s["cy"] / cn
        explained = torch.where(szz > 0, szy * szy / szz.clamp_min(1e-300), torch.zeros_like(szz))
        total = syy.sum()
        out["linear_r2_vs_preactivation"] = (
            float(explained.sum() / total) if float(total) > 0 else 1.0
        )
        out["tanh_saturated_fraction"] = self.saturated / n
        out["tanh_linear_fraction"] = self.linear / n
        return out


def _accuracy(model: nn.Module, loader: Iterable, device: torch.device, on_output=None) -> tuple[float, int]:
    correct = total = 0
    for features, labels in loader:
        logits = model(features.to(device))
        labels = labels.to(device)
        correct += int((logits.argmax(1) == labels).sum())
        total += int(labels.numel())
        if on_output is not None:
            on_output()
    return (correct / total if total else float("nan")), total


@torch.no_grad()
def dendrite_diagnostics(model: nn.Module, loader: Iterable, device: torch.device) -> dict:
    """Diagnostics for every clean dendrite module of ``model`` over ``loader``.

    ``loader`` yields ``(features, labels)``.  Accuracies are fractions in
    [0, 1], like the grow summary's ``val_acc``.  The model is left in eval
    mode with its skip weights exactly as they were.

    Returns ``{"val_acc_dendrite_on", "val_acc_dendrite_off", "n_samples",
    "modules": {name: {...}}}``; see the module docstring for the keys.
    """
    modules = dendrite_modules(model)
    if not modules:
        raise ValueError("model has no clean PAI dendrite modules (layer_array + skip_weights)")
    model.eval()
    stats = {name: _ModuleStats(module) for name, module in modules.items()}
    handles = []
    for st in stats.values():
        # One hook on the dendrite module, not on its branches: PAI's clean
        # wrapper runs the original module via ``.forward()``, so a hook on
        # ``layer_array[-1]`` never fires there and every metric but the skip
        # weights would silently go missing.
        handles.append(st.module.register_forward_hook(
            lambda _m, args, kwargs, out, st=st: st.observe(args, kwargs, out),
            with_kwargs=True))
    try:
        acc_on, n_samples = _accuracy(model, loader, device)
    finally:
        for handle in handles:
            handle.remove()

    saved = [
        (weight, weight.detach().clone())
        for module in modules.values()
        for weight in module.skip_weights
    ]
    try:
        for weight, _ in saved:
            weight.zero_()
        acc_off, _ = _accuracy(model, loader, device)
    finally:
        for weight, original in saved:
            weight.copy_(original)

    return {
        "val_acc_dendrite_on": acc_on,
        "val_acc_dendrite_off": acc_off,
        "n_samples": n_samples,
        "modules": {name: st.result() for name, st in stats.items()},
    }
