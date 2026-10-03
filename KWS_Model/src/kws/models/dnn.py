"""Hello Edge fully connected KWS model, with optional dendritic branches.

The input ``(B, 1, n_mfcc, T)`` is average-pooled over time by ``time_pool``
(``ceil_mode``, 101 -> 26 at the default 4), flattened time-major (index
``t * n_mfcc + m``, so a window of ``n_mfcc`` inputs is one whole frame), then
passed through ``Linear -> ReLU`` per entry of ``hidden`` and a final
``Linear(num_classes)`` named ``fc``.

With ``dendrites`` = d > 0 every Linear (hidden and head) gets an additive
:class:`DendriticPointwise` branch, ``y = linear(x) + dendrites(x)``, whose
dendrites read restricted local windows of ``dendrite_fan_in`` inputs
(Chavlis & Poirazi style), costing ``out * d * (f + 2)`` params per layer. The
fan-in is clamped to the layer's input width (``head_dendrite_fan_in``
overrides it for the head). On the flattened input the windows are frame
aligned and do not wrap (see :func:`kws.models.ei_conv1d.local_window_index`).

Ablation knobs: ``dendrite_layers`` ("all", "hidden" or "head") limits which
Linears get a branch; ``dendrite_activation: identity`` makes the branch linear
(a pure reparameterisation of the Linear); ``dendrite_layout: random`` gives
each dendrite ``fan_in`` inputs drawn without replacement from the whole input
(fixed generator seed, so the layout is identical at load time) instead of a
local window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kws.models.ei_conv1d import InputNorm, dendritic_branch


class DendriticLinear(nn.Module):
    """``Linear`` plus an optional additive dendritic branch."""

    def __init__(self, in_features: int, out_features: int, dendrites: int = 0,
                 fan_in: int = 0, align: int = 1, activation: str = "relu",
                 layout: str = "local"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dendrites = None
        if dendrites > 0:
            fan_in = min(fan_in, in_features)
            if fan_in < 1:
                raise ValueError("dendrite_fan_in must be positive when dendrites > 0")
            if activation not in ("relu", "identity"):
                raise ValueError(f"dendrite_activation must be relu or identity, got {activation!r}")
            if layout not in ("local", "random"):
                raise ValueError(f"dendrite_layout must be local or random, got {layout!r}")
            self.dendrites = dendritic_branch(out_features, dendrites, fan_in, in_features, align)
            self.dendrites.activation = activation
            if layout == "random":
                g = torch.Generator().manual_seed(0)
                self.dendrites.index = torch.cat([
                    torch.randperm(in_features, generator=g)[:fan_in]
                    for _ in range(out_features * dendrites)])

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.dendrites is not None:
            y = y + self.dendrites(x[:, :, None, None]).flatten(1)
        return y


class DNN(nn.Module):
    def __init__(self, n_mfcc: int, time_steps: int, num_classes: int,
                 hidden: list[int] | tuple[int, ...] = (16,), time_pool: int = 4,
                 dropout: float = 0.0, dendrites: int = 0, dendrite_fan_in: int = 0,
                 head_dendrite_fan_in: int | None = None, input_norm: str = "cmvn",
                 dendrite_layers: str = "all", dendrite_activation: str = "relu",
                 dendrite_layout: str = "local"):
        super().__init__()
        if dendrite_layers not in ("all", "hidden", "head"):
            raise ValueError(f"dendrite_layers must be all, hidden or head, got {dendrite_layers!r}")
        hidden_d = dendrites if dendrite_layers in ("all", "hidden") else 0
        head_d = dendrites if dendrite_layers in ("all", "head") else 0
        branch = dict(activation=dendrite_activation, layout=dendrite_layout)
        self.input_norm = InputNorm(input_norm, n_mfcc)
        self.input_shape = (n_mfcc, time_steps)
        self.time_pool = time_pool
        frames = -(-time_steps // time_pool)
        width = n_mfcc * frames
        layers: list[nn.Module] = []
        align = n_mfcc
        for h in hidden:
            layers += [DendriticLinear(width, h, hidden_d, dendrite_fan_in, align, **branch),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            width, align = h, 1
        self.hidden = nn.Sequential(*layers)
        head_fan_in = dendrite_fan_in if head_dendrite_fan_in is None else head_dendrite_fan_in
        self.fc = DendriticLinear(width, num_classes, head_d, head_fan_in, align, **branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.input_norm(x).unsqueeze(1)
        if self.time_pool > 1:
            x = F.avg_pool2d(x, (1, self.time_pool), ceil_mode=True)
        x = x.squeeze(1).transpose(1, 2).flatten(1)  # time-major
        return self.fc(self.hidden(x))


def build_dnn(model_cfg: dict, input_shape: tuple[int, int], num_classes: int) -> DNN:
    return DNN(
        n_mfcc=input_shape[0],
        time_steps=input_shape[1],
        num_classes=num_classes,
        hidden=list(model_cfg.get("hidden", [16])),
        time_pool=model_cfg.get("time_pool", 4),
        dropout=model_cfg.get("dropout", 0.0),
        dendrites=model_cfg.get("dendrites", 0),
        dendrite_fan_in=model_cfg.get("dendrite_fan_in", 0),
        head_dendrite_fan_in=model_cfg.get("head_dendrite_fan_in"),
        input_norm=model_cfg.get("input_norm", "cmvn"),
        dendrite_layers=model_cfg.get("dendrite_layers", "all"),
        dendrite_activation=model_cfg.get("dendrite_activation", "relu"),
        dendrite_layout=model_cfg.get("dendrite_layout", "local"),
    )
