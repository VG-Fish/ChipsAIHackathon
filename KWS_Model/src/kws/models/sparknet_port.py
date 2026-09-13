"""Port a released SparkNet Lightning checkpoint to this project's format.

    python -m kws.models.sparknet_port --ckpt … --data-config … --out …

The reference checkpoints (`ckpt/kws_C_{4,8,16,32}.ckpt` in
<https://github.com/jsvir/sparknet>, commit e66915e, MIT license) pickle
OmegaConf/NeMo objects this project does not depend on. A restricted
unpickler resolves only the classes needed to read the plain tensor
`state_dict` -- ``torch``, ``torch._utils``, ``torch.storage``,
``collections.OrderedDict``, and plain builtin containers/scalars -- and
stands in an inert stub for everything else (PLAN.md Finding 3).
"""
import argparse
import builtins
import collections
import pickle
import types
from pathlib import Path

import torch
import yaml

from kws.data.dataset import build_label_map
from kws.data.features import build_feature_extractor
from kws.models.sparknet import SparkNet
from kws.utils.artifacts import sha256_path
from kws.utils.checkpointing import atomic_torch_save
from kws.utils.logging import get_logger

logger = get_logger(__name__)

REFERENCE_LABEL_ORDER = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "_unknown_", "_silence_",
]

# Reference tensor suffix (under `fs.encoder.{i}.`) -> this project's TCSBlock
# submodule name.
_BLOCK_TENSOR_MAP = {
    "mconv.0": "depthwise",
    "mconv.1": "pointwise",
    "mconv.2": "bn",
    "res.0.0": "res_conv",
    "res.0.1": "res_bn",
}

_ALLOWED_BUILTIN_NAMES = {
    "dict", "list", "tuple", "set", "frozenset", "int", "float", "str",
    "bytes", "bytearray", "bool", "complex",
}


class _InertStub:
    """Stands in for any pickled class this port does not need.

    The checkpoint's OmegaConf hyperparameters and Lightning callback state
    need those packages installed to unpickle for real. This port reads only
    the tensor ``state_dict``, so every other class is replaced with a
    harmless placeholder that accepts any constructor args and any
    ``__setstate__`` payload.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return self


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in ("torch", "torch._utils", "torch.storage"):
            return getattr(__import__(module, fromlist=["_"]), name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module in ("__builtin__", "builtins"):
            if name == "long":  # Python 2 pickle protocol name for int.
                return int
            if name in _ALLOWED_BUILTIN_NAMES:
                return getattr(builtins, name)
        return _InertStub


# torch.load only requires this to look like a module exposing (Un)Pickler.
_restricted_pickle_module = types.ModuleType("kws_sparknet_restricted_pickle")
_restricted_pickle_module.Unpickler = _RestrictedUnpickler
_restricted_pickle_module.Pickler = pickle.Pickler
_restricted_pickle_module.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def load_lightning_state_dict(ckpt_path: str | Path) -> dict:
    """Load the full tensor ``state_dict`` (encoder + preprocessor buffers)."""
    state = torch.load(
        ckpt_path, map_location="cpu",
        pickle_module=_restricted_pickle_module, weights_only=False,
    )
    return state["state_dict"]


def map_state_dict(source: dict) -> tuple[dict, int, int]:
    """Map reference tensor names onto this project's SparkNet module names.

    Returns ``(mapped_state_dict, channels, gate_channels)``, both inferred
    from tensor shapes. Raises if any non-``preprocessor.*`` source tensor is
    left unmapped.
    """
    mapped: dict[str, torch.Tensor] = {}
    unused: list[str] = []

    for key, value in source.items():
        if key.startswith("preprocessor."):
            continue
        if key.startswith("fs.encoder."):
            index_str, _, tensor_suffix = key[len("fs.encoder."):].partition(".")
            for prefix, dest_module in _BLOCK_TENSOR_MAP.items():
                if tensor_suffix.startswith(prefix + "."):
                    dest_suffix = tensor_suffix[len(prefix) + 1:]
                    mapped[f"blocks.{index_str}.{dest_module}.{dest_suffix}"] = value
                    break
            else:
                unused.append(key)
        elif key.startswith("output_layer.0."):
            mapped[f"gate_conv.{key[len('output_layer.0.'):]}"] = value
        elif key.startswith("output_layer.1."):
            mapped[f"gate_bn.{key[len('output_layer.1.'):]}"] = value
        elif key.startswith("freq_linear_proj."):
            mapped[f"fc.{key[len('freq_linear_proj.'):]}"] = value
        else:
            unused.append(key)

    if unused:
        raise ValueError(f"unused source tensors after mapping: {unused}")

    # `TCSBlock`'s Conv2d weights need the kernel-height axis the reference's
    # Conv1d weights do not have.
    for key in list(mapped):
        if key.endswith(".weight") and mapped[key].dim() == 3:
            mapped[key] = mapped[key].unsqueeze(2)

    channels = mapped["blocks.1.depthwise.weight"].shape[0]
    gate_channels = mapped["gate_conv.weight"].shape[0]
    return mapped, channels, gate_channels


def _check_front_end_matches(extractor, source_state: dict, *, atol: float = 1e-4) -> None:
    checks = [
        ("dct_mat", extractor.mfcc.dct_mat, "preprocessor.featurizer.dct_mat"),
        (
            "mel filterbank",
            extractor.mfcc.MelSpectrogram.mel_scale.fb,
            "preprocessor.featurizer.MelSpectrogram.mel_scale.fb",
        ),
        (
            "window",
            extractor.mfcc.MelSpectrogram.spectrogram.window,
            "preprocessor.featurizer.MelSpectrogram.spectrogram.window",
        ),
    ]
    for label, local, key in checks:
        if key not in source_state:
            raise ValueError(f"checkpoint is missing front-end buffer {key!r}")
        reference = source_state[key]
        diff = (local - reference).abs().max().item()
        if diff >= atol:
            raise ValueError(
                f"front-end mismatch on {label}: max abs diff {diff:.3e} against the "
                f"checkpoint's {key!r} (tolerance {atol:.0e}); the data config's "
                "features block does not reproduce the checkpoint's preprocessor"
            )


def _read_upstream_commit(ckpt_path: Path) -> str:
    commit_path = ckpt_path.parent / "UPSTREAM_COMMIT"
    if not commit_path.exists():
        raise FileNotFoundError(
            f"expected an UPSTREAM_COMMIT file next to {ckpt_path} recording the "
            "jsvir/sparknet commit these checkpoints came from"
        )
    return commit_path.read_text(encoding="utf-8").strip()


def port_checkpoint(ckpt_path: str, data_config_path: str, out_path: str) -> dict:
    ckpt_path_resolved = Path(ckpt_path).resolve()
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    source_state = load_lightning_state_dict(ckpt_path_resolved)
    mapped_state, channels, gate_channels = map_state_dict(source_state)
    n_feat = mapped_state["blocks.0.depthwise.weight"].shape[0]  # depthwise: (F, 1, 1, K) post-unsqueeze

    extractor = build_feature_extractor(data_cfg)
    _check_front_end_matches(extractor, source_state)

    label_map = build_label_map(data_cfg["target_keywords"])
    expected_label_map = {name: index for index, name in enumerate(REFERENCE_LABEL_ORDER)}
    if label_map != expected_label_map:
        raise ValueError(
            f"data config label order {label_map} does not match the reference order "
            f"{expected_label_map}"
        )
    num_classes = len(REFERENCE_LABEL_ORDER)
    num_keywords = len(data_cfg["target_keywords"])

    clip_len = int(data_cfg["dataset"]["sample_rate"] * data_cfg["dataset"]["clip_seconds"])
    sample_features = extractor(torch.zeros(1, clip_len))
    if sample_features.shape[-2] != n_feat:
        raise ValueError(
            f"data config produces {sample_features.shape[-2]} feature bins, "
            f"checkpoint expects {n_feat}"
        )
    frames = sample_features.shape[-1]

    model = SparkNet(n_feat=n_feat, num_classes=num_classes, channels=channels,
                      gate_channels=gate_channels)
    model.load_state_dict(mapped_state, strict=True)
    model.eval()

    checkpoint = {
        "model_family": "sparknet",
        "model_cfg": {
            "name": f"sparknet_c{channels}",
            "channels": channels,
            "gate_channels": gate_channels,
        },
        "input_shape": [n_feat, frames],
        "num_classes": num_classes,
        "num_keywords": num_keywords,
        "label_map": label_map,
        "model_state_dict": model.state_dict(),
        "source": {
            "path": str(ckpt_path_resolved),
            "upstream_commit": _read_upstream_commit(ckpt_path_resolved),
            "sha256": sha256_path(ckpt_path_resolved),
        },
    }
    atomic_torch_save(out_path, checkpoint)
    logger.info(
        "Ported %s (C=%d, G=%d) -> %s (params=%d)",
        ckpt_path_resolved, channels, gate_channels, out_path,
        sum(p.numel() for p in model.parameters()),
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Released SparkNet Lightning checkpoint")
    parser.add_argument("--data-config", required=True,
                        help="Data config whose features block the checkpoint's front end must match")
    parser.add_argument("--out", required=True, help="Path to write the ported project-style checkpoint")
    args = parser.parse_args()
    port_checkpoint(args.ckpt, args.data_config, args.out)


if __name__ == "__main__":
    main()
