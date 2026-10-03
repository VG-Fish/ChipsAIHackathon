"""Robustness of dendritic checkpoints against their equal-parameter controls.

Evaluation only: no training, no selection.  Every checkpoint is one that a
finished run already selected on validation (``best.pt``); this script only
measures it on the held-out TEST split under perturbations, and compares each
dendritic arm with its matched controls paired by training seed.

Perturbations, all deterministic and identical across arms:

* ``snr``   -- a crop of a ``_background_noise_`` recording is mixed into each
  test *waveform* at 20/10/5/0 dB SNR before MFCC extraction.  The noise file
  and crop offset are drawn from ``random.Random(f"robustness-bgnoise:{seed}:{i}")``
  for clip ``i``, so every model -- and every SNR -- sees the same crop; only
  its gain changes with the SNR.  SNR is relative to the clip's own mean power,
  exactly as ``kws.data.augment.mix_background_noise`` defines it.
* ``wn``    -- multiplicative analog weight variation: the ``weight`` of every
  ``nn.Linear`` / ``nn.Conv1d`` / ``nn.Conv2d`` is multiplied by
  ``1 + sigma * eps`` with ``eps ~ N(0, 1)``.  Biases, BatchNorm and DTNet time
  constants are left alone.  Draw ``d`` uses ``torch.Generator`` seeded
  ``WEIGHT_NOISE_BASE_SEED + d``, walked over the modules in registration
  order, and the same ``eps`` is reused for every ``sigma`` (common random
  numbers).
* ``q``     -- symmetric fake quantization of the same weights to 8/6/4 bits,
  per output channel (``per_channel``) and per tensor (``per_tensor``).
  BatchNorm is not folded; for the per-output-channel mode that is exact,
  because every BN here scales the output channels of the conv it follows and
  symmetric per-channel quantization commutes with a per-channel scale.
* ``onnx_int8`` (``--onnx``) -- the project's own PTQ path
  (``kws.optimize.quantize_ptq``: ONNX export, ``quantize_dynamic`` with QInt8
  weights), scored batch-1 through onnxruntime like ``quantize_ptq`` does.
* ``ece``   -- expected calibration error, 15 equal-width confidence bins, on
  clean test.

Data, test-set construction and model loading are ``scripts/report_test_accuracy.py``'s
own code paths (same data config, same fixed eval seed, same loader, batch 128,
same device), so the clean accuracy here must reproduce ``eval_b*.json``.

Usage::

    uv run --env-file .env python scripts/robustness_eval.py evaluate \\
        --output-dir OUT --group msd --group dtnet
    uv run --env-file .env python scripts/robustness_eval.py analyze --output-dir OUT
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_test_accuracy import (  # noqa: E402
    DEFAULT_DATA_CONFIG,
    load_target_model,
    targets_from_checkpoints,
)

from kws.data.dataset import build_datasets  # noqa: E402
from kws.data.splits import TEST  # noqa: E402
from kws.models.registry import checkpoint_input_shape  # noqa: E402
from kws.utils.device import get_device  # noqa: E402
from kws.utils.seed import set_seed  # noqa: E402

CKPT_SUFFIX = Path("models/checkpoints/paper_replication/best.pt")
SNRS_DB = (20.0, 10.0, 5.0, 0.0)
SIGMAS = (0.05, 0.1, 0.2)
WEIGHT_NOISE_DRAWS = 5
WEIGHT_NOISE_BASE_SEED = 20260930
BITS = (8, 6, 4)
ECE_BINS = 15
BATCH_SIZE = 128  # report_test_accuracy's loader batch size
WEIGHT_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d)

# Each comparison is (dendritic arm, control arm, control kind). "plain" is a
# non-dendritic control (dense / wider / deeper / point neuron); "ablation" is
# a dendritic-structure ablation (linear branches, random layout, no tau...).
GROUPS: dict[str, dict[str, Any]] = {
    "msd": {
        "root": "outputs/students-msd",
        "seeds": [0, 1, 2],
        "comparisons": [
            ("sparknet_msd_a13_relu", "sparknet_msd_a13_dense", "plain"),
            ("sparknet_msd_a13_relu", "sparknet_msd_a13_lin", "ablation"),
            ("sparknet_msd_b_relu", "sparknet_msd_b_dense", "plain"),
            ("sparknet_msd_b_relu", "sparknet_msd_b_lin", "ablation"),
            ("sparknet_msd_a_relu", "sparknet_msd_a_dense", "plain"),
            ("sparknet_msd_a_relu", "sparknet_msd_a_lin", "ablation"),
            ("sparknet_msd_a_relu", "sparknet_msd_a_1scale", "ablation"),
        ],
    },
    "dtnet": {
        "root": "outputs/students-dtnet",
        "seeds": [0, 1, 2],
        "comparisons": [
            ("dtnet_a_het", "dtnet_a_point", "plain"),
            ("dtnet_a_het", "dtnet_a_point3", "plain"),
            ("dtnet_a_het", "dtnet_a_notau", "ablation"),
            ("dtnet_a_het", "dtnet_a_rnd", "ablation"),
            ("dtnet_a_het", "dtnet_a_lin", "ablation"),
            ("dtnet_b_het", "dtnet_b_point", "plain"),
        ],
    },
    "dnn": {
        "root": "outputs/students-dnn",
        "seeds": [0, 1, 2, 3, 4],
        "comparisons": [
            ("dnn_h16_d2f32", "dnn_h18", "plain"),
            ("dnn_h16_d4f32", "dnn_h20", "plain"),
            ("dnn_h16_d4f32", "dnn_h24", "plain"),
            ("dnn_h16_d4f32", "dnn_h16x112", "plain"),
            ("dnn_h16_d4f32", "dnn_h16_d4f32lin", "ablation"),
            ("dnn_h16_d4f32", "dnn_h16_d4f32rnd", "ablation"),
            ("dnn_h16x59_hd4", "dnn_h16x112", "plain"),
            ("dnn_h16x59_hd4", "dnn_h16x59x32", "plain"),
            ("dnn_h32_d2f32", "dnn_h35", "plain"),
        ],
    },
    "scratch": {
        # Context only: plain SparkNet trained from scratch, no pairing.
        "root": "outputs/sparknet-dendritic-study-v2/scratch",
        "seeds": [0, 1, 2, 3, 4],
        "arms": ["c16g16", "c9g8"],
        "comparisons": [],
    },
}


def group_arms(group: dict[str, Any]) -> list[str]:
    arms = list(group.get("arms", []))
    for dend, ctrl, _ in group["comparisons"]:
        for arm in (dend, ctrl):
            if arm not in arms:
                arms.append(arm)
    return arms


def checkpoint_path(group: dict[str, Any], arm: str, seed: int) -> Path:
    return Path(group["root"]) / f"{arm}-seed{seed}" / CKPT_SUFFIX


# ---------------------------------------------------------------- perturbations


def noise_crop(bank: list[torch.Tensor], clip_len: int, rng: random.Random) -> tuple[int, int, torch.Tensor]:
    """One ``clip_len`` crop of a background recording, drawn from ``rng``."""
    index = rng.randrange(len(bank))
    noise = bank[index]
    start = rng.randint(0, max(noise.shape[-1] - clip_len, 0))
    crop = noise[:, start:start + clip_len]
    if crop.shape[-1] < clip_len:
        crop = torch.nn.functional.pad(crop, (0, clip_len - crop.shape[-1]))
    return index, start, crop


def mix_at_snr(waveform: torch.Tensor, crop: torch.Tensor, snr_db: float) -> torch.Tensor:
    """``kws.data.augment.mix_background_noise``'s SNR definition, with a fixed crop."""
    signal_power = waveform.pow(2).mean()
    noise_power = crop.pow(2).mean().clamp_min(1e-10)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    return waveform + crop * torch.sqrt(target_noise_power / noise_power)


def fake_quantize(weight: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    """Symmetric round-to-nearest fake quantization, dim 0 = output channel."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        amax = weight.detach().abs().flatten(1).amax(dim=1)
        amax = amax.view(-1, *([1] * (weight.dim() - 1)))
    else:
        amax = weight.detach().abs().amax()
    scale = amax.clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def weight_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """The perturbable synaptic weights, in registration order."""
    return [
        (name, module.weight)
        for name, module in model.named_modules()
        if isinstance(module, WEIGHT_MODULES) and module.weight is not None
    ]


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = ECE_BINS) -> float:
    confidence, prediction = probs.max(dim=1)
    correct = (prediction == labels).double()
    confidence = confidence.double()
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    ece = 0.0
    total = labels.numel()
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        count = int(in_bin.sum())
        if count:
            ece += abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean())) * count / total
    return ece


# ------------------------------------------------------------------ test data


def build_test_features(data_config: Path, eval_seed: int, noise_seed: int, cache: Path) -> dict[str, Any]:
    """Clean and noisy TEST features, cached; clean is ``dataset[i]`` itself."""
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        if (
            meta["data_config"] == str(data_config)
            and meta["eval_seed"] == eval_seed
            and meta["noise_seed"] == noise_seed
            and tuple(meta["snrs_db"]) == SNRS_DB
        ):
            print(f"loaded cached test features from {cache}")
            return payload
        raise ValueError(f"{cache} was built with different settings: {meta}")

    data_cfg = yaml.safe_load(Path(data_config).read_text())
    # Exactly report_test_accuracy.evaluate: seed, then build the TEST split.
    set_seed(eval_seed)
    datasets, label_map = build_datasets(data_cfg, augment=False, seed=eval_seed, splits={TEST})
    dataset = datasets[TEST]
    bank = dataset.silence_sampler.noise_waveforms
    noise_files = [p.name for p in sorted((Path(data_cfg["dataset"]["root"]) / data_cfg["silence"]["background_noise_dir"]).glob("*.wav"))]
    clip_len = dataset.clip_len

    clean, labels = [], []
    noisy: dict[float, list[torch.Tensor]] = {snr: [] for snr in SNRS_DB}
    crops_used = []
    realized_snr_err = 0.0
    parity_err = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for i in range(len(dataset)):
            features, label = dataset[i]
            clean.append(features)
            labels.append(label)
            waveform = dataset._load_waveform(dataset.entries[i])
            if i % 500 == 0:  # the waveform path reproduces the dataset's own features
                parity_err = max(parity_err, float((dataset.feature_extractor(waveform) - features).abs().max()))
            rng = random.Random(f"robustness-bgnoise:{noise_seed}:{i}")
            file_index, start, crop = noise_crop(bank, clip_len, rng)
            crops_used.append((file_index, start))
            signal_power = float(waveform.pow(2).mean())
            for snr in SNRS_DB:
                mixed = mix_at_snr(waveform, crop, snr)
                if signal_power > 1e-8:
                    added = float((mixed - waveform).pow(2).mean())
                    realized = 10 * math.log10(signal_power / added)
                    realized_snr_err = max(realized_snr_err, abs(realized - snr))
                noisy[snr].append(dataset.feature_extractor(mixed))
            if (i + 1) % 1000 == 0:
                print(f"  features {i + 1}/{len(dataset)} ({time.monotonic() - started:.0f}s)")

    file_counts = [0] * len(bank)
    for file_index, _ in crops_used:
        file_counts[file_index] += 1
    payload = {
        "labels": torch.tensor(labels, dtype=torch.long),
        "clean": torch.stack(clean),
        "noisy": {snr: torch.stack(noisy[snr]) for snr in SNRS_DB},
        "label_names": [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])],
        "meta": {
            "data_config": str(data_config),
            "eval_seed": eval_seed,
            "noise_seed": noise_seed,
            "snrs_db": list(SNRS_DB),
            "n": len(dataset),
            "noise_files": noise_files,
            "noise_file_counts": file_counts,
            "max_abs_realized_snr_error_db": realized_snr_err,
            "waveform_path_feature_parity_max_abs": parity_err,
            "n_silence_clips": sum(
                1 for e in dataset.entries if e.rel_path is None
            ),
        },
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    print(f"wrote {cache}  meta={json.dumps(payload['meta'])}")
    return payload


# ----------------------------------------------------------------- evaluation


@torch.no_grad()
def logits_of(model: nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = []
    for start in range(0, features.shape[0], BATCH_SIZE):
        out.append(model(features[start:start + BATCH_SIZE].to(device)).float().cpu())
    return torch.cat(out)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).double().mean())


def onnx_int8_accuracy(model: nn.Module, input_shape: tuple[int, int], features: torch.Tensor,
                       labels: torch.Tensor, workdir: Path) -> dict[str, Any]:
    """``kws.optimize.quantize_ptq``'s path on a live module: export, dynamic QInt8, batch-1 ORT."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from kws.export.to_onnx import export_module_to_onnx

    workdir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = workdir / "fp32.onnx", workdir / "int8.onnx"
    import onnx

    parity = export_module_to_onnx(copy.deepcopy(model).cpu(), input_shape, str(fp32_path), seed=0)
    # quantize_ptq's call as-is fails on this torch/onnxruntime pair: the
    # dynamo exporter writes intermediate value_info that the quantizer's
    # shape inference then contradicts ("Inferred shape and existing shape
    # differ"). Dropping value_info (pure shape annotations) fixes it.
    graph = onnx.load(str(fp32_path))
    graph.graph.ClearField("value_info")
    onnx.save(graph, str(fp32_path))
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(int8_path), options, providers=["CPUExecutionProvider"])
    batch = features.numpy()
    predictions = np.empty(batch.shape[0], dtype=np.int64)
    for index in range(batch.shape[0]):
        logits = session.run(None, {"features": batch[index:index + 1]})[0]
        predictions[index] = int(np.argmax(logits, axis=1)[0])
    ops = {}
    for node in onnx.load(str(int8_path)).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {
        "accuracy": float((torch.from_numpy(predictions) == labels).double().mean()),
        "fp32_parity_max_abs_diff": parity,
        "int8_ops": ops,
    }


def evaluate_checkpoint(path: Path, data: dict[str, Any], device: torch.device, eval_seed: int,
                        run_onnx: bool, workdir: Path) -> dict[str, Any]:
    (target,) = targets_from_checkpoints([path])
    set_seed(eval_seed)
    model, checkpoint = load_target_model(target, device)
    model.eval()
    labels = data["labels"]
    clean_x = data["clean"]
    if tuple(checkpoint_input_shape(checkpoint)) != tuple(clean_x.shape[-2:]):
        raise ValueError(f"{path} expects {checkpoint_input_shape(checkpoint)}, data is {tuple(clean_x.shape[-2:])}")

    record: dict[str, Any] = {
        "checkpoint": str(path),
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    clean_logits = logits_of(model, clean_x, device)
    clean_pred = clean_logits.argmax(dim=1)
    probs = torch.softmax(clean_logits.double(), dim=1)
    record["clean_acc"] = accuracy(clean_logits, labels)
    record["ece_clean"] = expected_calibration_error(probs, labels)
    record["nll_clean"] = float(torch.nn.functional.cross_entropy(clean_logits.double(), labels))
    record["mean_conf_clean"] = float(probs.max(dim=1).values.mean())

    for snr in SNRS_DB:
        record[f"snr{int(snr)}_acc"] = accuracy(logits_of(model, data["noisy"][snr], device), labels)

    weights = weight_tensors(model)
    record["n_weight_tensors"] = len(weights)
    record["n_weights"] = sum(w.numel() for _, w in weights)
    originals = [w.detach().clone() for _, w in weights]
    try:
        for draw in range(WEIGHT_NOISE_DRAWS):
            generator = torch.Generator().manual_seed(WEIGHT_NOISE_BASE_SEED + draw)
            eps = [torch.randn(o.shape, generator=generator, dtype=torch.float32).to(o.device) for o in originals]
            for sigma in SIGMAS:
                with torch.no_grad():
                    for (_, w), o, e in zip(weights, originals, eps):
                        w.copy_(o * (1.0 + sigma * e))
                record.setdefault(f"wn{sigma}_draws", []).append(
                    accuracy(logits_of(model, clean_x, device), labels)
                )
        for sigma in SIGMAS:
            record[f"wn{sigma}_acc"] = statistics.fmean(record[f"wn{sigma}_draws"])
        for per_channel in (True, False):
            mode = "pc" if per_channel else "pt"
            for bits in BITS:
                with torch.no_grad():
                    for (_, w), o in zip(weights, originals):
                        w.copy_(fake_quantize(o, bits, per_channel))
                record[f"q{bits}{mode}_acc"] = accuracy(logits_of(model, clean_x, device), labels)
    finally:
        with torch.no_grad():
            for (_, w), o in zip(weights, originals):
                w.copy_(o)
    # Restoration check: the model must be bit-identical to the one scored clean.
    record["restore_identical"] = bool(torch.equal(logits_of(model, clean_x, device).argmax(dim=1), clean_pred))

    if run_onnx:
        try:
            result = onnx_int8_accuracy(model, tuple(checkpoint_input_shape(checkpoint)), clean_x, labels, workdir)
            record["onnx_int8_acc"] = result["accuracy"]
            record["onnx_fp32_parity"] = result["fp32_parity_max_abs_diff"]
            record["onnx_int8_ops"] = result["int8_ops"]
        except Exception as error:  # report, don't hide: some graphs may not export
            record["onnx_error"] = f"{type(error).__name__}: {error}"
    return record


def load_reference(pattern: str) -> dict[str, float]:
    reference: dict[str, float] = {}
    for name in sorted(glob.glob(pattern)):
        try:
            payload = json.loads(Path(name).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or payload.get("data_config") != DEFAULT_DATA_CONFIG or payload.get("eval_seed") != 0:
            continue
        for evaluation in payload.get("evaluations", []):
            reference.setdefault(str(evaluation["checkpoint"]), float(evaluation["test_accuracy"]))
    return reference


def cmd_evaluate(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_test_features(Path(args.data_config), args.eval_seed, args.noise_seed, out / "test_features.pt")
    if args.features_only:
        return 0
    reference = load_reference(args.reference_glob) if args.reference_glob else {}
    device = get_device()
    print(f"device {device}; {data['labels'].numel()} test clips")

    results_path = out / f"raw_{'_'.join(args.group)}.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["checkpoint"])

    missing = []
    for group_name in args.group:
        group = GROUPS[group_name]
        for arm in group_arms(group):
            for seed in group["seeds"]:
                path = checkpoint_path(group, arm, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                if str(path) in done:
                    continue
                started = time.monotonic()
                record = evaluate_checkpoint(path, data, device, args.eval_seed, args.onnx, out / "onnx" / f"{arm}-seed{seed}")
                record.update(group=group_name, arm=arm, seed=seed)
                ref = reference.get(str(path))
                record["reference_clean_acc"] = ref
                record["clean_matches_reference"] = None if ref is None else abs(ref - record["clean_acc"]) < 1e-4  # 0.01 pts
                with results_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(
                    f"{group_name} {arm} seed{seed}: clean {record['clean_acc']:.4f} (ref {ref})"
                    f" snr0 {record['snr0_acc']:.4f} wn0.2 {record['wn0.2_acc']:.4f}"
                    f" q4pc {record['q4pc_acc']:.4f} ece {record['ece_clean']:.4f}"
                    + (f" int8 {record.get('onnx_int8_acc', float('nan')):.4f}" if args.onnx else "")
                    + f" [{time.monotonic() - started:.1f}s]",
                    flush=True,
                )
    if missing:
        (out / f"missing_{'_'.join(args.group)}.txt").write_text("\n".join(missing) + "\n")
        print(f"missing {len(missing)} checkpoint(s): " + ", ".join(missing))
    return 0


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature minimizing NLL, by a log-spaced grid (robust, no optimizer state)."""
    grid = torch.logspace(math.log10(0.05), math.log10(10.0), 801, dtype=torch.float64)
    logits = logits.double()
    nll = torch.stack([torch.nn.functional.cross_entropy(logits / t, labels) for t in grid])
    return float(grid[int(nll.argmin())])


def cmd_calibrate(args: argparse.Namespace) -> int:
    """ECE after temperature scaling, T fitted on VALIDATION (never on test).

    DNN/DTNet students train with label smoothing and are underconfident, so a
    raw-ECE difference may be nothing a one-scalar post-hoc fix would not
    remove. The VAL split is built on its own (``splits=(VAL,)``) so the TEST
    build -- and therefore the test set -- is untouched.
    """
    from kws.data.splits import VAL

    out = Path(args.output_dir)
    data = build_test_features(Path(DEFAULT_DATA_CONFIG), 0, 0, out / "test_features.pt")
    data_cfg = yaml.safe_load(Path(DEFAULT_DATA_CONFIG).read_text())
    set_seed(0)
    datasets, _ = build_datasets(data_cfg, augment=False, seed=0, splits=(VAL,))
    val = datasets[VAL]
    val_x = torch.stack([val[i][0] for i in range(len(val))])
    val_y = torch.tensor([val.entries[i].label for i in range(len(val))], dtype=torch.long)
    device = get_device()
    results_path = out / "calib.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["checkpoint"] for line in results_path.read_text().splitlines() if line.strip()}
    for group_name, group in GROUPS.items():
        for arm in group_arms(group):
            for seed in group["seeds"]:
                path = checkpoint_path(group, arm, seed)
                if not path.exists() or str(path) in done:
                    continue
                (target,) = targets_from_checkpoints([path])
                set_seed(0)
                model, _ = load_target_model(target, device)
                model.eval()
                temperature = fit_temperature(logits_of(model, val_x, device), val_y)
                test_logits = logits_of(model, data["clean"], device)
                labels = data["labels"]
                record = {
                    "checkpoint": str(path), "arm": arm, "seed": seed, "group": group_name,
                    "n_val": len(val), "temperature": temperature,
                    "ece_clean_ts": expected_calibration_error(torch.softmax(test_logits.double() / temperature, 1), labels),
                    "nll_clean_ts": float(torch.nn.functional.cross_entropy(test_logits.double() / temperature, labels)),
                }
                with results_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"{arm} seed{seed}: T={temperature:.3f} ece_ts={record['ece_clean_ts']:.4f}", flush=True)
    return 0


# ------------------------------------------------------------------- analysis


METRICS = (
    ["clean_acc", "ece_clean", "ece_clean_ts", "nll_clean", "nll_clean_ts", "temperature"]
    + [f"snr{int(s)}_acc" for s in SNRS_DB]
    + [f"wn{s}_acc" for s in SIGMAS]
    + [f"q{b}pc_acc" for b in BITS]
    + [f"q{b}pt_acc" for b in BITS]
    + ["onnx_int8_acc"]
)
PERTURBED = [m for m in METRICS if m not in ("clean_acc", "ece_clean", "ece_clean_ts", "nll_clean", "nll_clean_ts", "temperature")]
FLOAT_WEIGHT_OPS = ("MatMul", "Conv", "Gemm")


def paired(values_d: list[float], values_c: list[float]) -> dict[str, Any]:
    diffs = [d - c for d, c in zip(values_d, values_c)]
    n = len(diffs)
    mean = statistics.fmean(diffs) if n else float("nan")
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t = mean / (sd / math.sqrt(n))
    elif n > 1 and mean != 0:
        t = math.copysign(float("inf"), mean)
    else:
        t = float("nan")
    p = float("nan")
    try:
        from scipy import stats

        if n > 1 and math.isfinite(t):
            p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        pass
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p_two_sided": p, "diffs": diffs}


def cmd_analyze(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for name in sorted(out.glob("raw_*.jsonl")):
        for line in name.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records[(rec["arm"], rec["seed"])] = rec
    calib_path = out / "calib.jsonl"
    if calib_path.exists():
        for line in calib_path.read_text().splitlines():
            if line.strip():
                cal = json.loads(line)
                if (cal["arm"], cal["seed"]) in records:
                    records[(cal["arm"], cal["seed"])].update(
                        {k: cal[k] for k in ("temperature", "ece_clean_ts", "nll_clean_ts")})
    features_meta = torch.load(out / "test_features.pt", map_location="cpu", weights_only=False)["meta"]
    missing = []
    for group in GROUPS.values():
        for arm in group_arms(group):
            for seed in group["seeds"]:
                if (arm, seed) not in records and checkpoint_path(group, arm, seed).exists():
                    missing.append(f"{arm}-seed{seed} (exists, not evaluated)")
                elif not checkpoint_path(group, arm, seed).exists():
                    missing.append(f"{arm}-seed{seed} (no checkpoint)")

    # Sanity checks.
    refs = [r for r in records.values() if r.get("reference_clean_acc") is not None]
    sanity = {
        "n_checkpoints": len(records),
        "n_with_reference": len(refs),
        "n_clean_exact_match": sum(1 for r in refs if r["clean_matches_reference"]),
        "max_abs_clean_diff_vs_reference_pts": max(
            (abs(r["clean_acc"] - r["reference_clean_acc"]) * 100 for r in refs), default=None
        ),
        "all_weights_restored": all(r["restore_identical"] for r in records.values()),
        "max_abs_q8pc_minus_clean_pts": max(abs(r["q8pc_acc"] - r["clean_acc"]) * 100 for r in records.values()),
        "onnx_errors": {f"{r['arm']}-seed{r['seed']}": r["onnx_error"] for r in records.values() if "onnx_error" in r},
        "features": features_meta,
        # Arms whose int8 ORT graph still runs some weight MatMul/Conv/Gemm in
        # float (weights built at runtime, e.g. DTNet's scattered branch/soma
        # matrices), i.e. only partially quantized.
        "onnx_int8_partial_arms": sorted({
            r["arm"] for r in records.values()
            if any((r.get("onnx_int8_ops") or {}).get(op, 0) for op in FLOAT_WEIGHT_OPS)
        }),
    }

    arms: dict[str, dict[str, Any]] = {}
    for (arm, seed), rec in sorted(records.items()):
        entry = arms.setdefault(arm, {"group": rec["group"], "seeds": [], "num_params": rec["num_params"], "per_seed": {}})
        entry["seeds"].append(seed)
        entry["per_seed"][seed] = {m: rec.get(m) for m in METRICS}
    for arm, entry in arms.items():
        summary = {}
        for m in METRICS:
            vals = [v[m] for v in entry["per_seed"].values() if v.get(m) is not None]
            if vals:
                summary[m] = {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None, "n": len(vals)}
        for m in PERTURBED:
            drops = [v["clean_acc"] - v[m] for v in entry["per_seed"].values() if v.get(m) is not None]
            if drops:
                summary[f"drop_{m}"] = {"mean": statistics.fmean(drops), "sd": statistics.stdev(drops) if len(drops) > 1 else None}
        entry["summary"] = summary

    comparisons = []
    flags: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for group_name, group in GROUPS.items():
        for dend, ctrl, kind in group["comparisons"]:
            if dend not in arms or ctrl not in arms:
                continue
            seeds = sorted(set(arms[dend]["per_seed"]) & set(arms[ctrl]["per_seed"]))
            comp = {
                "group": group_name, "dendritic": dend, "control": ctrl, "control_kind": kind,
                "seeds": seeds, "params": [arms[dend]["num_params"], arms[ctrl]["num_params"]], "metrics": {},
            }
            partial = sanity["onnx_int8_partial_arms"]
            comp["int8_partial"] = dend in partial or ctrl in partial
            for m in METRICS:
                d_vals = [arms[dend]["per_seed"][s].get(m) for s in seeds]
                c_vals = [arms[ctrl]["per_seed"][s].get(m) for s in seeds]
                if any(v is None for v in d_vals + c_vals):
                    continue
                stats_abs = paired(d_vals, c_vals)
                block = {"abs": stats_abs}
                if m in PERTURBED:
                    d_drop = [arms[dend]["per_seed"][s]["clean_acc"] - v for s, v in zip(seeds, d_vals)]
                    c_drop = [arms[ctrl]["per_seed"][s]["clean_acc"] - v for s, v in zip(seeds, c_vals)]
                    block["drop"] = paired(d_drop, c_drop)
                    # A dendritic win: higher absolute accuracy under the perturbation, t > 3.
                    if abs(stats_abs["t"]) > 3:
                        (flags if stats_abs["t"] > 0 else losses).append({
                            "group": group_name, "dendritic": dend, "control": ctrl, "kind": kind,
                            "metric": m, "mean_pts": stats_abs["mean"] * 100, "t": stats_abs["t"],
                            "p": stats_abs["p_two_sided"], "n": stats_abs["n"],
                            "caveat": "int8 graph only partially quantized" if m == "onnx_int8_acc" and comp["int8_partial"] else None,
                        })
                comp["metrics"][m] = block
            comparisons.append(comp)

    payload = {
        "description": "Robustness of dendritic KWS checkpoints vs equal-param controls; TEST split, eval seed 0.",
        "settings": {
            "data_config": DEFAULT_DATA_CONFIG, "eval_seed": 0, "snrs_db": list(SNRS_DB), "sigmas": list(SIGMAS),
            "weight_noise_draws": WEIGHT_NOISE_DRAWS, "weight_noise_base_seed": WEIGHT_NOISE_BASE_SEED,
            "bits": list(BITS), "ece_bins": ECE_BINS, "batch_size": BATCH_SIZE,
            "perturbed_weights": "weight of every nn.Linear/nn.Conv1d/nn.Conv2d; biases, BN, DTNet rho untouched",
            "calibration": "ECE 15 equal-width bins on clean TEST; ECE-TS after temperature scaling with T fitted by NLL grid search on a separately built VAL split",
            "quantization": "symmetric RTN fake quant; pc = per output channel (BN folding is a no-op), pt = per tensor (unfolded)",
            "onnx_int8": "kws.optimize.quantize_ptq path: quantize_dynamic QInt8 weights (per-tensor), dynamic uint8 activations, batch-1 ORT",
        },
        "sanity": sanity,
        "missing": missing,
        "arms": arms,
        "comparisons": comparisons,
        "dendritic_wins_t_gt_3": flags,
        "control_wins_t_lt_minus_3": losses,
        "records": list(records.values()),
    }
    (out / "robustness_results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    (out / "robustness_summary.md").write_text(render_markdown(payload))
    print(f"wrote {out / 'robustness_results.json'} and {out / 'robustness_summary.md'}")
    return 0


LABELS = {
    "clean_acc": "clean", "ece_clean": "ECE", "ece_clean_ts": "ECE-TS", "nll_clean": "NLL",
    "nll_clean_ts": "NLL-TS", "temperature": "T",
    **{f"snr{int(s)}_acc": f"SNR{int(s)}" for s in SNRS_DB},
    **{f"wn{s}_acc": f"wσ{s}" for s in SIGMAS},
    **{f"q{b}pc_acc": f"q{b}ch" for b in BITS},
    **{f"q{b}pt_acc": f"q{b}ten" for b in BITS},
    "onnx_int8_acc": "int8ORT",
}


def fmt_t(t: float) -> str:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "nan"
    if math.isinf(t):
        return "+inf" if t > 0 else "-inf"
    return f"{t:+.1f}"


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload["sanity"]
    lines = ["# Dendritic robustness vs equal-parameter controls", ""]
    lines.append("TEST split, eval seed 0, `configs/data/speech_commands_v2_mfcc32_paper.yaml`; checkpoints already selected on validation. "
                 "All numbers are accuracy in percentage points (pts) unless noted; paired differences are dendritic − control by training seed.")
    lines.append("")
    lines.append("## Sanity checks")
    lines.append(f"- {s['n_checkpoints']} checkpoints evaluated; clean accuracy reproduces the eval_b*.json reference on "
                 f"{s['n_clean_exact_match']}/{s['n_with_reference']} (max |diff| {s['max_abs_clean_diff_vs_reference_pts']} pts).")
    f = s["features"]
    lines.append(f"- Waveform path reproduces the dataset's clean features (max |diff| {f['waveform_path_feature_parity_max_abs']:.2e}); "
                 f"realized SNR error ≤ {f['max_abs_realized_snr_error_db']:.2e} dB; noise crops per file {dict(zip(f['noise_files'], f['noise_file_counts']))}.")
    lines.append(f"- Weights restored bit-exactly after every perturbation: {s['all_weights_restored']}; "
                 f"max |q8 per-channel − clean| = {s['max_abs_q8pc_minus_clean_pts']:.2f} pts.")
    if s["onnx_errors"]:
        lines.append(f"- ONNX int8 failures: {s['onnx_errors']}")
    if s["onnx_int8_partial_arms"]:
        lines.append(f"- ONNX int8 graph only partially quantized (float MatMul/Conv/Gemm left) for: {', '.join(s['onnx_int8_partial_arms'])}; "
                     "their int8ORT column is not a like-for-like quantization comparison.")
    if payload["missing"]:
        lines.append(f"- Missing: {payload['missing']}")
    lines.append("")

    cols = ["clean_acc"] + PERTURBED + ["ece_clean", "ece_clean_ts"]
    lines.append("## Arm means (pts; ECE in %)")
    lines.append("")
    lines.append("| arm | params | n | " + " | ".join(LABELS[c] for c in cols) + " |")
    lines.append("|---|---|---|" + "---|" * len(cols))
    for arm, entry in payload["arms"].items():
        row = [arm, str(entry["num_params"]), str(len(entry["seeds"]))]
        for c in cols:
            v = entry["summary"].get(c)
            row.append("" if v is None else f"{v['mean'] * 100:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Paired differences, dendritic − control (pts, mean [t])")
    lines.append("")
    lines.append("Absolute accuracy under each perturbation. `*` marks |t| > 3.")
    lines.append("")
    lines.append("| group | dendritic vs control | kind | n | " + " | ".join(LABELS[c] for c in cols) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(cols))
    for comp in payload["comparisons"]:
        row = [comp["group"], f"{comp['dendritic']} vs {comp['control']}", comp["control_kind"], str(len(comp["seeds"]))]
        for c in cols:
            block = comp["metrics"].get(c)
            if block is None:
                row.append("")
                continue
            a = block["abs"]
            star = "*" if math.isfinite(a["t"]) and abs(a["t"]) > 3 or math.isinf(a["t"]) else ""
            row.append(f"{a['mean'] * 100:+.2f} [{fmt_t(a['t'])}]{star}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Paired difference in accuracy DROP from own clean, dendritic − control (pts, mean [t])")
    lines.append("")
    lines.append("Negative = dendritic loses less. A lower-clean model can lose less trivially; read with the table above.")
    lines.append("")
    lines.append("| group | dendritic vs control | " + " | ".join(LABELS[c] for c in PERTURBED) + " |")
    lines.append("|---|---|" + "---|" * len(PERTURBED))
    for comp in payload["comparisons"]:
        row = [comp["group"], f"{comp['dendritic']} vs {comp['control']}"]
        for c in PERTURBED:
            block = comp["metrics"].get(c)
            if block is None:
                row.append("")
                continue
            d = block["drop"]
            star = "*" if math.isfinite(d["t"]) and abs(d["t"]) > 3 or math.isinf(d["t"]) else ""
            row.append(f"{d['mean'] * 100:+.2f} [{fmt_t(d['t'])}]{star}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Dendritic wins in absolute accuracy under perturbation (t > 3)")
    lines.append("")
    if payload["dendritic_wins_t_gt_3"]:
        for w in payload["dendritic_wins_t_gt_3"]:
            lines.append(f"- {w['group']}: {w['dendritic']} vs {w['control']} ({w['kind']}), {LABELS[w['metric']]}: "
                         f"{w['mean_pts']:+.2f} pts, t={fmt_t(w['t'])}, p={w['p']:.3g}, n={w['n']}"
                         + (f" [{w['caveat']}]" if w.get("caveat") else ""))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Control wins in absolute accuracy under perturbation (t < -3)")
    lines.append("")
    by_comp: dict[str, list[str]] = {}
    for w in payload["control_wins_t_lt_minus_3"]:
        by_comp.setdefault(f"{w['dendritic']} vs {w['control']} ({w['kind']})", []).append(
            f"{LABELS[w['metric']]} {w['mean_pts']:+.2f}")
    for key, items in by_comp.items():
        lines.append(f"- {key}: " + ", ".join(items))
    if not by_comp:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--group", action="append", choices=sorted(GROUPS), default=None)
    ev.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    ev.add_argument("--eval-seed", type=int, default=0)
    ev.add_argument("--noise-seed", type=int, default=0)
    ev.add_argument("--reference-glob", default=None, help="eval_*.json files holding reference clean test accuracies")
    ev.add_argument("--onnx", action="store_true", help="also score the ONNX dynamic-int8 PTQ path")
    ev.add_argument("--features-only", action="store_true")
    an = sub.add_parser("analyze")
    an.add_argument("--output-dir", required=True)
    ca = sub.add_parser("calibrate", help="temperature scaling fitted on VAL, ECE on TEST")
    ca.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "evaluate":
        if args.data_config != DEFAULT_DATA_CONFIG:
            parser.error("clean parity with eval_b*.json needs the default data config")
        args.group = args.group or sorted(GROUPS)
        return cmd_evaluate(args)
    return cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
