"""Deploy a trained SparkNet to the RP2040, end to end.

    python -m kws.export.rp2040_pipeline --checkpoint RUN/models/checkpoints/.../best.pt --output-dir OUT
    python -m kws.export.rp2040_pipeline --grow-run outputs/sparknet-grow-dendrites-v3/ARM/cW-seedS --output-dir OUT

Steps (see :mod:`kws.export.rp2040` for the integer design):

1. load the model -- a scratch checkpoint, or a grow run's clean dendritic
   state (``best_clean`` by default: the run's validation-selected epoch);
2. fold every BatchNorm and check the folded float graph against torch;
3. calibrate activation ranges on training clips (no augmentation);
4. choose activation bits x calibration method on validation accuracy;
5. report float and integer metrics on the test split -- measured once, after
   every choice is made;
6. emit the C model, compile ``kws_engine.c`` for the host and require
   bit-exact logits against the numpy reference on the whole test split;
7. write a Pico SDK self-test firmware project and, when the SDK and an ARM
   toolchain with a C library are available, build it into a ``.uf2``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from kws.data.dataset import build_datasets
from kws.data.splits import TEST, TRAIN, VAL
from kws.evaluate import load_model_from_checkpoint
from kws.export import rp2040
from kws.optimize.grow_clean_rebuild import load_clean_state, rebuild_clean_model
from kws.utils.artifacts import ArtifactLayout, sha256_path
from kws.utils.logging import get_logger, run_session
from kws.utils.metrics import compute_metrics
from kws.utils.seed import set_seed

logger = get_logger(__name__)

C_SOURCES = Path(__file__).resolve().parent / "rp2040_c"
DEFAULT_DATA_CONFIG = "configs/data/speech_commands_v2_mfcc32_paper.yaml"
FOLD_TOLERANCE = 1e-4
UF2_FAMILY_RP2040 = 0xE48BFF56
RP2040_FLASH_BASE = 0x10000000


# ---------------------------------------------------------------------------
# model and data


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict]:
    """The eval-mode model and a description of where it came from."""
    if args.checkpoint:
        model, ckpt = load_model_from_checkpoint(args.checkpoint, torch.device("cpu"))
        source = {
            "kind": "checkpoint",
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_path(args.checkpoint),
            "model": ckpt["model_cfg"].get("name"),
            "val_acc_at_training": 100 * float(ckpt["val_acc"]) if "val_acc" in ckpt else None,
        }
        return model.eval(), source
    run = Path(args.grow_run)
    summary = yaml.safe_load((run / "reports" / "grow_summary.yaml").read_text())
    if summary.get("status") != "complete":
        raise ValueError(f"{run}: grow run is {summary.get('status')!r}, not complete")
    state_path = run / summary["artifacts"][args.grow_checkpoint]
    state, _ = load_clean_state(state_path)
    model_cfg = yaml.safe_load(Path(summary["model_config"]).read_text())
    model = rebuild_clean_model(model_cfg, summary["input_shape"], summary["num_classes"], state, torch.tanh)
    results = summary.get("results", {})
    val = results.get("best_val_acc_post_switch" if args.grow_checkpoint == "best_clean" else "final_val_acc")
    source = {
        "kind": "grow_run",
        "path": str(run.resolve()),
        "arm": summary.get("arm"),
        "width": summary.get("width"),
        "seed": summary.get("seed"),
        "state": args.grow_checkpoint,
        "state_sha256": sha256_path(state_path),
        "distillation": (summary.get("variant") or {}).get("distillation"),
        "val_acc_at_training": None if val is None else 100 * float(val),
    }
    return model.eval(), source


def load_split(data_cfg: dict, split: str, seed: int, *, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(features (N, F, T) float32, labels, label names)`` of one split, no augmentation.

    Each split is built alone, right after seeding, exactly as the frontier
    evaluation built it, so test numbers are comparable with that table.
    ``limit`` draws a fixed random subset (the calibration clips).
    """
    set_seed(seed)
    datasets, label_map = build_datasets(data_cfg, augment=False, seed=seed, splits={split})
    dataset = datasets[split]
    if limit is not None and limit < len(dataset):
        chosen = np.random.default_rng(seed).choice(len(dataset), limit, replace=False)
        dataset = Subset(dataset, sorted(int(i) for i in chosen))
    features, labels = [], []
    for x, y in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
        features.append(x.squeeze(1).numpy())
        labels.append(y.numpy())
    names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    return np.concatenate(features), np.concatenate(labels), names


def batched(fn, x: np.ndarray, size: int = 512) -> np.ndarray:
    return np.concatenate([fn(x[i:i + size]) for i in range(0, len(x), size)])


def metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str], num_keywords: int) -> dict:
    m = compute_metrics(y_true, y_pred, num_keywords, names)
    keyword_f1 = [m["f1_per_class"][name] for name in names[:num_keywords]]
    return {
        "accuracy": round(100 * float(m["accuracy"]), 3),
        "frr": round(100 * m["frr"], 3),
        "far": round(100 * m["far"], 3),
        "keyword_f1": round(100 * float(np.mean(keyword_f1)), 3),
    }


# ---------------------------------------------------------------------------
# C artifacts


def scratch_elems(features: int, channels: int, frames: int) -> int:
    """int16 scratch the engine needs; mirrors ``kws_scratch_elems`` in kws_engine.c."""
    return 3 * max(features, channels) * frames


def write_test_vectors(path: Path, xq: np.ndarray, logits: np.ndarray, labels: np.ndarray, channels: int) -> None:
    """``kws_test_vectors.h`` for the firmware self-test."""
    count, features, frames = xq.shape
    classes = logits.shape[1]
    lines = [
        "/* Generated by kws.export.rp2040_pipeline -- do not edit. */\n#pragma once\n#include <stdint.h>\n\n",
        f"#define KWS_N_VECTORS {count}\n#define KWS_N_CLASSES {classes}\n",
        f"#define KWS_SCRATCH_ELEMS {scratch_elems(features, channels, frames)}\n\n",
        f"static const int16_t kws_vectors[{count}][{features * frames}] = {{\n",
    ]
    for clip in xq.astype(np.int64):
        lines.append("    {" + ",".join(str(int(v)) for v in clip.reshape(-1)) + "},\n")
    lines.append("};\n")
    lines.append(f"static const int32_t kws_expected[{count}][{classes}] = {{\n")
    for row in logits:
        lines.append("    {" + ", ".join(str(int(v)) for v in row) + "},\n")
    lines.append("};\n")
    lines.append(f"static const uint8_t kws_labels[{count}] = {{ " + ", ".join(str(int(v)) for v in labels) + " };\n")
    path.write_text("".join(lines))


def run_checked(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def host_parity(model: rp2040.QuantizedSparkNet, header: Path, symbol: str, xq: np.ndarray,
                expected: np.ndarray, compiler: str) -> dict:
    """Compile the engine for the host and compare its logits with ``forward_int``.

    Built with UBSan, so signed overflow or a bad shift aborts the run instead
    of passing silently.
    """
    with tempfile.TemporaryDirectory(prefix="kws_host_") as scratch:
        scratch = Path(scratch)
        binary = scratch / "kws_host"
        command = [
            compiler, "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=undefined", "-fno-sanitize-recover=all",
            f"-I{C_SOURCES}", f"-I{header.parent}",
            f"-DKWS_MODEL_HEADER=\"{header.name}\"", f"-DKWS_MODEL={symbol}",
            str(C_SOURCES / "kws_host.c"), str(C_SOURCES / "kws_engine.c"), "-o", str(binary),
        ]
        run_checked(command)
        inputs, outputs = scratch / "inputs.bin", scratch / "logits.bin"
        xq.astype("<i2").tofile(inputs)
        start = time.perf_counter()
        run_checked([str(binary), str(inputs), str(len(xq)), str(outputs)])
        elapsed = time.perf_counter() - start
        logits = np.fromfile(outputs, dtype="<i4").reshape(len(xq), -1).astype(np.int64)
    mismatched = np.flatnonzero(np.any(logits != expected, axis=1))
    return {
        "compiler": compiler,
        "clips": int(len(xq)),
        "bit_exact_clips": int(len(xq) - len(mismatched)),
        "bit_exact": bool(len(mismatched) == 0),
        "max_abs_logit_difference_q16": int(np.abs(logits - expected).max()),
        "same_prediction": bool(np.all(logits.argmax(1) == expected.argmax(1))),
        "host_seconds": round(elapsed, 3),
    }


def bin_to_uf2(data: bytes, base: int = RP2040_FLASH_BASE, family: int = UF2_FAMILY_RP2040) -> bytes:
    """UF2 image of a flat flash binary (256-byte payloads, family ID set)."""
    count = (len(data) + 255) // 256
    blocks = []
    for index in range(count):
        payload = data[index * 256:(index + 1) * 256].ljust(256, b"\0")
        header = struct.pack("<8I", 0x0A324655, 0x9E5D5157, 0x00002000, base + index * 256, 256, index, count, family)
        blocks.append(header + payload + bytes(476 - 256) + struct.pack("<I", 0x0AB16F30))
    return b"".join(blocks)


def write_firmware_project(directory: Path, header: Path, symbol: str, name: str, channels: int,
                           xq: np.ndarray, logits: np.ndarray, labels: np.ndarray, sdk: Path | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source, target in (("firmware_main.c", "main.c"), ("kws_engine.c", "kws_engine.c"),
                           ("kws_engine.h", "kws_engine.h"), ("firmware_CMakeLists.txt", "CMakeLists.txt")):
        shutil.copyfile(C_SOURCES / source, directory / target)
    shutil.copyfile(header, directory / header.name)
    (directory / "kws_firmware_model.h").write_text(
        f'#pragma once\n#include "{header.name}"\n#define KWS_MODEL {symbol}\n#define KWS_MODEL_NAME "{name}"\n'
    )
    write_test_vectors(directory / "kws_test_vectors.h", xq, logits, labels, channels)
    if sdk is not None and (sdk / "external" / "pico_sdk_import.cmake").exists():
        shutil.copyfile(sdk / "external" / "pico_sdk_import.cmake", directory / "pico_sdk_import.cmake")


def build_firmware(directory: Path, sdk: Path) -> dict:
    """Configure and build with the Pico SDK; the UF2 is written here, not by picotool."""
    build = directory / "build"
    configure = [
        "cmake", "-S", str(directory), "-B", str(build), "-G", "Ninja",
        f"-DPICO_SDK_PATH={sdk}", "-DPICO_BOARD=pico", "-DPICO_NO_PICOTOOL=1",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    run_checked(configure)
    run_checked(["cmake", "--build", str(build)])
    elf, image = build / "kws_rp2040.elf", build / "kws_rp2040.bin"
    uf2 = directory / "kws_rp2040.uf2"
    uf2.write_bytes(bin_to_uf2(image.read_bytes()))
    size = run_checked(["arm-none-eabi-size", "-A", str(elf)]).stdout
    sections = {}
    for line in size.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            sections[parts[0]] = int(parts[1])
    flash = sum(v for k, v in sections.items() if k in (".boot2", ".text", ".rodata", ".binary_info", ".data", ".ARM.exidx"))
    ram = sum(v for k, v in sections.items() if k in (".data", ".bss", ".heap", ".stack_dummy", ".tdata", ".tbss"))
    return {
        "uf2": str(uf2),
        "uf2_sha256": hashlib.sha256(uf2.read_bytes()).hexdigest(),
        "binary_bytes": image.stat().st_size,
        "flash_bytes_approx": flash,
        "static_ram_bytes_approx": ram,
        "sections": sections,
    }


def toolchain_problem() -> str | None:
    """Why the firmware cannot be built here, or None."""
    for tool in ("cmake", "ninja", "arm-none-eabi-gcc", "arm-none-eabi-size"):
        if shutil.which(tool) is None:
            return f"{tool} not found"
    libc = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m0plus", "-print-file-name=libc.a"],
                          capture_output=True, text=True).stdout.strip()
    if not Path(libc).is_absolute():
        return "arm-none-eabi-gcc has no C library (newlib); install the Arm GNU Toolchain"
    return None


# ---------------------------------------------------------------------------


def run(args: argparse.Namespace, layout: ArtifactLayout) -> dict:
    model, source = load_model(args)
    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    num_keywords = len(data_cfg["target_keywords"])
    folded = rp2040.fold_sparknet(model)

    x_cal, _, _ = load_split(data_cfg, TRAIN, args.seed, limit=args.calibration_clips)
    x_val, y_val, names = load_split(data_cfg, VAL, args.seed)
    x_test, y_test, _ = load_split(data_cfg, TEST, args.seed)
    logger.info("clips: calibration %d (train), val %d, test %d", len(x_cal), len(x_val), len(x_test))

    with torch.no_grad():
        torch_val = batched(lambda x: model(torch.from_numpy(x).unsqueeze(1).float()).double().numpy(), x_val)
    float_val = batched(folded.forward, x_val)
    fold_error = float(np.abs(torch_val - float_val).max())
    if fold_error > FOLD_TOLERANCE:
        raise ValueError(f"folded graph differs from torch by {fold_error:.2e} (> {FOLD_TOLERANCE})")

    # 4. choose on validation
    calibrations = {
        method: rp2040.calibrate(folded, [x_cal[i:i + 250] for i in range(0, len(x_cal), 250)], method)
        for method in args.calibration
    }
    grid = []
    for method, calibration in calibrations.items():
        for bits in args.act_bits:
            quantized = rp2040.quantize(folded, calibration, act_bits=bits, n_frames=x_val.shape[2])
            int_val = batched(lambda x: quantized.forward_int(quantized.quantize_input(x)), x_val)
            grid.append({
                "calibration": method,
                "act_bits": bits,
                "val_accuracy": round(100 * float(np.mean(int_val.argmax(1) == y_val)), 3),
                "val_agreement_with_float": round(100 * float(np.mean(int_val.argmax(1) == float_val.argmax(1))), 3),
            })
            logger.info("val: %s %d-bit -> acc %.2f, agreement %.2f", method, bits,
                        grid[-1]["val_accuracy"], grid[-1]["val_agreement_with_float"])
    chosen = max(grid, key=lambda row: (row["val_accuracy"], row["val_agreement_with_float"], -row["act_bits"]))
    quantized = rp2040.quantize(folded, calibrations[chosen["calibration"]], act_bits=chosen["act_bits"],
                                n_frames=x_val.shape[2])
    bounds = rp2040.check_integer_bounds(quantized)

    # 5. test, once
    float_test = batched(folded.forward, x_test)
    xq_test = quantized.quantize_input(x_test)
    int_test = batched(quantized.forward_int, xq_test)
    test = {
        "float": metrics(y_test, float_test.argmax(1), names, num_keywords),
        "int": metrics(y_test, int_test.argmax(1), names, num_keywords),
        "agreement": round(100 * float(np.mean(int_test.argmax(1) == float_test.argmax(1))), 3),
        "clips": int(len(y_test)),
    }
    val = {
        "float": metrics(y_val, float_val.argmax(1), names, num_keywords),
        "int": {k: chosen[k] for k in ("val_accuracy", "val_agreement_with_float")},
        "clips": int(len(y_val)),
    }
    logger.info("test: float %.2f, int %.2f (agreement %.2f)", test["float"]["accuracy"],
                test["int"]["accuracy"], test["agreement"])

    # 6. C model and host parity
    exported = layout.exported_path("rp2040")
    exported.mkdir(parents=True, exist_ok=True)
    header = rp2040.write_c_model(quantized, exported / f"{args.symbol}.h", args.symbol, comment=args.name)
    np.savez_compressed(exported / "test_reference.npz", inputs=xq_test.astype(np.int16),
                        logits_q16=int_test.astype(np.int32), labels=y_test.astype(np.uint8))
    parity = host_parity(quantized, header, args.symbol, xq_test, int_test, args.cc)
    if not parity["bit_exact"]:
        raise RuntimeError(f"C engine is not bit-exact: {parity}")
    logger.info("host parity: %d/%d clips bit-exact", parity["bit_exact_clips"], parity["clips"])

    # 7. firmware
    picks = np.linspace(0, len(x_test) - 1, args.firmware_clips).round().astype(int)
    channels = quantized.blocks[-1].cout
    firmware_dir = exported / "firmware"
    sdk = Path(args.pico_sdk).expanduser() if args.pico_sdk else None
    write_firmware_project(firmware_dir, header, args.symbol, args.name, channels,
                           xq_test[picks], int_test[picks], y_test[picks], sdk)
    firmware: dict = {"project": str(firmware_dir), "self_test_clips": int(len(picks))}
    problem = "no --pico-sdk" if sdk is None else toolchain_problem()
    if args.skip_build:
        problem = "--skip-build"
    if problem is None:
        firmware.update(build_firmware(firmware_dir, sdk))
        logger.info("firmware: %s (%d bytes)", firmware["uf2"], firmware["binary_bytes"])
    else:
        firmware["build_skipped"] = problem
        logger.warning("firmware not built: %s", problem)

    report = {
        "format_version": 1,
        "name": args.name,
        "source": source,
        "data_config": args.data_config,
        "seed": args.seed,
        "selection": "act_bits x calibration chosen on validation accuracy; test measured once afterwards",
        "calibration_clips": int(len(x_cal)),
        "calibration_split": "train (no augmentation)",
        "float_parameters": int(sum(p.numel() for p in model.parameters())),
        "folded_parameters": folded.parameter_count,
        "fold_max_abs_logit_difference": fold_error,
        "grid": grid,
        "chosen": {"calibration": chosen["calibration"], "act_bits": chosen["act_bits"]},
        "validation": val,
        "test": test,
        "integer_bounds_log2": {k: round(v, 2) for k, v in bounds.items()},
        "int8_weight_bytes": quantized.weight_bytes,
        "activation_scratch_bytes": 2 * scratch_elems(x_test.shape[1], channels, x_test.shape[2]),
        "c_model": str(header),
        "c_model_sha256": sha256_path(header),
        "host_parity": parity,
        "firmware": firmware,
    }
    layout.atomic_yaml(layout.report_path("rp2040.yaml"), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="scratch SparkNet checkpoint (best.pt)")
    source.add_argument("--grow-run", help="grow_dendrites run directory")
    parser.add_argument("--grow-checkpoint", choices=("best_clean", "final_clean"), default="best_clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", help="model name in the report and firmware (default: output dir name)")
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-clips", type=int, default=1000)
    parser.add_argument("--calibration", nargs="+", default=["max", "p99.99", "p99.9"])
    parser.add_argument("--act-bits", nargs="+", type=int, default=[8, 16], choices=(8, 16))
    parser.add_argument("--symbol", default="kws_model", help="C identifier of the model")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"), help="host C compiler for the parity check")
    parser.add_argument("--firmware-clips", type=int, default=20, help="test clips embedded in the self-test")
    parser.add_argument("--pico-sdk", default=os.environ.get("PICO_SDK_PATH"))
    parser.add_argument("--skip-build", action="store_true", help="write the firmware project without building it")
    args = parser.parse_args(argv)
    args.name = args.name or Path(args.output_dir).name
    inputs = [(args.checkpoint or args.grow_run, "checkpoint" if args.checkpoint else "grow_run")]
    layout = ArtifactLayout(args.output_dir)
    with run_session(layout=layout, command="kws.export.rp2040_pipeline", argv=sys.argv if argv is None else argv,
                     seed=args.seed, inputs=inputs):
        report = run(args, layout)
    print(yaml.safe_dump({k: report[k] for k in ("name", "chosen", "test", "host_parity", "firmware")}, sort_keys=False))


if __name__ == "__main__":
    main()
