"""Framework step 6: export and benchmark the exact inference graph.

Everything measured before this point was measured on a PyTorch object. What
ships is an ONNX graph run by a different runtime, and the two can disagree --
on numerics (fused kernels, different accumulation order) and on speed (a graph
optimizer may fuse conv+BN, or may not). So this stage re-measures accuracy and
latency *through the exported graph itself*, and reports the parity gap between
the two rather than assuming there is none.

Latency is measured at batch 1 on the runtime's own session, which is the shape
and the code path an always-on wake-word device actually runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch.utils.data import DataLoader

from kws.data.dataset import build_datasets
from kws.data.splits import TEST, VAL
from kws.evaluate import load_model_from_checkpoint
from kws.export.to_onnx import export_to_onnx
from kws.utils.logging import get_logger
from kws.utils.metrics import compute_metrics
from kws.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """What the exported graph does on the intended target."""

    target: str
    onnx_path: str
    onnx_bytes: int
    providers: list[str]
    split: str
    accuracy: float
    far: float
    frr: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    parity_max_abs_diff: float | None
    f1_per_class: dict
    confusion_matrix: list
    graph_format: str = "onnx"
    measurement_scope: str = "host_runtime_proxy"

    def as_dict(self) -> dict:
        return asdict(self)


def run_split_through_graph(
    session: ort.InferenceSession,
    input_name: str,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a split through the exported graph one sample at a time.

    The deployment export fixes the batch dimension at 1, because that is the
    graph an always-on device runs -- so accuracy has to be measured the same
    way. Feeding it batches would need a differently shaped graph, and then the
    number would describe something other than what ships.
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    for features, labels in loader:
        batch = features.numpy()
        for index in range(batch.shape[0]):
            logits = session.run(None, {input_name: batch[index : index + 1]})[0]
            y_pred.append(int(np.argmax(logits, axis=1)[0]))
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def benchmark_session_latency(
    session: ort.InferenceSession,
    input_name: str,
    input_shape: tuple[int, int],
    *,
    iterations: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    """Batch-1 latency percentiles through the runtime's own session."""
    if iterations < 1:
        raise ValueError("latency iterations must be positive")
    sample = np.zeros((1, 1, *input_shape), dtype=np.float32)
    for _ in range(max(warmup, 0)):
        session.run(None, {input_name: sample})

    timings: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        session.run(None, {input_name: sample})
        timings.append((perf_counter() - started) * 1000.0)
    timings.sort()

    def percentile(fraction: float) -> float:
        return timings[min(int(len(timings) * fraction), len(timings) - 1)]

    return {
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p50": percentile(0.50),
        "latency_ms_p90": percentile(0.90),
        "latency_ms_p99": percentile(0.99),
    }


def measure_parity(
    session: ort.InferenceSession,
    input_name: str,
    torch_model: torch.nn.Module,
    input_shape: tuple[int, int],
    *,
    samples: int = 16,
) -> float:
    """Largest logit disagreement between PyTorch and the exported graph.

    Reported rather than asserted: the export already enforces a tolerance, and
    what a reader of the benchmark wants to know is how much of any accuracy
    difference is numerics and how much is the graph.
    """
    torch_model.eval()
    batch = torch.randn(samples, 1, *input_shape)
    with torch.no_grad():
        torch_out = torch_model(batch).numpy()
    onnx_out = np.concatenate(
        [
            session.run(None, {input_name: batch[index : index + 1].numpy()})[0]
            for index in range(samples)
        ]
    )
    return float(np.abs(torch_out - onnx_out).max())


def export_and_benchmark(
    checkpoint_path: str,
    onnx_path: str,
    data_cfg: dict,
    *,
    target: str = "onnxruntime-cpu",
    split: str = TEST,
    seed: int = 0,
    latency_iterations: int = 200,
    batch_size: int = 128,
    report_path: str | None = None,
) -> BenchmarkResult:
    """Export, then measure accuracy and latency on the exported graph itself."""
    if target != "onnxruntime-cpu":
        raise NotImplementedError(
            f"target runtime {target!r} has no registered benchmark adapter; "
            "ESP32/MRAM measurements must be supplied by a device harness"
        )
    export_to_onnx(checkpoint_path, onnx_path)

    set_seed(seed)
    device = torch.device("cpu")
    torch_model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    input_shape = tuple(checkpoint["input_shape"])

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    datasets, label_map = build_datasets(data_cfg, augment=False, seed=seed)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    if split not in (TEST, VAL):
        raise ValueError(f"benchmark split must be {TEST!r} or {VAL!r}, got {split!r}")
    loader = DataLoader(datasets[split], batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = run_split_through_graph(session, input_name, loader)
    metrics = compute_metrics(y_true, y_pred, checkpoint["num_keywords"], label_names)
    latency = benchmark_session_latency(
        session, input_name, input_shape, iterations=latency_iterations,
    )
    result = BenchmarkResult(
        target=target,
        onnx_path=str(onnx_path),
        onnx_bytes=Path(onnx_path).stat().st_size,
        providers=list(session.get_providers()),
        split=split,
        accuracy=metrics["accuracy"],
        far=metrics["far"],
        frr=metrics["frr"],
        parity_max_abs_diff=measure_parity(session, input_name, torch_model, input_shape),
        f1_per_class=metrics["f1_per_class"],
        confusion_matrix=metrics["confusion_matrix"],
        **latency,
    )

    logger.info(
        "Benchmark on %s: %s accuracy %.4f, %.3f ms p50 / %.3f ms p99, "
        "%d bytes on disk, parity %.2e",
        target,
        split,
        result.accuracy,
        result.latency_ms_p50,
        result.latency_ms_p99,
        result.onnx_bytes,
        result.parity_max_abs_diff,
    )

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result.as_dict(), f, indent=2)
        logger.info("Wrote benchmark report to %s", report_path)
    return result


def _run_split_through_torchscript(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Score the exact TorchScript inference artifact one sample at a time."""
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            for index in range(features.shape[0]):
                logits = model(features[index : index + 1])
                y_pred.append(int(logits.argmax(dim=1)[0]))
            y_true.extend(labels.tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def _torchscript_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    *,
    iterations: int,
    warmup: int = 20,
) -> dict[str, float]:
    if iterations < 1:
        raise ValueError("latency iterations must be positive")
    sample = torch.zeros(1, 1, *input_shape)
    model.eval()
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            model(sample)
        timings = []
        for _ in range(iterations):
            started = perf_counter()
            model(sample)
            timings.append((perf_counter() - started) * 1000.0)
    timings.sort()

    def percentile(fraction: float) -> float:
        return timings[min(int(len(timings) * fraction), len(timings) - 1)]

    return {
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_p50": percentile(0.50),
        "latency_ms_p90": percentile(0.90),
        "latency_ms_p99": percentile(0.99),
    }


def benchmark_scripted_model(
    model: torch.nn.Module,
    graph_path: str,
    input_shape: tuple[int, int],
    data_cfg: dict,
    num_keywords: int,
    *,
    target: str = "torchscript-cpu",
    split: str = TEST,
    seed: int = 0,
    latency_iterations: int = 200,
    batch_size: int = 128,
    report_path: str | None = None,
) -> BenchmarkResult:
    """Benchmark the converted QAT TorchScript graph itself.

    Eager quantized PyTorch modules are not exportable to ONNX on the supported
    PyTorch versions (their packed convolution weights are opaque). The traced
    TorchScript file is the actual converted inference graph produced by step 5
    and is therefore the faithful target artifact to score in that case.
    """
    if target != "torchscript-cpu":
        raise NotImplementedError(
            f"target runtime {target!r} has no registered TorchScript adapter; "
            "ESP32/MRAM measurements must be supplied by a device harness"
        )
    if split not in (TEST, VAL):
        raise ValueError(f"benchmark split must be {TEST!r} or {VAL!r}, got {split!r}")
    set_seed(seed)
    model = model.to("cpu").eval()
    datasets, label_map = build_datasets(data_cfg, augment=False, seed=seed)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    loader = DataLoader(
        datasets[split], batch_size=batch_size, shuffle=False, num_workers=0,
    )
    y_true, y_pred = _run_split_through_torchscript(model, loader)
    metrics = compute_metrics(y_true, y_pred, num_keywords, label_names)
    latency = _torchscript_latency(
        model, input_shape, iterations=latency_iterations,
    )
    result = BenchmarkResult(
        target=target,
        onnx_path=str(graph_path),
        onnx_bytes=Path(graph_path).stat().st_size,
        providers=["torchscript-cpu"],
        split=split,
        accuracy=metrics["accuracy"],
        far=metrics["far"],
        frr=metrics["frr"],
        latency_ms_mean=latency["latency_ms_mean"],
        latency_ms_p50=latency["latency_ms_p50"],
        latency_ms_p90=latency["latency_ms_p90"],
        latency_ms_p99=latency["latency_ms_p99"],
        # There is no second runtime here to compare against: this is the
        # exact converted artifact being measured, not an ONNX translation.
        parity_max_abs_diff=None,
        f1_per_class=metrics["f1_per_class"],
        confusion_matrix=metrics["confusion_matrix"],
        graph_format="torchscript",
    )
    logger.info(
        "Benchmark on %s: %s accuracy %.4f, %.3f ms p50 / %.3f ms p99, "
        "%d bytes on disk",
        target,
        split,
        result.accuracy,
        result.latency_ms_p50,
        result.latency_ms_p99,
        result.onnx_bytes,
    )
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result.as_dict(), f, indent=2)
        logger.info("Wrote benchmark report to %s", report_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--target", default="onnxruntime-cpu")
    parser.add_argument("--split", default=TEST, choices=[TEST, VAL])
    parser.add_argument("--latency-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    export_and_benchmark(
        args.checkpoint,
        args.onnx_path,
        data_cfg,
        target=args.target,
        split=args.split,
        seed=args.seed,
        latency_iterations=args.latency_iterations,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()


def benchmark_module(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    onnx_path: str,
    data_cfg: dict,
    num_keywords: int,
    *,
    target: str = "onnxruntime-cpu",
    split: str = TEST,
    seed: int = 0,
    latency_iterations: int = 200,
    batch_size: int = 128,
    report_path: str | None = None,
) -> BenchmarkResult:
    """Step 6 for a live module whose architecture is not rebuildable from config.

    The dendritic deployment graph and the codebook-parametrized student both
    reach this path; the checkpoint-driven variant above is for plain DS-CNNs.
    """
    if target != "onnxruntime-cpu":
        raise NotImplementedError(
            f"target runtime {target!r} has no registered ONNX adapter; "
            "ESP32/MRAM measurements must be supplied by a device harness"
        )
    from kws.export.to_onnx import export_module_to_onnx

    model = model.to("cpu").eval()
    parity = export_module_to_onnx(model, input_shape, onnx_path)

    set_seed(seed)
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    if split not in (TEST, VAL):
        raise ValueError(f"benchmark split must be {TEST!r} or {VAL!r}, got {split!r}")
    datasets, label_map = build_datasets(data_cfg, augment=False, seed=seed)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    loader = DataLoader(datasets[split], batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = run_split_through_graph(session, input_name, loader)
    metrics = compute_metrics(y_true, y_pred, num_keywords, label_names)
    latency = benchmark_session_latency(
        session, input_name, input_shape, iterations=latency_iterations,
    )
    result = BenchmarkResult(
        target=target,
        onnx_path=str(onnx_path),
        onnx_bytes=Path(onnx_path).stat().st_size,
        providers=list(session.get_providers()),
        split=split,
        accuracy=metrics["accuracy"],
        far=metrics["far"],
        frr=metrics["frr"],
        parity_max_abs_diff=parity,
        f1_per_class=metrics["f1_per_class"],
        confusion_matrix=metrics["confusion_matrix"],
        **latency,
    )
    logger.info(
        "Benchmark on %s: %s accuracy %.4f, %.3f ms p50, %d bytes on disk",
        target, split, result.accuracy, result.latency_ms_p50, result.onnx_bytes,
    )
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result.as_dict(), f, indent=2)
        logger.info("Wrote benchmark report to %s", report_path)
    return result
