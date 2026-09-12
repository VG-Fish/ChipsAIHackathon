"""Framework step 4: layer-wise weight clustering and codebook learning.

Each layer's weights are replaced by an index into a small per-layer codebook,
in the style of Deep Compression (Han et al., 2016). Two things matter for this
project:

- **Memory.** An MRAM-backed MCU pays for weight *storage*, and a k-entry
  codebook turns a 32-bit weight into a ``ceil(log2 k)``-bit index plus a
  handful of shared FP32 centroids. At 16 clusters that is 4 bits per weight
  before any quantization runs.
- **Accuracy.** The codebook is *learned*, not just fitted. Registering it as a
  parametrization makes ``weight = centroids[indices]`` part of the autograd
  graph, so every weight sharing a centroid contributes its gradient to that
  centroid -- which is exactly the codebook update rule, obtained for free
  rather than hand-rolled.

Clustering is per layer because the weight distributions of a depthwise conv, a
pointwise conv, and the classifier have very different scales; one global
codebook would spend most of its centroids on whichever layer has the widest
range.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from kws.utils.artifacts import ArtifactLayout
from kws.utils.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    MetricsRecorder,
    atomic_torch_save,
)
from kws.train import resolve_manifest_run_id, validate_checkpoint_run_id
from kws.utils.logging import get_logger

logger = get_logger(__name__)

CLUSTERABLE_TYPES = (nn.Conv2d, nn.Linear)
BEST_CHECKPOINT_KIND = "kws_best_model"


def _infer_manifest_run_id(path: str | Path) -> str | None:
    """Find an existing run manifest for a direct artifact-writer call."""
    resolved = Path(path).expanduser().resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "manifest.yaml").is_file():
            return ArtifactLayout(parent).manifest_run_id
    return None


@dataclass(frozen=True)
class ClusterSpec:
    """How aggressively to share weights."""

    bits: int = 4
    min_weights: int = 64
    kmeans_iterations: int = 30

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 16:
            raise ValueError("bits must be between 1 and 16")
        if self.min_weights < 1:
            raise ValueError("min_weights must be positive")

    @property
    def clusters(self) -> int:
        return 2 ** self.bits

    @classmethod
    def from_config(cls, config: dict | None) -> "ClusterSpec":
        config = config or {}
        return cls(
            bits=int(config.get("bits", 4)),
            min_weights=int(config.get("min_weights", 64)),
            kmeans_iterations=int(config.get("kmeans_iterations", 30)),
        )

    def as_dict(self) -> dict:
        return {
            "bits": self.bits,
            "clusters": self.clusters,
            "min_weights": self.min_weights,
            "kmeans_iterations": self.kmeans_iterations,
        }


def cluster_weights(
    weight: torch.Tensor, clusters: int, *, iterations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D k-means over a weight tensor; returns sorted centroids and indices.

    Centroids start on a linear grid between the tensor's min and max rather
    than on random samples: weight distributions are unimodal and sharply
    peaked at zero, and random seeding routinely lands several centroids inside
    the peak and leaves the tails -- the weights that actually carry signal --
    sharing one. Deep Compression reports the same finding.
    """
    flat = weight.detach().reshape(-1).float()
    unique = torch.unique(flat)
    effective = min(clusters, unique.numel())
    if effective <= 1:
        centroids = flat.mean().reshape(1)
        return centroids, torch.zeros_like(flat, dtype=torch.long).reshape(weight.shape)

    centroids = torch.linspace(
        float(flat.min()), float(flat.max()), effective, device=flat.device,
    )
    indices = torch.zeros_like(flat, dtype=torch.long)
    for _ in range(max(iterations, 1)):
        indices = torch.argmin((flat.unsqueeze(1) - centroids.unsqueeze(0)).abs(), dim=1)
        totals = torch.zeros_like(centroids).index_add_(0, indices, flat)
        counts = torch.zeros_like(centroids).index_add_(0, indices, torch.ones_like(flat))
        occupied = counts > 0
        updated = centroids.clone()
        updated[occupied] = totals[occupied] / counts[occupied]
        if torch.allclose(updated, centroids):
            centroids = updated
            break
        centroids = updated

    # Sorting makes the codebook deterministic across runs and lets an exporter
    # emit a monotonically increasing table, which most MCU runtimes expect.
    order = torch.argsort(centroids)
    remap = torch.empty_like(order)
    remap[order] = torch.arange(order.numel(), device=order.device)
    return centroids[order].contiguous(), remap[indices].reshape(weight.shape)


class CodebookParametrization(nn.Module):
    """Materializes ``weight = centroids[indices]`` inside the autograd graph."""

    centroids: nn.Parameter
    indices: torch.Tensor

    def __init__(self, centroids: torch.Tensor, indices: torch.Tensor):
        super().__init__()
        self.centroids = nn.Parameter(centroids.clone())
        self.register_buffer("indices", indices.clone())

    def forward(self, _original: torch.Tensor) -> torch.Tensor:
        return self.centroids[self.indices]

    @property
    def clusters(self) -> int:
        return self.centroids.numel()


@dataclass(frozen=True)
class ClusteringReport:
    """What clustering did, and what it bought."""

    spec: dict
    clustered_layers: dict[str, int]
    skipped_layers: list[str]
    clustered_weights: int
    total_weights: int
    total_params: int
    codebook_bytes: int
    dense_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.dense_bytes / max(self.codebook_bytes, 1)

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "clustered_layers": self.clustered_layers,
            "skipped_layers": self.skipped_layers,
            "clustered_weights": self.clustered_weights,
            "total_weights": self.total_weights,
            "total_params": self.total_params,
            "codebook_bytes": self.codebook_bytes,
            "dense_bytes": self.dense_bytes,
            "compression_ratio": self.compression_ratio,
        }


def apply_weight_clustering(model: nn.Module, spec: ClusterSpec) -> ClusteringReport:
    """Cluster every eligible layer in place and register its codebook.

    Layers below ``min_weights`` are left dense: a codebook costs
    ``clusters * 4`` bytes of centroids, so on a tensor of a few dozen weights
    the table is larger than the weights it replaces.
    """
    clustered: dict[str, int] = {}
    skipped: list[str] = []
    dense_params = sum(parameter.numel() for parameter in model.parameters())
    clusterable_weights = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, CLUSTERABLE_TYPES)
    )
    # Biases, BatchNorm tensors, and other non-clustered parameters remain
    # dense and must be included in the reported whole-model footprint.
    codebook_bytes = (dense_params - clusterable_weights) * 4

    for name, module in list(model.named_modules()):
        if not isinstance(module, CLUSTERABLE_TYPES):
            continue
        if parametrize.is_parametrized(module, "weight"):
            skipped.append(name)
            codebook_bytes += module.weight.numel() * 4
            continue
        if module.weight.numel() < spec.min_weights:
            skipped.append(name)
            codebook_bytes += module.weight.numel() * 4
            continue

        centroids, indices = cluster_weights(
            module.weight, spec.clusters, iterations=spec.kmeans_iterations,
        )
        parametrize.register_parametrization(
            module, "weight", CodebookParametrization(centroids, indices),
        )
        # The original dense tensor is kept by `parametrize` but is no longer
        # the thing being learned; freezing it keeps optimizers from touching a
        # tensor that no longer affects the forward pass.
        module.parametrizations.weight.original.requires_grad = False
        clustered[name] = int(centroids.numel())
        codebook_bytes += _codebook_bytes(indices.numel(), int(centroids.numel()))

    total_weights = clusterable_weights
    clustered_weights = sum(
        cast(nn.Conv2d | nn.Linear, model.get_submodule(name)).weight.numel()
        for name in clustered
    )
    report = ClusteringReport(
        spec=spec.as_dict(),
        clustered_layers=clustered,
        skipped_layers=skipped,
        clustered_weights=clustered_weights,
        total_weights=total_weights,
        total_params=dense_params,
        codebook_bytes=codebook_bytes,
        dense_bytes=dense_params * 4,
    )
    logger.info(
        "Clustered %d/%d layers at %d bits: %d -> %d whole-model bytes (%.2fx)",
        len(clustered),
        len(clustered) + len(skipped),
        spec.bits,
        report.dense_bytes,
        report.codebook_bytes,
        report.compression_ratio,
    )
    return report


def _codebook_bytes(weight_count: int, clusters: int) -> int:
    """Centroid table plus packed indices, the form the target actually stores."""
    index_bits = max(math.ceil(math.log2(max(clusters, 2))), 1)
    return clusters * 4 + (weight_count * index_bits + 7) // 8


def codebook_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The centroids -- the only weight-domain parameters left to learn."""
    return [
        module.centroids
        for module in model.modules()
        if isinstance(module, CodebookParametrization)
    ]


def clustered_layer_indices(model: nn.Module) -> dict[str, torch.Tensor]:
    """Map ``module name -> cluster assignment``, for use after baking.

    Step 5 swaps modules for their fake-quantized equivalents, which drops the
    parametrization. Carrying the assignments by name lets the projector below
    keep enforcing weight sharing through quantization-aware fine-tuning.
    """
    assignments: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module, "weight"):
            continue
        for entry in module.parametrizations.weight:
            if isinstance(entry, CodebookParametrization):
                # Strip the `parametrizations` path so the name still resolves
                # once the codebook has been baked away.
                assignments[name] = entry.indices.detach().clone()
    return assignments


class CodebookProjector:
    """Re-imposes weight sharing after each optimizer step, by module name.

    Once the parametrization is gone, nothing stops gradients from pulling
    weights that share a centroid apart. Re-averaging each cluster and writing
    the shared value back is the Deep Compression codebook update applied
    directly in the weight domain, and it survives module swaps.
    """

    def __init__(
        self,
        assignments: dict[str, torch.Tensor],
        *,
        bits: int | None = None,
    ):
        self.assignments = assignments
        self.bits = bits

    def __len__(self) -> int:
        return len(self.assignments)

    def __call__(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, indices in self.assignments.items():
                try:
                    module = cast(nn.Conv2d | nn.Linear, model.get_submodule(name))
                except AttributeError:
                    continue
                weight = module.weight
                flat = weight.reshape(-1)
                flat_indices = indices.reshape(-1).to(weight.device)
                clusters = int(flat_indices.max().item()) + 1
                totals = torch.zeros(clusters, device=weight.device, dtype=flat.dtype)
                totals.index_add_(0, flat_indices, flat)
                counts = torch.zeros(clusters, device=weight.device, dtype=flat.dtype)
                counts.index_add_(0, flat_indices, torch.ones_like(flat))
                centroids = totals / counts.clamp(min=1)
                weight.copy_(centroids[flat_indices].reshape(weight.shape))

    def describe(self) -> dict:
        return {
            "projected_layers": sorted(self.assignments),
            "projected_weights": sum(
                indices.numel() for indices in self.assignments.values()
            ),
            "bits": self.bits,
        }


def _module_weight(module: nn.Module) -> torch.Tensor:
    """Read a float or eager-quantized module's logical weight tensor."""
    weight_or_getter = cast(
        torch.Tensor | Callable[[], torch.Tensor], getattr(module, "weight")
    )
    if isinstance(weight_or_getter, torch.Tensor):
        weight: torch.Tensor = weight_or_getter
    else:
        weight = cast(Callable[[], torch.Tensor], weight_or_getter)()
    return weight.dequantize() if weight.is_quantized else weight


def _codebook_payload(
    model: nn.Module,
    projector: CodebookProjector,
    *,
    bits: int | None = None,
) -> dict:
    """Build a serializable codebook/index payload for a deployment runtime."""
    effective_bits = bits or projector.bits
    layers: dict[str, dict] = {}
    for name, indices in projector.assignments.items():
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(
                f"codebook assignment refers to missing module {name!r}"
            ) from exc
        weight = _module_weight(module).detach().cpu().float()
        flat_indices = indices.detach().cpu().reshape(-1).long()
        if flat_indices.numel() != weight.numel():
            raise ValueError(
                f"codebook assignment for {name!r} has {flat_indices.numel()} "
                f"entries, expected {weight.numel()}"
            )
        cluster_count = int(flat_indices.max().item()) + 1
        centroids = torch.zeros(cluster_count, dtype=torch.float32)
        counts = torch.zeros(cluster_count, dtype=torch.float32)
        centroids.index_add_(0, flat_indices, weight.reshape(-1))
        counts.index_add_(0, flat_indices, torch.ones_like(weight.reshape(-1)))
        centroids = centroids / counts.clamp(min=1)
        layer_bits = effective_bits or max(math.ceil(math.log2(max(cluster_count, 2))), 1)
        if cluster_count > 2 ** layer_bits:
            raise ValueError(
                f"codebook {name!r} has {cluster_count} clusters but only "
                f"{layer_bits} bits were configured"
            )
        layers[name] = {
            "shape": list(weight.shape),
            "centroids": centroids,
            "indices": pack_indices(flat_indices, layer_bits),
            "index_count": flat_indices.numel(),
            "index_bits": layer_bits,
        }
    return {
        "format": "kws-codebook-v1",
        "bits": effective_bits,
        "layers": layers,
    }


def pack_indices(indices: torch.Tensor, bits: int) -> bytes:
    """Pack little-endian fixed-width codebook indices into bytes."""
    if bits < 1 or bits > 16:
        raise ValueError("packed codebook indices require 1-16 bits")
    limit = 1 << bits
    output = bytearray()
    accumulator = 0
    available = 0
    for value in indices.reshape(-1).tolist():
        value = int(value)
        if not 0 <= value < limit:
            raise ValueError(f"codebook index {value} does not fit in {bits} bits")
        accumulator |= value << available
        available += bits
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_indices(packed: bytes, count: int, bits: int) -> torch.Tensor:
    """Inverse of :func:`pack_indices`, useful to target-runtime adapters."""
    if bits < 1 or bits > 16 or count < 0:
        raise ValueError("invalid packed-index shape")
    mask = (1 << bits) - 1
    values: list[int] = []
    accumulator = 0
    available = 0
    for byte in packed:
        accumulator |= int(byte) << available
        available += 8
        while available >= bits and len(values) < count:
            values.append(accumulator & mask)
            accumulator >>= bits
            available -= bits
    if len(values) != count:
        raise ValueError(
            f"packed indices contain {len(values)} values, expected {count}"
        )
    return torch.tensor(values, dtype=torch.long)


def save_codebook_artifact(
    model: nn.Module,
    projector: CodebookProjector,
    path: str | Path,
    *,
    graph_path: str | Path,
    bits: int | None = None,
    run_id: str | None = None,
) -> dict:
    """Persist packed indices and centroids alongside the compatibility graph.

    The TorchScript graph remains available for host validation, while this
    payload is the representation a target runtime can consume instead of
    storing the clustered layers as dense weights.
    """
    run_id = run_id or _infer_manifest_run_id(path)
    payload = _codebook_payload(model, projector, bits=bits)
    payload["run_id"] = run_id
    payload["graph_path"] = str(graph_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(path, payload)
    storage_bytes = sum(
        len(layer["indices"]) + layer["centroids"].numel() * 4
        for layer in payload["layers"].values()
    )
    logger.info("Saved packed codebook artifact -> %s (%d bytes)", path, storage_bytes)
    return {
        "path": str(path),
        "format": payload["format"],
        "run_id": run_id,
        "bits": payload["bits"],
        "layers": sorted(payload["layers"]),
        "storage_bytes": storage_bytes,
    }


def bake_codebooks(model: nn.Module, *, bits: int | None = None) -> CodebookProjector:
    """Fold codebooks back into dense weights, returning the sharing pattern.

    Deployment and ``prepare_qat`` both want plain weight tensors. The returned
    projector preserves the sharing so later stages can keep enforcing it.
    """
    projector = CodebookProjector(clustered_layer_indices(model), bits=bits)
    for name in list(projector.assignments):
        module = model.get_submodule(name)
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    # This is a stage boundary: QAT must be able to adapt BN affine values and
    # biases as well as the baked weights. Codebook parametrization is gone, so
    # there is no reason to retain codebook-finetune's temporary freeze map.
    for parameter in model.parameters():
        parameter.requires_grad = True
    logger.info("Baked %d codebooks into dense weights", len(projector))
    return projector


def save_clustered_model(
    model: nn.Module,
    projector: CodebookProjector,
    path: str | Path,
    *,
    input_shape: tuple[int, int],
    candidate_id: str | None = None,
    recipe_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Persist the baked graph and assignments needed by the next stage.

    A PAI clean graph cannot be rebuilt from the DS-CNN YAML alone. Saving only
    the clustering report would therefore make a later ``--stages quantize``
    invocation silently restart from the unclustered candidate. The graph is
    reconstructed by the pipeline from the PAI artifact and this state dict is
    then applied exactly; assignments are retained so QAT can keep sharing
    weights after its module swaps.
    """
    path = Path(path)
    run_id = run_id or _infer_manifest_run_id(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codebook_payload = _codebook_payload(model, projector)
    atomic_torch_save(path, {
            "stage": "clustered",
            "candidate_id": candidate_id,
            "recipe_id": recipe_id,
            "run_id": run_id,
            "input_shape": list(input_shape),
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "cluster_assignments": {
                name: indices.detach().cpu().clone()
                for name, indices in projector.assignments.items()
            },
            "cluster_bits": projector.bits,
            "cluster_codebooks": codebook_payload["layers"],
        })
    logger.info("Saved clustered deployment graph -> %s", path)


def load_clustered_model(
    model: nn.Module,
    path: str | Path,
    *,
    expected_candidate: str | None = None,
    expected_recipe: str | None = None,
    expected_run_id: str | None = None,
) -> CodebookProjector:
    """Load a clustered graph onto a freshly reconstructed PAI clean model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "clustered":
        raise ValueError(f"{path} is not a clustered deployment artifact")
    validate_checkpoint_run_id(checkpoint, expected_run_id, source=path)
    if (
        expected_candidate is not None
        and checkpoint.get("candidate_id") != expected_candidate
    ):
        raise ValueError(
            f"clustered artifact {path} belongs to candidate "
            f"{checkpoint.get('candidate_id')!r}, expected {expected_candidate!r}"
        )
    if expected_recipe is not None and checkpoint.get("recipe_id") != expected_recipe:
        raise ValueError(
            f"clustered artifact {path} used recipe "
            f"{checkpoint.get('recipe_id')!r}, expected {expected_recipe!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assignments = {
        name: indices.clone()
        for name, indices in checkpoint.get("cluster_assignments", {}).items()
    }
    projector = CodebookProjector(assignments, bits=checkpoint.get("cluster_bits"))
    logger.info("Loaded clustered deployment graph from %s", path)
    return projector


def codebook_finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    train_cfg: dict,
    *,
    kd=None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    resume_from: str | Path | None = None,
    recipe: dict | None = None,
    reset_metrics: bool = False,
    run_id: str | None = None,
) -> dict:
    """Learn the codebook: train the centroids only, with the rest frozen.

    Fitting a codebook by k-means minimizes weight reconstruction error, which
    is not the objective anyone cares about. Training the centroids against the
    task and KD losses instead moves each shared value to where the *loss*
    wants it, and recovers most of what naive clustering gives up -- at a cost
    of a few dozen trainable scalars per layer.
    """
    from kws.train import run_finetune

    centroids = codebook_parameters(model)
    if not centroids:
        return {"status": "skipped", "reason": "no codebooks are registered"}

    for parameter in model.parameters():
        parameter.requires_grad = False
    for centroid in centroids:
        centroid.requires_grad = True

    logger.info(
        "Learning %d codebooks (%d trainable centroids)",
        len(centroids),
        sum(centroid.numel() for centroid in centroids),
    )
    best_state: dict[str, torch.Tensor] = {}
    layout = ArtifactLayout(output_dir) if output_dir is not None else None
    run_id = resolve_manifest_run_id(layout, run_id)
    recorder = (
        MetricsRecorder(
            layout.metrics_path("cluster", "codebook"),
            layout=layout,
            stage="cluster",
            phase="codebook",
        )
        if layout is not None else None
    )
    latest_path = (
        layout.checkpoint_path("cluster", "codebook", "latest") if layout else None
    )
    best_path = (
        layout.checkpoint_path("cluster", "codebook", "best") if layout else None
    )
    if resume_from is None and resume and latest_path is not None and latest_path.exists():
        resume_from = latest_path
    resume_state = (
        torch.load(resume_from, map_location=device, weights_only=False)
        if resume_from is not None else None
    )
    if resume_state is not None:
        validate_checkpoint_run_id(resume_state, run_id, source=resume_from)
    run_id = run_id or (
        resume_state.get("run_id") if resume_state is not None else None
    ) or uuid.uuid4().hex

    def save_best_checkpoint(
        state_dict: dict[str, torch.Tensor], val_acc: float, epoch: int
    ) -> None:
        if best_path is None:
            return
        atomic_torch_save(
            best_path,
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "kind": BEST_CHECKPOINT_KIND,
                "stage": "cluster",
                "phase": "codebook",
                "run_id": run_id,
                "model_format": "cluster-parametrized-state-dict",
                "model_state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in state_dict.items()
                },
                "best_metric_name": "validation_accuracy",
                "best_metric_value": float(val_acc),
                "best_epoch": int(epoch),
                # Retain the conventional alias used by inference checkpoints.
                "val_acc": float(val_acc),
                "stage_specific_state": {
                    "trainable_centroids": sum(
                        centroid.numel() for centroid in centroids
                    ),
                },
            },
        )

    def keep_best(module: nn.Module, _val_acc: float) -> None:
        best_state.clear()
        best_state.update(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }
        )
        epoch = (
            int(recorder.records[-1]["epoch"])
            if recorder is not None and recorder.records
            else 0
        )
        save_best_checkpoint(best_state, _val_acc, epoch)

    result = run_finetune(
        model,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        parameters=centroids,
        on_best=keep_best,
        recorder=recorder,
        latest_path=latest_path,
        resume_state=resume_state,
        best_path=None,
        stage="cluster",
        phase="codebook",
        run_id=run_id,
        recipe={
            **(recipe or {}),
            "train": train_cfg,
            "centroid_count": sum(p.numel() for p in centroids),
        },
        reset_metrics=reset_metrics,
        epochs=epochs if epochs is not None else train_cfg.get("cluster_epochs"),
        label="codebook",
    )
    if best_state:
        model.load_state_dict(best_state, strict=True)
    elif resume_state is not None and resume_state.get("best_model_state_dict"):
        model.load_state_dict(resume_state["best_model_state_dict"], strict=True)
    if layout is not None and latest_path is not None and latest_path.exists():
        latest_state = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_checkpoint_run_id(latest_state, run_id, source=latest_path)
        durable_best_state = latest_state.get("best_model_state_dict") or best_state
        if durable_best_state:
            save_best_checkpoint(
                durable_best_state,
                float(latest_state.get("best_metric_value", result.best_val_acc)),
                int(latest_state.get("best_epoch", 0)),
            )
    # Do not leak centroid-only freezing into QAT when this function is used as
    # part of the end-to-end pipeline.
    for parameter in model.parameters():
        parameter.requires_grad = True
    return {
        "status": "complete",
        "best_val_acc": result.best_val_acc,
        "final_val_acc": result.final_val_acc,
        "epochs": result.epochs,
        "trainable_centroids": sum(centroid.numel() for centroid in centroids),
        "history": result.history,
        "run_id": run_id,
    }
