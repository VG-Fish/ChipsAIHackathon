"""Framework step 5: quantization-aware distillation fine-tuning.

The objective here is task loss + KD loss from the fixed teacher, evaluated
through a fake-quantized forward pass. All three parts matter:

- the **fake-quantized forward** puts the rounding error into the loss, so the
  weights move somewhere int8 can represent instead of being rounded after the
  fact;
- the **KD loss** is what makes this *distillation* rather than plain QAT --
  int8 logits are coarse, and a full-precision teacher's soft targets recover
  far more of the decision boundary than one-hot labels do at that resolution;
- the **task loss** keeps the student anchored to the labels when the teacher
  is wrong.

If step 4 clustered the weights, pass its projector so weight sharing is
re-imposed after each step; ``prepare_qat`` swaps modules and would otherwise
quietly break the codebook.
"""
import argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import yaml

from kws.data.dataset import build_datasets
from kws.data.loader import build_data_loader
from kws.data.splits import TRAIN, VAL
from kws.models.ds_cnn import build_ds_cnn
from kws.models.ds_cnn import FeatureModel
from kws.optimize.kd import (
    DistillationCriterion,
    FrozenTeacher,
    KDWeights,
    supports_pooled_features,
)
from kws.optimize.cluster import save_codebook_artifact
from kws.optimize.quantization_compat import (
    DeQuantStub,
    QuantStub,
    convert,
    fuse_modules_qat,
    get_default_qat_qconfig,
    prepare_qat,
)
from kws.train import run_finetune
from kws.utils.logging import get_logger
from kws.utils.profile import profile_model
from kws.utils.seed import set_seed

logger = get_logger(__name__)


class QATWrapper(nn.Module):
    """Quant/dequant stubs around the student.

    ``forward_features`` and ``classify_features`` are forwarded so feature
    distillation still works through the wrapper: the pooled features come back
    fake-quantized, which is what the teacher's features should be matched
    against if the deployed encoder is int8.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward_features(self, x):
        return cast(FeatureModel, self.model).forward_features(self.quant(x))

    def classify_features(self, features):
        return self.dequant(cast(FeatureModel, self.model).classify_features(features))

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


class QuantizableDendriticResidual(nn.Module):
    """Quantization-safe equivalent of PAI's clean deployment wrapper.

    PAI's clean wrapper combines a quantized branch tensor with a floating-point
    per-channel skip coefficient using ordinary Python multiplication. Eager
    int8 conversion cannot promote that pair and crashes at inference time.
    This local deployment wrapper keeps the exact branch order and state-dict
    names, but explicitly dequantizes branch outputs for the learned residual
    arithmetic and requantizes the combined output for the next layer.

    The conversion deliberately rejects PAI processors. The configured
    ``DSConvBlock`` and ``Linear`` perforations do not use them; silently
    dropping one on a future graph would change its function.
    """

    def __init__(self, pai_module: nn.Module):
        super().__init__()
        processors = list(cast(list[object], getattr(pai_module, "processor_array")))
        if any(processor is not None for processor in processors):
            raise ValueError(
                "quantization-safe PAI conversion does not support pre/post "
                "processors"
            )
        if not hasattr(pai_module, "skip_weights"):
            raise ValueError("PAI deployment wrapper has no residual skip weights")

        # Preserve these names so clustered-layer assignments and clean PAI
        # checkpoints still address the same tensors after replacement.
        self.layer_array: nn.ModuleList = cast(
            nn.ModuleList, getattr(pai_module, "layer_array")
        )
        self.skip_weights: nn.ParameterList = cast(
            nn.ParameterList, getattr(pai_module, "skip_weights")
        )
        view_tuple = cast(torch.Tensor, getattr(pai_module, "view_tuple"))
        self.view_shape: tuple[int, ...] = tuple(
            int(value) for value in view_tuple.tolist()
        )
        self.branch_dequant: nn.ModuleList = nn.ModuleList(
            [DeQuantStub() for _ in self.layer_array]
        )
        self.output_quant: nn.Module = QuantStub()

    @property
    def pai_skip_connection_count(self) -> int:
        return sum(weight.shape[0] for weight in self.skip_weights)

    def forward(self, x):
        outputs = []
        final_index = len(self.layer_array) - 1
        for out_index, layer in enumerate(self.layer_array):
            current = self.branch_dequant[out_index](layer(x))
            for in_index in range(out_index):
                scale = self.skip_weights[out_index - 1][in_index].view(
                    self.view_shape
                )
                current = current + scale * outputs[in_index]
            if out_index < final_index:
                current = torch.tanh(current)
            outputs.append(current)
        return self.output_quant(outputs[-1])


def replace_clean_pai_modules(model: nn.Module) -> list[str]:
    """Replace opaque PAI clean wrappers with quantization-safe equivalents."""
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if hasattr(child, "layer_array") and hasattr(child, "processor_array"):
                replacement = QuantizableDendriticResidual(child)
                replacement.train(child.training)
                setattr(parent, child_name, replacement)
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(model)
    return replaced


def _select_backend() -> str:
    # fbgemm (the torch.ao.quantization default) targets x86; qnnpack is the
    # correct backend for the ARM CPUs this deploys to.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        return "qnnpack"
    return "fbgemm"


def _fuse_conv_bn_for_qat(model: nn.Module) -> list[str]:
    """Fuse adjacent Conv2d/BatchNorm2d pairs before preparing QAT.

    The clean PAI graph is not guaranteed to use the DS-CNN attribute names,
    so discover pairs from each parent's module order. Fusing only Conv+BN
    preserves the shared ReLU in ``DSConvBlock`` while matching the topology
    used by MCU inference kernels.
    """
    model.train()
    pairs: list[tuple[nn.Module, list[str]]] = []
    for parent in model.modules():
        child_names = list(parent._modules)
        for conv_name, bn_name in zip(child_names, child_names[1:]):
            conv = parent._modules[conv_name]
            bn = parent._modules[bn_name]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                pairs.append((parent, [conv_name, bn_name]))

    fused: list[str] = []
    for parent, names in pairs:
        fuse_modules_qat(parent, [names], inplace=True)
        fused.append(f"{type(parent).__name__}:{names[0]}+{names[1]}")
    return fused


def quantize_aware_distill_model(
    model: nn.Module,
    input_shape: tuple[int, int],
    data_cfg: dict,
    train_cfg: dict,
    out_checkpoint: Path,
    *,
    teacher_checkpoint: str | None = None,
    codebook_projector=None,
    num_keywords: int | None = None,
    candidate_id: str | None = None,
    source_artifact_id: str | None = None,
    recipe_id: str | None = None,
) -> dict:
    """Fine-tune a live module under fake quantization with task + KD loss.

    Taking a module rather than a checkpoint is what lets the pipeline quantize
    the dendritic deployment graph, whose architecture cannot be rebuilt from a
    model config.
    """
    device = torch.device("cpu")  # QAT fake-quant and the int8 convert target CPU
    model = model.to(device)

    replaced_pai_modules = replace_clean_pai_modules(model)
    if replaced_pai_modules:
        logger.info(
            "Replaced %d clean PAI wrappers with quantization-safe residuals: %s",
            len(replaced_pai_modules),
            replaced_pai_modules,
        )

    backend = _select_backend()
    torch.backends.quantized.engine = backend

    fused_pairs = _fuse_conv_bn_for_qat(model)
    logger.info("Fused %d Conv+BN pairs before QAT", len(fused_pairs))
    wrapped = QATWrapper(model)
    wrapped.qconfig = get_default_qat_qconfig(backend)
    prepare_qat(wrapped, inplace=True)
    logger.info("Prepared fake-quantized graph on the %s backend", backend)

    set_seed(train_cfg["seed"])
    datasets, label_map = build_datasets(
        data_cfg, augment=train_cfg["augment"], seed=train_cfg["seed"],
    )
    train_loader = build_data_loader(datasets[TRAIN], train_cfg, shuffle=True)
    val_loader = build_data_loader(datasets[VAL], train_cfg, shuffle=False)

    kd = None
    if teacher_checkpoint is not None:
        teacher = FrozenTeacher(teacher_checkpoint, device)
        kd = DistillationCriterion(
            teacher,
            KDWeights.from_config(train_cfg.get("distillation")),
            train_cfg["label_smoothing"],
            device,
            # `prepare_qat` swaps modules, so whether the pooled features are
            # still reachable has to be probed on the prepared graph.
            student_feature_dim=(
                _prepared_feature_dim(wrapped, input_shape)
                if supports_pooled_features(wrapped, input_shape)
                else None
            ),
        )
        logger.info(
            "Quantization-aware distillation from %s (feature loss %s)",
            teacher_checkpoint,
            "on" if kd.uses_features else "off",
        )
    else:
        logger.info("No teacher supplied; running plain quantization-aware training")

    post_step = None
    if codebook_projector is not None and len(codebook_projector):
        # The projector addresses modules by name, which survives the QAT swap.
        def post_step(module):
            codebook_projector(module.model)

        logger.info(
            "Re-imposing weight sharing on %d clustered layers after each step",
            len(codebook_projector),
        )

    best_state: dict[str, torch.Tensor] = {}

    def keep_best(module: nn.Module, _val_acc: float) -> None:
        best_state.clear()
        best_state.update(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }
        )

    result = run_finetune(
        wrapped,
        train_loader,
        val_loader,
        device,
        train_cfg,
        kd=kd,
        post_step=post_step,
        on_best=keep_best,
        label="qad",
    )
    if best_state:
        wrapped.load_state_dict(best_state, strict=True)

    wrapped.eval()
    quantized = convert(wrapped, inplace=False)
    torch_cost = profile_model(
        quantized,
        input_shape,
        device=device,
        bits_per_weight=8,
        latency_iterations=10,
        latency_warmup=2,
    ).as_dict()
    # Quantized BatchNorm is executable eagerly and through tracing on the
    # supported qnnpack build, but torch.jit.script lowers its scalar arguments
    # incorrectly in PyTorch 2.14. The input shape is fixed by the deployment
    # graph, so tracing is both valid and the artifact we can execute/benchmark.
    example_input = torch.zeros(1, 1, *input_shape, device=device)
    scripted = torch.jit.trace(quantized, example_input, check_trace=True)

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(out_checkpoint))
    codebook_artifact = None
    if codebook_projector is not None and len(codebook_projector):
        codebook_path = out_checkpoint.with_suffix(
            out_checkpoint.suffix + ".codebook.pt"
        )
        codebook_artifact = save_codebook_artifact(
            quantized.model,
            codebook_projector,
            codebook_path,
            graph_path=out_checkpoint,
        )
    metadata_path = out_checkpoint.with_suffix(out_checkpoint.suffix + ".yaml")
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "stage": "quantize",
                "candidate_id": candidate_id,
                "source_artifact_id": source_artifact_id,
                "recipe_id": recipe_id,
                "teacher_checkpoint": teacher_checkpoint,
                "backend": backend,
                "input_shape": list(input_shape),
                "num_keywords": num_keywords,
                "torch_cost": torch_cost,
                "graph_format": "torchscript",
                "fused_conv_bn_pairs": fused_pairs,
                "quantizable_pai_modules": replaced_pai_modules,
                "codebook_artifact": codebook_artifact,
            },
            sort_keys=False,
        )
    )
    logger.info(
        "Saved int8 TorchScript model -> %s (best fake-quant val_acc=%.4f)",
        out_checkpoint,
        result.best_val_acc,
    )
    return {
        "best_val_acc": result.best_val_acc,
        "final_val_acc": result.final_val_acc,
        "epochs": result.epochs,
        "backend": backend,
        "input_shape": list(input_shape),
        "label_map": label_map,
        "num_keywords": num_keywords,
        "checkpoint": str(out_checkpoint),
        "artifact_metadata": str(metadata_path),
        "candidate_id": candidate_id,
        "source_artifact_id": source_artifact_id,
        "recipe_id": recipe_id,
        "fused_conv_bn_pairs": fused_pairs,
        "quantizable_pai_modules": replaced_pai_modules,
        "torch_cost": torch_cost,
        "distillation": kd.describe() if kd else None,
        "codebook": codebook_projector.describe() if codebook_projector else None,
        "codebook_artifact": codebook_artifact,
        "history": result.history,
        "fake_quantized_model": wrapped,
        "int8_model": quantized,
        "scripted_model": scripted,
    }


def _prepared_feature_dim(model: nn.Module, input_shape: tuple[int, int]) -> int:
    with torch.no_grad():
        feature_model = cast(FeatureModel, model)
        return feature_model.forward_features(
            torch.zeros(1, 1, *input_shape)
        ).shape[1]


def quantize_aware_distill(
    checkpoint_path: str,
    data_cfg: dict,
    train_cfg: dict,
    out_checkpoint: Path,
    *,
    teacher_checkpoint: str | None = None,
    codebook_projector=None,
    candidate_id: str | None = None,
    source_artifact_id: str | None = None,
    recipe_id: str | None = None,
) -> dict:
    """Checkpoint-driven step 5, for a student whose architecture is a config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_ds_cnn(
        checkpoint["model_cfg"], tuple(checkpoint["input_shape"]), checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return quantize_aware_distill_model(
        model,
        tuple(checkpoint["input_shape"]),
        data_cfg,
        train_cfg,
        out_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        codebook_projector=codebook_projector,
        candidate_id=candidate_id,
        source_artifact_id=source_artifact_id,
        recipe_id=recipe_id,
        num_keywords=checkpoint["num_keywords"],
    )


def quantize_aware_train(
    checkpoint_path: str, data_cfg: dict, train_cfg: dict, out_checkpoint: Path,
) -> float:
    """Plain QAT with no teacher -- the ablation against step 5's full objective."""
    return quantize_aware_distill(
        checkpoint_path, data_cfg, train_cfg, out_checkpoint,
    )["best_val_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/speech_commands_v2.yaml")
    parser.add_argument("--train-config", default="configs/train/qat.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Fixed teacher for the KD term; omit to run plain QAT",
    )
    parser.add_argument("--out-checkpoint", required=True)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    quantize_aware_distill(
        args.checkpoint,
        data_cfg,
        train_cfg,
        Path(args.out_checkpoint),
        teacher_checkpoint=args.teacher_checkpoint,
    )


if __name__ == "__main__":
    main()
