"""Tests for the grow-one-dendrite-during-training driver.

The unit tests exercise the driver's pure helpers directly.  The ``test_pai_*``
tests run ``run_grow`` for real, on a tiny SparkNet and a synthetic dataset,
each in a fresh subprocess because PerforatedAI's tracker is process-global.
Deselect them with ``-k "not test_pai_"``.  They need the PAI license, so run
the file as ``uv run --env-file .env python -m pytest ...``.
"""

import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors import safe_open
from torch.utils.data import DataLoader, TensorDataset

from kws.models.registry import build_model
from kws.optimize import sparknet_grow_dendrites as grow
from kws.optimize.dendritic import fold_dendrite_input_scale
from kws.optimize.grow_clean_rebuild import (
    PlainSequential,
    RebuiltDendriteModule,
    group_conv_with_batchnorm,
    rebuild_clean_model,
)
from kws.train import build_lr_scheduler, build_optimizer
from kws.utils.artifacts import ArtifactLayout

ROOT = Path(__file__).parents[1]
GROW_TRAIN_CONFIG = ROOT / "configs/train/sparknet_grow_dendrites_paper.yaml"


def _paper_train_cfg() -> dict:
    return yaml.safe_load(GROW_TRAIN_CONFIG.read_text())


def _grow_config(epochs: int, switch_epoch: int, candidate_epochs: int) -> grow.GrowConfig:
    return grow.GrowConfig(
        epochs=epochs,
        switch_epoch=switch_epoch,
        candidate_epochs=candidate_epochs,
        placement="fc",
        module_ids=(".fc",),
        candidate_optimizer={"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        dendrite_weight_decay=0.0,
        carry_momentum=True,
        max_wall_clock_minutes=None,
    )


# --------------------------------------------------------------------------
# resolve_grow_config


def test_resolve_grow_config_reads_the_paper_defaults():
    config = grow.resolve_grow_config(_paper_train_cfg())

    assert config == grow.GrowConfig(
        epochs=200,
        switch_epoch=120,
        candidate_epochs=15,
        placement="fc",
        module_ids=(".fc",),
        candidate_optimizer={"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
        dendrite_weight_decay=0.0,
        carry_momentum=True,
        max_wall_clock_minutes=45.0,
    )
    assert config.total_epochs == 215
    # Order is PAI's conversion order, so it is preserved from the config.
    assert grow.resolve_grow_config(
        _paper_train_cfg(), placement="pointwise"
    ).module_ids == (".blocks.3.pointwise", ".blocks.2.pointwise")


def test_pointwise_blocks_are_available_as_independent_placements():
    train_cfg = _paper_train_cfg()

    assert grow.resolve_grow_config(
        train_cfg, placement="pointwise_b2"
    ).module_ids == (".blocks.2.pointwise",)
    assert grow.resolve_grow_config(
        train_cfg, placement="pointwise_b3"
    ).module_ids == (".blocks.3.pointwise",)
    assert grow.resolve_grow_config(
        train_cfg, placement="pointwise_b123"
    ).module_ids == (".blocks.3.pointwise", ".blocks.2.pointwise", ".blocks.1.pointwise")


def test_cli_overrides_win_and_the_train_config_is_not_mutated():
    train_cfg = _paper_train_cfg()
    pristine = copy.deepcopy(train_cfg)

    config = grow.resolve_grow_config(
        train_cfg, placement="pointwise", switch_epoch=50, candidate_epochs=3, max_minutes=5
    )

    assert (config.placement, config.switch_epoch, config.candidate_epochs) == ("pointwise", 50, 3)
    assert config.module_ids == (".blocks.3.pointwise", ".blocks.2.pointwise")
    assert config.max_wall_clock_minutes == 5.0
    assert config.total_epochs == 203
    assert train_cfg == pristine


def test_every_configured_placement_names_real_sparknet_modules():
    """PAI matches module ids exactly, so a typo would perforate nothing."""
    train_cfg = _paper_train_cfg()
    model = build_model(
        {"family": "sparknet", "name": "c4", "channels": 4, "gate_channels": 8}, (32, 101), 12
    )
    module_ids = {f".{name}" for name, _module in model.named_modules() if name}

    for placement in train_cfg["grow_dendrites"]["placements"]:
        config = grow.resolve_grow_config(train_cfg, placement=placement)
        assert set(config.module_ids) <= module_ids, placement


def test_candidate_optimizer_is_normalized():
    train_cfg = _paper_train_cfg()
    train_cfg["grow_dendrites"]["candidate_optimizer"] = {"name": "SGD", "lr": 0.05}
    train_cfg["grow_dendrites"]["max_wall_clock_minutes"] = None

    config = grow.resolve_grow_config(train_cfg)

    assert config.candidate_optimizer == {
        "name": "sgd", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
    }
    assert config.max_wall_clock_minutes is None


def _set_grow(key, value):
    def edit(train_cfg):
        train_cfg["grow_dendrites"][key] = value
    return edit


def _set_placement(name, module_ids):
    def edit(train_cfg):
        train_cfg["grow_dendrites"]["placements"][name] = module_ids
        train_cfg["grow_dendrites"]["placement"] = name
    return edit


def _no_edit(_train_cfg):
    pass


@pytest.mark.parametrize(
    ("edit", "overrides", "message"),
    [
        (_no_edit, {"switch_epoch": 0}, "switch_epoch must be in"),
        (_no_edit, {"switch_epoch": 200}, "switch_epoch must be in"),
        (_no_edit, {"switch_epoch": 201}, "switch_epoch must be in"),
        (_set_grow("switch_epoch", 0), {}, "switch_epoch must be in"),
        (_no_edit, {"candidate_epochs": 0}, "candidate_epochs must be at least 1"),
        (_no_edit, {"candidate_epochs": -2}, "candidate_epochs must be at least 1"),
        (_no_edit, {"placement": "everywhere"}, "unknown placement 'everywhere'"),
        (_set_grow("placement", None), {}, "unknown placement None"),
        (_set_grow("placements", {}), {}, "placements must map"),
        (_set_placement("empty", []), {}, "names no modules"),
        (_set_placement("undotted", ["fc"]), {}, "dot-prefixed"),
        (_set_placement("indexed", [".blocks[3].pointwise"]), {}, "dot-prefixed"),
        (_set_grow("candidate_optimizer", {"name": "rmsprop"}), {}, "'adamw' or 'sgd'"),
        (_set_grow("candidate_optimizer", {"name": "adamw", "lr": 0}), {}, "lr > 0"),
        (
            _set_grow("candidate_optimizer", {"name": "adamw", "weight_decay": -1e-4}),
            {},
            "weight_decay >= 0",
        ),
        (_set_grow("dendrite_weight_decay", -0.1), {}, "dendrite_weight_decay"),
        (_no_edit, {"dendrite_weight_decay": -1e-4}, "dendrite_weight_decay"),
        (_no_edit, {"dendrite_weight_decay": float("nan")}, "dendrite_weight_decay"),
        (_no_edit, {"dendrite_weight_decay": float("inf")}, "dendrite_weight_decay"),
        (_set_grow("dendrite_input_scale", 0), {}, "dendrite_input_scale"),
        (_no_edit, {"dendrite_input_scale": -2.0}, "dendrite_input_scale"),
        (_no_edit, {"dendrite_input_scale": float("nan")}, "dendrite_input_scale"),
        (_no_edit, {"dendrite_input_scale": float("inf")}, "dendrite_input_scale"),
        (_no_edit, {"arm": ""}, "arm must be a non-empty name"),
        (_no_edit, {"arm": "fc sham"}, "arm must be a non-empty name"),
        (_no_edit, {"arm": "runs/fc"}, "arm must be a non-empty name"),
        (_no_edit, {"arm": "fc\n"}, "arm must be a non-empty name"),
        (_no_edit, {"max_minutes": 0}, "max_wall_clock_minutes must be positive"),
        (_no_edit, {"max_minutes": -5}, "max_wall_clock_minutes must be positive"),
        (_set_grow("max_wall_clock_minutes", 0), {}, "max_wall_clock_minutes must be positive"),
    ],
)
def test_resolve_grow_config_rejects_invalid_settings(edit, overrides, message):
    train_cfg = _paper_train_cfg()
    edit(train_cfg)
    with pytest.raises(ValueError, match=message):
        grow.resolve_grow_config(train_cfg, **overrides)


def test_resolve_grow_config_accepts_the_extreme_switch_epochs():
    for switch_epoch in (1, 199):
        config = grow.resolve_grow_config(_paper_train_cfg(), switch_epoch=switch_epoch)
        assert config.switch_epoch == switch_epoch


def test_resolve_grow_config_requires_the_grow_block():
    train_cfg = _paper_train_cfg()
    del train_cfg["grow_dendrites"]
    with pytest.raises(ValueError, match="no grow_dendrites block"):
        grow.resolve_grow_config(train_cfg)


# --------------------------------------------------------------------------
# Run variants: --dendrite-weight-decay, --sham, --arm


def test_a_default_run_is_a_real_arm_named_after_its_placement():
    config = grow.resolve_grow_config(_paper_train_cfg(), placement="pointwise")

    assert config.sham is False
    assert config.arm == "pointwise"
    assert config.variant() == {
        "sham": False, "dendrite_weight_decay": 0.0, "switch_epoch": 120, "candidate_epochs": 15,
        "dendrite_input_scale": 1.0,
    }


def test_variant_flags_resolve_into_the_config_and_its_variant_record():
    train_cfg = _paper_train_cfg()
    pristine = copy.deepcopy(train_cfg)

    config = grow.resolve_grow_config(
        train_cfg, placement="pointwise", switch_epoch=100, candidate_epochs=5,
        dendrite_weight_decay=5e-4, sham=True, dendrite_input_scale=80,
    )

    assert config.arm == "pointwise-sham"
    assert config.dendrite_weight_decay == 5e-4
    assert config.variant() == {
        "sham": True, "dendrite_weight_decay": 5e-4, "switch_epoch": 100, "candidate_epochs": 5,
        "dendrite_input_scale": 80.0,
    }
    assert train_cfg == pristine


@pytest.mark.parametrize("arm", ["fc-wd1e-3", "pointwise.sham_2", "A", "fc"])
def test_an_explicit_arm_is_kept_verbatim(arm):
    for sham in (False, True):
        assert grow.resolve_grow_config(_paper_train_cfg(), sham=sham, arm=arm).arm == arm


def test_the_weight_decay_override_beats_the_config_even_when_zero():
    train_cfg = _paper_train_cfg()
    train_cfg["grow_dendrites"]["dendrite_weight_decay"] = 0.01

    assert grow.resolve_grow_config(train_cfg).dendrite_weight_decay == 0.01
    assert grow.resolve_grow_config(train_cfg, dendrite_weight_decay=0.0).dendrite_weight_decay == 0.0
    assert grow.resolve_grow_config(train_cfg, dendrite_weight_decay=2).dendrite_weight_decay == 2.0


def test_the_input_scale_override_beats_the_config():
    train_cfg = _paper_train_cfg()
    assert grow.resolve_grow_config(train_cfg).dendrite_input_scale == 1.0

    train_cfg["grow_dendrites"]["dendrite_input_scale"] = 40
    assert grow.resolve_grow_config(train_cfg).dendrite_input_scale == 40.0
    assert grow.resolve_grow_config(train_cfg, dendrite_input_scale=1).dendrite_input_scale == 1.0
    assert grow.resolve_grow_config(train_cfg, dendrite_input_scale=75).variant()[
        "dendrite_input_scale"
    ] == 75.0


def test_switch_input_scale_calibration_is_an_explicit_run_variant():
    config = grow.resolve_grow_config(
        _paper_train_cfg(),
        placement="pointwise_b2",
        calibrate_dendrite_input_scale=True,
    )

    assert config.calibrate_dendrite_input_scale is True
    assert config.dendrite_input_scale == 1.0


def test_switch_input_scale_calibration_uses_the_geometric_mean():
    assert grow.calibrated_dendrite_input_scale({"blocks.2.pointwise": 81.0}) == 81.0
    assert grow.calibrated_dendrite_input_scale(
        {"b2": 25.0, "b3": 100.0}
    ) == pytest.approx(50.0)

    for values in ({}, {"b2": 0.0}, {"b2": float("nan")}):
        with pytest.raises(grow.GrowInvariantError, match="input std"):
            grow.calibrated_dendrite_input_scale(values)


# --------------------------------------------------------------------------
# --group-batchnorm


def _paper_sparknet(width: int = 12) -> tuple[dict, nn.Module]:
    model_cfg = yaml.safe_load((ROOT / f"configs/model/sparknet_c{width}_paper.yaml").read_text())
    torch.manual_seed(0)
    return model_cfg, build_model(model_cfg, (32, 101), 12)


def test_group_batchnorm_is_a_run_variant_with_its_own_arm():
    config = grow.resolve_grow_config(
        _paper_train_cfg(), placement="pointwise_b2", group_batchnorm=True
    )
    sham = grow.resolve_grow_config(
        _paper_train_cfg(), placement="pointwise_b2", group_batchnorm=True, sham=True
    )

    assert config.group_batchnorm is True
    assert config.arm == "pointwise_b2-bn"
    assert config.variant()["group_batchnorm"] is True
    assert config.dendrite_input_scale == 1.0
    assert sham.arm == "pointwise_b2-bn-sham"
    assert "group_batchnorm" not in grow.resolve_grow_config(_paper_train_cfg()).variant()


def test_the_last_group_batchnorm_flag_wins():
    # a job queue that always passes --group-batchnorm can still run an fc arm
    parser = grow.build_arg_parser()
    required = ["--model-config", "m.yaml", "--output-dir", "out"]

    assert parser.parse_args(required).group_batchnorm is False
    assert parser.parse_args(required + ["--group-batchnorm"]).group_batchnorm is True
    assert parser.parse_args(required + ["--group-batchnorm", "--no-group-batchnorm"]).group_batchnorm is False


@pytest.mark.parametrize(
    ("edit", "overrides", "message"),
    [
        (_no_edit, {"placement": "fc"}, "needs 'pointwise' placements"),
        (_no_edit, {"placement": "depthwise"}, "needs 'pointwise' placements"),
        (_no_edit, {"placement": "pointwise_b2", "dendrite_input_scale": 34.8}, "leave it at 1"),
        (_set_grow("dendrite_input_scale", 50), {"placement": "pointwise_b2"}, "leave it at 1"),
        (
            _no_edit,
            {"placement": "pointwise_b2", "calibrate_dendrite_input_scale": True},
            "leave it at 1",
        ),
    ],
)
def test_group_batchnorm_rejects_what_it_cannot_group(edit, overrides, message):
    train_cfg = _paper_train_cfg()
    edit(train_cfg)
    with pytest.raises(ValueError, match=message):
        grow.resolve_grow_config(train_cfg, group_batchnorm=True, **overrides)


# --------------------------------------------------------------------------
# --teacher-checkpoint (knowledge distillation)


def _save_sparknet_checkpoint(path: Path, seed: int, width: int = 4) -> nn.Module:
    model_cfg, _ = _paper_sparknet(width)
    torch.manual_seed(seed)
    model = build_model(model_cfg, (32, 101), 12)
    torch.save(
        {"model_cfg": model_cfg, "input_shape": [32, 101], "num_classes": 12,
         "model_state_dict": model.state_dict()},
        path,
    )
    return model.eval()


def test_distillation_is_a_run_variant_with_its_own_arm(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"")
    config = grow.resolve_grow_config(
        _paper_train_cfg(), placement="pointwise_b2", group_batchnorm=True,
        teacher_checkpoints=[str(teacher)], kd_alpha=0.7, kd_temperature=2.0,
    )
    plain = grow.resolve_grow_config(_paper_train_cfg(), kd_alpha=0.7, kd_temperature=2.0)

    assert config.arm == "pointwise_b2-bn-kd"
    assert config.variant()["distillation"] == {
        "teachers": [str(teacher)], "alpha": 0.7, "temperature": 2.0,
    }
    assert (plain.kd_alpha, plain.kd_temperature, plain.teacher_checkpoints) == (0.0, 1.0, ())
    assert "distillation" not in plain.variant()
    explicit = grow.resolve_grow_config(
        _paper_train_cfg(), teacher_checkpoints=[str(teacher)], arm="mine"
    )
    assert explicit.arm == "mine"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kd_alpha": 0.0}, "kd_alpha"),
        ({"kd_alpha": 1.5}, "kd_alpha"),
        ({"kd_temperature": 0.0}, "kd_temperature"),
        ({"teacher_checkpoints": ["/nonexistent/teacher.pt"]}, "not found"),
    ],
)
def test_distillation_rejects_invalid_settings(tmp_path, overrides, message):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"")
    kwargs = {"teacher_checkpoints": [str(teacher)], **overrides}
    with pytest.raises(ValueError, match=message):
        grow.resolve_grow_config(_paper_train_cfg(), **kwargs)


def test_distillation_loss_is_the_temperature_scaled_kl():
    torch.manual_seed(0)
    student, teacher = torch.randn(5, 12), torch.randn(5, 12)
    expected = sum(
        (torch.softmax(t / 3, 0) * (torch.log_softmax(t / 3, 0) - torch.log_softmax(s / 3, 0))).sum()
        for s, t in zip(student, teacher)
    ) / 5 * 9
    assert grow.distillation_loss(student, teacher, 3.0) == pytest.approx(float(expected), rel=1e-5)
    assert grow.distillation_loss(teacher, teacher, 3.0) == pytest.approx(0.0, abs=1e-6)


def test_the_teacher_averages_member_logits_and_never_trains(tmp_path):
    members = [_save_sparknet_checkpoint(tmp_path / f"t{seed}.pt", seed) for seed in (1, 2)]
    teacher = grow.LogitEnsembleTeacher(
        [str(tmp_path / "t1.pt"), str(tmp_path / "t2.pt")], torch.device("cpu")
    )
    teacher.train()
    features = torch.randn(3, 1, 32, 101)
    with torch.no_grad():
        expected = (members[0](features) + members[1](features)) / 2

    assert not teacher.training and not any(member.training for member in teacher.members)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert torch.allclose(teacher(features), expected, atol=1e-6)
    # Eval mode draws no gate noise, so the teacher is deterministic.
    assert torch.equal(teacher(features), teacher(features))


def test_a_distilled_epoch_mixes_the_losses_and_logs_both(tmp_path):
    _save_sparknet_checkpoint(tmp_path / "t.pt", seed=1)
    teacher = grow.LogitEnsembleTeacher([str(tmp_path / "t.pt")], torch.device("cpu"))
    model_cfg, _ = _paper_sparknet(4)
    torch.manual_seed(0)
    model = build_model(model_cfg, (32, 101), 12)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 1, 32, 101), torch.randint(0, 12, (8,))), batch_size=8
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    losses, _ = grow._train_epoch(
        model, loader, optimizer, None, nn.CrossEntropyLoss(), 100.0, torch.device("cpu"),
        pin_base_batchnorm=False, teacher=teacher, kd_alpha=0.25, kd_temperature=2.0,
    )
    auxiliary = sum(value for name, value in losses.items() if name not in {"total", "ce", "kd"})

    assert losses["total"] == pytest.approx(
        100.0 * (0.75 * losses["ce"] + 0.25 * losses["kd"]) + auxiliary, rel=1e-5
    )


def test_grouping_a_pointwise_conv_with_its_batchnorm_changes_only_the_layout():
    _model_cfg, model = _paper_sparknet()
    grouped = copy.deepcopy(model)
    rng = torch.random.get_rng_state()

    assert group_conv_with_batchnorm(grouped, [".blocks.2.pointwise"]) == ["blocks.2.pointwise"]

    assert torch.equal(torch.random.get_rng_state(), rng)
    block = grouped.blocks[2]
    assert isinstance(block.pointwise, PlainSequential)
    assert isinstance(block.bn, nn.Identity)
    # The same tensors in the same order, three of them renamed: an optimizer
    # built over either model steps the same list.
    before, after = list(model.named_parameters()), list(grouped.named_parameters())
    assert len(before) == len(after)
    assert all(torch.equal(left, right) for (_, left), (_, right) in zip(before, after))
    assert {left: right for (left, _), (right, _) in zip(before, after) if left != right} == {
        "blocks.2.pointwise.weight": "blocks.2.pointwise.model.0.weight",
        "blocks.2.bn.weight": "blocks.2.pointwise.model.1.weight",
        "blocks.2.bn.bias": "blocks.2.pointwise.model.1.bias",
    }

    features = torch.randn(8, 1, 32, 101)
    for training in (True, False):
        outputs = []
        for net in (model, grouped):
            net.train(training)
            torch.manual_seed(1)  # the training gate noise
            outputs.append(net(features))
        assert torch.equal(outputs[0], outputs[1]), training
    assert torch.equal(model.blocks[2].bn.running_var, block.pointwise.model[1].running_var)


@pytest.mark.parametrize(
    ("module_id", "message"),
    [
        (".fc", "only a TCSBlock's 'pointwise'"),
        (".blocks.2.depthwise", "only a TCSBlock's 'pointwise'"),
        (".blocks.2", "only a TCSBlock's 'pointwise'"),
        (".nowhere.pointwise", "no module 'nowhere'"),
    ],
)
def test_grouping_refuses_anything_but_a_block_pointwise_conv(module_id, message):
    _model_cfg, model = _paper_sparknet()
    with pytest.raises(ValueError, match=message):
        group_conv_with_batchnorm(model, [module_id])


def test_grouping_refuses_to_group_twice():
    _model_cfg, model = _paper_sparknet()
    group_conv_with_batchnorm(model, [".blocks.2.pointwise"])
    with pytest.raises(ValueError, match="already grouped"):
        group_conv_with_batchnorm(model, [".blocks.2.pointwise"])


@pytest.mark.parametrize("blocks", [(2,), (3, 2, 1)])
def test_the_rebuild_regroups_a_grouped_export(blocks):
    """A clean state whose dendrite branches are (conv, BN) pairs rebuilds exactly."""
    model_cfg, model = _paper_sparknet()
    group_conv_with_batchnorm(model, [f".blocks.{index}.pointwise" for index in blocks])
    for index in blocks:
        main = model.blocks[index].pointwise
        dendrite = copy.deepcopy(main)
        with torch.no_grad():
            for parameter in dendrite.parameters():
                parameter.normal_()
            dendrite.model[1].running_mean.normal_()
            dendrite.model[1].running_var.uniform_(0.5, 2.0)
        module = RebuiltDendriteModule([dendrite, main], 12, [1, -1, 1, 1])
        with torch.no_grad():
            module.skip_weights[0].normal_()
        model.blocks[index].pointwise = module
    model.eval()

    rebuilt = rebuild_clean_model(model_cfg, (32, 101), 12, model.state_dict())

    for index in range(4):
        assert isinstance(rebuilt.blocks[index].bn, nn.Identity) == (index in blocks)
    features = torch.randn(4, 1, 32, 101)
    with torch.no_grad():
        assert torch.equal(rebuilt(features), model(features))


def test_a_scaled_forward_function_divides_the_pre_activation():
    z = torch.linspace(-300, 300, 13)
    scaled = grow.ScaledForwardFunction(torch.tanh, 75)

    assert torch.equal(scaled(z), torch.tanh(z / 75))
    assert repr(scaled) == "tanh(z / 75)"
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            grow.ScaledForwardFunction(torch.tanh, bad)


def test_the_forward_function_context_restores_and_the_plain_one_unwraps():
    original = grow.GPA.pc.get_pai_forward_function()
    scaled = grow.ScaledForwardFunction(torch.tanh, 10)
    try:
        with pytest.raises(RuntimeError, match="inside"):
            with grow.pai_forward_function(scaled):
                assert grow.GPA.pc.get_pai_forward_function() is scaled
                assert grow.plain_forward_function() is torch.tanh
                raise RuntimeError("inside")
        assert grow.GPA.pc.get_pai_forward_function() is original
        with grow.pai_forward_function(torch.sigmoid):
            assert grow.plain_forward_function() is torch.sigmoid
    finally:
        grow.GPA.pc.set_pai_forward_function(original)


class _CleanDendriteModule(nn.Module):
    """PAI's clean one-dendrite forward: main(x) + skip * f(dendrite(x))."""

    def __init__(self, dendrite: nn.Module, main: nn.Module, channels: int) -> None:
        super().__init__()
        self.layer_array = nn.ModuleList([dendrite, main])
        self.skip_weights = nn.ParameterList([nn.Parameter(torch.randn(1, channels))])

    def forward(self, x: torch.Tensor, f=torch.tanh) -> torch.Tensor:
        return self.layer_array[-1](x) + self.skip_weights[0] * f(self.layer_array[0](x))


@pytest.mark.parametrize("bias", [True, False])
def test_folding_the_input_scale_reproduces_the_scaled_graph_under_plain_tanh(bias):
    torch.manual_seed(0)
    module = _CleanDendriteModule(nn.Linear(6, 4, bias=bias), nn.Linear(6, 4), 4)
    x = torch.randn(32, 6) * 80
    scaled = grow.ScaledForwardFunction(torch.tanh, 75.0)
    with torch.no_grad():
        expected = module(x, f=scaled)
    dendrite = module.layer_array[0]
    trained_weight = dendrite.weight

    assert fold_dendrite_input_scale(module, 75.0) == 1

    with torch.no_grad():
        folded = module(x)
    assert torch.allclose(folded, expected, rtol=1e-5, atol=1e-5)
    # New Parameters: nothing that shares storage with the trained graph moved.
    assert dendrite.weight is not trained_weight
    assert torch.allclose(dendrite.weight * 75.0, trained_weight)
    # The original module is never touched.
    main_before = module.layer_array[-1].weight.detach().clone()
    fold_dendrite_input_scale(module, 2.0)
    assert torch.equal(module.layer_array[-1].weight, main_before)


def test_folding_is_a_no_op_at_scale_one_and_refuses_what_it_cannot_fold():
    module = _CleanDendriteModule(nn.Linear(3, 2), nn.Linear(3, 2), 2)
    weight = module.layer_array[0].weight
    assert fold_dendrite_input_scale(module, 1.0) == 0
    assert module.layer_array[0].weight is weight

    for bad in (0.0, -3.0, float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            fold_dendrite_input_scale(module, bad)
    with pytest.raises(ValueError, match="no clean dendrite branches"):
        fold_dendrite_input_scale(nn.Sequential(nn.Linear(3, 2)), 5.0)
    nonlinear = _CleanDendriteModule(nn.Sequential(nn.Linear(3, 2), nn.ReLU()), nn.Linear(3, 2), 2)
    with pytest.raises(ValueError, match="only Linear and Conv"):
        fold_dendrite_input_scale(nonlinear, 5.0)


def test_folding_reaches_a_pointwise_conv_dendrite():
    torch.manual_seed(1)
    module = _CleanDendriteModule(
        nn.Conv2d(8, 8, 1, bias=False), nn.Conv2d(8, 8, 1, bias=False), 8
    )
    module.skip_weights[0] = nn.Parameter(torch.randn(1, 8, 1, 1))
    x = torch.randn(4, 8, 1, 101) * 60
    scaled = grow.ScaledForwardFunction(torch.tanh, 60.0)
    with torch.no_grad():
        expected = module(x, f=scaled)
    fold_dendrite_input_scale(module, 60.0)
    with torch.no_grad():
        assert torch.allclose(module(x), expected, rtol=1e-5, atol=1e-4)


def test_a_grow_config_built_directly_gets_the_default_arm():
    config = _grow_config(epochs=10, switch_epoch=4, candidate_epochs=3)
    assert (config.sham, config.arm) == (False, "fc")
    assert dataclasses.replace(config, sham=True, arm="").arm == "fc-sham"
    assert dataclasses.replace(config, arm="mine").arm == "mine"
    assert grow.default_arm("gate_conv", sham=True) == "gate_conv-sham"


def test_configure_grow_pai_refuses_the_capacity_test_before_touching_pai():
    with pytest.raises(ValueError, match="testing_dendrite_capacity"):
        grow.configure_grow_pai(
            {"testing_dendrite_capacity": True}, (".fc",), torch.device("cpu")
        )


# --------------------------------------------------------------------------
# GrowConfig timeline


@pytest.mark.parametrize(
    ("epoch", "segment", "base_epoch"),
    [
        (1, "pre_switch", 1),
        (120, "pre_switch", 120),  # S
        (121, "candidate", None),  # S + 1
        (135, "candidate", None),  # S + C
        (136, "post_switch", 121),  # S + C + 1
        (215, "post_switch", 200),  # E + C
    ],
)
def test_segment_and_base_epoch_across_the_boundaries(epoch, segment, base_epoch):
    config = grow.resolve_grow_config(_paper_train_cfg())
    assert config.total_epochs == 215
    assert config.segment(epoch) == segment
    assert config.base_epoch(epoch) == base_epoch


def test_base_epochs_cover_the_recipe_exactly_once():
    config = _grow_config(epochs=10, switch_epoch=4, candidate_epochs=3)
    base_epochs = [config.base_epoch(epoch) for epoch in range(1, config.total_epochs + 1)]
    assert [epoch for epoch in base_epochs if epoch is not None] == list(range(1, 11))
    assert base_epochs.count(None) == 3


# --------------------------------------------------------------------------
# build_base_scheduler


def _lr_trace(train_cfg, total_steps, *, steps, start_step=0, two_groups=False):
    """Learning rates (one per group) in force before each of ``steps`` steps."""
    first, second = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))
    if two_groups:
        optimizer = build_optimizer(
            [{"params": [first]}, {"params": [second], "weight_decay": 0.0}], train_cfg
        )
    else:
        optimizer = build_optimizer([first], train_cfg)
    scheduler = grow.build_base_scheduler(optimizer, train_cfg, total_steps, start_step=start_step)
    trace = []
    for _ in range(steps):
        trace.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return trace


def _cosine_train_cfg() -> dict:
    return {
        "optimizer": "sgd", "lr": 0.05, "momentum": 0.9, "weight_decay": 1e-3,
        "scheduler": "cosine", "warmup_fraction": 0.1,
    }


TOTAL_STEPS = 200


@pytest.mark.parametrize("train_cfg_factory", [_paper_train_cfg, _cosine_train_cfg])
@pytest.mark.parametrize("split_step", [1, 10, 57, 90, 150, 199])
def test_rebuilt_scheduler_continues_the_uninterrupted_schedule(train_cfg_factory, split_step):
    train_cfg = train_cfg_factory()
    uninterrupted = _lr_trace(train_cfg, TOTAL_STEPS, steps=TOTAL_STEPS)

    before = _lr_trace(train_cfg, TOTAL_STEPS, steps=split_step)
    # A new optimizer, as after the candidate phase, entered mid-schedule.
    after = _lr_trace(
        train_cfg, TOTAL_STEPS, steps=TOTAL_STEPS - split_step, start_step=split_step
    )

    assert before + after == uninterrupted
    # Not vacuous: the schedule really moves, so a restart would differ.
    assert len({lr for (lr,) in uninterrupted}) > 10
    assert after[0] != uninterrupted[0]


def test_paper_schedule_trace_is_warmup_hold_decay():
    trace = [lr for (lr,) in _lr_trace(_paper_train_cfg(), TOTAL_STEPS, steps=TOTAL_STEPS)]
    # warmup = 10 steps, hold until step 90, then polynomial decay.
    assert trace[0] == pytest.approx(0.01 / 11)
    assert trace[10:90] == [pytest.approx(0.01)] * 80
    assert trace[150] < trace[90]
    assert trace[-1] < 0.01 * 0.01


@pytest.mark.parametrize("split_step", [0, 57, 150])
def test_rebuilt_scheduler_drives_both_parameter_groups(split_step):
    train_cfg = _paper_train_cfg()
    single = _lr_trace(
        train_cfg, TOTAL_STEPS, steps=TOTAL_STEPS - split_step, start_step=split_step
    )
    both = _lr_trace(
        train_cfg, TOTAL_STEPS, steps=TOTAL_STEPS - split_step, start_step=split_step,
        two_groups=True,
    )
    assert both == [[lr, lr] for (lr,) in single]


def test_rebuilt_scheduler_past_the_end_holds_min_lr():
    trace = _lr_trace(_paper_train_cfg(), TOTAL_STEPS, steps=3, start_step=TOTAL_STEPS + 5)
    assert trace == [[pytest.approx(1e-6)]] * 3


def test_build_base_scheduler_rejects_a_negative_start():
    optimizer = build_optimizer([nn.Parameter(torch.zeros(()))], _paper_train_cfg())
    with pytest.raises(ValueError, match="start_step"):
        grow.build_base_scheduler(optimizer, _paper_train_cfg(), TOTAL_STEPS, start_step=-1)


# --------------------------------------------------------------------------
# capture_optimizer_state_by_name / restore_optimizer_state_by_name


def _mlp(out_features: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, out_features))


def _sgd(parameters):
    return torch.optim.SGD(parameters, lr=0.1, momentum=0.9, weight_decay=1e-3)


def _train_step(model, optimizer, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(8, 4, generator=generator)
    optimizer.zero_grad()
    model(features).pow(2).mean().backward()
    optimizer.step()


def _trained_twins(make_optimizer=_sgd):
    """A model with optimizer history, and a copy on new Parameter objects."""
    torch.manual_seed(0)
    source = _mlp()
    optimizer = make_optimizer(source.parameters())
    for seed in range(3):
        _train_step(source, optimizer, seed)
    twin = copy.deepcopy(source)
    assert all(a is not b for a, b in zip(source.parameters(), twin.parameters()))
    return source, optimizer, twin


def _same_parameters(left: nn.Module, right: nn.Module) -> bool:
    return all(
        name_a == name_b and torch.equal(a, b)
        for (name_a, a), (name_b, b) in zip(left.named_parameters(), right.named_parameters())
    )


@pytest.mark.parametrize(
    "make_optimizer",
    [_sgd, lambda parameters: torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.01)],
    ids=["sgd", "adamw"],
)
def test_restored_state_continues_like_the_uninterrupted_optimizer(make_optimizer):
    source, source_optimizer, twin = _trained_twins(make_optimizer)
    captured = grow.capture_optimizer_state_by_name(source, source_optimizer)
    assert set(captured) == {name for name, _ in source.named_parameters()}

    twin_optimizer = make_optimizer(twin.parameters())
    restored = grow.restore_optimizer_state_by_name(twin, twin_optimizer, captured)
    assert restored == 4

    cold = copy.deepcopy(twin)
    cold_optimizer = make_optimizer(cold.parameters())

    _train_step(source, source_optimizer, seed=99)
    _train_step(twin, twin_optimizer, seed=99)
    _train_step(cold, cold_optimizer, seed=99)
    assert _same_parameters(source, twin)
    # Without the carried state the same step lands elsewhere.
    assert not _same_parameters(source, cold)


def test_captured_state_is_a_snapshot_not_an_alias():
    source, optimizer, _twin = _trained_twins()
    captured = grow.capture_optimizer_state_by_name(source, optimizer)
    frozen = {name: state["momentum_buffer"].clone() for name, state in captured.items()}

    _train_step(source, optimizer, seed=7)

    for name, parameter in source.named_parameters():
        assert captured[name]["momentum_buffer"] is not optimizer.state[parameter]["momentum_buffer"]
        assert torch.equal(captured[name]["momentum_buffer"], frozen[name])


def test_capture_skips_parameters_without_state():
    model = _mlp()
    assert grow.capture_optimizer_state_by_name(model, _sgd(model.parameters())) == {}


def test_restore_skips_shape_mismatches():
    source, optimizer, _twin = _trained_twins()
    captured = grow.capture_optimizer_state_by_name(source, optimizer)
    wider = _mlp(out_features=5)  # same names, "2.weight"/"2.bias" change shape
    wider_optimizer = _sgd(wider.parameters())

    assert grow.restore_optimizer_state_by_name(wider, wider_optimizer, captured) == 2
    assert set(wider_optimizer.state) == {wider[0].weight, wider[0].bias}


def test_restore_skips_parameters_outside_the_optimizer():
    source, optimizer, twin = _trained_twins()
    captured = grow.capture_optimizer_state_by_name(source, optimizer)
    partial = _sgd([twin[0].weight])

    assert grow.restore_optimizer_state_by_name(twin, partial, captured) == 1
    assert set(partial.state) == {twin[0].weight}


def test_restore_into_a_base_plus_dendrite_optimizer():
    """The post-switch layout: base group restored, new dendrite group cold."""
    source, optimizer, twin = _trained_twins()
    captured = grow.capture_optimizer_state_by_name(source, optimizer)
    twin.register_parameter("dendrite", nn.Parameter(torch.zeros(3)))
    base = [parameter for name, parameter in twin.named_parameters() if name != "dendrite"]
    grouped = _sgd([{"params": base}, {"params": [twin.dendrite], "weight_decay": 0.0}])

    assert grow.restore_optimizer_state_by_name(twin, grouped, captured) == 4
    assert twin.dendrite not in grouped.state
    for name, parameter in twin.named_parameters():
        if name != "dendrite":
            assert torch.equal(
                grouped.state[parameter]["momentum_buffer"], captured[name]["momentum_buffer"]
            )


def test_optimizer_numel_counts_each_requested_parameter_once():
    model = _mlp()
    optimizer = _sgd([{"params": [model[0].weight, model[0].bias]}, {"params": [model[2].weight]}])
    optimizer.param_groups[1]["params"].append(model[0].weight)  # a duplicate reference
    wanted = [model[0].weight, model[2].weight, model[2].bias]
    assert grow.optimizer_numel(optimizer, wanted) == 12 + 6


# --------------------------------------------------------------------------
# summarize_history


BASE_ACCURACY = {
    1: 0.10, 2: 0.30, 3: 0.25, 4: 0.40, 5: 0.35, 6: 0.38,
    7: 0.50, 8: 0.62, 9: 0.55, 10: 0.62, 11: 0.58, 12: 0.60,
}
# Deliberately above every neuron epoch: candidate epochs must not count.
CANDIDATE_ACCURACY = [0.99, 0.95]


def _history(config: grow.GrowConfig, last_epoch: int | None = None) -> list[dict]:
    candidate = iter(CANDIDATE_ACCURACY)
    records = []
    for epoch in range(1, (last_epoch or config.total_epochs) + 1):
        base_epoch = config.base_epoch(epoch)
        records.append({
            "epoch": epoch,
            "segment": config.segment(epoch),
            "base_epoch": base_epoch,
            "val_acc": BASE_ACCURACY[base_epoch] if base_epoch is not None else next(candidate),
        })
    return records


def test_summarize_history_of_a_complete_run():
    config = _grow_config(epochs=12, switch_epoch=6, candidate_epochs=2)

    results = grow.summarize_history(_history(config), config)

    assert results == {
        "val_acc_at_switch": 0.38,
        "best_val_acc_pre_switch": 0.40,
        "best_val_acc_post_switch": 0.62,
        # base epochs 8 and 10 tie; the earliest wins.
        "best_base_epoch_post_switch": 8,
        "final_val_acc": 0.60,
        "last5_mean_val_acc": pytest.approx((0.62 + 0.55 + 0.62 + 0.58 + 0.60) / 5),
        "last10_mean_val_acc": pytest.approx(
            sum(BASE_ACCURACY[epoch] for epoch in range(3, 13)) / 10
        ),
        "best_val_acc_overall": 0.62,
        "candidate_val_acc_span": pytest.approx(0.04),
    }


def test_summarize_history_of_a_run_stopped_after_the_candidate_phase():
    config = _grow_config(epochs=12, switch_epoch=6, candidate_epochs=2)

    results = grow.summarize_history(_history(config, last_epoch=11), config)  # base 9

    assert results["best_val_acc_post_switch"] == 0.62
    assert results["best_base_epoch_post_switch"] == 8
    assert results["final_val_acc"] is None
    assert results["last5_mean_val_acc"] is None
    assert results["last10_mean_val_acc"] is None
    assert results["candidate_val_acc_span"] == pytest.approx(0.04)


def test_summarize_history_of_a_run_stopped_before_the_switch():
    config = _grow_config(epochs=12, switch_epoch=6, candidate_epochs=2)

    results = grow.summarize_history(_history(config, last_epoch=4), config)

    assert results["val_acc_at_switch"] is None
    assert results["best_val_acc_pre_switch"] == 0.40
    assert results["best_val_acc_overall"] == 0.40
    assert results["best_val_acc_post_switch"] is None
    assert results["best_base_epoch_post_switch"] is None
    assert results["final_val_acc"] is None
    assert results["candidate_val_acc_span"] is None


def test_summarize_history_of_nothing_is_all_none():
    config = _grow_config(epochs=12, switch_epoch=6, candidate_epochs=2)
    assert set(grow.summarize_history([], config).values()) == {None}


def test_summary_windows_need_enough_base_epochs():
    config = _grow_config(epochs=4, switch_epoch=2, candidate_epochs=1)
    history = [
        {"epoch": epoch, "segment": config.segment(epoch),
         "base_epoch": config.base_epoch(epoch), "val_acc": 0.5}
        for epoch in range(1, config.total_epochs + 1)
    ]
    results = grow.summarize_history(history, config)
    assert results["final_val_acc"] == 0.5
    assert results["last5_mean_val_acc"] is None
    assert results["last10_mean_val_acc"] is None


# --------------------------------------------------------------------------
# existing_run_reason


def test_a_missing_or_empty_output_dir_can_host_a_run(tmp_path):
    assert grow.existing_run_reason(tmp_path / "missing") is None
    (tmp_path / "empty").mkdir()
    assert grow.existing_run_reason(tmp_path / "empty") is None


def test_a_fresh_layout_with_empty_output_dirs_can_host_a_run(tmp_path):
    layout = ArtifactLayout(tmp_path / "run").ensure_tree()
    (layout.root / "metrics" / grow.STAGE).mkdir()
    # Unrelated output does not block a fresh run either.
    (layout.root / "metrics" / "train").mkdir()
    (layout.root / "metrics" / "train" / "scratch.jsonl").write_text("{}\n")
    (layout.root / "logs" / "earlier.log").write_text("log\n")

    assert grow.existing_run_reason(layout.root) is None


def test_a_written_summary_blocks_a_rerun(tmp_path):
    summary = tmp_path / "reports" / grow.SUMMARY_NAME
    summary.parent.mkdir(parents=True)
    summary.write_text("status: complete\n")

    reason = grow.existing_run_reason(tmp_path)
    assert reason is not None and grow.SUMMARY_NAME in reason


@pytest.mark.parametrize(
    "leftover",
    ["pai/candidates/sparknet_c4_fc/latest.pt", "metrics/grow/sparknet_c4.jsonl"],
)
def test_partial_output_blocks_a_rerun(tmp_path, leftover):
    path = tmp_path / leftover
    path.parent.mkdir(parents=True)
    path.write_text("partial\n")

    reason = grow.existing_run_reason(tmp_path)
    assert reason is not None and "earlier attempt" in reason


# --------------------------------------------------------------------------
# load_weights_for_export


class _WithTracker(nn.Module):
    """A parameterized module plus a PAI-style variable-length tracker buffer."""

    def __init__(self, tracker_length: int):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.norm = nn.BatchNorm1d(2)
        self.register_buffer("tracker_string", torch.zeros(tracker_length, dtype=torch.uint8))


def _saved_state() -> dict[str, torch.Tensor]:
    saved = _WithTracker(tracker_length=9)
    with torch.no_grad():
        saved.linear.weight.fill_(1.0)
        saved.linear.bias.fill_(-1.0)
        saved.norm.running_mean.fill_(0.5)
    return {name: tensor.clone() for name, tensor in saved.state_dict().items()}


def test_export_reload_skips_a_resized_tracker_buffer():
    live = _WithTracker(tracker_length=4)
    state = _saved_state()
    state["stale.extra"] = torch.ones(2)

    skipped = grow.load_weights_for_export(live, state)

    assert skipped == ["stale.extra", "tracker_string"]
    assert torch.equal(live.linear.weight, torch.ones(2, 3))
    assert torch.equal(live.linear.bias, -torch.ones(2))
    assert torch.equal(live.norm.running_mean, torch.full((2,), 0.5))
    assert live.tracker_string.shape == (4,)


@pytest.mark.parametrize("damage", ["reshape", "drop"])
def test_export_reload_refuses_to_mix_epochs(damage):
    live = _WithTracker(tracker_length=4)
    before = {name: tensor.clone() for name, tensor in live.state_dict().items()}
    state = _saved_state()
    if damage == "reshape":
        state["linear.bias"] = torch.zeros(5)
    else:
        del state["linear.bias"]

    with pytest.raises(grow.GrowInvariantError, match="linear.bias"):
        grow.load_weights_for_export(live, state)
    # All or nothing: the live graph is untouched.
    assert all(torch.equal(before[name], tensor) for name, tensor in live.state_dict().items())


# --------------------------------------------------------------------------
# base_state_snapshot / max_abs_change


class _Wrapped(nn.Module):
    """Mimics PAI's wrapper naming: the original module lives in main_module."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.main_module = module


class _PaiLikeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = _Wrapped(nn.Linear(3, 2))
        self.bn = _Wrapped(nn.BatchNorm1d(2))
        self.head = nn.Linear(2, 2)  # not under a wrapper: not base state


def test_snapshot_needs_pai_wrapped_modules():
    with pytest.raises(grow.GrowInvariantError, match="main_module"):
        grow.base_state_snapshot(_mlp())


def test_snapshot_holds_floating_base_tensors_only():
    net = _PaiLikeNet()
    snapshot = grow.base_state_snapshot(net)

    assert set(snapshot) == {
        "fc.main_module.weight", "fc.main_module.bias",
        "bn.main_module.weight", "bn.main_module.bias",
        "bn.main_module.running_mean", "bn.main_module.running_var",
    }
    assert grow.max_abs_change(snapshot, net) == 0.0


def test_max_abs_change_sees_weights_and_batchnorm_statistics():
    net = _PaiLikeNet()
    with torch.no_grad():
        net.fc.main_module.weight.zero_()
    snapshot = grow.base_state_snapshot(net)

    with torch.no_grad():
        net.head.weight.add_(10.0)  # outside the base: ignored
    assert grow.max_abs_change(snapshot, net) == 0.0

    with torch.no_grad():
        net.fc.main_module.weight[0, 0] = 0.25
    assert grow.max_abs_change(snapshot, net) == 0.25

    net.bn.main_module.running_mean[1] = -0.75
    assert grow.max_abs_change(snapshot, net) == 0.75


def test_max_abs_change_refuses_missing_or_reshaped_base_tensors():
    net = _PaiLikeNet()
    snapshot = grow.base_state_snapshot(net)

    net.fc.main_module = nn.Linear(3, 4)
    with pytest.raises(grow.GrowInvariantError, match="changed shape"):
        grow.max_abs_change(snapshot, net)

    net.fc = nn.Linear(3, 2)  # the wrapper is gone
    with pytest.raises(grow.GrowInvariantError, match="disappeared"):
        grow.max_abs_change(snapshot, net)


# --------------------------------------------------------------------------
# build_post_switch_optimizer / zero_and_freeze_dendrite / skip weights


class _DendriteModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 2)])  # the integrated dendrite
        self.parent_module = nn.Linear(3, 2)  # PAI's copy of the base: base


class _IntegratedNet(nn.Module):
    """PAI's post-integration naming for one wrapped module."""

    def __init__(self):
        super().__init__()
        self.fc = _Wrapped(nn.Linear(3, 2))
        self.fc.dendrite_module = _DendriteModule()
        self.fc.dendrites_to_top = nn.ParameterList([nn.Parameter(torch.tensor([[0.5, -0.75]]))])

    def dendrite_names(self) -> set[str]:
        return {
            "fc.dendrite_module.layers.0.weight", "fc.dendrite_module.layers.0.bias",
            "fc.dendrites_to_top.0",
        }


def test_the_fake_integrated_net_splits_like_pai():
    net = _IntegratedNet()
    base, dendrite = grow.split_base_and_dendrite_parameters(net)
    names = {id(parameter): name for name, parameter in net.named_parameters()}
    assert {names[id(parameter)] for parameter in dendrite} == net.dendrite_names()
    assert len(base) == 4  # main_module and parent_module weight + bias


@pytest.mark.parametrize("override", [None, 0.0, 0.05])
def test_the_weight_decay_override_reaches_the_dendrite_group(override):
    train_cfg = _paper_train_cfg()
    config = grow.resolve_grow_config(train_cfg, dendrite_weight_decay=override)
    net = _IntegratedNet()

    optimizer, base, dendrite = grow.build_post_switch_optimizer(net, train_cfg, config)

    assert isinstance(optimizer, torch.optim.SGD)
    base_group, dendrite_group = optimizer.param_groups
    assert base_group["params"] == base and dendrite_group["params"] == dendrite
    assert base_group["weight_decay"] == train_cfg["weight_decay"] == 0.001
    expected = train_cfg["grow_dendrites"]["dendrite_weight_decay"] if override is None else override
    assert dendrite_group["weight_decay"] == expected
    assert base_group["momentum"] == dendrite_group["momentum"] == train_cfg["momentum"]
    assert grow.optimizer_numel(optimizer, dendrite) == 2 * 3 + 2 + 2


def test_a_sham_post_switch_optimizer_holds_the_base_only():
    train_cfg = _paper_train_cfg()
    config = grow.resolve_grow_config(train_cfg, sham=True, dendrite_weight_decay=0.05)
    net = _IntegratedNet()

    optimizer, base, dendrite = grow.build_post_switch_optimizer(net, train_cfg, config)

    assert [group["params"] for group in optimizer.param_groups] == [base]
    assert optimizer.param_groups[0]["weight_decay"] == train_cfg["weight_decay"]
    assert grow.optimizer_numel(optimizer, dendrite) == 0
    assert len(dendrite) == 3


def test_zero_and_freeze_touches_the_dendrite_side_only():
    net = _IntegratedNet()
    base_before = {
        name: parameter.detach().clone() for name, parameter in net.named_parameters()
        if name not in net.dendrite_names()
    }

    assert grow.zero_and_freeze_dendrite(net) == (3, 2 * 3 + 2 + 2)

    for name, parameter in net.named_parameters():
        if name in net.dendrite_names():
            assert not parameter.requires_grad, name
            assert torch.count_nonzero(parameter) == 0, name
        else:
            assert parameter.requires_grad, name
            assert torch.equal(parameter, base_before[name]), name
    assert grow.dendrite_skip_weight_max_abs(net) == 0.0


def test_zeroing_a_network_without_a_dendrite_is_an_error():
    with pytest.raises(grow.GrowInvariantError, match="no dendrite parameters"):
        grow.zero_and_freeze_dendrite(_mlp())


def test_skip_weight_max_abs_reads_every_dendrites_to_top_entry():
    net = _IntegratedNet()
    assert grow.dendrite_skip_weight_max_abs(net) == 0.75
    with torch.no_grad():
        net.fc.dendrites_to_top[0].zero_()
    assert grow.dendrite_skip_weight_max_abs(net) == 0.0
    with pytest.raises(grow.GrowInvariantError, match="vacuous"):
        grow.dendrite_skip_weight_max_abs(_mlp())


# --------------------------------------------------------------------------
# integration_probe / probe_logits: measuring without perturbing


class _Noisy(nn.Module):
    """Draws from the global RNG on every forward, like SparkNet's gate noise."""

    def forward(self, features):
        return features + 0.01 * torch.rand_like(features)


def _probe_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), _Noisy(), nn.Dropout(0.5))
    model.train()
    model[1].eval()  # a pinned base BatchNorm inside a training model
    return model


def test_probe_logits_restores_modes_rng_and_leaves_no_graph():
    model = _probe_model()
    probe = torch.randn(16, 4)
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state()
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    logits = grow.probe_logits(model, probe, torch.device("cpu"))

    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[name], tensor) for name, tensor in model.state_dict().items())
    assert not logits.requires_grad and logits.device.type == "cpu"
    assert all(parameter.grad is None for parameter in model.parameters())
    # Eval semantics (no dropout), from the RNG state the run was in.
    reference = copy.deepcopy(model).eval()
    torch.set_rng_state(rng)
    with torch.no_grad():
        assert torch.equal(logits, reference(probe))


def _probe_input_std(device: torch.device) -> tuple[dict[str, float], torch.Tensor]:
    model = _probe_model().to(device)
    probe = torch.randn(16, 4, device=device) * 7
    stds: dict[str, float] = {}
    first = next(name for name, module in model.named_modules() if isinstance(module, nn.Linear))
    grow.probe_logits(model, probe, device, input_std=([f".{first}"], stds))
    return stds, probe


def test_probe_logits_reports_the_input_std_of_the_listed_modules():
    stds, probe = _probe_input_std(torch.device("cpu"))
    (name,) = stds
    assert stds[name] == pytest.approx(float(probe.double().std()), rel=1e-9)


def test_probe_logits_rejects_an_unknown_input_module():
    with pytest.raises(grow.GrowInvariantError, match="no module 'nope'"):
        grow.probe_logits(
            _probe_model(), torch.randn(3, 4), torch.device("cpu"),
            input_std=([".nope"], {}),
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_probe_input_std_works_on_a_device_without_float64():
    # MPS cannot hold float64: the std must be widened only after the copy to CPU.
    stds, probe = _probe_input_std(torch.device("mps"))
    (name,) = stds
    assert stds[name] == pytest.approx(float(probe.cpu().double().std()), rel=1e-5)


def test_probe_logits_restores_an_all_eval_model_too():
    model = _probe_model().eval()
    grow.probe_logits(model, torch.randn(3, 4), torch.device("cpu"))
    assert not any(module.training for module in model.modules())


def test_integration_probe_is_the_parity_probe_and_moves_no_generator():
    count = grow.PARITY_PROBE_EXAMPLES + 44
    features = torch.randn(count, 1, 2, 3)
    dataset = TensorDataset(features, torch.zeros(count, dtype=torch.long))
    generator = torch.Generator().manual_seed(7)
    loader = DataLoader(dataset, batch_size=100, generator=generator)
    generator_state, rng = generator.get_state(), torch.get_rng_state()

    probe = grow.integration_probe(loader, torch.device("cpu"))

    assert torch.equal(probe, features[: grow.PARITY_PROBE_EXAMPLES])
    assert torch.equal(generator.get_state(), generator_state)
    assert torch.equal(torch.get_rng_state(), rng)
    # Not vacuous: starting an iterator does draw from the loader's generator.
    iter(loader)
    assert not torch.equal(generator.get_state(), generator_state)


# --------------------------------------------------------------------------
# diagnose_exports


def _fake_diagnostics(on_by_model):
    def diagnostics(model, loader, device):
        return {
            "val_acc_dendrite_on": on_by_model[id(model)],
            "val_acc_dendrite_off": 0.5,
            "n_samples": 8,
            "modules": {"fc": {"n_dendrites": 1}},
        }
    return diagnostics


def test_diagnose_exports_records_each_model_and_its_agreement(monkeypatch):
    final, best = nn.Identity(), nn.Identity()
    monkeypatch.setattr(
        grow, "dendrite_diagnostics", _fake_diagnostics({id(final): 0.75, id(best): 0.625})
    )
    checks = {"existing": 1}

    diagnostics = grow.diagnose_exports(
        {"final": final, "best": best}, [], torch.device("cpu"),
        {"final_val_acc": 0.75, "best_val_acc_post_switch": 0.875}, checks,
    )

    assert diagnostics == {
        "final": {
            "val_acc_dendrite_on": 0.75, "val_acc_dendrite_off": 0.5, "n_samples": 8,
            "modules": {"fc": {"n_dendrites": 1}}, "device": "cpu",
        },
        "best": {
            "val_acc_dendrite_on": 0.625, "val_acc_dendrite_off": 0.5, "n_samples": 8,
            "modules": {"fc": {"n_dendrites": 1}}, "device": "cpu",
        },
    }
    assert checks == {
        "existing": 1,
        "diagnostics_val_acc_on_minus_final": 0.0,
        "diagnostics_val_acc_on_minus_best": -0.25,
    }


def test_diagnose_exports_skips_the_agreement_without_a_reference(monkeypatch):
    final = nn.Identity()
    monkeypatch.setattr(grow, "dendrite_diagnostics", _fake_diagnostics({id(final): 0.75}))
    checks: dict = {}
    diagnostics = grow.diagnose_exports(
        {"final": final}, [], torch.device("cpu"), {"final_val_acc": None}, checks
    )
    assert set(diagnostics) == {"final"}
    assert checks == {}


@pytest.mark.parametrize("fail_on", ["final", "best"])
def test_diagnose_exports_reports_a_failure_instead_of_raising(monkeypatch, fail_on):
    models = {"final": nn.Identity(), "best": nn.Identity()}
    working = _fake_diagnostics({id(model): 0.5 for model in models.values()})

    def diagnostics(model, loader, device):
        if model is models[fail_on]:
            raise RuntimeError("no clean PAI dendrite modules")
        return working(model, loader, device)

    monkeypatch.setattr(grow, "dendrite_diagnostics", diagnostics)
    checks = {"existing": 1}

    result = grow.diagnose_exports(
        models, [], torch.device("cpu"),
        {"final_val_acc": 0.5, "best_val_acc_post_switch": 0.5}, checks,
    )

    assert result == {"error": "RuntimeError: no clean PAI dendrite modules"}
    assert checks == {"existing": 1}  # no partial agreement checks


# --------------------------------------------------------------------------
# main(): validation and refusal happen before any output is written


def _cli_args(output_dir: Path, *extra: str) -> list[str]:
    return [
        "--data-config", str(ROOT / "configs/data/speech_commands_v2_mfcc32_paper.yaml"),
        "--model-config", str(ROOT / "configs/model/sparknet_c4_paper.yaml"),
        "--train-config", str(GROW_TRAIN_CONFIG),
        "--output-dir", str(output_dir),
        *extra,
    ]


def test_cli_refuses_a_directory_holding_an_earlier_attempt(tmp_path, capsys):
    output_dir = tmp_path / "run"
    leftover = output_dir / "metrics" / grow.STAGE / "sparknet_c4_paper.jsonl"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("{}\n")

    assert grow.main(_cli_args(output_dir)) == grow.EXIT_REFUSED
    assert "refusing to start" in capsys.readouterr().err
    assert not (output_dir / "manifest.yaml").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--switch-epoch", "200"], "switch_epoch must be in"),
        (["--placement", "everywhere"], "unknown placement"),
        (["--max-minutes", "0"], "max_wall_clock_minutes"),
        (["--seed", "-1"], "--seed must be non-negative"),
        (["--dendrite-weight-decay", "-0.5"], "dendrite_weight_decay"),
        (["--dendrite-weight-decay", "nan"], "dendrite_weight_decay"),
        (["--dendrite-weight-decay", "lots"], "invalid float value"),
        (["--dendrite-input-scale", "0"], "dendrite_input_scale"),
        (["--dendrite-input-scale", "-80"], "dendrite_input_scale"),
        (["--dendrite-input-scale", "inf"], "dendrite_input_scale"),
        (["--arm", ""], "arm must be a non-empty name"),
        (["--arm", "fc sham"], "arm must be a non-empty name"),
        (["--sham", "--arm", "../fc"], "arm must be a non-empty name"),
        (["--group-batchnorm"], "needs 'pointwise' placements"),
        (
            ["--placement", "pointwise_b2", "--group-batchnorm", "--calibrate-dendrite-input-scale"],
            "leave it at 1",
        ),
    ],
)
def test_cli_rejects_bad_overrides_before_writing_output(tmp_path, capsys, extra, message):
    output_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as exit_info:
        grow.main(_cli_args(output_dir, *extra))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not output_dir.exists()


def test_cli_variant_flags_default_to_a_real_run():
    args = grow.build_arg_parser().parse_args(["--model-config", "m.yaml", "--output-dir", "o"])
    assert (args.sham, args.arm, args.dendrite_weight_decay) == (False, None, None)
    assert args.dendrite_input_scale is None
    assert args.calibrate_dendrite_input_scale is False
    assert args.group_batchnorm is False

    args = grow.build_arg_parser().parse_args([
        "--model-config", "m.yaml", "--output-dir", "o",
        "--sham", "--arm", "fc-x", "--dendrite-weight-decay", "1e-3",
    ])
    assert (args.sham, args.arm, args.dendrite_weight_decay) == (True, "fc-x", 1e-3)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], (False, "fc", 0.0)),
        (["--sham"], (True, "fc-sham", 0.0)),
        (["--placement", "pointwise", "--sham"], (True, "pointwise-sham", 0.0)),
        (["--dendrite-weight-decay", "5e-4"], (False, "fc", 5e-4)),
        (["--sham", "--arm", "floor.1", "--dendrite-weight-decay", "0.01"], (True, "floor.1", 0.01)),
    ],
)
def test_cli_variant_flags_reach_the_run(tmp_path, monkeypatch, extra, expected):
    """main() hands the resolved variant to run_grow (stubbed: no training)."""
    received = []

    def fake_run_grow(data_cfg, model_cfg, train_cfg, config, **kwargs):
        received.append(config)
        return {}

    monkeypatch.setattr(grow, "run_grow", fake_run_grow)
    assert grow.main(_cli_args(tmp_path / "run", *extra)) == 0

    (config,) = received
    assert (config.sham, config.arm, config.dendrite_weight_decay) == expected
    assert config.variant()["sham"] is expected[0]
    assert config.variant()["dendrite_weight_decay"] == expected[2]
    assert config.variant()["dendrite_input_scale"] == 1.0


def test_cli_input_scale_reaches_the_run(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(
        grow, "run_grow", lambda *args, **kwargs: received.append(args[3]) or {}
    )
    extra = ["--placement", "pointwise", "--dendrite-input-scale", "75", "--arm", "pw-in75"]
    assert grow.main(_cli_args(tmp_path / "run", *extra)) == 0

    (config,) = received
    assert (config.placement, config.arm, config.dendrite_input_scale) == ("pointwise", "pw-in75", 75.0)
    assert config.variant()["dendrite_input_scale"] == 75.0


def test_cli_switch_input_scale_calibration_reaches_the_run(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(
        grow, "run_grow", lambda *args, **kwargs: received.append(args[3]) or {}
    )
    extra = [
        "--placement", "pointwise_b2",
        "--calibrate-dendrite-input-scale",
        "--arm", "pointwise_b2-auto",
    ]

    assert grow.main(_cli_args(tmp_path / "run", *extra)) == 0

    (config,) = received
    assert config.calibrate_dendrite_input_scale is True
    assert config.variant()["dendrite_input_scale_calibration"] == "switch_input_std"


def test_cli_group_batchnorm_reaches_the_run(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(
        grow, "run_grow", lambda *args, **kwargs: received.append(args[3]) or {}
    )
    extra = ["--placement", "pointwise_b2", "--group-batchnorm"]

    assert grow.main(_cli_args(tmp_path / "run", *extra)) == 0

    (config,) = received
    assert (config.group_batchnorm, config.arm) == (True, "pointwise_b2-bn")
    assert config.variant()["group_batchnorm"] is True


# --------------------------------------------------------------------------
# Real PerforatedAI runs, one fresh interpreter each.

TINY_MODEL = {
    "family": "sparknet", "name": "sparknet_c4_test",
    "channels": 4, "gate_channels": 8, "sparsity_weight": 1.0,
}
TRAIN_EXAMPLES, VAL_EXAMPLES, NUM_CLASSES, BATCH_SIZE = 96, 64, 12, 16
STEPS_PER_EPOCH = TRAIN_EXAMPLES // BATCH_SIZE
EPOCHS, CANDIDATE_EPOCHS = 4, 2
PARITY_TOLERANCE = 1e-5
PRE, CANDIDATE, POST = "pre_switch", "candidate", "post_switch"
SUBPROCESS_TIMEOUT_SECONDS = 600

# Runs inside the subprocess.  The spec (argv[1]) carries every setting so the
# test and the run cannot disagree; the harness only patches seams.
_HARNESS = r'''
import json
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

spec = json.loads(Path(sys.argv[1]).read_text())
result = {"raised": None}


def synthetic_datasets(data_cfg=None, augment=False, seed=0, **_options):
    """(features[1, 32, 101], label) pairs; the real dataset is never touched."""
    from kws.data.splits import TRAIN, VAL

    generator = torch.Generator().manual_seed(1234)

    def split(count):
        features = torch.randn(count, 1, 32, 101, generator=generator)
        return TensorDataset(features, torch.arange(count) % spec["num_classes"])

    datasets = {TRAIN: split(spec["train_examples"]), VAL: split(spec["val_examples"])}
    return datasets, {f"word{index}": index for index in range(spec["num_classes"])}


def run_plain():
    """The paired control: the scratch entry point kws.train.train, unmodified."""
    import kws.train as scratch

    scratch.build_datasets = synthetic_datasets
    scratch.get_device = lambda: torch.device("cpu")
    trained = scratch.train(
        {"target_keywords": [f"word{index}" for index in range(10)]},
        spec["model_cfg"], spec["train_cfg"], Path("best.pt"),
        output_dir=spec["output_dir"],
    )
    result["history"] = trained.history
    result["pai_modules_loaded"] = sorted(
        name for name in sys.modules if name.startswith("perforated")
    )


def script_validation(grow):
    """Override chosen epochs' val_acc; optionally swallow the best-score reset.

    ``evaluate_loss_acc`` runs once per wall-clock epoch, so its call count is
    the epoch.  The reset guard drops the driver's two zero writes at the
    switch epoch and nothing else, to show the rewind check is live.
    """
    script = {int(epoch): acc for epoch, acc in (spec.get("val_acc_script") or {}).items()}
    swallow_at = spec.get("swallow_best_reset_at")
    real_evaluate = grow.evaluate_loss_acc
    calls = [0]

    class ResetGuard(dict):
        armed = set()
        swallowed = []

        def __setitem__(self, key, value):
            if key in ResetGuard.armed and value == 0:
                ResetGuard.armed.discard(key)
                ResetGuard.swallowed.append(key)
                return
            super().__setitem__(key, value)

    def evaluate(*args, **kwargs):
        calls[0] += 1
        loss, accuracy = real_evaluate(*args, **kwargs)
        if calls[0] == swallow_at:
            tracker = grow.GPA.pai_tracker
            tracker.member_vars = ResetGuard(tracker.member_vars)
            ResetGuard.armed = {"current_best_validation_score", "global_best_validation_score"}
        return loss, script.get(calls[0], accuracy)

    grow.evaluate_loss_acc = evaluate
    return ResetGuard


def probe_momentum(grow, first_post_switch_epoch):
    """Watch the first post-switch SGD step through PAI's step wrapper.

    Records whether each captured base parameter enters that step holding its
    captured momentum buffer, and whether the step folds that buffer in
    (buf = momentum * captured + grad + wd * param) rather than starting cold.
    """
    probe = result["momentum_probe"] = {"checked": 0, "restored": 0, "carried": 0}
    captured = {}
    real_capture = grow.capture_optimizer_state_by_name
    real_train_epoch = grow._train_epoch
    epochs = [0]

    def capture(model, optimizer):
        state = real_capture(model, optimizer)
        captured.update({
            name: {key: value.clone() for key, value in entry.items()}
            for name, entry in state.items()
        })
        return state

    def train_epoch(model, loader, optimizer, *args, **kwargs):
        epochs[0] += 1
        if epochs[0] == first_post_switch_epoch:
            names = {id(parameter): name for name, parameter in model.named_parameters()}
            real_step = optimizer.step

            def first_step(*step_args, **step_kwargs):
                optimizer.step = real_step
                pending = []
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        name = names[id(parameter)]
                        if name in captured and parameter.grad is not None:
                            buffer = optimizer.state[parameter].get("momentum_buffer")
                            pending.append((
                                name, group, parameter,
                                None if buffer is None else buffer.clone(),
                                parameter.detach().clone(), parameter.grad.detach().clone(),
                            ))
                outcome = real_step(*step_args, **step_kwargs)
                for name, group, parameter, before, value, grad in pending:
                    saved = captured[name]["momentum_buffer"]
                    cold = grad.add(value, alpha=group["weight_decay"])
                    warm = saved.mul(group["momentum"]).add(cold)
                    after = optimizer.state[parameter]["momentum_buffer"]
                    probe["checked"] += 1
                    probe["restored"] += int(before is not None and torch.equal(before, saved))
                    probe["carried"] += int(
                        torch.allclose(after, warm, rtol=1e-6, atol=1e-10)
                        and not torch.allclose(after, cold, rtol=1e-6, atol=1e-10)
                    )
                return outcome

            optimizer.step = first_step
        return real_train_epoch(model, loader, optimizer, *args, **kwargs)

    grow.capture_optimizer_state_by_name = capture
    grow._train_epoch = train_epoch


def record_post_switch_groups(grow, first_post_switch_epoch):
    """The optimizer groups the first post-switch epoch really trains with."""
    real_train_epoch = grow._train_epoch
    epochs = [0]

    def train_epoch(model, loader, optimizer, *args, **kwargs):
        epochs[0] += 1
        if epochs[0] == first_post_switch_epoch:
            result["post_switch_groups"] = [
                {
                    "weight_decay": group["weight_decay"],
                    "numel": sum(parameter.numel() for parameter in group["params"]),
                }
                for group in optimizer.param_groups
            ]
        return real_train_epoch(model, loader, optimizer, *args, **kwargs)

    grow._train_epoch = train_epoch


def record_final_dendrite(grow):
    """Dendrite-side parameters of the live graph when training has ended.

    The first export is of the last-epoch graph, before the best epoch's
    weights are reloaded for the second one.
    """
    real_export = grow._export_with_parity

    def export(model, *args, **kwargs):
        if "final_dendrite" not in result:
            _base, dendrite = grow.split_base_and_dendrite_parameters(model)
            result["final_dendrite"] = {
                "tensors": len(dendrite),
                "numel": sum(parameter.numel() for parameter in dendrite),
                "max_abs": max(float(parameter.detach().abs().max()) for parameter in dendrite),
                "any_requires_grad": any(parameter.requires_grad for parameter in dendrite),
            }
        return real_export(model, *args, **kwargs)

    grow._export_with_parity = export


def disable_integration_probe(grow):
    """Skip the probe entirely: the control for 'the probe perturbs nothing'."""
    grow.integration_probe = lambda val_loader, device: torch.zeros(1)
    grow.probe_logits = lambda model, probe, device, **_options: torch.zeros(1)


def check_rebuild(grow):
    """Rebuild each saved export under plain tanh and compare it with the live graph.

    The live graph runs under the run's own (possibly scaled) forward function;
    the file must reproduce it with PAI nowhere in the loop.  A tiny run's skip
    weights barely leave zero, which would hide a wrong dendrite branch, so
    they are set to 2 first: the dendrite then carries real weight.
    """
    from kws.optimize.grow_clean_rebuild import load_clean_state, rebuild_clean_model

    real_export = grow._export_with_parity
    result["rebuild"] = []

    def export(model, run_dir, save_dir, probe, *args, **kwargs):
        rng = torch.random.get_rng_state()
        model.eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "dendrites_to_top" in name:
                    parameter.fill_(2.0)
            reference = model(probe).detach().cpu()
        forward_function = repr(grow.GPA.pc.get_pai_forward_function())
        outcome = real_export(model, run_dir, save_dir, probe, *args, **kwargs)
        state, metadata = load_clean_state(Path(save_dir) / "final_clean_pai.pt")
        rebuilt = rebuild_clean_model(
            spec["model_cfg"], (32, 101), spec["num_classes"], state, forward_function=torch.tanh
        ).eval()
        with torch.no_grad():
            difference = float((rebuilt(probe.cpu()) - reference).abs().max())
        result["rebuild"].append({
            "forward_function": forward_function,
            "folded": metadata.get("dendrite_input_scale_folded"),
            "max_abs_diff": difference,
        })
        torch.random.set_rng_state(rng)
        return outcome

    grow._export_with_parity = export


def run_grow():
    import kws.optimize.sparknet_grow_dendrites as grow

    grow.build_datasets = synthetic_datasets
    grow.get_device = lambda: torch.device("cpu")
    config = grow.resolve_grow_config(
        spec["train_cfg"], placement=spec["placement"], max_minutes=spec["max_minutes"],
        dendrite_weight_decay=spec["dendrite_weight_decay"], sham=spec["sham"], arm=spec["arm"],
        dendrite_input_scale=spec["dendrite_input_scale"],
        calibrate_dendrite_input_scale=spec["calibrate_dendrite_input_scale"],
        group_batchnorm=spec["group_batchnorm"],
    )
    guard = script_validation(grow)
    if spec["probe_momentum"]:
        probe_momentum(grow, config.switch_epoch + config.candidate_epochs + 1)
    record_post_switch_groups(grow, config.switch_epoch + config.candidate_epochs + 1)
    record_final_dendrite(grow)
    if spec["check_rebuild"]:
        check_rebuild(grow)
    if spec["disable_integration_probe"]:
        disable_integration_probe(grow)
    try:
        grow.run_grow({}, spec["model_cfg"], spec["train_cfg"], config,
                      output_dir=spec["output_dir"])
    except Exception as exc:
        result.update(raised=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc())
    result["best_reset_swallowed"] = sorted(guard.swallowed)


run_plain() if spec["mode"] == "plain" else run_grow()
Path(spec["result_path"]).write_text(json.dumps(result))
'''


def _tiny_train_cfg(switch_epoch: int) -> dict:
    train_cfg = _paper_train_cfg()
    train_cfg.update(epochs=EPOCHS, batch_size=BATCH_SIZE, num_workers=0, persistent_workers=False)
    train_cfg.pop("prefetch_factor")
    train_cfg["grow_dendrites"].update(
        switch_epoch=switch_epoch, candidate_epochs=CANDIDATE_EPOCHS
    )
    train_cfg["perforatedai"]["initial_correlation_batches"] = 2
    return train_cfg


def _launch(workdir: Path, options: dict) -> SimpleNamespace:
    cwd = workdir / "cwd"
    cwd.mkdir()
    spec = {
        "mode": options["mode"],
        "placement": options.get("placement", "fc"),
        "max_minutes": options.get("max_minutes"),
        "val_acc_script": options.get("val_acc_script"),
        "swallow_best_reset_at": options.get("swallow_best_reset_at"),
        "probe_momentum": options.get("probe_momentum", False),
        "sham": options.get("sham", False),
        "arm": options.get("arm"),
        "dendrite_weight_decay": options.get("dendrite_weight_decay"),
        "disable_integration_probe": options.get("disable_integration_probe", False),
        "dendrite_input_scale": options.get("dendrite_input_scale"),
        "calibrate_dendrite_input_scale": options.get("calibrate_dendrite_input_scale", False),
        "group_batchnorm": options.get("group_batchnorm", False),
        "check_rebuild": options.get("check_rebuild", False),
        "train_cfg": _tiny_train_cfg(options.get("switch_epoch", 2)),
        "model_cfg": TINY_MODEL,
        "train_examples": TRAIN_EXAMPLES,
        "val_examples": VAL_EXAMPLES,
        "num_classes": NUM_CLASSES,
        "output_dir": str(workdir / "run"),
        "result_path": str(workdir / "result.json"),
    }
    spec_path = workdir / "spec.json"
    spec_path.write_text(json.dumps(spec))
    completed = subprocess.run(
        [sys.executable, "-c", _HARNESS, str(spec_path)],
        cwd=cwd,  # empty, so anything PAI writes outside the run dir shows up
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,  # a license prompt must fail, not hang
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,  # the assertion below reports PAI's output on failure
    )
    assert completed.returncode == 0, (
        f"harness failed ({completed.returncode})\n--- stdout ---\n"
        f"{completed.stdout[-4000:]}\n--- stderr ---\n{completed.stderr[-6000:]}"
    )
    return SimpleNamespace(
        spec=spec,
        cwd=cwd,
        output_dir=workdir / "run",
        result=json.loads((workdir / "result.json").read_text()),
    )


@pytest.fixture(scope="module")
def pai_run(tmp_path_factory):
    """Run (or reuse) one subprocess per distinct option set."""
    cache: dict[str, SimpleNamespace] = {}

    def run(**options) -> SimpleNamespace:
        key = json.dumps(options, sort_keys=True)
        if key not in cache:
            cache[key] = _launch(tmp_path_factory.mktemp(options["mode"]), options)
        return cache[key]

    return run


def _records(run: SimpleNamespace) -> list[dict]:
    path = run.output_dir / "metrics" / grow.STAGE / f"{TINY_MODEL['name']}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summary(run: SimpleNamespace) -> dict:
    return yaml.safe_load((run.output_dir / "reports" / grow.SUMMARY_NAME).read_text())


def _reference_lrs(train_cfg: dict, total_steps: int) -> list[float]:
    """The recipe's uninterrupted schedule: index k is the LR after k steps."""
    optimizer = build_optimizer([nn.Parameter(torch.zeros(()))], train_cfg)
    scheduler = build_lr_scheduler(
        optimizer, total_steps, train_cfg["warmup_fraction"],
        name=train_cfg["scheduler"], hold_fraction=train_cfg["hold_fraction"],
        min_lr=train_cfg["min_lr"], power=train_cfg["polynomial_power"],
    )
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


EXPECTED_TIMELINE = {
    2: ([PRE, PRE, CANDIDATE, CANDIDATE, POST, POST], [1, 2, None, None, 3, 4]),
    1: ([PRE, CANDIDATE, CANDIDATE, POST, POST, POST], [1, None, None, 2, 3, 4]),
}


@pytest.mark.parametrize(
    ("placement", "switch_epoch"),
    [
        ("fc", 2),
        ("pointwise", 2),
        # PAI never records epoch 1 as a best (its enough_time gate), so the
        # earliest legal switch is its own case for the reload at n->p.
        ("fc", 1),
    ],
)
def test_pai_grow_run_holds_every_invariant(pai_run, placement, switch_epoch):
    run = pai_run(mode="grow", placement=placement, switch_epoch=switch_epoch, probe_momentum=True)
    assert run.result["raised"] is None, run.result.get("traceback")
    train_cfg = run.spec["train_cfg"]
    config = grow.resolve_grow_config(train_cfg, placement=placement)
    summary = _summary(run)
    records = _records(run)
    checks = summary["checks"]
    base_params = summary["cost"]["base"]["params"]

    # Timeline, as recorded.
    segments, base_epochs = EXPECTED_TIMELINE[switch_epoch]
    assert [record["segment"] for record in records] == segments
    assert [record["base_epoch"] for record in records] == base_epochs
    assert [record["pai_mode"] for record in records] == [
        "p" if segment == CANDIDATE else "n" for segment in segments
    ]
    assert [record["global_step"] for record in records] == [
        STEPS_PER_EPOCH * epoch for epoch in range(1, len(records) + 1)
    ]
    assert [record["base_step"] for record in records] == [
        STEPS_PER_EPOCH * (base_epoch if base_epoch is not None else switch_epoch)
        for base_epoch in base_epochs
    ]

    # The base schedule is one uninterrupted schedule over base steps; the
    # candidate optimizer runs at its own fixed LR.
    reference = _reference_lrs(train_cfg, EPOCHS * STEPS_PER_EPOCH)
    for record in records:
        if record["segment"] == CANDIDATE:
            assert record["learning_rate"] == [config.candidate_optimizer["lr"]]
        else:
            groups = 1 if record["segment"] == PRE else 2  # base + dendrite
            assert record["learning_rate"] == [reference[record["base_step"]]] * groups
    assert summary["schedule"]["steps_per_epoch"] == STEPS_PER_EPOCH
    assert summary["schedule"]["base_lr_at_switch"] == reference[switch_epoch * STEPS_PER_EPOCH]

    # The candidate phase does not change the network's function at all.
    at_switch = records[switch_epoch - 1]
    for record in records:
        if record["segment"] == CANDIDATE:
            assert (record["val_loss"], record["val_acc"]) == (
                at_switch["val_loss"], at_switch["val_acc"]
            )
    for record in records:
        if record["segment"] == POST:
            assert record["parameter_count"] > base_params
        else:
            assert record["parameter_count"] == base_params

    # The summary's own checks.
    assert summary["status"] == "complete"
    assert summary["incomplete_reason"] is None
    assert summary["test_split_used"] is False
    assert summary["module_ids"] == list(config.module_ids)
    assert checks["n_to_p_base_max_abs_change"] == 0
    assert checks["candidate_phase_base_max_abs_drift"] == 0
    assert checks["base_params_in_optimizer_candidate_phase"] == 0
    assert checks["base_params_in_optimizer_post_switch"] == base_params
    assert checks["dendrite_params_in_optimizer_post_switch"] > 0
    assert checks["momentum_buffers_restored"] == checks["momentum_buffers_expected"] > 0
    assert checks["candidate_phase_val_acc_span"] == 0
    assert checks["clean_parity_max_abs_diff_final"] <= PARITY_TOLERANCE
    assert checks["clean_parity_max_abs_diff_best"] <= PARITY_TOLERANCE
    assert summary["dendrite"]["num_dendrites_added"] == 1
    assert summary["cost"]["deployed"]["params"] > base_params
    assert summary["results"] == grow.summarize_history(records, config)

    # The restored momentum really feeds the first post-switch step, through
    # PAI's replacement of optimizer.step.
    probe = run.result["momentum_probe"]
    assert probe["checked"] == checks["momentum_buffers_restored"]
    assert probe["restored"] == probe["checked"]
    assert probe["carried"] == probe["checked"]

    # Deployable graphs, each carrying every one-dendrite skip coefficient.
    run_dir = run.output_dir / "pai" / "candidates" / f"{TINY_MODEL['name']}_{placement}"
    for relative in ("final_clean_pai.pt", "best_dendritic/final_clean_pai.pt"):
        path = run_dir / relative
        assert path.is_file(), relative
        with safe_open(str(path), "pt") as clean:
            metadata = clean.metadata()
        assert metadata["single_dendrite_skip_weights_restored"] == str(len(config.module_ids))
        assert metadata["run_id"] == summary["run_id"]
    assert summary["artifacts"]["final_clean"] == (
        (run_dir / "final_clean_pai.pt").relative_to(run.output_dir).as_posix()
    )

    # PAI's writes stayed inside the run directory.
    assert list(run.cwd.iterdir()) == []
    assert grow.existing_run_reason(run.output_dir) is not None


def test_pai_pre_switch_epochs_are_the_paired_scratch_run(pai_run):
    """Before the switch the grow run is bit-identical to kws.train for the seed."""
    plain = pai_run(mode="plain")
    grown = pai_run(mode="grow", placement="fc", switch_epoch=2, probe_momentum=True)
    assert grown.result["raised"] is None, grown.result.get("traceback")
    # The control never loaded PAI, as the real scratch runs did not.
    assert plain.result["pai_modules_loaded"] == []
    scratch = plain.result["history"]
    records = _records(grown)
    assert len(scratch) == EPOCHS

    for epoch in (1, 2):
        grown_record, scratch_record = records[epoch - 1], scratch[epoch - 1]
        assert grown_record["segment"] == PRE
        for key in (
            "train_loss", "train_gate_sparsity", "train_accuracy",
            "val_loss", "val_acc", "learning_rate", "parameter_count",
        ):
            assert grown_record[key] == scratch_record[key], (epoch, key)

    # After the dendrite, the base LR is where scratch's would be.
    for grown_record, scratch_record in zip(records[4:], scratch[2:]):
        assert grown_record["segment"] == POST
        assert grown_record["learning_rate"] == scratch_record["learning_rate"] * 2
    assert list(plain.cwd.iterdir()) == []


def test_pai_wall_clock_cap_stops_the_run_with_an_incomplete_summary(pai_run):
    run = pai_run(mode="grow", placement="fc", switch_epoch=2, max_minutes=1e-6)

    assert run.result["raised"] == "WallClockExceeded"
    assert run.result["message"].startswith("stopped after epoch 1:")
    summary = _summary(run)
    assert summary["status"] == "incomplete"
    assert summary["incomplete_reason"].startswith("wall_clock_cap_exceeded")
    records = _records(run)
    assert [record["epoch"] for record in records] == [1]
    assert summary["results"]["best_val_acc_pre_switch"] == records[0]["val_acc"]
    assert summary["results"]["final_val_acc"] is None
    # The partial run cannot be silently reused.
    assert grow.existing_run_reason(run.output_dir) is not None


# Validation peaks at base epoch 2 and collapses at the switch (epoch 3), so an
# un-reset PAI tracker holds epoch 2 as its best and reloads it on n->p.
REWIND_SCRIPT = {"1": 0.5, "2": 0.9, "3": 0.1}


def test_pai_best_score_reset_keeps_the_switch_from_rewinding(pai_run):
    run = pai_run(mode="grow", placement="fc", switch_epoch=3, val_acc_script=REWIND_SCRIPT)

    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    assert [record["val_acc"] for record in _records(run)][:3] == [0.5, 0.9, 0.1]
    assert run.result["best_reset_swallowed"] == []
    assert summary["status"] == "complete"
    assert summary["checks"]["n_to_p_base_max_abs_change"] == 0
    assert summary["results"]["best_val_acc_pre_switch"] == 0.9
    assert summary["results"]["val_acc_at_switch"] == 0.1


def test_pai_rewind_check_catches_the_reload_without_the_reset(pai_run):
    """Mutation check: swallow the reset and PAI rewinds to epoch 2's weights."""
    run = pai_run(
        mode="grow", placement="fc", switch_epoch=3,
        val_acc_script=REWIND_SCRIPT, swallow_best_reset_at=3,
    )

    assert run.result["best_reset_swallowed"] == [
        "current_best_validation_score", "global_best_validation_score",
    ]
    assert run.result["raised"] == "GrowInvariantError"
    assert "rewound the base at the n->p switch" in run.result["message"]
    summary = _summary(run)
    assert summary["status"] == "incomplete"
    assert summary["incomplete_reason"].startswith("error: GrowInvariantError: PAI rewound")
    assert [record["epoch"] for record in _records(run)] == [1, 2, 3]


# --------------------------------------------------------------------------
# Run variants, the integration check and the dendrite diagnostics, for real.

REAL_FC = {"mode": "grow", "placement": "fc", "switch_epoch": 2, "probe_momentum": True}
# Every recorded per-epoch value that describes training, not wall time.
TRAJECTORY_KEYS = (
    "epoch", "segment", "pai_mode", "base_epoch", "global_step", "base_step",
    "train_loss", "train_gate_sparsity", "train_accuracy", "val_loss", "val_acc",
    "learning_rate", "parameter_count",
)
DIAGNOSTICS_KEYS = {
    "val_acc_dendrite_on", "val_acc_dendrite_off", "n_samples", "modules", "device",
}


def _trajectory(records: list[dict]) -> list[tuple]:
    return [tuple(record[key] for key in TRAJECTORY_KEYS) for record in records]


def _post_switch_groups(run: SimpleNamespace) -> list[tuple[float, int]]:
    """(weight_decay, numel) of each group the first post-switch epoch used."""
    return [(group["weight_decay"], group["numel"]) for group in run.result["post_switch_groups"]]


def _recipe_weight_decay(run: SimpleNamespace) -> float:
    return run.spec["train_cfg"]["weight_decay"]


@pytest.mark.parametrize("placement", ["fc", "pointwise"])
def test_pai_real_run_records_the_integration_check_and_diagnostics(pai_run, placement):
    run = pai_run(**{**REAL_FC, "placement": placement})
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    checks = summary["checks"]
    config = grow.resolve_grow_config(run.spec["train_cfg"], placement=placement)
    base_params = summary["cost"]["base"]["params"]

    assert summary["arm"] == placement
    assert summary["variant"] == {
        "sham": False, "dendrite_weight_decay": 0.0,
        "switch_epoch": 2, "candidate_epochs": CANDIDATE_EPOCHS, "dendrite_input_scale": 1.0,
    }
    assert "sham_skip_weight_max_abs_final" not in checks

    # PAI starts the skip weights at zero, so integrating changes nothing.
    assert checks["integration_output_max_abs_diff"] <= grow.INTEGRATION_WARN_TOLERANCE

    # The dendrite really trains after the switch, in its own group.
    assert _post_switch_groups(run) == [
        (_recipe_weight_decay(run), base_params),
        (0.0, checks["dendrite_params_in_optimizer_post_switch"]),
    ]
    assert run.result["final_dendrite"]["any_requires_grad"] is True
    assert run.result["final_dendrite"]["max_abs"] > 0

    diagnostics = summary["dendrite_diagnostics"]
    module_names = {module_id.lstrip(".") for module_id in config.module_ids}
    for label, reference, skip_key in (
        ("final", "final_val_acc", "skip_weight_mean_abs"),
        ("best", "best_val_acc_post_switch", "skip_weight_mean_abs_best"),
    ):
        report = diagnostics[label]
        assert set(report) == DIAGNOSTICS_KEYS, label
        assert report["device"] == "cpu"
        assert report["n_samples"] == VAL_EXAMPLES
        assert set(report["modules"]) == module_names
        for name, module in report["modules"].items():
            assert module["n_dendrites"] == 1
            assert module["skip_weight_mean_abs"] == pytest.approx(
                summary["dendrite"][skip_key][name]
            )
            assert {"corr_with_base_output", "linear_r2_vs_preactivation"} <= set(module)
        # The clean export reproduces the live graph's recorded accuracy.
        agreement = checks[f"diagnostics_val_acc_on_minus_{label}"]
        assert agreement == pytest.approx(
            report["val_acc_dendrite_on"] - summary["results"][reference]
        )
        assert abs(agreement) <= 1 / report["n_samples"]


def test_pai_integration_probe_does_not_perturb_the_run(pai_run):
    """The same run with the probe skipped entirely trains bit-identically."""
    probed = pai_run(**REAL_FC)
    unprobed = pai_run(**REAL_FC, disable_integration_probe=True)
    for run in (probed, unprobed):
        assert run.result["raised"] is None, run.result.get("traceback")
    probed_summary, unprobed_summary = _summary(probed), _summary(unprobed)

    assert _trajectory(_records(probed)) == _trajectory(_records(unprobed))
    assert probed.result["momentum_probe"] == unprobed.result["momentum_probe"]
    assert probed.result["final_dendrite"] == unprobed.result["final_dendrite"]
    for key in ("pb_scores_by_candidate_epoch", "pb_scores_at_integration", "dendrite", "results"):
        assert probed_summary[key] == unprobed_summary[key], key
    assert probed_summary["dendrite_diagnostics"] == unprobed_summary["dendrite_diagnostics"]
    assert probed_summary["checks"] == unprobed_summary["checks"]


def test_pai_sham_run_zeroes_and_freezes_the_dendrite(pai_run):
    sham = pai_run(**REAL_FC, sham=True)
    real = pai_run(**REAL_FC)
    assert sham.result["raised"] is None, sham.result.get("traceback")
    summary, real_summary = _summary(sham), _summary(real)
    checks, real_checks = summary["checks"], real_summary["checks"]
    records, real_records = _records(sham), _records(real)
    base_params = summary["cost"]["base"]["params"]
    boundary = 2 + CANDIDATE_EPOCHS  # last epoch before post-switch training

    assert summary["status"] == "complete"
    assert summary["arm"] == "fc-sham"
    assert summary["variant"] == {
        "sham": True, "dendrite_weight_decay": 0.0,
        "switch_epoch": 2, "candidate_epochs": CANDIDATE_EPOCHS, "dendrite_input_scale": 1.0,
    }

    # Identical to the real run through the candidate phase and the switch.
    assert _trajectory(records[:boundary]) == _trajectory(real_records[:boundary])
    for key in ("pb_scores_by_candidate_epoch", "pb_scores_at_integration"):
        assert summary[key] == real_summary[key], key
    for key in (
        "n_to_p_base_max_abs_change", "candidate_phase_base_max_abs_drift",
        "base_params_in_optimizer_candidate_phase", "candidate_phase_val_acc_span",
        "momentum_buffers_restored", "momentum_buffers_expected",
        "base_params_in_optimizer_post_switch",
    ):
        assert checks[key] == real_checks[key], key
    assert summary["dendrite"]["num_dendrites_added"] == 1

    # Then the dendrite is zeroed, frozen and out of the optimizer; the base
    # alone trains, with its momentum carried and its schedule resumed.
    assert checks["dendrite_params_in_optimizer_post_switch"] == 0
    assert checks["base_params_in_optimizer_post_switch"] == base_params
    assert _post_switch_groups(sham) == [(_recipe_weight_decay(sham), base_params)]
    probe = sham.result["momentum_probe"]
    assert probe["restored"] == probe["carried"] == probe["checked"] == checks["momentum_buffers_restored"] > 0
    reference = _reference_lrs(sham.spec["train_cfg"], EPOCHS * STEPS_PER_EPOCH)
    for record in records[boundary:]:
        assert record["segment"] == POST
        assert record["learning_rate"] == [reference[record["base_step"]]]

    final_dendrite = sham.result["final_dendrite"]
    assert final_dendrite["tensors"] == real.result["final_dendrite"]["tensors"] > 0
    assert final_dendrite["max_abs"] == 0.0
    assert final_dendrite["any_requires_grad"] is False
    assert checks["sham_skip_weight_max_abs_final"] == 0.0
    assert checks["integration_output_max_abs_diff"] == 0.0
    assert set(summary["dendrite"]["skip_weight_mean_abs"].values()) == {0.0}

    # Exports and diagnostics still run; the dendrite is inert in both.
    assert checks["clean_parity_max_abs_diff_final"] <= PARITY_TOLERANCE
    assert checks["clean_parity_max_abs_diff_best"] <= PARITY_TOLERANCE
    for label in ("final", "best"):
        report = summary["dendrite_diagnostics"][label]
        assert set(report) == DIAGNOSTICS_KEYS
        assert report["val_acc_dendrite_on"] == report["val_acc_dendrite_off"]
        assert {module["skip_weight_mean_abs"] for module in report["modules"].values()} == {0.0}
        assert abs(checks[f"diagnostics_val_acc_on_minus_{label}"]) <= 1 / report["n_samples"]
    assert list(sham.cwd.iterdir()) == []


def test_pai_dendrite_weight_decay_override_reaches_the_dendrite_group(pai_run):
    run = pai_run(**REAL_FC, dendrite_weight_decay=0.05, arm="fc-wd0.05")
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    checks = summary["checks"]

    assert _post_switch_groups(run) == [
        (_recipe_weight_decay(run), summary["cost"]["base"]["params"]),
        (0.05, checks["dendrite_params_in_optimizer_post_switch"]),
    ]
    assert checks["dendrite_params_in_optimizer_post_switch"] > 0
    assert summary["arm"] == "fc-wd0.05"
    assert summary["variant"]["dendrite_weight_decay"] == 0.05
    assert summary["variant"]["sham"] is False
    assert summary["schedule"]["dendrite_weight_decay"] == 0.05
    assert summary["status"] == "complete"


def test_pai_incomplete_summaries_carry_the_arm_and_variant(pai_run):
    capped = pai_run(mode="grow", placement="fc", switch_epoch=2, max_minutes=1e-6)
    rewound = pai_run(
        mode="grow", placement="fc", switch_epoch=3,
        val_acc_script=REWIND_SCRIPT, swallow_best_reset_at=3,
    )
    for run, switch_epoch in ((capped, 2), (rewound, 3)):
        summary = _summary(run)
        assert summary["status"] == "incomplete"
        assert summary["arm"] == "fc"
        assert summary["variant"] == {
            "sham": False, "dendrite_weight_decay": 0.0,
            "switch_epoch": switch_epoch, "candidate_epochs": CANDIDATE_EPOCHS,
            "dendrite_input_scale": 1.0,
        }
        assert "dendrite_diagnostics" not in summary


SCALED_POINTWISE = {**REAL_FC, "placement": "pointwise", "check_rebuild": True}


@pytest.mark.parametrize("scale", [1.0, 50.0])
def test_pai_input_scale_trains_scaled_and_exports_a_plain_tanh_graph(pai_run, scale):
    run = pai_run(**SCALED_POINTWISE, dendrite_input_scale=scale, arm=f"pointwise-in{scale:g}")
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    checks = summary["checks"]
    config = grow.resolve_grow_config(run.spec["train_cfg"], placement="pointwise")
    module_names = {module_id.lstrip(".") for module_id in config.module_ids}

    assert summary["status"] == "complete"
    assert summary["variant"]["dendrite_input_scale"] == scale
    assert summary["schedule"]["dendrite_input_scale"] == scale
    # Measured on the integration probe before the n->p switch, per module.
    stds = summary["dendrite_input_std_at_switch"]
    assert set(stds) == module_names
    assert all(value > 0 for value in stds.values())

    # Training ran under f(z / scale); both exports are plain-tanh graphs with
    # the scale folded in, and the saved files reproduce the live graph.
    rebuilds = run.result["rebuild"]
    assert len(rebuilds) == 2  # final and best
    for entry in rebuilds:
        if scale == 1.0:
            assert entry["forward_function"].startswith("<built-in method tanh")
        else:
            assert entry["forward_function"] == f"tanh(z / {scale:g})"
        assert entry["folded"] == repr(scale)
        assert entry["max_abs_diff"] <= grow.PARITY_TOLERANCE
    for key in ("clean_parity_max_abs_diff_final", "clean_parity_max_abs_diff_best"):
        assert checks[key] <= grow.PARITY_TOLERANCE
    for label in ("final", "best"):
        # The on-accuracy is of the export with skip weights 2, not the live
        # graph's, so only its existence is checked here.
        assert f"diagnostics_val_acc_on_minus_{label}" in checks
        assert summary["dendrite_diagnostics"][label]["modules"].keys() == module_names
    assert checks["integration_output_max_abs_diff"] <= grow.INTEGRATION_WARN_TOLERANCE
    assert run.result["final_dendrite"]["max_abs"] > 0


def test_pai_calibrates_input_scale_at_the_switch_before_candidate_training(pai_run):
    run = pai_run(**{
        **REAL_FC,
        "placement": "pointwise_b2",
        "calibrate_dendrite_input_scale": True,
        "check_rebuild": True,
        "arm": "pointwise_b2-auto",
    })
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)

    input_std = summary["dendrite_input_std_at_switch"]["blocks.2.pointwise"]
    assert input_std > 0
    assert summary["schedule"]["dendrite_input_scale"] == pytest.approx(input_std)
    assert summary["variant"]["dendrite_input_scale"] == pytest.approx(input_std)
    assert summary["variant"]["dendrite_input_scale_calibration"] == "switch_input_std"
    assert summary["dendrite_input_scale_calibration"] == {
        "method": "geometric_mean_switch_input_std",
        "module_input_std": {"blocks.2.pointwise": input_std},
        "scale": input_std,
    }
    for entry in run.result["rebuild"]:
        assert entry["forward_function"] == f"tanh(z / {input_std:g})"
        assert entry["folded"] == repr(input_std)
        assert entry["max_abs_diff"] <= grow.PARITY_TOLERANCE


GROUPED_POINTWISE = {
    **REAL_FC, "placement": "pointwise_b2", "group_batchnorm": True, "check_rebuild": True,
}


def test_pai_group_batchnorm_grows_a_conv_and_batchnorm_dendrite(pai_run):
    plain = pai_run(mode="plain")
    run = pai_run(**GROUPED_POINTWISE)
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    checks = summary["checks"]
    records = _records(run)
    name = "blocks.2.pointwise"

    assert summary["status"] == "complete"
    assert summary["arm"] == "pointwise_b2-bn"
    assert summary["variant"]["group_batchnorm"] is True
    assert summary["schedule"]["group_batchnorm"] is True

    # The regrouping is exact: the pre-switch epochs are still the scratch run.
    for grown_record, scratch_record in zip(records[:2], plain.result["history"][:2]):
        assert grown_record["segment"] == PRE
        for key in (
            "train_loss", "train_gate_sparsity", "train_accuracy",
            "val_loss", "val_acc", "learning_rate", "parameter_count",
        ):
            assert grown_record[key] == scratch_record[key], key

    # Post-switch records report the skip weights as they train.
    for record in records:
        assert ("dendrite_skip_weight_mean_abs" in record) == (record["segment"] == POST)
    assert records[-1]["dendrite_skip_weight_mean_abs"][name] == pytest.approx(
        summary["dendrite"]["skip_weight_mean_abs"][name]
    )

    # The dendrite is a conv + BatchNorm copy, and the export keeps both.
    path = run.output_dir / summary["artifacts"]["final_clean"]
    with safe_open(str(path), "pt") as clean:
        keys = set(clean.keys())
    prefix = f"{name}.layer_array"
    assert {
        f"{prefix}.0.model.0.weight", f"{prefix}.0.model.1.running_var",
        f"{prefix}.1.model.0.weight", f"{prefix}.1.model.1.running_var",
    } <= keys
    assert not any(key.startswith("blocks.2.bn.") for key in keys)
    channels = TINY_MODEL["channels"]
    # conv (C x C) + BatchNorm affine (2C) + skip weights (C)
    assert summary["cost"]["deployed"]["params"] - summary["cost"]["base"]["params"] == (
        channels * channels + 3 * channels
    )
    for entry in run.result["rebuild"]:
        assert entry["max_abs_diff"] <= grow.PARITY_TOLERANCE
    for key in ("clean_parity_max_abs_diff_final", "clean_parity_max_abs_diff_best"):
        assert checks[key] <= grow.PARITY_TOLERANCE
    assert checks["integration_output_max_abs_diff"] <= grow.INTEGRATION_WARN_TOLERANCE
    assert run.result["final_dendrite"]["max_abs"] > 0
    assert set(summary["dendrite_diagnostics"]["final"]["modules"]) == {name}


def test_pai_group_batchnorm_grows_one_dendrite_per_grouped_block(pai_run):
    run = pai_run(**{**GROUPED_POINTWISE, "placement": "pointwise_b123"})
    assert run.result["raised"] is None, run.result.get("traceback")
    summary = _summary(run)
    checks = summary["checks"]
    blocks = (1, 2, 3)

    assert summary["status"] == "complete"
    assert summary["arm"] == "pointwise_b123-bn"
    path = run.output_dir / summary["artifacts"]["final_clean"]
    with safe_open(str(path), "pt") as clean:
        keys = set(clean.keys())
    for index in blocks:
        assert f"blocks.{index}.pointwise.layer_array.0.model.1.running_var" in keys
        assert not any(key.startswith(f"blocks.{index}.bn.") for key in keys)
    assert "blocks.0.bn.running_var" in keys
    channels = TINY_MODEL["channels"]
    assert summary["cost"]["deployed"]["params"] - summary["cost"]["base"]["params"] == (
        len(blocks) * (channels * channels + 3 * channels)
    )
    for entry in run.result["rebuild"]:
        assert entry["max_abs_diff"] <= grow.PARITY_TOLERANCE
    for key in ("clean_parity_max_abs_diff_final", "clean_parity_max_abs_diff_best"):
        assert checks[key] <= grow.PARITY_TOLERANCE
    assert checks["integration_output_max_abs_diff"] <= grow.INTEGRATION_WARN_TOLERANCE
    assert set(summary["dendrite_diagnostics"]["final"]["modules"]) == {
        f"blocks.{index}.pointwise" for index in blocks
    }
