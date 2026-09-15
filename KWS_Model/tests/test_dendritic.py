from types import SimpleNamespace
from typing import Any
from typing import cast

import torch
import yaml

from kws.models.ds_cnn import build_ds_cnn
from kws.optimize.dendritic import (
    build_cycle_base,
    estimate_one_dendrite_params,
    read_pai_architecture_results,
    _restore_single_dendrite_skip_weights,
)
from kws.optimize.dendritic_prune_loop import candidate_widths, judge_candidate


def test_cycle1_base_hits_expected_size_and_widths(tmp_path):
    source_cfg = {
        "name": "ds_cnn_xs",
        "initial_channels": 18,
        "initial_kernel": 5,
        "initial_stride": 2,
        "block_channels": [40, 40],
        "dropout": 0.2,
    }
    input_shape = (40, 98)
    source = build_ds_cnn(source_cfg, input_shape, num_classes=6)
    checkpoint_path = tmp_path / "xs.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "model_cfg": source_cfg,
            "input_shape": input_shape,
            "num_classes": 6,
        },
        checkpoint_path,
    )

    with open("configs/model/ds_cnn_xxs.yaml") as f:
        target_model_cfg = yaml.safe_load(f)
    model, checkpoint, model_cfg = build_cycle_base(
        str(checkpoint_path),
        keep_ratio=0.45,
        target_model_cfg=target_model_cfg,
    )

    assert checkpoint["model_cfg"]["block_channels"] == [40, 40]
    assert model_cfg["name"] == "ds_cnn_xxs"
    assert model_cfg["block_channels"] == [18, 18]
    assert sum(parameter.numel() for parameter in model.parameters()) == 1716
    assert estimate_one_dendrite_params(model) == 2946
    assert estimate_one_dendrite_params(model, "fc_only") == 1830

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, *checkpoint["input_shape"]))
    assert output.shape == (2, checkpoint["num_classes"])


def test_read_pai_architecture_results_selects_best_deployable_row(tmp_path):
    run_dir = tmp_path / "pai_w18"
    run_dir.mkdir()
    (run_dir / "pai_w18_best_arch_scores.csv").write_text(
        "Param Counts,Max Valid Scores,Train\n"
        "1716,0.8895,0.69\n"
        "2946,0.9061,0.75\n"
    )

    accuracy, parameters = read_pai_architecture_results(str(run_dir))

    assert accuracy == 0.9061
    assert parameters == 2946


def test_pruning_widths_descend_to_configured_minimum():
    assert candidate_widths(18, 14, 1) == [18, 17, 16, 15, 14]
    assert candidate_widths(18, 12, 3) == [18, 15, 12]
    assert candidate_widths(18, 13, 3) == [18, 15, 13]


def test_pruning_stops_below_accuracy_floor():
    accepted = judge_candidate(0.901, 0.90, 0.906, None)
    degraded = judge_candidate(0.899, 0.90, 0.901, None)

    assert accepted.accepted
    assert not degraded.accepted
    assert "below" in degraded.reason


def test_optional_relative_drop_rule():
    decision = judge_candidate(0.902, 0.90, 0.910, 0.005)

    assert not decision.accepted
    assert "dropped" in decision.reason


def test_restore_single_dendrite_skip_weights():
    class CleanModule(torch.nn.Module):
        layer_array: torch.nn.ModuleList
        skip_weights: torch.nn.ParameterList

        def __init__(self):
            super().__init__()
            self.layer_array = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    class CleanNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([CleanModule()])

    model = CleanNetwork()
    weights = {"blocks.0": [torch.tensor([[0.25, -0.5]])]}

    restored = _restore_single_dendrite_skip_weights(model, weights)
    clean_module = cast(CleanModule, model.blocks[0])

    assert restored == 1
    assert torch.equal(clean_module.skip_weights[0], weights["blocks.0"][0])
    assert not clean_module.skip_weights[0].requires_grad


def test_restore_single_dendrite_skip_weights_from_pai_sequential_cleanup():
    """PAI 3.2.8 emits nn.Sequential, not ModuleList, for clean branches."""

    class CleanModule(torch.nn.Module):
        layer_array: torch.nn.Sequential
        skip_weights: torch.nn.ParameterList

        def __init__(self):
            super().__init__()
            self.layer_array = torch.nn.Sequential(
                torch.nn.Identity(), torch.nn.Identity()
            )

    model = torch.nn.Module()
    model.fc = CleanModule()
    weights = {"fc": [torch.tensor([[0.25, -0.5]])]}

    restored = _restore_single_dendrite_skip_weights(model, weights)
    clean_module = cast(CleanModule, model.fc)

    assert restored == 1
    assert torch.equal(clean_module.skip_weights[0], weights["fc"][0])
    assert not clean_module.skip_weights[0].requires_grad


def test_clean_reload_recreates_optional_pai_skip_coefficients():
    from kws.optimize.dendritic import ensure_clean_dendrite_skip_weights

    class CleanModule(torch.nn.Module):
        layer_array: torch.nn.ModuleList
        skip_weights: torch.nn.ParameterList

        def __init__(self):
            super().__init__()
            self.layer_array = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    class CleanNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.blocks = torch.nn.ModuleList([CleanModule()])

    model = CleanNetwork()
    state = {"anchor": torch.ones(1), "blocks.0.skip_weights.0": torch.tensor([0.25])}
    ensure_clean_dendrite_skip_weights(model, state)
    model.load_state_dict(state, strict=True)
    clean_module = cast(CleanModule, model.blocks[0])

    assert torch.equal(clean_module.skip_weights[0], torch.tensor([0.25]))
    assert not clean_module.skip_weights[0].requires_grad


def _fake_perforated_model():
    """A stand-in with the parameter names PAI gives base and dendrite tensors."""
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(4, 4))          # base
    model.bias = torch.nn.Parameter(torch.zeros(4))                # base
    model.dendrite_weight = torch.nn.Parameter(torch.zeros(4, 4))  # dendritic
    model.dendrites_to_top = torch.nn.Parameter(torch.zeros(1, 4))  # dendritic
    return model


def test_learning_phase_splits_base_from_dendritic_parameters():
    from kws.optimize.dendritic import describe_learning_phase

    phase = describe_learning_phase(_fake_perforated_model())

    assert phase["base"]["trainable"] == 20   # 4x4 weight + 4 bias
    assert phase["dendrite"]["trainable"] == 20  # 4x4 dendrite + 1x4 to_top
    assert phase["base"]["frozen"] == 0


def test_legacy_freeze_helper_stops_base_weights_and_leaves_dendrites_alone():
    from kws.optimize.dendritic import describe_learning_phase, enforce_base_weight_freeze

    model = _fake_perforated_model()
    frozen = enforce_base_weight_freeze(model)

    assert frozen == 20
    phase = describe_learning_phase(model)
    assert phase["base"]["trainable"] == 0
    assert phase["base"]["frozen"] == 20
    # This pins the legacy helper's mechanical behavior. The PAI pipeline must
    # not use it because dendrite scoring still needs base autograd.
    assert phase["dendrite"]["trainable"] == 20


def test_legacy_freeze_helper_twice_reports_nothing_left_to_freeze():
    from kws.optimize.dendritic import enforce_base_weight_freeze

    model = _fake_perforated_model()
    enforce_base_weight_freeze(model)
    assert enforce_base_weight_freeze(model) == 0


def test_selected_dendrites_freeze_and_hand_the_base_back_for_the_kd_resume():
    from kws.optimize.dendritic import describe_learning_phase, freeze_selected_dendrites

    model = _fake_perforated_model()
    frozen = freeze_selected_dendrites(model)

    assert frozen == 20
    phase = describe_learning_phase(model)
    assert phase["dendrite"]["trainable"] == 0
    assert phase["base"]["trainable"] == 20


def test_clean_pai_parameter_names_classify_selected_branches_as_dendrites():
    from kws.optimize.dendritic import _is_dendrite_parameter

    # PAI stores residual branches first and the original/base branch last.
    assert _is_dendrite_parameter("blocks.0.layer_array.0.pointwise.weight")
    assert _is_dendrite_parameter("blocks.0.skip_weights.0")
    assert _is_dendrite_parameter("blocks.0.dendrite_module.layers.0.weight")
    assert not _is_dendrite_parameter(
        "blocks.0.dendrite_module.parent_module.pointwise.weight"
    )
    assert not _is_dendrite_parameter("blocks.0.layer_array.1.pointwise.weight")


def test_clean_pai_branch_classification_uses_the_module_length():
    from kws.optimize.dendritic import _is_dendrite_parameter

    class Branches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_array = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
            )

    model = torch.nn.Module()
    model.block = Branches()
    assert _is_dendrite_parameter("block.layer_array.0.weight", model)
    assert _is_dendrite_parameter("block.layer_array.1.weight", model)
    assert not _is_dendrite_parameter("block.layer_array.2.weight", model)


def test_legacy_freeze_helper_is_symmetric():
    from kws.optimize.dendritic import (
        describe_learning_phase,
        enforce_base_weight_freeze,
        restore_base_weight_training,
    )

    model = _fake_perforated_model()
    enforce_base_weight_freeze(model)
    restored = restore_base_weight_training(model)

    assert restored == 20
    assert describe_learning_phase(model)["base"]["trainable"] == 20


def test_an_unreadable_pai_mode_leaves_requires_grad_alone():
    """A PAI internals change must degrade to the library's own behaviour."""
    from kws.optimize import dendritic

    model = _fake_perforated_model()
    original = {name: p.requires_grad for name, p in model.named_parameters()}

    mode = dendritic.apply_phase_freezing(model)  # no live PAI tracker here

    assert mode == "?"
    assert {name: p.requires_grad for name, p in model.named_parameters()} == original


def test_pai_optimizer_keeps_kd_adapter_outside_pai_parameter_filter(monkeypatch):
    from kws.optimize import dendritic

    class FakeTracker:
        def set_optimizer_instance(self, optimizer):
            self.optimizer = optimizer

    class FakeKD:
        def __init__(self):
            self.adapter = torch.nn.Linear(2, 3, bias=False)

        def extra_parameters(self):
            return list(self.adapter.parameters())

    tracker = FakeTracker()
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)
    model = torch.nn.Linear(2, 2)
    kd = FakeKD()
    train_cfg = {
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_fraction": 0.05,
        "epochs": 2,
        "dendritic_schedule_epochs": 2,
    }

    optimizer, scheduler, adapter_optimizer, adapter_scheduler = (
        dendritic._make_optimizer_and_scheduler(
            model, train_cfg, loader_length=4, kd=kd, lr_multiplier=0.25
        )
    )

    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    adapter_parameter_ids = {id(parameter) for parameter in kd.extra_parameters()}
    pai_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    adapter_optimizer_ids = {
        id(parameter)
        for group in adapter_optimizer.param_groups
        for parameter in group["params"]
    }

    assert tracker.optimizer is optimizer
    assert pai_parameter_ids == model_parameter_ids
    assert adapter_optimizer_ids == adapter_parameter_ids
    assert scheduler.base_lrs == [0.00025]
    assert adapter_scheduler.base_lrs == [0.00025]


def test_fc_only_configuration_targets_exact_classifier(monkeypatch):
    from kws.optimize import dendritic

    class FakePAIConfig:
        def __init__(self):
            self.values = {}

        def __getattr__(self, name):
            if not name.startswith("set_"):
                raise AttributeError(name)

            def setter(value):
                self.values[name.removeprefix("set_")] = value

            return setter

    pc = FakePAIConfig()
    monkeypatch.setattr(dendritic, "GPA", SimpleNamespace(pc=pc))
    monkeypatch.setattr(dendritic, "disarm_pai_debugger", lambda: None)

    dendritic.configure_perforatedai(
        {
            "testing_dendrite_capacity": False,
            "conversion": "fc_only",
            "max_dendrites": 1,
            "n_epochs_to_switch": 25,
            "improvement_threshold": [0.001, 0.0001, 0.0],
            "candidate_weight_initialization_multiplier": 0.01,
            "initial_correlation_batches": 40,
            "max_dendrite_tries": 2,
            "forward_function": "tanh",
        },
        torch.device("cpu"),
    )

    assert pc.values["module_ids_to_perforate"] == [".fc"]
    assert pc.values["modules_to_perforate"] == []
    assert pc.values["module_names_to_perforate"] == []
    assert pc.values["modules_to_track"] == [
        dendritic.DSConvBlock,
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
    ]


def test_module_id_configuration_targets_only_requested_placements(monkeypatch):
    from kws.optimize import dendritic

    class FakePAIConfig:
        def __init__(self):
            self.values = {}

        def __getattr__(self, name):
            if not name.startswith("set_"):
                raise AttributeError(name)

            def setter(value):
                self.values[name.removeprefix("set_")] = value

            return setter

    pc = FakePAIConfig()
    monkeypatch.setattr(dendritic, "GPA", SimpleNamespace(pc=pc))
    monkeypatch.setattr(dendritic, "disarm_pai_debugger", lambda: None)

    dendritic.configure_perforatedai(
        {
            "testing_dendrite_capacity": False,
            "conversion": "module_ids",
            "module_ids_to_perforate": ["blocks.0", ".fc"],
            "max_dendrites": 1,
            "n_epochs_to_switch": 25,
            "improvement_threshold": [0.001, 0.0001, 0.0],
            "candidate_weight_initialization_multiplier": 0.01,
            "initial_correlation_batches": 40,
            "max_dendrite_tries": 2,
            "forward_function": "tanh",
        },
        torch.device("cpu"),
    )

    assert pc.values["module_ids_to_perforate"] == [".blocks.0", ".fc"]
    assert pc.values["modules_to_perforate"] == []
    assert pc.values["modules_to_track"] == [
        dendritic.DSConvBlock,
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.Linear,
    ]


def test_configuration_without_conversion_uses_historical_selector(monkeypatch):
    from kws.optimize import dendritic

    class FakePAIConfig:
        def __init__(self):
            self.values = {}

        def __getattr__(self, name):
            if not name.startswith("set_"):
                raise AttributeError(name)

            def setter(value):
                self.values[name.removeprefix("set_")] = value

            return setter

    pc = FakePAIConfig()
    monkeypatch.setattr(dendritic, "GPA", SimpleNamespace(pc=pc))
    monkeypatch.setattr(dendritic, "disarm_pai_debugger", lambda: None)

    dendritic.configure_perforatedai(
        {
            "testing_dendrite_capacity": False,
            "max_dendrites": 1,
            "n_epochs_to_switch": 25,
            "improvement_threshold": [0.001, 0.0001, 0.0],
            "candidate_weight_initialization_multiplier": 0.01,
            "initial_correlation_batches": 40,
            "max_dendrite_tries": 2,
            "forward_function": "tanh",
        },
        torch.device("cpu"),
    )

    assert pc.values["module_ids_to_perforate"] == []
    assert pc.values["modules_to_perforate"] == [
        dendritic.DSConvBlock,
        torch.nn.Linear,
    ]
    assert pc.values["modules_to_track"] == [
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
    ]


def test_gentle_restart_applies_only_after_dendrite_integration():
    from kws.optimize.dendritic import _restructure_lr_multiplier

    assert _restructure_lr_multiplier("p", "n", 0.25) == 0.25
    # At the terminal dendrite limit PAI performs n -> p -> n inside one
    # add_validation_score call, so both externally sampled modes are n.
    assert (
        _restructure_lr_multiplier(
            "n", "n", 0.25, dendrite_integrated=True
        )
        == 0.25
    )
    assert _restructure_lr_multiplier("n", "p", 0.25) == 1.0
    assert _restructure_lr_multiplier("n", "n", 0.25) == 1.0


def test_completed_pai_tracker_boundary_is_export_only(monkeypatch):
    from kws.optimize import dendritic

    tracker = cast(Any, SimpleNamespace(
        member_vars={
            "doing_pai": True,
            "num_dendrites_integrated": 1,
        }
    ))
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)

    assert dendritic.pai_tracker_at_terminal_boundary(1)
    assert not dendritic.pai_tracker_at_terminal_boundary(2)

    tracker.member_vars["doing_pai"] = False
    assert dendritic.pai_tracker_at_terminal_boundary(2)
