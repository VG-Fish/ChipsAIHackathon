from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from kws.optimize import dendritic


class _Stateful:
    def __init__(self, name: str, events: list[str] | None = None):
        self.name = name
        self.events = events
        self.state = {}
        self.param_groups = [{"lr": 0.01}]

    def zero_grad(self):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {"name": self.name}

    def load_state_dict(self, state):
        if self.events is not None:
            self.events.append(self.name)
        self.loaded_state = state


def test_fresh_standalone_run_refuses_a_nonempty_native_candidate(tmp_path):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    (run_dir / "native.csv").write_text("existing PAI output\n")

    with pytest.raises(FileExistsError, match="nonempty.*--resume"):
        dendritic._select_pai_resume_sidecar(
            run_dir,
            tmp_path / "kws_latest.pt",
            resume=False,
            resume_from=None,
        )


def test_explicit_resume_requires_the_selected_sidecar(tmp_path):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="resume sidecar not found"):
        dendritic._select_pai_resume_sidecar(
            run_dir,
            tmp_path / "missing.pt",
            resume=True,
            resume_from=None,
        )


def test_pai_pair_save_persists_kd_state_and_requires_matching_native_digest(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    sidecar_path = tmp_path / "kws_latest.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    adapter = torch.nn.Linear(2, 3, bias=False)
    adapter_optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.02)
    adapter_scheduler = torch.optim.lr_scheduler.LambdaLR(
        adapter_optimizer, lambda _step: 1.0
    )

    class FakeKD:
        def state_dict(self):
            return {"adapter_state_dict": adapter.state_dict(), "marker": "persisted"}

    def save_system(_model, folder, name):
        # Mirror the real PerforatedAI contract: save_net writes
        # <folder>/<name>.pt as safetensors (using_safe_tensors defaults on),
        # NOT a torch pickle and NOT <folder>/<name>/latest.pt.  A fake that
        # encodes our assumption instead of PAI's hid a crash that only
        # surfaced on the first real dendrite save.
        target = Path(folder) / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file({"native": torch.zeros(1)}, target)

    monkeypatch.setattr(dendritic.UPA, "save_system", save_system)
    loaders = (
        SimpleNamespace(generator=torch.Generator().manual_seed(1)),
        SimpleNamespace(generator=torch.Generator().manual_seed(2)),
    )

    saved = dendritic._save_pai_restart_pair(
        model=model,
        run_dir=run_dir,
        pai_run_name=run_dir.name,
        sidecar_path=sidecar_path,
        optimizer=optimizer,
        scheduler=scheduler,
        kd=FakeKD(),
        adapter_optimizer=adapter_optimizer,
        adapter_scheduler=adapter_scheduler,
        phase_trail=[{"epoch": 0, "mode": "n"}],
        completed_epoch=1,
        global_step=4,
        history_length=1,
        last_metric_digest="metrics-digest",
        teacher_checkpoint="teacher.pt",
        recipe_fingerprint="recipe",
        loaders=loaders,
    )

    assert saved["kd_state_dict"]["marker"] == "persisted"
    assert saved["adapter_optimizer_state_dict"] is not None
    assert set(saved) == dendritic.PAI_SIDECAR_KEYS
    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    assert loaded["completed_epoch"] == 1

    (run_dir / "latest.pt").write_bytes(b"corrupt replacement")
    with pytest.raises(ValueError, match="does not match sidecar"):
        dendritic._load_pai_sidecar(
            sidecar_path, run_dir, "recipe", torch.device("cpu")
        )


def test_pai_sidecar_schema_rejects_missing_native_digest(tmp_path):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    native = run_dir / "latest.pt"
    native.write_bytes(b"native")
    sidecar = tmp_path / "sidecar.pt"
    state = {key: None for key in dendritic.PAI_SIDECAR_KEYS}
    state.update(
        {
            "format_version": dendritic.FRAMEWORK_CYCLE_VERSION,
            "kind": "kws_pai_training_state",
            "stage": "sparsity",
            "phase": "pai",
            "completed_epoch": 1,
            "next_epoch": 2,
            "global_step": 1,
            "history_length": 1,
            "phase_trail": [{"mode": "n"}],
            "loader_generator_states": [None, None],
            "pai_run_dir": str(run_dir),
            "native_pai_latest": str(native),
            "recipe_fingerprint": "recipe",
        }
    )
    torch.save(state, sidecar)

    with pytest.raises(ValueError, match="no valid native digest"):
        dendritic._load_pai_sidecar(
            sidecar, run_dir, "recipe", torch.device("cpu")
        )


def test_restore_loads_kd_adapter_before_its_optimizer_and_loads_model_strictly():
    events: list[str] = []

    class FakeModel:
        def load_state_dict(self, state, *, strict):
            events.append(f"model-strict-{strict}")

    class FakeKD:
        def load_state_dict(self, state, *, strict):
            events.append(f"kd-strict-{strict}")

    model_optimizer = _Stateful("model-optimizer", events)
    model_scheduler = _Stateful("model-scheduler", events)
    adapter_optimizer = _Stateful("adapter-optimizer", events)
    adapter_scheduler = _Stateful("adapter-scheduler", events)
    state = {
        "model_state_dict": {"weight": torch.ones(1)},
        "kd_state_dict": {"adapter_state_dict": {"weight": torch.ones(1)}},
        "optimizer_state_dict": {"model": True},
        "scheduler_state_dict": {"model_scheduler": True},
        "adapter_optimizer_state_dict": {"adapter": True},
        "adapter_scheduler_state_dict": {"adapter_scheduler": True},
    }

    dendritic._restore_pai_training_state(
        state,
        model=FakeModel(),
        optimizer=model_optimizer,
        scheduler=model_scheduler,
        kd=FakeKD(),
        adapter_optimizer=adapter_optimizer,
        adapter_scheduler=adapter_scheduler,
        device=torch.device("cpu"),
    )

    assert events[0] == "model-strict-True"
    assert events[1] == "kd-strict-True"
    assert events.index("kd-strict-True") < events.index("adapter-optimizer")


def test_restore_pai_tracker_uses_the_sidecars_committed_tracker_string(monkeypatch):
    committed = (
        "num_epochs_run,98\n"
        "epoch_last_improved,91\n"
        "accuracies,\n"
        "0.8,0.81\n"
    )
    restored: list[str] = []
    tracker = SimpleNamespace(from_string=restored.append)
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)

    dendritic._restore_pai_tracker_state(
        {
            "tracker_string": torch.tensor(
                list(committed.encode("utf-8")), dtype=torch.uint8
            )
        }
    )

    assert restored == [committed]


def test_restore_pai_tracker_rejects_a_missing_tracker_string(monkeypatch):
    monkeypatch.setattr(
        dendritic.GPA, "pai_tracker", SimpleNamespace(from_string=lambda _value: None)
    )

    with pytest.raises(ValueError, match="missing its tracker_string"):
        dendritic._restore_pai_tracker_state({})


def test_pai_epoch_metrics_include_named_kd_losses_and_adapter_lr(monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = _Stateful("model")
    adapter_optimizer = _Stateful("adapter")
    optimizer.param_groups[0]["lr"] = 0.003
    adapter_optimizer.param_groups[0]["lr"] = 0.007
    tracker = SimpleNamespace(member_vars={"mode": "p"})
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)
    monkeypatch.setattr(
        dendritic.UPA,
        "count_params",
        lambda value: sum(parameter.numel() for parameter in value.parameters()),
    )

    record = dendritic._pai_epoch_record(
        epoch=2,
        global_step=8,
        elapsed_seconds=0.5,
        train_losses={
            "total": 1.0,
            "classification": 0.8,
            "response": 0.15,
            "feature": 0.05,
        },
        train_accuracy=0.75,
        val_loss=0.9,
        val_accuracy=0.5,
        optimizer=optimizer,
        adapter_optimizer=adapter_optimizer,
        mode="p",
        next_mode="n",
        base_params_in_optimizer_count=0,
        restructured=True,
        model=model,
        seed=17,
    )

    assert record["train_classification"] == 0.8
    assert record["train_response"] == 0.15
    assert record["train_feature"] == 0.05
    assert record["adapter_learning_rates"] == [0.007]
    assert record["optimizer_learning_rates"] == {
        "model": [0.003],
        "kd_adapter": [0.007],
    }
    assert record["pai_mode"] == "p"
    assert record["next_pai_mode"] == "n"
    assert record["base_params_in_optimizer"] == 0


def test_restructure_resets_optimizer_before_the_only_paired_save(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    run_dir = tmp_path / "candidate"
    model = torch.nn.Linear(2, 2)
    checkpoint = {
        "input_shape": (1, 2),
        "num_keywords": 1,
        "num_classes": 2,
        "val_acc": 0.0,
    }
    model_cfg = {"block_channels": [2], "name": "tiny"}
    dataset = [(torch.ones(2), torch.tensor(0))]
    train_cfg = {
        "seed": 3,
        "pruning": {"kind": "structured", "keep_ratio": 1.0},
        "augment": False,
        "perforatedai": {"enforce_base_weight_freeze": False},
        "label_smoothing": 0.0,
        "objective": "accuracy",
        "deployment": {
            "profile_device": "cpu",
            "bits_per_weight": 32,
            "latency_iterations": 1,
        },
    }
    tracker = SimpleNamespace(
        member_vars={"mode": "n"},
        add_extra_score=lambda *_args: None,
        add_validation_score=lambda _score, value: (value, True, True),
    )
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)
    monkeypatch.setattr(dendritic, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        dendritic,
        "build_cycle_base",
        lambda *_args, **_kwargs: (model, checkpoint, model_cfg),
    )
    monkeypatch.setattr(dendritic, "cycle_fingerprint", lambda *_args: "recipe")
    monkeypatch.setattr(dendritic, "estimate_one_dendrite_params", lambda _model: 2)
    monkeypatch.setattr(
        dendritic,
        "build_datasets",
        lambda *_args, **_kwargs: (
            {dendritic.TRAIN: dataset, dendritic.VAL: dataset},
            {"keyword": 0},
        ),
    )
    monkeypatch.setattr(
        dendritic,
        "build_data_loader",
        lambda value, *_args, **_kwargs: [
            (torch.stack([item[0] for item in value]), torch.stack([item[1] for item in value]))
        ],
    )
    monkeypatch.setattr(dendritic, "configure_perforatedai", lambda *_args: None)
    monkeypatch.setattr(dendritic.UPA, "perforate_model", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        dendritic.UPA,
        "count_params",
        lambda value: sum(parameter.numel() for parameter in value.parameters()),
    )
    old_optimizer = _Stateful("old")
    reset_optimizer = _Stateful("reset")
    optimizers = iter(
        [
            (old_optimizer, _Stateful("old-scheduler"), None, None),
            (reset_optimizer, _Stateful("reset-scheduler"), None, None),
        ]
    )
    monkeypatch.setattr(
        dendritic, "_make_optimizer_and_scheduler", lambda *_args, **_kwargs: next(optimizers)
    )
    paired_optimizers = []
    monkeypatch.setattr(
        dendritic,
        "_save_pai_restart_pair",
        lambda **kwargs: paired_optimizers.append(kwargs["optimizer"]),
    )

    def export(value, save_name):
        path = Path(save_name) / "final_clean_pai.pt"
        path.write_bytes(b"clean")
        return value

    monkeypatch.setattr(dendritic, "export_final_pai_model", export)
    monkeypatch.setattr(dendritic, "read_pai_architecture_results", lambda _name: (0.0, 6))
    monkeypatch.setattr(
        dendritic,
        "profile_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            as_dict=lambda: {
                "params": 6,
                "macs": 6,
                "weight_bytes": 24,
                "latency_ms_p50": 0.1,
            }
        ),
    )

    dendritic.run_cycle(
        str(source),
        {},
        model_cfg,
        train_cfg,
        str(run_dir),
        resume=False,
    )

    assert paired_optimizers == [reset_optimizer]


def test_post_pai_kd_materializes_best_checkpoint_when_zero_does_not_improve(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(2, 2)
    initial = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    teacher = SimpleNamespace(
        checkpoint_path="teacher.pt",
        num_classes=2,
        num_keywords=1,
    )

    class FakeKD:
        uses_features = False

        def __init__(self, *_args, **_kwargs):
            pass

        def describe(self):
            return {"teacher_checkpoint": "teacher.pt"}

    monkeypatch.setattr(dendritic, "DistillationCriterion", FakeKD)
    monkeypatch.setattr(dendritic, "supports_pooled_features", lambda *_args: False)
    monkeypatch.setattr(dendritic, "build_data_loader", lambda *_args, **_kwargs: [])

    def fake_finetune(value, *_args, **_kwargs):
        with torch.no_grad():
            for parameter in value.parameters():
                parameter.add_(10)
        return SimpleNamespace(best_val_acc=0.0, history=[{"val_acc": 0.0}])

    monkeypatch.setattr(dendritic, "run_finetune", fake_finetune)
    output_dir = tmp_path / "run"
    result = dendritic.resume_kd_finetune(
        model,
        {dendritic.TRAIN: [], dendritic.VAL: []},
        {
            "resume_epochs": 1,
            "seed": 0,
            "label_smoothing": 0.0,
            "distillation": {},
        },
        torch.device("cpu"),
        teacher,
        (1, 2),
        "candidate",
        baseline_val_acc=0.0,
        output_dir=output_dir,
    )

    best_path = dendritic.ArtifactLayout(output_dir).checkpoint_path(
        "sparsity", "resume_kd", "best", candidate="candidate"
    )
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    assert result["status"] == "no_improvement"
    assert result["checkpoint"] == str(best_path)
    assert checkpoint["val_acc"] == 0.0
    for name, tensor in initial.items():
        assert torch.equal(checkpoint["model_state_dict"][name], tensor)


@pytest.mark.parametrize(
    ("cli_args", "expected_resume", "expected_resume_from"),
    [
        (["--resume"], True, None),
        (["--resume-from", "explicit-sidecar.pt"], False, "explicit-sidecar.pt"),
    ],
)
def test_standalone_cli_forwards_resume_options(
    cli_args, expected_resume, expected_resume_from, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "sys.argv", ["kws.optimize.dendritic", *cli_args]
    )
    monkeypatch.setattr(dendritic, "load_yaml", lambda path: {"path": path})
    monkeypatch.setattr(dendritic, "run_session", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        dendritic,
        "run_cycle",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    dendritic.main()

    assert captured["resume"] is expected_resume
    assert captured["resume_from"] == expected_resume_from


def test_step_3c_check_separates_unmeasurable_from_violated():
    """``-1`` means "could not read", not "the base is live".

    The caller logs a step-3c violation on this number, and that log line is an
    alert pattern for the run monitor, so a truthiness test here would turn an
    unreadable optimizer into a false crash alert reporting "-1 base
    parameters". The sentinel has to stay distinguishable from a real count.
    """
    model = torch.nn.Linear(4, 3)

    # No param_groups at all, and param_groups without a "params" key: both are
    # "unmeasured", and neither may be reported as a clean zero.
    assert dendritic.base_params_in_optimizer(model, SimpleNamespace()) == -1
    assert dendritic.base_params_in_optimizer(model, _Stateful("stub")) == -1
    assert dendritic.base_params_in_optimizer(
        model,
        SimpleNamespace(param_groups=[{"params": []}, {"lr": 0.1}]),
    ) == -1

    # A foreign or stale tensor cannot safely be called a base parameter or a
    # dendrite, so it also leaves the audit unmeasured rather than producing a
    # false violation count.
    foreign = torch.nn.Parameter(torch.ones(2))
    assert dendritic.base_params_in_optimizer(
        model,
        SimpleNamespace(param_groups=[{"params": [foreign]}]),
    ) == -1

    # A real optimizer holding the base reports the true live count.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert dendritic.base_params_in_optimizer(model, optimizer) == 15

    # Count parameter identities, not repeated references in unusual wrappers.
    duplicated = SimpleNamespace(
        param_groups=[
            {"params": list(model.parameters())},
            {"params": list(model.parameters())},
        ]
    )
    assert dendritic.base_params_in_optimizer(model, duplicated) == 15

    # An optimizer PAI has stripped reports a genuine zero.
    optimizer.param_groups = [{"params": []}]
    assert dendritic.base_params_in_optimizer(model, optimizer) == 0


def test_base_batchnorm_stats_are_pinned_without_severing_autograd():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_bn = torch.nn.BatchNorm1d(3)
            self.dendrite_bn = torch.nn.BatchNorm1d(3)

        def forward(self, value):
            return self.base_bn(value) + self.dendrite_bn(value)

    model = Model().train()
    before_base_mean = model.base_bn.running_mean.clone()
    before_dendrite_mean = model.dendrite_bn.running_mean.clone()

    assert dendritic.freeze_base_batchnorm_stats(model) == 1
    assert not model.base_bn.training
    assert model.dendrite_bn.training
    assert model.base_bn.weight.requires_grad

    model(torch.full((4, 3), 2.0)).sum().backward()

    assert torch.equal(model.base_bn.running_mean, before_base_mean)
    assert not torch.equal(model.dendrite_bn.running_mean, before_dendrite_mean)
    assert model.base_bn.weight.grad is not None


def test_cycle_rejects_the_known_bad_base_freeze_before_starting_work():
    with pytest.raises(ValueError, match="enforce_base_weight_freeze must be false"):
        dendritic.run_cycle(
            "missing-checkpoint.pt",
            {},
            {},
            {
                "seed": 0,
                "pruning": {"kind": "structured"},
                "perforatedai": {"enforce_base_weight_freeze": True},
            },
            "unused-candidate",
        )
