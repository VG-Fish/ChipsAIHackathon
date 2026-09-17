from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import load_file, save_file

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


def test_automatic_resume_starts_clean_in_empty_candidate_and_selects_canonical_sidecar(tmp_path):
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    canonical = tmp_path / "kws_latest.pt"

    assert dendritic._select_pai_resume_sidecar(
        run_dir, canonical, resume=None, resume_from=None
    ) is None

    canonical.write_bytes(b"sidecar")
    assert dendritic._select_pai_resume_sidecar(
        run_dir, canonical, resume=None, resume_from=None
    ) == canonical.resolve()


def test_completed_prune_kd_is_reused_when_only_pai_recipe_changes(tmp_path):
    source = tmp_path / "student.pt"
    teacher = tmp_path / "teacher.pt"
    source.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    run_id = "run-123"
    model_cfg = {"name": "candidate_w18", "block_channels": [18, 18]}
    old_train = {
        "epochs": 40,
        "lr": 0.001,
        "seed": 0,
        "pruning": {"kind": "structured", "keep_ratio": 0.45},
        "distillation": {"temperature": 1.0},
        "perforatedai": {"conversion": "blocks_and_linear"},
        "dendritic_schedule_epochs": 130,
    }
    new_train = {
        **old_train,
        "perforatedai": {
            "conversion": "fc_only",
            "post_integration_lr_multiplier": 0.25,
        },
    }
    torch.save(
        {
            "run_id": run_id,
            "stage": "sparsity",
            "phase": "prune_kd",
            "target_epochs": 40,
            "completed_epoch": 40,
            "best_metric_value": 0.81,
            "best_model_state_dict": {"weight": torch.ones(1)},
            "recipe": {
                "source_checkpoint": str(source),
                "teacher_checkpoint": str(teacher),
                "train": old_train,
                "pruning": old_train["pruning"],
            },
        },
        latest_path,
    )
    torch.save(
        {
            "run_id": run_id,
            "stage": "pruned_kd_finetune",
            "model_cfg": model_cfg,
            "sparsity": old_train["pruning"],
            "distillation": {
                "teacher_checkpoint": str(teacher),
                "teacher_sha256": dendritic.file_sha256(str(teacher)),
            },
            "val_acc": 0.81,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        best_path,
    )

    reused = dendritic._load_reusable_prune_finetune_checkpoint(
        best_path,
        latest_path,
        checkpoint_path=str(source),
        teacher_checkpoint=str(teacher),
        model_cfg=model_cfg,
        train_cfg=new_train,
        expected_run_id=run_id,
    )

    assert reused is not None
    assert reused["val_acc"] == 0.81


def test_completed_supervised_prune_is_reused_without_teacher(tmp_path):
    source = tmp_path / "student.pt"
    source.write_bytes(b"student")
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    model_cfg = {"name": "sparknet_c8", "channels": 8, "gate_channels": 32}
    train_cfg = {
        "epochs": 40,
        "lr": 0.001,
        "pruning": {"kind": "structured", "target_channels": 8},
        "perforatedai": {"module_ids": [".blocks.3", ".gate_conv", ".fc"]},
    }
    model_state = {"weight": torch.ones(1)}
    phase_train = dendritic._prune_finetune_train_recipe(train_cfg)
    torch.save(
        {
            "stage": "sparsity",
            "phase": "prune_supervised",
            "target_epochs": 40,
            "completed_epoch": 40,
            "best_metric_value": 0.82,
            "best_model_state_dict": model_state,
            "recipe": {
                "source_checkpoint": str(source),
                "teacher_checkpoint": None,
                "train": phase_train,
                "pruning": train_cfg["pruning"],
            },
        },
        latest_path,
    )
    torch.save(
        {
            "stage": "pruned_supervised_finetune",
            "model_cfg": model_cfg,
            "sparsity": train_cfg["pruning"],
            "distillation": None,
            "val_acc": 0.82,
            "model_state_dict": model_state,
        },
        best_path,
    )

    reused = dendritic._load_reusable_prune_finetune_checkpoint(
        best_path,
        latest_path,
        checkpoint_path=str(source),
        teacher_checkpoint=None,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        expected_run_id=None,
    )

    assert reused is not None
    assert reused["distillation"] is None


def test_completed_prune_kd_is_not_reused_after_phase_local_change(tmp_path):
    source = tmp_path / "student.pt"
    teacher = tmp_path / "teacher.pt"
    source.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    model_cfg = {"name": "candidate", "block_channels": [18, 18]}
    recorded_train = {
        "epochs": 40,
        "lr": 0.001,
        "pruning": {"kind": "structured", "keep_ratio": 0.45},
    }
    torch.save(
        {
            "stage": "sparsity",
            "phase": "prune_kd",
            "target_epochs": 40,
            "completed_epoch": 40,
            "recipe": {
                "source_checkpoint": str(source),
                "teacher_checkpoint": str(teacher),
                "train": recorded_train,
                "pruning": recorded_train["pruning"],
            },
        },
        latest_path,
    )
    torch.save(
        {
            "stage": "pruned_kd_finetune",
            "model_cfg": model_cfg,
            "sparsity": recorded_train["pruning"],
            "distillation": {
                "teacher_checkpoint": str(teacher),
                "teacher_sha256": dendritic.file_sha256(str(teacher)),
            },
            "val_acc": 0.81,
            "model_state_dict": {"weight": torch.ones(1)},
        },
        best_path,
    )

    changed = {**recorded_train, "lr": 0.0005}
    assert dendritic._load_reusable_prune_finetune_checkpoint(
        best_path,
        latest_path,
        checkpoint_path=str(source),
        teacher_checkpoint=str(teacher),
        model_cfg=model_cfg,
        train_cfg=changed,
        expected_run_id=None,
    ) is None


def test_completed_prune_kd_rejects_mismatched_best_and_latest_pair(tmp_path):
    source = tmp_path / "student.pt"
    teacher = tmp_path / "teacher.pt"
    source.write_bytes(b"student")
    teacher.write_bytes(b"teacher")
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    run_id = "run-123"
    model_cfg = {"name": "candidate", "block_channels": [18, 18]}
    train_cfg = {
        "epochs": 40,
        "lr": 0.001,
        "pruning": {"kind": "structured", "keep_ratio": 0.45},
    }
    torch.save(
        {
            "run_id": run_id,
            "stage": "sparsity",
            "phase": "prune_kd",
            "target_epochs": 40,
            "completed_epoch": 40,
            "best_metric_value": 0.81,
            "best_model_state_dict": {"weight": torch.zeros(1)},
            "recipe": {
                "source_checkpoint": str(source),
                "teacher_checkpoint": str(teacher),
                "train": train_cfg,
                "pruning": train_cfg["pruning"],
            },
        },
        latest_path,
    )
    torch.save(
        {
            "run_id": run_id,
            "stage": "pruned_kd_finetune",
            "model_cfg": model_cfg,
            "sparsity": train_cfg["pruning"],
            "distillation": {
                "teacher_checkpoint": str(teacher),
                "teacher_sha256": dendritic.file_sha256(str(teacher)),
            },
            "val_acc": 0.81,
            # This stale state does not match latest.pt's durable best.
            "model_state_dict": {"weight": torch.ones(1)},
        },
        best_path,
    )

    assert dendritic._load_reusable_prune_finetune_checkpoint(
        best_path,
        latest_path,
        checkpoint_path=str(source),
        teacher_checkpoint=str(teacher),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        expected_run_id=run_id,
    ) is None


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
        kd=cast(Any, FakeKD()),
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
        training_complete=True,
    )

    assert saved["kd_state_dict"]["marker"] == "persisted"
    assert saved["adapter_optimizer_state_dict"] is not None
    assert set(saved) == dendritic.PAI_SIDECAR_KEYS
    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    assert loaded["completed_epoch"] == 1
    assert loaded["training_complete"] is True

    # Sidecars written before the optional terminal marker was added remain
    # resumable and conservatively fall back to tracker-based detection.
    legacy = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    legacy.pop("training_complete")
    torch.save(legacy, sidecar_path)
    loaded_legacy = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    assert loaded_legacy["training_complete"] is False

    # The digest guard binds the sidecar to the KWS-owned copy it attested, not
    # to PAI's own latest.pt, which PAI overwrites on its own schedule.
    (run_dir / "kws_native_latest.pt").write_bytes(b"corrupt replacement")
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
            "training_complete": False,
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
        model=cast(Any, FakeModel()),
        optimizer=cast(Any, model_optimizer),
        scheduler=model_scheduler,
        kd=cast(Any, FakeKD()),
        adapter_optimizer=cast(Any, adapter_optimizer),
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
        optimizer=cast(Any, optimizer),
        adapter_optimizer=cast(Any, adapter_optimizer),
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


@pytest.mark.parametrize("with_kd", [False, True])
def test_pai_auxiliary_losses_are_logged_and_contribute_gradients(monkeypatch, with_kd):
    class _AuxiliaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.penalty = torch.nn.Parameter(torch.tensor(2.0))

        def auxiliary_losses(self):
            return {"gate": (self.penalty, 0.25)}

    model = _AuxiliaryModel()
    losses = {"total": model.penalty * 0}
    if with_kd:
        losses["classification"] = model.penalty * 0

    result = dendritic._add_auxiliary_losses(losses, model)
    result["total"].backward()

    assert result["gate"] is model.penalty
    assert result["total"].item() == pytest.approx(0.5)
    assert model.penalty.grad is not None
    assert model.penalty.grad.item() == pytest.approx(0.25)


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
        "lr": 0.001,
        "pruning": {"kind": "structured", "keep_ratio": 1.0},
        "augment": False,
        "perforatedai": {
            "enforce_base_weight_freeze": False,
            "post_integration_lr_multiplier": 0.25,
        },
        "label_smoothing": 0.0,
        "objective": "accuracy",
        "deployment": {
            "profile_device": "cpu",
            "bits_per_weight": 32,
            "latency_iterations": 1,
        },
    }
    tracker: Any = SimpleNamespace(member_vars={"mode": "n", "num_dendrites_integrated": 0})
    tracker.add_extra_score = lambda *_args: None

    def finish_with_internal_p_to_n_transition(_score, value):
        # PAI's terminal path enters p and returns to n within this call.  The
        # integration counter is the only externally observable transition.
        tracker.member_vars["num_dendrites_integrated"] = 1
        return value, True, True

    tracker.add_validation_score = finish_with_internal_p_to_n_transition
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)
    monkeypatch.setattr(dendritic, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        dendritic,
        "build_cycle_base",
        lambda *_args, **_kwargs: (model, checkpoint, model_cfg),
    )
    monkeypatch.setattr(dendritic, "cycle_fingerprint", lambda *_args: "recipe")
    estimated_conversions = []

    def estimate_params(_model, conversion="blocks_and_linear"):
        estimated_conversions.append(conversion)
        return 2

    monkeypatch.setattr(dendritic, "estimate_one_dendrite_params", estimate_params)
    dataset_build_kwargs = {}

    def build_train_and_val_datasets(*_args, **kwargs):
        dataset_build_kwargs.update(kwargs)
        return {dendritic.TRAIN: dataset, dendritic.VAL: dataset}, {"keyword": 0}

    monkeypatch.setattr(dendritic, "build_datasets", build_train_and_val_datasets)
    monkeypatch.setattr(
        dendritic,
        "build_data_loader",
        lambda value, *_args, **_kwargs: [
            (torch.stack([item[0] for item in value]), torch.stack([item[1] for item in value]))
        ],
    )

    def fake_prune_finetune(
        value,
        _datasets,
        _label_map,
        _model_cfg,
        _train_cfg,
        checkpoint_path,
        *_args,
        **_kwargs,
    ):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": value.state_dict(), "val_acc": 0.0},
            checkpoint_path,
        )
        return SimpleNamespace(best_val_acc=0.0)

    monkeypatch.setattr(dendritic, "train_model", fake_prune_finetune)
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
    optimizer_lr_multipliers = []

    def make_optimizer(*_args, **kwargs):
        optimizer_lr_multipliers.append(kwargs.get("lr_multiplier", 1.0))
        return next(optimizers)

    monkeypatch.setattr(dendritic, "_make_optimizer_and_scheduler", make_optimizer)
    paired_optimizers = []
    paired_training_complete = []

    def save_pair(**kwargs):
        paired_optimizers.append(kwargs["optimizer"])
        paired_training_complete.append(kwargs["training_complete"])
        return {"model_state_dict": kwargs["model"].state_dict()}

    monkeypatch.setattr(
        dendritic,
        "_save_pai_restart_pair",
        save_pair,
    )

    def export(value, save_name):
        path = Path(save_name) / "final_clean_pai.pt"
        path.write_bytes(b"clean")
        return value

    monkeypatch.setattr(dendritic, "export_final_pai_model", export)
    monkeypatch.setattr(dendritic, "read_pai_architecture_results", lambda _name: (0.0, 6))
    # ``run_cycle`` also reads PAI's minimum-parameter row, which lives in
    # the same CSV this test never writes.
    monkeypatch.setattr(dendritic, "read_pai_zero_dendrite_score", lambda _name: (0.0, 6))
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
    assert paired_training_complete == [True]
    assert optimizer_lr_multipliers == [1.0, 0.25]
    assert dataset_build_kwargs["splits"] == (dendritic.TRAIN, dendritic.VAL)
    assert estimated_conversions == ["blocks_and_linear"]
    cycle_paths = list((run_dir / "cycle_checkpoints").glob("*.pt"))
    assert [path.name for path in cycle_paths] == ["cycle_01_epoch_0001.pt"]
    cycle_state = torch.load(cycle_paths[0], map_location="cpu", weights_only=False)
    assert cycle_state["kind"] == "kws_pai_cycle_checkpoint"
    assert cycle_state["dendrites_integrated"] == 1


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
        cast(Any, teacher),
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


def test_post_pai_supervised_resume_never_constructs_distillation(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(dendritic, "freeze_selected_dendrites", lambda _model: 0)
    monkeypatch.setattr(dendritic, "build_data_loader", lambda *_args, **_kwargs: [])

    def reject_kd(*_args, **_kwargs):
        raise AssertionError("DistillationCriterion must not be built for --no-KD")

    monkeypatch.setattr(dendritic, "DistillationCriterion", reject_kd)

    def fake_finetune(value, *_args, **kwargs):
        with torch.no_grad():
            for parameter in value.parameters():
                parameter.add_(1)
        kwargs["on_best"](value, 0.75)
        return SimpleNamespace(best_val_acc=0.75, history=[{"val_acc": 0.75}])

    monkeypatch.setattr(dendritic, "run_finetune", fake_finetune)
    output_dir = tmp_path / "run"
    result = dendritic.resume_kd_finetune(
        model,
        {dendritic.TRAIN: [], dendritic.VAL: []},
        {
            "resume_epochs": 1,
            "seed": 0,
            "label_smoothing": 0.1,
            "distillation": {"temperature": 4.0},
        },
        torch.device("cpu"),
        None,
        (1, 2),
        "candidate",
        baseline_val_acc=0.5,
        output_dir=output_dir,
        num_classes=2,
        num_keywords=1,
    )

    best_path = dendritic.ArtifactLayout(output_dir).checkpoint_path(
        "sparsity", "resume_supervised", "best", candidate="candidate"
    )
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    assert result["status"] == "complete"
    assert result["best_val_acc"] == 0.75
    assert result["checkpoint"] == str(best_path)
    assert checkpoint["distillation"] is None
    assert checkpoint["teacher_checkpoint"] is None


@pytest.mark.parametrize(
    ("cli_args", "expected_resume", "expected_resume_from", "expected_teacher"),
    [
        (["--resume"], True, None, "models/checkpoints/ds_cnn_l_12class.pt"),
        (
            ["--resume-from", "explicit-sidecar.pt"],
            False,
            "explicit-sidecar.pt",
            "models/checkpoints/ds_cnn_l_12class.pt",
        ),
        (["--no-KD"], False, None, None),
    ],
)
def test_standalone_cli_forwards_resume_options(
    cli_args, expected_resume, expected_resume_from, expected_teacher, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "sys.argv", ["kws.optimize.dendritic", *cli_args]
    )
    monkeypatch.setattr(dendritic, "load_yaml", lambda path: {"path": path})
    monkeypatch.setattr(dendritic, "run_session", lambda *_args, **_kwargs: cast(Any, nullcontext)())
    monkeypatch.setattr(
        dendritic,
        "run_cycle",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    dendritic.main()

    assert captured["resume"] is expected_resume
    assert captured["resume_from"] == expected_resume_from
    assert captured["teacher_checkpoint"] == expected_teacher


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
    assert model.base_bn.running_mean is not None
    assert model.dendrite_bn.running_mean is not None
    before_base_mean = model.base_bn.running_mean.clone()
    before_dendrite_mean = model.dendrite_bn.running_mean.clone()

    assert dendritic.freeze_base_batchnorm_stats(model) == 1
    assert not model.base_bn.training
    assert model.dendrite_bn.training
    assert model.base_bn.weight.requires_grad

    model(torch.full((4, 3), 2.0)).sum().backward()

    assert model.base_bn.running_mean is not None
    assert model.dendrite_bn.running_mean is not None
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


def _save_one_pair(tmp_path, monkeypatch, *, completed_epoch=1, native_value=0.0):
    """Commit one real PAI/KWS restart pair through the production writer."""
    run_dir = tmp_path / "candidate"
    run_dir.mkdir(exist_ok=True)
    sidecar_path = tmp_path / "kws_latest.pt"
    saved = _save_one_pair_into(
        run_dir,
        sidecar_path,
        monkeypatch,
        completed_epoch=completed_epoch,
        native_value=native_value,
    )
    return run_dir, sidecar_path, saved


def _save_one_pair_into(
    run_dir, sidecar_path, monkeypatch, *, completed_epoch=1, native_value=0.0
):
    model = torch.nn.Linear(2, 2)
    # PAI serializes the very model whose state_dict the sidecar stores in the
    # same commit.  The fixture has to preserve that identity: it is what lets
    # an attested native blob be rebuilt from the sidecar alone.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(native_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    def save_system(saved_model, folder, name):
        target = Path(folder) / f"{name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                key: value.contiguous()
                for key, value in saved_model.state_dict().items()
            },
            target,
        )

    monkeypatch.setattr(dendritic.UPA, "save_system", save_system)
    saved = dendritic._save_pai_restart_pair(
        model=model,
        run_dir=run_dir,
        pai_run_name=run_dir.name,
        sidecar_path=sidecar_path,
        optimizer=optimizer,
        scheduler=scheduler,
        kd=None,
        adapter_optimizer=None,
        adapter_scheduler=None,
        phase_trail=[{"epoch": 0, "mode": "n"}],
        completed_epoch=completed_epoch,
        global_step=4,
        history_length=1,
        last_metric_digest="metrics-digest",
        teacher_checkpoint=None,
        recipe_fingerprint="recipe",
        loaders=(SimpleNamespace(generator=None), SimpleNamespace(generator=None)),
    )
    return saved


def test_attested_pair_survives_pai_overwriting_its_own_latest_mid_epoch(
    tmp_path, monkeypatch
):
    # PAI's tracker writes <candidate>/latest.pt at validation, one step before
    # the epoch's pair is committed.  A kill inside that window used to destroy
    # the bytes the newest sidecar had signed, so the candidate could never be
    # resumed.  The attested half must be a file PAI does not write.
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)

    save_file({"native": torch.full((1,), 9.0)}, run_dir / "latest.pt")

    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    assert loaded["completed_epoch"] == 1


def test_resume_loads_the_native_half_the_sidecar_attests(tmp_path, monkeypatch):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    state = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    requested = []

    def load_system(net, folder, name, **kwargs):
        requested.append((Path(folder), name, kwargs))
        return net

    monkeypatch.setattr(dendritic.UPA, "load_system", load_system)
    model = torch.nn.Linear(2, 2)

    assert (
        dendritic._load_paired_native_state(model, run_dir, state, sidecar_path)
        is model
    )
    assert requested == [
        (run_dir, "kws_native_latest", {"load_from_restart": True})
    ]


def test_legacy_sidecar_attesting_pais_own_latest_still_resumes(tmp_path, monkeypatch):
    # Candidates written before the split attest <candidate>/latest.pt.  They
    # keep the old tear risk, but refusing them would strand every in-flight run.
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    legacy_native = run_dir / "latest.pt"
    state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    state["native_pai_latest"] = str(legacy_native)
    state["native_pai_latest_sha256"] = dendritic.sha256_path(legacy_native)
    torch.save(state, sidecar_path)

    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )
    assert (
        dendritic._attested_native_path(run_dir, loaded, source=sidecar_path)
        == legacy_native.resolve()
    )


def test_sidecar_rejects_a_native_reference_outside_the_candidate(
    tmp_path, monkeypatch
):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    intruder = tmp_path / "kws_native_latest.pt"
    intruder.write_bytes((run_dir / "kws_native_latest.pt").read_bytes())
    state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    state["native_pai_latest"] = str(intruder)
    torch.save(state, sidecar_path)

    with pytest.raises(ValueError, match="references native state"):
        dendritic._load_pai_sidecar(
            sidecar_path, run_dir, "recipe", torch.device("cpu")
        )


def test_torn_pair_error_names_the_newest_cycle_snapshot(tmp_path, monkeypatch):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    snapshots = run_dir / "cycle_checkpoints"
    snapshots.mkdir()
    (snapshots / "cycle_01_epoch_0007.pt").write_bytes(b"first")
    (snapshots / "cycle_02_epoch_0311.pt").write_bytes(b"newest")
    (run_dir / "kws_native_latest.pt").write_bytes(b"corrupt replacement")
    _make_tear_unrepairable(sidecar_path)

    with pytest.raises(ValueError, match="cycle_02_epoch_0311.pt"):
        dendritic._load_pai_sidecar(
            sidecar_path, run_dir, "recipe", torch.device("cpu")
        )


def test_torn_pair_error_says_so_when_no_snapshot_exists(tmp_path, monkeypatch):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    (run_dir / "kws_native_latest.pt").write_bytes(b"corrupt replacement")
    _make_tear_unrepairable(sidecar_path)

    with pytest.raises(ValueError, match="no cycle_checkpoints snapshot"):
        dendritic._load_pai_sidecar(
            sidecar_path, run_dir, "recipe", torch.device("cpu")
        )


def test_pair_status_reports_an_intact_pair_without_the_cycle_recipe(
    tmp_path, monkeypatch
):
    run_dir, sidecar_path, _ = _save_one_pair(
        tmp_path, monkeypatch, completed_epoch=655
    )

    status = dendritic.pai_restart_pair_status(sidecar_path, run_dir)

    assert status.paired is True
    assert status.completed_epoch == 655
    assert status.native == (run_dir / "kws_native_latest.pt").resolve()


def test_pair_status_reports_a_torn_pair_instead_of_raising(tmp_path, monkeypatch):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    (run_dir / "kws_native_latest.pt").write_bytes(b"corrupt replacement")

    status = dendritic.pai_restart_pair_status(sidecar_path, run_dir)

    assert status.paired is False
    assert status.completed_epoch == 1
    assert "digest" in status.detail


def test_run_pair_statuses_cover_every_candidate_with_a_committed_sidecar(
    tmp_path, monkeypatch
):
    # A candidate that has not committed a pair yet has nothing attested and so
    # nothing to tear; only candidates with a sidecar are a liveness concern.
    run_root = tmp_path / "seed2"
    intact_dir = run_root / "pai" / "candidates" / "intact"
    torn_dir = run_root / "pai" / "candidates" / "torn"
    (run_root / "pai" / "candidates" / "not_started_yet").mkdir(parents=True)
    for candidate_dir in (intact_dir, torn_dir):
        candidate_dir.mkdir(parents=True)
        sidecar = (
            run_root
            / "models"
            / "checkpoints"
            / "sparsity"
            / candidate_dir.name
            / "pai"
            / "latest.pt"
        )
        sidecar.parent.mkdir(parents=True)
        _save_one_pair_into(
            candidate_dir, sidecar, monkeypatch, completed_epoch=655
        )
    (torn_dir / "kws_native_latest.pt").write_bytes(b"clobbered by PAI")

    statuses = dendritic.pai_restart_pair_statuses(run_root)

    assert [
        (status.sidecar.parents[1].name, status.paired) for status in statuses
    ] == [("intact", True), ("torn", False)]
    assert all(status.completed_epoch == 655 for status in statuses)


def test_a_live_candidate_inside_the_vulnerable_window_is_not_reported_as_torn():
    # A candidate still training reads as torn for the 1-2 s between PAI's own
    # latest.pt write and the epoch's pair, so one sample cannot distinguish a
    # live writer from a dead one.  An advancing epoch can.
    sidecar = Path("sidecar.pt")
    first = dendritic.PaiPairStatus(sidecar, None, 464, False, "digest mismatch")
    later = dendritic.PaiPairStatus(sidecar, None, 465, False, "digest mismatch")

    assert dendritic.is_confirmed_torn(first, later) is False


def test_a_stalled_candidate_that_stays_unpaired_is_reported_as_torn():
    sidecar = Path("sidecar.pt")
    first = dendritic.PaiPairStatus(sidecar, None, 655, False, "digest mismatch")
    later = dendritic.PaiPairStatus(sidecar, None, 655, False, "digest mismatch")

    assert dendritic.is_confirmed_torn(first, later) is True


def test_a_pair_that_healed_by_the_second_reading_is_not_torn():
    sidecar = Path("sidecar.pt")
    first = dendritic.PaiPairStatus(sidecar, None, 655, False, "digest mismatch")
    later = dendritic.PaiPairStatus(sidecar, None, 655, True, "paired")

    assert dendritic.is_confirmed_torn(first, later) is False


def _make_tear_unrepairable(sidecar_path):
    """Leave a sidecar whose own model state can no longer reproduce its digest.

    Rebuilding the attested blob from the sidecar heals an ordinary tear, so a
    test about the *unrecoverable* case has to destroy the second copy too.
    """
    state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    state["model_state_dict"] = {"weight": torch.zeros(2, 2)}
    torch.save(state, sidecar_path)


def test_a_torn_pair_is_rebuilt_from_the_sidecars_own_model_state(
    tmp_path, monkeypatch
):
    # The sidecar stores the same state_dict PAI serialized in the same commit,
    # so the attested bytes are never really lost: they can be regenerated and
    # checked against the digest that names them.
    run_dir, sidecar_path, saved = _save_one_pair(
        tmp_path, monkeypatch, native_value=1.5
    )
    attested = run_dir / "kws_native_latest.pt"
    save_file({"native": torch.full((1,), 9.0)}, attested)

    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )

    assert loaded["completed_epoch"] == 1
    assert dendritic.sha256_path(attested) == saved["native_pai_latest_sha256"]
    assert bool(load_file(attested)["weight"].eq(1.5).all())


def test_a_missing_native_half_is_rebuilt_from_the_sidecar(tmp_path, monkeypatch):
    run_dir, sidecar_path, saved = _save_one_pair(tmp_path, monkeypatch)
    (run_dir / "kws_native_latest.pt").unlink()

    dendritic._load_pai_sidecar(sidecar_path, run_dir, "recipe", torch.device("cpu"))

    assert (
        dendritic.sha256_path(run_dir / "kws_native_latest.pt")
        == saved["native_pai_latest_sha256"]
    )


def test_a_legacy_torn_pair_is_rebuilt_under_the_kws_owned_name(tmp_path, monkeypatch):
    # A pre-split sidecar attests PAI's own latest.pt.  Healing it must not
    # write through PAI's file: the attested bytes are published under the name
    # KWS owns and the sidecar is repointed there, which also migrates the
    # candidate off the shared path that tore it.
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch, native_value=2.5)
    legacy_native = run_dir / "latest.pt"
    legacy_digest = dendritic.sha256_path(legacy_native)
    state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    state["native_pai_latest"] = str(legacy_native)
    state["native_pai_latest_sha256"] = legacy_digest
    torch.save(state, sidecar_path)
    (run_dir / "kws_native_latest.pt").unlink()
    save_file({"native": torch.full((1,), 9.0)}, legacy_native)

    loaded = dendritic._load_pai_sidecar(
        sidecar_path, run_dir, "recipe", torch.device("cpu")
    )

    repaired = dendritic._attested_native_path(run_dir, loaded, source=sidecar_path)
    assert repaired == (run_dir / "kws_native_latest.pt").resolve()
    assert dendritic.sha256_path(repaired) == legacy_digest
    assert load_file(legacy_native)["native"].item() == 9.0


def test_a_sidecar_that_cannot_reproduce_its_digest_still_refuses_to_resume(
    tmp_path, monkeypatch
):
    run_dir, sidecar_path, _ = _save_one_pair(tmp_path, monkeypatch)
    (run_dir / "kws_native_latest.pt").write_bytes(b"corrupt replacement")
    _make_tear_unrepairable(sidecar_path)

    with pytest.raises(ValueError, match="refusing to resume an unpaired candidate"):
        dendritic._load_pai_sidecar(
            sidecar_path, run_dir, "recipe", torch.device("cpu")
        )


def test_a_tear_the_sidecar_can_heal_is_not_reported_as_torn(tmp_path, monkeypatch):
    # Monitoring exists to say "this candidate cannot resume".  A tear the
    # sidecar heals by itself resumes fine, so paging on it is a false alarm.
    run_dir, sidecar_path, _ = _save_one_pair(
        tmp_path, monkeypatch, completed_epoch=655
    )
    (run_dir / "kws_native_latest.pt").write_bytes(b"clobbered by PAI")
    before = sorted(path.name for path in run_dir.iterdir())

    status = dendritic.pai_restart_pair_status(sidecar_path, run_dir)

    assert status.paired is False
    assert status.repairable is True
    assert dendritic.is_confirmed_torn(status, status) is False
    # Looking at a candidate must not change it.
    assert sorted(path.name for path in run_dir.iterdir()) == before
    assert (run_dir / "kws_native_latest.pt").read_bytes() == b"clobbered by PAI"


def test_a_tear_the_sidecar_cannot_heal_is_still_confirmed_torn(tmp_path, monkeypatch):
    run_dir, sidecar_path, _ = _save_one_pair(
        tmp_path, monkeypatch, completed_epoch=655
    )
    (run_dir / "kws_native_latest.pt").write_bytes(b"clobbered by PAI")
    _make_tear_unrepairable(sidecar_path)

    status = dendritic.pai_restart_pair_status(sidecar_path, run_dir)

    assert status.repairable is False
    assert dendritic.is_confirmed_torn(status, status) is True
