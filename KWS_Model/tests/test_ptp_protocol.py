"""The "Pruning Then Perforating" protocol: plateau/early stopping, group
prune rates in the SparkNet experiment, PAI-phase plateau + eval logging, and
the offline analysis math."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import yaml

from kws import train as train_module
from kws.models.sparknet import SparkNet
from kws.optimize import dendritic
from kws.optimize.sparknet_dendritic_prune_experiment import (
    load_config,
    resolve_group_prune_plan,
    run_experiment,
    validate_config,
)
from kws.train import (
    EarlyStopping,
    build_plateau_scheduler,
    resolve_scheduler_name,
    run_finetune,
)
from scripts import analyze_ptp

ROOT = Path(__file__).parents[1]


# --------------------------------------------------------------------------
# train.py: plateau scheduler and early stopping
# --------------------------------------------------------------------------


def test_scheduler_name_alias_and_default_are_backward_compatible():
    assert resolve_scheduler_name({}) == "cosine"
    assert resolve_scheduler_name({"scheduler": "polynomial_hold"}) == "polynomial_hold"
    assert resolve_scheduler_name({"lr_scheduler": "plateau"}) == "plateau"
    assert resolve_scheduler_name({"scheduler": "plateau", "lr_scheduler": "PLATEAU"}) == "plateau"
    with pytest.raises(ValueError, match="disagree"):
        resolve_scheduler_name({"scheduler": "cosine", "lr_scheduler": "plateau"})
    assert EarlyStopping.from_config({}) is None


def test_plateau_scheduler_follows_the_paper_defaults():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    scheduler = build_plateau_scheduler(optimizer, {"plateau": {"patience": 3}})
    assert scheduler.mode == "max"
    assert scheduler.factor == pytest.approx(0.1)
    assert scheduler.min_lrs == [pytest.approx(5e-6)]
    lrs = []
    for _ in range(12):
        scheduler.step(0.5)  # a flat validation accuracy
        lrs.append(optimizer.param_groups[0]["lr"])
    # First step sets the best; the 4th bad epoch (> patience 3) decays.
    assert lrs[:4] == [pytest.approx(5e-4)] * 4
    assert lrs[4] == pytest.approx(5e-5)
    assert lrs[-1] == pytest.approx(5e-6)  # floored at 0.01x the initial rate
    with pytest.raises(ValueError, match="unknown plateau"):
        build_plateau_scheduler(optimizer, {"plateau": {"patiance": 3}})


def test_early_stopping_counts_only_min_delta_improvements():
    stopper = EarlyStopping(patience=3, min_delta=0.001)
    values = [0.50, 0.5005, 0.5009, 0.5020, 0.5025, 0.5029, 0.5030]
    stops = [stopper.update(v, epoch) for epoch, v in enumerate(values, start=1)]
    # 0.5005 and 0.5009 are within min_delta of 0.50; 0.5020 resets; then
    # three epochs without a >0.001 gain stop at epoch 7.
    assert stops == [False, False, False, False, False, False, True]
    assert stopper.stopped_epoch == 7
    clone = EarlyStopping(patience=3, min_delta=0.001)
    clone.load_state_dict(stopper.state_dict())
    assert clone.state_dict() == stopper.state_dict()
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping.from_config({"early_stopping": {"patience": 0}})


def _toy_loaders():
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(16, 4, generator=generator)
    labels = (features[:, 0] > 0).long()
    batches = [(features[:8], labels[:8]), (features[8:], labels[8:])]
    return batches, batches


def _ptp_train_cfg(**overrides):
    cfg = {
        "epochs": 400,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "optimizer": "adamw",
        "lr_scheduler": "plateau",
        "plateau": {"mode": "max", "factor": 0.1, "patience": 3, "min_lr_scale": 0.01},
        "early_stopping": {"patience": 8, "min_delta": 0.001},
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def test_run_finetune_stops_early_restores_best_and_resumes_as_done(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    train_loader, val_loader = _toy_loaders()
    # Scripted validation accuracy: best at epoch 3, a sub-min-delta blip at
    # epoch 5, then flat -> stop at epoch 3 + 8 = 11 (0.8005 is not > 0.801).
    scripted = iter([0.5, 0.7, 0.8, 0.79, 0.8005] + [0.6] * 100)
    monkeypatch.setattr(
        train_module, "evaluate_loss_acc", lambda *_args, **_kwargs: (1.0, next(scripted))
    )
    snapshots: dict[int, dict[str, torch.Tensor]] = {}

    def on_epoch_end(record):
        snapshots[record["epoch"]] = copy.deepcopy(model.state_dict())

    latest = tmp_path / "latest.pt"
    result = run_finetune(
        model, train_loader, val_loader, torch.device("cpu"), _ptp_train_cfg(),
        on_epoch_end=on_epoch_end, latest_path=latest,
    )
    assert result.completed_epoch == 11
    assert result.epochs == 11
    # The strictly-best epoch is 5 (0.8005 > 0.8) even though it did not
    # count as an early-stopping improvement; its weights are restored.
    assert result.best_val_acc == pytest.approx(0.8005)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, snapshots[5][name])
    lrs = [record["learning_rate"][0] for record in result.history]
    assert lrs[0] == pytest.approx(5e-4)
    assert min(lrs) < 5e-4  # the plateau decayed during the flat tail
    assert result.history[-1]["early_stopping_wait"] == 8

    state = torch.load(latest, map_location="cpu", weights_only=False)
    assert state["stage_specific_state"]["early_stopping"]["stopped"] is True
    assert dendritic._prune_finetune_early_stopped(state)
    resumed = run_finetune(
        nn.Linear(4, 2), train_loader, val_loader, torch.device("cpu"), _ptp_train_cfg(),
        resume_state=state, latest_path=tmp_path / "latest2.pt",
    )
    assert resumed.completed_epoch == 11  # nothing more was trained
    assert len(resumed.history) == 11


def test_run_finetune_without_new_keys_keeps_fixed_epochs(monkeypatch):
    model = nn.Linear(4, 2)
    train_loader, val_loader = _toy_loaders()
    monkeypatch.setattr(train_module, "evaluate_loss_acc", lambda *_a, **_k: (1.0, 0.5))
    result = run_finetune(
        model, train_loader, val_loader, torch.device("cpu"),
        {"epochs": 12, "lr": 1e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
         "warmup_fraction": 0.1, "seed": 0},
    )
    assert result.completed_epoch == 12
    assert "early_stopping_wait" not in result.history[-1]


# --------------------------------------------------------------------------
# Experiment runner: pruning.prune_rates with group_prune
# --------------------------------------------------------------------------


@pytest.fixture
def ptp_config(tmp_path):
    input_shape = (32, 5)
    source_cfg = {"family": "sparknet", "name": "sparknet_c16_paper", "channels": 16,
                  "gate_channels": 32, "sparsity_weight": 1.0}
    torch.manual_seed(0)
    source = SparkNet(32, 12, channels=16, gate_channels=32, input_shape=input_shape)
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict(), "model_cfg": source_cfg,
                "input_shape": input_shape, "num_classes": 12, "num_keywords": 10,
                "val_acc": 0.95}, checkpoint)
    for name, payload in (
        ("data.yaml", {"features": {"type": "mfcc", "n_mels": 32}}),
        ("model.yaml", source_cfg),
    ):
        (tmp_path / name).write_text(yaml.safe_dump(payload))
    pai = copy.deepcopy(load_config(ROOT / "configs/train/sparknet_ptp_paper.yaml")["perforatedai"])
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(
        {"objective": {"metric": "validation_accuracy", "use_test": False},
         "perforatedai": pai, "pai_eval_splits": ["test", "train"], "seed": 0}
    ))
    return {
        "source_checkpoint": str(checkpoint),
        "source_channels": 16,
        "teacher_checkpoint": None,
        "data_config": str(tmp_path / "data.yaml"),
        "model_config": str(tmp_path / "model.yaml"),
        "train_config": str(tmp_path / "train.yaml"),
        "output_dir": str(tmp_path / "run"),
        "seed": 0,
        "objective": {"metric": "validation_accuracy", "use_test": False},
        "perforatedai": pai,
        "pruning": {"method": "group", "criterion": "l2_group",
                    "prune_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
    }


EXPECTED_C16_PLAN = {  # rate -> (width, trainable params) for C16 / gate 32
    0.1: (15, 4309), 0.2: (13, 3691), 0.3: (11, 3121), 0.4: (10, 2854),
    0.5: (8, 2356), 0.6: (6, 1906), 0.7: (3, 1321),
}


def test_shipped_ptp_configs_validate_and_match_their_train_config():
    train = load_config(ROOT / "configs/train/sparknet_ptp_paper.yaml")
    for seed in range(6):
        cfg = validate_config(load_config(ROOT / f"configs/experiment/sparknet_c16_ptp_seed{seed}.yaml"))
        assert cfg["perforatedai"] == train["perforatedai"]
        assert cfg["pruning"]["prune_rates"] == list(EXPECTED_C16_PLAN)
        assert cfg["seed"] == seed


def test_group_rates_resolve_to_widths_and_reject_collisions(ptp_config):
    source = SparkNet(32, 12, channels=16, gate_channels=32, input_shape=(32, 5))
    plan = resolve_group_prune_plan(source, list(EXPECTED_C16_PLAN))
    assert {item["prune_rate"]: (item["width"], item["pruned_params"]) for item in plan} == EXPECTED_C16_PLAN
    assert all(item["source_params"] == 4636 for item in plan)
    with pytest.raises(ValueError, match="both resolve to width 1"):
        resolve_group_prune_plan(source, [0.78, 0.8])
    bad = copy.deepcopy(ptp_config)
    bad["widths"] = [8]
    with pytest.raises(ValueError, match="remove widths"):
        validate_config(bad)
    bad = copy.deepcopy(ptp_config)
    bad["pruning"]["prune_rates"] = [0.5, 0.3]
    with pytest.raises(ValueError, match="ascending"):
        validate_config(bad)
    bad["pruning"] = {"method": "group", "prune_rates": [0.3], "criterion": "l1"}
    with pytest.raises(ValueError, match="criterion"):
        validate_config(bad)


def test_group_prune_path_accepts_a_reduced_gate_source(tmp_path, ptp_config):
    cfg = copy.deepcopy(ptp_config)
    source_cfg = {"family": "sparknet", "name": "sparknet_c16g16", "channels": 16,
                  "gate_channels": 16, "sparsity_weight": 1.0}
    source = SparkNet(32, 12, channels=16, gate_channels=16, input_shape=(32, 5))
    torch.save({"model_state_dict": source.state_dict(), "model_cfg": source_cfg,
                "input_shape": (32, 5), "num_classes": 12, "val_acc": 0.9},
               tmp_path / "g16.pt")
    (tmp_path / "g16.yaml").write_text(yaml.safe_dump(source_cfg))
    cfg.update(source_checkpoint=str(tmp_path / "g16.pt"), model_config=str(tmp_path / "g16.yaml"))
    cfg["pruning"]["prune_rates"] = [0.3, 0.5]
    report = run_experiment(cfg, dry_run=True, output_dir=tmp_path / "g16-run")
    assert [c["group_prune"]["prune_rate"] for c in report["candidates"]] == [0.3, 0.5]
    # The legacy l1_filter path still refuses a non-32 gate.
    legacy = copy.deepcopy(cfg)
    legacy["pruning"] = {"method": "l1_filter"}
    legacy["widths"] = [8]
    with pytest.raises(ValueError, match="gate width"):
        run_experiment(legacy, dry_run=True, output_dir=tmp_path / "legacy")


def test_group_prune_execution_materialises_the_cycle_source(tmp_path, ptp_config):
    ptp_config["pruning"]["prune_rates"] = [0.2, 0.5]
    calls: list[dict[str, Any]] = []

    def fake_cycle(checkpoint_path, data_cfg, model_cfg, train_cfg, save_name, **kwargs):
        width = model_cfg["channels"]
        # The real cycle loads this checkpoint as an identity-width source.
        base, _, built_cfg = dendritic.build_cycle_base(
            checkpoint_path, train_cfg["pruning"]["keep_ratio"],
            target_model_cfg=model_cfg, expected_run_id=kwargs["run_id"],
        )
        calls.append({"path": checkpoint_path, "train_cfg": train_cfg, "base": base,
                      "model_cfg": built_cfg})
        params = sum(p.numel() for p in base.parameters())
        return {
            "best_val_acc": 0.9, "pai_best_val_acc": 0.9,
            "deployed_params": params + 396, "cost": {"macs": 1000 * width},
            "prune_finetune": {"best_val_acc": 0.88, "checkpoint": "x.pt",
                               "reused": False, "distillation": None},
            "resume": {"status": "skipped"},
        }

    report = run_experiment(ptp_config, output_dir=tmp_path / "run", cycle_runner=fake_cycle)
    assert report["status"] == "complete"
    assert report["widths"] == [13, 8]
    assert report["test_split_used"] is True
    assert report["test_split_role"] == "per_epoch_logging_only"
    for call, (rate, width, params) in zip(calls, [(0.2, 13, 3691), (0.5, 8, 2356)]):
        assert "group_prune" in call["path"] and call["path"] != ptp_config["source_checkpoint"]
        assert call["train_cfg"]["pruning"]["keep_ratio"] == 1.0
        assert call["train_cfg"]["pruning"]["prune_rate"] == rate
        assert call["train_cfg"]["pruning"]["method"] == "group"
        assert call["base"].blocks[0].pointwise.out_channels == width
        assert sum(p.numel() for p in call["base"].parameters()) == params
        saved = torch.load(call["path"], map_location="cpu", weights_only=False)
        assert saved["group_prune"]["criterion"] == "l2_group"
        assert saved["group_prune"]["target_rate"] == rate
        assert saved["model_cfg"]["channels"] == width
    for candidate, (rate, params) in zip(report["candidates"], [(0.2, 3691), (0.5, 2356)]):
        group = candidate["group_prune"]
        assert group["prune_rate"] == rate
        assert group["pruned_params"] == params
        assert group["achieved_rate"] == pytest.approx(1 - params / 4636)
        assert candidate["baseline"]["deployed_params"] == params


# --------------------------------------------------------------------------
# dendritic.py: PAI settings, plateau in the PAI phase, per-epoch eval logging
# --------------------------------------------------------------------------


def test_configure_perforatedai_applies_explicit_retain_and_output_dimensions():
    pc = dendritic.GPA.pc
    before = (pc.get_retain_all_dendrites(), list(pc.get_output_dimensions()))
    pai = load_config(ROOT / "configs/train/sparknet_ptp_paper.yaml")["perforatedai"]
    try:
        pc.set_retain_all_dendrites(True)
        dendritic.configure_perforatedai(pai, torch.device("cpu"))
        assert pc.get_retain_all_dendrites() is False
        assert list(pc.get_output_dimensions()) == [-1, 0]
        assert pc.get_max_dendrites() == 3
        assert pc.get_max_dendrite_tries() == 3
        with pytest.raises(ValueError, match="output_dimensions"):
            dendritic.configure_perforatedai(
                {**pai, "output_dimensions": [-1, -1]}, torch.device("cpu")
            )
    finally:
        pc.set_retain_all_dendrites(before[0])
        pc.set_output_dimensions(before[1])


def test_pai_phase_knobs_default_off():
    assert dendritic.pai_uses_plateau({}) is False
    assert dendritic.pai_eval_splits({}) == ()
    assert dendritic.pai_uses_plateau({"pai_lr_scheduler": "plateau"}) is True
    with pytest.raises(ValueError):
        dendritic.pai_eval_splits({"pai_eval_splits": ["val"]})


class _Named(list):
    def __init__(self, name, items):
        super().__init__(items)
        self.name = name


class _Stop(Exception):
    pass


def test_pai_phase_plateau_and_per_epoch_test_train_logging(tmp_path, monkeypatch):
    model = nn.Linear(2, 2)
    checkpoint = {"input_shape": (1, 2), "num_keywords": 1, "num_classes": 2, "val_acc": 0.0}
    model_cfg = {"block_channels": [2], "name": "tiny"}
    sample = [(torch.ones(2), torch.tensor(0))]
    train_cfg = {
        "seed": 3, "lr": 1e-3, "weight_decay": 0.0, "label_smoothing": 0.0,
        "augment": False, "objective": "accuracy",
        "pruning": {"kind": "structured", "keep_ratio": 1.0},
        "pai_lr_scheduler": "plateau",
        "plateau": {"patience": 3, "factor": 0.1, "min_lr_scale": 0.01},
        "pai_eval_splits": ["test", "train"],
        "perforatedai": {"enforce_base_weight_freeze": False,
                         "post_integration_lr_multiplier": 1.0},
    }
    epochs = 8
    tracker: Any = SimpleNamespace(member_vars={"mode": "n", "num_dendrites_integrated": 0})
    tracker.add_extra_score = lambda *_a: None
    tracker.set_optimizer_instance = lambda *_a: None
    counter = {"epoch": 0}

    def add_validation_score(_score, value):
        counter["epoch"] += 1
        return value, False, counter["epoch"] >= epochs

    tracker.add_validation_score = add_validation_score
    monkeypatch.setattr(dendritic.GPA, "pai_tracker", tracker)
    monkeypatch.setattr(dendritic, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(dendritic, "build_cycle_base", lambda *_a, **_k: (model, checkpoint, model_cfg))
    monkeypatch.setattr(dendritic, "cycle_fingerprint", lambda *_a: "recipe")
    monkeypatch.setattr(dendritic, "estimate_one_dendrite_params", lambda *_a, **_k: 2)
    requested = {}

    def build_datasets(*_a, **kwargs):
        requested.update(kwargs)
        return ({dendritic.TRAIN: _Named("train", sample), dendritic.VAL: _Named("val", sample),
                 dendritic.TEST: _Named("test", sample)}, {"keyword": 0})

    monkeypatch.setattr(dendritic, "build_datasets", build_datasets)
    monkeypatch.setattr(dendritic, "clean_train_view", lambda ds: _Named("train_eval", ds))

    def loader(value, *_a, **_k):
        batch = (torch.stack([x for x, _ in value]), torch.stack([y for _, y in value]))
        return _Named(value.name, [batch])

    monkeypatch.setattr(dendritic, "build_data_loader", loader)
    accuracy = {"val": 0.5, "test": 0.4, "train_eval": 0.6}
    monkeypatch.setattr(
        dendritic, "evaluate_loss_acc",
        lambda _m, data, *_a: (1.0, accuracy[data.name]),
    )

    def fake_prune_finetune(value, _d, _l, _m, _t, checkpoint_path, *_a, **_k):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": value.state_dict(), "val_acc": 0.0}, checkpoint_path)
        return SimpleNamespace(best_val_acc=0.0)

    monkeypatch.setattr(dendritic, "train_model", fake_prune_finetune)
    monkeypatch.setattr(dendritic, "configure_perforatedai", lambda *_a: None)
    monkeypatch.setattr(dendritic.UPA, "perforate_model", lambda value, **_k: value)
    monkeypatch.setattr(dendritic.UPA, "count_params",
                        lambda value: sum(p.numel() for p in value.parameters()))
    monkeypatch.setattr(dendritic, "_save_pai_restart_pair",
                        lambda **kw: {"model_state_dict": kw["model"].state_dict()})
    monkeypatch.setattr(dendritic, "export_final_pai_model", lambda value, _name: value)

    def stop(_name):
        raise _Stop

    monkeypatch.setattr(dendritic, "read_pai_architecture_results", stop)
    out = tmp_path / "run"
    with pytest.raises(_Stop):
        dendritic.run_cycle("source.pt", {}, model_cfg, train_cfg, "tiny_cycle",
                            output_dir=out, resume=False)

    assert requested["splits"] == (dendritic.TRAIN, dendritic.VAL, dendritic.TEST)
    log = out / "metrics" / "sparsity" / "tiny_cycle" / "pai.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == epochs
    assert all(r["test_accuracy"] == 0.4 for r in records)
    assert all(r["train_eval_accuracy"] == 0.6 for r in records)
    assert all(r["evaluated_parameter_count"] == 6 for r in records)
    lrs = [r["learning_rate"][0] for r in records]
    # Flat validation: epochs 1-5 at 1e-3, the 5th step decays (patience 3).
    assert lrs[:5] == [pytest.approx(1e-3)] * 5
    assert lrs[5] == pytest.approx(1e-4)


# --------------------------------------------------------------------------
# scripts/analyze_ptp.py
# --------------------------------------------------------------------------


def _record(epoch, val, test, params, *, evaluated=None, train=None, mode="n"):
    record = {"epoch": epoch, "val_acc": val, "test_accuracy": test,
              "parameter_count": params, "pai_mode": mode, "train_accuracy": 0.9}
    if evaluated is not None:
        record["evaluated_parameter_count"] = evaluated
    if train is not None:
        record["train_eval_accuracy"] = train
    return record


def test_dendrite_count_uses_evaluated_params_or_the_previous_epoch():
    legacy = [_record(1, .5, .5, 100), _record(2, .5, .5, 110), _record(3, .5, .5, 110),
              _record(4, .5, .5, 120), _record(5, .5, .5, 110)]
    counts = analyze_ptp.evaluated_parameter_counts(legacy)
    assert counts == [100, 100, 110, 110, 120]
    assert analyze_ptp.infer_dendrite_step(counts) == 10
    assert analyze_ptp.dendrite_counts(counts, 10) == [0, 0, 1, 1, 2]
    logged = [_record(1, .5, .5, 110, evaluated=100)]
    assert analyze_ptp.evaluated_parameter_counts(logged) == [100]
    with pytest.raises(ValueError, match="multiple"):
        analyze_ptp.dendrite_counts([100, 105], 10)


def test_budget_selection_is_best_val_among_epochs_with_at_most_n_dendrites():
    records = [_record(1, .80, .70, 0), _record(2, .82, .72, 0), _record(3, .81, .75, 0),
               _record(4, .85, .74, 0), _record(5, .85, .80, 0), _record(6, .84, .81, 0)]
    dendrites = [0, 0, 1, 1, 2, 3]
    params = [100, 100, 110, 110, 120, 130]
    budgets = analyze_ptp.select_budgets(records, dendrites, params)
    assert budgets[0].epoch == 2 and budgets[0].test_acc == .72
    assert budgets[1].epoch == 4 and budgets[1].params == 110
    assert budgets[2].epoch == 4  # tie at .85 keeps the earlier epoch
    assert budgets[3].epoch == 4


def test_log_curve_interpolates_in_log10_and_extends_flat():
    curve = analyze_ptp.LogCurve.from_points([(100, 0.6), (1000, 0.8), (1000, 0.9)])
    assert curve(100) == pytest.approx(0.6)
    assert curve(1000) == pytest.approx(0.85)  # duplicate params are averaged
    assert curve(math.sqrt(100 * 1000)) == pytest.approx(0.725)
    assert curve(10) == pytest.approx(0.6)
    assert curve(10_000) == pytest.approx(0.85)


def _write_run(root: Path, seed: int, candidates):
    report = {"seed": seed, "candidates": []}
    for width, rate, pruned, epochs in candidates:
        name = f"sparknet_c{width}_multilayer"
        report["candidates"].append({
            "width": width,
            "group_prune": {"prune_rate": rate, "pruned_params": pruned},
            "baseline": {"deployed_params": pruned},
            "dendritic": {"checkpoint": f"pai/candidates/{name}/final_clean_pai.pt"},
        })
        log = root / "metrics" / "sparsity" / name / "pai.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(json.dumps(r) for r in epochs(pruned)) + "\n")
    path = root / "reports" / "sparknet_dendritic_prune_experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report))


def test_analysis_decomposition_and_statistics_on_synthetic_runs(tmp_path):
    # PAI's counter starts at a wrapped count (offset 7); the analysis must
    # re-anchor on the true pruned count.  Step 40 per dendrite.
    def run_epochs(a_pre, a_final, dendrites_final):
        def build(pruned):
            base = pruned + 7
            rows = [_record(1, .80, a_pre - .01, base, evaluated=base, train=.95),
                    _record(2, .81, a_pre, base, evaluated=base, train=.96)]
            for d in range(1, dendrites_final + 1):
                rows.append(_record(2 + d, .81 + .01 * d, a_final if d == dendrites_final else a_pre,
                                    base + 40 * d, evaluated=base + 40 * d, train=.97))
            return rows
        return build

    # Two rates x two seeds; rate 0.5 at 1000 params, rate 0.7 at 500.
    _write_run(tmp_path / "s0", 0, [(8, 0.5, 1000, run_epochs(.80, .82, 1)),
                                    (4, 0.7, 500, run_epochs(.70, .74, 2))])
    _write_run(tmp_path / "s1", 1, [(8, 0.5, 1000, run_epochs(.82, .83, 1)),
                                    (4, 0.7, 500, run_epochs(.72, .75, 2))])
    runs = analyze_ptp.load_runs([tmp_path / "s0", tmp_path / "s1"])
    assert {r.step for r in runs} == {40}
    result = analyze_ptp.analyze(runs, budget=3)
    z = analyze_ptp.LogCurve.from_points([(1000, .81), (500, .71)])
    assert [(p["params"], p["test_acc"]) for p in result["zero_dendrite_reference"]] == [
        (500, pytest.approx(.71)), (1000, pytest.approx(.81))]

    rows = {(r["seed"], r["prune_rate"]): r for r in result["runs"]}
    dec = rows[(0, 0.7)]["decomposition"]
    assert dec["n_start"] == 500 and dec["n_final"] == 580 and dec["dendrites"] == 2
    assert dec["raw_gain"] == pytest.approx(.04)
    assert dec["param_cost"] == pytest.approx(z(580) - z(500))
    assert dec["gain"] == pytest.approx(.74 - z(580))
    assert dec["gain"] == pytest.approx(dec["pre_gain"] + dec["raw_gain"] - dec["param_cost"])
    # rate 0.5 at n_final 1040 is beyond the largest point: Z is flat there.
    assert rows[(0, 0.5)]["decomposition"]["param_cost"] == pytest.approx(0.0)
    assert rows[(0, 0.5)]["train_acc_budget0"] == pytest.approx(.96)

    by_rate = {r["prune_rate"]: r for r in result["per_rate"]}
    gains = [.74 - z(580), .75 - z(580)]
    from scipy import stats
    expected = stats.ttest_1samp(gains, 0.0)
    ci = stats.t.interval(0.95, 1, loc=sum(gains) / 2, scale=stats.sem(gains))
    summary = by_rate[0.7]["gain"]
    assert summary["mean"] == pytest.approx(sum(gains) / 2)
    assert summary["p"] == pytest.approx(expected.pvalue)
    assert (summary["ci_low"], summary["ci_high"]) == (pytest.approx(ci[0]), pytest.approx(ci[1]))
    # The seed-mean pre-dendrite term vanishes against Z by construction.
    pre = [rows[(s, 0.7)]["decomposition"]["pre_gain"] for s in (0, 1)]
    assert sum(pre) == pytest.approx(0.0)
    assert by_rate[0.7]["params_added_fraction"] == pytest.approx(80 / 500)

    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps([{"params": 500, "test_acc": .70},
                                   {"params": 2000, "test_acc": .90}]))
    with_scratch = analyze_ptp.analyze(
        runs, budget=3, scratch=analyze_ptp.load_scratch_frontier(scratch)
    )
    curve = analyze_ptp.LogCurve.from_points([(500, .70), (2000, .90)])
    row = next(r for r in with_scratch["runs"] if r["seed"] == 1 and r["prune_rate"] == 0.7)
    assert row["gain_vs_scratch"] == pytest.approx(.75 - curve(580))
    table = analyze_ptp.format_table(with_scratch)
    assert "vs scratch frontier" in table and "pooled: n=4" in table


def test_taylor_criterion_prunes_with_calibration_batches(tmp_path, ptp_config, monkeypatch):
    from kws.optimize import sparknet_dendritic_prune_experiment as experiment

    ptp_config["pruning"] = {"method": "group", "criterion": "taylor",
                             "prune_rates": [0.5], "taylor_batches": 2}
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.randn(4, 1, 32, 5, generator=generator), torch.randint(0, 12, (4,), generator=generator))
               for _ in range(2)]
    monkeypatch.setattr(experiment, "_taylor_batches", lambda *_a: (lambda: iter(batches)))
    seen = []

    def fake_cycle(checkpoint_path, data_cfg, model_cfg, train_cfg, save_name, **kwargs):
        seen.append(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        return {"best_val_acc": 0.9, "deployed_params": 2356, "cost": {"macs": 1},
                "prune_finetune": {"best_val_acc": 0.88, "checkpoint": "x.pt"},
                "resume": {"status": "skipped"}}

    report = run_experiment(ptp_config, output_dir=tmp_path / "run", cycle_runner=fake_cycle)
    assert report["candidates"][0]["group_prune"]["criterion"] == "taylor"
    assert seen[0]["group_prune"]["criterion"] == "taylor"
    assert seen[0]["model_cfg"]["channels"] == 8
