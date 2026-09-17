import importlib
import importlib.util
import subprocess
from pathlib import Path


def test_study_matrix_uses_scratch_sources_and_prioritizes_pointwise():
    spec = importlib.util.find_spec("scripts.run_sparknet_dendritic_study")
    assert spec is not None
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")

    runs = launcher.build_arm_runs(Path("outputs/new-study"))

    # Derived from the launcher's own constants: a width added to the study
    # should not need this test edited to keep passing, but the matrix must
    # stay a complete arm x width x seed product with no cell dropped.
    expected = len(launcher.ARM_ORDER) * len(launcher.WIDTHS) * len(launcher.SEEDS)
    assert len(runs) == expected
    assert len({(r.arm, r.width, r.seed) for r in runs}) == expected
    assert {run.width for run in runs} == set(launcher.WIDTHS)
    assert {run.seed for run in runs} == set(launcher.SEEDS)
    # The narrow end is the point of the study; keep it in the matrix.
    assert {2, 4, 6, 8, 10, 12} <= set(launcher.WIDTHS)
    assert {run.arm for run in runs} == {
        "pointwise",
        "fc",
        "gate_conv",
        "depthwise",
        "control",
    }
    lead = len(launcher.WIDTHS) * len(launcher.SEEDS)
    assert [run.arm for run in runs[:lead]] == ["pointwise"] * lead
    for run in runs:
        assert run.source_checkpoint == Path(
            f"outputs/new-study/scratch/c{run.width}-seed{run.seed}/"
            "models/checkpoints/paper_replication/best.pt"
        )
        assert run.config["source_channels"] == run.width
        assert run.config["widths"] == [run.width]
        assert run.config["pruning"] == {"method": "identity"}
        assert run.config["seed"] == run.seed
        assert run.config["objective"] == {
            "metric": "validation_accuracy",
            "use_test": False,
        }


def test_frozen_selection_is_not_recomputed(tmp_path):
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")
    selection = tmp_path / "selection/selected_arms.json"
    selection.parent.mkdir(parents=True)
    selection.write_text('{"selection_split": "validation"}\n')

    returned = launcher.freeze_selection(tmp_path, dry_run=False)

    assert returned == selection
    assert selection.read_text() == '{"selection_split": "validation"}\n'


def test_launcher_configs_and_selector_registry_name_the_same_arms():
    """Two files spell out each arm's placement; drift would mislabel results."""
    import yaml

    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")
    selector = importlib.import_module("scripts.select_sparknet_arms")

    assert set(launcher.ARM_TRAIN_CONFIG) == set(selector.ARM_MODULE_IDS)
    for arm, config_path in launcher.ARM_TRAIN_CONFIG.items():
        train = yaml.safe_load((launcher.ROOT / config_path).read_text())
        module_ids = frozenset(train["perforatedai"]["module_ids"])
        assert module_ids == selector.ARM_MODULE_IDS[arm], arm
        # The control is the budget match, not a shorter run; equality of the
        # epoch budget is the invariant that makes the comparison mean anything.
        reference = yaml.safe_load(
            (launcher.ROOT / launcher.ARM_TRAIN_CONFIG["fc"]).read_text()
        )
        for key in ("epochs", "dendritic_schedule_epochs", "resume_epochs"):
            assert train[key] == reference[key], (arm, key)


def test_a_failed_arm_does_not_abort_the_rest_of_the_matrix(tmp_path, monkeypatch):
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")

    runs = launcher.build_arm_runs(tmp_path)[:3]
    monkeypatch.setattr(launcher, "build_arm_runs", lambda root: runs)
    for run in runs:
        run.source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        run.source_checkpoint.write_bytes(b"checkpoint")

    attempted: list[str] = []
    completed: set[str] = set()

    def fake_run(command, *, env=None):
        config = command[command.index("--config") + 1]
        attempted.append(config)
        if len(attempted) == 1:
            raise subprocess.CalledProcessError(1, command)
        completed.add(str(Path(config).parent))

    monkeypatch.setattr(launcher, "_run", fake_run)
    monkeypatch.setattr(
        launcher, "_report_complete", lambda run: str(run.output_dir) in completed
    )

    failures = launcher.run_arms(tmp_path, dry_run=False)

    # The first run dies; the two behind it still run, and only the dead one is
    # reported, so a re-invocation retries exactly that one.
    assert len(attempted) == 3
    assert failures == [f"{runs[0].arm} C{runs[0].width} seed{runs[0].seed}"]


def _mark_scratch_complete(scratch_root, width, seed):
    run = scratch_root / f"c{width}-seed{seed}"
    (run / "models/checkpoints/paper_replication").mkdir(parents=True, exist_ok=True)
    (run / "models/checkpoints/paper_replication/best.pt").write_bytes(b"ck")
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    (run / "metrics/summaries.yaml").write_text("best_val_acc: 0.9\n")


def test_one_failed_baseline_does_not_discard_the_whole_sweep(tmp_path, monkeypatch):
    """The sweep exits non-zero after any bad run; that must not kill the study."""
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")
    scratch_root = tmp_path / "scratch"
    calls = []

    def fake_run(command, *, env=None):
        calls.append(command)
        # Every baseline trains except C10 seed4, which keeps failing.
        for width in launcher.WIDTHS:
            for seed in launcher.SEEDS:
                if (width, seed) != (10, 4):
                    _mark_scratch_complete(scratch_root, width, seed)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(launcher, "_run", fake_run)
    launcher.run_baselines(tmp_path, dry_run=False)

    # Retried once, then returned rather than raising, so the arms whose
    # baselines do exist still get to run.
    assert len(calls) == 2
    assert launcher._missing_baselines(scratch_root) == [(10, 4)]


def test_an_interrupted_baseline_is_not_mistaken_for_a_finished_one(tmp_path):
    """A lone best.pt is what an interrupted 77/200-epoch run leaves behind."""
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")
    scratch_root = tmp_path / "scratch"
    for width in launcher.WIDTHS:
        for seed in launcher.SEEDS:
            _mark_scratch_complete(scratch_root, width, seed)
    assert launcher._missing_baselines(scratch_root) == []

    partial = scratch_root / "c8-seed2"
    (partial / "metrics/summaries.yaml").unlink()
    assert (partial / "models/checkpoints/paper_replication/best.pt").exists()
    assert launcher._missing_baselines(scratch_root) == [(8, 2)]
