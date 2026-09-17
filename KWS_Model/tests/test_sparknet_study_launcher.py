import importlib
import importlib.util
import subprocess
from pathlib import Path


def test_study_matrix_uses_scratch_sources_and_prioritizes_pointwise():
    spec = importlib.util.find_spec("scripts.run_sparknet_dendritic_study")
    assert spec is not None
    launcher = importlib.import_module("scripts.run_sparknet_dendritic_study")

    runs = launcher.build_arm_runs(Path("outputs/new-study"))

    assert len(runs) == 75
    assert {run.width for run in runs} == {8, 10, 12}
    assert {run.seed for run in runs} == {0, 1, 2, 3, 4}
    assert {run.arm for run in runs} == {
        "pointwise",
        "fc",
        "gate_conv",
        "depthwise",
        "control",
    }
    assert [run.arm for run in runs[:15]] == ["pointwise"] * 15
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
