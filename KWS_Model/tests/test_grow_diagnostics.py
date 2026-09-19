"""Tests for kws.optimize.grow_diagnostics and kws.optimize.grow_clean_rebuild."""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from kws.models.registry import build_model
from kws.optimize.grow_clean_rebuild import (
    PAI_BOOKKEEPING,
    RebuiltDendriteModule,
    dendrite_prefixes,
    load_clean_state,
    rebuild_clean_model,
)
from kws.optimize.grow_diagnostics import _as_rows, dendrite_diagnostics, dendrite_modules

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG_PATH = REPO_ROOT / "configs/model/sparknet_c8_paper.yaml"
INPUT_SHAPE = (32, 101)
NUM_CLASSES = 12
CPU = torch.device("cpu")


def identity(z: torch.Tensor) -> torch.Tensor:
    return z


# --------------------------------------------------------------------------
# Synthetic clean modules for the diagnostics
# --------------------------------------------------------------------------


class FakeClean(nn.Module):
    """A one-dendrite clean PAI module: ``main(x) + skip * f(dendrite(x))``.

    ``pai_style`` calls the original module through ``.forward()`` the way
    PAI's compiled clean wrapper does, which bypasses hooks on it.
    """

    def __init__(self, main, dendrite, skip, view, f=torch.tanh, pai_style=False):
        super().__init__()
        self.layer_array = nn.ModuleList([dendrite, main])
        self.processor_array = [None, None]
        self.skip_weights = nn.ParameterList([nn.Parameter(skip.clone())])
        self.register_buffer("view_tuple", torch.tensor(view))
        self.f = f
        self.pai_style = pai_style

    def forward(self, x):
        z = self.layer_array[0](x)
        main = self.layer_array[-1].forward(x) if self.pai_style else self.layer_array[-1](x)
        return main + self.skip_weights[0][0].view(self.view_tuple.tolist()) * self.f(z)


def linear_model(f=torch.tanh, skip=None, same=False, pai_style=False, seed=0):
    torch.manual_seed(seed)
    main = nn.Linear(6, 4)
    dendrite = copy.deepcopy(main) if same else nn.Linear(6, 4)
    skip = torch.randn(1, 4) if skip is None else skip
    return nn.Sequential(FakeClean(main, dendrite, skip, [1, -1], f, pai_style))


def conv_model(f=torch.tanh, skip=None, same=False, seed=0):
    torch.manual_seed(seed)
    main = nn.Conv2d(3, 4, kernel_size=(1, 3), padding=(0, 1))
    dendrite = copy.deepcopy(main) if same else nn.Conv2d(3, 4, kernel_size=(1, 3), padding=(0, 1))
    skip = torch.randn(1, 4) if skip is None else skip
    return nn.Sequential(
        FakeClean(main, dendrite, skip, [1, -1, 1, 1], f), nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )


def batches_for(kind, n=96, batch=32, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(n, 6) if kind == "linear" else torch.randn(n, 3, 1, 9)
    y = torch.randint(0, 4, (n,))
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


MODELS = {"linear": linear_model, "conv": conv_model}


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_on_off_accuracy_match_manual_evaluation(kind):
    model = MODELS[kind]()
    clean = model[0]
    with torch.no_grad():
        # Labels the dendrite-on model gets right, so switching it off must cost.
        batches = [(x, model(x).argmax(1)) for x, _ in batches_for(kind)]
        on = sum(int((model(x).argmax(1) == y).sum()) for x, y in batches) / 96
        saved = clean.skip_weights[0].detach().clone()
        clean.skip_weights[0].zero_()
        off = sum(int((model(x).argmax(1) == y).sum()) for x, y in batches) / 96
        clean.skip_weights[0].copy_(saved)
    result = dendrite_diagnostics(model, batches, CPU)
    assert result["val_acc_dendrite_on"] == on
    assert result["val_acc_dendrite_off"] == off
    assert result["n_samples"] == 96
    assert on == 1.0 and off < 1.0


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_skip_weights_restored_and_model_left_in_eval(kind):
    model = MODELS[kind]()
    model.train()
    before = model[0].skip_weights[0].detach().clone()
    dendrite_diagnostics(model, batches_for(kind), CPU)
    assert torch.equal(model[0].skip_weights[0], before)
    assert not model.training


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_identity_forward_function_is_perfectly_linear(kind):
    model = MODELS[kind](f=identity)
    stats = dendrite_diagnostics(model, batches_for(kind), CPU)["modules"]["0"]
    assert stats["linear_r2_vs_preactivation"] == pytest.approx(1.0, abs=1e-9)
    assert stats["n_dendrites"] == 1


@pytest.mark.parametrize("kind", ["linear", "conv"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_correlation_sign_follows_skip_sign_for_a_copy_of_the_base(kind, sign):
    # Dendrite == main and f == identity: the contribution is skip * main(x).
    model = MODELS[kind](f=identity, skip=torch.full((1, 4), 0.7 * sign), same=True)
    stats = dendrite_diagnostics(model, batches_for(kind), CPU)["modules"]["0"]
    assert stats["corr_with_base_output"] == pytest.approx(sign, abs=1e-9)
    assert stats["dendrite_to_base_std_ratio"] == pytest.approx(0.7, rel=1e-6)  # float32 0.7
    assert stats["skip_weight_mean_abs"] == pytest.approx(0.7, rel=1e-6)


def test_saturation_fractions_on_known_preactivations():
    main = nn.Linear(3, 3)
    dendrite = nn.Linear(3, 3)
    with torch.no_grad():
        dendrite.weight.copy_(torch.eye(3))
        dendrite.bias.zero_()
    model = nn.Sequential(FakeClean(main, dendrite, torch.ones(1, 3), [1, -1]))
    # Pre-activations are the inputs: 4 of 12 saturated (|z| > 2), 4 of 12 linear (|z| < 0.5).
    x = torch.tensor([
        [-3.0, 2.5, 0.1],
        [-0.4, 0.0, 1.0],
        [2.0, -2.1, 0.49],
        [0.5, 1.5, 9.0],
    ])
    y = torch.zeros(4, dtype=torch.long)
    stats = dendrite_diagnostics(model, [(x, y)], CPU)["modules"]["0"]
    assert stats["tanh_saturated_fraction"] == pytest.approx(4 / 12)
    assert stats["tanh_linear_fraction"] == pytest.approx(4 / 12)


def test_pai_style_forward_call_still_gets_every_metric():
    # PAI's clean wrapper runs the original module with ``.forward()``, so a
    # hook on it never fires.  The metrics must not silently disappear.
    plain = dendrite_diagnostics(linear_model(), batches_for("linear"), CPU)["modules"]["0"]
    pai = dendrite_diagnostics(linear_model(pai_style=True), batches_for("linear"), CPU)
    pai_stats = pai["modules"]["0"]
    for key in (
        "linear_r2_vs_preactivation", "corr_with_base_output",
        "dendrite_to_base_std_ratio", "tanh_saturated_fraction", "tanh_linear_fraction",
    ):
        assert pai_stats[key] == pytest.approx(plain[key], rel=1e-12), key


def test_multi_dendrite_module_reports_a_note_instead_of_linearity():
    torch.manual_seed(3)
    layers = [nn.Linear(6, 4) for _ in range(3)]
    module = RebuiltDendriteModule(layers, 4, [1, -1])
    with torch.no_grad():
        for weight in module.skip_weights:
            weight.normal_()
    stats = dendrite_diagnostics(nn.Sequential(module), batches_for("linear"), CPU)["modules"]["0"]
    assert stats["n_dendrites"] == 2
    assert "note" in stats
    assert "linear_r2_vs_preactivation" not in stats
    assert "tanh_saturated_fraction" not in stats
    assert -1.0 <= stats["corr_with_base_output"] <= 1.0
    assert stats["dendrite_to_base_std_ratio"] > 0


def test_modules_without_a_dendrite_are_not_diagnosed():
    empty = RebuiltDendriteModule([nn.Linear(6, 4)], 4, [1, -1])
    assert dendrite_modules(nn.Sequential(empty)) == {}
    with pytest.raises(ValueError, match="no clean PAI dendrite modules"):
        dendrite_diagnostics(nn.Sequential(empty), batches_for("linear"), CPU)
    mixed = nn.Sequential(linear_model()[0], nn.ReLU(), RebuiltDendriteModule([nn.Linear(4, 4)], 4, [1, -1]))
    assert list(dendrite_modules(mixed)) == ["0"]


def test_rows_are_cpu_float64():
    rows = _as_rows(torch.randn(2, 3, 1, 5), 1)
    assert rows.shape == (10, 3) and rows.dtype == torch.float64 and rows.device.type == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_matches_cpu():
    # MPS has no float64, and a fused .to(cpu, float64) of an MPS tensor
    # returns zeros; the statistics must still match the CPU ones.
    cpu = dendrite_diagnostics(conv_model(), batches_for("conv"), CPU)
    mps_model = conv_model().to("mps")
    mps = dendrite_diagnostics(mps_model, batches_for("conv"), torch.device("mps"))
    for key, value in cpu["modules"]["0"].items():
        assert mps["modules"]["0"][key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


# --------------------------------------------------------------------------
# Rebuild from a clean state
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_cfg():
    return yaml.safe_load(MODEL_CFG_PATH.read_text())


def plain_model(model_cfg, seed=0):
    torch.manual_seed(seed)
    model = build_model(model_cfg, INPUT_SHAPE, NUM_CLASSES)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith("running_var"):
                tensor.uniform_(0.5, 1.5)
            elif name.endswith("running_mean"):
                tensor.normal_(0.0, 0.1)
    return model.eval()


def synthetic_clean_state(model_cfg, prefix, n_dendrites=1, seed=0, view=None):
    """A clean state with ``prefix`` carrying ``n_dendrites`` random dendrites."""
    base = plain_model(model_cfg, seed)
    torch.manual_seed(seed + 100)
    module = base.get_submodule(prefix)
    channels = getattr(module, "out_channels", None) or module.out_features
    state = {k: v.clone() for k, v in base.state_dict().items() if not k.startswith(prefix + ".")}
    for index in range(n_dendrites + 1):
        for name, tensor in module.state_dict().items():
            state[f"{prefix}.layer_array.{index}.{name}"] = torch.randn_like(tensor) * 0.3
    for k in range(n_dendrites):
        state[f"{prefix}.skip_weights.{k}"] = torch.randn(k + 1, channels)
    if view is None:
        view = [1, -1] if isinstance(module, nn.Linear) else [1, -1, 1, 1]
    state[f"{prefix}.view_tuple"] = torch.tensor(view)
    state[f"{prefix}.node_index"] = torch.tensor(1)
    state[f"{prefix}.num_cycles"] = torch.tensor([2.0 * n_dendrites])
    state["tracker_string"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    return state


def manual_model(model_cfg, state, prefix, n_dendrites, f=torch.tanh):
    """Plain model whose ``prefix`` output is replaced by PAI's clean formula, by hand."""
    model = build_model(model_cfg, INPUT_SHAPE, NUM_CLASSES)
    marker = f"{prefix}.layer_array."
    main = f"{marker}{n_dendrites}."
    plain = {
        k: v for k, v in state.items()
        if not k.startswith(prefix + ".") and k != "tracker_string"
    }
    plain.update({f"{prefix}.{k[len(main):]}": v for k, v in state.items() if k.startswith(main)})
    model.load_state_dict(plain)
    module = model.get_submodule(prefix)

    def branch(index, x):
        weight = state[f"{marker}{index}.weight"]
        bias = state.get(f"{marker}{index}.bias")
        if isinstance(module, nn.Linear):
            return F.linear(x, weight, bias)
        return F.conv2d(x, weight, bias, module.stride, module.padding, module.dilation, module.groups)

    def hook(_module, inputs, _output):
        x = inputs[0]
        shape = [1, -1] if isinstance(module, nn.Linear) else [1, -1, 1, 1]
        outs = []
        for index in range(n_dendrites):
            value = branch(index, x)
            for j in range(index):
                value = value + state[f"{prefix}.skip_weights.{index - 1}"][j].view(shape) * outs[j]
            outs.append(f(value))
        value = branch(n_dendrites, x)
        for j in range(n_dendrites):
            value = value + state[f"{prefix}.skip_weights.{n_dendrites - 1}"][j].view(shape) * outs[j]
        return value

    module.register_forward_hook(hook)
    return model.eval()


def random_input(seed=5, batch=4):
    torch.manual_seed(seed)
    return torch.randn(batch, 1, *INPUT_SHAPE)


@pytest.mark.parametrize(
    "prefix,n_dendrites",
    [("fc", 1), ("blocks.3.pointwise", 1), ("gate_conv", 1), ("blocks.2.depthwise", 1), ("fc", 2)],
)
def test_rebuild_matches_manual_clean_forward(model_cfg, prefix, n_dendrites):
    state = synthetic_clean_state(model_cfg, prefix, n_dendrites)
    rebuilt = rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state)
    reference = manual_model(model_cfg, state, prefix, n_dendrites)
    x = random_input()
    with torch.no_grad():
        got, want = rebuilt(x), reference(x)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert not rebuilt.training
    assert all(p.device.type == "cpu" for p in rebuilt.parameters())
    module = rebuilt.get_submodule(prefix)
    assert isinstance(module, RebuiltDendriteModule)
    assert len(module.layer_array) == n_dendrites + 1
    assert module.processor_array == [None] * (n_dendrites + 1)
    assert module.view_tuple.tolist() == state[f"{prefix}.view_tuple"].tolist()
    for k, weight in enumerate(module.skip_weights):
        assert torch.equal(weight, state[f"{prefix}.skip_weights.{k}"])


def test_rebuild_of_two_placements_at_once(model_cfg):
    state = synthetic_clean_state(model_cfg, "blocks.3.pointwise")
    other = synthetic_clean_state(model_cfg, "blocks.2.pointwise", seed=1)
    state = {k: v for k, v in state.items() if not k.startswith("blocks.2.pointwise.")}
    state.update({k: v for k, v in other.items() if k.startswith("blocks.2.pointwise.")})
    assert dendrite_prefixes(state) == ["blocks.2.pointwise", "blocks.3.pointwise"]
    rebuilt = rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state)
    assert sorted(dendrite_modules(rebuilt)) == ["blocks.2.pointwise", "blocks.3.pointwise"]


def test_rebuilt_model_diagnostics_are_linear_with_identity(model_cfg):
    state = synthetic_clean_state(model_cfg, "blocks.3.pointwise")
    rebuilt = rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state, forward_function=identity)
    batches = [(random_input(seed), torch.zeros(4, dtype=torch.long)) for seed in range(3)]
    before = rebuilt.get_submodule("blocks.3.pointwise").skip_weights[0].detach().clone()
    stats = dendrite_diagnostics(rebuilt, batches, CPU)["modules"]["blocks.3.pointwise"]
    assert stats["linear_r2_vs_preactivation"] == pytest.approx(1.0, abs=1e-9)
    assert torch.equal(rebuilt.get_submodule("blocks.3.pointwise").skip_weights[0], before)


def test_bookkeeping_is_the_documented_set(model_cfg):
    assert set(PAI_BOOKKEEPING) == {"node_index", "num_cycles", "tracker_string", "module_id"}
    state = synthetic_clean_state(model_cfg, "fc")
    state["fc.module_id"] = torch.tensor(7)
    state["blocks.0.module_id.extra"] = torch.tensor(7)
    rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state)  # accepted


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s: s.__setitem__("fc.extra", torch.zeros(1)), "unexpected keys: fc.extra"),
        (lambda s: s.__setitem__("blocks.0.num_cycles", torch.zeros(1)), "unexpected keys: blocks.0.num_cycles"),
        (lambda s: s.pop("blocks.0.bn.running_mean"), "missing keys: blocks.0.bn.running_mean"),
        (lambda s: s.pop("fc.skip_weights.0"), "missing keys: fc.skip_weights.0.*drops a one-dendrite"),
        (lambda s: s.__setitem__("fc.skip_weights.0", torch.zeros(1, 11)), "shape mismatches: fc.skip_weights.0"),
        (lambda s: s.__setitem__("fc.skip_weights.0", torch.zeros(2, 12)), "shape mismatches: fc.skip_weights.0"),
        (lambda s: s.__setitem__("fc.layer_array.0.weight", torch.zeros(12, 31)), "shape mismatches: fc.layer_array.0.weight"),
        (lambda s: s.__setitem__("fc.skip_weights.1", torch.zeros(2, 12)), "unexpected keys: fc.skip_weights.1"),
        (lambda s: s.pop("fc.view_tuple"), "missing key 'fc.view_tuple'"),
        (lambda s: s.__setitem__("fc.view_tuple", torch.tensor([1, 1])), "expected one -1"),
        (lambda s: s.__setitem__("fc.node_index", torch.tensor(0)), "disagrees with the channel axis"),
        (lambda s: s.__setitem__("nope.layer_array.0.weight", torch.zeros(1)), "the model lacks"),
        (lambda s: s.__setitem__("fc.layer_array.3.weight", torch.zeros(12, 32)), "has gaps"),
    ],
)
def test_rebuild_is_strict(model_cfg, mutate, match):
    state = synthetic_clean_state(model_cfg, "fc")
    mutate(state)
    with pytest.raises(ValueError, match=match):
        rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state)


def test_load_clean_state_round_trips_safetensors(tmp_path, model_cfg):
    from safetensors.torch import save_file

    state = synthetic_clean_state(model_cfg, "fc")
    path = tmp_path / "final_clean_pai.pt"
    save_file(state, str(path), metadata={"format": "perforatedai_clean_inference"})
    loaded, metadata = load_clean_state(path)
    assert metadata == {"format": "perforatedai_clean_inference"}
    assert set(loaded) == set(state)
    rebuilt = rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, loaded)
    with torch.no_grad():
        torch.testing.assert_close(
            rebuilt(random_input()), manual_model(model_cfg, state, "fc", 1)(random_input())
        )


def test_rebuild_does_not_import_perforatedai(tmp_path, model_cfg):
    from safetensors.torch import save_file

    path = tmp_path / "clean.pt"
    save_file(synthetic_clean_state(model_cfg, "blocks.3.pointwise"), str(path))
    script = textwrap.dedent(
        f"""
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("perforatedai", "perforatedbp"):
                    raise ImportError("blocked: " + name)
                return None

        sys.meta_path.insert(0, Block())
        import torch, yaml
        from kws.optimize.grow_clean_rebuild import load_clean_state, rebuild_clean_model
        from kws.optimize.grow_diagnostics import dendrite_diagnostics
        cfg = yaml.safe_load(open({str(MODEL_CFG_PATH)!r}))
        state, _ = load_clean_state({str(path)!r})
        model = rebuild_clean_model(cfg, (32, 101), 12, state)
        x = torch.randn(2, 1, 32, 101)
        dendrite_diagnostics(model, [(x, torch.zeros(2, dtype=torch.long))], torch.device("cpu"))
        loaded = sorted(m for m in sys.modules if m.startswith(("perforatedai", "perforatedbp")))
        assert not loaded, loaded
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


REAL_BEST = (
    REPO_ROOT / "outputs/sparknet-grow-dendrites-v3/fc/c8-seed0/pai/candidates"
    / "sparknet_c8_paper_fc/best_dendritic/final_clean_pai.pt"
)


@pytest.mark.skipif(not REAL_BEST.is_file(), reason="needs the v3 fc c8 seed0 run")
def test_rebuilds_the_real_fc_export(model_cfg):
    state, metadata = load_clean_state(REAL_BEST)
    assert metadata.get("format") == "perforatedai_clean_inference"
    rebuilt = rebuild_clean_model(model_cfg, INPUT_SHAPE, NUM_CLASSES, state)
    reference = manual_model(model_cfg, state, "fc", 1)
    x = random_input(batch=8)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), reference(x), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# scripts/diagnose_grow_run.py refusals (no data needed)
# --------------------------------------------------------------------------


def _run_dir(tmp_path, status="complete"):
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    summary = {"status": status, "seed": 0, "artifacts": {}}
    (run / "reports/grow_summary.yaml").write_text(yaml.safe_dump(summary))
    return run


def test_diagnose_script_refuses_an_incomplete_run(tmp_path, capsys):
    from scripts import diagnose_grow_run

    run = _run_dir(tmp_path, status="incomplete")
    before = (run / "reports/grow_summary.yaml").read_bytes()
    assert diagnose_grow_run.main(["--run-dir", str(run)]) == diagnose_grow_run.EXIT_REFUSED
    assert "not 'complete'" in capsys.readouterr().err
    assert not (run / "reports/grow_diagnostics.yaml").exists()
    assert (run / "reports/grow_summary.yaml").read_bytes() == before


def test_diagnose_script_will_not_overwrite_without_force(tmp_path, capsys):
    from scripts import diagnose_grow_run

    run = _run_dir(tmp_path)
    existing = run / "reports/grow_diagnostics.yaml"
    existing.write_text("keep: me\n")
    assert diagnose_grow_run.main(["--run-dir", str(run)]) == diagnose_grow_run.EXIT_REFUSED
    assert "--force" in capsys.readouterr().err
    assert existing.read_text() == "keep: me\n"


def test_diagnose_script_refuses_a_missing_requested_export(tmp_path, capsys):
    from scripts import diagnose_grow_run

    run = _run_dir(tmp_path)
    assert diagnose_grow_run.main(["--run-dir", str(run), "--export", "best"]) == diagnose_grow_run.EXIT_REFUSED
    assert "artifacts.best_clean" in capsys.readouterr().err
