import math
from pathlib import Path

import pytest
import torch
import torchaudio
import yaml

from kws.data.features import FeatureExtractor
from kws.models.dtnet import (
    DTLayer,
    DTNet,
    IDCT,
    build_dtnet,
    leaky_integrate,
    rho_to_tau,
)
from kws.models.registry import build_model, build_model_from_checkpoint

SHAPE = (32, 101)
CLASSES = 12
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"

HET = {"family": "dtnet", "name": "het", "neurons": [6, 5], "branches": 3, "fan_in": [4, 3],
       "tau": "hetero", "branch_norm": True, "soma_tau": True, "input_transform": "idct"}
POINT = {"family": "dtnet", "name": "point", "neurons": [7, 6, 5], "branches": 1,
         "fan_in": "full", "branch_activation": "identity", "tau": "none",
         "branch_norm": False, "soma_tau": True}


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_params(c_in, n, k, f, tau, branch_norm, soma_tau):
    f = c_in if f == "full" else min(f, c_in)
    rho = {"hetero": n * k, "shared": n, "none": 0}[tau]
    return n * k * (f + 2) + rho + 2 * n * k * branch_norm + n * soma_tau + 2 * n


def expected_params(neurons, branches, fan_in, tau, branch_norm, soma_tau, c_in=32):
    total = 0
    branches = branches if isinstance(branches, list) else [branches] * len(neurons)
    fan_in = fan_in if isinstance(fan_in, list) else [fan_in] * len(neurons)
    for n, k, f in zip(neurons, branches, fan_in):
        total += layer_params(c_in, n, k, f, tau, branch_norm, soma_tau)
        c_in = n
    return total + c_in * CLASSES + CLASSES


def naive_leaky(u, rho):
    a = torch.sigmoid(rho)[None, :]
    v = torch.zeros_like(u[..., 0])
    out = []
    for t in range(u.shape[-1]):
        v = a * v + (1 - a) * u[..., t]
        out.append(v)
    return torch.stack(out, -1)


@pytest.mark.parametrize("cfg", [HET, POINT, {**HET, "tau": "shared", "layout": "random"},
                                 {**HET, "tau": "none", "branches": [2, 4], "dropout": 0.1}])
def test_output_and_sequence_shapes(cfg):
    model = build_model(cfg, SHAPE, CLASSES)
    assert isinstance(model, DTNet)
    x = torch.randn(3, 1, *SHAPE)
    assert model(x).shape == (3, CLASSES)
    model.eval()
    seq = model.forward_sequence(x)
    assert seq.shape == (3, cfg["neurons"][-1], SHAPE[1])
    assert model.fc.in_features == cfg["neurons"][-1]
    assert torch.allclose(model(x), model.fc(seq.mean(-1)), atol=1e-6)


@pytest.mark.parametrize("tau", ["hetero", "shared", "none"])
@pytest.mark.parametrize("branch_norm", [True, False])
@pytest.mark.parametrize("soma_tau", [True, False])
@pytest.mark.parametrize("fan_in", [[8, 8], [4, "full"], [40, 3]])
def test_param_count_matches_formula(tau, branch_norm, soma_tau, fan_in):
    cfg = {"neurons": [9, 7], "branches": [4, 2], "fan_in": fan_in, "tau": tau,
           "branch_norm": branch_norm, "soma_tau": soma_tau}
    model = build_dtnet(cfg, SHAPE, CLASSES)
    assert n_params(model) == expected_params([9, 7], [4, 2], fan_in, tau, branch_norm, soma_tau)


def test_point_control_is_a_leaky_dense_layer():
    model = build_dtnet(POINT, SHAPE, CLASSES)
    layer = model.layers[0]
    assert layer.index is None and layer.branch.weight.shape == (7, 32, 1)
    assert layer.branch_rho is None and layer.branch_bn is None
    assert torch.equal(layer.soma.weight, torch.ones_like(layer.soma.weight))
    assert n_params(model) == expected_params([7, 6, 5], 1, "full", "none", False, True)


def test_idct_inverts_torchaudio_mfcc_dct():
    torch.manual_seed(0)
    mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=32, log_mels=True,
                                      melkwargs={"n_mels": 32, "n_fft": 512})
    log_mel = torch.randn(2, 32, 101) * 5 - 3
    coeffs = torch.matmul(log_mel.transpose(-1, -2), mfcc.dct_mat).transpose(-1, -2)  # torchaudio
    idct = IDCT(32)
    assert torch.allclose(idct(coeffs), log_mel, atol=2e-5)
    assert torch.allclose(idct.inverse, mfcc.dct_mat.t(), atol=1e-5)  # the ortho DCT's transpose
    assert list(idct.parameters()) == [] and "inverse" not in idct.state_dict()


def test_idct_recovers_log_mels_from_project_features():
    torch.manual_seed(0)
    extractor = FeatureExtractor(16000, 32, 25, 10, feature_type="mfcc", log_mels=True)
    wave = torch.randn(1, 16000) * 0.1
    features = extractor(wave)
    assert features.shape == (1, *SHAPE)
    log_mel = torch.log(extractor.mfcc.MelSpectrogram(wave) + 1e-6)  # torchaudio's log_mels path
    assert torch.allclose(IDCT(32)(features), log_mel, atol=1e-4)
    model = build_dtnet({"input_transform": "idct"}, SHAPE, CLASSES)
    assert n_params(model) == n_params(build_dtnet({}, SHAPE, CLASSES))
    assert torch.equal(model.input_transform(features), IDCT(32)(features))


def test_leaky_integration_matches_python_recurrence_and_gradients():
    torch.manual_seed(0)
    u = torch.randn(3, 5, 13, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([-6.0, -0.5, 0.3, 2.0, 9.0], dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 5, 13, dtype=torch.float64)
    fast = leaky_integrate(u, rho)
    slow = naive_leaky(u, rho)
    assert torch.allclose(fast, slow, atol=1e-12)
    g_fast = torch.autograd.grad((fast * w).sum(), (u, rho))
    g_slow = torch.autograd.grad((slow * w).sum(), (u, rho))
    for a, b in zip(g_fast, g_slow):
        assert torch.allclose(a, b, atol=1e-12)
    assert torch.autograd.gradcheck(lambda r: leaky_integrate(u.detach(), r), (rho,))


@pytest.mark.parametrize("fan_in,layout,branches", [(8, "local", 4), (5, "random", 3),
                                                     ("full", "local", 1), ("full", "local", 2)])
def test_dense_forward_matches_index_select_grouped_conv(fan_in, layout, branches):
    torch.manual_seed(0)
    layer = DTLayer(12, 5, branches, fan_in, layout=layout).double().eval()
    x = torch.randn(3, 12, 17, dtype=torch.float64, requires_grad=True)

    def reference(x):
        u = layer.branch(x if layer.index is None else x.index_select(1, layer.index))
        u = torch.relu(layer.branch_bn(leaky_integrate(u, layer.expanded_branch_rho())))
        return torch.relu(layer.bn(leaky_integrate(layer.soma(u), layer.soma_rho)))

    w = torch.randn(3, 5, 17, dtype=torch.float64)
    fast, ref = layer(x), reference(x)
    assert torch.allclose(fast, ref, atol=1e-12)
    params = [x] + list(layer.parameters())
    for a, b in zip(torch.autograd.grad((fast * w).sum(), params),
                    torch.autograd.grad((ref * w).sum(), params)):
        assert torch.allclose(a, b, atol=1e-12)


def test_leaky_integration_is_finite_at_extreme_rho():
    u = torch.randn(2, 3, 20)
    rho = torch.tensor([-200.0, 0.0, 200.0], requires_grad=True)
    v = leaky_integrate(u, rho)
    v.sum().backward()
    assert torch.isfinite(v).all() and torch.isfinite(rho.grad).all()
    assert torch.allclose(v[:, 0], u[:, 0], atol=1e-6)  # a -> 0: no memory


def test_tau_init_is_log_spaced_and_deterministic():
    layer = DTLayer(32, 3, 4, 8, tau="hetero", tau_min=1.0, tau_max=50.0)
    taus = layer.branch_taus().view(3, 4)
    expected = torch.logspace(0, math.log10(50), 4)
    assert torch.allclose(taus, expected.expand(3, 4), rtol=1e-4)
    shared = DTLayer(32, 5, 4, 8, tau="shared")
    assert torch.allclose(shared.branch_taus().view(5, 4)[:, 0],
                          torch.logspace(0, math.log10(50), 5), rtol=1e-4)
    assert torch.equal(shared.branch_taus().view(5, 4), shared.branch_taus().view(5, 4)[:, :1].expand(5, 4))
    assert torch.allclose(rho_to_tau(layer.soma_rho), torch.logspace(0, math.log10(50), 3), rtol=1e-4)
    single = DTLayer(32, 4, 1, 8, tau="hetero")  # K = 1 spreads across neurons
    assert torch.allclose(single.branch_taus(), torch.logspace(0, math.log10(50), 4), rtol=1e-4)
    assert torch.equal(DTLayer(32, 3, 4, 8).branch_rho, layer.branch_rho)


@pytest.mark.parametrize("input_norm", ["none", "batchnorm"])
@pytest.mark.parametrize("cfg", [HET, POINT, {**HET, "tau": "shared", "layout": "random"}])
def test_every_layer_is_causal(cfg, input_norm):
    torch.manual_seed(0)
    model = build_model({**cfg, "input_norm": input_norm}, SHAPE, CLASSES).eval()
    outputs = []
    for layer in model.layers:
        layer.register_forward_hook(lambda m, i, o: outputs.append(o))
    x = torch.randn(2, 1, *SHAPE)
    t0 = 40
    y = x.clone()
    y[..., t0] += torch.randn_like(y[..., t0]) * 3
    with torch.no_grad():
        model(x)
        model(y)
    n = len(model.layers)
    for before, after in zip(outputs[:n], outputs[n:]):
        assert torch.equal(before[..., :t0], after[..., :t0])
        assert not torch.allclose(before[..., t0:], after[..., t0:])


def test_local_and_random_layouts_differ_and_are_deterministic():
    local = DTLayer(32, 6, 4, 8, layout="local")
    rnd = DTLayer(32, 6, 4, 8, layout="random")
    assert torch.equal(rnd.index, DTLayer(32, 6, 4, 8, layout="random").index)
    assert torch.equal(local.index, DTLayer(32, 6, 4, 8, layout="local").index)
    assert not torch.equal(local.index, rnd.index)
    starts = local.index.view(24, 8)[:, 0]
    assert starts.tolist() == [round(u / 23 * 24 + 1e-9) for u in range(24)]
    for branch in rnd.index.view(24, 8):
        assert len(set(branch.tolist())) == 8  # drawn without replacement
    assert "index" not in local.state_dict()


@pytest.mark.parametrize("in_ch,n,k,f", [(32, 34, 4, 8), (34, 34, 4, 8), (17, 16, 4, 8),
                                         (32, 6, 4, 8), (12, 1, 1, 5), (9, 3, 2, 9)])
def test_every_local_window_is_contiguous_and_in_range(in_ch, n, k, f):
    layer = DTLayer(in_ch, n, k, f, layout="local")
    f = min(f, in_ch)
    if layer.index is None:
        assert f == in_ch
        return
    windows = layer.index.view(n * k, f)
    starts = windows[:, 0]
    assert torch.equal(windows, starts[:, None] + torch.arange(f))  # contiguous, no wrap
    assert starts[0].item() == 0
    if n * k > 1:
        assert starts[-1].item() == in_ch - f  # windows spread over the whole input
    assert windows.min().item() >= 0 and windows.max().item() < in_ch
    assert (starts[1:] >= starts[:-1]).all()  # neuron-major, low to high


def test_checkpoint_rebuild_roundtrip():
    model = build_model(HET, SHAPE, CLASSES).eval()
    checkpoint = {"model_state_dict": model.state_dict(), "model_family": "dtnet",
                  "model_cfg": HET, "input_shape": model.input_shape,
                  "num_classes": model.fc.out_features}
    rebuilt = build_model_from_checkpoint(checkpoint).eval()
    rebuilt.load_state_dict(checkpoint["model_state_dict"])
    x = torch.randn(2, 1, *SHAPE)
    assert torch.equal(model(x), rebuilt(x))


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        build_dtnet({"tau": "fast"}, SHAPE, CLASSES)
    with pytest.raises(ValueError):
        build_dtnet({"layout": "grid"}, SHAPE, CLASSES)
    with pytest.raises(ValueError):
        build_dtnet({"neurons": [4, 4], "fan_in": [8, 8, 8]}, SHAPE, CLASSES)
    with pytest.raises(ValueError):
        build_dtnet({"input_transform": "dct"}, SHAPE, CLASSES)


DTNET_CONFIGS = sorted(CONFIGS.glob("dtnet_*.yaml"))


@pytest.mark.parametrize("path", DTNET_CONFIGS, ids=lambda p: p.stem)
def test_config_header_param_count(path):
    text = path.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["name"] == path.stem and cfg["family"] == "dtnet"
    assert text.startswith("# ") and "params (12 classes, input (32, 101))." in text.splitlines()[0]
    stated = int(text.split(" params", 1)[0].lstrip("# "))
    model = build_model(cfg, SHAPE, CLASSES)
    assert n_params(model) == stated
    assert model(torch.randn(2, 1, *SHAPE)).shape == (2, CLASSES)


@pytest.mark.parametrize("budget", ["a", "b"])
def test_budget_arms_are_param_matched(budget):
    counts = {p.stem: n_params(build_model(yaml.safe_load(p.read_text()), SHAPE, CLASSES))
              for p in DTNET_CONFIGS if p.stem.startswith(f"dtnet_{budget}_")}
    assert len(counts) >= 2
    assert max(counts.values()) <= 1.015 * min(counts.values()), counts
