import torch
from torch import nn

from kws.hardware.neurosim.graph_capture import capture_graph


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.conv = nn.Conv2d(1, 2, 3, padding=1, groups=1)

    def forward(self, x):
        y = self.shared(x.mean(dim=2))
        y = self.shared(y)
        z = self.conv(x.unsqueeze(1))
        return y, z


def test_capture_records_runtime_shapes_and_shared_executions():
    captured = capture_graph(SharedModel().eval(), torch.ones(2, 4, 4))

    assert [layer.execution_name for layer in captured.ir.layers] == [
        "shared#0",
        "shared#1",
        "conv#0",
    ]
    assert captured.ir.layers[0].input_shape == (2, 4)
    assert captured.ir.layers[2].output_shape == (2, 2, 4, 4)
    assert captured.ir.layers[0].weight_key == captured.ir.layers[1].weight_key


def test_capture_supports_a_weighted_root_module():
    captured = capture_graph(nn.Linear(2, 3).eval(), torch.ones(1, 2))

    assert len(captured.ir.layers) == 1
    assert captured.ir.layers[0].op_type == "linear"


def test_capture_rejects_unsupported_weighted_module():
    class Unsupported(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 3)

        def forward(self, x):
            return self.conv(x)

    try:
        capture_graph(Unsupported().eval(), torch.ones(1, 2, 8))
    except ValueError as exc:
        assert "unsupported weighted" in str(exc)
    else:
        raise AssertionError("unsupported weighted module was accepted")


def test_capture_rejects_custom_learned_parameter_even_without_weight_name():
    class Unsupported(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Parameter(torch.ones(2, 2))

        def forward(self, x):
            return x @ self.matrix

    try:
        capture_graph(Unsupported().eval(), torch.ones(1, 2))
    except ValueError as exc:
        assert "unsupported weighted" in str(exc)
    else:
        raise AssertionError("custom learned parameter was accepted")


def test_capture_observes_clean_pai_direct_branch_forward_calls():
    class CleanWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_array = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

        def forward(self, x):
            return self.layer_array[0](x) + self.layer_array[1].forward(x)

    captured = capture_graph(CleanWrapper().eval(), torch.ones(1, 2))

    assert len(captured.ir.layers) == 2


def test_capture_counts_clean_pai_skip_edges_like_project_profiler():
    class CleanResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_array = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
            self.skip_weights = nn.ParameterList([nn.Parameter(torch.ones(1, 2))])

        def forward(self, x):
            return self.layer_array[0](x) + self.layer_array[1].forward(x)

    captured = capture_graph(CleanResidual().eval(), torch.ones(1, 2))

    assert captured.ir.total_macs == 10
