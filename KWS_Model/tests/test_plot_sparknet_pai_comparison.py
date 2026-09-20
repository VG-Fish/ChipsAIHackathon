import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_sparknet_pai_comparison.py"
SPEC = importlib.util.spec_from_file_location("plot_sparknet_pai_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collect_selects_g16_real_cells_and_paired_scratch():
    report = {
        "cells": [
            {
                "arm": "pointwise_b2-bn", "sham": False, "model_name": "c4g16",
                "width": 4, "grow_mean": {"best": 0.84},
                "scratch_mean_paired": {"best": 0.83},
                "deployed": {"params": 1532, "macs": 98960},
            },
            {
                "arm": "pointwise_b2-bn", "sham": True, "model_name": "c6g16",
                "width": 6, "grow_mean": {"best": 0.88},
                "scratch_mean_paired": {"best": 0.87},
                "deployed": {"params": 1960, "macs": 138956},
            },
            {
                "arm": "fc", "sham": False, "model_name": "c8g16",
                "width": 8, "grow_mean": {"best": 0.91},
                "scratch_mean_paired": {"best": 0.90},
                "deployed": {"params": 2000, "macs": 1},
            },
        ]
    }
    assert MODULE.collect(report) == [(4, 0.83, 0.84, 1532)]


def test_collect_exports_reads_int_test_accuracy_and_flash_bytes(tmp_path):
    root = tmp_path / "rp2040"
    run = root / "c6g16-b2-seed0"
    (run / "reports").mkdir(parents=True)
    (run / "models/exported/rp2040/firmware").mkdir(parents=True)
    (run / "reports/rp2040.yaml").write_text(
        "float_parameters: 1624\ntest:\n  int:\n    accuracy: 85.787\n"
    )
    (run / "models/exported/rp2040/firmware/kws_rp2040.uf2").write_bytes(b"x" * 123)
    rows = MODULE.collect_exports(root, ["c6g16-b2-seed0"])
    assert rows == [("PAI C6", 85.787, 1624, 123)]
