"""Re-measure MACs for dendritic candidates profiled before the counter fix.

``count_macs`` used to miss the base branch of every PerforatedAI-wrapped
module, because PAI's clean export calls it as ``layer_array[-1].forward(...)``
and a direct ``.forward`` call is invisible to forward hooks. Reports written
by a run that started before ``kws.utils.profile`` was fixed therefore carry a
dendritic ``macs`` value that is one branch too cheap -- and a long run keeps
using the code it imported at launch, so restarting the process is the only
other way to get the corrected number.

This rebuilds each completed candidate's exported graph and re-profiles it. It
prints what would change and writes nothing unless ``--apply`` is given, and it
refuses to write to a run whose lock is still held, so it cannot race the
experiment's own report writes.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from kws.models.registry import checkpoint_input_shape
from kws.optimize.dendritic import (
    build_cycle_base,
    configure_perforatedai,
    ensure_clean_dendrite_skip_weights,
)
from kws.pipeline import import_pai_module
from kws.utils.artifacts import ArtifactLayout
from kws.utils.profile import (
    count_macs,
    deployed_parameter_count,
    measure_peak_activation_bytes,
)

REPORT_NAME = "sparknet_dendritic_prune_experiment.yaml"


def load_clean_graph(candidate_dir: Path, device: torch.device) -> tuple[Any, tuple[int, int]]:
    """Rebuild the exact clean inference graph a finished PAI cycle exported."""
    # PAI resolves the candidate path while this process is chdir'd elsewhere.
    candidate_dir = candidate_dir.resolve()
    UPA: Any = import_pai_module("perforatedai.utils_perforatedai")
    metadata = yaml.safe_load((candidate_dir / "cycle_metadata.yaml").read_text())
    base_model_cfg = metadata["base_model_cfg"]
    # Relative to the repo root, exactly as the experiment recorded it.
    source_path = metadata["source_checkpoint"]
    source_channels = torch.load(source_path, map_location="cpu", weights_only=False)[
        "model_cfg"
    ]["channels"]
    base, checkpoint, _ = build_cycle_base(
        source_path,
        float(base_model_cfg["channels"]) / float(source_channels),
        target_model_cfg=base_model_cfg,
    )
    configure_perforatedai(metadata["perforatedai"], device)

    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as scratch:
        # PAI resolves its own writes against the working directory, so keep
        # them out of the run being inspected.
        os.chdir(scratch)
        try:
            model = UPA.perforate_model(
                base,
                doing_pai=False,
                save_name="reload",
                making_graphs=False,
                maximizing_score=True,
            )
            model = UPA.load_pretrained_model(
                model, str(candidate_dir), "best_model", remove_dendrite_scaffolding=True
            ).to(device)
        finally:
            os.chdir(previous_cwd)

    clean_state = UPA.load_file(str(candidate_dir / "final_clean_pai.pt"))
    ensure_clean_dendrite_skip_weights(model, clean_state)
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected or set(missing) - {"tracker_string"}:
        raise RuntimeError(
            f"{candidate_dir} did not match its reconstructed graph: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model.eval(), tuple(checkpoint_input_shape(checkpoint))


def remeasure(run_root: Path, *, apply: bool) -> int:
    layout = ArtifactLayout(run_root)
    report_path = layout.report_path(REPORT_NAME)
    if not report_path.exists():
        print(f"{run_root}: no {REPORT_NAME} report, skipped")
        return 0
    report = yaml.safe_load(report_path.read_text())
    device = torch.device("cpu")

    changed = 0
    for record in report["candidates"]:
        if record.get("status") != "complete":
            continue
        candidate_dir = run_root / "pai" / "candidates" / f"sparknet_c{record['width']}_multilayer"
        if not (candidate_dir / "final_clean_pai.pt").exists():
            print(f"{run_root} C{record['width']}: no clean export, skipped")
            continue
        model, input_shape = load_clean_graph(candidate_dir, device)
        dendritic = record["dendritic"]
        # The parameter count was never affected by the hook blind spot, so it
        # is an independent check that this rebuilt graph really is the one the
        # report describes -- and therefore that its MACs may replace the old
        # ones. Profiling a graph that differs would write a plausible lie.
        params = deployed_parameter_count(model)
        if params != int(dendritic["deployed_params"]):
            raise RuntimeError(
                f"{candidate_dir}: rebuilt graph has {params} deployed parameters "
                f"but the report recorded {dendritic['deployed_params']}; "
                "refusing to trust its MACs"
            )
        macs = int(count_macs(model, input_shape))
        activation_bytes = int(measure_peak_activation_bytes(model, input_shape))

        baseline = record["baseline"]
        if int(dendritic["macs"]) == macs:
            print(f"{run_root} C{record['width']}: already {macs} MACs")
            continue
        recorded_bytes = (dendritic.get("full_cost") or {}).get("activation_peak_bytes")
        print(
            f"{run_root} C{record['width']}: macs {dendritic['macs']} -> {macs}"
            f"  (delta vs baseline {macs - int(baseline['macs'])}, "
            f"growth {(macs - int(baseline['macs'])) / int(baseline['macs']):.6f}), "
            f"activation peak {recorded_bytes} -> {activation_bytes} bytes"
        )
        changed += 1
        if not apply:
            continue
        dendritic["macs"] = macs
        if isinstance(dendritic.get("full_cost"), dict):
            dendritic["full_cost"]["macs"] = macs
            dendritic["full_cost"]["activation_peak_bytes"] = activation_bytes
        comparison = record.get("comparison")
        if isinstance(comparison, dict):
            comparison["mac_delta"] = macs - int(baseline["macs"])
            comparison["mac_growth_fraction"] = comparison["mac_delta"] / int(baseline["macs"])

    if apply and changed:
        layout.atomic_yaml(report_path, report)
        print(f"{run_root}: rewrote {report_path}")
    return changed


def run_is_live(run_root: Path) -> bool:
    """True while a training process still holds the run's advisory lock."""
    lock_path = run_root / ".run.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path, help="experiment run directories")
    parser.add_argument("--apply", action="store_true", help="write the corrected reports")
    args = parser.parse_args()

    total = 0
    for run_root in args.run_roots:
        if args.apply and run_is_live(run_root):
            print(f"{run_root}: still running, refusing to rewrite its report")
            continue
        total += remeasure(run_root, apply=args.apply)
    if total and not args.apply:
        print(f"\n{total} candidate(s) would change; rerun with --apply to write them")


if __name__ == "__main__":
    main()
