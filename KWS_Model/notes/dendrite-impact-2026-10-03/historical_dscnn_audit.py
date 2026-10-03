"""Read-only audit of historical DS-CNN reports and canonical PAI score files.

Run from the repository root with KWS_Model/.venv/bin/python. Only writes
derived evidence beside this script; does not import PerforatedAI or load models.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
KWS = HERE.parents[1]


def architecture_rows(directory: Path) -> list[dict]:
    path = directory / f"{directory.name}_best_arch_scores.csv"
    if not path.exists():
        return []
    with path.open() as stream:
        return [
            {"native_params": int(float(row["Param Counts"])),
             "best_validation": float(row["Max Valid Scores"])}
            for row in csv.DictReader(stream)
        ]


def metrics(path: Path) -> dict:
    if not path.exists():
        return {"epochs": 0}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    accuracies = [r.get("val_acc", r.get("val_accuracy")) for r in rows]
    accuracies = [v for v in accuracies if v is not None]
    return {
        "epochs": len(rows),
        "best_validation": max(accuracies) if accuracies else None,
        "recorded_seeds": sorted({r.get("seed") for r in rows if r.get("seed") is not None}),
        "summed_epoch_seconds": sum(r.get("elapsed_seconds", 0) for r in rows),
    }


def main() -> None:
    evidence = []
    for family in ["compression-run", "full-run-20260912T063022Z"]:
        root = KWS / "outputs" / family
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        for directory in sorted((root / "pai" / "candidates").iterdir()):
            if not directory.is_dir():
                continue
            arches = architecture_rows(directory)
            if not arches:
                continue
            key = directory.name
            pai_metrics = metrics(root / "metrics" / "sparsity" / key / "pai.jsonl")
            resume_metrics = metrics(root / "metrics" / "sparsity" / key / "resume_kd.jsonl")
            before = arches[0]
            selected = max(arches, key=lambda row: row["best_validation"])
            evidence.append({
                "family": family,
                "candidate": key,
                "run_status": manifest.get("status"),
                "source": str(directory.relative_to(KWS)),
                "canonical_architectures": arches,
                "zero_validation": before["best_validation"],
                "selected_validation": selected["best_validation"],
                "within_search_gain_pp": 100 * (selected["best_validation"] - before["best_validation"]),
                "native_param_delta": selected["native_params"] - before["native_params"],
                "pai_metrics": pai_metrics,
                "resume_metrics": resume_metrics,
                "held_out_test_available": False,
            })

    framework = KWS / "dendritic_framework_w18"
    evidence.append({
        "family": "dendritic_framework_w18",
        "source": str(framework.relative_to(KWS)),
        "canonical_architectures": architecture_rows(framework),
        "interpretation": "one zero-dendrite architecture, empty canonical switch CSV; partial baseline evidence",
        "held_out_test_available": False,
    })
    (HERE / "historical-dscnn-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for row in evidence:
        print(row["family"], row.get("candidate", ""),
              "architectures", len(row["canonical_architectures"]),
              "within_search_gain_pp", row.get("within_search_gain_pp"))


if __name__ == "__main__":
    main()
