"""Census of local run/report artifacts, without reading credentials or models."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
KWS = HERE.parents[1]


def kind(path: Path) -> str | None:
    if path.name == "manifest.yaml":
        return "run_manifest"
    if path.name == "result.json":
        return "faithful_result"
    if path.parent.name == "reports" and path.suffix in {".yaml", ".json"}:
        return "stage_report"
    if path.name == "summaries.yaml" and path.parent.name == "metrics":
        return "training_summary"
    if path.name == f"{path.parent.name}_best_arch_scores.csv":
        return "canonical_pai_architecture_scores"
    if path.suffix == ".jsonl" and ("metrics" in path.parts or path.name == "epochs.jsonl"):
        return "canonical_epoch_log"
    if path.name in {"rp2040.yaml", "robustness.json"}:
        return "deployment_or_robustness_report"
    return None


def main() -> None:
    rows = []
    counts = defaultdict(Counter)
    for root in [KWS / "outputs", KWS / "dendritic_framework_w18"]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            category = kind(path)
            if category is None:
                continue
            relative = path.relative_to(KWS)
            family = relative.parts[1] if relative.parts[0] == "outputs" else relative.parts[0]
            contents = path.read_bytes()
            rows.append({
                "family": family,
                "artifact_kind": category,
                "path_relative_to_KWS_Model": str(relative),
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            })
            counts[family][category] += 1
    with (HERE / "source-artifact-census.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(rows),
        "warning": "Artifact counts are not independent run/replicate counts. PAI snapshots excluded; active outputs can change.",
        "families": {key: dict(value) for key, value in sorted(counts.items())},
    }
    (HERE / "source-artifact-census-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
