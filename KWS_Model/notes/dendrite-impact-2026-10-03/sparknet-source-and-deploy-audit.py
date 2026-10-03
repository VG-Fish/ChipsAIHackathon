"""Reproduce source checkpoint shapes, manifest provenance, deployment metadata.

Read only original artifacts. Write only this research directory. No PAI import,
dataset construction, training, board execution, or private environment reads.
"""
from pathlib import Path
import hashlib, json, sys
import yaml

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kws.models.sparknet import build_sparknet
from kws.models.sparknet_port import load_lightning_state_dict, map_state_dict
from kws.utils.profile import count_macs

released = []
for c in [4, 8, 16, 32]:
    path = ROOT / f"models/checkpoints/external/sparknet/kws_C_{c}.ckpt"
    state = load_lightning_state_dict(path)  # repository's restricted unpickler
    mapped, width, gate = map_state_dict(state)
    cfg = yaml.safe_load((ROOT / "configs/model/sparknet_c16_paper.yaml").read_text())
    cfg["channels"] = width
    cfg["gate_channels"] = gate
    model = build_sparknet(cfg, (32, 101), 12)
    model.load_state_dict(mapped, strict=True)
    released.append(dict(file=str(path.relative_to(ROOT.parent)), width=width,
                         gate_channels=gate, n_features=int(state["fs.encoder.0.mconv.0.weight"].shape[0]),
                         params=sum(p.numel() for p in model.parameters()),
                         local_macs=count_macs(model, (32, 101)),
                         shapes={k: list(v.shape) for k, v in state.items()
                                 if k.startswith("preprocessor.") or k.endswith("mconv.0.weight")
                                 or k == "output_layer.0.weight"}))
(P / "sparknet-released-checkpoint-audit.json").write_text(json.dumps(released, indent=2))

rows = json.loads((P / "sparknet-all-runs.json").read_text())
provenance = []
for row in rows:
    manifest = yaml.safe_load((ROOT / row["run"] / "manifest.yaml").read_text())
    for source in manifest.get("inputs", []):
        if source.get("role") not in ["data_config", "model_config", "train_config"]:
            continue
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        provenance.append(dict(run=row["run"], family=row["family"], role=source["role"],
                               path=str(path), recorded_hash=source.get("sha256"),
                               exists=path.exists(), current_hash=current_hash,
                               matches=current_hash == source.get("sha256")))
(P / "sparknet-config-provenance.json").write_text(json.dumps(provenance, indent=2))

deploy = []
for path in sorted((ROOT / "outputs/rp2040").glob("*/reports/rp2040.yaml")):
    report = yaml.safe_load(path.read_text())
    deploy.append(dict(name=report["name"], source=report["source"],
                       float_params=report["float_parameters"], folded_params=report["folded_parameters"],
                       chosen=report["chosen"], test_float=report["test"]["float"],
                       test_int=report["test"]["int"], agreement=report["test"]["agreement"],
                       weight_int8_bytes=report["int8_weight_bytes"],
                       scratch_bytes=report["activation_scratch_bytes"],
                       host_parity=report["host_parity"],
                       firmware={k: report["firmware"].get(k) for k in
                                 ["uf2", "uf2_sha256", "binary_bytes", "static_ram_bytes_approx", "self_test_clips"]}))
(P / "sparknet-rp2040-evidence.json").write_text(json.dumps(deploy, indent=2))
assert not any(k.startswith(("perforatedai", "perforatedbp")) for k in sys.modules)
print(f"Released graphs: {len(released)}; configuration hash records: {len(provenance)}; deployment reports: {len(deploy)}")
