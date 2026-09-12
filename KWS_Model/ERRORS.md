# Training run errors

Append-only log of errors hit during monitored `kws.pipeline` runs, kept
alongside the run's own `manifest.yaml`/`reports/` so a crash has a
human-readable trail in addition to the durable logs under the run's
`--output-dir`.

Each entry: when it happened, which run/stage/command, the failure signature,
what subagent was spawned to investigate, and how it was resolved. Schema
defects and other non-fatal problems noticed while monitoring a run belong here
too, marked OPEN until they are fixed.

---

## 2026-09-12 — OPEN — duplicate `val_acc` key in the metrics JSONL

**Run:** `outputs/full-run-20260912T063022Z` (stage 1 in progress; not a failure —
found by inspecting the metrics schema, the run is healthy).

**Signature:** every epoch record carries the validation accuracy twice, under
two names with byte-identical values:

```python
# src/kws/train.py:380-381
"val_acc": val_acc,
"val_accuracy": val_acc,
```

Same duplication at `src/kws/optimize/dendritic.py:369-370`. Confirmed across all
194 records written so far: the two fields differ in 0 of them. `learning_rate` /
`learning_rates` (`train.py:382-383`, `dendritic.py:371-372`) are the same defect
and should be fixed in the same pass; note both are *lists*, one entry per
optimizer param group, so the singular name is wrong as well as redundant.

**Fix:** drop `val_acc` from the epoch record only, keeping `val_accuracy`.

- `src/kws/train.py:381` and `src/kws/optimize/dendritic.py:369` — delete the key.
- `src/kws/train.py:289` — the one reader of the record's `val_acc`
  (`history[-1].get("val_acc", 0.0)`); repoint it at `val_accuracy`. Safe on old
  runs, because every record ever written carries both names.
- `src/kws/utils/graphs.py:352` already prefers `val_accuracy`, and its chart
  de-duplication drops the alias, so no change needed there.

**Do not touch** the `val_acc` key in checkpoint and report payloads
(`train.py:414,501,560`, `distill.py:228`, `cluster.py:675`,
`quantize_qat.py:368`, `dendritic.py:1019`, `pipeline.py:269,293,381,413`). That
is a different dict with the same key name, and it is read back by stage reuse
(`train.py:547`, `pipeline.py:257,375`), KD teacher loading (`kd.py:180`), and
dendritic provenance (`dendritic.py:1738`). Removing it there would break resume.
`tests/test_distill.py:129` and `tests/test_dendritic_resume_regressions.py:411`
assert on that payload and must keep passing unchanged.

**Status:** deferred until the current run finishes — it is a live schema change
against a JSONL that stage 1 is still appending to, and no stage depends on the
duplicate today.

---

## 2026-09-12 — OPEN — stage 2 silently trained the student from scratch

**Run:** `outputs/full-run-20260912T063022Z`, stage 2, started 12:36:18. Not a
failure; the run is healthy and still going.

**Signature:** stage 2 emitted a single WARNING and continued on a path the
config explicitly says is the worse one:

```
[WARNING] Configured optional student warm-start checkpoint
.../models/checkpoints/ds_cnn_xs_12class.pt is missing;
training the student from scratch
```

`_resolve_optional_checkpoint` (`pipeline.py:296-307`) downgrades a missing
warm start to `None` and logs at WARNING. `configs/train/pipeline.yaml:28-30`
meanwhile asserts warm starting "beat distilling from scratch by 6.3 points of
test accuracy (79.3% -> 85.6%), so it stays on." It did not stay on. Confirmed
from the loss curve: `distill-ds_cnn_xs epoch 1/200 val_acc=0.1998`, i.e. random
init.

**Root cause:** no pipeline stage produces that checkpoint. `STAGES` is
`(teacher, student, sparsity, cluster, quantize, benchmark)`, where `student`
is the distillation itself; the warm start is an *external prerequisite* that
must come from a separate hard-label `kws.train` run on `ds_cnn_xs`. A fresh
clone can therefore never satisfy it, and the pipeline will always take the
degraded path without failing.

**The 6.3-point claim is stale and does not describe this task.** Its evidence
is 6-class:

| report | accuracy | params | classes |
| --- | --- | --- | --- |
| `reports/ds_cnn_xs_distilled.json` | 0.7934 | 3850 | 6 (`yes no on off _unknown_ _silence_`) |
| `reports/ds_cnn_xs_distilled_warm.json` | 0.8563 | 3850 | 6 |

This pipeline is 12-class at ~4,096 params, so capacity per class is roughly
halved and both arms are pushed harder against the capacity ceiling. The gain
has never been measured in the 12-class regime, and it is a single-seed n=1
comparison on a model small enough for seed variance to be material. The one
12-class artifact that exists, `ds_cnn_xs_distilled_warm_12class.pt`, sits at
val 0.7302 — well below 85.6% — though it may be from an aborted run.

Neither checkpoint on disk can serve as the warm start: `ds_cnn_xs.pt` is
6-class (`num_classes: 6`), so `load_state_dict` fails against the 12-wide head
even though `model_cfg` matches; `ds_cnn_xs_distilled_warm_12class.pt` is a
distillation output, not a pretrain.

**Decision:** let stage 2 finish from scratch. It yields the 12-class
from-scratch baseline that the 6-class number is missing, at no extra cost —
land near 85% and warm starting has little headroom, land near 79% and it is
clearly worth the ~3.3 h pretrain. Re-deciding then is free; deciding now costs
~7 h (3.3 h pretrain + restarting the ~4 h stage 2) on an untested assumption.

**Fix, once measured:**

- If the gain holds: add a student-pretrain step that writes
  `ds_cnn_xs_12class.pt` before stage 2, so the prerequisite is produced rather
  than assumed.
- Either way, make the mismatch loud. A config that names a warm start whose
  file is absent should fail fast, or at minimum record the downgrade in the
  run report instead of only the log — this cost a 4-hour stage before anyone
  noticed.
- Replace the `pipeline.yaml:28-30` comment with a 12-class measurement, or
  mark it explicitly as 6-class provenance.

**Status:** open, pending the stage 2 result (~16:40 local).

---
