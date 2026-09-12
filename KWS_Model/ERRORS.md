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
