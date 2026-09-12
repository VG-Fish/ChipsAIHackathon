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

**Measured 2026-09-12 17:02 — the claim does not hold at 12 classes.** Stage 2
finished 200/200 (best val 0.8507 at epoch 167, final val 0.8457) and the
from-scratch student scores **0.8364 test**, 4,096 params:

| run | classes | params | test acc |
| --- | --- | --- | --- |
| 6-class, from scratch | 6 | 3850 | 0.7934 |
| 6-class, warm-started | 6 | 3850 | 0.8563 |
| 12-class, from scratch (this run) | 12 | 4096 | 0.8364 |

From-scratch at 12 classes lands 4.3 points *above* the 79.3% baseline the
6.3-point claim rests on, on a harder task. The deficit the config warns about
does not exist in this regime.

This refutes the baseline, not warm-starting itself: no warm-started 12-class
run exists, so a smaller gain there remains possible but unmeasured. Evidence it
would be small — val flattened from roughly epoch 130 while `train_loss` held
near 0.92, which is capacity-bound, and initialization has least to offer in
that regime.

**Resolution:** keep from-scratch. Do not build the pretrain stage on the
strength of the 6-class number. Revisit only with a measured 12-class A/B, and
only if the student is given more capacity.

**Status:** resolved for this run — the warm-start config entry is still a
silent-degradation hazard, which is tracked as the "make the mismatch loud" fix
above and in the stale-paths entry below. Stage 3 began 16:53.

---

## 2026-09-12 — OPEN — stale paths and legacy-format shims to retire

**Run:** found while investigating the warm-start entry above; no run has failed
because of any of it. Filed so the cleanup happens deliberately rather than by
someone deleting the wrong thing.

The repo still carries a 6-class / 4-keyword generation of artifacts that the
12-class pipeline no longer uses, plus a set of compatibility shims that exist
only to accept those older shapes. Three groups, in descending order of risk.

### 1. Provenance defect — a snapshot that names the wrong teacher

`metadata/configs/sparsity_search.yaml` in every run records:

```yaml
teacher_checkpoint: models/checkpoints/ds_cnn_l_12class.pt
```

That is the repo-root teacher from 2026-09-11, **not** the teacher the run
actually used. At runtime `pipeline.py:871` passes
`config["teacher"]["checkpoint"]` (rebased to the run root) and
`dendritic_prune_loop.py:469` takes the explicit argument over the config value
(`teacher_checkpoint or search_cfg.get("teacher_checkpoint")`), so the run is
correct — but the snapshot that exists to document the run is not. Anyone
auditing stage 3 reads a teacher that was never loaded.

Fix: rebase the value into the snapshot before writing it, or drop the key from
`configs/train/dendritic_prune_loop.yaml:35` entirely and make the pipeline the
only source of the sparsity teacher. This is the one item worth doing even if
nothing else here is touched.

### 2. Stale 6-class artifacts still on disk

Six of the eight repo-root checkpoints and seven of the eight repo-root reports
predate the 12-class relabelling:

| path | classes | keep? |
| --- | --- | --- |
| `models/checkpoints/ds_cnn_l_12class.pt` | 12 | keep — current teacher fallback |
| `models/checkpoints/ds_cnn_xs_distilled_warm_12class.pt` | 12 | keep — only 12-class student artifact |
| `models/checkpoints/{ds_cnn_l,ds_cnn_m,ds_cnn_s,ds_cnn_xs,ds_cnn_xs_distilled,ds_cnn_xs_distilled_warm}.pt` | 6 | retire |
| `reports/ds_cnn_l_teacher_current.json` | 12 | keep |
| `reports/{ds_cnn_l,ds_cnn_m,ds_cnn_s,ds_cnn_xs,ds_cnn_xs_distilled,ds_cnn_xs_distilled_warm,dendritic_prune_w15_test}.json` | 6 | retire |

The 6-class checkpoints are not merely unused — they are unusable. Their
`model_cfg` matches the current XS config exactly, so nothing rejects them up
front, but `load_state_dict` fails on the 6-wide classifier head. A 6-class file
with a matching `model_cfg` is a trap for anyone wiring up a warm start.

Five `dendritic_prune_w{14,15,15_reload,16,17}/` run directories are likewise
superseded; `configs/train/dendritic_prune_loop.yaml:4-6` already states they
"predate KD pre-fine-tuning, the corrected freeze, KD resume, and cost
profiling" and are "evidence only".

Retiring should mean moving to an `archive/` directory with a note, not
deleting: the 6-class reports are the sole evidence behind the warm-start claim
in the entry above, and that comparison still needs re-measuring before its
numbers are thrown away.

### 3. Legacy-format shims

Each accepts a shape the current pipeline never produces:

- `artifacts.py:259` `legacy_output_path()` — validates an old flat
  `--checkpoint some/path.pt` flag, then discards everything but the basename
  and rebases it into a fixed category. Callers: `train.py:614`,
  `distill.py:192`.
- `pipeline.py:1390-1410` — the same basename trick for pipeline output fields,
  commented "Existing YAML uses paths such as models/exported/foo.pt. Preserve
  that legacy spelling". Retiring it means rewriting the output paths in
  `configs/train/pipeline.yaml` to be category-relative.
- `checkpointing.py:140` — the "legacy best-only checkpoints" branch of
  `require_training_state`, for checkpoints written before `format_version`.
- `dendritic_prune_loop.py:319,330` — rejects candidates missing
  `cycle_metadata.yaml` or carrying an older `framework_cycle_version`. Tied to
  the `dendritic_prune_w*` directories in group 2; retire together.

**Do not touch:**

- `src/kws/optimize/quantization_compat.py` — despite every "deprecated" and
  "legacy" mention in it, this is *forward*-compat: it bypasses
  `torch.ao.quantization` eager entry points that PyTorch 2.10+ deprecates.
  Removing it reintroduces a deprecation warning on every QAT step and breaks
  stage 5 on newer PyTorch.
- `src/kws/data/splits.py:5` — the hash fallback is deliberate, for files absent
  from the official split lists.

**Sequencing:** group 1 is a one-line provenance fix and can go in now. Groups 2
and 3 must wait for the current run — stage 3 has not started, and it is the
stage that reads `framework_cycle_version` and the repo-root teacher fallback.

**Status:** open. Group 1 ready; groups 2-3 deferred until the run completes.

---

## 2026-09-12 — INVESTIGATING — stage 3 crashed: PAI native checkpoint path mismatch

**First actual run failure.** Everything above this entry was found by inspection; this
one killed the pipeline.

**Run:** `outputs/full-run-20260912T063022Z`, stage 3 (sparsity), candidate `w18`,
dendrite-growth phase. Crashed 2026-09-12 17:41:21, exit code 1, after 15h11m of
wall clock. Background task `b7dnaga6n`.

**Signature:**

```
RuntimeError: PerforatedAI did not create the required native checkpoint:
.../outputs/full-run-20260912T063022Z/pai/candidates/candidate_w18/latest.pt
```

Raised at `src/kws/optimize/dendritic.py:417` inside `_save_pai_restart_pair`,
called from `run_cycle` at `dendritic.py:1589`, via
`dendritic_prune_loop.py:563 run_pruning_search` ← `pipeline.py:465 stage_sparsity`.

**When:** 74 seconds into the dendrite phase, at the *first* checkpoint save. The
PAI wrapper had already initialised cleanly (`PAI-wrapped initial parameter
count: 1830`), run one validation (`Adding validation score 0.79093539`), and
logged one switch check (`Returning False - no triggers to switch have been
hit`). So PAI itself is working; only the save contract is wrong.

**Root cause (high confidence, pending subagent confirmation):** a path contract
mismatch between how we call PAI's `save_system` and where we look for its
output. `dendritic.py:411` calls

```python
save_system(model, str(run_dir.parent), pai_run_name)
```

with `run_dir = pai/candidates/candidate_w18`, i.e. `folder=pai/candidates`,
`name=candidate_w18`. The installed PAI signature is
`def save_system(net, folder, name)` (`utils_perforatedai.py:955`, read from the
embedded source in the compiled `utils_perforatedai.c`), and it writes
`<folder>/<name>.pt`. Line 414 then requires `<run_dir>/latest.pt`, a path PAI
never produces. Confirmed by what is actually on disk after the crash:

```
pai/candidates/candidate_w18.pt          <- PAI wrote this
pai/candidates/candidate_w18_pai.pt      <- and this
pai/candidates/candidate_w18/candidate_w18_config.json   <- only the config
```

The guard is correct to be strict — it exists so a missing native checkpoint
cannot be silently attested in the sidecar — it is simply guarding the wrong
path.

**Why it survived until now:** this is the first time any run has reached the
dendrite phase's first save. Stages 1-2 never touch PAI, and the
`dendritic_prune_w{14..17}` evidence runs predate the current framework cycle
(`configs/train/dendritic_prune_loop.yaml:4-6` says as much). Whether the
existing resume tests mock `save_system` and therefore never exercise the real
path is part of the subagent's brief.

**State preserved — nothing before the crash was lost:**

| artifact | status |
| --- | --- |
| stage 1 teacher | complete, val 0.9769 / test 0.9735 |
| stage 2 student | complete, val 0.8507 / test 0.8364 |
| candidate w18 prune-KD | complete, 40/40 epochs, best val 0.7938 |
| `sparsity/candidate_w18/prune_kd/best.pt` | intact, 17:38 |
| `sparsity/candidate_w18/prune_kd/latest.pt` | intact, 17:40 |

Only the 74-second dendrite phase is lost. Resuming into the same
`--output-dir` should reuse stages 1-2 and the finished prune-KD phase.

**Subagent:** spawned 17:43 with the traceback, the `_save_pai_restart_pair`
body, the on-disk evidence, and instructions to extract the real `save_system`
and `load_system` source out of the Cython `.c`, find every reader/writer of the
native path, and propose a minimal fix that preserves resume. Told not to edit.

### What the investigation actually found: four defects, not one

Fixing only the crash would have failed again seconds later. The subagent
extracted the real PAI source from the embedded Python in
`utils_perforatedai.c` and established the contract exactly:
`save_system(net, folder, name)` writes `<folder>/<name>.pt`, `load_system`
reads the same, and `latest` is PAI's conventional *name*, never a synthesized
directory entry. PAI's own tracker does
`UPA.save_system(net, GPA.pc.get_save_name(), "latest")` on every
`add_validation_score` (`tracker_perforatedai.py:3536-3540`). So our code was
written against the right convention with the two arguments transposed.

| # | defect | fix |
| --- | --- | --- |
| A | `save_system(model, run_dir.parent, pai_run_name)` writes `<parent>/<candidate>.pt`, but line 414 demands `<run_dir>/latest.pt` | `save_system(model, str(run_dir), "latest")` |
| B | `load_system` inverted identically (`dendritic.py:1395`) — save and load were self-consistent with each other and both wrong, so fixing A alone would have **silently broken resume** | mirror the same contract |
| C | the native file is **safetensors**, not a torch pickle (`using_safe_tensors` defaults on). `torch.load` on it throws, and the `run_id` stamp + `atomic_torch_save` underneath would have **overwritten PAI's safetensors with a pickle** that `load_net` can never read | drop the load/stamp/re-save on both the write (`:421-432`) and read (`:281-282`) sides; the sidecar's `native_pai_latest_sha256` already binds the pair, which is stronger than an in-file field |
| D | `perforate_model` rejects a path separator in `save_name` (`utils_perforatedai.py:99`, `sys.exit(1)`), so PAI keeps a *relative* name and resolves every write against the cwd. The `chdir` was scoped to `perforate_model` only, so **PAI leaked the whole run into the repo root** | `_pai_cwd()` context manager around each PAI-writing site |

Defect C verified directly against the artifact: `torch.load` raises
`UnpicklingError: invalid load key, '\xb8'`; `safetensors.load_file` returns 149
tensors. Defect D verified by the leaked `KWS_Model/candidate_w18/` directory,
which contained `latest.pt`, `best_model.pt`, the architecture CSVs and the PNG
— including the very `latest.pt` the assertion wanted, written against the
process cwd instead of the run.

D also had a delayed second bite: `read_pai_architecture_results` (`:1645`)
reads `<run_dir>/<name>_best_arch_scores.csv`, which PAI would have written to
the repo root — a `FileNotFoundError` *after* the full dendrite phase completed.

**`chdir` scope decision.** The subagent offered a simpler variant holding the
`chdir` across the whole PAI section. Rejected: `configs/data/speech_commands_v2.yaml:5`
sets `root: "data/raw/speech_commands_v0.02"`, a **relative** path, so a
process-wide cwd change would break data loading every epoch. The scope is kept
narrow deliberately and `_pai_cwd`'s docstring records why.

### The test that hid defect A actively asserted the wrong contract

`tests/test_dendritic_resume_regressions.py:79-82` monkeypatched `save_system`
with a fake writing `<folder>/<name>/latest.pt` as a **torch pickle** — encoding
our assumption as truth rather than PAI's behaviour. That single fake hid A (wrong
path shape) and C (wrong format, since `torch.load` then succeeded), and since
`load_system` was never exercised it hid B too. The fake now mirrors the real
contract (`save_file({...}, Path(folder)/f"{name}.pt")`), which is what makes
this class of bug catchable. It failed immediately when the code became correct
— the correct signal.

### Two further defects surfaced on relaunch, unrelated to PAI

**E — epoch-resume has never worked on MPS.** `train.py:487` loaded the resume
checkpoint with `map_location=device`, dragging the uint8 RNG tensor onto MPS;
`torch.set_rng_state` requires a CPU `ByteTensor`. Crashed at
`checkpointing.py:52`, then again at `train.py:279` on the DataLoader generator
states — the same bug at a second site. Lines 545/552 already used
`map_location="cpu"`; 487 was the odd one out. This never fired before because
nothing had ever resumed. Fixed at the source (`map_location="cpu"`, the tensors
that belong on the device get there via `load_state_dict` /
`move_optimizer_state_to_device`) plus defensive CPU/uint8 normalisation at both
`set_state` sites.

A sweep for the same class found exactly two RNG-bearing resume paths — `train.py`
and the PAI sidecar resume — so `dendritic.py:192` (`map_location=device`) and
`dendritic.py:1473` were fixed the same way before they could bite. The other
nine `map_location=device` sites restore no RNG state and are unaffected.
Verified against the real checkpoint: `restore_rng_state` now succeeds even on a
deliberately MPS-loaded payload.

**F — self-inflicted: manifest referenced the removed debris.** Clearing the
crash debris left three `runtime_artifact` entries in `manifest.yaml` pointing at
moved files, so manifest finalization failed. Removed exactly those three
entries (original backed up to the scratchpad). The `status: failed` invocation
record at `manifest.yaml:32-34` was **kept** — that is the audit trail of the
crash, not stale state.

**Debris handling:** moved to the scratchpad rather than deleted, so it stays
inspectable. `models/checkpoints/sparsity/candidate_w18/prune_kd/` was preserved
deliberately — that is the completed 40-epoch phase being reused.

**Verification:** full suite 193 passed after every change.

### G — OPEN — duplicate epoch 1 in `metrics/sparsity/candidate_w18/pai.jsonl`

Left by the crash. The dendrite phase wrote one record before dying at 17:41,
and the relaunch wrote its own epoch 1 at 18:03, so the file carries two
`"epoch": 1` records that differ only in noise:

| line | elapsed_seconds | val_accuracy | source |
| --- | --- | --- | --- |
| 1 | 73.639 | 0.7909353905 | crashed attempt |
| 2 | 75.838 | 0.7907425265 | current run |

`MetricsRecorder.reconcile()` — which exists for exactly this — is only called
on the *resume* path (`train.py:268`, `dendritic.py:1475`). The dendrite phase
began fresh rather than resuming, so `append()` ran against a file that still
held a dead run's record. `reset_for_fresh_run()` / the `.archive.jsonl`
rotation never fired.

Consequence is cosmetic but real: the live chart plots two points at epoch 1,
and any consumer keying on `epoch` sees a duplicate. It does not affect training
or checkpoint selection.

**Not fixed during the run** — the file is held open for append by a live
process; rewriting it mid-run risks interleaving. Fix afterwards by dropping
line 1, and separately make a fresh (non-resume) phase rotate a pre-existing
metrics file the way a resume does. This is the same shape of defect as the
`val_acc` duplicate in entry 1: nothing validates the epoch sequence in a JSONL.

**Status:** fixed and relaunched 18:12 into the same
`--output-dir outputs/full-run-20260912T063022Z`. Monitoring at 5-minute
cadence at the user's request. Remaining risk is the dendrite phase's *later*
stages, which no run has ever reached: the `read_pai_architecture_results` CSV
read (guarded by the D fix but unexercised) and the first true PAI *resume*
(guarded by the B and E fixes but likewise unexercised).

---
