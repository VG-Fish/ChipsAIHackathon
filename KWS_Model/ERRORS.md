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

## 2026-09-13 — stale cache test after enabling fixed augmented views

- **Symptom:** `test_precomputed_features_skip_repeated_deterministic_extraction`
  expected one cached tensor for a two-entry fixture and failed with
  `assert 2 == 1`.
- **Diagnosis:** The test encoded the previous policy that synthesized silence
  was always regenerated. The requested augmented-training cache deliberately
  materializes silence and its augmentation once at startup, so `2` is the
  correct count.
- **Resolution:** Updated the expectation and documented that repeated silence
  accesses reuse the cached augmented feature. No production defect found.

## 9. Epoch-301 `.fc` cleanup rejected PAI's real branch container and missed its terminal integration

**Date:** 2026-09-13
**Status:** RESOLVED; THE SAVED EPOCH-301 PAI/KWS PAIR IS RECOVERABLE

### Production failure

Invocation `c94ed2f30330` reached PAI's normal terminal condition at epoch 301
with `conversion: fc_only`. PAI logged the complete final transition and
integration:

```text
Module .fc calling set mode p : tensor([1.], device='mps:0')
Module .fc calling set mode n : tensor([2.], device='mps:0')
Final dendrites successfully integrated! Total integrated: 1
```

The KWS wrapper then logged the wrong optimizer policy and failed while
repeating PAI's final cleanup:

```text
PAI restructured the network; resetting optimizer phase with LR multiplier 1.0000
RuntimeError: PAI cleanup module 'fc' has no two-branch layer_array
```

No pipeline restart was attempted. The failure happened after the epoch-301
native/KWS restart pair was committed and before step 3d KD resume, clustering,
quantization, or benchmarking.

### Root cause 1: the cleanup graph was valid, but the type check was not

`_restore_single_dendrite_skip_weights()` required `layer_array` to be an
`nn.ModuleList`. That reflected the PAI *training* wrapper, not its deployed
wrapper. In installed PerforatedAI 3.2.8, `prepare_final_model()` calls
`blockwise_network()` before `clean_perforatedai.refresh_net()`, and the real
clean `.fc` is a `PAIModulePyThread` whose two registered branches are held in
an **`nn.Sequential`**.

The exception text therefore reported a missing two-branch graph even though
the actual object had exactly two branches. Direct inspection of the saved
artifacts and a reconstruction from the paired sidecar confirmed:

```text
clean.fc type:         PAIModulePyThread
clean.fc.layer_array:  nn.Sequential
branch count:          2
```

The native `latest_pai.pt` and vendor-written `final_clean_pai.pt` both contain
`fc.layer_array.0.*` and `fc.layer_array.1.*`. The live epoch-301 sidecar also
contains the learned `fc.dendrites_to_top.0` coefficient (shape `[1, 12]`).
PAI's clean-wrapper constructor intentionally omits `skip_weights` when there
is one integrated dendrite, so that coefficient still has to be restored;
silently skipping the module would have produced an inference checkpoint with
the wrong forward function.

The fix accepts both registered ordered containers PAI uses (`nn.ModuleList`
and `nn.Sequential`) while retaining the branch-count and module-presence
guards. Clean-graph dendrite/base classification now recognizes both types as
well. A production-state reconstruction in a temporary directory restored
exactly `fc.skip_weights.0` and matched the live PAI model exactly on a random
four-example batch:

```text
max absolute output difference: 0.0
torch.allclose(...):             True
```

No file under the saved production run was changed by that check.

Independent review found the same stale `ModuleList`-only assumption in the
step-3e activation-memory profiler. It would not have blocked recovery, but it
could have undercounted the live residual-branch activation set after this real
`Sequential` cleanup and therefore distorted one Pareto cost axis. The profiler
now accepts both ordered containers, with a parameterized liveness regression
covering each representation.

### Root cause 2: pre/post mode sampling cannot see PAI's terminal `p` phase

The LR selector inferred an integration only from externally sampled
`previous_mode == "p" and next_mode == "n"`. At the configured one-dendrite
maximum, PAI performs `n -> p -> n` entirely inside one
`add_validation_score()` call. Both samples were consequently `n`, even though
the dendrite was integrated, and `_restructure_lr_multiplier()` returned 1.0
instead of the configured 0.25.

This is visible in the durable phase trail: ordinary externally visible
integrations at epochs 142 and 213 recorded the requested 0.25 multiplier,
while terminal epoch 301 alone recorded 1.0. The training loss and score were
not the cause; this was solely a transition-observation defect.

The loop now samples PAI's monotonic `num_dendrites_integrated` counter before
and after `add_validation_score()`. A counter increase applies the gentle
restart even when the sampled modes are both `n`; the old `p -> n` check remains
as a compatibility fallback if that private counter is unavailable. The phase
trail records the before/after counters and the derived integration decision
for future audit.

### Root cause 3: the sweep mistook PAI's vendor export for KWS completion

PAI writes `final_clean_pai.pt` inside `add_validation_score()`, before the KWS
export shim runs and before KD resume, profiling, or `cycle_metadata.yaml`.
`run_pruning_search()` nevertheless used existence of that vendor file alone
to select its completed-candidate reuse path. Restarting after this exact crash
would therefore have tried `_load_completed_result()` and failed on the absent
cycle metadata instead of reaching the valid sidecar resume.

Completed-candidate selection now requires both `final_clean_pai.pt` and
`cycle_metadata.yaml`. A vendor final file without KWS metadata remains in the
partial-candidate branch, where the paired sidecar selects normal recovery.
This distinction is covered directly and requires no artifact move or delete.

### Recovery of invocation `c94ed2f30330`

The existing recovery pair is internally valid and should be reused:

```text
KWS sidecar completed_epoch: 301
KWS sidecar next_epoch:      302
run_id:                      9e163d95fbca4f80b2c7536151ce36e4
native latest SHA-256:       6c50b17fb71be9ef17822112861578aac45426f0e32637e4bcdaf69de877dcd2
tracker mode:                n
num_dendrites_added:         1
num_dendrites_integrated:    1
configured max_dendrites:    1
```

`_load_pai_sidecar()` verified the path, schema, run ID, and native digest. The
sidecar retains the full live graph and learned top coefficient. Because its
tracker is already terminal, resume now retries export directly rather than
training an unrequested epoch 302. This also prevents the historically saved
1.0 optimizer reset from taking a step; the bad reset occurred after epoch 301
and before the paired save, but no batch ever executed with it.

The vendor-written `final_clean_pai.pt` in the failed run is not by itself the
finished KWS deployment artifact because it lacks the restored skip
coefficient. It must not be treated as proof that steps 3d-3f completed. The
paired epoch-301 live state is the authoritative recovery source, and the
normal resume path will replace the clean artifact atomically after applying
the corrected export.

### Regression coverage and verification

Coverage now includes:

1. restoration into PAI's actual `nn.Sequential` clean-branch shape;
2. preservation of legacy `nn.ModuleList` support;
3. an internal terminal `n -> p -> n` simulation that increments only the
   integration counter and proves the reset uses 0.25 before the paired save;
4. terminal-boundary detection for both the configured maximum and PAI's
   `doing_pai=False` completion path;
5. classification of a vendor-only final artifact as resumable rather than a
   completed KWS cycle;
6. reconstruction and parity export from the exact production state in an
   isolated temporary directory; and
7. identical residual activation-liveness accounting for clean `Sequential`
   and training `ModuleList` branch containers.

```text
uv run --env-file .env python -m pytest \
  tests/test_dendritic.py tests/test_dendritic_resume_regressions.py \
  tests/test_prune_loop.py tests/test_profile.py -q
```

All **62/62** focused tests pass. `python -m compileall -q src tests` and
`git diff --check` also pass. Per the PerforatedAI debugging protocol, no
training script or pipeline was started during diagnosis.

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

## 5. PAI drops into `pdb` at the first dendrite switch (mode `n` -> `p`)

**When:** 2026-09-12 19:56:42, exit code 1. Candidate `w18`, logged epoch 92,
1h54m into the relaunched run (`outputs/full-run-20260912T063022Z`, invocation
`1e43964ef188`). This is the first time any run has reached a dendrite switch.

**How it got here:** PAI's patience counter (`n_epochs_to_switch: 25`) expired
at logged epoch 92 with `last improved epoch 66`, `global_best = 0.8204`. The
switch to dendrite mode (`p`) fired as designed, and the crash happened inside
that first dendrite epoch.

### Traceback

```
File "src/kws/optimize/dendritic.py", line 1563, in run_cycle
    GPA.pai_tracker.add_validation_score(val_acc, model)
File "perforatedai/tracker_perforatedai.py", line 3534, in add_validation_score
File "perforatedbp/tracker_pbp.py", line 578, in check_best_pai_score_improvement
File "perforatedbp/tracker_pbp.py", line 312, in best_pai_score_improved_this_epoch
File "perforatedbp/tracker_pbp.py", line 361, in best_cascor_score_improved_this_epoch
File "perforatedai/modules_perforatedai.py", line 490, in PAINeuronModule.__getattr__
File "torch/nn/modules/module.py", line 1959, in __getattr__
-> def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
(Pdb)
...
bdb.BdbQuit
```

The process runs with `stdin=/dev/null`, so entering `pdb` is immediately fatal:
`bdb.BdbQuit` propagates and the pipeline exits 1. The `stdin` choice is not the
bug -- it converts a silent hang into a loud failure -- but it does mean any PAI
`set_trace()` is unrecoverable.

### The decisive diagnostic (printed immediately before the pdb)

```
Adding validation score 0.81967213
You set GPA.pc.set_initial_correlation_batches() to be greater than an entire
epoch 0 < 40. ... Start over or Load from 'latest' for candidate_w18.
It was caught on layer .blocks.0
If your epoch is larger than this number it means the layer is not being
included in autograd backwards.
```

**The message is misleading.** Our `initial_correlation_batches: 40` is not
larger than an epoch -- the train loader has 337 batches. The `0` in `0 < 40`
is the *observed* count: the correlation-batch counter never incremented past
zero. PAI's own second sentence is the real diagnosis -- "the layer is not being
included in autograd backwards."

Corroborating output, repeated every batch of the dendrite epoch:

```
skipping pb batch for .blocks.0
skipping pb batch for .blocks.1
skipping pb batch for .fc
.blocks.0 is in backwards graph multiple times.
Dendrite outs and neuron errors are currently stacked (336/0) times
```

`(336/0)` is the tell: 336 dendrite outputs recorded, **zero neuron errors**.
The backward hooks that record neuron errors never fired for any tracked layer.

### Leading hypothesis (NOT yet verified -- subagent investigating)

Our own base-weight freeze is starving PAI's correlation machinery of gradients.

- `dendritic.py:863` `apply_phase_freezing()` runs every epoch; on `mode == "p"`
  it calls `enforce_base_weight_freeze(model)`.
- `dendritic.py:833` `enforce_base_weight_freeze()` sets `requires_grad = False`
  on **every** non-dendrite parameter. Its docstring calls this "belt and braces
  over PAI's own freezing".
- The loop comment at ~1500 states the intent plainly: *"PAI may change
  requires_grad during the call above"* -- i.e. we deliberately override PAI.
- `configs/train/dendritic_cycle1.yaml:64` `enforce_base_weight_freeze: true`.

If PAI's cascade-correlation scoring needs gradients to *flow to* base neuron
modules (to capture neuron errors via backward hooks) while simply not
*updating* them, then `requires_grad=False` severs exactly that path. Autograd
never reaches the base modules, hooks never fire, neuron-error count stays 0,
the correlation counter stays 0, and PAI trips its guard.

If this holds, it is a well-intentioned safety measure that silently disabled
the library's core algorithm -- and note the shape: the docstring worried about
PAI *under*-freezing in some future release, and defended against it by
over-freezing now. Any fix must still preserve step 3c's real requirement (base
weights must not move while dendrites are scored); satisfying that by excluding
base params from the optimizer, or by zeroing their grads before `step()`, would
keep the guarantee without cutting the autograd path.

### Open questions for the investigation

1. Does PAI itself set base `requires_grad=False` in dendrite mode, or keep it
   `True`? This is the crux and must be read from PAI's real source (the package
   is compiled Cython; original Python survives as comments in the generated
   `.c` files, `inspect.getsource` fails).
2. What attribute lookup in `best_cascor_score_improved_this_epoch`
   (`perforatedbp/tracker_pbp.py:361`) misses `PAINeuronModule.__getattr__` and
   falls through to torch's, and is the `pdb` entry gated by a config flag that
   would make PAI raise instead of trap?

**State of the run:** dead at 19:56:42. PAI's message says `Load from 'latest'
for candidate_w18` is possible, so the 91 completed neuron-mode epochs need not
be retrained -- and taking that path would finally exercise the PAI *resume*
code guarded by entry 4's B and E fixes, which no run has reached either.

### Root cause (confirmed from source + experiment)

The hypothesis above was right about the culprit and **wrong about the premise**,
and the wrong premise is the whole lesson.

PAI **never** sets `requires_grad=False` on base parameters. Every
`requires_grad_(False)` in `modules_perforatedai` is on `dendrites_to_top`. PAI
holds the base still by **filtering it out of the optimizer**
(`set_optimizer_instance` -> `setup_optimizer_pb`). Our code re-asserted a
guarantee PAI never made, using a mechanism that destroys PAI's:

```python
# modules_perforatedai.py:14927 -- the hook that records neuron errors
if out.requires_grad and (not self._fb_init_done or ...):
    out.register_hook(lambda grad: filter_backward(grad, ...))
```

Freeze the base and `.blocks.0`'s output has no `grad_fn`, the hook never
registers, `filter_backward` is never called, neuron errors stay 0 forever
(`stacked (336/0)`), the correlation counter never leaves 0, and PAI trips its
guard and calls `pdb.set_trace()`. `Best PBScores.csv` was written with a header
and **zero data rows** -- independent confirmation that nothing was ever scored.

The `requires_grad`-based optimizer filtering that does exist in PAI
(`tracker_perforatedai.c:30729`) lives inside `setup_optimizer`, which is PAI's
*Pattern 1*. We use *Pattern 2* (`set_optimizer_instance`), so it never applied
to us. The old comment was written as if we were on Pattern 1.

Also: the bottom two traceback frames are a **red herring**.
`modules_perforatedai.py:490` is the `try:` arm of `__getattr__`, not a fallback.
pdb read EOF, set `quitting=True`, and `BdbQuit` surfaced at the next trace
dispatch -- which happened to be a `--Call--` on torch's `__getattr__`. The
frame has nothing to do with the fault.

### Fixes applied

| # | Fix | File |
|---|-----|------|
| 1 | `enforce_base_weight_freeze: false` (+ code default flipped to `False`) | `dendritic_cycle1.yaml:66`, `dendritic.py` |
| 2 | New `freeze_base_batchnorm_stats()` -- pins base BN via `.eval()`, never `requires_grad` | `dendritic.py` |
| 3 | New `base_params_in_optimizer()` -- checks 3c the way PAI actually enforces it; logs, never raises; returns `-1` for "unmeasurable" | `dendritic.py` |
| 4 | New `disarm_pai_debugger()` -- converts PAI's `pdb.set_trace()` into an exception at the real call site | `dendritic.py` |
| 5 | Corrected three docstrings/comments carrying the false premise | `dendritic.py`, `dendritic_cycle1.yaml` |

**Fix 2 closes a gap the investigation missed.** `keep_frozen_batchnorm_eval`
only pins BatchNorm when its affine params have `requires_grad=False`. Simply
removing the freeze would therefore have let base BN running statistics drift
through the whole dendrite phase -- a real 3c violation by a path no optimizer
filter covers. Measured drift without the pin: **3.9e-2 per forward pass**, so
this was not hypothetical. Verified: BN pinned, `requires_grad` still `True`,
output still has `grad_fn`, gradients still flow, drift exactly `0.0`.

Fix 4 verified against a live PAI trap: the traceback now points at
`utils_perforatedai.py:902` (the real call site) instead of a torch frame.
`PYTHONBREAKPOINT=0` would not have worked -- PAI never calls `breakpoint()`.

### Two self-inflicted relaunch failures (both mine, both fixed)

1. `resume recipe mismatch for sparsity/prune_kd` -- editing the train config
   changed the recipe fingerprint, correctly invalidating the 40 kept prune+KD
   epochs. Coarse fingerprinting erring toward retraining is the safe direction;
   accepted the 40-epoch cost rather than weaken the check mid-recovery.
2. `manifest artifact is missing: .../pai.jsonl` -- I filtered the manifest
   *before* moving that file. Same mistake as entry 4. Correct order is: move
   everything first, then filter once. Did so; 62 -> 31 artifacts, two backups
   in scratchpad.

**Status:** FIXED and relaunched 20:30 into the same
`--output-dir outputs/full-run-20260912T063022Z`. 193/193 tests pass. Candidate
`w18` restarts from prune+KD epoch 1 (~40 epochs) before re-entering the
dendrite cycle. Deliberately a clean restart rather than `Load from 'latest'`:
PAI had already recorded `switch_1` with zero scores, and resuming risked the
tracker counting a consumed dendrite try against `max_dendrite_tries: 2`.
Crash debris preserved in scratchpad `crash-debris-20260912T1956Z/`.
Pre-crash run reached val **0.8204** at epoch 67, up from 0.7909 at epoch 1.

**Still unexercised:** `read_pai_architecture_results` and the first true PAI
resume. The dendrite scoring path itself is now the immediate test.

### 5b. Audit of the entry-5 fix (20:45) — core fix confirmed, two defects in the new check

Re-reviewed the fix rather than trusting it. The substantive fix is **correct**,
and I verified it by experiment instead of by reading source, on a real
PAI-wrapped width-18 model driven through a real switch into mode `p`:

| Property | Measured |
| --- | --- |
| base parameter drift over a full dendrite-mode epoch | **0.0** (0 of 31 tensors moved) |
| base tensors receiving nonzero gradient | **17 / 31** (PAI's hook is alive) |
| base parameters in the optimizer, mode `p` | **0** |
| base BatchNorm pinned / dendrite BatchNorm left training | **9 / 8**, correct both pre- and post-switch |
| `pdb.set_trace` trap | raises at the call site; PAI holds no private alias that would bypass the patch |

This also settles the mechanism, which the entry-5 writeup asserted but had not
demonstrated: `_make_optimizer_and_scheduler` builds `AdamW` over
`list(model.parameters())` -- *all* parameters -- and `set_optimizer_instance`
then strips the base out of that optimizer **in place**. The base therefore
keeps `requires_grad=True` (so PAI's neuron-error hook registers) while being
unable to move. Both halves are now measured, not assumed.

Two defects **in the new `base_params_in_optimizer` check itself**:

1. **It read the wrong optimizer.** The call sat in the `phase_changed` block,
   which runs *before* `if restructured:` rebuilds the optimizer. At a switch
   epoch it therefore measured the previous phase's optimizer -- which
   legitimately still holds the whole base -- against the new mode. Measured in
   production order: **1542** live base parameters. Measured after the rebuild:
   **0**. The check was guaranteed to report a violation at exactly the one
   epoch it exists to validate. Moved the measurement below the rebuild.

2. **The `-1` "unmeasurable" sentinel is truthy.** The guard was
   `if mode == "p" and base_live:`, so an unreadable optimizer would log
   `Step 3c violated: -1`. Now `base_live > 0`, with a regression test
   (`test_step_3c_check_separates_unmeasurable_from_violated`) pinning the
   three-way contract: `-1` unmeasured, `0` clean, `>0` violated.

Neither defect could corrupt training -- the check only logs -- but both would
have produced a false alert, and `Step 3c violated` is an alert pattern for the
run monitor. A false alarm there triggers the crash procedure against a healthy
multi-hour run, which is how a diagnostic becomes worse than no diagnostic.

**Consequence for the run in flight:** process 21501 imported the pre-fix module
at 20:30, so it is still running defect 1. It **will** log
`Step 3c violated: <~1542>` at its first dendrite switch and the monitor **will**
raise an `[ALERT]`. That specific line, in the same epoch as
`PAI restructured the network`, with a count equal to the base parameter count,
is this known false positive and **is not grounds to stop the run**. A real
violation looks different: a nonzero count on a *non*-switch epoch, or one that
persists across subsequent dendrite-mode epochs. Not restarting to pick up the
fix -- the defect is log-only and a restart would cost the run.

194/194 tests pass.

---

### 5c. Final audit of the entry-5 changes

The training fix remained sound, but the audit found three residual defects in
its safeguards:

1. `enforce_base_weight_freeze: true` was still accepted even though it is a
   known-invalid PAI configuration. It now fails before checkpoint loading,
   dataset construction, or prune/KD training can consume time.
2. The optimizer audit ran only when a phase changed. A process resumed directly
   into dendrite mode, or an optimizer mutated later in that phase, therefore
   had no per-epoch check. Every epoch now records and validates the optimizer
   that will actually execute it; transition-time validation remains in the
   phase trail for the paired restart boundary.
3. A transition epoch's metric used the post-validation (next) PAI mode while
   its loss and learning-rate fields described the mode that just trained. The
   record now separates `pai_mode` from `next_pai_mode` and includes the live
   base-parameter count for the executed optimizer.

The optimizer reader was also hardened so malformed/foreign parameter groups
remain `-1` (unmeasured), and repeated parameter references are not double
counted. Direct regression coverage now verifies that base BatchNorm statistics
stay fixed while base affine gradients still flow. **196/196 tests pass.**

---

## 6. Headless selector smoke check prompted for a PAI license and raised `EOFError`

**Date:** 2026-09-13
**Status:** RESOLVED AS AN INVOCATION/DIAGNOSTIC DEFECT; THE LIVE PIPELINE NEVER FAILED

### Symptom

A one-off smoke check intended to inspect the proposed `.fc`-only selector was
started from a temporary working directory with the virtual environment's
interpreter directly. During import it printed a PerforatedAI license message,
fell back to an `email:` prompt, and then raised:

```text
EOFError: EOF when reading a line
```

No training epoch or selector code ran. The existing six-step pipeline process
continued normally, so it was neither stopped nor restarted for this event.

### Root cause

This check bypassed both parts of the repository's documented PAI invocation
contract:

1. it used `.venv/bin/python` rather than `uv run --env-file .env ...`; and
2. it changed the working directory from `KWS_Model` to a temporary directory.

The proprietary `perforatedbp` extension validates its license during module
import, before `configure_perforatedai()` handles the conversion selector. With
the configured run environment absent, its fallback is an interactive prompt.
The smoke process was headless, so `input()` immediately encountered EOF.

This was **not** evidence that `.fc` selection is invalid, and it was not a
training, resume, checkpoint, graph, or model defect. In particular, the
failure occurred before any call to `set_modules_to_perforate()`.

### Diagnostic hardening

| Change | File |
| --- | --- |
| Added `import_pai_module()`, which preserves the original `EOFError` as the exception cause but raises an actionable `RuntimeError` explaining the supported root/env-aware invocation | `src/kws/optimize/pai_import.py` |
| Routed every compiled PAI import through that boundary, including the two dendritic startup imports and the downstream candidate reload | `src/kws/optimize/dendritic.py`, `src/kws/pipeline.py` |
| Documented that PAI-dependent commands must run from `KWS_Model` with `uv run --env-file .env ...` | `README.md` |
| Added isolated tests for the success path, the headless-license-prompt path, and preservation of unrelated import failures; tests substitute the import function and never inspect credentials or contact the licensing service | `tests/test_pai_import.py` |

The wrapper deliberately catches only `EOFError`. Import errors, ABI failures,
and actual PAI defects retain their original exception types instead of being
misreported as credential problems. No credential value or `.env` content was
read, logged, or added to an artifact.

### Verification

```text
uv run --env-file .env python -m pytest tests/test_pai_import.py tests/test_dendritic.py \
  tests/test_dendritic_resume_regressions.py tests/test_pipeline.py
```

All 45 focused tests pass. The same env-aware invocation also imports both PAI
compiled modules and `kws.pipeline` successfully.

The supported operational rule for future selector probes is the same one used
by production runs:

```text
cd KWS_Model
uv run --env-file .env python -m <module-or-smoke-check>
```

---

## 7. Legacy/minimal dendritic configs raised `KeyError: 'conversion'`

**Date:** 2026-09-13
**Status:** RESOLVED

### Symptom

The focused resume regression
`test_restructure_resets_optimizer_before_the_only_paired_save` failed before
it reached the optimizer-reset behavior it is designed to exercise:

```text
KeyError: 'conversion'
```

The exception came from `run_cycle()` while estimating the candidate's
deployed parameter count. The regression's deliberately minimal
`perforatedai` mapping contains only `enforce_base_weight_freeze`, matching the
shape accepted before the `.fc`-only selector was added.

### Root cause

The selector change added a required positional conversion argument to the
parameter estimator call and obtained it with `pai_cfg["conversion"]`. That
made the new key mandatory even though the only historical behavior was
`blocks_and_linear`, and other new PAI settings such as the post-integration LR
multiplier already use backward-compatible defaults. The configuration path
inside `configure_perforatedai()` had the same unconditional lookup, so fixing
only the logging call would have moved the same failure later in a real legacy
run.

This was a regression in configuration compatibility. It did not indicate a
training-data, checkpoint, graph, optimizer, or PerforatedAI runtime failure,
and no pipeline was restarted from this failed unit test.

Once that lookup was corrected, the same heavily mocked regression reached the
new restart-audit assignment and exposed `KeyError: 'lr'`. Unlike conversion,
`lr` was already mandatory for every real run through
`_make_optimizer_and_scheduler()`; the test had omitted it only because that
constructor is replaced by a stub. The fixture now supplies `lr: 0.001`, making
its mock configuration faithful to the production contract while leaving
runtime behavior unchanged.

### Fix

Both consumers now interpret an omitted conversion as
`blocks_and_linear`, which is exactly the pre-change behavior:

```python
pai_cfg.get("conversion", "blocks_and_linear")
```

Explicit `conversion: fc_only` remains unchanged and continues to target only
`.fc`. Unsupported explicit values still fail validation. The optimizer-reset
regression retains its intentionally minimal configuration, and its estimator
stub now accepts the optional conversion argument so the test continues to
verify the production call contract as well as the optimizer/save ordering.

### Verification

Run the original failing regression plus the focused dendritic suites:

```text
uv run --env-file .env python -m pytest \
  tests/test_dendritic_resume_regressions.py::test_restructure_resets_optimizer_before_the_only_paired_save \
  tests/test_dendritic.py tests/test_dendritic_resume_regressions.py
```

The original failing regression passes, and the two focused dendritic suites
pass **36/36** tests. Coverage now asserts both `run_cycle()`'s default and the
historical module selector applied by `configure_perforatedai()` when the key
is omitted.

---

## 8. Prune/KD reuse did not prove `best.pt` and `latest.pt` were one pair

**Date:** 2026-09-13
**Status:** RESOLVED DURING INDEPENDENT REVIEW

### Symptom

The phase-local step-3a/3b reuse helper checked the recipe, run ID, completion
boundary, model configuration, pruning recipe, and teacher provenance, but it
validated the best and latest checkpoint files independently. Two individually
plausible files could therefore pass even when `best.pt` was stale and did not
contain the durable best model recorded by `latest.pt`.

No production run raised an exception from this defect. It was found by review
before restarting the `.fc`-only experiment. The completed width-18 checkpoint
pair was also checked directly and contains identical best-state tensors, so
the stricter validation does not prevent the requested step-3c restart.

### Root cause

`train_model()` commits `best.pt` and `latest.pt` with separate atomic writes.
That protects each file from partial bytes, but it cannot make the two-file
update transactional. An interruption between writes, or a copied/stale file,
can leave metadata that looks compatible while the model tensors differ. The
reuse helper's comment said both files attested the same phase, but there was
no cross-file state check enforcing that claim.

### Fix

The reuse boundary now requires both of the following:

1. `best.pt`'s validation accuracy equals `latest.pt`'s durable best metric;
2. every key and tensor in `best.pt["model_state_dict"]` exactly equals the
   corresponding tensor in `latest.pt["best_model_state_dict"]`.

A focused regression constructs a pair with matching metadata and metrics but
different tensors and verifies it is rejected. The real completed width-18
pair passes the same tensor-identity check.

### Verification

```text
uv run --env-file .env python -m pytest tests/test_pai_import.py \
  tests/test_dendritic.py tests/test_dendritic_resume_regressions.py \
  tests/test_pipeline.py
uv run --env-file .env python -m compileall -q src tests
git diff --check
```

---
## 2026-09-13 — stale cache test after enabling fixed augmented views

- **Symptom:** `test_precomputed_features_skip_repeated_deterministic_extraction`
  expected one cached tensor for a two-entry fixture and failed with
  `assert 2 == 1`.
- **Diagnosis:** The test encoded the previous policy that synthesized silence
  was always regenerated. The requested augmented-training cache deliberately
  materializes silence and its augmentation once at startup, so `2` is the
  correct count.
- **Resolution:** Updated the expectation and documented that repeated silence
  accesses reuse the cached augmented feature. No production defect found.
