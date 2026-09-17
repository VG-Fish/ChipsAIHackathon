# Torn PAI/KWS checkpoint pair: a hard kill can make a candidate unresumable

**Date:** 2026-09-16
**Status:** FIXED — see [The fix](#the-fix). New candidates cannot tear, and a
candidate torn by the old code now repairs itself on its next resume, so no
operator decision and no retraining is needed.
**First observed:** `outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed2`,
candidate `sparknet_c10_multilayer`, killed at epoch 656 by the macOS
low-memory killer on 2026-09-16 13:34:50.

## Symptom

A dendritic candidate that was mid-training when its process died refuses to
resume, even though every file it needs is present and readable:

```text
ValueError: PAI native latest state does not match sidecar
  .../seed2/models/checkpoints/sparsity/sparknet_c10_multilayer/pai/latest.pt;
  refusing to resume an unpaired candidate
```

The run then exits within ~12 s of launch, so under a supervisor with a seed
pool the slot is silently handed to the next seed and the failed seed is
skipped for the rest of the arm.

## What the resume contract is

`_save_pai_restart_pair` (`src/kws/optimize/dendritic.py:739`) commits *two*
files per epoch and binds them together:

1. `UPA.save_system(model, run_dir, "latest")` writes PAI's native
   `<candidate>/latest.pt` (safetensors).
2. The KWS sidecar at
   `models/checkpoints/sparsity/<candidate>/pai/latest.pt` is written
   atomically with `native_pai_latest_sha256 = sha256(latest.pt)` plus the
   optimizer, scheduler, RNG, DataLoader generator and phase-trail state.

On resume, `_load_pai_sidecar` (`dendritic.py:435`) re-hashes the native file
and refuses if it differs. The native blob is safetensors and carries no
`run_id` of its own, so that digest is the *only* thing binding the two halves
to the same epoch — the check is correct and should stay.

## Root cause: two writers share one path

`latest.pt` is written twice per epoch, by two different owners:

| step | line | writer | file |
|---|---|---|---|
| 2 | `dendritic.py:2386` | `GPA.pai_tracker.add_validation_score(...)`, inside `_pai_cwd(run_dir)` | PAI's own `latest.pt` (+ `latest_pai.pt`, CSVs, PNGs) |
| 4 | `dendritic.py:2517` | `_save_pai_restart_pair` | the same `latest.pt`, then the sidecar attesting it |

Step 2 runs right after validation; step 4 runs at the end of the same epoch's
bookkeeping. **Between them the on-disk native file belongs to epoch N while
the newest sidecar still attests epoch N−1.** A `SIGKILL` anywhere in that
window leaves a pair that can never be reassembled: the attested bytes are
gone, overwritten by PAI's own write.

Neither file is corrupt. In the observed case the native blob loads fine —
127,001 bytes, 101 tensors via `safetensors.torch.load_file`. It is simply one
epoch ahead of the state the sidecar signed.

### Evidence from seed2

```text
13:34:42.369  log: epoch 655 mode=n ... val_acc=0.9021 params=4042
13:34:42      candidates/sparknet_c10_multilayer/latest.pt      (step 4, epoch 655)
13:34:42      checkpoints/.../pai/latest.pt                     (sidecar, completed_epoch: 655)
13:34:49      candidates/sparknet_c10_multilayer/latest.pt      (step 2, epoch 656)
13:34:50      candidates/sparknet_c10_multilayer/latest_pai.pt  (step 2, epoch 656)
13:34:50      <process killed — step 4 of epoch 656 never ran>
```

Sidecar attests `94531a19f47135ed…`; the file now hashes to `6827b30f03e5eb33…`.

### How wide is the window

Epochs on this run took 7–9 s wall-clock (three seeds concurrent, C10). PAI's
step-2 write landed ~1–2 s before the epoch would have ended, so roughly
**15–25 % of run time is spent inside the vulnerable window**. Four of the five
seeds killed at 13:34:50 survived; seed2 lost the coin flip. This is not a rare
race — expect it whenever runs are killed by an OOM killer or `kill -9` rather
than shut down cleanly.

## Detection gap

Checking that a sidecar's `completed_epoch` matches the seed's last logged
epoch does **not** prove the candidate is resumable — seed2 passed that check
while being unresumable. The real test is the digest, and the run monitor now
has it:

```bash
.venv/bin/python scripts/check_pai_restart_pairs.py \
    outputs/sparknet-c16-dendritic-prune-no-kd-fc-only-d3/seed*
```

It exits 1 when any pair is confirmed torn, so it works directly as a cron
predicate. An unpaired candidate whose sidecar can still rebuild what it
attests is reported as `unpaired on disk, rebuildable from its sidecar` and is
**not** an alert — it resumes without help. A first unpaired reading is confirmed by a second one taken
`--confirm-after` seconds later (default 30, which must exceed one epoch of the
run being watched): a live writer advances `completed_epoch`, a stalled one
does not. Without that second reading the check is useless against a *legacy*
candidate — one written before the fix, still attesting PAI's own `latest.pt`
— because such a candidate genuinely reads as torn for the 1–2 s of every
epoch it spends inside the window. Sampling seed1's live `sparknet_c8_multilayer`
three times over four seconds returned `paired`, `TORN`, `paired`. Candidates
written by the fixed code have no window and confirm on the first reading.

## Recovery for an already-torn candidate

**A torn pair is not data loss, and nothing has to be re-run.** The sidecar
already carries a second copy of the bytes it attests: `_save_pai_restart_pair`
hands PAI a model to serialize *and* stores that same `model.state_dict()` in
the sidecar, in one commit. Re-serializing the sidecar's own model state
therefore reproduces the attested native file exactly — verified against all
eight real candidates in this sweep, where the rebuild matched the recorded
`native_pai_latest_sha256` byte for byte, including seed2's torn one.

`_load_pai_sidecar` does this automatically (`_rebuild_attested_native_state`,
`dendritic.py:272`). When the attested file is missing or its digest does not
match, the sidecar's model state is re-serialized to a scratch file, and it is
published **only if it hashes to the digest the sidecar attests**. So the guard
is never weakened: either the restored bytes are provably the ones committed
alongside that optimizer, scheduler, RNG and loader state, or the resume is
refused exactly as before. The rebuild is published under the KWS-owned name
even when the sidecar attested PAI's own `latest.pt`, so healing a legacy
candidate also migrates it off the shared path, and PAI's file is never written
through.

The two recoveries this document originally proposed are both obsolete:
re-running the width throws away real work, and re-attesting the digest pairs
epoch N+1 weights with epoch N optimizer state — the very thing the guard
exists to prevent. Neither was applied.

A tear that even the sidecar cannot heal — a truncated sidecar, or a PAI build
with `using_safe_tensors` off — still raises, and the error still names the
newest `cycle_checkpoints/` snapshot.

## The fix

The two writers no longer share a path. `_save_pai_restart_pair` still lets PAI
write its own `latest.pt` through `save_system` — PAI's internal bookkeeping is
unchanged — and then publishes a KWS-owned copy at
`<candidate>/kws_native_latest.pt` with `atomic_copy_file`
(`src/kws/utils/checkpointing.py`), which stages the bytes beside the
destination and renames them into place. The sidecar attests *that* copy and
points `native_pai_latest` at it. Both halves of the pair are now written back
to back by one owner, the rename publishes the copy only once it is complete,
and the last complete pair therefore survives a kill at any instant.

On resume, `_load_paired_native_state` (`dendritic.py:553`) takes the PAI
`save_name` from the sidecar rather than a constant, so a candidate always
reloads the exact bytes its digest signed. `_attested_native_path`
(`dendritic.py:240`) resolves that reference and still refuses anything outside
the candidate directory; the digest guard itself is unchanged.

**Sidecars written before the fix stay resumable.** They attest
`<candidate>/latest.pt`, which is accepted as a legacy name and reloaded under
that name. Those candidates keep the old tear risk until their next paired save
— the first epoch after a relaunch — at which point they move to the KWS-owned
copy on their own. Nothing needs to be migrated by hand, and no format version
was bumped, so no in-flight run's `recipe_fingerprint` changes.

A process that was already running when the fix landed keeps the code it
imported at launch, so seeds live at 2026-09-16 15:23 can still *write* a torn
pair. That no longer matters: whatever they leave behind is rebuilt from the
sidecar by the next process to resume it.

### Alongside it

- A digest mismatch now names the newest `cycle_checkpoints/` snapshot, or says
  plainly that none exists. The message does **not** offer it as a
  `--resume-from` argument: those snapshots carry model state only — no
  optimizer, scheduler, RNG or loader state — so they are not sidecars and
  `_load_pai_sidecar` would reject one. Naming it gives the operator the
  concrete artifact and an honest description of what it is.
- `pai_restart_pair_status` / `pai_restart_pair_statuses` / `is_confirmed_torn`
  (`dendritic.py:356`, `:422`, `:407`) expose the digest check to monitoring
  without requiring the cycle recipe or run ID, and never raise. The cron
  itself lives outside this repo and has to be pointed at
  `scripts/check_pai_restart_pairs.py`.

Regression coverage is in `tests/test_dendritic_resume_regressions.py`
(`test_attested_pair_survives_pai_overwriting_its_own_latest_mid_epoch` is the
reproduction of this bug) and `tests/test_training_resume.py`.
