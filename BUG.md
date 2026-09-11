# BUG: `torch_shm_manager` prevents multiprocessing DataLoader startup on macOS

## Status

Resolved on branch `KWS_Model_torch_shm_manager`.

## Environment

- Hardware: Apple M3 Pro MacBook, 36 GB RAM
- OS: macOS 26.5.2, arm64
- Python: 3.13.11
- PyTorch: 2.14.0
- MPS: built and available when run outside the terminal sandbox
- Python multiprocessing start method: `spawn`
- PyTorch sharing strategy: `file_system` (the only available strategy in this environment)
- Dataset: Speech Commands v2, extracted under `data/raw/speech_commands_v0.02`

## Failure

The teacher stage was configured with four persistent DataLoader workers and a
prefetch factor of two. The first launch failed during DataLoader worker startup,
before the first training batch, with:

```text
RuntimeError: no response from torch_shm_manager
```

The traceback went through PyTorch CPU tensor filename sharing:

```text
DataLoader worker startup
  -> torch.storage._share_filename_cpu_
  -> THManagedMapAllocatorInit
  -> no response from torch_shm_manager
```

The failure reproduced in an unsandboxed process, so it was not caused only by the
editor's terminal sandbox or by MPS device detection.

## Root cause and limitation

On this macOS/PyTorch combination, multiprocessing DataLoader workers use PyTorch's
file-backed CPU tensor sharing path. The associated `torch_shm_manager` process did
not respond during worker initialization. The available sharing strategy could not
be changed to a working alternative in this environment.

This report does not claim that every macOS or PyTorch version has the same failure.
The workaround is intentionally platform-specific and conservative: it avoids the
unreliable multiprocessing path on Darwin while retaining configured worker behavior
on Linux and Windows.

## Fix

`src/kws/data/loader.py` now:

1. Validates the configured worker and prefetch values as before.
2. On macOS (`sys.platform == "darwin"`), logs a warning and forces
   `num_workers=0` when multiprocessing was requested.
3. Omits `persistent_workers` and `prefetch_factor` from the resulting DataLoader
   when it is single-process, as required by PyTorch.
4. Leaves the configured multiprocessing behavior unchanged on non-macOS systems.

The teacher log confirms the fallback was applied:

```text
macOS DataLoader workers are disabled because this PyTorch build cannot start torch_shm_manager reliably; using num_workers=0
DataLoader workers=0 persistent=False prefetch_factor=None
```

## Verification

Focused loader tests passed:

```text
.venv/bin/pytest -q tests/test_loader.py
4 passed
```

A minimal unsandboxed DataLoader run also successfully constructed and iterated a
single-process loader. The resumed teacher training then completed epoch 1/40 and
saved a readable checkpoint:

```text
ds_cnn_l epoch 1/40 train_loss=2.1788 val_loss=1.4593 val_acc=0.6750
Saved new best checkpoint (val_acc=0.6750) -> models/checkpoints/ds_cnn_l_12class.pt
```

## Performance tradeoff

`num_workers=0` performs data loading and preprocessing in the training process. It
removes worker startup/sharing failures and is reliable for this Mac, but it can
reduce input-pipeline throughput compared with four persistent workers on a system
where multiprocessing works. The tradeoff is acceptable here because the previous
configuration could not start training at all. A future PyTorch/macOS upgrade can be
retested before restoring multiprocessing on Darwin.

---

# BUG: Pipeline assumed an unavailable optional student warm-start checkpoint

## Status

Resolved on branch `KWS_Model_missing_student_warm_start`.

## Failure

After the teacher completed successfully, the downstream launch:

```text
python -m kws.pipeline --stages student,sparsity,cluster,quantize,benchmark
```

failed before student training began with:

```text
FileNotFoundError: [Errno 2] No such file or directory:
'models/checkpoints/ds_cnn_xs_12class.pt'
```

The missing file was configured as `student.warm_start_checkpoint` in
`configs/train/pipeline.yaml`.

## Root cause

The distillation API accepts a warm-start checkpoint as optional, and the command-line
help explicitly says that omitting it trains the student from scratch. However,
`stage_student` passed the configured path to `distillation_fingerprint` without first
checking that the artifact existed. Fingerprinting calls `file_sha256`, so a missing
optional artifact raised `FileNotFoundError`. Passing the same missing path to the
training function would also have failed when loading the initial model.

The repository contains the pipeline configuration and code, but not the independent
XS checkpoint referenced by the warm-start setting. Therefore this was an artifact
availability/configuration mismatch, not a teacher-checkpoint or DataLoader failure.

## Fix

`src/kws/pipeline.py` now resolves the configured warm-start path before fingerprinting
or training:

- Existing files are retained and used as warm starts.
- A missing configured warm start emits a warning and is normalized to `None`.
- The student then trains from scratch, which is already a supported distillation path.
- The effective `None` value is included in the recipe fingerprint, so future reuse
  validation remains deterministic.

## Verification

Regression tests for the pipeline and distillation code passed:

```text
.venv/bin/pytest -q tests/test_pipeline.py tests/test_distill.py
13 passed
```

Diagnostics for `src/kws/pipeline.py` and `tests/test_pipeline.py` are clean.
