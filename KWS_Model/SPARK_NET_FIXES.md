# SparkNet / Phase B Audit Findings

Audit date: 2026-09-13

Scope: the uncommitted SparkNet/Phase B changes, checked against `PLAN.md` and
`README.md`. This file records the audit findings and their resolutions.

## Findings

### 1. Resolved — the distillation CLI did not validate the data feature shape

`kws.optimize.distill.distill()` now probes the selected data feature recipe
with a zero waveform and checks its `(n_mels, frames)` shape against the teacher
checkpoint before building the student or datasets. An incompatible recipe is
rejected immediately with a clear `ValueError`.

Previously, selecting the MFCC-32 config, or any incompatible feature config,
for a KD run failed later during a model forward with a low-level channel/shape
error. This was especially easy to trigger around the new SparkNet configs
because `PLAN.md` adds both 40-bin log-mel and 32-bin MFCC configurations.

### 2. Resolved — the README documented a test command that did not run here

`KWS_Model/README.md` previously said:

```bash
uv run pytest tests/
```

From `KWS_Model`, this fails with `Failed to spawn: pytest` because the local
`.venv/bin/pytest` has a stale shebang. The README now uses the robust module
invocation already documented in `PLAN.md`:

```bash
uv run python -m pytest tests/
```

### 3. Resolved — README caching guidance contradicted the new light recipe

`KWS_Model/README.md` previously said augmented training configs set
`cache_train_features: true` and therefore reuse one augmented view for the
whole run. The new Phase B configs are augmented but explicitly set
`cache_train_features: false` (`configs/train/light.yaml` and
`configs/train/light_kd.yaml`) so they resample augmentation every epoch.

The README now describes caching conditionally and calls out that the Phase B
light recipes resample augmentation every epoch.

### 4. Resolved — the README contained a duplicated fragment

`KWS_Model/README.md` read:

```text
Replaying the
An audit-only replay of the historical runs ...
```

The incomplete `Replaying the` fragment has been removed.

## Intentional limitations, not counted as bugs

The plan explicitly leaves pruning, PAI, QAT, and the rest of the pipeline
DS-CNN-only until Phase C. I did not report those as regressions because the
current Phase B implementation documents that boundary and the requested scope
does not claim full SparkNet framework integration yet.

## Verification

- `UV_CACHE_DIR=/private/tmp/kws-uv-cache uv run python -m pytest tests/` — **242 passed**.
- `UV_CACHE_DIR=/private/tmp/kws-uv-cache uv run python -m compileall -q src tests` — passed.
- `git diff --check` — passed.
- The new distillation regression verifies that incompatible feature shapes are
  rejected before the student is built.
