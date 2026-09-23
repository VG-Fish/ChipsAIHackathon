# ReRAM simulation implementation journal

## 2026-09-20 — start

- Task: implement the standalone NeuroSim V2.1 ReRAM estimator described in
  `ReRAM Simulation Implementation.md`.
- Repository: `/Users/vishy/Desktop/ChipsAIHackathon`, branch `KWS_Model`.
- User-owned work preserved: the guide directory was already untracked before
  implementation began.
- Compatibility audit completed before coding:
  - model registry reconstructs DS-CNN and SparkNet from checkpoint metadata;
  - `kws.evaluate` provides strict checkpoint loading and evaluation setup;
  - the existing profiler counts grouped/depthwise convolution MACs;
  - clean PerforatedAI checkpoints have an existing safetensors rebuild path;
  - no guide/codebase conflict was found.
- Scope decision: keep this as a standalone subsystem. Do not modify
  `kws.pipeline` until the subsystem has independent tests and a working
  external-backend seam.
- Test-first status: `tests/test_neurosim_config.py` was added first and
  intentionally failed with `ModuleNotFoundError: No module named
  'kws.hardware'`. Production implementation starts after that red test.
- Environment note: repository tests are run from `KWS_Model` with
  `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run ...` because the default UV
  cache is not writable in this workspace.

## 2026-09-20 — standalone subsystem slices

- Configuration implemented in `src/kws/hardware/neurosim/config.py` with
  frozen dataclasses, YAML loading, reference defaults, and fail-closed
  validation for backend, inference-only mode, precision, arrays, technology,
  memory, trace, graph, and runtime settings.
- Model loading implemented in `model_loader.py`. Normal checkpoints reuse
  `kws.evaluate.load_model_from_checkpoint`; clean PAI safetensors require
  explicit model config, input shape, and class count and reuse
  `kws.optimize.grow_clean_rebuild`.
- Runtime graph capture and IR implemented in `graph_capture.py` and `ir.py`.
  Captured executions use runtime tensor shapes; shared modules get separate
  execution names while retaining one weight key; unsupported learned
  operators and dilation fail closed. Grouped/depthwise Conv2d conversion is
  block-diagonal and preserves disconnected channels.
- Quantization/traces implemented in `quantization.py` and `traces.py`:
  deterministic per-tensor symmetric integer quantization, MSB-first signed
  two's-complement bit traces, exact bit activity, activation calibration over
  selected real samples, and trace provenance files.
- Network export/report seams implemented in `network_csv.py`, `parser.py`,
  and `report.py`. Parser normalization is centralized and uses SI units;
  reports retain raw fields, validity labels, JSON/CSV/Markdown outputs, and
  explicit inference-energy/TOPS conventions.
- `backend.py` exports model IR, network CSV, quantized weights, activation
  traces, a manifest, and a MAC cross-check against `kws.utils.profile`.
  It also runs one explicit subprocess per sample and aggregates parsed
  metrics.
- The graph capture reuses the existing profiler's `_hookable_branch_calls`
  adapter so clean PAI branches invoked via direct `.forward()` are not
  omitted. PAI residual skip-edge MACs are represented as non-CIM operations
  and included in the project-profiler MAC cross-check.
- External source/build boundary implemented in `source.py`, `build.py`, and
  `runner.py`. Source provenance is recorded; builds copy into a cache and
  preserve compiler/simulator stdout/stderr. The original NeuroSim checkout
  is never edited.
- CLI implemented with exactly `inspect`, `export`, `run`, and `validate`;
  `inspect` can use a real dataset sample when `--data-config` is supplied,
  while final `export` always requires a configured dataset split.
- Current NeuroSim-specific executable flags are intentionally caller-supplied
  through `--simulator-arg` and `{network}`, `{weights}`, `{config}`,
  `{sample_dir}`, `{sample_id}`, `{export_dir}` placeholders. The external
  checkout and its exact executable interface are not present in this
  workspace, so no real upstream compilation or simulator run has been
  claimed. This is an external-environment limitation, not a codebase
  conflict.

## Verification so far

- `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run python -m pytest tests/test_neurosim_*.py -q`
  → 49 passed.
- The 49-test suite also covers root-level weighted modules, arbitrary custom
  learned parameters, PAI direct branch calls, PAI skip-edge MACs, timeout
  diagnostics, report warnings, and multi-sample FPS statistics.
- Regression excluding the environment-gated PAI integration file:
  `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run python -m pytest -q --ignore=tests/test_sparknet_grow_dendrites.py`
  → 613 passed, 1 skipped, 6 warnings.
- Default config validation:
  `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run python -m kws.hardware.neurosim validate --hardware-config configs/hardware/neurosim/reram_v21.yaml --skip-source-check`
  → configuration valid.
- Project-wide regression:
  `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run python -m pytest -q`
  → 778 passed, 2 skipped, 17 failed. The 17 failures are existing
  `tests/test_sparknet_grow_dendrites.py` PerforatedAI integration cases; all
  stop before the requested code at `perforatedbp.check_license` with
  `EOFError: EOF when reading a line` because this environment has no
  interactive PerforatedBP license token. No failing traceback imports or
  executes `kws.hardware.neurosim`.

## Handoff instructions

1. From `KWS_Model/`, validate only the checked-in reference config with:
   `UV_CACHE_DIR=/private/tmp/codex-uv-cache uv run python -m kws.hardware.neurosim validate --hardware-config configs/hardware/neurosim/reram_v21.yaml --skip-source-check`.
2. For a normal project checkpoint, use `inspect` with `--checkpoint` and
   optionally `--model-config`/`--data-config`; use `export` with all of
   `--checkpoint`, `--data-config`, `--hardware-config`, and `--output-dir`.
   Exported trace samples are real dataset items and are stored under
   `traces/sample_NNN/`.
3. `run` requires `NEUROSIM_V21_ROOT`, a valid source checkout, `make`, and
   an explicit `--executable`. Its repeated `--simulator-arg` values support
   placeholders `{network}`, `{weights}`, `{config}`, `{sample_dir}`,
   `{sample_id}`, and `{export_dir}`. This is deliberate: the exact
   executable argument contract was not available in the workspace and must
   be confirmed against the supplied V2.1 source before adding a revision-
   specific adapter.
4. If adding that adapter, keep source changes confined to the copied cache
   tree returned by `prepare_isolated_build`; add a golden parser test from
   the actual checkout output before changing `parser.py`.
5. No files under `kws.pipeline` were changed. Do not integrate the backend
   into the training pipeline until the external V2.1 source, patch behavior,
   and executable output fields are verified.

## Handoff protocol

Update this file after each substantive implementation slice. Record files,
tests/commands, known limitations, and any conflict with the guide. Never
claim an external NeuroSim run succeeded unless the upstream source and its
toolchain are actually present and the subprocess result is known.
