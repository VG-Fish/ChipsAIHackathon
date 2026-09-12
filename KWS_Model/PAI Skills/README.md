# Dashboard Package

Customer-deliverable package that adds the Dashboard MCP Server and its Claude Code Skills to a project.

## Contents

- `install.sh` / `uninstall.sh` — setup and teardown scripts
- `dashboard-run.sh` — runtime launcher copied into `.perforated_tools/` on install (logs container stderr to `dashboard.log`)
- `skills/` — Claude Code Skills copied into the project on install
- `test.sh` — smoke test for a completed install

The Dashboard MCP Server itself is distributed as a public Docker Hub image, `rorrybrenner/perforated_dashboard_mcp:v0.1.0` (multi-arch), pulled by `install.sh` — it is not bundled in this package.

## Install

From the project root (the Codebase you want to add the Dashboard to):

```sh
/path/to/package/install.sh
```

This will:

1. Pull `rorrybrenner/perforated_dashboard_mcp:v0.1.0` from Docker Hub and smoke-test that it runs on this machine.
2. Install a runtime launcher at `.perforated_tools/dashboard-run.sh` that logs the container's stderr to `.perforated_tools/dashboard.log`.
3. Register a `dashboard` entry in the project's `.mcp.json`, pointing at the launcher and mounting the Codebase read-only plus a read-write `.perforated_tools/` directory.
4. Copy every skill in `skills/` into `.claude/skills/` in the project.

By default the MCP Server listens on port `3002`. Use `--port` to change it. Use `--artifact-root PATH` (or `PAI_ARTIFACT_ROOT=PATH`) to route runtime logs and visualizer/training exports into a selected run directory:

```sh
/path/to/package/install.sh --port 4000
/path/to/package/install.sh --artifact-root "$PWD/outputs/pai"
```

Restart Claude Code (or start a new session) after installing so it picks up the updated `.mcp.json`.

## Usage

Once installed, invoke any of the skills from Claude Code:

- `/dashboard` — verify the MCP Server is reachable and open the Dashboard in a browser.
- `/train-my-model` — configure and launch a training run with live dashboard monitoring.
- `/compare-models` — export two or more models and compare their architectures side by side.
- `visualize-model` — export a PyTorch model to ONNX and view its architecture in the Visualizer (triggered by asking to visualize a model, or via `ml_export_model`).
- `perforatedai` — set up or debug a PerforatedAI dendrite integration in a PyTorch model.
- `perforatedai-analyze` — review results from a completed PerforatedAI training run and get optimization recommendations.
- `perforatedai-distributed` — multi-GPU (DataParallel/DDP) setup for PerforatedAI; invoked automatically by the `perforatedai` skill when needed.

## Uninstall

```sh
/path/to/package/uninstall.sh --artifact-root "$PWD/outputs/pai"
```

This removes the `dashboard` entry from `.mcp.json`, deletes the installed skills from `.claude/skills/`, removes only the runtime launcher and log from the selected artifact root (preserving other user artifacts), and removes the `rorrybrenner/perforated_dashboard_mcp:v0.1.0` Docker image.

## MCP Server

The MCP Server is a Python (fastMCP) process running inside the `dashboard-mcp` Docker container. It talks to Claude Code over stdio MCP transport and serves the compiled React build (the Dashboard) as static files over HTTP on the configured port (default `3002`). It mounts the Codebase read-only at `/workspace` and `.perforated_tools/` read-write at `/perforated_tools` — the only channel the container can write output back through.

Tools are grouped into two plugins, each namespacing its tool names with a prefix:

- `dashboard` plugin — `ping`, `dashboard_visualize_model`, `dashboard_show_variant`, `dashboard_remove_variant`, `dashboard_clear_variants`, `dashboard_open_training`, `dashboard_show_training_chart`, `dashboard_hide_training_chart`
- `ml` plugin — `ml_export_model`, `ml_configure_runner`

## Dashboard views

The Dashboard is a single-page app with three routes:

- **Visualizer** (`/visualize/<ModelClassName>`) — a React Flow DAG of one model's forward pass, built from a Model Artifact (`ml_export_model` output). Click a node to open a sidebar with its type, path, output shape, and `__init__` params. Perforated layers are highlighted. Opened via `dashboard_visualize_model` or the `visualize-model` skill, one model per browser tab.
- **Comparison View** (`/compare`) — multiple models side by side, each in its own React Flow panel. Panel membership (which models are shown) is controlled by Claude via `dashboard_show_variant` / `dashboard_remove_variant` / `dashboard_clear_variants`, pushed to all connected browser tabs over the `/events` SSE channel. Use the `compare-models` skill to drive this.
- **Training View** (`/training`) — live view of the in-progress or most recent Training Run. Opened ahead of time by `dashboard_open_training` (called from the `train-my-model` skill) so it's ready before the training script starts posting events. Shows:
  - An **Epoch Progress Bar** (current epoch out of `total_epochs`, from the run's `run_start` event).
  - The permanent **score chart** (validation score, train score, with Switch boundaries marked as vertical phase lines).
  - Optional **Training Charts**, hidden by default, shown/hidden via `dashboard_show_training_chart` / `dashboard_hide_training_chart` by chart id. Visibility resets to empty at the start of every new run.
  - A **Training Log** scrollback of `log`-type events for the current run (not persisted — cleared on reload). `error`-level log events also trigger a Toast notification.

## How training events reach the dashboard

PerforatedAI (running in the customer's own environment, not inside the container) POSTs Training Events as JSON to an `events_url` the MCP Server exposes at `/training-events`. The `train-my-model` skill wires this up via `ml_configure_runner`, which writes `events_url` into the Perforation Config (`{save_name}/{save_name}_config.json`) so PerforatedAI can read it at runtime. The server updates its in-memory run state from each event and fans it out to all connected Dashboard tabs over the `/training-events` SSE channel.

Five event types exist:

| type | when | carries |
|---|---|---|
| `run_start` | training begins | model class name, timestamp, `total_epochs` — resets run state |
| `epoch` | after each epoch | validation score, train score, learning rate, normal epoch time, PAI epoch time |
| `switch` | a PerforatedAI learning-phase boundary | switch number, param count, epoch number |
| `log` | free-text message | `message`, `level` (`info` \| `warning` \| `error`) |
| `run_end` | training finishes | no-op today |

Only `run_start`, `epoch`, and `switch` events are persisted with the run; `log` events are ephemeral (Training Log scrollback only).

### Adding a graph to the Training View

The score chart is permanent; everything else is an optional Training Chart, toggled by Claude — not shown by default. To add a new one:

1. **Server**: add the new id to the `TrainingChartId` literal in `mcp_server/tools/dashboard/__init__.py`. This is the id `dashboard_show_training_chart` / `dashboard_hide_training_chart` validate against.
2. **Client**: add a chart component (e.g. modeled on `client/src/components/LearningRateChart/`) and register it — with a label and the component — in the `OPTIONAL_CHARTS` map in `client/src/components/TrainingView/TrainingView.tsx`.
3. The two lists are kept in sync by convention, not a shared source of truth (see `docs/adr/0012-server-side-training-chart-id-registry.md`) — a drift fails safely: the tool call is rejected, or the chart id is accepted but nothing renders.
4. Call `dashboard_show_training_chart(chart_id=...)` for a run to make the new chart visible; visibility resets to empty on the next `run_start` (see `docs/adr/0013-training-chart-visibility-scoped-per-run.md`), so re-issue the call per run.

If a training script has metrics PerforatedAI itself doesn't know about (e.g. a loop-bound `total_epochs`), it can POST additional events directly to `GPA.pc.events_url`, e.g.:

```python
import requests

requests.post(GPA.pc.events_url, json={"type": "run_config", "total_epochs": epochs})
```

See `docs/adr/0014-run-config-event-for-training-script-known-values.md` for why this is a separate event rather than part of `run_start`.

No real PerforatedAI run needed to develop against the Training View — `scripts/simulate_training.py` posts a realistic sequence of Training Events to a running MCP Server.
