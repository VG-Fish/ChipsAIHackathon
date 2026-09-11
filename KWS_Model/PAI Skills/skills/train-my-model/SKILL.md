---
name: train-my-model
description: Guide the Operator through launching a training run with live dashboard monitoring. Use when the Operator wants to start training, asks to train a model, or invokes /train-my-model.
---

# Train My Model

## Workflow

1. **Collect inputs** — ask the Operator for the following if not already known:
   - `save_name` — the PAI save name (e.g. `PAI`)
   - `training_script` — path to the training script relative to the Codebase root (e.g. `train.py`)
   - `training_args` — any arguments to pass to the script (e.g. `--epochs 50`); default is none
   - `model_file` — path to the model file (e.g. `model.py`)
   - `model_class` — the model class name (e.g. `MyNet`)

2. **Configure** — call `ml_configure_runner` with the collected inputs:
   ```
   ml_configure_runner(
     save_name=...,
     training_script=...,
     training_args=[...],
     model_file=...,
     model_class=...,
   )
   ```
   This writes `events_url` into the Perforation Config (`{save_name}/{save_name}_config.json`) — the URL PerforatedAI posts Training Events to.

3. **Enable dashboard events in the training script (required)** — the training script must call `GPA.pc.set_dashboard_events_enabled(True)` for PerforatedAI to actually post Training Events to `events_url`; without it the Training View stays empty even though the runner is configured. Check whether the script's PAI configuration block already has this call (it's typically set alongside `GPA.pc.set_testing_dendrite_capacity(...)`). If missing, add it there:
   ```python
   GPA.pc.set_dashboard_events_enabled(True)
   ```
   If the Operator would rather add it themselves, give them the snippet instead of editing their script.

4. **Wire up `total_epochs` (optional but recommended)** — PerforatedAI's own `run_start` post has no visibility into the training script's loop bounds, so the Epoch Progress Bar has nothing to show a total against unless the script tells it directly. If the training script has a known epoch count (e.g. an `epochs` variable or CLI arg), offer to add a couple of lines near the start of its training loop that POST a `run_config` event to the same `events_url` PerforatedAI uses:
   ```python
   import requests

   requests.post(GPA.pc.events_url, json={"type": "run_config", "total_epochs": epochs})
   ```
   This is a no-op server-side if posted before `run_start` arrives or if there's no active run, so it's safe to place early in the script. If the Operator would rather add it themselves, give them the snippet instead of editing their script. See `docs/adr/0014-run-config-event-for-training-script-known-values.md` for why this isn't just part of `run_start`.

5. **Open the Training View** — call `dashboard_open_training` so the view is ready before events arrive:
   ```
   dashboard_open_training()
   ```

6. **Suggest the command** — output the exact command for the Operator to run in their terminal:
   ```
   python <training_script> <training_args>
   ```
   Explain that training logs will appear in the terminal and the Training View will update live as events arrive.

## Example (all inputs known)

```
ml_configure_runner(
  save_name="PAI",
  training_script="train.py",
  training_args=["--epochs", "50"],
  model_file="model.py",
  model_class="MyNet",
)
```

If `train.py` has a known epoch count, add the `run_config` snippet near the top of its training loop (or offer it to the Operator), then:

```
dashboard_open_training()
```

Then tell the Operator:

> Run this in your terminal to start training:
> ```
> python train.py --epochs 50
> ```
> Training logs will be visible in your terminal. The Training View will update live as events arrive.
