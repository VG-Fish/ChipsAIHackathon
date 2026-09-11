---
name: visualize-model
description: Export a PyTorch model to ONNX and open it in the Netron Visualizer. Use when the Operator wants to visualize a model architecture, calls ml_export_model, or asks how to set up a model file or dataloader for visualization.
---

# Visualize Model

## Workflow

When invoked, always run this discovery step before calling any tool:

1. **Scan** the Codebase for `nn.Module` subclasses:
   ```
   grep -rn "class .*nn\.Module" --include="*.py" .
   ```
2. **If more than one class is found**, present the list and ask the Operator which to visualize. Include the file path next to each name so they can distinguish identically-named classes across files.
3. **Identify the dataloader** — ask the Operator for `module_path:variable_name` if not already known. If they don't know it, grep for `DataLoader\|torch\.utils\.data` to surface candidates.
4. **Call the tools** once you have both:
   ```
   ml_export_model(model_file=..., model_class=..., dataloader=...)
   dashboard_visualize_model("<ModelClass>")
   ```
5. **If the Operator wants to compare multiple models**, repeat steps 2–4 for each — `dashboard_visualize_model` opens each in its own browser tab.

## Quick start (model already known)

```
ml_export_model(
    model_file="path/to/model.py",    # relative to Codebase root
    model_class="MyNet",
    dataloader="path/to/loader.py:train_loader",
    save_name="PAI",                  # default; omit if using PAI
)
dashboard_visualize_model("MyNet")
```

## Model file requirements

The class must be importable with no constructor arguments:

```python
# model.py
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)
```

- No `if __name__ == "__main__"` required — the tool imports the file directly.
- No weights needed — the export captures architecture only (Model Artifact).

## Dataloader reference format

The `dataloader` parameter is `"module_path:variable_name"` where `module_path` is a `.py` path relative to the Codebase root (with the `.py` extension).

```python
# loaders/train.py
import torch
train_loader = [(torch.zeros(1, 3, 224, 224), torch.tensor([0]))]
```

Called as: `dataloader="loaders/train.py:train_loader"`

The tool calls `next(iter(dataloader))` and takes the **first element** as the sample input. If batches are plain tensors use them directly; if they are `(inputs, labels)` tuples the tool unpacks index `[0]`.

**The dataloader variable must be defined at module level** — not inside a function or `if __name__` guard.

## Outputs

Both files are written to `.perforated_tools/exports/` in the Codebase:

| File | Contents |
|---|---|
| `<ModelClass>.pt` | Model Artifact — loadable by Netron |
| `<ModelClass>.meta.json` | Alteration Manifest — perforated layer list + ONNX node mapping |

If `{save_name}/{save_name}_config.json` does not exist, an empty Alteration Manifest is written (no layers highlighted).

## Common errors

| Error message | Fix |
|---|---|
| `Cannot import model file` | Check path is relative to Codebase root and the file exists |
| `Cannot import dataloader` | Ensure the variable is at module level and the path+name match exactly |
| `Class not found` | `model_class` must match the class name exactly (case-sensitive) |
| Blank Visualizer page | Check browser console — ONNX fetch likely 404; confirm export succeeded |
