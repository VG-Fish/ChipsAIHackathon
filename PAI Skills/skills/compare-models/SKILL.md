---
name: compare-models
description: Export two or more PyTorch models and open them side-by-side in the Comparison View. Use when the Operator wants to compare model architectures, asks to diff two models, or invokes /compare-models.
---

# Compare Models

## Workflow

1. **Identify the models** — if not specified, scan the Codebase:
   ```
   grep -rn "class .*nn\.Module" --include="*.py" .
   ```
   Present the list and ask the Operator which models to compare.

2. **Identify a dataloader for each model** — ask if not already known. Grep for candidates if needed:
   ```
   grep -rn "DataLoader\|torch\.utils\.data" --include="*.py" .
   ```

3. **Export and load each model** — repeat for every model being compared:
   ```
   ml_export_model(model_file=..., model_class="ModelA", dataloader=...)
   dashboard_show_variant("ModelA")
   ```
   `dashboard_show_variant` opens the Comparison View automatically and the panel appears without a page reload.

4. **Tell the Operator** the Comparison View is ready and how many panels are loaded.

## Quick start (models already known)

```
ml_export_model(model_file="models/a.py", model_class="ModelA", dataloader="loaders/train.py:train_loader")
dashboard_show_variant("ModelA")

ml_export_model(model_file="models/b.py", model_class="ModelB", dataloader="loaders/train.py:train_loader")
dashboard_show_variant("ModelB")
```

## Managing panels

| Goal | Tool |
|---|---|
| Add a model to the view | `dashboard_show_variant("ModelClass")` |
| Remove one panel | `dashboard_remove_variant("ModelClass")` |
| Clear all panels | `dashboard_clear_variants()` |

Calling `dashboard_show_variant` with an already-loaded model class has no effect (no duplicate panels).

## Notes

- See `/visualize-model` for model file and dataloader format requirements.
- Each panel is independent — zoom, pan, and node selection in one panel do not affect others.
- The Comparison View layout: 1 panel fills the screen, 2 panels split side-by-side, 3 panels show two on top and one full-width below, 4+ panels form a 2-column grid.
