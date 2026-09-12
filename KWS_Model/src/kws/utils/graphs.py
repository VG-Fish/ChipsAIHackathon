"""Live per-epoch charts written beside a run's JSONL metrics.

``MetricsRecorder.append`` is the single point every epoch-level stage already
passes through, so ``--graphs`` is a run-wide switch rather than a parameter
threaded through six stage signatures.  Each phase gets a CSV of its records
plus a self-contained HTML chart that reloads itself, which keeps the per-epoch
cost at two small text writes instead of a plot render, and keeps the chart
openable with nothing but a browser (no server, no CDN, no extra install).

The same code runs as a CLI so an already-finished or already-running output
directory can be charted after the fact::

    python -m kws.utils.graphs outputs/my-run --watch
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


GRAPHS_ENV_VAR = "KWS_GRAPHS"
GRAPHS_DIRNAME = "graphs"
RELOAD_SECONDS = 3

logger = logging.getLogger(__name__)

# Columns that identify the run rather than trace its learning curve.
_NON_SERIES_COLUMNS = frozenset(
    {
        "schema_version",
        "stage",
        "phase",
        "seed",
        "epoch",
        "global_step",
        "elapsed_seconds",
        "parameter_count",
        "learning_rate",
        "learning_rates",
    }
)
_LEADING_COLUMNS = ("epoch", "global_step", "elapsed_seconds")
_ACCURACY_TOKENS = ("acc", "f1", "precision", "recall", "score")
_LOSS_TOKENS = ("loss", "error", "nll", "perplexity")
_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
)

_enabled: bool | None = None
_index_signatures: dict[Path, tuple] = {}


# --------------------------------------------------------------------------
# Switch
# --------------------------------------------------------------------------


def enabled() -> bool:
    """Return whether per-epoch graph updates are on for this process."""
    if _enabled is not None:
        return _enabled
    return os.environ.get(GRAPHS_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def enable(value: bool = True) -> None:
    """Turn graph updates on for this process and anything it launches."""
    global _enabled
    _enabled = bool(value)
    os.environ[GRAPHS_ENV_VAR] = "1" if value else "0"


def add_cli_flag(parser: argparse.ArgumentParser) -> None:
    """Register ``--graphs`` so every entry point words it the same way."""
    parser.add_argument(
        "--graphs",
        action="store_true",
        help=(
            "After every epoch, refresh a CSV and a self-refreshing HTML chart "
            f"under <output-dir>/{GRAPHS_DIRNAME}/ (requires --output-dir)"
        ),
    )


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """Reduce one recorded value to something a CSV cell can hold."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalar(value[0])
        return ";".join(str(_scalar(item)) for item in value)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _columns(records: Sequence[dict]) -> list[str]:
    seen: list[str] = []
    for record in records:
        for key in record:
            if key not in seen:
                seen.append(key)
    leading = [key for key in _LEADING_COLUMNS if key in seen]
    return leading + sorted(key for key in seen if key not in leading)


def render_csv(records: Sequence[dict]) -> str:
    """Render the same records the JSONL holds, one row per epoch."""
    columns = _columns(records)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({key: _scalar(record.get(key)) for key in columns})
    return buffer.getvalue()


def _axis_for(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in _LOSS_TOKENS):
        return "loss"
    if any(token in lowered for token in _ACCURACY_TOKENS):
        return "accuracy"
    return None


def build_series(records: Sequence[dict]) -> list[dict]:
    """Pick the numeric columns worth drawing and bucket them by axis."""
    points: dict[str, list[list[float]]] = {}
    for index, record in enumerate(records):
        x = record.get("epoch")
        x = float(x) if _is_number(x) else float(index + 1)
        for key, raw in record.items():
            if key in _NON_SERIES_COLUMNS:
                continue
            value = _scalar(raw)
            if not _is_number(value):
                continue
            points.setdefault(key, []).append([x, float(value)])

    # ``val_acc`` and ``val_accuracy`` are the same curve under two names; keep
    # whichever spelling is more explicit so the legend reads cleanly.
    preferred: dict[tuple, str] = {}
    for name in sorted(points):
        signature = tuple(round(value, 12) for _, value in points[name])
        chosen = preferred.get(signature)
        if chosen is None or len(name) > len(chosen):
            preferred[signature] = name
    kept = set(preferred.values())

    series = [
        {"name": name, "axis": _axis_for(name), "points": points[name]}
        for name in sorted(points)
        if name in kept
    ]
    if any(item["axis"] for item in series):
        series = [item for item in series if item["axis"]]
    else:
        # An unfamiliar phase schema still deserves a chart.
        for item in series:
            item["axis"] = "loss"
    for index, item in enumerate(series):
        item["color"] = _PALETTE[index % len(_PALETTE)]
    return series


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


_CHART_HTML = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111827; --muted:#6b7280;
          --grid:#e5e7eb; --card:#fff; --line:#e5e7eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0b0f19; --fg:#e5e7eb; --muted:#9ca3af; --grid:#1f2937;
            --card:#111827; --line:#1f2937; }
  }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:860px; margin:0 auto; padding:20px 16px 32px; }
  h1 { font-size:17px; margin:0; letter-spacing:-0.01em; }
  .meta { margin:2px 0 14px; color:var(--muted); font-size:12px;
          font-variant-numeric:tabular-nums; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:10px 8px 4px; }
  svg { display:block; width:100%; height:auto; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 16px; list-style:none;
            margin:12px 0 0; padding:0; font-size:12px;
            font-variant-numeric:tabular-nums; }
  .legend li { display:flex; align-items:center; gap:6px; }
  .swatch { width:10px; height:3px; border-radius:2px; flex:none; }
  .legend b { font-weight:600; }
  .legend span { color:var(--muted); }
  footer { display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center;
           margin-top:14px; font-size:12px; color:var(--muted); }
  footer a { color:inherit; }
  label { display:flex; align-items:center; gap:6px; cursor:pointer; }
</style>
<main>
  <h1>__TITLE__</h1>
  <p class="meta" id="meta"></p>
  <div class="card"><div id="chart"></div></div>
  <ul class="legend" id="legend"></ul>
  <footer>
    <label><input type="checkbox" id="live"> live &middot; reloads every __RELOAD__s</label>
    <a href="__CSV__">download CSV</a>
    <span id="stamp"></span>
  </footer>
</main>
<script>
const DATA = __PAYLOAD__;
const NS = "http://www.w3.org/2000/svg";
const W = 760, H = 300, PAD = {t: 14, r: 54, b: 34, l: 54};

function el(name, attrs, text) {
  const node = document.createElementNS(NS, name);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(value) {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(1);
  return (Math.round(value * 1000) / 1000).toString();
}

function bounds(series) {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const [, y] of s.points) { if (y < lo) lo = y; if (y > hi) hi = y; }
  if (!isFinite(lo)) return [0, 1];
  if (lo === hi) { const pad = Math.abs(lo) * 0.1 || 0.5; return [lo - pad, hi + pad]; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function render() {
  const series = DATA.series.filter(s => s.points.length);
  const host = document.getElementById("chart");
  host.textContent = "";
  if (!series.length) { host.textContent = "No numeric series yet."; return; }

  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img"});
  const xs = series.flatMap(s => s.points.map(p => p[0]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const axes = {};
  for (const key of ["loss", "accuracy"]) {
    const group = series.filter(s => s.axis === key);
    if (group.length) axes[key] = bounds(group);
  }
  const px = v => PAD.l + (x1 === x0 ? 0 : (v - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const py = (v, key) => {
    const [lo, hi] = axes[key];
    return H - PAD.b - (hi === lo ? 0.5 : (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);
  };

  const primary = axes.loss ? "loss" : "accuracy";
  for (let i = 0; i <= 4; i++) {
    const [lo, hi] = axes[primary];
    const value = lo + (hi - lo) * (i / 4);
    const y = py(value, primary);
    svg.appendChild(el("line", {x1: PAD.l, x2: W - PAD.r, y1: y, y2: y,
      stroke: "var(--grid)", "stroke-width": 1}));
    svg.appendChild(el("text", {x: PAD.l - 8, y: y + 4, "text-anchor": "end",
      "font-size": 10, fill: "var(--muted)"}, fmt(value)));
    if (axes.loss && axes.accuracy) {
      const [alo, ahi] = axes.accuracy;
      svg.appendChild(el("text", {x: W - PAD.r + 8, y: y + 4, "text-anchor": "start",
        "font-size": 10, fill: "var(--muted)"}, fmt(alo + (ahi - alo) * (i / 4))));
    }
  }
  for (let i = 0; i <= 4; i++) {
    const value = x0 + (x1 - x0) * (i / 4);
    svg.appendChild(el("text", {x: px(value), y: H - PAD.b + 16, "text-anchor": "middle",
      "font-size": 10, fill: "var(--muted)"}, Math.round(value).toString()));
  }
  svg.appendChild(el("text", {x: (PAD.l + W - PAD.r) / 2, y: H - 2,
    "text-anchor": "middle", "font-size": 10, fill: "var(--muted)"}, DATA.xlabel));

  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)} ${py(p[1], s.axis).toFixed(1)}`).join(" ");
    svg.appendChild(el("path", {d, fill: "none", stroke: s.color, "stroke-width": 1.75,
      "stroke-linejoin": "round", "stroke-linecap": "round"}));
    const last = s.points[s.points.length - 1];
    svg.appendChild(el("circle", {cx: px(last[0]), cy: py(last[1], s.axis), r: 2.5, fill: s.color}));
  }
  host.appendChild(svg);

  const legend = document.getElementById("legend");
  legend.textContent = "";
  for (const s of series) {
    const last = s.points[s.points.length - 1][1];
    const item = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = s.color;
    const name = document.createElement("span");
    name.textContent = s.name;
    const value = document.createElement("b");
    value.textContent = fmt(last);
    item.append(swatch, name, value);
    legend.appendChild(item);
  }

  document.getElementById("meta").textContent =
    `${DATA.count} records` + (DATA.best ? ` · ${DATA.best}` : "");
  document.getElementById("stamp").textContent = "written " + DATA.updated;
}

render();

const live = document.getElementById("live");
live.checked = sessionStorage.getItem("kws-graph-live") !== "0";
live.addEventListener("change", () => {
  sessionStorage.setItem("kws-graph-live", live.checked ? "1" : "0");
  if (live.checked) schedule();
});
function schedule() { setTimeout(() => { if (live.checked) location.reload(); }, __RELOAD__ * 1000); }
schedule();
</script>
"""


def _best_summary(records: Sequence[dict]) -> str:
    """Describe the best validation point, when the phase reports one."""
    for key in ("val_accuracy", "val_acc", "val_loss"):
        values = [
            (record[key], record.get("epoch", index + 1))
            for index, record in enumerate(records)
            if _is_number(record.get(key))
        ]
        if not values:
            continue
        pick = min(values) if "loss" in key else max(values)
        return f"best {key} {pick[0]:.4f} @ epoch {pick[1]}"
    return ""


def render_chart_html(title: str, records: Sequence[dict], csv_name: str) -> str:
    payload = {
        "series": build_series(records),
        "xlabel": "epoch" if any("epoch" in record for record in records) else "record",
        "count": len(records),
        "best": _best_summary(records),
        "updated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return (
        _CHART_HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
        .replace("__TITLE__", html.escape(title))
        .replace("__CSV__", html.escape(csv_name, quote=True))
        .replace("__RELOAD__", str(RELOAD_SECONDS))
    )


_INDEX_HTML = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111827; --muted:#6b7280;
          --card:#fff; --line:#e5e7eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0b0f19; --fg:#e5e7eb; --muted:#9ca3af; --card:#111827; --line:#1f2937; }
  }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:640px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:17px; margin:0 0 2px; letter-spacing:-0.01em; }
  p { color:var(--muted); font-size:12px; margin:0 0 16px; }
  ul { list-style:none; margin:0; padding:0; display:grid; gap:8px; }
  li a { display:flex; justify-content:space-between; gap:12px; padding:10px 12px;
         border:1px solid var(--line); border-radius:8px; background:var(--card);
         text-decoration:none; color:inherit; }
  li a:hover { border-color:var(--muted); }
  li span { color:var(--muted); font-size:12px; }
</style>
<main>
  <h1>__TITLE__</h1>
  <p>__SUBTITLE__</p>
  <ul>__ITEMS__</ul>
</main>
<script>setTimeout(() => location.reload(), __RELOAD__ * 1000);</script>
"""


def _render_index(root: Path, charts: Sequence[Path]) -> str:
    items = []
    for chart in charts:
        relative = chart.relative_to(root / GRAPHS_DIRNAME)
        label = "/".join(relative.with_suffix("").parts)
        href = html.escape(relative.as_posix(), quote=True)
        items.append(
            f'<li><a href="{href}"><b>{html.escape(label)}</b>'
            f"<span>chart &amp; CSV</span></a></li>"
        )
    return (
        _INDEX_HTML.replace("__ITEMS__", "".join(items) or "<li><span>No metrics yet.</span></li>")
        .replace("__TITLE__", html.escape(root.name))
        .replace("__SUBTITLE__", f"{len(items)} phase(s) &middot; live charts for this run")
        .replace("__RELOAD__", str(RELOAD_SECONDS * 3))
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace a derived file in one step so a reader never sees a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def graphs_dir(root: str | Path) -> Path:
    return Path(root) / GRAPHS_DIRNAME


def run_root_for(metrics_path: str | Path) -> Path | None:
    """Recover the run root from ``<root>/metrics/<stage>/[…/]<phase>.jsonl``."""
    path = Path(metrics_path)
    for parent in path.parents:
        if parent.name == "metrics":
            return parent.parent
    return None


def update_phase(
    root: str | Path, metrics_path: str | Path, records: Sequence[dict]
) -> Path | None:
    """Refresh one phase's CSV and chart, and the run's chart index."""
    root = Path(root)
    metrics_path = Path(metrics_path)
    if not records:
        return None
    relative = metrics_path.relative_to(root / "metrics").with_suffix("")
    base = graphs_dir(root) / relative
    title = "/".join(relative.parts)
    csv_path = base.with_suffix(".csv")
    html_path = base.with_suffix(".html")
    _atomic_write_text(csv_path, render_csv(records))
    _atomic_write_text(html_path, render_chart_html(title, records, csv_path.name))
    _refresh_index(root)
    return html_path


def _refresh_index(root: Path) -> None:
    """Rewrite the index only when the set of charts actually changes."""
    directory = graphs_dir(root)
    charts = sorted(p for p in directory.rglob("*.html") if p.name != "index.html")
    signature = tuple(str(p) for p in charts)
    if _index_signatures.get(root) == signature:
        return
    _atomic_write_text(directory / "index.html", _render_index(root, charts))
    _index_signatures[root] = signature


def update_from_recorder(recorder: Any) -> None:
    """Refresh a phase's chart after an epoch was appended.

    Never raises: a chart is derived convenience data, and losing it must not
    end a training run that is otherwise healthy.
    """
    if not enabled():
        return
    try:
        root = run_root_for(recorder.path)
        if root is None:
            return
        update_phase(root, recorder.path, recorder.records)
    except Exception:  # noqa: BLE001 - see docstring
        logger.warning("could not update graphs for %s", recorder.path, exc_info=True)


# --------------------------------------------------------------------------
# Offline rebuild / watch
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run still being written can end in a partial final line.
                break
            if isinstance(record, dict):
                records.append(record)
    return records


def metrics_files(root: str | Path) -> list[Path]:
    metrics_root = Path(root) / "metrics"
    if not metrics_root.is_dir():
        return []
    return sorted(
        path
        for path in metrics_root.rglob("*.jsonl")
        if ".archive." not in path.name
    )


def rebuild(root: str | Path) -> list[Path]:
    """Regenerate every chart in a run from the JSONL already on disk."""
    root = Path(root).expanduser().resolve()
    written: list[Path] = []
    for path in metrics_files(root):
        chart = update_phase(root, path, _read_jsonl(path))
        if chart is not None:
            written.append(chart)
    if not written:
        _refresh_index(root)
    return written


def watch(root: str | Path, interval: float = 10.0) -> None:
    """Keep a run's charts current by polling its JSONL for changes."""
    root = Path(root).expanduser().resolve()
    seen: dict[Path, tuple[int, float]] = {}
    while True:
        for path in metrics_files(root):
            try:
                stat = path.stat()
            except OSError:
                continue
            fingerprint = (stat.st_size, stat.st_mtime)
            if seen.get(path) == fingerprint:
                continue
            seen[path] = fingerprint
            try:
                update_phase(root, path, _read_jsonl(path))
            except Exception:  # noqa: BLE001 - a watcher must outlive one bad file
                logger.warning("could not chart %s", path, exc_info=True)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="A run output root containing metrics/")
    parser.add_argument(
        "--watch",
        nargs="?",
        type=float,
        const=10.0,
        default=None,
        metavar="SECONDS",
        help="Keep charting while the run writes (default poll: every 10s)",
    )
    args = parser.parse_args()

    root = Path(args.run_dir).expanduser().resolve()
    charts = rebuild(root)
    print(f"{len(charts)} chart(s) under {graphs_dir(root)}")
    print(f"open {graphs_dir(root) / 'index.html'}")
    if args.watch is not None:
        print(f"watching every {args.watch:g}s; Ctrl-C to stop")
        try:
            watch(root, interval=args.watch)
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()
