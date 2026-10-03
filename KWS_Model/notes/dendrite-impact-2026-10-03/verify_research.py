"""Check the newly generated evidence files without loading trained models."""
from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree


HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "DENDRITE_IMPACT_RESEARCH_2026-10-03.md"
REPORT = HERE / "research-verification.json"
SENSITIVE_KEYS = {"token", "pai_token", "api_key", "password", "secret", "pai_email"}


def check_keys(value: object, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                errors.append(f"Credential-like key in {location}: {key}")
            check_keys(item, location, errors)
    elif isinstance(value, list):
        for item in value:
            check_keys(item, location, errors)


def main() -> None:
    errors: list[str] = []
    checked: dict[str, int] = {"json": 0, "csv": 0, "python": 0, "svg": 0,
                               "markdown": 0, "local_links": 0}
    csv_rows: dict[str, int] = {}
    for path in sorted(HERE.iterdir()):
        if path == REPORT or not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text())
                check_keys(data, path.name, errors)
                checked["json"] += 1
            elif path.suffix == ".csv":
                with path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                if any(None in row for row in rows):
                    errors.append(f"CSV row has extra cells: {path.name}")
                csv_rows[path.name] = len(rows)
                checked["csv"] += 1
            elif path.suffix == ".py":
                ast.parse(path.read_text(), filename=str(path))
                checked["python"] += 1
            elif path.suffix == ".svg":
                ElementTree.parse(path)
                checked["svg"] += 1
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    for path in [MAIN, *sorted(HERE.glob("*.md"))]:
        checked["markdown"] += 1
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            target = re.sub(r":\d+$", "", target)
            resolved = Path(target) if target.startswith("/") else path.parent / target
            checked["local_links"] += 1
            if not resolved.exists():
                errors.append(f"Broken link in {path.name}: {target}")

    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checked, "csv_rows": csv_rows, "errors": errors,
              "scope": "New research artifacts only; no model execution or training."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
