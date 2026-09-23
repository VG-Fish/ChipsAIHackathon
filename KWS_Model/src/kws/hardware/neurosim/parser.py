"""Parser for the small, explicit subset of NeuroSim output used by reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedNeuroSimResult:
    forward_latency_s: float
    forward_dynamic_energy_j: float
    leakage_power_w: float | None
    chip_area_m2: float | None
    raw_fields: dict[str, str]


_LINE = re.compile(
    r"^\s*(?P<key>[^:]+):\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>[^\s]+)?\s*$"
)
_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "j": 1.0,
    "mj": 1e-3,
    "uj": 1e-6,
    "µj": 1e-6,
    "μj": 1e-6,
    "nj": 1e-9,
    "pj": 1e-12,
    "w": 1.0,
    "mw": 1e-3,
    "uw": 1e-6,
    "µw": 1e-6,
    "μw": 1e-6,
    "nw": 1e-9,
    "mm^2": 1e-6,
    "mm2": 1e-6,
    "um^2": 1e-12,
    "µm^2": 1e-12,
    "um2": 1e-12,
}


def _parse_value(value: str, unit: str | None) -> float:
    scale = _SCALE.get((unit or "").lower(), 1.0)
    return float(value) * scale


def parse_neurosim_output(text: str) -> ParsedNeuroSimResult:
    raw_fields: dict[str, str] = {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        key = match.group("key").strip()
        raw_fields[key] = f"{match.group('value')} {match.group('unit') or ''}".strip()
        values[key.casefold()] = _parse_value(match.group("value"), match.group("unit"))

        # The official V2.1 executable uses descriptive labels rather than
        # the short labels used by the standalone backend.
        normalized = key.casefold().replace(" ", "")
        if normalized == "chiparea":
            values["chip area"] = values[key.casefold()]
        elif normalized.startswith("chipreadlatencyofforward"):
            values["forward latency"] = values[key.casefold()]
        elif normalized.startswith("chipreaddynamicenergyofforward"):
            values["forward dynamic energy"] = values[key.casefold()]
        elif normalized.startswith("chipleakagepower"):
            values["leakage power"] = values[key.casefold()]

    def find(*names: str) -> float | None:
        for name in names:
            if name.casefold() in values:
                return values[name.casefold()]
        return None

    latency = find("forward latency", "inference latency")
    if latency is None:
        raise ValueError("NeuroSim output is missing forward latency")
    dynamic = find("forward dynamic energy", "forward energy", "inference dynamic energy")
    if dynamic is None:
        raise ValueError("NeuroSim output is missing forward dynamic energy")
    return ParsedNeuroSimResult(
        forward_latency_s=latency,
        forward_dynamic_energy_j=dynamic,
        leakage_power_w=find("leakage power", "leakage"),
        chip_area_m2=find("chip area", "area"),
        raw_fields=raw_fields,
    )


def parse_neurosim_file(path: str | Path) -> ParsedNeuroSimResult:
    return parse_neurosim_output(Path(path).read_text(encoding="utf-8"))
