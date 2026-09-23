"""Immutable and validated configuration for the ReRAM backend.

The configuration is deliberately narrower than NeuroSim's complete option
set.  This keeps the v1 command line contract explicit and makes it possible
to include the exact assumptions in every exported manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class HardwareConfigError(ValueError):
    """Raised when a hardware configuration cannot be used safely."""


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise HardwareConfigError(f"configuration section {name!r} is required")
    return value


def _required(section: Mapping[str, Any], name: str) -> Any:
    if name not in section:
        raise HardwareConfigError(f"missing configuration value {name!r}")
    return section[name]


def _bool(section: Mapping[str, Any], name: str) -> bool:
    value = _required(section, name)
    if not isinstance(value, bool):
        raise HardwareConfigError(f"{name} must be a boolean")
    return value


def _int(section: Mapping[str, Any], name: str) -> int:
    value = _required(section, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareConfigError(f"{name} must be an integer")
    return value


def _float(section: Mapping[str, Any], name: str) -> float:
    value = _required(section, name)
    if isinstance(value, bool):
        raise HardwareConfigError(f"{name} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise HardwareConfigError(f"{name} must be numeric") from exc
    raise HardwareConfigError(f"{name} must be numeric")


@dataclass(frozen=True)
class SourceConfig:
    root_env: str


@dataclass(frozen=True)
class SimulationConfig:
    inference_only: bool
    mapping: str
    pipeline: bool


@dataclass(frozen=True)
class PrecisionConfig:
    weight_bits: int
    activation_bits: int
    cell_bits: int
    adc_bits: int


@dataclass(frozen=True)
class ArrayConfig:
    rows: int
    cols: int
    columns_per_adc: int


@dataclass(frozen=True)
class TechnologyConfig:
    node_nm: int
    temperature_k: float
    clock_hz: float


@dataclass(frozen=True)
class MemoryConfig:
    access_type: str
    resistance_on_ohm: float
    resistance_off_ohm: float
    read_voltage_v: float
    read_pulse_width_s: float
    write_voltage_v: float
    write_pulse_width_s: float
    access_resistance_ohm: float


@dataclass(frozen=True)
class TraceConfig:
    split: str
    samples: int
    seed: int


@dataclass(frozen=True)
class GraphConfig:
    grouped_conv_mode: str
    reject_unsupported_ops: bool


@dataclass(frozen=True)
class RuntimeConfig:
    timeout_seconds: float


@dataclass(frozen=True)
class HardwareConfig:
    schema_version: int
    backend: str
    source: SourceConfig
    simulation: SimulationConfig
    precision: PrecisionConfig
    array: ArrayConfig
    technology: TechnologyConfig
    memory: MemoryConfig
    trace: TraceConfig
    graph: GraphConfig
    runtime: RuntimeConfig
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy suitable for manifests."""

        result = asdict(self)
        if self.config_path is not None:
            result["config_path"] = str(self.config_path)
        return result


def _validate(config: HardwareConfig) -> None:
    if config.schema_version != 1:
        raise HardwareConfigError("schema_version must be 1")
    if config.backend != "neurosim_v21_reram":
        raise HardwareConfigError("backend must be neurosim_v21_reram")
    if not config.source.root_env.strip():
        raise HardwareConfigError("source.root_env must not be empty")
    if not config.simulation.inference_only:
        raise HardwareConfigError("simulation.inference_only must be true")
    if config.simulation.mapping not in {"novel", "conventional"}:
        raise HardwareConfigError("simulation.mapping must be novel or conventional")

    precision = config.precision
    for name, value in (
        ("weight_bits", precision.weight_bits),
        ("activation_bits", precision.activation_bits),
        ("cell_bits", precision.cell_bits),
        ("adc_bits", precision.adc_bits),
    ):
        if not 1 <= value <= 16:
            raise HardwareConfigError(f"{name} must be between 1 and 16")
    if precision.cell_bits > precision.weight_bits:
        raise HardwareConfigError("cell_bits cannot exceed weight_bits")

    array = config.array
    if array.rows not in {32, 64, 128, 256}:
        raise HardwareConfigError("array.rows must be one of 32, 64, 128, 256")
    if array.cols not in {32, 64, 128, 256}:
        raise HardwareConfigError("array.cols must be one of 32, 64, 128, 256")
    if not 1 <= array.columns_per_adc <= array.cols:
        raise HardwareConfigError("array.columns_per_adc must be between 1 and array.cols")
    if array.cols % array.columns_per_adc:
        raise HardwareConfigError("array.cols must be divisible by columns_per_adc")

    technology = config.technology
    if technology.node_nm <= 0:
        raise HardwareConfigError("technology.node_nm must be positive")
    if technology.temperature_k <= 0:
        raise HardwareConfigError("technology.temperature_k must be positive")
    if technology.clock_hz <= 0:
        raise HardwareConfigError("technology.clock_hz must be positive")

    memory = config.memory
    if memory.resistance_on_ohm <= 0:
        raise HardwareConfigError("resistance_on_ohm must be positive")
    if memory.resistance_off_ohm <= memory.resistance_on_ohm:
        raise HardwareConfigError("resistance_off_ohm must exceed resistance_on_ohm")
    for name, value in (
        ("read_voltage_v", memory.read_voltage_v),
        ("read_pulse_width_s", memory.read_pulse_width_s),
        ("write_voltage_v", memory.write_voltage_v),
        ("write_pulse_width_s", memory.write_pulse_width_s),
    ):
        if value <= 0:
            raise HardwareConfigError(f"{name} must be positive")
    if memory.access_resistance_ohm < 0:
        raise HardwareConfigError("access_resistance_ohm must not be negative")

    if not config.trace.split.strip():
        raise HardwareConfigError("trace.split must not be empty")
    if config.trace.samples <= 0:
        raise HardwareConfigError("trace.samples must be positive")
    if config.trace.seed < 0:
        raise HardwareConfigError("trace.seed must not be negative")
    if config.graph.grouped_conv_mode not in {"reject", "dense_upper_bound", "patched"}:
        raise HardwareConfigError(
            "graph.grouped_conv_mode must be reject, dense_upper_bound, or patched"
        )
    if config.runtime.timeout_seconds <= 0:
        raise HardwareConfigError("runtime.timeout_seconds must be positive")


def load_hardware_config(path: str | Path) -> HardwareConfig:
    """Load and validate a YAML hardware configuration."""

    config_path = Path(path)
    if not config_path.exists():
        raise HardwareConfigError(f"configuration file {config_path} does not exist")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardwareConfigError(f"could not read configuration {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise HardwareConfigError("configuration root must be a mapping")

    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise HardwareConfigError("schema_version must be an integer")
    backend = raw.get("backend")
    if not isinstance(backend, str):
        raise HardwareConfigError("backend must be a string")

    source = _section(raw, "source")
    simulation = _section(raw, "simulation")
    precision = _section(raw, "precision")
    array = _section(raw, "array")
    technology = _section(raw, "technology")
    memory = _section(raw, "memory")
    trace = _section(raw, "trace")
    graph = _section(raw, "graph")
    runtime = _section(raw, "runtime")
    config = HardwareConfig(
        schema_version=schema_version,
        backend=backend,
        source=SourceConfig(root_env=str(_required(source, "root_env"))),
        simulation=SimulationConfig(
            inference_only=_bool(simulation, "inference_only"),
            mapping=str(_required(simulation, "mapping")),
            pipeline=_bool(simulation, "pipeline"),
        ),
        precision=PrecisionConfig(
            weight_bits=_int(precision, "weight_bits"),
            activation_bits=_int(precision, "activation_bits"),
            cell_bits=_int(precision, "cell_bits"),
            adc_bits=_int(precision, "adc_bits"),
        ),
        array=ArrayConfig(
            rows=_int(array, "rows"),
            cols=_int(array, "cols"),
            columns_per_adc=_int(array, "columns_per_adc"),
        ),
        technology=TechnologyConfig(
            node_nm=_int(technology, "node_nm"),
            temperature_k=_float(technology, "temperature_k"),
            clock_hz=_float(technology, "clock_hz"),
        ),
        memory=MemoryConfig(
            access_type=str(_required(memory, "access_type")),
            resistance_on_ohm=_float(memory, "resistance_on_ohm"),
            resistance_off_ohm=_float(memory, "resistance_off_ohm"),
            read_voltage_v=_float(memory, "read_voltage_v"),
            read_pulse_width_s=_float(memory, "read_pulse_width_s"),
            write_voltage_v=_float(memory, "write_voltage_v"),
            write_pulse_width_s=_float(memory, "write_pulse_width_s"),
            access_resistance_ohm=_float(memory, "access_resistance_ohm"),
        ),
        trace=TraceConfig(
            split=str(_required(trace, "split")),
            samples=_int(trace, "samples"),
            seed=_int(trace, "seed"),
        ),
        graph=GraphConfig(
            grouped_conv_mode=str(_required(graph, "grouped_conv_mode")),
            reject_unsupported_ops=_bool(graph, "reject_unsupported_ops"),
        ),
        runtime=RuntimeConfig(timeout_seconds=_float(runtime, "timeout_seconds")),
        config_path=config_path.resolve(),
    )
    _validate(config)
    return config
