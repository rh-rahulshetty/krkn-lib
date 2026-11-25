"""Scenario telemetry payloads for GPU chaos scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from krkn_lib.gpu.models import GpuSnapshot, JsonRecord


@dataclass
class GpuRestoration(JsonRecord):
    """Device policy restoration evidence, distinct from observed metrics."""

    status: str
    reason: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuRestoration:
        return cls(value["status"], value.get("reason"))


@dataclass
class GpuWorkerTelemetry(JsonRecord):
    """GPU injector outcome for a worker node."""

    node_name: str
    phase: str | None = None
    pod_deleted: bool | None = None
    device_execution_stats: dict[
        str, dict[str, int | float | bool]
    ] = field(default_factory=dict)
    observation_errors: dict[str, list[str]] = field(default_factory=dict)
    restoration_safe: bool | None = None
    device_restorations: dict[str, GpuRestoration] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuWorkerTelemetry:
        return cls(
            node_name=value["node_name"],
            phase=value.get("phase"),
            pod_deleted=value.get("pod_deleted"),
            device_execution_stats={
                uuid: dict(stats)
                for uuid, stats in (
                    (value.get("device_execution_stats") or {}).items()
                )
            },
            observation_errors={
                phase: list(errors)
                for phase, errors in (
                    (value.get("observation_errors") or {}).items()
                )
            },
            restoration_safe=value.get("restoration_safe"),
            device_restorations={
                uuid: item if isinstance(item, GpuRestoration) else GpuRestoration.from_dict(item)
                for uuid, item in (value.get("device_restorations") or {}).items()
            },
        )


@dataclass
class GpuObservation(JsonRecord):
    """A GPU metrics snapshot captured for a node and scenario phase."""

    node_name: str
    phase: str
    snapshot: GpuSnapshot

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuObservation:
        snapshot = value["snapshot"]
        if isinstance(snapshot, dict):
            snapshot = GpuSnapshot.from_dict(snapshot)
        return cls(
            node_name=value["node_name"],
            phase=value["phase"],
            snapshot=snapshot,
        )


@dataclass
class GpuScenarioTelemetry(JsonRecord):
    """GPU-specific telemetry associated with one scenario run."""

    fault_type: str
    workers: list[GpuWorkerTelemetry] = field(default_factory=list)
    observations: list[GpuObservation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GpuScenarioTelemetry:
        return cls(
            fault_type=value["fault_type"],
            workers=[
                item
                if isinstance(item, GpuWorkerTelemetry)
                else GpuWorkerTelemetry.from_dict(item)
                for item in (value.get("workers") or [])
            ],
            observations=[
                item
                if isinstance(item, GpuObservation)
                else GpuObservation.from_dict(item)
                for item in (value.get("observations") or [])
            ],
        )


