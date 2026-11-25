"""Read-only NVIDIA GPU discovery and on-demand measurements."""

from .inspector import NvidiaGpuInspector
from .local import MEMORY_FIELDS, snapshot_local_gpus
from .metadata import GpuPodCandidate, list_gpu_nodes, list_gpu_pod_candidates
from .models import (
    Availability,
    GpuInventory,
    GpuNode,
    GpuSnapshot,
    Measurement,
    WorkloadMapping,
)
from .nvml import LocalNvmlSource
from .reader import AccessPod
from .telemetry import (
    GpuObservation,
    GpuRestoration,
    GpuScenarioTelemetry,
    GpuWorkerTelemetry,
)

__all__ = [
    "AccessPod",
    "Availability",
    "GpuInventory",
    "GpuNode",
    "GpuObservation",
    "GpuPodCandidate",
    "GpuRestoration",
    "GpuScenarioTelemetry",
    "GpuSnapshot",
    "GpuWorkerTelemetry",
    "Measurement",
    "LocalNvmlSource",
    "NvidiaGpuInspector",
    "WorkloadMapping",
    "MEMORY_FIELDS",
    "list_gpu_nodes",
    "list_gpu_pod_candidates",
    "snapshot_local_gpus",
]
