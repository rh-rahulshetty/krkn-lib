"""Read GPU measurements on the current host without a cluster client."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from xml.etree.ElementTree import ParseError

from .models import (
    GpuSnapshot,
    _select_snapshot_measurements,
    utc_now,
)
from .parsers import field_definition, parse_gpu_xml, supported_fields

MEMORY_FIELDS = (
    "memory_total_bytes",
    "memory_free_bytes",
    "memory_used_bytes",
)


def snapshot_local_gpus(
    *,
    fields: Sequence[str] | None = None,
    device_uuids: Sequence[str] | None = None,
    timeout_seconds: float = 5,
    executable: str = "nvidia-smi",
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> GpuSnapshot:
    """Sample devices visible to the local NVIDIA driver via read-only CLI.

    Call this on the host (or in a container with NVIDIA device access).
    The result does not imply any Kubernetes node or workload identity.
    Pass an executable path if the NVIDIA driver is mounted outside PATH.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    known_fields = supported_fields()
    selected = set(fields) if fields is not None else set(known_fields)
    if invalid := selected - known_fields:
        raise ValueError(f"unknown GPU fields: {', '.join(sorted(invalid))}")

    source = "local:nvidia-smi -q -x"
    try:
        result = (run or subprocess.run)(
            (executable, "-q", "-x"),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        error = "local nvidia-smi timed out"
    except OSError:
        error = "local nvidia-smi could not be started"
    else:
        if result.returncode != 0:
            error = f"local nvidia-smi exited with status {result.returncode}"
        else:
            try:
                queried = parse_gpu_xml(result.stdout, selected)
            except (ParseError, ValueError):
                error = "local nvidia-smi returned invalid inventory XML"
            else:
                if device_uuids is None and not any(
                    device.uuid for device in queried.gpus
                ):
                    error = "local nvidia-smi reported no identifiable GPUs"
                else:
                    for measurement in queried.measurements:
                        measurement.source = source
                    if device_uuids is None:
                        return GpuSnapshot(
                            None, utc_now(), queried.measurements
                        )
                    scopes = {
                        device.uuid: "physical_gpu"
                        for device in queried.gpus
                        if device.uuid
                    }
                    scopes.update(
                        {
                            device.uuid: "mig_partition"
                            for device in queried.mig_partitions
                            if device.uuid
                        }
                    )
                    measurements, missing = _select_snapshot_measurements(
                        queried.measurements,
                        device_uuids,
                        set(scopes),
                        selected,
                        {
                            field_name: field_definition(field_name)
                            for field_name in selected
                        },
                        scopes,
                        source,
                    )
                    errors = [
                        f"device {uuid} absent from fresh NVIDIA query"
                        for uuid in missing
                    ]
                    return GpuSnapshot(None, utc_now(), measurements, errors)

    if device_uuids is None:
        return GpuSnapshot(None, utc_now(), [], [error])
    measurements, _ = _select_snapshot_measurements(
        [],
        device_uuids,
        set(),
        selected,
        {
            field_name: field_definition(field_name)
            for field_name in selected
        },
        {},
        source,
        unavailable_reason=error,
    )
    return GpuSnapshot(None, utc_now(), measurements, [error])
