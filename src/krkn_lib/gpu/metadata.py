"""Read Kubernetes GPU node and pod metadata with bounded list calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import chain
import math
import time
from typing import Any

from kubernetes.utils.quantity import parse_quantity


_DEFAULT_RESOURCE_NAME = "nvidia.com/gpu"
_PAGE_SIZE = 500


@dataclass(frozen=True)
class GpuPodCandidate:
    """Metadata for a pod matched by the caller's selector."""

    namespace: str
    name: str
    uid: str | None
    node_name: str | None
    phase: str | None
    requests_gpu: bool


def is_gpu_resource(name: str, resource_name: str | None = None) -> bool:
    """Match advertised GPU resources, or the standard and configured names."""
    return name != "nvidia.com/hostdev" and (
        name in ("nvidia.com/gpu", "nvidia.com/gpu.shared")
        or name.startswith("nvidia.com/mig-")
        or name == resource_name
        or (resource_name is None and name.startswith("nvidia.com/gpu."))
    )


def _candidate_resource_names(names, resource_name: str):
    return (name for name in names if is_gpu_resource(name, resource_name))


def normalize_count(value: object) -> Decimal | None:
    """Parse a Kubernetes quantity without rounding through binary floats."""
    if value is None or isinstance(value, bool):
        return None
    try:
        quantity = parse_quantity(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    try:
        result = Decimal(quantity)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _positive_timeout(timeout_seconds: float) -> float:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    return timeout


def _list_pages(api_call: Any, timeout_seconds: float) -> list[Any]:
    deadline = time.monotonic() + _positive_timeout(timeout_seconds)
    items: list[Any] = []
    continuation: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Kubernetes metadata listing timed out")
        kwargs: dict[str, Any] = {
            "limit": _PAGE_SIZE,
            "_request_timeout": remaining,
        }
        if continuation:
            kwargs["_continue"] = continuation
        response = api_call(**kwargs)
        items.extend(response.items or [])
        metadata = getattr(response, "metadata", None)
        continuation = getattr(metadata, "_continue", None)
        if not continuation:
            return items


def _node_ready_status(status: Any) -> str | None:
    return next(
        (condition.status for condition in (getattr(status, "conditions", None) or ())
         if condition.type == "Ready"),
        None,
    )


def list_gpu_nodes(
    core_api: Any,
    label_selector: str,
    *,
    resource_name: str = _DEFAULT_RESOURCE_NAME,
    ready_only: bool = False,
    timeout_seconds: float = 10,
) -> list[str]:
    """List nodes advertising a positive allocatable GPU resource.

    Set ``ready_only`` to require the Kubernetes ``Ready=True`` condition.
    Capacity is used only when the allocatable map does not contain that
    resource key. Kubernetes API failures are propagated without returning a
    partial list.
    """
    nodes = _list_pages(
        lambda **kwargs: core_api.list_node(
            label_selector=label_selector, **kwargs
        ),
        timeout_seconds,
    )
    result: list[str] = []
    for node in nodes:
        status = getattr(node, "status", None)
        if ready_only and _node_ready_status(status) != "True":
            continue
        capacity = getattr(status, "capacity", None) or {}
        allocatable = getattr(status, "allocatable", None) or {}
        names = _candidate_resource_names(capacity.keys() | allocatable.keys(), resource_name)
        if any(
            (count := normalize_count(
                allocatable[name] if name in allocatable else capacity.get(name)
            )) is not None
            and count > 0
            for name in names
        ):
            name = getattr(getattr(node, "metadata", None), "name", None)
            if name is not None:
                result.append(name)
    return result


def _pod_requests_gpu(pod: Any, resource_name: str) -> bool:
    spec = getattr(pod, "spec", None)
    containers = chain(
        getattr(spec, "containers", None) or (),
        getattr(spec, "init_containers", None) or (),
    )
    for container in containers:
        resources = getattr(container, "resources", None)
        requests = getattr(resources, "requests", None) or {}
        limits = getattr(resources, "limits", None) or {}
        for name in _candidate_resource_names(requests.keys() | limits.keys(), resource_name):
            value = requests[name] if name in requests else limits[name]
            count = normalize_count(value)
            if count is not None and count > 0:
                return True
    return False


def list_gpu_pod_candidates(
    core_api: Any,
    namespace: str,
    label_selector: str,
    *,
    resource_name: str = _DEFAULT_RESOURCE_NAME,
    timeout_seconds: float = 10,
) -> list[GpuPodCandidate]:
    """List matching pods and flag GPU requests in regular and init containers.

    This describes resource declarations, not current GPU assignments.
    No pods are filtered out by phase or GPU request.
    """
    pods = _list_pages(
        lambda **kwargs: core_api.list_namespaced_pod(
            namespace, label_selector=label_selector, **kwargs
        ),
        timeout_seconds,
    )
    candidates: list[GpuPodCandidate] = []
    for pod in pods:
        metadata = getattr(pod, "metadata", None)
        spec = getattr(pod, "spec", None)
        status = getattr(pod, "status", None)
        pod_namespace = getattr(metadata, "namespace", None) or namespace
        name = getattr(metadata, "name", None)
        if name is None:
            continue
        candidates.append(
            GpuPodCandidate(
                namespace=pod_namespace,
                name=name,
                uid=getattr(metadata, "uid", None),
                node_name=getattr(spec, "node_name", None),
                phase=getattr(status, "phase", None),
                requests_gpu=_pod_requests_gpu(pod, resource_name),
            )
        )
    return candidates
