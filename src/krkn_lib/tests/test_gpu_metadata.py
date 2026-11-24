"""Tests for Kubernetes GPU metadata listing helpers."""

from types import SimpleNamespace as Obj
import unittest

from krkn_lib.gpu.inventory import _gpu_resource, _resource_count
from krkn_lib.gpu.metadata import (
    GpuPodCandidate,
    list_gpu_nodes,
    list_gpu_pod_candidates,
)


def node(name, capacity=None, allocatable=None):
    return Obj(
        metadata=Obj(name=name),
        status=Obj(capacity=capacity or {}, allocatable=allocatable or {}),
    )


def pod(
    name, *, namespace="jobs", node_name=None, phase=None, containers=None,
    init_containers=None,
):
    return Obj(
        metadata=Obj(namespace=namespace, name=name, uid=f"uid-{name}"),
        spec=Obj(
            node_name=node_name,
            containers=containers or [],
            init_containers=init_containers,
        ),
        status=Obj(phase=phase),
    )


def container(requests=None, limits=None):
    return Obj(resources=Obj(requests=requests or {}, limits=limits or {}))


class GpuMetadataTests(unittest.TestCase):
    def test_nodes_paginate_filter_custom_resources_and_preserve_zero(self):
        pages = [
            Obj(
                items=[
                    node(
                        "zero-allocatable",
                        {"nvidia.com/gpu": "4"},
                        {"nvidia.com/gpu": "0"},
                    ),
                    node("capacity-fallback", {"nvidia.com/gpu": "2"}),
                    node("nonstandard", {"nvidia.com/gpu.other": "2"}),
                    node(
                        "custom",
                        {"example.com/accelerator": "3", "nvidia.com/hostdev": "8"},
                        {"example.com/accelerator": "1"},
                    ),
                ],
                metadata=Obj(_continue="next"),
            ),
            Obj(
                items=[
                    node(
                        "mig",
                        allocatable={
                            "nvidia.com/mig-1g.10gb": "2",
                            "nvidia.com/gpu.shared": "1",
                        },
                    )
                ],
                metadata=Obj(_continue=None),
            ),
        ]
        calls = []

        class Api:
            def list_node(self, **kwargs):
                calls.append(kwargs)
                return pages.pop(0)

        names = list_gpu_nodes(
            Api(), "accelerated=true", resource_name="example.com/accelerator"
        )

        self.assertEqual(names, ["capacity-fallback", "custom", "mig"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["label_selector"], "accelerated=true")
        self.assertEqual(calls[0]["limit"], 500)
        self.assertEqual(calls[1]["_continue"], "next")
        self.assertTrue(all(call["_request_timeout"] > 0 for call in calls))

    def test_ready_only_excludes_unreachable_gpu_nodes(self):
        nodes = []
        for name, state in (("ready", "True"), ("unknown", "Unknown"), ("down", "False")):
            item = node(name, allocatable={"nvidia.com/gpu": "1"})
            item.status.conditions = [Obj(type="Ready", status=state)]
            nodes.append(item)
        nodes.append(node("no-condition", allocatable={"nvidia.com/gpu": "1"}))
        api = Obj(list_node=lambda **_: Obj(items=nodes, metadata=None))

        self.assertEqual(list_gpu_nodes(api, "", ready_only=True), ["ready"])
        self.assertEqual(list_gpu_nodes(api, ""), ["ready", "unknown", "down", "no-condition"])

    def test_pods_include_all_matches_and_apply_request_precedence(self):
        pages = [
            Obj(
                items=[
                    pod(
                        "worker",
                        node_name="gpu-node",
                        phase="Running",
                        containers=[
                            container(
                                requests={"nvidia.com/gpu": "0"},
                                limits={"nvidia.com/gpu": "2"},
                            ),
                            container(
                                limits={"nvidia.com/mig-1g.10gb": "4"}
                            ),
                            container(
                                requests={"example.com/accelerator": "1.25"},
                                limits={"example.com/accelerator": "5"},
                            ),
                        ],
                    )
                ],
                metadata=Obj(_continue="more"),
            ),
            Obj(
                items=[pod("pending", phase="Pending"), pod("no-resources")],
                metadata=Obj(_continue=None),
            ),
        ]
        calls = []

        class Api:
            def list_namespaced_pod(self, namespace, **kwargs):
                calls.append((namespace, kwargs))
                return pages.pop(0)

        candidates = list_gpu_pod_candidates(
            Api(), "jobs", "selected=yes", resource_name="example.com/accelerator"
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0], GpuPodCandidate(
            "jobs", "worker", "uid-worker", "gpu-node", "Running", True
        ))
        self.assertEqual(candidates[1].phase, "Pending")
        self.assertFalse(candidates[1].requests_gpu)
        self.assertEqual(candidates[2].node_name, None)
        self.assertEqual(calls[0][0], "jobs")
        self.assertEqual(calls[0][1]["label_selector"], "selected=yes")
        self.assertEqual(calls[1][1]["_continue"], "more")

    def test_pods_report_init_container_gpu_requests(self):
        class Api:
            def list_namespaced_pod(self, namespace, **kwargs):
                return Obj(
                    items=[
                        pod(
                            "init-only",
                            containers=[container()],
                            init_containers=[container(limits={"nvidia.com/gpu": "1"})],
                        ),
                        pod(
                            "zero-init-request",
                            init_containers=[
                                container(
                                    requests={"nvidia.com/gpu": "0"},
                                    limits={"nvidia.com/gpu": "1"},
                                ),
                            ],
                        ),
                    ],
                    metadata=Obj(_continue=None),
                )

        candidates = list_gpu_pod_candidates(Api(), "jobs", "selected=yes")

        self.assertEqual(
            {pod.name: pod.requests_gpu for pod in candidates},
            {"init-only": True, "zero-init-request": False},
        )

    def test_api_failure_on_later_page_is_propagated(self):
        class Api:
            def __init__(self):
                self.calls = 0

            def list_node(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return Obj(
                        items=[node("partial", allocatable={"nvidia.com/gpu": "1"})],
                        metadata=Obj(_continue="next"),
                    )
                raise RuntimeError("second page denied")

        with self.assertRaisesRegex(RuntimeError, "second page denied"):
            list_gpu_nodes(Api(), "")

    def test_resource_normalization_is_shared_with_inventory(self):
        self.assertTrue(_gpu_resource("nvidia.com/gpu.shared"))
        self.assertTrue(_gpu_resource("nvidia.com/gpu.other"))
        self.assertTrue(_gpu_resource("nvidia.com/mig-1g.10gb"))
        self.assertFalse(_gpu_resource("nvidia.com/hostdev"))
        count = _resource_count(
            "nvidia.com/gpu", {"nvidia.com/gpu": "9007199254740993"},
            {"nvidia.com/gpu": "0"}
        )
        self.assertEqual(count.capacity, 9007199254740993)
        self.assertEqual(count.allocatable, 0)


if __name__ == "__main__":
    unittest.main()
