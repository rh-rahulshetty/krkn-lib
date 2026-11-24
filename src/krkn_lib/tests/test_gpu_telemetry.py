import json
import unittest

from krkn_lib.gpu.models import Availability, Evidence, GpuSnapshot, Measurement
from krkn_lib.gpu.telemetry import (
    GpuObservation,
    GpuScenarioTelemetry,
    GpuWorkerTelemetry,
)
from krkn_lib.models.elastic.models import ElasticChaosRunTelemetry
from krkn_lib.models.telemetry import ChaosRunTelemetry, ScenarioTelemetry


class GpuTelemetryTests(unittest.TestCase):
    def _gpu_payload(self):
        snapshot = GpuSnapshot(
            context="test-context",
            observed_at="2026-11-19T12:00:00+00:00",
            measurements=[
                Measurement(
                    device_uuid="GPU-1",
                    scope="device",
                    field="temperature",
                    value=72,
                    unit="C",
                    value_type="integer",
                    source="nvml",
                    observed_at="2026-11-19T12:00:00+00:00",
                    status=Availability.AVAILABLE,
                    alternatives=[
                        Evidence(
                            source="nvidia-smi",
                            value=71,
                            observed_at="2026-11-19T12:00:00+00:00",
                        )
                    ],
                )
            ],
        )
        return GpuScenarioTelemetry(
            fault_type="gpu_memory",
            workers=[
                GpuWorkerTelemetry(
                    node_name="worker-0",
                    phase="during",
                    pod_deleted=False,
                    device_execution_stats={
                        "GPU-1": {"injected": True, "bytes": 1024}
                    },
                    observation_errors={"started": ["sample unavailable"]},
                )
            ],
            observations=[
                GpuObservation(
                    node_name="worker-0", phase="post", snapshot=snapshot
                )
            ],
        )

    def _scenario(self, gpu):
        return ScenarioTelemetry(
            {
                "start_timestamp": 1,
                "end_timestamp": 2,
                "scenario": "gpu.yaml",
                "scenario_type": "gpu",
                "exit_status": 0,
                "parameters_base64": "",
                "parameters": {},
                "affected_pods": {},
                "affected_vmis": {},
                "affected_nodes": [],
                "cluster_events": [],
                "gpu": gpu,
            }
        )

    def _run_telemetry(self, scenario):
        return ChaosRunTelemetry(
            {
                "scenarios": [json.loads(scenario.to_json())],
                "node_summary_infos": [],
                "node_taints": [],
                "total_node_count": 1,
                "cloud_infrastructure": "Unknown",
                "cloud_type": "self-managed",
                "cluster_version": "Unknown",
                "major_version": 0,
                "kubernetes_objects_count": {},
                "network_plugins": [],
                "run_uuid": "run-1",
                "timestamp": "2026-11-19T12:00:00Z",
                "health_checks": [],
                "virt_checks": [],
                "object_state_checks": [],
                "job_status": True,
                "build_url": "manual",
                "tag": "",
                "error_logs": [],
            }
        )

    def test_json_and_elastic_round_trip_rebuilds_typed_gpu_records(self):
        scenario = self._scenario(None)
        scenario.gpu = self._gpu_payload()
        run = self._run_telemetry(scenario)
        decoded = ChaosRunTelemetry(json.loads(run.to_json()))
        json_round_trip = decoded.scenarios[0]

        gpu = json_round_trip.gpu
        self.assertIsInstance(gpu, GpuScenarioTelemetry)
        self.assertIsInstance(gpu.workers[0], GpuWorkerTelemetry)
        self.assertIsInstance(gpu.observations[0], GpuObservation)
        measurement = gpu.observations[0].snapshot.measurements[0]
        self.assertIsInstance(measurement, Measurement)
        self.assertIs(measurement.status, Availability.AVAILABLE)
        self.assertIsInstance(measurement.alternatives[0], Evidence)
        self.assertEqual(measurement.alternatives[0].value, 71)
        self.assertEqual(
            gpu.workers[0].device_execution_stats["GPU-1"]["bytes"], 1024
        )
        self.assertEqual(
            gpu.workers[0].observation_errors,
            {"started": ["sample unavailable"]},
        )
        rebuilt = GpuSnapshot.from_dict(
            gpu.observations[0].snapshot.to_dict()
        )
        self.assertEqual(
            rebuilt.to_dict(), gpu.observations[0].snapshot.to_dict()
        )

        elastic = ElasticChaosRunTelemetry(decoded)
        indexed = elastic.to_dict()
        self.assertEqual(indexed["scenarios"][0]["gpu"], gpu.to_dict())
        reconstructed = ElasticChaosRunTelemetry(**indexed)
        self.assertEqual(
            reconstructed.to_dict()["scenarios"][0]["gpu"], gpu.to_dict()
        )

    def test_legacy_scenario_without_gpu_defaults_to_none(self):
        scenario = self._scenario(None)
        legacy = json.loads(scenario.to_json())
        legacy.pop("gpu")

        self.assertIsNone(ScenarioTelemetry(legacy).gpu)
        elastic = ElasticChaosRunTelemetry(
            self._run_telemetry(ScenarioTelemetry(legacy))
        )
        reconstructed = ElasticChaosRunTelemetry(**elastic.to_dict())
        self.assertNotIn("gpu", reconstructed.to_dict()["scenarios"][0])


if __name__ == "__main__":
    unittest.main()
