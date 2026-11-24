import subprocess
import unittest
from unittest.mock import patch

from krkn_lib.gpu import Availability, snapshot_local_gpus


class LocalGpuSnapshotTests(unittest.TestCase):
    def test_local_snapshot_reports_values_and_missing_fields(self):
        xml = """<nvidia_smi_log><gpu><uuid>GPU-a</uuid>
            <fb_memory_usage><free>0 MiB</free><total>1024 MiB</total>
            </fb_memory_usage></gpu></nvidia_smi_log>"""

        def run(command, **kwargs):
            self.assertEqual(
                command, ("/host-driver/usr/bin/nvidia-smi", "-q", "-x")
            )
            self.assertEqual(kwargs["timeout"], 0.5)
            return subprocess.CompletedProcess(command, 0, xml, "")

        snapshot = snapshot_local_gpus(
            fields=["memory_free_bytes", "memory_used_bytes"],
            executable="/host-driver/usr/bin/nvidia-smi",
            timeout_seconds=0.5,
            run=run,
        )
        fields = {item.field: item for item in snapshot.measurements}
        self.assertEqual(fields["memory_free_bytes"].value, 0)
        self.assertEqual(
            fields["memory_free_bytes"].status, Availability.AVAILABLE
        )
        self.assertEqual(
            fields["memory_used_bytes"].status, Availability.UNKNOWN
        )
        self.assertIsNone(snapshot.context)

    def test_device_selection_filters_and_reports_missing_uuids(self):
        xml = """<nvidia_smi_log>
            <gpu><uuid>GPU-a</uuid><fb_memory_usage>
                <free>512 MiB</free><total>1024 MiB</total>
            </fb_memory_usage></gpu>
            <gpu><uuid>GPU-b</uuid><fb_memory_usage>
                <free>256 MiB</free><total>1024 MiB</total>
            </fb_memory_usage></gpu>
        </nvidia_smi_log>"""
        result = subprocess.CompletedProcess([], 0, xml, "")

        snapshot = snapshot_local_gpus(
            fields=["memory_free_bytes"],
            device_uuids=["GPU-a", "GPU-missing"],
            run=lambda *args, **kwargs: result,
        )
        measurements = {
            item.device_uuid: item for item in snapshot.measurements
        }
        self.assertEqual(set(measurements), {"GPU-a", "GPU-missing"})
        self.assertEqual(
            measurements["GPU-a"].value, 512 * 1024 * 1024
        )
        self.assertEqual(
            measurements["GPU-a"].status, Availability.AVAILABLE
        )
        self.assertEqual(
            measurements["GPU-a"].source, "local:nvidia-smi -q -x"
        )
        self.assertEqual(
            measurements["GPU-missing"].status, Availability.UNAVAILABLE
        )
        self.assertEqual(
            measurements["GPU-missing"].source, "local:nvidia-smi -q -x"
        )
        self.assertEqual(
            snapshot.errors,
            ["device GPU-missing absent from fresh NVIDIA query"],
        )

    def test_selected_devices_are_unavailable_when_local_query_fails(self):
        with patch(
            "krkn_lib.gpu.local.subprocess.run", side_effect=FileNotFoundError
        ):
            snapshot = snapshot_local_gpus(
                fields=["memory_free_bytes"], device_uuids=["GPU-a"]
            )

        self.assertEqual(
            snapshot.measurements[0].status, Availability.UNAVAILABLE
        )
        self.assertEqual(
            snapshot.measurements[0].source, "local:nvidia-smi -q -x"
        )
        self.assertEqual(
            snapshot.errors, ["local nvidia-smi could not be started"]
        )
