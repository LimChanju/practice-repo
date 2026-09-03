"""Pure-Python tests for synchronized A/C spectator-video bookkeeping."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from v3_chan.ac_feedback.video import SynchronizedSpectatorVideoRecorder


class _FakeVideoBackend:
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        self.interval_steps = int(kwargs["interval_steps"])
        self._capture_count = 0
        self._step_count = 0
        self.setup_called = False
        self.close_called = False

    def setup(self) -> bool:
        self.setup_called = True
        return True

    def capture(self) -> None:
        self._step_count += 1
        if self._step_count % self.interval_steps == 0:
            self._capture_count += 1

    def close(self) -> None:
        self.close_called = True


class _SilentFailureVideoBackend(_FakeVideoBackend):
    def capture(self) -> None:
        self._step_count += 1


class SynchronizedSpectatorVideoTests(unittest.TestCase):
    def test_disabled_default_creates_no_backend_or_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            called = []

            def factory(**kwargs):  # type: ignore[no-untyped-def]
                called.append(kwargs)
                return _FakeVideoBackend(**kwargs)

            output = Path(directory) / "session.hdf5"
            recorder = SynchronizedSpectatorVideoRecorder(
                output, recorder_factory=factory
            )
            self.assertFalse(recorder.setup())
            self.assertIsNone(recorder.capture(0.1, 1, "trial-1"))
            recorder.close()

            self.assertEqual(called, [])
            self.assertFalse(recorder.csv_sidecar_path.exists())
            self.assertFalse(recorder.json_sidecar_path.exists())
            self.assertFalse(recorder.record_dir.exists())
            self.assertFalse(recorder.metadata["enabled"])

    def test_enabled_capture_writes_csv_and_json_clock_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backends = []

            def factory(**kwargs):  # type: ignore[no-untyped-def]
                backend = _FakeVideoBackend(**kwargs)
                backends.append(backend)
                return backend

            output = Path(directory) / "participant_01.hdf5.partial"
            config = {
                "recording": {
                    "spectator_video_enabled": True,
                    "camera_prim_path": "/World/TestSpectatorCamera",
                    "resolution": [640, 360],
                    "fps": 15,
                    "capture_interval_steps": 2,
                }
            }
            recorder = SynchronizedSpectatorVideoRecorder.from_config(
                output, config, recorder_factory=factory
            )
            self.assertTrue(recorder.setup())
            self.assertEqual(len(backends), 1)
            self.assertEqual(backends[0].kwargs["resolution"], "640,360")
            self.assertEqual(backends[0].kwargs["interval_steps"], 2)
            self.assertEqual(
                backends[0].kwargs["prim_path"], "/World/TestSpectatorCamera"
            )

            self.assertIsNone(recorder.capture(0.10, 1, "trial-1"))
            first = recorder.capture(
                sim_time_s=0.20, control_step=2, trial_id="trial-1"
            )
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.frame_index, 0)
            self.assertEqual(first.trial_frame_index, 0)
            self.assertEqual(first.simulation_time_s, 0.20)
            self.assertEqual(first.control_step, 2)
            self.assertEqual(first.trial_id, "trial-1")

            self.assertIsNone(recorder.capture(0.30, 3, "trial-1"))
            # Per-trial clocks may reset; the session frame index remains global.
            second = recorder.capture(0.05, 1, "trial-2")
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.frame_index, 1)
            self.assertEqual(second.trial_frame_index, 0)
            self.assertEqual(second.trial_id, "trial-2")

            metadata_before_close = recorder.metadata
            self.assertTrue(metadata_before_close["fixed_camera_pose"])
            self.assertFalse(metadata_before_close["condition_identity_visible"])
            self.assertEqual(metadata_before_close["resolution"], [640, 360])
            self.assertEqual(metadata_before_close["fps"], 15)
            self.assertEqual(metadata_before_close["video_start_sim_time_s"], 0.20)
            self.assertEqual(metadata_before_close["video_start_control_step"], 2)
            self.assertEqual(metadata_before_close["video_start_trial_id"], "trial-1")
            self.assertEqual(metadata_before_close["frame_count"], 2)

            recorder.close()
            recorder.close()  # idempotent
            self.assertTrue(backends[0].close_called)
            self.assertTrue(recorder.csv_sidecar_path.exists())
            self.assertTrue(recorder.json_sidecar_path.exists())
            self.assertEqual(
                recorder.csv_sidecar_path.parent, output.resolve().parent
            )

            with recorder.csv_sidecar_path.open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["frame_index"] for row in rows], ["0", "1"])
            self.assertEqual(
                [row["simulation_time_s"] for row in rows], ["0.2", "0.05"]
            )
            self.assertEqual([row["trial_id"] for row in rows], ["trial-1", "trial-2"])

            payload = json.loads(recorder.json_sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["frame_count"], 2)
            self.assertEqual(
                [frame["control_step"] for frame in payload["frames"]], [2, 1]
            )
            self.assertEqual(
                payload["metadata"]["timestamp_semantics"],
                "caller_supplied_simulation_state_rendered_with_delta_time_zero",
            )

    def test_clocks_fail_closed_within_one_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedSpectatorVideoRecorder(
                Path(directory) / "session.hdf5",
                enabled=True,
                capture_interval_steps=1,
                recorder_factory=_FakeVideoBackend,
            )
            self.assertTrue(recorder.setup())
            recorder.capture(0.2, 2, "trial-1")
            with self.assertRaisesRegex(ValueError, "control_step"):
                recorder.capture(0.3, 2, "trial-1")
            with self.assertRaisesRegex(ValueError, "moved backwards"):
                recorder.capture(0.1, 3, "trial-1")
            recorder.close()

    def test_due_frame_silent_backend_failure_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedSpectatorVideoRecorder(
                Path(directory) / "session.hdf5",
                enabled=True,
                capture_interval_steps=2,
                recorder_factory=_SilentFailureVideoBackend,
            )
            self.assertTrue(recorder.setup())
            self.assertIsNone(recorder.capture(0.1, 1, "trial-1"))
            with self.assertRaisesRegex(RuntimeError, "capture cadence"):
                recorder.capture(0.2, 2, "trial-1")
            self.assertTrue(recorder.metadata["capture_error"])
            recorder.close()


if __name__ == "__main__":
    unittest.main()
