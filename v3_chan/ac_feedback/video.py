"""Optional fixed-camera video with an auditable simulation-time sidecar.

The Isaac/Replicator dependency is imported lazily only when recording is
enabled.  Consequently the disabled path and all synchronization bookkeeping
remain testable with ordinary Python and create no output artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO


VIDEO_SYNC_SCHEMA_VERSION = "ac_spectator_video_sync_v1"
DEFAULT_CAMERA_EYE_M = (1.35, -1.15, 1.75)
DEFAULT_CAMERA_TARGET_M = (0.45, 0.0, 1.05)
DEFAULT_CAMERA_UP = (0.0, 0.0, 1.0)


def _finite_vec3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain three finite values")
    return values


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result != value or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sidecar_base(output_path: Path) -> Path:
    """Remove collection suffixes while retaining the requested directory."""

    base = output_path
    if base.suffix == ".partial":
        base = base.with_suffix("")
    if base.suffix.lower() in (".h5", ".hdf5"):
        base = base.with_suffix("")
    return base.parent / f"{base.name}_spectator"


@dataclass(frozen=True)
class VideoFrameRecord:
    """Clock mapping for one frame accepted by the underlying recorder."""

    frame_index: int
    trial_frame_index: int
    simulation_time_s: float
    control_step: int
    trial_id: str
    monotonic_ns: int
    unix_ns: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SynchronizedSpectatorVideoRecorder:
    """Wrap ``HRIOverviewVideoRecorder`` with frame-clock provenance.

    ``capture`` returns ``None`` on interval-skipped calls and a
    :class:`VideoFrameRecord` only when the backend reports a newly captured
    frame.  The CSV is flushed after every frame for crash recovery; the JSON
    sidecar is atomically finalized by :meth:`close`.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        enabled: bool = False,
        camera_prim_path: str = "/World/ACFeedbackSpectatorCamera",
        resolution: Sequence[int] = (1280, 720),
        fps: int = 20,
        capture_interval_steps: int = 3,
        eye_m: Sequence[float] = DEFAULT_CAMERA_EYE_M,
        target_m: Sequence[float] = DEFAULT_CAMERA_TARGET_M,
        recorder_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.enabled = bool(enabled)
        self.camera_prim_path = str(camera_prim_path).strip()
        if not self.camera_prim_path.startswith("/"):
            raise ValueError("camera_prim_path must be an absolute USD prim path")
        dimensions = tuple(resolution)
        if len(dimensions) != 2:
            raise ValueError("resolution must contain width and height")
        self.resolution = (
            _positive_int(dimensions[0], name="resolution width"),
            _positive_int(dimensions[1], name="resolution height"),
        )
        self.fps = _positive_int(fps, name="fps")
        self.capture_interval_steps = _positive_int(
            capture_interval_steps, name="capture_interval_steps"
        )
        self.eye_m = _finite_vec3(eye_m, name="eye_m")
        self.target_m = _finite_vec3(target_m, name="target_m")
        if self.eye_m == self.target_m:
            raise ValueError("camera eye and target must differ")
        self.up = DEFAULT_CAMERA_UP
        self._recorder_factory = recorder_factory

        base = _sidecar_base(self.output_path)
        self.record_dir = base.parent / f"{base.name}_frames"
        self.mp4_path = base.with_suffix(".mp4")
        self.csv_sidecar_path = base.with_name(f"{base.name}_frames.csv")
        self.json_sidecar_path = base.with_name(f"{base.name}_frames.json")

        self._backend: Any = None
        self._csv_stream: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._frames: list[VideoFrameRecord] = []
        self._trial_frame_counts: dict[str, int] = {}
        self._setup_called = False
        self._available = False
        self._closed = False
        self._capture_call_count = 0
        self._last_trial_id = ""
        self._last_control_step = -1
        self._last_simulation_time_s = -1.0
        self._start_simulation_time_s: Optional[float] = None
        self._start_control_step: Optional[int] = None
        self._start_trial_id = ""
        self._setup_error = ""
        self._capture_error = ""

    @classmethod
    def from_config(
        cls,
        output_path: str | Path,
        config: Mapping[str, Any],
        *,
        recorder_factory: Optional[Callable[..., Any]] = None,
    ) -> "SynchronizedSpectatorVideoRecorder":
        """Construct from either the full pilot config or its recording block."""

        recording = config.get("recording", config)
        if not isinstance(recording, Mapping):
            raise ValueError("recording configuration must be a mapping")
        return cls(
            output_path,
            enabled=bool(recording.get("spectator_video_enabled", False)),
            camera_prim_path=str(
                recording.get("camera_prim_path", "/World/ACFeedbackSpectatorCamera")
            ),
            resolution=recording.get("resolution", (1280, 720)),
            fps=recording.get("fps", 20),
            capture_interval_steps=recording.get("capture_interval_steps", 3),
            eye_m=recording.get("eye_m", DEFAULT_CAMERA_EYE_M),
            target_m=recording.get("target_m", DEFAULT_CAMERA_TARGET_M),
            recorder_factory=recorder_factory,
        )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def frame_records(self) -> tuple[VideoFrameRecord, ...]:
        return tuple(self._frames)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": VIDEO_SYNC_SCHEMA_VERSION,
            "enabled": self.enabled,
            "available": self._available,
            "fixed_camera_pose": True,
            "condition_identity_visible": False,
            "camera_prim_path": self.camera_prim_path,
            "camera_eye_m": list(self.eye_m),
            "camera_target_m": list(self.target_m),
            "camera_up": list(self.up),
            "resolution": list(self.resolution),
            "fps": self.fps,
            "capture_interval_steps": self.capture_interval_steps,
            "video_start_sim_time_s": self._start_simulation_time_s,
            "video_start_control_step": self._start_control_step,
            "video_start_trial_id": self._start_trial_id,
            "frame_count": len(self._frames),
            "capture_call_count": self._capture_call_count,
            "timestamp_semantics": (
                "caller_supplied_simulation_state_rendered_with_delta_time_zero"
            ),
            "record_dir": str(self.record_dir),
            "mp4_path": str(self.mp4_path),
            "frame_csv_path": str(self.csv_sidecar_path),
            "frame_json_path": str(self.json_sidecar_path),
            "setup_error": self._setup_error,
            "capture_error": self._capture_error,
        }

    def setup(self) -> bool:
        """Initialize the fixed camera and sidecars; no-op when disabled."""

        if self._closed:
            raise RuntimeError("video recorder is already closed")
        if self._setup_called:
            return self._available
        self._setup_called = True
        if not self.enabled:
            return False
        try:
            factory = self._recorder_factory or self._default_recorder_factory()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            backend = factory(
                prim_path=self.camera_prim_path,
                enabled=True,
                record_dir=str(self.record_dir),
                resolution=f"{self.resolution[0]},{self.resolution[1]}",
                interval_steps=self.capture_interval_steps,
                fps=self.fps,
                eye=",".join(str(value) for value in self.eye_m),
                target=",".join(str(value) for value in self.target_m),
                mp4_path=str(self.mp4_path),
            )
            setup_result = backend.setup()
            if isinstance(setup_result, bool):
                ready = setup_result
            elif hasattr(backend, "_writer"):
                ready = getattr(backend, "_writer") is not None
            else:
                ready = True
            if not ready:
                self._setup_error = "underlying video recorder is unavailable"
                try:
                    backend.close()
                except Exception:
                    pass
                return False
            self._backend = backend
            self._open_csv_sidecar()
            self._available = True
            return True
        except Exception as error:
            self._setup_error = f"{type(error).__name__}: {error}"
            self._available = False
            return False

    def capture(
        self,
        sim_time_s: float,
        control_step: int,
        trial_id: str,
    ) -> Optional[VideoFrameRecord]:
        """Capture when due and return the corresponding synchronized record."""

        if not self.enabled or not self._available or self._closed:
            return None
        sim_time = float(sim_time_s)
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        if isinstance(control_step, bool):
            raise ValueError("control_step must be a non-negative integer")
        step = int(control_step)
        if step != control_step or step < 0:
            raise ValueError("control_step must be a non-negative integer")
        trial = str(trial_id).strip()
        if not trial:
            raise ValueError("trial_id must be non-empty")
        if trial == self._last_trial_id:
            if step <= self._last_control_step:
                raise ValueError("control_step must increase within a trial")
            if sim_time < self._last_simulation_time_s:
                raise ValueError("simulation_time_s moved backwards within a trial")
        self._last_trial_id = trial
        self._last_control_step = step
        self._last_simulation_time_s = sim_time
        self._capture_call_count += 1

        before = getattr(self._backend, "_capture_count", None)
        try:
            capture_result = self._backend.capture()
        except Exception as error:
            self._capture_error = f"{type(error).__name__}: {error}"
            raise RuntimeError("spectator video capture failed") from error
        after = getattr(self._backend, "_capture_count", None)
        if before is not None and after is not None:
            captured = int(after) > int(before)
        elif isinstance(capture_result, bool):
            captured = capture_result
        else:
            # A custom backend without a count must explicitly return True.
            captured = False
        capture_due = (
            self._capture_call_count % self.capture_interval_steps == 0
        )
        if captured != capture_due:
            self._capture_error = (
                "backend capture cadence/failure mismatch: "
                f"call={self._capture_call_count}, due={capture_due}, "
                f"captured={captured}"
            )
            raise RuntimeError(self._capture_error)
        if not captured:
            return None

        trial_frame_index = self._trial_frame_counts.get(trial, 0)
        record = VideoFrameRecord(
            frame_index=len(self._frames),
            trial_frame_index=trial_frame_index,
            simulation_time_s=sim_time,
            control_step=step,
            trial_id=trial,
            monotonic_ns=time.monotonic_ns(),
            unix_ns=time.time_ns(),
        )
        self._trial_frame_counts[trial] = trial_frame_index + 1
        self._frames.append(record)
        if self._start_simulation_time_s is None:
            self._start_simulation_time_s = sim_time
            self._start_control_step = step
            self._start_trial_id = trial
        self._append_csv(record)
        return record

    def close(self) -> None:
        """Finalize encoding and JSON mapping; safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        close_error: Optional[BaseException] = None
        if self._backend is not None:
            try:
                self._backend.close()
            except BaseException as error:  # preserve sidecars before propagating
                close_error = error
        if self._csv_stream is not None:
            try:
                self._csv_stream.flush()
                os.fsync(self._csv_stream.fileno())
            finally:
                self._csv_stream.close()
                self._csv_stream = None
                self._csv_writer = None
        if self.enabled and self._setup_called:
            self._write_json_sidecar()
        self._available = False
        if close_error is not None:
            raise RuntimeError("spectator video close failed") from close_error

    def _open_csv_sidecar(self) -> None:
        self._csv_stream = self.csv_sidecar_path.open("w", encoding="utf-8", newline="")
        fieldnames = tuple(VideoFrameRecord.__dataclass_fields__)
        self._csv_writer = csv.DictWriter(self._csv_stream, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._csv_stream.flush()

    def _append_csv(self, record: VideoFrameRecord) -> None:
        if self._csv_stream is None or self._csv_writer is None:
            raise RuntimeError("video CSV sidecar is not open")
        self._csv_writer.writerow(record.as_dict())
        self._csv_stream.flush()

    def _write_json_sidecar(self) -> None:
        payload = {
            "metadata": self.metadata,
            "frames": [record.as_dict() for record in self._frames],
        }
        temporary = self.json_sidecar_path.with_name(
            f".{self.json_sidecar_path.name}.tmp"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.json_sidecar_path)

    @staticmethod
    def _default_recorder_factory() -> Callable[..., Any]:
        # Importing hri_video_recorder imports pxr, so keep this out of the
        # default-disabled and unit-test paths.
        from v3_chan.hri_video_recorder import HRIOverviewVideoRecorder

        return HRIOverviewVideoRecorder


__all__ = [
    "DEFAULT_CAMERA_EYE_M",
    "DEFAULT_CAMERA_TARGET_M",
    "SynchronizedSpectatorVideoRecorder",
    "VIDEO_SYNC_SCHEMA_VERSION",
    "VideoFrameRecord",
]
