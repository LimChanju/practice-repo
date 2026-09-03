"""Live-pose provenance snapshots and fail-closed tracking checks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


TRACKING_UNKNOWN = -1
TRACKING_NOT_TRACKED = 0
TRACKING_TRACKED = 1
ENTITY_NAMES = ("head", "left", "right")
DEFAULT_ALLOWED_SOURCES = {
    "head": frozenset({"hmd_xr_physical"}),
    "left": frozenset(
        {
            "xr_physical",
            "xr_raw_physical",
            "openxr_joint",
            "external_hand_tracking",
        }
    ),
    "right": frozenset(
        {
            "xr_physical",
            "xr_raw_physical",
            "openxr_joint",
            "external_hand_tracking",
        }
    ),
}


@dataclass(frozen=True)
class TrackedPose:
    position_world: np.ndarray
    orientation_wxyz: np.ndarray
    pose_valid: bool
    position_tracked: int
    tracking_status_known: bool
    source_name: str
    source_path: str
    acquisition_monotonic_ns: int
    pose_age_ms: float
    source_switched: bool

    @classmethod
    def from_pose_sample(cls, sample: Any) -> "TrackedPose":
        position = _fixed_finite(getattr(sample, "position_world", None), 3)
        orientation = _fixed_finite(
            getattr(sample, "orientation_wxyz", None), 4
        )
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return cls(
            position_world=(
                np.full(3, np.nan, dtype=np.float64)
                if position is None
                else position
            ),
            orientation_wxyz=orientation,
            pose_valid=bool(getattr(sample, "pose_valid", False)),
            position_tracked=int(
                getattr(sample, "position_tracked", TRACKING_UNKNOWN)
            ),
            tracking_status_known=bool(
                getattr(sample, "tracking_status_known", False)
            ),
            source_name=str(getattr(sample, "source_name", "missing")),
            source_path=str(getattr(sample, "source_path", "")),
            acquisition_monotonic_ns=int(
                getattr(sample, "acquisition_monotonic_ns", 0)
            ),
            pose_age_ms=float(getattr(sample, "pose_age_ms", -1.0)),
            source_switched=bool(getattr(sample, "source_switched", False)),
        )


@dataclass(frozen=True)
class TrackingSnapshot:
    sample_monotonic_ns: int
    head: TrackedPose
    left: TrackedPose
    right: TrackedPose
    valid_mask: np.ndarray
    invalid_reasons: tuple[str, ...]

    def state_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "human_valid_mask": self.valid_mask.astype(np.float32),
            "tracking_sample_monotonic_ns": int(self.sample_monotonic_ns),
            "intentional_human_absence": False,
        }
        for index, name in enumerate(ENTITY_NAMES):
            pose = getattr(self, name)
            if bool(self.valid_mask[index]):
                key = "human_head_pos" if name == "head" else f"human_{name}_hand_pos"
                payload[key] = pose.position_world.astype(np.float32)
            payload[f"{name}_pose_valid"] = bool(pose.pose_valid)
            payload[f"{name}_position_tracked"] = int(pose.position_tracked)
            payload[f"{name}_tracking_status_known"] = bool(
                pose.tracking_status_known
            )
            payload[f"{name}_pose_source"] = pose.source_name
            payload[f"{name}_pose_source_path"] = pose.source_path
            payload[f"{name}_pose_acquisition_monotonic_ns"] = int(
                pose.acquisition_monotonic_ns
            )
            payload[f"{name}_pose_age_ms"] = float(pose.pose_age_ms)
            payload[f"{name}_pose_source_switched"] = bool(pose.source_switched)
        return payload


class TrackingDropout(RuntimeError):
    def __init__(self, reasons: Iterable[str], snapshot: TrackingSnapshot) -> None:
        self.reasons = tuple(str(reason) for reason in reasons)
        self.snapshot = snapshot
        super().__init__("live tracking invalid: " + ", ".join(self.reasons))


class TrackingWatchdog:
    """Validate current live samples without treating a cached pose as presence."""

    def __init__(
        self,
        *,
        required_mask: tuple[int, int, int] = (1, 1, 1),
        max_pose_age_ms: float = 100.0,
        allowed_sources: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        mask = np.asarray(required_mask, dtype=np.int8).reshape(-1)
        if mask.shape != (3,) or not np.all(np.isin(mask, (0, 1))):
            raise ValueError("required_mask must be three exact binary values")
        if not math.isfinite(float(max_pose_age_ms)) or max_pose_age_ms <= 0.0:
            raise ValueError("max_pose_age_ms must be finite and positive")
        self.required_mask = mask
        self.max_pose_age_ms = float(max_pose_age_ms)
        selected = allowed_sources or DEFAULT_ALLOWED_SOURCES
        self.allowed_sources = {
            name: frozenset(str(value) for value in selected.get(name, ()))
            for name in ENTITY_NAMES
        }

    def evaluate(
        self,
        samples: tuple[Any, Any, Any],
        *,
        now_monotonic_ns: int | None = None,
    ) -> TrackingSnapshot:
        if len(samples) != 3:
            raise ValueError("samples must contain head, left, and right")
        now_ns = int(time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns)
        poses = tuple(TrackedPose.from_pose_sample(sample) for sample in samples)
        valid = np.zeros(3, dtype=np.int8)
        reasons: list[str] = []
        for index, (name, pose) in enumerate(zip(ENTITY_NAMES, poses)):
            entity_reasons = self._pose_reasons(name, pose, now_ns=now_ns)
            valid[index] = int(not entity_reasons)
            if bool(self.required_mask[index]):
                reasons.extend(entity_reasons)
        return TrackingSnapshot(
            sample_monotonic_ns=now_ns,
            head=poses[0],
            left=poses[1],
            right=poses[2],
            valid_mask=valid,
            invalid_reasons=tuple(reasons),
        )

    def require_valid(self, snapshot: TrackingSnapshot) -> None:
        missing = self.required_mask.astype(bool) & ~snapshot.valid_mask.astype(bool)
        if np.any(missing):
            raise TrackingDropout(snapshot.invalid_reasons, snapshot)

    def require_current(
        self,
        snapshot: TrackingSnapshot,
        *,
        now_monotonic_ns: int | None = None,
    ) -> None:
        """Re-check a cached observation at the instant it will drive control.

        ``valid_mask`` is true at the snapshot's sampling time.  A policy input
        can nevertheless become stale while inference or other work is queued,
        so actuation-boundary checks must recompute freshness against *now*.
        """

        if not isinstance(snapshot, TrackingSnapshot):
            raise TypeError("snapshot must be a TrackingSnapshot")
        now_ns = int(
            time.monotonic_ns()
            if now_monotonic_ns is None
            else now_monotonic_ns
        )
        reasons: list[str] = []
        valid = np.zeros(3, dtype=np.int8)
        poses = (snapshot.head, snapshot.left, snapshot.right)
        for index, (name, pose) in enumerate(zip(ENTITY_NAMES, poses)):
            entity_reasons = self._pose_reasons(name, pose, now_ns=now_ns)
            valid[index] = int(not entity_reasons)
            if bool(self.required_mask[index]):
                reasons.extend(entity_reasons)
        refreshed = TrackingSnapshot(
            sample_monotonic_ns=snapshot.sample_monotonic_ns,
            head=snapshot.head,
            left=snapshot.left,
            right=snapshot.right,
            valid_mask=valid,
            invalid_reasons=tuple(reasons),
        )
        missing = self.required_mask.astype(bool) & ~valid.astype(bool)
        if np.any(missing):
            raise TrackingDropout(reasons, refreshed)

    def _pose_reasons(
        self,
        name: str,
        pose: TrackedPose,
        *,
        now_ns: int,
    ) -> list[str]:
        reasons: list[str] = []
        prefix = f"{name}:"
        if not pose.pose_valid:
            reasons.append(prefix + "pose_invalid")
        if not np.all(np.isfinite(pose.position_world)):
            reasons.append(prefix + "position_nonfinite")
        if pose.source_name not in self.allowed_sources[name]:
            reasons.append(prefix + "source_not_allowed")
        if not pose.source_path.strip():
            reasons.append(prefix + "source_path_missing")
        if pose.source_switched:
            reasons.append(prefix + "source_switched")
        acquired = int(pose.acquisition_monotonic_ns)
        observed_age_ms: float | None = None
        if acquired <= 0 or acquired > now_ns:
            reasons.append(prefix + "acquisition_time_invalid")
        else:
            observed_age_ms = (now_ns - acquired) / 1e6
            if observed_age_ms > self.max_pose_age_ms:
                reasons.append(prefix + "pose_stale")
        if not math.isfinite(pose.pose_age_ms) or pose.pose_age_ms < 0.0:
            reasons.append(prefix + "source_pose_age_invalid")
        elif pose.pose_age_ms > self.max_pose_age_ms:
            reasons.append(prefix + "source_pose_stale")
        elif (
            observed_age_ms is not None
            and pose.pose_age_ms + observed_age_ms > self.max_pose_age_ms
        ):
            reasons.append(prefix + "effective_pose_stale")
        if pose.position_tracked not in (
            TRACKING_UNKNOWN,
            TRACKING_NOT_TRACKED,
            TRACKING_TRACKED,
        ):
            reasons.append(prefix + "tracking_status_invalid")
        elif pose.tracking_status_known:
            if pose.position_tracked != TRACKING_TRACKED:
                reasons.append(prefix + "not_position_tracked")
        elif pose.position_tracked != TRACKING_UNKNOWN:
            reasons.append(prefix + "tracking_status_inconsistent")
        return reasons


def _fixed_finite(value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < size or not np.all(np.isfinite(array[:size])):
        return None
    return array[:size].copy()
