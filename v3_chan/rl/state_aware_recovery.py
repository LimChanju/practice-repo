from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np


STATE_AWARE_RECOVERY_SCHEMA = "state_aware_pick_place_recovery_bridge_v3"

RecoveryMode = Literal["inactive", "place", "regrasp"]
RecoveryStage = Literal[
    "inactive",
    "place_lift",
    "place_transport",
    "place_descend",
    "regrasp_prepose",
]


@dataclass(frozen=True)
class StateAwareRecoveryConfig:
    """Frozen, expert-compatible anchors for post-CBF task recovery.

    The controller target offsets are taken from the release-capable expert
    collector.  The bridge deliberately stops at a state represented in the
    BC data: event-1 pre-grasp for re-grasp and the physically placed event-6
    endpoint for placement.  The frozen BC retains ownership of the
    subsequent grasp or release/settle primitive.
    """

    enabled: bool = False
    controller_target_y_offset_m: float = 0.005
    prepose_height_m: float = 0.17425
    observed_ee_frame_offset_x_m: float = 0.0
    observed_ee_frame_offset_y_m: float = 0.040
    observed_ee_frame_offset_z_m: float = 0.042
    # The bridge hands back at the physically placed event-6 anchor.  This is
    # intentionally later than the pre-place anchor: dev diagnostics showed
    # that EE-only pre-place alignment could leave CBF-shifted joint states
    # outside the frozen BC manifold and induce 7--10 cm lateral drift.
    place_handoff_progress: float = 1.0
    placement_z_offset_m: float = 0.015
    placement_xy_tolerance_m: float = 0.020
    placement_z_tolerance_m: float = 0.010
    minimum_transport_clearance_m: float = 0.10
    anchor_position_tolerance_m: float = 0.035
    maximum_cube_speed_mps: float = 0.05
    maximum_joint_speed_radps: float = 0.25
    anchor_confirmation_steps: int = 6
    maximum_recovery_steps: int = 600

    def validated(self) -> "StateAwareRecoveryConfig":
        if not isinstance(self.enabled, bool):
            raise ValueError("state-aware recovery enabled must be boolean")
        finite_values = (
            self.controller_target_y_offset_m,
            self.prepose_height_m,
            self.observed_ee_frame_offset_x_m,
            self.observed_ee_frame_offset_y_m,
            self.observed_ee_frame_offset_z_m,
            self.place_handoff_progress,
            self.placement_z_offset_m,
            self.placement_xy_tolerance_m,
            self.placement_z_tolerance_m,
            self.minimum_transport_clearance_m,
            self.anchor_position_tolerance_m,
            self.maximum_cube_speed_mps,
            self.maximum_joint_speed_radps,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("state-aware recovery thresholds must be finite")
        positive_values = (
            self.prepose_height_m,
            self.minimum_transport_clearance_m,
            self.anchor_position_tolerance_m,
            self.maximum_cube_speed_mps,
            self.maximum_joint_speed_radps,
            self.placement_xy_tolerance_m,
            self.placement_z_tolerance_m,
        )
        if any(float(value) <= 0.0 for value in positive_values):
            raise ValueError("state-aware recovery thresholds must be positive")
        if not math.isclose(
            float(self.place_handoff_progress), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "v3 place handoff progress must equal the placed endpoint (1.0)"
            )
        if float(self.placement_z_offset_m) < 0.0:
            raise ValueError("placement z offset must be non-negative")
        if float(self.minimum_transport_clearance_m) >= float(
            self.prepose_height_m
        ):
            raise ValueError(
                "minimum transport clearance must be below prepose height"
            )
        for name, value in (
            ("anchor_confirmation_steps", self.anchor_confirmation_steps),
            ("maximum_recovery_steps", self.maximum_recovery_steps),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": STATE_AWARE_RECOVERY_SCHEMA, **asdict(self)}

    @property
    def observed_ee_frame_offset_m(self) -> np.ndarray:
        return np.array(
            [
                self.observed_ee_frame_offset_x_m,
                self.observed_ee_frame_offset_y_m,
                self.observed_ee_frame_offset_z_m,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class StateAwareRecoveryEvidence:
    ee_position_m: Sequence[float] | np.ndarray
    cube_position_m: Sequence[float] | np.ndarray
    place_target_position_m: Sequence[float] | np.ndarray
    cube_speed_mps: float
    joint_speed_radps: float
    has_grasped_cube: bool
    cbf_clear: bool

    def validated(self) -> "StateAwareRecoveryEvidence":
        for name, value in (
            ("ee_position_m", self.ee_position_m),
            ("cube_position_m", self.cube_position_m),
            ("place_target_position_m", self.place_target_position_m),
        ):
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size < 3 or not np.all(np.isfinite(array[:3])):
                raise ValueError(f"{name} must contain a finite 3D position")
        for name, value in (
            ("cube_speed_mps", self.cube_speed_mps),
            ("joint_speed_radps", self.joint_speed_radps),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.has_grasped_cube, bool):
            raise ValueError("has_grasped_cube must be boolean")
        if not isinstance(self.cbf_clear, bool):
            raise ValueError("cbf_clear must be boolean")
        return self

    def vector(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=float).reshape(-1)[:3].copy()


@dataclass
class StateAwareRecoveryState:
    active: bool = False
    mode: RecoveryMode = "inactive"
    stage: RecoveryStage = "inactive"
    source_event: int = -1
    source_progress: float = 0.0
    lift_anchor_x_m: float = 0.0
    lift_anchor_y_m: float = 0.0
    ready_streak: int = 0
    active_steps: int = 0
    effective_control_steps: int = 0
    cbf_blocked_steps: int = 0
    stage_steps: int = 0
    plan_id: int = 0
    activation_count: int = 0
    completion_count: int = 0
    replan_count: int = 0
    timeout_count: int = 0
    last_reason: str = "reset"
    last_target_position_m: tuple[float, float, float] | None = None
    last_expected_ee_position_m: tuple[float, float, float] | None = None
    last_anchor_error_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateAwareRecoveryDecision:
    active: bool
    plan_id: int
    mode: RecoveryMode
    stage: RecoveryStage
    target_position_m: np.ndarray | None
    desired_gripper_closed: bool | None
    anchor_ready: bool
    ready_streak: int
    anchor_error_m: float | None
    handoff: bool
    handoff_event: int | None
    handoff_progress: float | None
    stage_changed: bool
    timed_out: bool
    reason: str


class StateAwareRecoveryBridge:
    """Deterministic post-CBF bridge back to a frozen BC anchor state."""

    def __init__(self, config: StateAwareRecoveryConfig) -> None:
        self.config = config.validated()
        self.state = StateAwareRecoveryState()

    def reset(self) -> None:
        self.state = StateAwareRecoveryState()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.state = StateAwareRecoveryState(**dict(state))

    def request(
        self,
        mode: Literal["place", "regrasp"],
        evidence: StateAwareRecoveryEvidence,
        *,
        source_event: int,
        source_progress: float,
    ) -> None:
        if not self.config.enabled:
            return
        evidence = evidence.validated()
        if mode not in ("place", "regrasp"):
            raise ValueError(f"Unsupported recovery mode: {mode!r}")

        # The strict FSM owns grasp/loss confirmation.  Do not reinterpret a
        # confirmed request from a single instantaneous width-distance sample:
        # doing so could turn a transient false negative into an unsafe OPEN.
        requested_mode: Literal["place", "regrasp"] = mode
        self.state.plan_id += 1
        if self.state.active:
            self.state.replan_count += 1
            self.state.ready_streak = 0
            if self.state.mode == requested_mode:
                self.state.last_reason = "recovery_request_reconfirmed"
                return
        else:
            self.state.activation_count += 1
            self.state.active_steps = 0
            self.state.effective_control_steps = 0
            self.state.cbf_blocked_steps = 0

        cube = evidence.vector("cube_position_m")
        place = evidence.vector("place_target_position_m")
        self.state.active = True
        self.state.mode = requested_mode
        self.state.source_event = int(source_event)
        self.state.source_progress = float(source_progress)
        self.state.ready_streak = 0
        self.state.stage_steps = 0
        self.state.lift_anchor_x_m = float(cube[0])
        self.state.lift_anchor_y_m = float(cube[1])
        if requested_mode == "regrasp":
            self.state.stage = "regrasp_prepose"
            self.state.last_reason = "regrasp_recovery_requested"
        elif cube[2] < place[2] + float(
            self.config.minimum_transport_clearance_m
        ):
            self.state.stage = "place_lift"
            self.state.last_reason = "place_lift_recovery_requested"
        else:
            self.state.stage = "place_transport"
            self.state.last_reason = "place_transport_recovery_requested"

    def cancel_for_success(self) -> None:
        if self.state.active:
            self.state.last_reason = "terminal_success_cancelled_recovery"
        self.state.active = False
        self.state.mode = "inactive"
        self.state.stage = "inactive"
        self.state.ready_streak = 0

    def defer_handoff(self, reason: str) -> None:
        """Keep bridge ownership when the post-action state invalidates handoff."""

        if not self.state.active:
            raise RuntimeError("cannot defer an inactive recovery bridge")
        self.state.ready_streak = 0
        self.state.last_reason = str(reason)

    def accept_handoff(self) -> None:
        """Commit a handoff only after the environment revalidates it."""

        if not self.state.active:
            raise RuntimeError("cannot accept an inactive recovery bridge")
        mode = self.state.mode
        self.state.completion_count += 1
        self.state.active = False
        self.state.mode = "inactive"
        self.state.stage = "inactive"
        self.state.ready_streak = 0
        self.state.last_reason = (
            "place_endpoint_handoff"
            if mode == "place"
            else "regrasp_prepose_handoff"
        )

    def anchor_is_ready(
        self,
        evidence: StateAwareRecoveryEvidence,
        target_position_m: Sequence[float] | np.ndarray,
    ) -> tuple[bool, float]:
        """Revalidate a handoff against the post-action physical state."""

        evidence = evidence.validated()
        target = np.asarray(target_position_m, dtype=float).reshape(-1)
        if target.size < 3 or not np.all(np.isfinite(target[:3])):
            raise ValueError("recovery target must contain a finite 3D position")
        expected_ee = target[:3] + self.config.observed_ee_frame_offset_m
        anchor_error = float(
            np.linalg.norm(evidence.vector("ee_position_m") - expected_ee)
        )
        ready = bool(
            evidence.cbf_clear
            and anchor_error
            <= float(self.config.anchor_position_tolerance_m)
            and float(evidence.cube_speed_mps)
            <= float(self.config.maximum_cube_speed_mps)
            and float(evidence.joint_speed_radps)
            <= float(self.config.maximum_joint_speed_radps)
        )
        if self.state.stage == "place_descend":
            cube = evidence.vector("cube_position_m")
            place = evidence.vector("place_target_position_m")
            desired_cube_z = place[2] + float(
                self.config.placement_z_offset_m
            )
            ready = bool(
                ready
                and float(np.linalg.norm(cube[:2] - place[:2]))
                <= float(self.config.placement_xy_tolerance_m)
                and abs(float(cube[2]) - desired_cube_z)
                <= float(self.config.placement_z_tolerance_m)
            )
        return ready, anchor_error

    def update(
        self, evidence: StateAwareRecoveryEvidence
    ) -> StateAwareRecoveryDecision:
        evidence = evidence.validated()
        if not self.config.enabled or not self.state.active:
            return self._inactive_decision("recovery_inactive")

        self.state.active_steps += 1
        self.state.stage_steps += 1
        if evidence.cbf_clear:
            self.state.effective_control_steps += 1
        else:
            self.state.cbf_blocked_steps += 1
        if self.state.effective_control_steps >= int(
            self.config.maximum_recovery_steps
        ):
            self.state.timeout_count += 1
            self.state.active = False
            self.state.mode = "inactive"
            self.state.stage = "inactive"
            self.state.ready_streak = 0
            self.state.last_reason = "state_aware_recovery_timeout"
            return self._inactive_decision(
                "state_aware_recovery_timeout", timed_out=True
            )

        target = self._target_position(evidence)
        expected_ee = target + self.config.observed_ee_frame_offset_m
        anchor_ready, anchor_error = self.anchor_is_ready(evidence, target)
        mode_consistent = bool(
            (self.state.mode == "place" and evidence.has_grasped_cube)
            or (self.state.mode == "regrasp" and not evidence.has_grasped_cube)
        )
        anchor_ready = bool(anchor_ready and mode_consistent)
        if self.state.stage == "place_lift":
            cube = evidence.vector("cube_position_m")
            place = evidence.vector("place_target_position_m")
            anchor_ready = bool(
                anchor_ready
                and cube[2]
                >= place[2] + float(self.config.minimum_transport_clearance_m)
            )
        self.state.ready_streak = (
            self.state.ready_streak + 1 if anchor_ready else 0
        )
        self.state.last_target_position_m = tuple(float(value) for value in target)
        self.state.last_expected_ee_position_m = tuple(
            float(value) for value in expected_ee
        )
        self.state.last_anchor_error_m = anchor_error

        confirmed = self.state.ready_streak >= int(
            self.config.anchor_confirmation_steps
        )
        if confirmed and self.state.stage == "place_lift":
            self.state.stage = "place_transport"
            self.state.stage_steps = 0
            self.state.ready_streak = 0
            self.state.last_reason = "place_lift_anchor_confirmed"
            next_target = self._target_position(evidence)
            return self._active_decision(
                target=next_target,
                desired_gripper_closed=True,
                anchor_ready=True,
                anchor_error_m=anchor_error,
                stage_changed=True,
                reason="place_lift_anchor_confirmed",
            )
        if confirmed and self.state.stage == "place_transport":
            self.state.stage = "place_descend"
            self.state.stage_steps = 0
            self.state.ready_streak = 0
            self.state.last_reason = "place_transport_anchor_confirmed"
            next_target = self._target_position(evidence)
            return self._active_decision(
                target=next_target,
                desired_gripper_closed=True,
                anchor_ready=True,
                anchor_error_m=anchor_error,
                stage_changed=True,
                reason="place_transport_anchor_confirmed",
            )
        if confirmed:
            mode = self.state.mode
            handoff_event = 6 if mode == "place" else 1
            handoff_reason = (
                "place_endpoint_handoff" if mode == "place" else "regrasp_prepose_handoff"
            )
            return StateAwareRecoveryDecision(
                active=True,
                plan_id=int(self.state.plan_id),
                mode=mode,
                stage=self.state.stage,
                target_position_m=target.copy(),
                desired_gripper_closed=(mode == "place"),
                anchor_ready=True,
                ready_streak=int(self.config.anchor_confirmation_steps),
                anchor_error_m=anchor_error,
                handoff=True,
                handoff_event=handoff_event,
                handoff_progress=(
                    float(self.config.place_handoff_progress)
                    if mode == "place"
                    else 0.0
                ),
                stage_changed=True,
                timed_out=False,
                reason=handoff_reason,
            )

        self.state.last_reason = (
            "recovery_anchor_confirming"
            if anchor_ready
            else "recovery_anchor_seeking"
        )
        return self._active_decision(
            target=target,
            desired_gripper_closed=(self.state.mode == "place"),
            anchor_ready=anchor_ready,
            anchor_error_m=anchor_error,
            reason=str(self.state.last_reason),
        )

    def _target_position(
        self, evidence: StateAwareRecoveryEvidence
    ) -> np.ndarray:
        cube = evidence.vector("cube_position_m")
        place = evidence.vector("place_target_position_m")
        y_offset = float(self.config.controller_target_y_offset_m)
        height = float(self.config.prepose_height_m)
        if self.state.stage == "place_lift":
            return np.array(
                [
                    self.state.lift_anchor_x_m,
                    self.state.lift_anchor_y_m + y_offset,
                    place[2] + height,
                ],
                dtype=float,
            )
        if self.state.stage == "place_transport":
            return place + np.array([0.0, y_offset, height], dtype=float)
        if self.state.stage == "place_descend":
            # Preserve the live grasp transform instead of assuming that the
            # cube is centered under the nominal EE frame.  RMPflow targets
            # its controller frame while observations report an offset EE
            # frame, hence the explicit frame-offset subtraction.
            ee = evidence.vector("ee_position_m")
            desired_cube = place + np.array(
                [0.0, 0.0, float(self.config.placement_z_offset_m)],
                dtype=float,
            )
            return (
                ee
                + (desired_cube - cube)
                - self.config.observed_ee_frame_offset_m
            )
        if self.state.stage == "regrasp_prepose":
            return cube + np.array([0.0, y_offset, height], dtype=float)
        raise RuntimeError(f"No target for inactive recovery stage {self.state.stage!r}")

    def _active_decision(
        self,
        *,
        target: np.ndarray,
        desired_gripper_closed: bool,
        anchor_ready: bool,
        anchor_error_m: float,
        reason: str,
        stage_changed: bool = False,
    ) -> StateAwareRecoveryDecision:
        return StateAwareRecoveryDecision(
            active=True,
            plan_id=int(self.state.plan_id),
            mode=self.state.mode,
            stage=self.state.stage,
            target_position_m=np.asarray(target, dtype=float).reshape(3).copy(),
            desired_gripper_closed=bool(desired_gripper_closed),
            anchor_ready=bool(anchor_ready),
            ready_streak=int(self.state.ready_streak),
            anchor_error_m=float(anchor_error_m),
            handoff=False,
            handoff_event=None,
            handoff_progress=None,
            stage_changed=bool(stage_changed),
            timed_out=False,
            reason=str(reason),
        )

    def _inactive_decision(
        self, reason: str, *, timed_out: bool = False
    ) -> StateAwareRecoveryDecision:
        return StateAwareRecoveryDecision(
            active=False,
            plan_id=int(self.state.plan_id),
            mode="inactive",
            stage="inactive",
            target_position_m=None,
            desired_gripper_closed=None,
            anchor_ready=False,
            ready_streak=0,
            anchor_error_m=None,
            handoff=False,
            handoff_event=None,
            handoff_progress=None,
            stage_changed=False,
            timed_out=bool(timed_out),
            reason=str(reason),
        )


__all__ = [
    "STATE_AWARE_RECOVERY_SCHEMA",
    "RecoveryMode",
    "RecoveryStage",
    "StateAwareRecoveryBridge",
    "StateAwareRecoveryConfig",
    "StateAwareRecoveryDecision",
    "StateAwareRecoveryEvidence",
    "StateAwareRecoveryState",
]
