from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


STRICT_TASK_SEMANTICS_SCHEMA = "physical_event_driven_pick_place_v5"


@dataclass(frozen=True)
class StrictTaskSemanticsConfig:
    """Frozen physical predicates for the corrected pick-and-place FSM.

    Event identifiers stay compatible with the historical 10-event policy
    observation.  Only the conditions that permit an event transition change.
    """

    enabled: bool = False
    grasp_close_distance_m: float = 0.075
    grasp_confirmation_steps: int = 6
    grasp_confirmation_timeout_steps: int = 60
    grasp_approach_timeout_steps: int = 320
    grasp_lost_confirmation_steps: int = 6
    maximum_grasp_retries: int = 1
    place_xy_tolerance_m: float = 0.04
    place_z_tolerance_m: float = 0.03
    place_max_cube_speed_mps: float = 0.05
    minimum_cube_lift_m: float = 0.05
    place_confirmation_steps: int = 6
    place_confirmation_timeout_steps: int = 240
    maximum_place_reentries: int = 2
    release_settle_confirmation_steps: int = 12
    release_settle_timeout_steps: int = 240
    cbf_pause_enter_norm_radps: float = 0.05
    cbf_pause_exit_norm_radps: float = 0.01
    cbf_pause_exit_confirmation_steps: int = 6
    state_aware_recovery: bool = False

    def validated(self) -> "StrictTaskSemanticsConfig":
        if not isinstance(self.enabled, bool):
            raise ValueError("strict task semantics enabled must be boolean")
        if not isinstance(self.state_aware_recovery, bool):
            raise ValueError("state-aware recovery must be boolean")
        if self.state_aware_recovery and not self.enabled:
            raise ValueError("state-aware recovery requires strict task semantics")
        nonnegative = (
            self.grasp_close_distance_m,
            self.place_xy_tolerance_m,
            self.place_z_tolerance_m,
            self.place_max_cube_speed_mps,
            self.minimum_cube_lift_m,
            self.cbf_pause_enter_norm_radps,
            self.cbf_pause_exit_norm_radps,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in nonnegative):
            raise ValueError("strict task semantic thresholds must be finite and non-negative")
        if float(self.cbf_pause_exit_norm_radps) > float(
            self.cbf_pause_enter_norm_radps
        ):
            raise ValueError("CBF pause exit threshold must not exceed enter threshold")
        positive_ints = (
            self.grasp_confirmation_steps,
            self.grasp_confirmation_timeout_steps,
            self.grasp_approach_timeout_steps,
            self.grasp_lost_confirmation_steps,
            self.place_confirmation_steps,
            self.place_confirmation_timeout_steps,
            self.release_settle_confirmation_steps,
            self.release_settle_timeout_steps,
            self.cbf_pause_exit_confirmation_steps,
        )
        if any(
            isinstance(value, bool) or int(value) != value or int(value) <= 0
            for value in positive_ints
        ):
            raise ValueError("strict task confirmation/timeout steps must be positive integers")
        for name, value in (
            ("maximum_grasp_retries", self.maximum_grasp_retries),
            ("maximum_place_reentries", self.maximum_place_reentries),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": STRICT_TASK_SEMANTICS_SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class TaskPhysicalEvidence:
    ee_cube_distance_m: float
    cube_target_xy_error_m: float
    cube_target_z_error_m: float
    cube_speed_mps: float
    cube_z_m: float
    grasp_candidate: bool
    cbf_intervention_norm_radps: float

    def validated(self) -> "TaskPhysicalEvidence":
        values = (
            self.ee_cube_distance_m,
            self.cube_target_xy_error_m,
            self.cube_target_z_error_m,
            self.cube_speed_mps,
            self.cube_z_m,
            self.cbf_intervention_norm_radps,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("task physical evidence must be finite and non-negative")
        if not isinstance(self.grasp_candidate, bool):
            raise ValueError("grasp_candidate must be boolean")
        return self


@dataclass
class StrictTaskSemanticsState:
    initial_cube_z_m: float = 0.0
    maximum_cube_lift_m: float = 0.0
    grasp_observed: bool = False
    grasp_confirmation_streak: int = 0
    grasp_lost_streak: int = 0
    grasp_approach_wait_steps: int = 0
    grasp_confirmation_wait_steps: int = 0
    grasp_retry_count: int = 0
    place_ready_streak: int = 0
    place_ready_latched: bool = False
    place_wait_steps: int = 0
    place_reentry_count: int = 0
    release_settle_streak: int = 0
    release_wait_steps: int = 0
    cbf_hold_latched: bool = False
    cbf_clear_streak: int = 0
    cbf_pause_steps: int = 0
    cbf_intervention_observed: bool = False
    phase_hold_total_steps: int = 0
    phase_reentry_count: int = 0
    recovery_request_count: int = 0
    place_recovery_started: bool = False
    place_recovery_in_progress: bool = False
    place_recovery_completed: bool = False
    pending_retry_open: bool = False
    release_command_applied: bool = False
    success_latched: bool = False
    failure_reason: str = ""
    last_transition_reason: str = "reset"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictTaskPhaseDecision:
    event: int
    progress: float
    reason: str
    phase_changed: bool
    held: bool
    retry_started: bool
    reentry: bool
    success_latched: bool
    failure_reason: str
    recovery_request: str = "none"


class StrictTaskSemanticsController:
    """State-conditioned successor for the historical time-driven FSM."""

    def __init__(self, config: StrictTaskSemanticsConfig) -> None:
        self.config = config.validated()
        self.state = StrictTaskSemanticsState()

    def reset(self, *, initial_cube_z_m: float) -> None:
        initial_z = float(initial_cube_z_m)
        if not math.isfinite(initial_z):
            raise ValueError("initial cube z must be finite")
        self.state = StrictTaskSemanticsState(initial_cube_z_m=initial_z)

    def restore_state(self, state: Mapping[str, Any]) -> None:
        restored = StrictTaskSemanticsState(**dict(state))
        if restored.success_latched and not restored.release_command_applied:
            raise ValueError(
                "A strict-task success snapshot must include an applied release command"
            )
        if restored.success_latched and restored.failure_reason:
            raise ValueError(
                "A strict-task snapshot cannot be both successful and failed"
            )
        self.state = restored

    def consume_pending_retry_open(self) -> bool:
        pending = bool(self.state.pending_retry_open)
        self.state.pending_retry_open = False
        return pending

    def release_command_allowed(self) -> bool:
        return bool(
            self.state.place_ready_latched
            and not self.state.release_command_applied
            and not self.state.cbf_hold_latched
            and not self.state.failure_reason
        )

    def notify_gripper_command_applied(
        self,
        *,
        event: int,
        command: str | None,
        grasp_candidate_before_command: bool,
    ) -> bool:
        """Latch an actual guarded release only after robot.apply_action succeeds."""

        if command != "open" or int(event) != 7:
            return False
        if not isinstance(grasp_candidate_before_command, bool):
            raise ValueError("grasp candidate before release must be boolean")
        if not grasp_candidate_before_command:
            self.state.last_transition_reason = "release_open_after_grasp_loss_rejected"
            return False
        if not self.release_command_allowed():
            raise RuntimeError(
                "strict release OPEN was applied without a valid release gate"
            )
        self.state.release_command_applied = True
        self.state.release_settle_streak = 0
        self.state.release_wait_steps = 0
        self.state.last_transition_reason = "release_command_applied"
        return True

    def classify_external_reentry(
        self, *, event: int, evidence: TaskPhysicalEvidence
    ) -> str:
        """Classify an external re-entry without trusting one grasp bit."""

        evidence = evidence.validated()
        event = int(event)
        self._observe(evidence)
        if evidence.grasp_candidate:
            self.state.grasp_lost_streak = 0
            self.state.last_transition_reason = "external_reentry_confirmed_attached"
            return "place"
        if event >= 7 and self.state.release_command_applied:
            self.state.last_transition_reason = "external_reentry_confirmed_released"
            return "released"
        if event >= 4:
            self.state.grasp_lost_streak += 1
            if self.state.grasp_lost_streak < int(
                self.config.grasp_lost_confirmation_steps
            ):
                self.state.phase_hold_total_steps += 1
                self.state.last_transition_reason = (
                    "hold_for_external_reentry_grasp_loss_confirmation"
                )
                return "wait"
        self.state.last_transition_reason = "external_reentry_confirmed_missing_grasp"
        return "regrasp"

    def latch_failure(self, reason: str) -> None:
        failure = str(reason).strip()
        if not failure:
            raise ValueError("strict task failure reason must be non-empty")
        if not self.state.success_latched:
            self.state.failure_reason = failure
            self.state.last_transition_reason = failure

    def begin_recovery(
        self, *, mode: str, evidence: TaskPhysicalEvidence
    ) -> None:
        """Clear stale phase gates while a non-policy recovery owns control."""

        evidence = evidence.validated()
        recovery_mode = str(mode)
        if recovery_mode not in ("place", "regrasp"):
            raise ValueError(f"Unsupported recovery mode: {recovery_mode!r}")
        self.state.grasp_lost_streak = 0
        self.state.grasp_approach_wait_steps = 0
        self.state.grasp_confirmation_wait_steps = 0
        self.state.place_ready_streak = 0
        self.state.place_ready_latched = False
        self.state.place_wait_steps = 0
        self.state.release_settle_streak = 0
        self.state.release_wait_steps = 0
        self.state.release_command_applied = False
        if recovery_mode == "regrasp":
            self.state.initial_cube_z_m = float(evidence.cube_z_m)
            self.state.maximum_cube_lift_m = 0.0
            self.state.grasp_observed = False
            self.state.grasp_confirmation_streak = 0
        else:
            self.state.place_recovery_started = True
            self.state.place_recovery_in_progress = True
        self.state.last_transition_reason = (
            f"state_aware_{recovery_mode}_recovery_started"
        )

    def recovery_handoff(
        self,
        *,
        event: int,
        progress: float,
        mode: str,
        evidence: TaskPhysicalEvidence,
        event_before: int,
        progress_before: float,
    ) -> StrictTaskPhaseDecision:
        """Record a physical-anchor handoff without replaying an old primitive."""

        evidence = evidence.validated()
        recovery_mode = str(mode)
        if recovery_mode not in ("place", "regrasp"):
            raise ValueError(f"Unsupported recovery handoff mode: {recovery_mode!r}")
        self.state.cbf_hold_latched = False
        self.state.cbf_clear_streak = 0
        self.state.grasp_lost_streak = 0
        self.state.place_ready_streak = 0
        self.state.place_ready_latched = False
        self.state.place_wait_steps = 0
        self.state.release_settle_streak = 0
        self.state.release_wait_steps = 0
        self.state.release_command_applied = False
        if recovery_mode == "regrasp":
            self.state.pending_retry_open = False
            self.state.initial_cube_z_m = float(evidence.cube_z_m)
            self.state.maximum_cube_lift_m = 0.0
            self.state.grasp_observed = False
            self.state.grasp_confirmation_streak = 0
            self.state.grasp_approach_wait_steps = 0
            self.state.grasp_confirmation_wait_steps = 0
        else:
            self.state.place_recovery_started = True
            self.state.place_recovery_in_progress = False
            self.state.place_recovery_completed = True
        return self._decision(
            int(event),
            float(progress),
            f"state_aware_{recovery_mode}_handoff",
            int(event_before),
            float(progress_before),
            reentry=True,
        )

    def update(
        self,
        *,
        event: int,
        progress: float,
        proposed_event: int,
        proposed_progress: float,
        terminal_event: int,
        evidence: TaskPhysicalEvidence,
    ) -> StrictTaskPhaseDecision:
        evidence = evidence.validated()
        event = int(event)
        progress = float(progress)
        proposed_event = int(proposed_event)
        proposed_progress = float(proposed_progress)
        terminal_event = int(terminal_event)
        self._observe(evidence)

        if self.state.success_latched:
            return self._decision(
                terminal_event, 0.0, "terminal_success_latched", event, progress
            )
        if self.state.failure_reason:
            return self._decision(
                terminal_event,
                0.0,
                "terminal_task_failure_latched",
                event,
                progress,
            )

        # Once the cube is physically released, recognizing that it has
        # settled is an outcome check rather than a nominal-motion phase
        # transition.  A simultaneous CBF intervention must not prevent a
        # valid placement from accumulating its confirmation streak.
        if (
            event == 7
            and self.state.release_command_applied
            and not evidence.grasp_candidate
        ):
            return self._update_released_cube_settling(
                event=event,
                progress=progress,
                terminal_event=terminal_event,
                evidence=evidence,
            )

        released_from_cbf = self._update_cbf_hold(evidence)
        if self.state.cbf_hold_latched:
            if 4 <= event <= 6:
                self.state.grasp_lost_streak = (
                    0
                    if evidence.grasp_candidate
                    else self.state.grasp_lost_streak + 1
                )
            self.state.cbf_pause_steps += 1
            self.state.phase_hold_total_steps += 1
            return self._decision(
                event, progress, "cbf_intervention_pause", event, progress, held=True
            )
        if released_from_cbf:
            return self._reenter_after_cbf(
                event=event,
                progress=progress,
                terminal_event=terminal_event,
                evidence=evidence,
            )

        if event in (1, 2) and proposed_event != event:
            if (
                not evidence.grasp_candidate
                and evidence.ee_cube_distance_m
                > float(self.config.grasp_close_distance_m)
            ):
                self.state.grasp_approach_wait_steps += 1
                self.state.phase_hold_total_steps += 1
                if self.state.grasp_approach_wait_steps >= int(
                    self.config.grasp_approach_timeout_steps
                ):
                    return self._retry_grasp_or_fail(
                        reason="grasp_approach_timeout",
                        event_before=event,
                        progress_before=progress,
                        terminal_event=terminal_event,
                        evidence=evidence,
                    )
                return self._decision(
                    event,
                    progress,
                    "hold_for_grasp_approach",
                    event,
                    progress,
                    held=True,
                )

        if event == 3:
            self.state.grasp_confirmation_wait_steps += 1
            if proposed_event != event and self.state.grasp_confirmation_streak < int(
                self.config.grasp_confirmation_steps
            ):
                self.state.phase_hold_total_steps += 1
                if self.state.grasp_confirmation_wait_steps >= int(
                    self.config.grasp_confirmation_timeout_steps
                ):
                    return self._retry_grasp_or_fail(
                        reason="grasp_confirmation_timeout",
                        event_before=event,
                        progress_before=progress,
                        terminal_event=terminal_event,
                        evidence=evidence,
                    )
                return self._decision(
                    event,
                    progress,
                    "hold_for_grasp_confirmation",
                    event,
                    progress,
                    held=True,
                )

        if 4 <= event <= 6 and not evidence.grasp_candidate:
            self.state.grasp_lost_streak += 1
            self.state.phase_hold_total_steps += 1
            if self.state.grasp_lost_streak >= int(
                self.config.grasp_lost_confirmation_steps
            ):
                return self._retry_grasp_or_fail(
                    reason="grasp_lost_before_release",
                    event_before=event,
                    progress_before=progress,
                    terminal_event=terminal_event,
                    evidence=evidence,
                )
            return self._decision(
                event,
                progress,
                "hold_for_grasp_loss_confirmation",
                event,
                progress,
                held=True,
            )
        if evidence.grasp_candidate:
            self.state.grasp_lost_streak = 0

        if event == 6:
            place_ready = self._place_ready(evidence, require_grasp=True)
            self.state.place_ready_streak = (
                self.state.place_ready_streak + 1 if place_ready else 0
            )
            if not place_ready:
                self.state.place_ready_latched = False
            if self.state.place_ready_streak >= int(
                self.config.place_confirmation_steps
            ):
                self.state.place_ready_latched = True
            if self.state.place_ready_latched:
                self.state.place_wait_steps = 0
                self.state.release_command_applied = False
                return self._decision(
                    7,
                    0.0,
                    "event_driven_release_after_place_confirmation",
                    event,
                    progress,
                )
            if (
                self.config.state_aware_recovery
                and self.state.cbf_intervention_observed
                and not self.state.place_recovery_started
                and not self.state.place_recovery_in_progress
                and progress <= 0.05
                and not self._place_spatially_ready(evidence)
            ):
                self.state.place_recovery_started = True
                self.state.place_reentry_count += 1
                self.state.phase_reentry_count += 1
                self.state.recovery_request_count += 1
                self.state.place_wait_steps = 0
                self.state.place_ready_streak = 0
                return self._decision(
                    event,
                    progress,
                    "request_place_recovery_on_event6_entry",
                    event,
                    progress,
                    held=True,
                    reentry=True,
                    recovery_request="place",
                )
            if (
                self.config.state_aware_recovery
                and self.state.cbf_intervention_observed
                and not self.state.place_recovery_in_progress
                and progress >= 1.0 - 1e-9
                and not self._place_spatially_ready(evidence)
            ):
                self.state.place_recovery_started = True
                self.state.place_reentry_count += 1
                self.state.phase_reentry_count += 1
                self.state.recovery_request_count += 1
                self.state.place_wait_steps = 0
                self.state.place_ready_streak = 0
                return self._decision(
                    event,
                    progress,
                    "request_place_recovery_at_event6_endpoint",
                    event,
                    progress,
                    held=True,
                    reentry=True,
                    recovery_request="place",
                )
            if proposed_event != event and not self.state.place_ready_latched:
                self.state.place_wait_steps += 1
                self.state.phase_hold_total_steps += 1
                if self.state.place_wait_steps >= int(
                    self.config.place_confirmation_timeout_steps
                ):
                    if (
                        self.config.state_aware_recovery
                        and self.state.cbf_intervention_observed
                    ):
                        self.state.place_reentry_count += 1
                        self.state.phase_reentry_count += 1
                        self.state.recovery_request_count += 1
                        self.state.place_wait_steps = 0
                        self.state.place_ready_streak = 0
                        return self._decision(
                            event,
                            progress,
                            "request_place_recovery_after_readiness_timeout",
                            event,
                            progress,
                            held=True,
                            reentry=True,
                            recovery_request="place",
                        )
                    if self.state.place_reentry_count < int(
                        self.config.maximum_place_reentries
                    ):
                        self.state.place_reentry_count += 1
                        self.state.phase_reentry_count += 1
                        self.state.place_wait_steps = 0
                        self.state.place_ready_streak = 0
                        return self._decision(
                            5,
                            0.0,
                            "reenter_transport_before_release",
                            event,
                            progress,
                            reentry=True,
                        )
                    self.state.failure_reason = "place_readiness_timeout"
                    return self._decision(
                        terminal_event,
                        0.0,
                        "place_readiness_timeout",
                        event,
                        progress,
                    )
                return self._decision(
                    event,
                    progress,
                    "hold_for_place_readiness",
                    event,
                    progress,
                    held=True,
                )

        if event == 7:
            return self._update_released_cube_settling(
                event=event,
                progress=progress,
                terminal_event=terminal_event,
                evidence=evidence,
            )

        if proposed_event != event:
            self.state.grasp_approach_wait_steps = 0
            if proposed_event >= 4:
                self.state.grasp_confirmation_wait_steps = 0
        return self._decision(
            proposed_event,
            proposed_progress,
            "physical_transition_permitted",
            event,
            progress,
        )

    def _observe(self, evidence: TaskPhysicalEvidence) -> None:
        self.state.maximum_cube_lift_m = max(
            float(self.state.maximum_cube_lift_m),
            float(evidence.cube_z_m) - float(self.state.initial_cube_z_m),
        )
        self.state.grasp_observed = bool(
            self.state.grasp_observed or evidence.grasp_candidate
        )
        self.state.grasp_confirmation_streak = (
            self.state.grasp_confirmation_streak + 1
            if evidence.grasp_candidate
            else 0
        )

    def _update_cbf_hold(self, evidence: TaskPhysicalEvidence) -> bool:
        intervention = float(evidence.cbf_intervention_norm_radps)
        if not self.state.cbf_hold_latched:
            if intervention >= float(self.config.cbf_pause_enter_norm_radps):
                self.state.cbf_hold_latched = True
                self.state.cbf_clear_streak = 0
                self.state.cbf_intervention_observed = True
            return False
        if intervention <= float(self.config.cbf_pause_exit_norm_radps):
            self.state.cbf_clear_streak += 1
        else:
            self.state.cbf_clear_streak = 0
        if self.state.cbf_clear_streak < int(
            self.config.cbf_pause_exit_confirmation_steps
        ):
            return False
        self.state.cbf_hold_latched = False
        self.state.cbf_clear_streak = 0
        self.state.phase_reentry_count += 1
        return True

    def _reenter_after_cbf(
        self,
        *,
        event: int,
        progress: float,
        terminal_event: int,
        evidence: TaskPhysicalEvidence,
    ) -> StrictTaskPhaseDecision:
        if self.config.state_aware_recovery:
            self.state.place_ready_streak = 0
            self.state.place_ready_latched = False
            self.state.place_wait_steps = 0
            self.state.release_settle_streak = 0
            self.state.release_wait_steps = 0
            if evidence.grasp_candidate:
                if self._place_spatially_ready(evidence):
                    target_progress = progress if event == 6 else 1.0
                    return self._decision(
                        6,
                        target_progress,
                        "cbf_reentry_preserve_place_progress",
                        event,
                        progress,
                        reentry=True,
                    )
                self.state.recovery_request_count += 1
                return self._decision(
                    event,
                    progress,
                    "cbf_reentry_request_place_recovery",
                    event,
                    progress,
                    held=True,
                    reentry=True,
                    recovery_request="place",
                )
            if (
                self.state.grasp_observed
                and self.state.release_command_applied
                and self._place_ready(evidence, require_grasp=False)
            ):
                return self._decision(
                    7,
                    0.0,
                    "cbf_reentry_confirm_released_at_target",
                    event,
                    progress,
                    reentry=True,
                )
            if (
                4 <= event <= 6
                and self.state.grasp_observed
                and self.state.grasp_lost_streak
                < int(self.config.grasp_lost_confirmation_steps)
            ):
                self.state.phase_hold_total_steps += 1
                return self._decision(
                    event,
                    progress,
                    "hold_for_post_cbf_grasp_loss_confirmation",
                    event,
                    progress,
                    held=True,
                    reentry=True,
                )
            self.state.recovery_request_count += 1
            return self._decision(
                event,
                progress,
                "cbf_reentry_request_regrasp_recovery",
                event,
                progress,
                held=True,
                reentry=True,
                recovery_request="regrasp",
            )

        if event == 7 and evidence.grasp_candidate:
            still_place_ready = self._place_ready(
                evidence, require_grasp=True
            )
            self.state.place_ready_streak = 0
            self.state.place_ready_latched = False
            self.state.place_wait_steps = 0
            self.state.release_settle_streak = 0
            self.state.release_wait_steps = 0
            return self._decision(
                6 if still_place_ready else 5,
                0.0,
                (
                    "cbf_reentry_reconfirm_release"
                    if still_place_ready
                    else "cbf_reentry_reposition_before_release"
                ),
                event,
                progress,
                reentry=True,
            )
        if 4 <= event <= 6 and not evidence.grasp_candidate:
            return self._retry_grasp_or_fail(
                reason="cbf_reentry_missing_grasp",
                event_before=event,
                progress_before=progress,
                terminal_event=terminal_event,
                evidence=evidence,
                count_reentry=False,
            )
        if 4 <= event <= 6 and evidence.grasp_candidate:
            reentry_event = (
                4
                if self.state.maximum_cube_lift_m
                < float(self.config.minimum_cube_lift_m)
                else 5
            )
            self.state.place_ready_streak = 0
            self.state.place_ready_latched = False
            return self._decision(
                reentry_event,
                0.0,
                "cbf_reentry_realign_task_state",
                event,
                progress,
                reentry=True,
            )
        return self._decision(
            event,
            progress,
            "cbf_reentry_resume_current_phase",
            event,
            progress,
            reentry=True,
        )

    def _update_released_cube_settling(
        self,
        *,
        event: int,
        progress: float,
        terminal_event: int,
        evidence: TaskPhysicalEvidence,
    ) -> StrictTaskPhaseDecision:
        settled_release = bool(
            self.state.release_command_applied
            and self._place_ready(evidence, require_grasp=False)
            and not evidence.grasp_candidate
        )
        self.state.release_settle_streak = (
            self.state.release_settle_streak + 1 if settled_release else 0
        )
        self.state.release_wait_steps += 1
        if self.state.release_settle_streak >= int(
            self.config.release_settle_confirmation_steps
        ):
            self.state.success_latched = True
            self.state.last_transition_reason = "strict_success_latched"
            return self._decision(
                terminal_event,
                0.0,
                "strict_success_latched",
                event,
                progress,
            )
        if self.state.release_wait_steps >= int(
            self.config.release_settle_timeout_steps
        ):
            self.state.failure_reason = (
                "release_not_commanded"
                if not self.state.release_command_applied
                else "release_not_settled"
            )
            return self._decision(
                terminal_event,
                0.0,
                self.state.failure_reason,
                event,
                progress,
            )
        self.state.phase_hold_total_steps += 1
        return self._decision(
            event,
            progress,
            (
                "hold_for_release_command_application"
                if not self.state.release_command_applied
                else "hold_for_released_cube_settling"
            ),
            event,
            progress,
            held=True,
        )

    def _retry_grasp_or_fail(
        self,
        *,
        reason: str,
        event_before: int,
        progress_before: float,
        terminal_event: int,
        evidence: TaskPhysicalEvidence,
        count_reentry: bool = True,
    ) -> StrictTaskPhaseDecision:
        if self.state.grasp_retry_count < int(self.config.maximum_grasp_retries):
            self.state.grasp_retry_count += 1
            self.state.phase_reentry_count += int(count_reentry)
            if (
                self.config.state_aware_recovery
                and self.state.cbf_intervention_observed
                and reason in (
                    "grasp_lost_before_release",
                    "cbf_reentry_missing_grasp",
                )
            ):
                # A cube lost after a safety detour must be recovered from its
                # current physical pose.  Replaying event 0 here would target
                # the historical pre-detour cube pose and recreate the OOD
                # failure that the recovery bridge is intended to remove.
                self.state.recovery_request_count += 1
                self.state.pending_retry_open = False
                self.state.initial_cube_z_m = float(evidence.cube_z_m)
                self.state.maximum_cube_lift_m = 0.0
                self.state.grasp_observed = False
                self.state.grasp_confirmation_streak = 0
                self.state.grasp_lost_streak = 0
                self.state.grasp_approach_wait_steps = 0
                self.state.grasp_confirmation_wait_steps = 0
                self.state.place_ready_streak = 0
                self.state.place_ready_latched = False
                self.state.place_wait_steps = 0
                self.state.release_settle_streak = 0
                self.state.release_wait_steps = 0
                self.state.release_command_applied = False
                return self._decision(
                    event_before,
                    progress_before,
                    f"request_regrasp_recovery_after_{reason}",
                    event_before,
                    progress_before,
                    held=True,
                    retry_started=True,
                    reentry=True,
                    recovery_request="regrasp",
                )
            self.state.pending_retry_open = True
            self.state.initial_cube_z_m = float(evidence.cube_z_m)
            self.state.maximum_cube_lift_m = 0.0
            self.state.grasp_observed = False
            self.state.grasp_confirmation_streak = 0
            self.state.grasp_lost_streak = 0
            self.state.grasp_approach_wait_steps = 0
            self.state.grasp_confirmation_wait_steps = 0
            self.state.place_ready_streak = 0
            self.state.place_ready_latched = False
            self.state.place_wait_steps = 0
            self.state.release_settle_streak = 0
            self.state.release_wait_steps = 0
            self.state.release_command_applied = False
            return self._decision(
                0,
                0.0,
                f"retry_after_{reason}",
                event_before,
                progress_before,
                retry_started=True,
                reentry=True,
            )
        self.state.failure_reason = str(reason)
        return self._decision(
            terminal_event,
            0.0,
            reason,
            event_before,
            progress_before,
        )

    def _place_ready(
        self, evidence: TaskPhysicalEvidence, *, require_grasp: bool
    ) -> bool:
        return bool(
            evidence.cube_target_xy_error_m
            <= float(self.config.place_xy_tolerance_m)
            and evidence.cube_target_z_error_m
            <= float(self.config.place_z_tolerance_m)
            and evidence.cube_speed_mps
            <= float(self.config.place_max_cube_speed_mps)
            and self.state.maximum_cube_lift_m
            >= float(self.config.minimum_cube_lift_m)
            and self.state.grasp_observed
            and (evidence.grasp_candidate if require_grasp else True)
        )

    def _place_spatially_ready(self, evidence: TaskPhysicalEvidence) -> bool:
        return bool(
            evidence.cube_target_xy_error_m
            <= float(self.config.place_xy_tolerance_m)
            and evidence.cube_target_z_error_m
            <= float(self.config.place_z_tolerance_m)
            and self.state.maximum_cube_lift_m
            >= float(self.config.minimum_cube_lift_m)
            and self.state.grasp_observed
            and evidence.grasp_candidate
        )

    def _decision(
        self,
        event: int,
        progress: float,
        reason: str,
        event_before: int,
        progress_before: float,
        *,
        held: bool = False,
        retry_started: bool = False,
        reentry: bool = False,
        recovery_request: str = "none",
    ) -> StrictTaskPhaseDecision:
        self.state.last_transition_reason = str(reason)
        changed = bool(
            int(event) != int(event_before)
            or not math.isclose(float(progress), float(progress_before), abs_tol=1e-15)
        )
        return StrictTaskPhaseDecision(
            event=int(event),
            progress=float(progress),
            reason=str(reason),
            phase_changed=changed,
            held=bool(held),
            retry_started=bool(retry_started),
            reentry=bool(reentry),
            success_latched=bool(self.state.success_latched),
            failure_reason=str(self.state.failure_reason),
            recovery_request=str(recovery_request),
        )


__all__ = [
    "STRICT_TASK_SEMANTICS_SCHEMA",
    "StrictTaskPhaseDecision",
    "StrictTaskSemanticsConfig",
    "StrictTaskSemanticsController",
    "StrictTaskSemanticsState",
    "TaskPhysicalEvidence",
]
