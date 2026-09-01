from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


STRICT_TASK_SEMANTICS_SCHEMA = "physical_event_driven_pick_place_v3"


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

    def validated(self) -> "StrictTaskSemanticsConfig":
        if not isinstance(self.enabled, bool):
            raise ValueError("strict task semantics enabled must be boolean")
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
    phase_hold_total_steps: int = 0
    phase_reentry_count: int = 0
    pending_retry_open: bool = False
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
        self.state = StrictTaskSemanticsState(**dict(state))

    def consume_pending_retry_open(self) -> bool:
        pending = bool(self.state.pending_retry_open)
        self.state.pending_retry_open = False
        return pending

    def release_command_allowed(self) -> bool:
        return bool(
            self.state.place_ready_latched
            and not self.state.cbf_hold_latched
            and not self.state.failure_reason
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
        if event == 7 and not evidence.grasp_candidate:
            return self._update_released_cube_settling(
                event=event,
                progress=progress,
                terminal_event=terminal_event,
                evidence=evidence,
            )

        released_from_cbf = self._update_cbf_hold(evidence)
        if self.state.cbf_hold_latched:
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
                return self._decision(
                    7,
                    0.0,
                    "event_driven_release_after_place_confirmation",
                    event,
                    progress,
                )
            if proposed_event != event and not self.state.place_ready_latched:
                self.state.place_wait_steps += 1
                self.state.phase_hold_total_steps += 1
                if self.state.place_wait_steps >= int(
                    self.config.place_confirmation_timeout_steps
                ):
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
        settled_release = self._place_ready(
            evidence, require_grasp=False
        ) and not evidence.grasp_candidate
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
            self.state.failure_reason = "release_not_settled"
            return self._decision(
                terminal_event,
                0.0,
                "release_not_settled",
                event,
                progress,
            )
        self.state.phase_hold_total_steps += 1
        return self._decision(
            event,
            progress,
            "hold_for_released_cube_settling",
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
        )


__all__ = [
    "STRICT_TASK_SEMANTICS_SCHEMA",
    "StrictTaskPhaseDecision",
    "StrictTaskSemanticsConfig",
    "StrictTaskSemanticsController",
    "StrictTaskSemanticsState",
    "TaskPhysicalEvidence",
]
