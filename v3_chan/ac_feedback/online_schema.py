"""Frozen value objects for the A/C selective-smoothing collection schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral
from typing import Any, Mapping, Sequence


ONLINE_SCHEMA_VERSION = "bc_cbf_recovery_ac_explicit_feedback_v1"
ONLINE_PROTOCOL_VERSION = "ac_selective_smoothing_pilot_v1"
ONLINE_ROW_SEMANTICS = "obs_t_bc_t_rmpflow_t_cbf_ac_t_applied_t_obs_t_plus_1_v1"
FEATURE_PROVENANCE_VERSION = "runtime_observable_vs_future_outcome_v1"

TASK_PHASES = ("reach_approach", "grasp_lift", "transport", "place_release")
SEVERITIES = ("shallow", "threat")
DIRECTIONS = ("left_to_right", "right_to_left")
ACTUAL_CROSSING_DIRECTIONS = (*DIRECTIONS, "not_observed")
SPEEDS = ("slow", "fast")
Q1_ID = "reuse_robot_response_needs_modification_v1"
Q1_TEXT_KO = "같은 상황에서 방금 로봇 반응을 다시 사용한다면 수정이 필요합니까?"
Q1_RESPONSES = ("needs_modification", "acceptable_as_is", "uncertain")
LIKERT_QUESTIONS = {
    "q2_perceived_danger": "방금 로봇 반응에서 손과 충돌하거나 지나치게 가까워질 위험을 얼마나 느꼈습니까?",
    "q3_abruptness": "방금 로봇 움직임이 갑작스럽거나 예측하기 어려웠습니까?",
    "q4_excessive_duration": "방금 safety response가 필요 이상으로 오래 지속되었다고 느꼈습니까?",
    "q5_task_disruption": "방금 로봇 반응이 원래 pick-and-place 진행을 얼마나 방해했다고 느꼈습니까?",
    "q6_confidence": "방금 판단에 얼마나 확신합니까?",
}
MODIFICATION_REASONS = (
    "too_close_or_late",
    "excessive_or_unnecessary_motion",
    "abrupt_or_unpredictable",
    "response_too_long",
    "inappropriate_direction",
    "recovery_or_task_resumption_problem",
    "grasp_place_release_disruption",
    "other",
)
REJECTION_REASONS = MODIFICATION_REASONS
MARKER_TYPES = ("realtime_safety_concern", "realtime_behavior_anomaly")
RESPONSE_PHASES = (
    "PRE_RESPONSE",
    "CBF_ACTIVE",
    "SMOOTH_TAIL",
    "RECOVERY_ACTIVE",
    "BC_RESUMED",
    "STABLE_TASK_RESUMPTION",
)
TRIAL_STATES = (
    "RESET",
    "LOAD_BC_FEASIBLE_SCENARIO",
    "START_FROZEN_BC",
    "WAIT_FOR_TARGET_TASK_PHASE",
    "SHOW_HAND_CROSSING_CUE",
    "EXECUTE_SINGLE_CROSSING",
    "RUN_ASSIGNED_RESPONSE_A_OR_C",
    "TRACK_SAFETY_RESPONSE_EPISODE",
    "DETECT_STABLE_TASK_RESUMPTION",
    "SHOW_RETURN_HAND_TO_NEUTRAL_CUE",
    "PAUSE_NOMINAL_TASK_PROGRESSION",
    "SHOW_MANDATORY_FEEDBACK_UI",
    "SAVE_FEEDBACK",
    "RESUME_TASK_TO_COMPLETION_OR_TERMINAL",
    "SAVE_TRIAL",
    "VALIDATE_TRIAL",
    "NEXT_TRIAL_OR_END_SESSION",
)
CONDITION_CONTRACT = {
    "A_reactive": ("joint_nominal", 0.0),
    "C_smooth": ("smooth_intervention", 4.0),
}

RUNTIME_OBSERVABLE_FIELDS = (
    "current_gap",
    "current_ttc",
    "closing_speed",
    "current_task_phase",
    "hand_position_velocity",
    "robot_state",
    "nominal_command",
    "cbf_state_so_far",
)
FUTURE_OUTCOME_FIELDS = (
    "final_task_success",
    "final_recovery_time",
    "total_intervention_duration",
    "future_minimum_gap",
    "future_maximum_jerk",
    "eventual_object_drop",
    "post_encounter_outcome",
)


class OnlineProtocolViolation(RuntimeError):
    """A fail-closed invariant of the online protocol was violated."""


@dataclass(frozen=True)
class TrialPlan:
    trial_index: int
    trial_id: str
    query_id: str
    encounter_id: str
    task_phase: str
    severity: str
    direction: str
    speed: str
    crossing_hand: str
    condition_id: str
    objective_mode: str
    lambda_s: float
    schedule_seed: int
    block_id: str
    counterbalancing_group: str
    practice: bool = False
    anchor_repeat: bool = False
    pilot: bool = False
    analysis_exclude: bool = False
    source_trial_id: str = ""

    def __post_init__(self) -> None:
        if not _is_nonnegative_integer(self.trial_index):
            raise OnlineProtocolViolation("trial_index must be a non-negative integer")
        for name in (
            "trial_id", "query_id", "encounter_id", "block_id",
            "counterbalancing_group",
        ):
            if not str(getattr(self, name)).strip():
                raise OnlineProtocolViolation(f"{name} must be non-empty")
        if self.task_phase not in TASK_PHASES:
            raise OnlineProtocolViolation(f"invalid task phase: {self.task_phase!r}")
        if self.severity not in SEVERITIES:
            raise OnlineProtocolViolation(f"invalid severity: {self.severity!r}")
        if self.direction not in DIRECTIONS:
            raise OnlineProtocolViolation(f"invalid crossing direction: {self.direction!r}")
        if self.speed not in SPEEDS:
            raise OnlineProtocolViolation(f"invalid crossing speed: {self.speed!r}")
        if self.crossing_hand not in {"left", "right"}:
            raise OnlineProtocolViolation("crossing_hand must be left or right")
        expected = CONDITION_CONTRACT.get(self.condition_id)
        actual = (self.objective_mode, float(self.lambda_s))
        if expected is None or actual != expected:
            raise OnlineProtocolViolation(
                "condition_id/objective_mode/lambda_s violates the frozen A/C contract"
            )
        if not _is_nonnegative_integer(self.schedule_seed):
            raise OnlineProtocolViolation("schedule_seed must be a non-negative integer")
        for name in ("practice", "anchor_repeat", "pilot", "analysis_exclude"):
            if not isinstance(getattr(self, name), bool):
                raise OnlineProtocolViolation(f"{name} must be boolean")
        if not self.pilot:
            raise OnlineProtocolViolation("this collector only supports the feasibility pilot")
        if self.practice and not self.analysis_exclude:
            raise OnlineProtocolViolation("practice trials must be excluded from analysis")
        if self.anchor_repeat != bool(str(self.source_trial_id).strip()):
            raise OnlineProtocolViolation(
                "anchor_repeat requires exactly one non-empty source context identifier"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealtimeMarkerRecord:
    marker_id: str
    trial_id: str
    participant_id: str
    session_id: str
    encounter_id: str
    condition_id: str
    lambda_s: float
    marker_type: str
    simulation_time_s: float
    monotonic_ns: int
    unix_ns: int
    control_step: int
    response_phase: str
    time_since_response_onset_s: float
    time_since_recovery_onset_s: float
    controller_hand: str
    button: str
    button_identifier: str

    def __post_init__(self) -> None:
        for name in (
            "marker_id", "trial_id", "participant_id", "session_id",
            "encounter_id", "condition_id", "marker_type", "response_phase",
            "controller_hand", "button", "button_identifier",
        ):
            if not str(getattr(self, name)).strip():
                raise OnlineProtocolViolation(f"marker {name} must be non-empty")
        if self.marker_type not in MARKER_TYPES:
            raise OnlineProtocolViolation(f"invalid marker type: {self.marker_type!r}")
        if self.response_phase not in RESPONSE_PHASES:
            raise OnlineProtocolViolation(
                f"invalid marker response phase: {self.response_phase!r}"
            )
        if self.controller_hand not in {"left", "right"}:
            raise OnlineProtocolViolation("marker controller_hand must be left or right")
        if self.button_identifier != f"{self.controller_hand}.{self.button}":
            raise OnlineProtocolViolation(
                "marker button_identifier must equal controller_hand.button"
            )
        expected = CONDITION_CONTRACT.get(self.condition_id)
        if expected is None or float(self.lambda_s) != expected[1]:
            raise OnlineProtocolViolation("marker condition/lambda does not match A/C contract")
        simulation_time = float(self.simulation_time_s)
        if not math.isfinite(simulation_time) or simulation_time < 0.0:
            raise OnlineProtocolViolation(
                "marker simulation_time_s must be finite and non-negative"
            )
        for name in (
            "time_since_response_onset_s", "time_since_recovery_onset_s"
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (value < 0.0 and value != -1.0):
                raise OnlineProtocolViolation(
                    f"marker {name} must be -1 or finite and non-negative"
                )
        for name in ("monotonic_ns", "unix_ns", "control_step"):
            value = getattr(self, name)
            if not _is_nonnegative_integer(value):
                raise OnlineProtocolViolation(f"marker {name} must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    trial_id: str
    encounter_id: str
    issued_simulation_time_s: float
    issued_monotonic_ns: int
    issued_unix_ns: int
    issued_control_step: int
    completed_simulation_time_s: float
    completed_monotonic_ns: int
    completed_unix_ns: int
    completed_control_step: int
    response_status: str
    response_disposition: str
    q1_response: str = ""
    q2_perceived_danger: int = 0
    q3_abruptness: int = 0
    q4_excessive_duration: int = 0
    q5_task_disruption: int = 0
    q6_confidence: int = 0
    modification_reasons: tuple[str, ...] = field(default_factory=tuple)
    response_latency_ms: float = 0.0
    input_device: str = "vr_controller"
    first_input_simulation_time_s: float = -1.0
    first_input_monotonic_ns: int = 0
    first_input_unix_ns: int = 0
    first_input_control_step: int = -1
    back_correction_count: int = 0
    accidental_input_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["modification_reasons_json"] = _compact_json(self.modification_reasons)
        del result["modification_reasons"]
        return result


@dataclass(frozen=True)
class EncounterRecordV1:
    encounter_id: str
    trial_id: str
    onset_step: int
    onset_confirmed_step: int
    offset_step_exclusive: int
    onset_simulation_time_s: float
    offset_simulation_time_s: float
    onset_monotonic_ns: int
    offset_monotonic_ns: int
    window_start_step: int
    window_end_step_exclusive: int
    minimum_surface_gap_m: float
    maximum_intervention_norm_rad_s: float
    risk_onset_step: int
    cbf_intervention_start_step: int
    cbf_intervention_confirmed_step: int
    encounter_timeout: bool
    merged_reentry_count: int

    def __post_init__(self) -> None:
        if not self.encounter_id.strip() or not self.trial_id.strip():
            raise OnlineProtocolViolation("encounter identifiers must be non-empty")
        for name in (
            "onset_step", "onset_confirmed_step", "offset_step_exclusive",
            "onset_monotonic_ns", "offset_monotonic_ns", "window_start_step",
            "window_end_step_exclusive", "merged_reentry_count",
        ):
            if not _is_nonnegative_integer(getattr(self, name)):
                raise OnlineProtocolViolation(
                    f"encounter {name} must be a non-negative integer"
                )
        if (
            self.onset_step < 0
            or self.onset_confirmed_step < self.onset_step
            or self.onset_confirmed_step >= self.offset_step_exclusive
            or self.offset_step_exclusive <= self.onset_step
            or self.window_start_step < 0
            or self.window_start_step > self.onset_step
            or self.offset_step_exclusive > self.window_end_step_exclusive
            or self.window_end_step_exclusive <= self.window_start_step
        ):
            raise OnlineProtocolViolation("encounter step bounds are invalid")
        if self.offset_simulation_time_s < self.onset_simulation_time_s:
            raise OnlineProtocolViolation("encounter simulation timestamps are reversed")
        if self.offset_monotonic_ns < self.onset_monotonic_ns:
            raise OnlineProtocolViolation("encounter monotonic timestamps are reversed")
        for name in (
            "onset_simulation_time_s", "offset_simulation_time_s",
            "minimum_surface_gap_m", "maximum_intervention_norm_rad_s",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise OnlineProtocolViolation(f"encounter {name} must be finite")
        if self.onset_simulation_time_s < 0.0:
            raise OnlineProtocolViolation("encounter onset time must be non-negative")
        if self.risk_onset_step != self.onset_step:
            raise OnlineProtocolViolation("risk_onset_step must equal onset_step")
        if self.maximum_intervention_norm_rad_s < 0:
            raise OnlineProtocolViolation("encounter diagnostics must be non-negative")
        for name in (
            "risk_onset_step", "cbf_intervention_start_step",
            "cbf_intervention_confirmed_step",
        ):
            if not _is_integer_at_least(getattr(self, name), -1):
                raise OnlineProtocolViolation(f"encounter {name} must be >= -1")
        if not isinstance(self.encounter_timeout, bool):
            raise OnlineProtocolViolation("encounter_timeout must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_bool(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise OnlineProtocolViolation(f"{name} must be an exact boolean")


def validate_query_record(record: QueryRecord) -> None:
    if not record.query_id.strip() or not record.trial_id.strip() or not record.encounter_id.strip():
        raise OnlineProtocolViolation("query identifiers must be non-empty")
    if record.response_status not in {"completed", "timeout", "no_response"}:
        raise OnlineProtocolViolation("invalid query response_status")
    expected_disposition = (
        "missing_abstain"
        if record.response_status != "completed"
        else (
            "uncertain_abstain"
            if record.q1_response == "uncertain"
            else "answered"
        )
    )
    if record.response_disposition != expected_disposition:
        raise OnlineProtocolViolation(
            "response_disposition disagrees with response_status/Q1"
        )
    if record.response_status == "completed":
        if record.q1_response not in Q1_RESPONSES:
            raise OnlineProtocolViolation("completed query requires a valid Q1 response")
        ratings = (
            record.q2_perceived_danger,
            record.q3_abruptness,
            record.q4_excessive_duration,
            record.q5_task_disruption,
            record.q6_confidence,
        )
        if any(
            not _is_integer_at_least(value, 1) or int(value) > 5
            for value in ratings
        ):
            raise OnlineProtocolViolation("completed query requires five separate 1-5 ratings")
    else:
        missing_ratings = (
            record.q2_perceived_danger,
            record.q3_abruptness,
            record.q4_excessive_duration,
            record.q5_task_disruption,
            record.q6_confidence,
        )
        if record.q1_response or any(
            not _is_integer_at_least(value, 0) or int(value) != 0
            for value in missing_ratings
        ):
            raise OnlineProtocolViolation("missing/timeout response cannot carry inferred labels")
    reasons = tuple(str(value) for value in record.modification_reasons)
    if len(set(reasons)) != len(reasons) or any(
        reason not in MODIFICATION_REASONS for reason in reasons
    ):
        raise OnlineProtocolViolation("invalid or duplicate modification reason")
    if record.q1_response == "needs_modification" and not reasons:
        raise OnlineProtocolViolation("needs_modification requires a modification reason")
    if record.q1_response != "needs_modification" and reasons:
        raise OnlineProtocolViolation("modification reasons are only valid for needs_modification")
    if record.input_device != "vr_controller":
        raise OnlineProtocolViolation("mandatory feedback input_device must be vr_controller")
    for name in (
        "issued_simulation_time_s", "completed_simulation_time_s",
        "response_latency_ms", "first_input_simulation_time_s",
    ):
        if not math.isfinite(float(getattr(record, name))):
            raise OnlineProtocolViolation(f"{name} must be finite")
    if record.issued_simulation_time_s < 0:
        raise OnlineProtocolViolation("issued_simulation_time_s must be non-negative")
    if record.completed_simulation_time_s < record.issued_simulation_time_s:
        raise OnlineProtocolViolation("query completion precedes prompt display in simulation time")
    if record.completed_monotonic_ns < record.issued_monotonic_ns:
        raise OnlineProtocolViolation("query completion precedes prompt display")
    if record.completed_unix_ns < record.issued_unix_ns:
        raise OnlineProtocolViolation(
            "query completion precedes prompt display in Unix time"
        )
    if record.completed_control_step < record.issued_control_step:
        raise OnlineProtocolViolation(
            "query completion precedes its issued control step"
        )
    expected_latency_ms = (
        int(record.completed_monotonic_ns) - int(record.issued_monotonic_ns)
    ) / 1e6
    if record.response_latency_ms < 0 or not math.isclose(
        float(record.response_latency_ms), expected_latency_ms, rel_tol=0.0, abs_tol=1e-3
    ):
        raise OnlineProtocolViolation("response_latency_ms disagrees with monotonic clocks")
    for name in (
        "issued_monotonic_ns", "issued_unix_ns", "issued_control_step",
        "completed_monotonic_ns", "completed_unix_ns", "completed_control_step",
        "back_correction_count", "accidental_input_count",
    ):
        value = getattr(record, name)
        if not _is_nonnegative_integer(value):
            raise OnlineProtocolViolation(f"{name} must be non-negative")
    first_missing = float(record.first_input_simulation_time_s) < 0
    if record.response_status == "completed" and first_missing:
        raise OnlineProtocolViolation("completed query requires a first-input timestamp")
    if first_missing:
        if not (
            record.first_input_simulation_time_s == -1.0
            and record.first_input_monotonic_ns == 0
            and record.first_input_unix_ns == 0
            and record.first_input_control_step == -1
            and _is_nonnegative_integer(record.first_input_monotonic_ns)
            and _is_nonnegative_integer(record.first_input_unix_ns)
            and _is_integer_at_least(record.first_input_control_step, -1)
        ):
            raise OnlineProtocolViolation("missing first-input clocks must use exact sentinels")
    else:
        for name in (
            "first_input_monotonic_ns", "first_input_unix_ns",
            "first_input_control_step",
        ):
            if not _is_nonnegative_integer(getattr(record, name)):
                raise OnlineProtocolViolation(
                    f"{name} must be a non-negative integer when input exists"
                )
        if (
            record.first_input_simulation_time_s < record.issued_simulation_time_s
            or record.first_input_simulation_time_s > record.completed_simulation_time_s
            or record.first_input_monotonic_ns < record.issued_monotonic_ns
            or record.first_input_monotonic_ns > record.completed_monotonic_ns
            or record.first_input_unix_ns < record.issued_unix_ns
            or record.first_input_unix_ns > record.completed_unix_ns
            or record.first_input_control_step < record.issued_control_step
            or record.first_input_control_step > record.completed_control_step
        ):
            raise OnlineProtocolViolation("first input lies outside the query interval")


def _is_nonnegative_integer(value: Any) -> bool:
    return _is_integer_at_least(value, 0)


def _is_integer_at_least(value: Any, minimum: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and int(value) >= int(minimum)
    )


def mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_json(values: Sequence[Any]) -> str:
    import json

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
