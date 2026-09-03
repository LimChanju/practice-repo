"""Pure-Python study contract for A/C selective-smoothing feedback collection.

This module deliberately has no Isaac Sim, Torch, HDF5, or XR imports.  It
owns the randomized schedule, trial lifecycle, and safety-response phase
detector so those research-critical semantics can be tested off robot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import random
import re
import uuid
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "bc_cbf_recovery_ac_explicit_feedback_v1"
TASK_PHASES = (
    "reach_approach",
    "grasp_lift",
    "transport",
    "place_release",
)
SEVERITIES = ("shallow", "threat")
DIRECTIONS = ("left_to_right", "right_to_left")
SPEEDS = ("slow", "fast")
CONDITION_IDS = ("A_reactive", "C_smooth")


@dataclass(frozen=True)
class ResponseCondition:
    condition_id: str
    objective_mode: str
    lambda_s: float

    def __post_init__(self) -> None:
        expected = {
            "A_reactive": ("joint_nominal", 0.0),
            "C_smooth": ("smooth_intervention", 4.0),
        }
        if self.condition_id not in expected:
            raise ValueError(f"unsupported response condition: {self.condition_id!r}")
        if (self.objective_mode, float(self.lambda_s)) != expected[self.condition_id]:
            raise ValueError(f"condition contract mismatch for {self.condition_id}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


A_REACTIVE = ResponseCondition("A_reactive", "joint_nominal", 0.0)
C_SMOOTH = ResponseCondition("C_smooth", "smooth_intervention", 4.0)
CONDITIONS: Mapping[str, ResponseCondition] = {
    A_REACTIVE.condition_id: A_REACTIVE,
    C_SMOOTH.condition_id: C_SMOOTH,
}


@dataclass(frozen=True)
class SeverityDefinition:
    name: str
    minimum_gap_m: float
    maximum_gap_m: float
    target_gap_m: float

    def __post_init__(self) -> None:
        if self.name not in SEVERITIES:
            raise ValueError(f"unsupported severity: {self.name!r}")
        values = tuple(float(value) for value in (
            self.minimum_gap_m,
            self.maximum_gap_m,
            self.target_gap_m,
        ))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("severity gaps must be finite")
        if not 0.0 <= values[0] <= values[2] <= values[1]:
            raise ValueError("target gap must lie inside its planned range")


SEVERITY_DEFINITIONS: Mapping[str, SeverityDefinition] = {
    "shallow": SeverityDefinition("shallow", 0.08, 0.12, 0.10),
    "threat": SeverityDefinition("threat", 0.00, 0.04, 0.02),
}


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    query_id: str
    encounter_id: str
    order_index: int
    evaluated_index: int | None
    task_phase: str
    severity: str
    crossing_direction: str
    crossing_speed: str
    crossing_hand: str
    condition_id: str
    objective_mode: str
    lambda_s: float
    planned_gap_min_m: float
    planned_gap_max_m: float
    planned_gap_target_m: float
    schedule_seed: int
    block_id: str
    counterbalancing_group: str
    practice: bool = False
    analysis_exclude: bool = False
    anchor_repeat: bool = False
    anchor_context_id: str = ""
    pilot: bool = True

    def __post_init__(self) -> None:
        if any(not str(value).strip() for value in (
            self.trial_id,
            self.query_id,
            self.encounter_id,
            self.block_id,
            self.counterbalancing_group,
        )):
            raise ValueError("trial identifiers must be non-empty")
        if self.order_index < 0:
            raise ValueError("order_index must be non-negative")
        if self.task_phase not in TASK_PHASES:
            raise ValueError(f"invalid task phase: {self.task_phase!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity!r}")
        if self.crossing_direction not in DIRECTIONS:
            raise ValueError(f"invalid direction: {self.crossing_direction!r}")
        if self.crossing_speed not in SPEEDS:
            raise ValueError(f"invalid speed: {self.crossing_speed!r}")
        if self.crossing_hand not in ("left", "right"):
            raise ValueError("crossing_hand must be left or right")
        condition = CONDITIONS.get(self.condition_id)
        if condition is None:
            raise ValueError(f"invalid condition: {self.condition_id!r}")
        if self.objective_mode != condition.objective_mode or float(self.lambda_s) != condition.lambda_s:
            raise ValueError("condition mode/lambda does not match the frozen contract")
        severity = SEVERITY_DEFINITIONS[self.severity]
        if tuple(float(value) for value in (
            self.planned_gap_min_m,
            self.planned_gap_max_m,
            self.planned_gap_target_m,
        )) != (
            severity.minimum_gap_m,
            severity.maximum_gap_m,
            severity.target_gap_m,
        ):
            raise ValueError("planned gap does not match the frozen severity")
        if self.practice != (self.evaluated_index is None):
            raise ValueError("only practice trials may omit evaluated_index")
        if self.practice and not self.analysis_exclude:
            raise ValueError("practice trials must be excluded from analysis")
        if self.anchor_repeat != bool(self.anchor_context_id):
            raise ValueError("anchor repeats require an anchor_context_id")

    @property
    def target_phase(self) -> str:
        return self.task_phase

    @property
    def direction(self) -> str:
        return self.crossing_direction

    @property
    def speed(self) -> str:
        return self.crossing_speed

    @property
    def is_evaluated(self) -> bool:
        return not self.practice

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolSchedule(Sequence[TrialSpec]):
    participant_id: str
    session_id: str
    schedule_seed: int
    counterbalancing_group: str
    mode: str
    trials: tuple[TrialSpec, ...]

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.trials[index]

    def __iter__(self) -> Iterator[TrialSpec]:
        return iter(self.trials)

    @property
    def practice_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(trial for trial in self.trials if trial.practice)

    @property
    def evaluated_trials(self) -> tuple[TrialSpec, ...]:
        return tuple(trial for trial in self.trials if trial.is_evaluated)

    @property
    def trial_order(self) -> tuple[str, ...]:
        return tuple(trial.trial_id for trial in self.trials)


def build_participant_schedule(
    participant_id: str,
    *,
    session_id: str,
    seed: int = 11,
    mode: str = "minimal_pilot",
    practice_trials: int = 4,
) -> ProtocolSchedule:
    """Build the deterministic pre-assigned A/C schedule.

    Production cells are exactly phase(4) x severity(2) x condition(2).  A/C
    trials for each phase/severity context are adjacent in a randomized pair;
    pair orientation alternates, which guarantees a maximum condition run of
    two while balancing the first condition by counterbalancing group.
    """

    participant_id = str(participant_id).strip()
    session_id = str(session_id).strip()
    if not participant_id or not session_id:
        raise ValueError("participant_id and session_id must be non-empty pseudonyms")
    if mode not in ("minimal_pilot", "pilot_with_anchors"):
        raise ValueError("mode must be minimal_pilot or pilot_with_anchors")
    if int(practice_trials) != 4:
        raise ValueError("the frozen pilot protocol requires exactly four practice trials")
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    schedule_seed = _derived_seed(participant_id, session_id, int(seed))
    group_index = _counterbalancing_group_index(participant_id)
    group = f"G{group_index:02d}"
    namespace = uuid.UUID("f70d5163-6a69-4adc-8b49-f4eef5323a93")
    contexts = [(phase, severity) for phase in TASK_PHASES for severity in SEVERITIES]
    rng = random.Random(schedule_seed)
    rng.shuffle(contexts)
    direction_offset = (group_index >> 1) & 1
    speed_offset = (group_index >> 2) & 1
    first_condition = group_index & 1
    planned: list[TrialSpec] = []

    def make_trial(
        *,
        phase: str,
        severity_name: str,
        condition_index: int,
        direction_index: int,
        speed_index: int,
        block_id: str,
        order_index: int,
        practice: bool,
        evaluated_index: int | None,
        anchor_repeat: bool = False,
        anchor_context_id: str = "",
        token: str,
    ) -> TrialSpec:
        condition = (A_REACTIVE, C_SMOOTH)[condition_index]
        direction = DIRECTIONS[direction_index]
        speed = SPEEDS[speed_index]
        crossing_hand = "left" if direction == "left_to_right" else "right"
        severity = SEVERITY_DEFINITIONS[severity_name]
        trial_id = str(uuid.uuid5(namespace, f"{schedule_seed}|trial|{token}"))
        return TrialSpec(
            trial_id=trial_id,
            query_id=str(uuid.uuid5(namespace, f"{schedule_seed}|query|{token}")),
            encounter_id=str(uuid.uuid5(namespace, f"{schedule_seed}|encounter|{token}")),
            order_index=order_index,
            evaluated_index=evaluated_index,
            task_phase=phase,
            severity=severity_name,
            crossing_direction=direction,
            crossing_speed=speed,
            crossing_hand=crossing_hand,
            condition_id=condition.condition_id,
            objective_mode=condition.objective_mode,
            lambda_s=condition.lambda_s,
            planned_gap_min_m=severity.minimum_gap_m,
            planned_gap_max_m=severity.maximum_gap_m,
            planned_gap_target_m=severity.target_gap_m,
            schedule_seed=schedule_seed,
            block_id=block_id,
            counterbalancing_group=group,
            practice=practice,
            analysis_exclude=practice,
            anchor_repeat=anchor_repeat,
            anchor_context_id=anchor_context_id,
            pilot=True,
        )

    for index, phase in enumerate(TASK_PHASES):
        condition_index = (first_condition + index) % 2
        severity_index = index % 2
        planned.append(make_trial(
            phase=phase,
            severity_name=SEVERITIES[severity_index],
            condition_index=condition_index,
            direction_index=(index + direction_offset) % 2,
            speed_index=((index // 2) + speed_offset) % 2,
            block_id="practice",
            order_index=len(planned),
            practice=True,
            evaluated_index=None,
            token=f"practice|{index}",
        ))

    production: list[TrialSpec] = []
    for context_index, (phase, severity_name) in enumerate(contexts):
        phase_index = TASK_PHASES.index(phase)
        severity_index = SEVERITIES.index(severity_name)
        pair_first = (first_condition + context_index) % 2
        for within_pair, condition_index in enumerate((pair_first, 1 - pair_first)):
            direction_index = (
                phase_index + severity_index + condition_index + direction_offset
            ) % 2
            speed_index = (phase_index + severity_index + speed_offset) % 2
            production.append(make_trial(
                phase=phase,
                severity_name=severity_name,
                condition_index=condition_index,
                direction_index=direction_index,
                speed_index=speed_index,
                block_id=f"block_{context_index // 2 + 1:02d}",
                order_index=len(planned) + len(production),
                practice=False,
                evaluated_index=len(production),
                token=f"production|{context_index}|{within_pair}|{phase}|{severity_name}",
            ))
    planned.extend(production)

    if mode == "pilot_with_anchors":
        anchor_contexts = _select_anchor_contexts(contexts, schedule_seed)
        for anchor_index, (phase, severity_name) in enumerate(anchor_contexts):
            phase_index = TASK_PHASES.index(phase)
            severity_index = SEVERITIES.index(severity_name)
            context_id = f"anchor_{anchor_index + 1:02d}_{phase}_{severity_name}"
            for condition_index in (0, 1):
                planned.append(make_trial(
                    phase=phase,
                    severity_name=severity_name,
                    condition_index=condition_index,
                    direction_index=(phase_index + severity_index + condition_index + direction_offset) % 2,
                    speed_index=(phase_index + severity_index + speed_offset) % 2,
                    block_id="block_05_anchors",
                    order_index=len(planned),
                    practice=False,
                    evaluated_index=len([trial for trial in planned if not trial.practice]),
                    anchor_repeat=True,
                    anchor_context_id=context_id,
                    token=f"anchor|{anchor_index}|{condition_index}|{phase}|{severity_name}",
                ))

    _validate_schedule(planned, mode=mode)
    return ProtocolSchedule(
        participant_id=participant_id,
        session_id=session_id,
        schedule_seed=schedule_seed,
        counterbalancing_group=group,
        mode=mode,
        trials=tuple(planned),
    )


def _select_anchor_contexts(
    contexts: Sequence[tuple[str, str]], schedule_seed: int
) -> tuple[tuple[str, str], tuple[str, str]]:
    ordered = sorted(
        contexts,
        key=lambda value: hashlib.sha256(
            f"{schedule_seed}|anchor|{value[0]}|{value[1]}".encode("utf-8")
        ).digest(),
    )
    first = ordered[0]
    second = next(
        value for value in ordered[1:]
        if value[0] != first[0] and value[1] != first[1]
    )
    return first, second


def _derived_seed(participant_id: str, session_id: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}|{participant_id}|{session_id}|{seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _counterbalancing_group_index(participant_id: str) -> int:
    """Assign sequential lab pseudonyms to successive Latin-style groups."""

    match = re.search(r"(\d+)$", str(participant_id).strip())
    if match is not None and int(match.group(1)) > 0:
        return (int(match.group(1)) - 1) % 8
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}|counterbalance|{participant_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:2], "big") % 8


def _count(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _maximum_run(values: Sequence[str]) -> int:
    longest = current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def _validate_schedule(trials: Sequence[TrialSpec], *, mode: str) -> None:
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise RuntimeError("trial IDs are not unique")
    if [trial.order_index for trial in trials] != list(range(len(trials))):
        raise RuntimeError("trial order indices are not contiguous")
    practice = [trial for trial in trials if trial.practice]
    evaluated = [trial for trial in trials if not trial.practice]
    expected_count = 16 if mode == "minimal_pilot" else 20
    if len(practice) != 4 or len(evaluated) != expected_count:
        raise RuntimeError("schedule trial count mismatch")
    core = [trial for trial in evaluated if not trial.anchor_repeat]
    if len(core) != 16:
        raise RuntimeError("core factorial must have 16 trials")
    if _count(trial.condition_id for trial in core) != {"A_reactive": 8, "C_smooth": 8}:
        raise RuntimeError("A/C core counts are not balanced")
    for phase in TASK_PHASES:
        phase_trials = [trial for trial in core if trial.task_phase == phase]
        if _count(trial.condition_id for trial in phase_trials) != {"A_reactive": 2, "C_smooth": 2}:
            raise RuntimeError(f"A/C imbalance in phase {phase}")
    for severity in SEVERITIES:
        severity_trials = [trial for trial in core if trial.severity == severity]
        if _count(trial.condition_id for trial in severity_trials) != {"A_reactive": 4, "C_smooth": 4}:
            raise RuntimeError(f"A/C imbalance in severity {severity}")
    if _count(trial.crossing_direction for trial in core) != {
        "left_to_right": 8, "right_to_left": 8
    }:
        raise RuntimeError("crossing direction is not balanced")
    if _count(trial.crossing_speed for trial in core) != {"slow": 8, "fast": 8}:
        raise RuntimeError("crossing speed is not balanced")
    if _maximum_run([trial.condition_id for trial in core]) > 2:
        raise RuntimeError("condition run exceeds two trials")
    if mode == "pilot_with_anchors":
        anchors = [trial for trial in evaluated if trial.anchor_repeat]
        if len(anchors) != 4 or _count(trial.condition_id for trial in anchors) != {
            "A_reactive": 2, "C_smooth": 2
        }:
            raise RuntimeError("anchor-repeat schedule is not balanced")


class TrialState(str, Enum):
    RESET = "RESET"
    LOAD_BC_FEASIBLE_SCENARIO = "LOAD_BC_FEASIBLE_SCENARIO"
    START_FROZEN_BC = "START_FROZEN_BC"
    WAIT_FOR_TARGET_TASK_PHASE = "WAIT_FOR_TARGET_TASK_PHASE"
    SHOW_HAND_CROSSING_CUE = "SHOW_HAND_CROSSING_CUE"
    EXECUTE_SINGLE_CROSSING = "EXECUTE_SINGLE_CROSSING"
    RUN_ASSIGNED_RESPONSE_A_OR_C = "RUN_ASSIGNED_RESPONSE_A_OR_C"
    TRACK_SAFETY_RESPONSE_EPISODE = "TRACK_SAFETY_RESPONSE_EPISODE"
    DETECT_STABLE_TASK_RESUMPTION = "DETECT_STABLE_TASK_RESUMPTION"
    SHOW_RETURN_HAND_TO_NEUTRAL_CUE = "SHOW_RETURN_HAND_TO_NEUTRAL_CUE"
    PAUSE_NOMINAL_TASK_PROGRESSION = "PAUSE_NOMINAL_TASK_PROGRESSION"
    SHOW_MANDATORY_FEEDBACK_UI = "SHOW_MANDATORY_FEEDBACK_UI"
    SAVE_FEEDBACK = "SAVE_FEEDBACK"
    RESUME_TASK_TO_COMPLETION_OR_TERMINAL = "RESUME_TASK_TO_COMPLETION_OR_TERMINAL"
    SAVE_TRIAL = "SAVE_TRIAL"
    VALIDATE_TRIAL = "VALIDATE_TRIAL"
    NEXT_TRIAL_OR_END_SESSION = "NEXT_TRIAL_OR_END_SESSION"


_TRIAL_STATE_SEQUENCE = tuple(TrialState)


class TrialLifecycle:
    """Auditable one-way trial FSM with exact crossing/query cardinality."""

    def __init__(self, condition_id: str) -> None:
        if condition_id not in CONDITIONS:
            raise ValueError("assigned condition is not A or C")
        self.condition_id = condition_id
        self.state = TrialState.RESET
        self.intended_crossing_count = 0
        self.mandatory_query_count = 0
        self.feedback_saved_count = 0
        self.history = [self.state.value]

    def advance(self, next_state: TrialState | str) -> None:
        target = TrialState(next_state)
        current_index = _TRIAL_STATE_SEQUENCE.index(self.state)
        if current_index + 1 >= len(_TRIAL_STATE_SEQUENCE) or _TRIAL_STATE_SEQUENCE[current_index + 1] is not target:
            raise RuntimeError(f"illegal trial transition {self.state.value} -> {target.value}")
        self.state = target
        self.history.append(target.value)

    def record_crossing(self) -> None:
        if self.state is not TrialState.EXECUTE_SINGLE_CROSSING:
            raise RuntimeError("crossing recorded outside crossing state")
        self.intended_crossing_count += 1
        if self.intended_crossing_count > 1:
            raise RuntimeError("more than one intended crossing in a trial")

    def record_query(self) -> None:
        if self.state is not TrialState.SHOW_MANDATORY_FEEDBACK_UI:
            raise RuntimeError("query recorded outside feedback UI state")
        self.mandatory_query_count += 1
        if self.mandatory_query_count > 1:
            raise RuntimeError("more than one mandatory query in a trial")

    def record_feedback_saved(self) -> None:
        if self.state is not TrialState.SAVE_FEEDBACK:
            raise RuntimeError("feedback saved outside SAVE_FEEDBACK")
        self.feedback_saved_count += 1

    def validate_complete(self) -> None:
        if self.state is not TrialState.NEXT_TRIAL_OR_END_SESSION:
            raise RuntimeError("trial lifecycle is incomplete")
        if (self.intended_crossing_count, self.mandatory_query_count, self.feedback_saved_count) != (1, 1, 1):
            raise RuntimeError("trial must contain exactly one crossing, query, and feedback save")


class ResponsePhase(str, Enum):
    PRE_RESPONSE = "PRE_RESPONSE"
    CBF_ACTIVE = "CBF_ACTIVE"
    SMOOTH_TAIL = "SMOOTH_TAIL"
    RECOVERY_ACTIVE = "RECOVERY_ACTIVE"
    BC_RESUMED = "BC_RESUMED"
    STABLE_TASK_RESUMPTION = "STABLE_TASK_RESUMPTION"


@dataclass(frozen=True)
class ResponsePhaseConfig:
    onset_intervention_rad_s: float = 0.05
    onset_confirmation_frames: int = 3
    stable_intervention_rad_s: float = 0.01
    stable_duration_s: float = 0.5

    def __post_init__(self) -> None:
        if self.onset_intervention_rad_s <= 0 or self.onset_confirmation_frames < 1:
            raise ValueError("invalid response onset configuration")
        if self.stable_intervention_rad_s < 0 or self.stable_duration_s <= 0:
            raise ValueError("invalid stable-resumption configuration")


@dataclass(frozen=True)
class ResponsePhaseUpdate:
    phase: ResponsePhase
    onset_candidate_now: bool
    onset_confirmed_now: bool
    stable_resumption_now: bool
    recovery_onset_now: bool
    recovery_end_now: bool
    onset_candidate_step: int
    onset_confirmed_step: int
    recovery_onset_step: int
    recovery_end_step_exclusive: int
    stable_resumption_step: int
    onset_simulation_time_s: float
    recovery_onset_simulation_time_s: float
    recovery_end_simulation_time_s: float
    stable_resumption_simulation_time_s: float


class SafetyResponsePhaseDetector:
    """Separate CBF constraint activity, C correction tail, and Recovery."""

    def __init__(self, config: ResponsePhaseConfig | None = None) -> None:
        self.config = config or ResponsePhaseConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = ResponsePhase.PRE_RESPONSE
        self._onset_streak = 0
        self._stable_since_s: float | None = None
        self.onset_candidate_step = -1
        self.onset_confirmed_step = -1
        self.recovery_onset_step = -1
        self.recovery_end_step_exclusive = -1
        self.stable_resumption_step = -1
        self.onset_simulation_time_s = -1.0
        self.recovery_onset_simulation_time_s = -1.0
        self.recovery_end_simulation_time_s = -1.0
        self.stable_resumption_simulation_time_s = -1.0
        self._raw_recovery_active = False
        self._pending_recovery_onset_step = -1
        self._pending_recovery_onset_simulation_time_s = -1.0

    def update(
        self,
        *,
        control_step: int,
        simulation_time_s: float,
        cbf_constraint_active: bool,
        intervention_norm_rad_s: float,
        smooth_tail_active: bool,
        recovery_active: bool,
        bc_resumed: bool,
    ) -> ResponsePhaseUpdate:
        step = int(control_step)
        sim_time = float(simulation_time_s)
        intervention = float(intervention_norm_rad_s)
        if step < 0 or not math.isfinite(sim_time) or sim_time < 0:
            raise ValueError("invalid response detector clock")
        if not math.isfinite(intervention) or intervention < 0:
            raise ValueError("invalid intervention norm")
        candidate_now = False
        confirmed_now = False
        stable_now = False
        recovery_onset_now = bool(recovery_active and not self._raw_recovery_active)
        recovery_end_now = bool(not recovery_active and self._raw_recovery_active)
        if recovery_onset_now:
            self._pending_recovery_onset_step = step
            self._pending_recovery_onset_simulation_time_s = sim_time
        elif recovery_end_now and self.onset_confirmed_step < 0:
            # Do not promote a completed pre-response Recovery run into the
            # later confirmed safety-response episode.
            self._pending_recovery_onset_step = -1
            self._pending_recovery_onset_simulation_time_s = -1.0
        self._raw_recovery_active = bool(recovery_active)
        # Stable task resumption closes this response episode. Query and
        # task-completion rows retain its terminal label rather than reopening
        # the response if a later sample happens to look active.
        if self.stable_resumption_step >= 0:
            self.phase = ResponsePhase.STABLE_TASK_RESUMPTION
            return ResponsePhaseUpdate(
                phase=self.phase,
                onset_candidate_now=False,
                onset_confirmed_now=False,
                stable_resumption_now=False,
                recovery_onset_now=recovery_onset_now,
                recovery_end_now=recovery_end_now,
                onset_candidate_step=self.onset_candidate_step,
                onset_confirmed_step=self.onset_confirmed_step,
                recovery_onset_step=self.recovery_onset_step,
                recovery_end_step_exclusive=self.recovery_end_step_exclusive,
                stable_resumption_step=self.stable_resumption_step,
                onset_simulation_time_s=self.onset_simulation_time_s,
                recovery_onset_simulation_time_s=(
                    self.recovery_onset_simulation_time_s
                ),
                recovery_end_simulation_time_s=(
                    self.recovery_end_simulation_time_s
                ),
                stable_resumption_simulation_time_s=(
                    self.stable_resumption_simulation_time_s
                ),
            )
        onset_signal = bool(
            cbf_constraint_active
            and intervention >= self.config.onset_intervention_rad_s
        )
        if self.onset_confirmed_step < 0:
            if onset_signal:
                if self._onset_streak == 0:
                    self.onset_candidate_step = step
                    self.onset_simulation_time_s = sim_time
                    candidate_now = True
                self._onset_streak += 1
                if self._onset_streak >= self.config.onset_confirmation_frames:
                    self.onset_confirmed_step = step
                    confirmed_now = True
            else:
                self._onset_streak = 0
                self.onset_candidate_step = -1
                self.onset_simulation_time_s = -1.0

        # Recovery is an overlapping response component, not merely the
        # lowest-precedence categorical phase.  Promote its raw-mask onset as
        # soon as the response is confirmed, even when CBF or smoothing still
        # owns the categorical phase label.
        if self.onset_confirmed_step >= 0:
            if self.recovery_onset_step < 0 and recovery_active:
                pending_inside_response = bool(
                    self._pending_recovery_onset_step
                    >= self.onset_candidate_step
                )
                self.recovery_onset_step = int(
                    self._pending_recovery_onset_step
                    if pending_inside_response
                    else self.onset_candidate_step
                )
                self.recovery_onset_simulation_time_s = float(
                    self._pending_recovery_onset_simulation_time_s
                    if pending_inside_response
                    else self.onset_simulation_time_s
                )
            if self.recovery_onset_step >= 0:
                if recovery_active:
                    # A later Recovery run means the final end is not known
                    # until that run also releases authority.
                    self.recovery_end_step_exclusive = -1
                    self.recovery_end_simulation_time_s = -1.0
                elif recovery_end_now:
                    self.recovery_end_step_exclusive = step
                    self.recovery_end_simulation_time_s = sim_time

        if self.onset_confirmed_step < 0:
            self.phase = (
                ResponsePhase.CBF_ACTIVE
                if onset_signal
                else ResponsePhase.PRE_RESPONSE
            )
        elif cbf_constraint_active:
            self.phase = ResponsePhase.CBF_ACTIVE
        elif smooth_tail_active:
            self.phase = ResponsePhase.SMOOTH_TAIL
        elif recovery_active:
            self.phase = ResponsePhase.RECOVERY_ACTIVE
        elif bc_resumed:
            self.phase = ResponsePhase.BC_RESUMED

        stable_signal = bool(
            self.onset_confirmed_step >= 0
            and not cbf_constraint_active
            and not smooth_tail_active
            and not recovery_active
            and bc_resumed
            and intervention <= self.config.stable_intervention_rad_s
        )
        if stable_signal:
            if self._stable_since_s is None:
                self._stable_since_s = sim_time
            if sim_time - self._stable_since_s >= self.config.stable_duration_s:
                if self.stable_resumption_step < 0:
                    self.stable_resumption_step = step
                    self.stable_resumption_simulation_time_s = sim_time
                    stable_now = True
                self.phase = ResponsePhase.STABLE_TASK_RESUMPTION
        else:
            self._stable_since_s = None

        return ResponsePhaseUpdate(
            phase=self.phase,
            onset_candidate_now=candidate_now,
            onset_confirmed_now=confirmed_now,
            stable_resumption_now=stable_now,
            recovery_onset_now=recovery_onset_now,
            recovery_end_now=recovery_end_now,
            onset_candidate_step=self.onset_candidate_step,
            onset_confirmed_step=self.onset_confirmed_step,
            recovery_onset_step=self.recovery_onset_step,
            recovery_end_step_exclusive=self.recovery_end_step_exclusive,
            stable_resumption_step=self.stable_resumption_step,
            onset_simulation_time_s=self.onset_simulation_time_s,
            recovery_onset_simulation_time_s=self.recovery_onset_simulation_time_s,
            recovery_end_simulation_time_s=self.recovery_end_simulation_time_s,
            stable_resumption_simulation_time_s=self.stable_resumption_simulation_time_s,
        )


FORBIDDEN_DECISION_CONTEXT_KEYS = frozenset({
    "final_response_duration",
    "final_jerk",
    "final_recovery_duration",
    "completion_delay",
    "task_success",
    "future_minimum_gap",
    "object_drop",
})


def validate_decision_context(context: Mapping[str, Any]) -> None:
    overlap = sorted(set(context).intersection(FORBIDDEN_DECISION_CONTEXT_KEYS))
    if overlap:
        raise ValueError("decision context contains future outcomes: " + ", ".join(overlap))
    required = {
        "task_phase", "grasp_state", "attachment_state", "robot_joint_state",
        "ee_pose_velocity", "cube_pose", "goal_pose", "hand_pose_velocity",
        "surface_gap_m", "ttc_s", "closing_speed_m_s", "active_candidate",
        "nominal_rmpflow_joint_command", "bc_raw_action", "condition_id", "lambda_s",
    }
    missing = sorted(required - set(context))
    if missing:
        raise ValueError("decision context missing fields: " + ", ".join(missing))


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "A_REACTIVE", "C_SMOOTH", "CONDITIONS", "CONDITION_IDS", "DIRECTIONS",
    "ProtocolSchedule", "ResponseCondition", "ResponsePhase", "ResponsePhaseConfig",
    "ResponsePhaseUpdate", "SCHEMA_VERSION", "SEVERITIES", "SPEEDS",
    "SafetyResponsePhaseDetector", "TASK_PHASES", "TrialLifecycle", "TrialSpec",
    "TrialState", "build_participant_schedule", "validate_decision_context",
]
